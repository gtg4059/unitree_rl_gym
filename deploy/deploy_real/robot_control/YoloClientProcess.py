from multiprocessing import Process, Array
import traceback
import zmq
import struct
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml
from ultralytics.utils import YAML
from legged_gym import LEGGED_GYM_ROOT_DIR
import pyrealsense2 as rs
from itertools import combinations

# Configuration
PORT_IMAGE = 1234
PORT_SYNC = 7777
YOLO_MODEL_PATH = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/policy/YOLO/last.pt" #"yolov8_depth/pt/0528_jh_yolov8m_l.pt"
# BOX class ID : 1
KEYPOINT_CLASS_ID = 1

# 박스 실측 사이즈(키포인트 기준)
D_REF = {
    (0, 1): 0.345,  # p0-p1
    (1, 2): 0.220,  # p1-p2
    (2, 3): 0.345,  # p2-p3
    (0, 3): 0.220,  # p0-p3
    (0, 2): 0.410,  # 대각
    (1, 3): 0.410,  # 대각
}

detector = cv2.QRCodeDetector()

##################################
#           계산 함수            #
##################################

def make_rs_intrinsics(depth_intrinsics, width=640, height=480):
    intr = rs.intrinsics()
    intr.width  = int(width)
    intr.height = int(height)
    intr.ppx    = float(depth_intrinsics[0])  # cx
    intr.ppy    = float(depth_intrinsics[1])  # cy 
    intr.fx     = float(depth_intrinsics[2])
    intr.fy     = float(depth_intrinsics[3])
    intr.model  = rs.distortion.brown_conrady

    # coeffs는 길이 5의 배열 (k1,k2,p1,p2,k3)
    intr.coeffs = [0.0]*5
    for i in range(5):
        intr.coeffs[i] = float(depth_intrinsics[4 + i])
    return intr


# keypoint 주변 depth 데이터로 보정
def get_average_depth(depth_image, x, y, depth_scale, window_size=2):
    depth_values = []
    H, W = depth_image.shape[:2]
    for dx in range(-window_size, window_size + 1):
        for dy in range(-window_size, window_size + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                d = depth_image[ny, nx] * depth_scale
                if d > 0:
                    depth_values.append(d)
    return sum(depth_values) / len(depth_values) if depth_values else None

# 픽셀 좌표를 3D 좌표로 변환
def deproject_pixel_to_point(intrinsics, pixel, depth):
    point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)
    return point  # [X, Y, Z]

# 인덱스 번호로 접근해서 픽셀좌표 > depth 데이터 호출 > 3D 좌표로 변환
def kp3d(idx, lookup, depth_image, depth_intrinsics, depth_scale):
    data = lookup.get(idx)
    if not data or data["xy"] is None:
        return None
    x, y = map(int, data["xy"])
    d = get_average_depth(depth_image, x, y, depth_scale)
    if d is None or d <= 0:
        return None
    return deproject_pixel_to_point(depth_intrinsics, (x, y), d)  # (3,)

# 3D 좌표를 픽셀 좌표로 변환
def project_point_to_pixel(intrinsics, point):
    pixel = rs.rs2_project_point_to_pixel(intrinsics, point)
    return int(pixel[0]), int(pixel[1])

# plane 추정용 유틸
def fit_plane_from_points(points3d):
    P = np.asarray(points3d, dtype=float)
    c = P.mean(axis=0)
    Q = P - c
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    n = Vt[-1]
    n /= (np.linalg.norm(n) + 1e-12)
    return c, n

def rot_axis_angle(axis, theta):
    axis = np.asarray(axis, float)
    axis /= (np.linalg.norm(axis) + 1e-12)
    x, y, z = axis
    c = np.cos(theta); s = np.sin(theta); C = 1 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s,  x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,    y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s,  c + z*z*C   ]
    ], dtype=float)

# 키포인트 간 거리가 실측값과 오차가 크지 않은지 검사
def _key(i, j):  # (i,j) 정렬된 키
    return (i, j) if i <= j else (j, i)

def validate_points_by_distance(
    P3, D_REF, conf=None, abs_tol=0.03, rel_tol=0.10, use_conf=True, plane_state=None, single_tol=0.01
):
    """
    P3: {id: np.array([X,Y,Z])}  // 이번 프레임에서 3D 복원된 유효 키포인트만
    D_REF: {(i,j): d_ref_m}      // 실측 기준 거리(미터). (i<j) 키 권장(변+대각 권장)
    conf: {id: float in [0,1]}   // (선택) 키포인트 신뢰도. 없으면 전부 1.0
    abs_tol: 절대 허용오차(m)
    rel_tol: 상대 허용오차
    use_conf: True면 페어 비용에 1/min(conf_i, conf_j) 가중
    plane_state: dict(c0, n, pts0) 형태면 single-point 추정 시 참조
    single_tol: 입력점이 한 개인 경우, 이전 점과의 허용 3D 거리 임계 (m)
    """
    ids = sorted(P3.keys())
    if conf is None:
        conf = {i: 1.0 for i in ids}

    if len(ids) == 1:
        k = ids[0]
        if (plane_state is None) or ("pts0" not in plane_state) or (k >= len(plane_state["pts0"])):
            return [], ids, {}
        p_prev = np.asarray(plane_state["pts0"][k], float)
        p_now  = np.asarray(P3[k], float)
        if not np.all(np.isfinite(p_prev)) or not np.all(np.isfinite(p_now)):
            return [], ids, {}
        dist = float(np.linalg.norm(p_now - p_prev))
        if dist <= single_tol:
            return [k], [], {"(k,)": (dist, single_tol)}
        else:
            return [], [k], {"(k,)": (dist, single_tol)}
    
    pair_err = {}
    any_violation = False
    for i, j in combinations(ids, 2):
        k = _key(i, j)
        if k not in D_REF:
            pair_err[k] = (np.nan, np.nan, np.nan, np.nan, None)
            any_violation = True
            continue
        d_obs = float(np.linalg.norm(P3[i] - P3[j]))
        d_ref = float(D_REF[k])
        err   = abs(d_obs - d_ref)
        thr   = max(abs_tol, rel_tol * d_ref)
        ok    = err <= thr
        pair_err[k] = (d_obs, d_ref, err, thr, ok)
        if not ok:
            any_violation = True
            print(f"[DEBUG] pair {i}-{j}: d_obs={d_obs:.4f}, d_ref={d_ref:.4f}, err={err:.4f}, thr={thr:.4f} -> EXCEEDED")

    if not any_violation:
        return ids, [], pair_err

    best_subset = None
    best_score  = float("inf")
    for sz in range(len(ids), 1, -1):
        for S in combinations(ids, sz):
            S = list(S)
            pairs = list(combinations(S, 2))
            if not pairs:
                continue
            all_ok = True
            score = 0.0
            cnt = 0
            for a, b in pairs:
                k = _key(a, b)
                if k not in pair_err:
                    all_ok = False; break
                d_obs, d_ref, err, thr, ok = pair_err[k]
                if ok is False:
                    all_ok = False; break
                if ok is None:
                    continue
                w = 1.0 / max(1e-3, min(conf.get(a,1.0), conf.get(b,1.0))) if use_conf else 1.0
                score += w * (err / (thr + 1e-12))
                cnt   += 1
            if not all_ok or cnt == 0:
                continue
            score /= cnt
            if (best_subset is None) or (len(S) > len(best_subset)) or \
               (len(S) == len(best_subset) and score < best_score):
                best_subset = S
                best_score  = score

        if best_subset is not None:
            break

    if best_subset is None:
        return [], ids, pair_err

    inliers  = list(best_subset)
    outliers = [i for i in ids if i not in inliers]
    return inliers, outliers, pair_err


##################################
#     평면 상태(함수형) 유틸     #
##################################

# plane_state는 dict로 관리: {"c0": np.ndarray(3,), "n": np.ndarray(3,), "pts0": (4,3) array}

def tracked_plane_init(p0, p1, p2, p3):
    pts = np.vstack([p0, p1, p2, p3]).astype(float)  # (4,3)
    c0, n = fit_plane_from_points(pts)
    return {"c0": c0.copy(), "n": n.copy(), "pts0": pts.copy()}

def tracked_plane_apply(state, R, t, c_hat):
    # 상태 갱신: c0, pts0, n
    new_state = {
        "c0": np.asarray(c_hat, float).copy(),
        "pts0": (R @ state["pts0"].T).T + t,
        "n": (R @ state["n"])
    }
    new_state["n"] /= (np.linalg.norm(new_state["n"]) + 1e-12)
    return new_state

def tracked_plane_estimate_center_from_one_idx(state, k, p_new):
    pk0 = state["pts0"][int(k)]
    t = np.asarray(p_new, float) - pk0
    return state["c0"] + t  # c_hat

def tracked_plane_estimate_two(state, i, j, pi_new, pj_new, wi=1.0, wj=1.0, max_drift=None):
    n = state["n"]
    pi0, pj0 = state["pts0"][int(i)], state["pts0"][int(j)]
    v0 = pj0 - pi0
    v1 = np.asarray(pj_new) - np.asarray(pi_new)
    v0p = v0 - np.dot(v0, n) * n
    v1p = v1 - np.dot(v1, n) * n

    if np.linalg.norm(v0p) < 1e-9 or np.linalg.norm(v1p) < 1e-9:
        R = np.eye(3)
    else:
        u0 = v0p / np.linalg.norm(v0p)
        u1 = v1p / np.linalg.norm(v1p)
        sin_th = float(np.dot(n, np.cross(u0, u1)))
        cos_th = float(np.dot(u0, u1))
        theta = np.arctan2(sin_th, cos_th)
        R = rot_axis_angle(n, theta)

    t_i = np.asarray(pi_new) - R @ pi0
    t_j = np.asarray(pj_new) - R @ pj0
    t = (wi * t_i + wj * t_j) / (wi + wj + 1e-12)

    if max_drift is not None:
        norm_t = np.linalg.norm(t)
        if norm_t > max_drift:
            t = t * (max_drift / (norm_t + 1e-12))

    c_hat = R @ state["c0"] + t
    r = float(np.linalg.norm(v1p) / (np.linalg.norm(v0p) + 1e-12)) if np.linalg.norm(v0p) > 0 else 1.0
    residual = float(
        np.linalg.norm((R @ pi0 + t) - pi_new) +
        np.linalg.norm((R @ pj0 + t) - pj_new)
    )
    return c_hat, R, t, residual, r

def tracked_plane_estimate_three(state, idxs, P_new, w=None, area_tol=1e-6):
    A = state["pts0"][list(idxs)].astype(float)   # (3,3) 기준 3D
    B = np.asarray(P_new, float).reshape(3, 3)    # (3,3) 현재 3D

    area = 0.5 * np.linalg.norm(np.cross(A[1] - A[0], A[2] - A[0]))
    if area < area_tol:
        return None

    if w is None:
        ca, cb = A.mean(axis=0), B.mean(axis=0)
        Ac, Bc = A - ca, B - cb
        H = Ac.T @ Bc
    else:
        w = np.asarray(w, float).reshape(3, 1)
        w = w / (w.sum() + 1e-12)
        ca = (A * w).sum(axis=0)
        cb = (B * w).sum(axis=0)
        Ac, Bc = A - ca, B - cb
        H = (Ac * w).T @ Bc

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = cb - R @ ca
    c_hat = R @ state["c0"] + t
    residual = float(np.mean(np.linalg.norm((R @ A.T).T + t - B, axis=1)))
    return c_hat, R, t, residual


##################################
#          시각화 함수            #
##################################

def draw_label(img, text, x1, y1, x2, y2, color_bgr, margin=3):
    H, W = img.shape[:2]
    color_bgr = tuple(int(c) for c in color_bgr)

    tl = 2
    tf = max(tl - 1, 1)
    fs = tl / 3
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, tf)

    tx, ty = int(x1), int(y1 - th - margin)
    if ty < 0:
        tx = int(x2 - tw)
        ty = int(y2 + margin)

    tx = max(0, min(tx, W - tw - 1))
    ty = max(0, min(ty, H - (th + margin) - 1))

    cv2.rectangle(img, (tx, ty), (tx + tw, ty + th + margin), color_bgr, -1, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty + th), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)


#############################################################################
# Dummy image for testing
def dummy_image():
    return np.full((480, 640, 3), 255, dtype=np.uint8)
#############################################################################

class YoloClientProcess:
    def __init__(self, boxdata, server_ip="127.0.0.1",):
        self.server_ip = server_ip
        client_process = Process(target=self.run_client, args=(boxdata,))
        client_process.daemon = True
        client_process.start()

    # keypoint 주변 depth 데이터로 보정
    def _get_average_depth(self, depth_image, x, y, window_size=3):
        w = h = window_size // 2
        depth_values = []
        for dx in range(-w, w + 1):
            for dy in range(-h, h + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < depth_image.shape[0] and 0 <= ny < depth_image.shape[1]:
                    d = depth_image[nx,ny]
                    if d > 0:
                        depth_values.append(d)
        return np.median(depth_values) if depth_values else 0.0

    # 픽셀 좌표를 3D 좌표로 변환
    '''def deproject_pixel_to_point(intrinsics, pixel, depth):
        point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)
        return point  # [X, Y, Z]'''

    def _deproject_pixel_to_point(self,intrinsics, pixel, depth):
        ppx=intrinsics[0]
        ppy=intrinsics[1]
        fx=intrinsics[2]
        fy=intrinsics[3]
        coeffs=intrinsics[4:]

        x = (pixel[0] - ppx) / fx
        y = (pixel[1] - ppy) / fy
        xo = x
        yo = y
        point = np.array([0, 0, 0])
        for i in range(10):
            r2 = x * x + y * y
            icdist = 1 / (1 + ((coeffs[4] * r2 + coeffs[1]) * r2 + coeffs[0]) * r2)
            xq = x / icdist
            yq = y / icdist
            delta_x = 2 * coeffs[2] * xq * yq + coeffs[3] * (r2 + 2 * xq * xq)
            delta_y = 2 * coeffs[3] * xq * yq + coeffs[2] * (r2 + 2 * yq * yq)
            x = (xo - delta_x) * icdist
            y = (yo - delta_y) * icdist
        point[0] = depth * x
        point[1] = depth * y
        point[2] = depth
        return point

    # 3D 좌표를 픽셀 좌표로 변환
    '''def project_point_to_pixel(intrinsics, point):
        pixel = rs.rs2_project_point_to_pixel(intrinsics, point)
        return int(pixel[0]), int(pixel[1])'''

    def _project_point_to_pixel(self,intrinsics, point):
        ppx = intrinsics[0]
        ppy = intrinsics[1]
        fx = intrinsics[2]
        fy = intrinsics[3]
        coeffs = intrinsics[4:]

        x = point[0] / point[2]
        y = point[1] / point[2]
        pixel = np.array([0, 0])
        r2 = x * x + y * y
        f = 1 + coeffs[0] * r2 + coeffs[1] * r2 * r2 + coeffs[4] * r2 * r2 * r2
        x *= f
        y *= f
        dx = x + 2 * coeffs[2] * x * y + coeffs[3] * (r2 + 2 * x * x)
        dy = y + 2 * coeffs[3] * x * y + coeffs[2] * (r2 + 2 * y * y)
        x = dx
        y = dy
        pixel[0] = x * fx + ppx
        pixel[1] = y * fy + ppy
        return pixel

    # 대각선 예외처리
    def _compute_center_3d(self,points_3d):
        # if len(points_3d) != 4:
        #     return None  # 예외 처리

        p0, p1, p2, p3 = points_3d

        all_valid = all(p is not None for p in [p0, p1, p2, p3])
        pair_02_valid = all(p is not None for p in [p0, p2])  # 대각선 1
        pair_13_valid = all(p is not None for p in [p1, p3])  # 대각선 2

        if all_valid:
            return np.mean([p0, p1, p2, p3], axis=0)
        elif pair_13_valid:
            return np.mean([p1, p3], axis=0)
        elif pair_02_valid:
            return np.mean([p0, p2], axis=0)
        else:
            return None

    # 각면의 중점 구하기
    def _compute_pixel_center_from_diagonals(self,points):
        if len(points) != 4:
            return None

        p0, p1, p2, p3 = points
        pair_02_valid = p0 is not None and p2 is not None
        pair_13_valid = p1 is not None and p3 is not None

        if all(p is not None for p in [p0, p1, p2, p3]):
            return tuple(np.mean([p0, p1, p2, p3], axis=0).astype(int))
        elif pair_13_valid:
            return tuple(np.mean([p1, p3], axis=0).astype(int))
        elif pair_02_valid:
            return tuple(np.mean([p0, p2], axis=0).astype(int))
        else:
            return None

    # 면의 법선벡터 구하기
    def _compute_normal_from_points(self,points):
        valid_points = [np.array(p) for p in points if p is not None]

        if len(valid_points) < 3:
            return None  # 법선 계산 불가

        # 삼각형(3점)일 경우
        if len(valid_points) == 3:
            v1 = valid_points[1] - valid_points[0]
            v2 = valid_points[2] - valid_points[0]
            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            return n / norm if norm > 1e-6 else None

        # 사각형(4점)일 경우: 삼각형 2개로 나눠 평균
        if len(valid_points) == 4:
            v1a = valid_points[1] - valid_points[0]
            v2a = valid_points[2] - valid_points[0]
            n1 = np.cross(v1a, v2a)

            v1b = valid_points[2] - valid_points[0]
            v2b = valid_points[3] - valid_points[0]
            n2 = np.cross(v1b, v2b)

            n_avg = (n1 + n2) / 2
            norm = np.linalg.norm(n_avg)
            return n_avg / norm if norm > 1e-6 else None

        return None

    # 법선을 이용해서 박스의 중심 구하기
    def _compute_box_center_from_faces(self,face_centers_3d, face_normals_3d):
        Ps = []
        ns = []

        for face in ["top", "left", "right"]:
            p = face_centers_3d.get(face)
            n = face_normals_3d.get(face)

            if p is not None and n is not None:
                Ps.append(np.array(p))
                ns.append(np.array(n))

        if len(Ps) < 2:
            return None

        # 최소제곱 해를 구하기 위한 Ax = b 구성
        A = []
        b = []
        for P, n in zip(Ps, ns):
            n = n / np.linalg.norm(n)
            I = np.eye(3)
            A_i = I - np.outer(n, n)
            A.append(A_i)
            b.append(A_i @ P)

        A = np.sum(A, axis=0)
        b = np.sum(b, axis=0)

        # 최소제곱 해 (법선들이 가장 가까이 만나는 점)
        center = np.linalg.lstsq(A, b, rcond=None)[0]
        return center

    # Dummy image for testing
    def _dummy_image(self):
        return np.full((480, 640, 3), 255, dtype=np.uint8)

    def run_client(self, boxdata=None):
        context = zmq.Context()

        # SUB socket for receiving image
        sub_socket = context.socket(zmq.SUB)
        sub_socket.connect(f"tcp://{self.server_ip}:{PORT_IMAGE}")
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # REQ socket for sending YOLO result
        req_socket = context.socket(zmq.REQ)
        req_socket.connect(f"tcp://{self.server_ip}:{PORT_SYNC}")

        model = YOLO(YOLO_MODEL_PATH)
        NAMES = model.names   
        print("[Client] Started")

        # Create message: 1.0 + 7 dummy float values
        yolo_result = [1.0] + [0.0] * 7
        packed_msg = struct.pack("8f", *yolo_result)

        # Send to server and wait for reply
        req_socket.send(packed_msg)
        ack = req_socket.recv()

        rng = np.random.default_rng(42)
        colors = (rng.uniform(0, 255, size=(len(NAMES), 3))).astype(np.uint8)

        plane_state = None

        while True:
            try:
                # Receive image
                message = sub_socket.recv()
                header_size = struct.calcsize('iii10f')
                header = message[:header_size]
                lidar_len, color_len, depth_len, *depth_intrinsics, depth_scale = struct.unpack('iii10f', header)
                jpg_bytes = message[header_size:header_size + color_len]
                png_bytes = message[header_size + color_len:header_size + color_len + depth_len]
                lid_bytes = message[header_size + color_len + depth_len:header_size + color_len + depth_len + lidar_len]

                np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
                if np_img.shape[0] == 0:
                    continue
                color_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

                np_depth = np.frombuffer(png_bytes, dtype=np.uint8)
                if np_depth.shape[0] == 0:
                    continue
                depth_image = cv2.imdecode(np_depth, cv2.IMREAD_UNCHANGED)
                depth_intrinsics = make_rs_intrinsics(depth_intrinsics, width=640, height=480)

                # Run YOLO
                results = model(color_image, stream=True, conf=0.5)

                for res in results:
                    has_kpts = res.keypoints is not None
                    boxes = res.boxes

                    for i, b in enumerate(boxes):
                        conf = float(b.conf.item())
                        if conf < 0.5:
                            continue

                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        cls_id = int(b.cls.item())
                        name = NAMES.get(cls_id, str(cls_id))
                        color = tuple(int(c) for c in colors[cls_id])  # BGR

                        cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                        label = f"{name} {conf:.2f}"
                        draw_label(color_image, label, x1, y1, x2, y2, color)

                        if has_kpts and cls_id == KEYPOINT_CLASS_ID:
                            kpts_xy   = res.keypoints.xy[i].cpu().numpy()    # (K, 2)
                            kpts_conf = res.keypoints.conf[i].cpu().numpy()  # (K,)

                            KPT_THR = 0.5
                            EPS = 1e-6
                            H, W = color_image.shape[:2]

                            mask_conf    = (kpts_conf >= KPT_THR)
                            mask_nonzero = (np.abs(kpts_xy[:, 0]) > EPS) | (np.abs(kpts_xy[:, 1]) > EPS)
                            mask_bounds  = (kpts_xy[:, 0] >= 0) & (kpts_xy[:, 0] < W) & \
                                        (kpts_xy[:, 1] >= 0) & (kpts_xy[:, 1] < H)
                            mask = mask_conf & mask_nonzero & mask_bounds

                            valid_idx  = np.flatnonzero(mask)          # (M,)
                            valid_xy   = kpts_xy[valid_idx]            # (M, 2)
                            valid_conf = kpts_conf[valid_idx]          # (M,)

                            for (kx, ky) in valid_xy:
                                cv2.circle(color_image, (int(kx), int(ky)), 3, (0, 255, 0), -1)

                            lookup = {
                                int(k): {"xy": (float(kpts_xy[k, 0]), float(kpts_xy[k, 1])),
                                        "conf": float(kpts_conf[k])}
                                for k in valid_idx
                            }

                            need_ids = [0, 1, 2, 3]

                            # 3D 복원
                            P3 = {}  # {kpt_id: np.array([X,Y,Z])}
                            for k in need_ids:
                                if k in lookup:
                                    pt3 = kp3d(k, lookup, depth_image, depth_intrinsics, depth_scale)
                                    if pt3 is not None:
                                        P3[k] = np.array(pt3, dtype=float)

                            conf_dict = {k: float(kpts_conf[k]) for k in P3.keys()}
                            present, outliers, pair_err = validate_points_by_distance(
                                P3, D_REF, conf=conf_dict, abs_tol=0.5, rel_tol=0.20, plane_state=plane_state
                            )

                            # plane_state 있으면 업데이트
                            if plane_state is not None:
                                if len(present) == 4 and set(present) == set(need_ids):
                                    # 4점 모두 보임 → 재초기화
                                    p = [P3[k] for k in need_ids]
                                    plane_state = tracked_plane_init(*p)

                                elif len(present) == 3:
                                    tri = next((c for c in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
                                                if all(k in present for k in c)), None)
                                    if tri is not None:
                                        P_new = [P3[k] for k in tri]
                                        w = [float(kpts_conf[k]) for k in tri]
                                        out = tracked_plane_estimate_three(plane_state, tri, P_new, w=w)
                                        if out is not None:
                                            c_hat, R, t, res = out
                                            plane_state = tracked_plane_apply(plane_state, R, t, c_hat)
                                        else:
                                            pair = next((p for p in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
                                                        if p[0] in present and p[1] in present), None)
                                            if pair is not None:
                                                i0, j0 = pair
                                                pi_new, pj_new = P3[i0], P3[j0]
                                                wi = float(kpts_conf[i0]); wj = float(kpts_conf[j0])
                                                c_hat, R, t, residual, r = tracked_plane_estimate_two(
                                                    plane_state, i0, j0, pi_new, pj_new, wi=wi, wj=wj
                                                )
                                                plane_state = tracked_plane_apply(plane_state, R, t, c_hat)

                                elif len(present) == 2:
                                    pair = next((p for p in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
                                                if p[0] in present and p[1] in present), None)
                                    if pair is not None:
                                        i0, j0 = pair
                                        pi_new, pj_new = P3[i0], P3[j0]
                                        wi = float(kpts_conf[i0]); wj = float(kpts_conf[j0])
                                        c_hat, R, t, residual, r = tracked_plane_estimate_two(
                                            plane_state, i0, j0, pi_new, pj_new, wi=wi, wj=wj
                                        )
                                        plane_state = tracked_plane_apply(plane_state, R, t, c_hat)

                                elif len(present) == 1:
                                    k = present[0]
                                    c_hat = tracked_plane_estimate_center_from_one_idx(plane_state, k, P3[k])
                                    # pts0도 동일 평행이동 적용(해당 점 기준)
                                    delta = P3[k] - plane_state["pts0"][k]
                                    plane_state = {
                                        "c0": c_hat,
                                        "n": plane_state["n"],
                                        "pts0": plane_state["pts0"] + delta
                                    }

                            # plane_state 없으면 초기화
                            if plane_state is None and len(present) == 4 and set(present) == set(need_ids):
                                p = [P3[k] for k in need_ids]
                                plane_state = tracked_plane_init(*p)

                if plane_state is not None:
                    c0_text = f"({plane_state['c0'][0]:.2f}, {plane_state['c0'][1]:.2f}, {plane_state['c0'][2]:.2f})"
                    center_pixel = project_point_to_pixel(depth_intrinsics, plane_state["c0"])
                    u, v = map(int, center_pixel)
                    cv2.circle(color_image, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
                    cv2.putText(color_image, c0_text, (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    yolo_result = [1.0] + list(plane_state["c0"]) + [0.0] * 4
                else:
                    yolo_result = [1.0] + [0.0] * 7

                packed_msg = struct.pack("8f", *yolo_result)
                req_socket.send(packed_msg)
                ack = req_socket.recv()
                print(f"[Client] Sent YOLO info, got: {ack.decode()}")

                # if color_image is not None:
                #     cv2.imshow("Client View", color_image)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
                if plane_state is not None:
                    boxdata[:] = np.array(plane_state["c0"], dtype=np.float32)
                else:
                    boxdata[:] = np.array([0, 0, 0], dtype=np.float32)
                print(boxdata[:])
            except Exception as e:
                print(f"[Client] Exception: {e}")
                traceback.print_exc()
                break

        req_socket.close()
        sub_socket.close()
        context.term()
        # cv2.destroyAllWindows()

