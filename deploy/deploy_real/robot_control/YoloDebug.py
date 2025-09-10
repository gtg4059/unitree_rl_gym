from typing import Union
import numpy as np
from itertools import combinations
import traceback
import json
from multiprocessing import Process, Array

import os
import zmq
import struct
import cv2
from ultralytics import YOLO
import pyrealsense2 as rs

# Configuration
# ------------------------- Ports / Server -------------------------
PORT_IMAGE = int(os.environ.get("PORT_IMAGE", "2345"))  # 기존 5555 → 신규 규약 1234
PORT_SYNC  = int(os.environ.get("PORT_SYNC",  "6789"))  # 기존 6666 → 신규 규약 5678
#YOLO_MODEL_PATH = "js_9_8.pt" #"yolov8_depth/pt/0528_jh_yolov8m_l.pt"
YOLO_MODEL_PATH = "epoch400.pt"
model= YOLO(YOLO_MODEL_PATH)
NAMES=model.names


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
def get_average_depth(depth_image, x, y, depth_scale, window_size=3):
    depth_values = []
    H, W = depth_image.shape[:2]
    for dx in range(-window_size, window_size + 1):
        for dy in range(-window_size, window_size + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < W and 0 <= ny < H:
                d = depth_image[ny,nx]*depth_scale
                if d > 0:
                    depth_values.append(d)
    #return sum(depth_values) / len(depth_values) if depth_values else None
    return float(np.median(depth_values)) if depth_values else None

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

# plane_state 입력 시 중점 계산
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
    P3, D_REF, conf=None, abs_tol=0.03, rel_tol=0.10, use_conf=True, plane_state=None, single_tol=0.01,
    use_temporal_gate=False,
    jump_abs=0.03,   # 한 프레임에서 허용하는 3D 점프 한계 (m)
):
    """
    P3: {id: np.array([X,Y,Z])}  // 이번 프레임에서 3D 복원된 유효 키포인트만
    D_REF: {(i,j): d_ref_m}      // 실측 기준 거리(미터). (i<j) 키 권장(변+대각 권장)
    conf: {id: float in [0,1]}   // (선택) 키포인트 신뢰도. 없으면 전부 1.0
    abs_tol: 절대 허용오차(m)   // 예: 0.02 = 2 cm
    rel_tol: 상대 허용오차      // 예: 0.10 = 10 %
    use_conf: True면 페어 비용에 1/min(conf_i, conf_j) 가중
    plane_state: TrackedPlane (pts0[k]에 이전 프레임 기준 점 3D가 있음)
    single_tol: 입력점이 한 개인 경우, 이전 점과의 허용 3D 거리 임계 (m)

    반환:
      inliers:  기준을 만족하는 최대 일관 집합(list of ids)
      outliers: 나머지 ids
      pair_err: {(i,j): (d_obs, d_ref, err, thr, ok_bool)}
    """
    ids = sorted(P3.keys())
    if conf is None:
        conf = {i: 1.0 for i in ids}

    # ---------- 프레임 간 점프 게이트 ----------
    rejected_temporal = []
    if use_temporal_gate and plane_state is not None and getattr(plane_state, "pts0", None) is not None:
        prev_pts = plane_state.pts0
        keep_ids = []
        for k in ids:
            if k >= len(prev_pts) or k not in P3:
                continue
            p_prev = np.asarray(prev_pts[k], float)
            p_now  = np.asarray(P3[k], float)
            if (not np.all(np.isfinite(p_prev))) or (not np.all(np.isfinite(p_now))):
                rejected_temporal.append(k)
                continue
            # 유클리드 거리로만 판단 (방향/속도 고려 없음)
            if float(np.linalg.norm(p_now - p_prev)) > float(jump_abs):
                rejected_temporal.append(k)
                continue
            keep_ids.append(k)

        if rejected_temporal:
            P3 = {k: P3[k] for k in keep_ids}
            ids = sorted(P3.keys())
        if len(ids) == 0:
            return [], rejected_temporal, {}

    # 점이 1개인 경우 이동 거리로 처리
    if len(ids) == 1:
        k = ids[0]
        # plane_state/참조점이 없으면 보수적으로 제외하거나 정책에 맞게 결정
        if (plane_state is None) or (getattr(plane_state, "pts0", None) is None) or (k >= len(plane_state.pts0)):
            return [], ids, {}   # 보수적: 전부 outlier

        p_prev = np.asarray(plane_state.pts0[k], float)
        p_now  = np.asarray(P3[k], float)
        if not np.all(np.isfinite(p_prev)) or not np.all(np.isfinite(p_now)):
            return [], ids, {}

        dist = float(np.linalg.norm(p_now - p_prev))
        if dist <= single_tol:
            return [k], [], {"(k,)": (dist, single_tol)}
        else:
            return [], [k], {"(k,)": (dist, single_tol)}
    
    # 점이 2개 이상인 경우 모든 페어의 관측/기준/오차/허용치 계산
    pair_err = {}     # (i,j) -> (d_obs, d_ref, err, thr, ok)
    any_violation = False

    for i, j in combinations(ids, 2):
        k = _key(i, j)
        if k not in D_REF:
            # 기준이 없으면 평가 불가 → 이 페어는 'ok=False'로 기록(필터링에 사용)
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
            print(f"[DEBUG] pair {i}-{j}: d_obs={d_obs:.4f}, d_ref={d_ref:.4f}, "
              f"err={err:.4f}, thr={thr:.4f} -> EXCEEDED")

    # --- 위반이 하나도 없으면: 전부 inlier ---
    if not any_violation:
        return ids, [], pair_err

    # --- 위반이 있으면: '내부 모든 페어 ok'인 최대 부분집합 찾기 ---
    best_subset = None
    best_score  = float("inf")

    # 큰 집합부터(4→3→2) 검사
    for sz in range(len(ids), 1, -1):  # size >= 2만 고려 애매하면 전부 제외
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

                if ok is False:          # 명시적 불일치 → 탈락
                    all_ok = False; break
                if ok is None:           # 기준 없음 → 점수/카운트에서 제외
                    continue
                
                # 정규화 비용(작을수록 좋음). 신뢰도 낮으면 가중↑
                w = 1.0 / max(1e-3, min(conf.get(a,1.0), conf.get(b,1.0))) if use_conf else 1.0
                score += w * (err / (thr + 1e-12))
                cnt   += 1
            if not all_ok or cnt == 0:
                continue

            # 평균 비용으로 비교(쌍 수 보정)
            score /= cnt
            # 더 큰 집합 우선, 동일 크기면 점수 작은 것
            if (best_subset is None) or (len(S) > len(best_subset)) or \
               (len(S) == len(best_subset) and score < best_score):
                best_subset = S
                best_score  = score

        if best_subset is not None:
            break  # 최대 크기 집합을 찾았으니 종료

    if best_subset is None:
        # 기준을 만족하는 부분집합을 결정할 수 없다면 → 전부 이상점으로
        return [], ids, pair_err

    inliers  = list(best_subset)
    outliers = [i for i in ids if i not in inliers]
    if rejected_temporal:
        outliers = sorted(list(set(outliers) | set(rejected_temporal)))
    return inliers, outliers, pair_err

##################################
#  평면 상태 저장 및 추정용 클래스 #
##################################

class TrackedPlane:
    #4개 기준 키포인트(p0..p3)로 정의된 평면 상태 저장
    def __init__(self, p0, p1, p2, p3):
        pts = np.vstack([p0, p1, p2, p3])   # (4,3)
        self.c0, self.n = fit_plane_from_points(pts)
        self.pts0 = pts.copy()              # 초기 4점 3D

    # 점 하나만 보일 때: 회전 없음(평행이동만) 가정 → ĉ = c0 + (p'_k - p_k)
    def estimate_center_from_one_idx(self, k, p_new):
        
        pk0 = self.pts0[int(k)]
        t = np.asarray(p_new, float) - pk0
        return self.c0 + t

    # 점 두 개 보일 때
    def estimate_center_from_two(self, i, j, pi_new, pj_new, wi=1.0, wj=1.0, max_drift=None):
        """
        i, j: 기준 키포인트 인덱스
        pi_new, pj_new: 현재 프레임에서의 3D (카메라 좌표계)
        wi, wj: 키포인트 conf 가중치(0~1 권장)
        max_drift: (선택) 한 프레임에서 허용할 최대 평행이동(m). 크면 클램프.
        반환: c_hat, R, t, residual, r
        """
        n = self.n
        pi0, pj0 = self.pts0[int(i)], self.pts0[int(j)]
        v0 = pj0 - pi0
        v1 = np.asarray(pj_new) - np.asarray(pi_new)

        # 평면에 사영 후 '방향'만 사용(길이는 무시 → 스케일 불변)
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

        # 평행이동: 두 점의 번역을 가중 평균(스케일 차이는 이미 무시)
        t_i = np.asarray(pi_new) - R @ pi0
        t_j = np.asarray(pj_new) - R @ pj0
        t = (wi * t_i + wj * t_j) / (wi + wj + 1e-12)

        # (선택) 과한 점프 방지
        if max_drift is not None:
            norm_t = np.linalg.norm(t)
            if norm_t > max_drift:
                t = t * (max_drift / (norm_t + 1e-12))

        c_hat = R @ self.c0 + t

        # 모니터링용 잔차/스케일비 (알고리즘엔 미반영)
        r = float(np.linalg.norm(v1p) / (np.linalg.norm(v0p) + 1e-12)) if np.linalg.norm(v0p) > 0 else 1.0
        residual = float(
            np.linalg.norm((R @ pi0 + t) - pi_new) +
            np.linalg.norm((R @ pj0 + t) - pj_new)
        )
        return c_hat, R, t, residual, r
    
    # 점 세 개 보일 때
    def estimate_center_from_three_points(self, idxs, P_new, w=None, area_tol=1e-6):
        """
        plane_state: c0, n, pts0(4x3)을 가진 상태 객체
        idxs: (3,) 예: (0,1,2)
        P_new: (3,3) 현재 프레임 3D 점들, idxs 순서와 대응
        w: (3,) 키포인트 conf 가중치(없으면 균등)
        반환: c_hat, R, t, residual
        """
        A = self.pts0[list(idxs)].astype(float)   # (3,3) 기준 3D
        B = np.asarray(P_new, float).reshape(3, 3)       # (3,3) 현재 3D

        # 퇴화 방지: 삼각형 면적이 너무 작으면 실패 처리
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
        if np.linalg.det(R) < 0:   # 반사 방지
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        t = cb - R @ ca
        c_hat = R @ self.c0 + t
        residual = float(np.mean(np.linalg.norm((R @ A.T).T + t - B, axis=1)))  # 평균 오차(m)

        return c_hat, R, t, residual

##################################
#          시각화 함수            #
##################################

# 바운딩박스 라벨 표시 코드
def draw_label(img, text, x1, y1, x2, y2, color_bgr, margin=3):
    H, W = img.shape[:2]
    color_bgr = tuple(int(c) for c in color_bgr)

    tl = 2
    tf = max(tl - 1, 1)
    fs = tl / 3
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, tf)

    # 1순위: 박스 왼쪽-위 바깥(상단)
    tx, ty = int(x1), int(y1 - th - margin)

    # 여백 부족하면 2순위: 박스 오른쪽-아래 바깥
    if ty < 0:
        tx = int(x2 - tw)
        ty = int(y2 + margin)

    # 화면 안으로 클램프
    tx = max(0, min(tx, W - tw - 1))
    # 배경 사각형 높이는 th + margin으로 살짝 여유
    ty = max(0, min(ty, H - (th + margin) - 1))

    # 배경 박스 + 텍스트
    cv2.rectangle(img, (tx, ty), (tx + tw, ty + th + margin), color_bgr, -1, cv2.LINE_AA)
    cv2.putText(img, text, (tx, ty + th), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)


#############################################################################
# Dummy image for testing
def dummy_image():
    return np.full((480, 640, 3), 255, dtype=np.uint8)
#############################################################################


##################################
#            CLIENT              #
##################################

class YoloClientProcess:
    def __init__(self, boxdata, server_ip="127.0.0.1"):
        self.server_ip = server_ip
        self.boxdata = boxdata
        client_process = Process(target=self.run_client, args=(boxdata,))
        client_process.daemon = True
        client_process.start()

    def run_client(self, boxdata=None):
        # 클래스별 고정 색상
        rng = np.random.default_rng(42)
        colors = (rng.uniform(0, 255, size=(len(NAMES), 3))).astype(np.uint8)

        plane_state = None

        context = zmq.Context()

        # SUB: (topic, meta_json, color_jpg, depth_png, lidar_f32, imu_f32)
        sub_socket = context.socket(zmq.SUB)
        sub_socket.connect(f"tcp://{self.server_ip}:{PORT_IMAGE}")
        sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

        # REQ: keep-alive / YOLO dummy 8 floats
        req_socket = context.socket(zmq.REQ)
        req_socket.connect(f"tcp://{self.server_ip}:{PORT_SYNC}")

        print("[Client] Started")

        # Handshake (same 8 floats)
        yolo_result = [1.0] + [0.0] * 7
        req_socket.send(struct.pack("8f", *yolo_result))
        try:
            ack = req_socket.recv()
            print(f"[Client] Sent YOLO info, got: {ack.decode(errors='ignore')}")
        except Exception:
            print("[Client] Handshake ack recv failed (continue)")

        cv2.namedWindow("Client View", cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                # Keep REQ/REP alive
                try:
                    req_socket.send(struct.pack("8f", *([1.0] + [0.0]*7)))
                    _ = req_socket.recv()
                except Exception:
                    pass

                # Receive multipart
                try:
                    parts = sub_socket.recv_multipart(copy=False)
                except Exception as e:
                    print(f"[Client] recv_multipart error: {e}")
                    continue

                print(len(parts))
                # Expecting 4 or 6 parts
                if len(parts) == 6:
                    topic_b, meta_b, color_b, depth_b, lidar_b, imu_b = parts
                elif len(parts) == 4:
                    topic_b, meta_b, color_b, depth_b = parts
                    lidar_b = None
                    imu_b = None
                else:
                    print(f"[Client] unexpected parts: {len(parts)}")
                    continue

                # Parse meta
                try:
                    meta = json.loads(meta_b.bytes.decode("utf-8"))
                except Exception:
                    meta = {}

                # Color
                color_image = None
                try:
                    np_img = np.frombuffer(color_b.bytes, dtype=np.uint8)
                    if np_img.size > 0:
                        color_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                except Exception:
                    color_image = None

                # Depth (optional; only for display normalization if needed)
                depth_meta = meta.get("depth", {"format":"none"})
                if depth_meta.get("format") == "png" and depth_b is not None and len(depth_b.bytes) > 0:
                    depth_np = np.frombuffer(depth_b.bytes, dtype=np.uint8)
                    depth_image = cv2.imdecode(depth_np, cv2.IMREAD_UNCHANGED)

                if color_image is not None:
                    Hc, Wc, _ = color_image.shape
                else:
                    Hc, Wc = 480, 640

                if depth_meta.get("format") == "png":
                    intr = depth_meta.get("intrinsics", {})
                    intr_list = [intr.get("ppx",0), intr.get("ppy",0), intr.get("fx",0), intr.get("fy",0)] + list(intr.get("coeffs", [0,0,0,0,0]))
                    depth_scale = float(depth_meta["scale"])
                    print(f"[Client] Depth scale: {depth_scale}")
                    depth_intrinsics = make_rs_intrinsics(intr_list, width=Wc, height=Hc)  # 필요시 사용

                # LiDAR points (float32, Nx3) — shape from meta["lidar"]["shape"][0]
                xyz = np.empty((0, 3), np.float32)
                try:
                    if lidar_b is not None and len(lidar_b.bytes) > 0:
                        npts = int(meta.get("lidar", {}).get("shape", [0])[0])
                        raw = np.frombuffer(lidar_b.bytes, dtype=np.float32)
                        xyz = raw.reshape(npts, 3)
                except Exception:
                    xyz = np.empty((0, 3), np.float32)

                #cv2.imshow("Client View", color_image)

                ''' # DepthMap 시각화
                depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
                depth_display = depth_display.astype(np.uint8)
                depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)
                cv2.imshow("Depth", depth_display)
                '''

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
                        color = tuple(int(c) for c in colors[cls_id])  # BGR 튜플

                        # 바운딩 박스 시각화
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                        label = f"{name} {conf:.2f}"
                        draw_label(color_image, label, x1, y1, x2, y2, color)  

                        # 키포인트: 지정한 클래스만 처리
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

                            # 키포인트 시각화
                            for (kx, ky) in valid_xy:
                                cv2.circle(color_image, (int(kx), int(ky)), 3, (0, 255, 0), -1)

                            # 조회용 딕셔너리(키: 원래 키포인트 id)
                            lookup = {
                                int(k): {"xy": (float(kpts_xy[k, 0]), float(kpts_xy[k, 1])),
                                         "conf": float(kpts_conf[k])}
                                for k in valid_idx
                            }

                            need_ids = [0, 1, 2, 3]

                            # 1) 이번 프레임의 3D 포인트를 한 번만 복원해서 캐싱
                            P3 = {}  # {kpt_id: np.array([X,Y,Z])}
                            for k in need_ids:
                                if k in lookup:
                                    pt3 = kp3d(k, lookup, depth_image, depth_intrinsics, depth_scale)  # (3,) or None
                                    if pt3 is not None:
                                        P3[k] = np.array(pt3, dtype=float)

                            print(P3)

                            conf_dict = {k: float(kpts_conf[k]) for k in P3.keys()}
                            present, outliers, pair_err = validate_points_by_distance(P3, D_REF, conf=conf_dict,
                                                                                    abs_tol=0.03, rel_tol=0.05, plane_state=plane_state)

                            print(present)

                            # 2) 초기화 이후: 가시 포인트 수에 따라 갱신/추정 (초기화 코드는 하단)
                            if plane_state is not None:
                                if len(present) == 4 and set(present) == set(need_ids):
                                    # 4점 모두 보이면 재초기화
                                    p = [P3[k] for k in need_ids]
                                    #P = np.vstack(p)
                                    #box_center_3d = P.mean(axis=0)
                                    plane_state = TrackedPlane(*p)

                                elif len(present) == 3:
                                    # 3점: Kabsch(강체)로 R,t 추정 → c_hat
                                    tri = next((c for c in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)] if all(k in present for k in c)), None)
                                    
                                    if tri is not None:
                                        P_new = [P3[k] for k in tri]
                                        w = [float(kpts_conf[k]) for k in tri]  # conf 가중치
                                        out = plane_state.estimate_center_from_three_points(tri, P_new, w=w)
                                        if out is not None:
                                            c_hat, R, t, res = out
                                            
                                            # plane_state 갱신
                                            plane_state.c0   = c_hat
                                            plane_state.pts0 = (R @ plane_state.pts0.T).T + t
                                            plane_state.n    = (R @ plane_state.n); plane_state.n /= (np.linalg.norm(plane_state.n)+1e-12)
                                            #box_center_3d = c_hat
                                        else:
                                            pair = next((p for p in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)] if p[0] in present and p[1] in present), None)

                                            if pair is not None:
                                                i0, j0 = pair
                                                pi_new, pj_new = P3[i0], P3[j0]
                                                wi = float(kpts_conf[i0]); wj = float(kpts_conf[j0])
                                                c_hat, R, t, residual, r = plane_state.estimate_center_from_two(
                                                    i0, j0, pi_new, pj_new, wi=wi, wj=wj
                                                )

                                                # plane_state 갱신
                                                plane_state.c0   = c_hat
                                                plane_state.pts0 = (R @ plane_state.pts0.T).T + t
                                                plane_state.n    = (R @ plane_state.n); plane_state.n /= (np.linalg.norm(plane_state.n)+1e-12)
                                                #box_center_3d = c_hat

                                elif len(present) == 2:
                                    # 2점: 평면 내 회전 + 평행이동(스케일 무시, 강체)
                                    pair = next((p for p in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)] if p[0] in present and p[1] in present), None)

                                    if pair is not None:
                                        i0, j0 = pair
                                        pi_new, pj_new = P3[i0], P3[j0]
                                        wi = float(kpts_conf[i0]); wj = float(kpts_conf[j0])
                                        c_hat, R, t, residual, r = plane_state.estimate_center_from_two(
                                            i0, j0, pi_new, pj_new, wi=wi, wj=wj
                                        )

                                        # plane_state 갱신
                                        plane_state.c0   = c_hat
                                        plane_state.pts0 = (R @ plane_state.pts0.T).T + t
                                        plane_state.n    = (R @ plane_state.n); plane_state.n /= (np.linalg.norm(plane_state.n)+1e-12)
                                        #box_center_3d = c_hat

                                elif len(present) == 1:
                                    # 1점: 평행이동만
                                    k = present[0]
                                    c_hat = plane_state.estimate_center_from_one_idx(k, P3[k])

                                    # plane_state 갱신
                                    plane_state.c0   = c_hat
                                    plane_state.pts0 = plane_state.pts0 + (P3[k] - plane_state.pts0[k])
                                    #box_center_3d = c_hat

                            # 3) plane_state 없으면 4점으로 초기화
                            if plane_state is None and len(present) == 4 and set(present) == set(need_ids):
                                p = [P3[k] for k in need_ids]  # 순서 0,1,2,3
                                #P = np.vstack(p)
                                #box_center_3d = P.mean(axis=0)
                                plane_state = TrackedPlane(*p)

                # boxdata와 plane_state.c0 동기화
                if plane_state is not None:
                    boxdata[:] = np.array(plane_state.c0, dtype=np.float32)
                else:
                    boxdata[:] = np.array([0, 0, 0], dtype=np.float32)

                if plane_state is not None:
                    c0_text = f"({plane_state.c0[0]:.2f}, {plane_state.c0[1]:.2f}, {plane_state.c0[2]:.2f})"
                    center_pixel = project_point_to_pixel(depth_intrinsics, plane_state.c0)
                    u, v = map(int, center_pixel)
                    cv2.circle(color_image, (u, v), radius=5, color=(0, 0, 255), thickness=-1)
                    cv2.putText(color_image, c0_text, (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if color_image is not None:
                    cv2.imshow("Client View", color_image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                       break
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"[Client] Exception: {e}")
            traceback.print_exc()
        finally:
            try: req_socket.close()
            except: pass
            try: sub_socket.close()
            except: pass
            try: context.term()
            except: pass
            cv2.destroyAllWindows()

        req_socket.close()
        sub_socket.close()
        context.term()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # multiprocessing.Array를 사용하여 boxdata 공유
    boxdata = Array('f', [0.0, 0.0, 0.0])  # 3D 좌표 (x, y, z)
    
    # YoloClientProcess 인스턴스 생성
    yolo_client = YoloClientProcess(boxdata, server_ip="192.168.123.164")
    
    # 메인 프로세스에서 boxdata 모니터링
    try:
        while True:
            print(f"Box center: ({boxdata[0]:.3f}, {boxdata[1]:.3f}, {boxdata[2]:.3f})")
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Stopping YOLO client...")
