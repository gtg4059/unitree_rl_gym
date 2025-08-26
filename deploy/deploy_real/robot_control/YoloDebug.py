from typing import Union
import numpy as np
import traceback

import zmq
import struct
import cv2
from ultralytics import YOLO
from ultralytics.utils.checks import check_yaml
from ultralytics.utils import ROOT, YAML
import pyrealsense2 as rs

# Configuration
PORT_IMAGE = 5555
PORT_SYNC = 6666
YOLO_MODEL_PATH = "./deploy/deploy_real/policy/YOLO/0528_jh_yolov8m_l.pt" #"yolov8_depth/pt/0528_jh_yolov8m_l.pt"
CLASSES = YAML.load(check_yaml('coco128.yaml'))['names']
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
detector = cv2.QRCodeDetector()

def make_rs_intrinsics(depth_intrinsics, width=640, height=480):
    intr = rs.intrinsics()
    intr.width  = int(width)
    intr.height = int(height)
    intr.ppx    = float(depth_intrinsics[0])  # cx
    intr.ppy    = float(depth_intrinsics[1])  # cy  ← ppx로 잘못 넣지 않도록 주의!
    intr.fx     = float(depth_intrinsics[2])
    intr.fy     = float(depth_intrinsics[3])
    intr.model  = rs.distortion.brown_conrady

    # coeffs는 길이 5의 배열입니다. (k1,k2,p1,p2,k3)
    intr.coeffs = [0.0]*5
    for i in range(5):
        intr.coeffs[i] = float(depth_intrinsics[4 + i])
    return intr


# keypoint 주변 depth 데이터로 보정
def get_average_depth(depth_image, x, y,depth_scale, window_size=2):
    depth_values = []
    for dx in range(-window_size, window_size + 1):
        for dy in range(-window_size, window_size + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < depth_image.shape[0] and 0 <= ny < depth_image.shape[1]:
                d = depth_image[ny,nx]*depth_scale
                if d > 0:
                    depth_values.append(d)
    return sum(depth_values) / len(depth_values) if depth_values else 0.0

# 픽셀 좌표를 3D 좌표로 변환
def deproject_pixel_to_point(intrinsics, pixel, depth):
    point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel, depth)
    return point  # [X, Y, Z]
'''
def deproject_pixel_to_point(intrinsics, pixel, depth):
    ppx = intrinsics[0]
    ppy = intrinsics[1]
    fx = intrinsics[2]
    fy = intrinsics[3]
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
'''
# 3D 좌표를 픽셀 좌표로 변환
def project_point_to_pixel(intrinsics, point):
    pixel = rs.rs2_project_point_to_pixel(intrinsics, point)
    return int(pixel[0]), int(pixel[1])
'''
def project_point_to_pixel(intrinsics, point):
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
'''
# 대각선 예외처리
def compute_center_3d(points_3d):
    if len(points_3d) != 4:
        return None  # 예외 처리

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
def compute_pixel_center_from_diagonals(points):
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
def compute_normal_from_points(points):
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
def compute_box_center_from_faces(face_centers_3d, face_normals_3d):
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
def dummy_image():
    return np.full((480, 640, 3), 255, dtype=np.uint8)

##################################
#            CLIENT              #
##################################

def run_client(server_ip="127.0.0.1"):
    context = zmq.Context()

    # SUB socket for receiving image
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://{server_ip}:{PORT_IMAGE}")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

    # REQ socket for sending YOLO result
    req_socket = context.socket(zmq.REQ)
    req_socket.connect(f"tcp://{server_ip}:{PORT_SYNC}")

    model = YOLO(YOLO_MODEL_PATH)
    print("[Client] Started")

    # Create message: 1.0 + 7 dummy float values
    yolo_result = [1.0] + [0.0] * 7
    packed_msg = struct.pack("8f", *yolo_result)

    # Send to server and wait for reply
    req_socket.send(packed_msg)
    ack = req_socket.recv()
    #print(f"[Client] Sent YOLO info, got: {ack.decode()}")

    while True:
        try:
            # Receive image
            box_center_3d=np.array([0,0,0])
            message = sub_socket.recv()
            header_size = struct.calcsize('iii10f')
            header = message[:header_size]
            lidar_len, color_len, depth_len, *depth_intrinsics,depth_scale = struct.unpack('iii10f', header)
            jpg_bytes = message[header_size:header_size + color_len]
            png_bytes = message[header_size + color_len:header_size + color_len + depth_len]
            lid_bytes = message[header_size + color_len + depth_len:header_size + color_len + depth_len+lidar_len]
            np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
            np_lidar=np.frombuffer(lid_bytes,dtype=np.float32)
            xyz = np_lidar.reshape(-1, 3).copy()
            # Mount orientation 보정 (예: upside-down)
            xyz *= np.array([1.0, -1.0, -1.0], dtype=np.float32)
            #print(depth_intrinsics)
            np_img = np.frombuffer(jpg_bytes, dtype=np.uint8)
            if np_img.shape[0] == 0:
                continue
            color_image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            np_depth = np.frombuffer(png_bytes, dtype=np.uint8)
            if np_depth.shape[0] == 0:
                continue
            depth_image = cv2.imdecode(np_depth, cv2.IMREAD_UNCHANGED)
            depth_display = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_display.astype(np.uint8)
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            depth_intrinsics = make_rs_intrinsics(depth_intrinsics, width=640, height=480)

            #cv2.imshow("Depth", depth_display)

            # Run YOLO
            results = model(color_image)
            class_ids = []
            confidences = []
            bboxes = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = box.conf
                    if confidence > 0.5:
                        xyxy = box.xyxy.tolist()[0]
                        bboxes.append(xyxy)
                        confidences.append(float(confidence))
                        class_ids.append(box.cls.tolist())

            result_boxes = cv2.dnn.NMSBoxes(bboxes, confidences, 0.25, 0.45, 0.5)
            for i in range(len(bboxes)):
                # calss id 0번(박스)인 경우
                if int(box.cls) == 0:
                    if i in result_boxes:
                        bbox = list(map(int, bboxes[i]))
                        keypoints = result.keypoints
                        x1, y1, x2, y2 = bbox

                        label = "box"

                        color = colors[i]
                        color = (int(color[0]), int(color[1]), int(color[2]))

                        tl = 2  # line/font thickness
                        tf = max(tl - 1, 1)  # font thickness
                        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                        c2 = x1 + t_size[0], y1 - t_size[1] - 3

                        # color rectangle
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), color, 2)
                        # label rectangle
                        cv2.rectangle(color_image, (x1, y1), c2, color, -1, cv2.LINE_AA)
                        # label
                        cv2.putText(color_image, label, (x1, y1 - 2), 0, tl / 3, [255, 255, 255], thickness=tf,
                                    lineType=cv2.LINE_AA)

                        if keypoints is not None:
                            kps = keypoints.xy[i].cpu().numpy()
                            kcs = keypoints.conf[i].cpu().numpy()

                            keypoints_pixel_dict = {}

                            for idx, (kp, conf) in enumerate(zip(kps, kcs)):
                                kx, ky = int(kp[0]), int(kp[1])
                                valid_pixel = 1
                                if (kx == 0 and ky == 0) or (conf < 0.01):
                                    valid_pixel = 0
                                keypoints_pixel_dict[idx] = {"pixel": (kx, ky), "valid": valid_pixel}

                            diagonals = {
                                "top1": [0, 2],
                                "top2": [1, 3],
                                "left1": [0, 5],
                                "left2": [1, 4],
                                "right1": [1, 6],
                                "right2": [2, 5]
                            }

                            valid_flags = {name: all(keypoints_pixel_dict[idx]["valid"] for idx in indices) for
                                           name, indices in diagonals.items()}

                            faces = {
                                "top": ("top1", "top2"),
                                "left": ("left1", "left2"),
                                "right": ("right1", "right2")
                            }
                            temp_face_centers_pixel = {}

                            # 픽셀로 면의 중점 구하기
                            for face_name, (diag1, diag2) in faces.items():
                                diag1_ok = valid_flags[diag1]
                                diag2_ok = valid_flags[diag2]

                                temp_centers = []
                                if diag1_ok and diag2_ok:
                                    for idx in diagonals[diag1] + diagonals[diag2]:
                                        temp_centers.append(keypoints_pixel_dict[idx]["pixel"])
                                elif diag1_ok:
                                    for idx in diagonals[diag1]:
                                        temp_centers.append(keypoints_pixel_dict[idx]["pixel"])
                                elif diag2_ok:
                                    for idx in diagonals[diag2]:
                                        temp_centers.append(keypoints_pixel_dict[idx]["pixel"])

                                if temp_centers:
                                    points_np = np.array(temp_centers)
                                    center_x, center_y = np.mean(points_np, axis=0)
                                    temp_face_centers_pixel[face_name] = (center_x, center_y)
                                else:
                                    temp_face_centers_pixel[face_name] = None

                            face_keypoints = {
                                "top": [0, 1, 2, 3],
                                "left": [0, 1, 5, 4],
                                "right": [1, 2, 6, 5]
                            }
                            offset = 0.05  # 이동 비율
                            corrected_faces = {}  # 보정된 포인트 저장

                            # 중점 방향으로 keypoint 이동시키기
                            for face_name, indices in face_keypoints.items():
                                face_center = temp_face_centers_pixel.get(face_name)

                                if face_center is None:
                                    corrected_faces[face_name] = None  # 이 face는 invalid
                                    continue

                                face_center = np.array(face_center)
                                corrected_points = []

                                for idx in indices:
                                    pixel = np.array(keypoints_pixel_dict[idx]["pixel"])
                                    valid = keypoints_pixel_dict[idx]["valid"]

                                    if not valid:
                                        corrected_points.append(None)  # invalid 모서리는 None으로 표시
                                        continue

                                    # face 중심 방향으로 offset 이동
                                    direction = face_center - pixel
                                    corrected_pixel = pixel + offset * direction

                                    corrected_points.append(tuple(corrected_pixel.astype(int)))

                                corrected_faces[face_name] = corrected_points

                            face_3d_points = {}  # face별 3D 포인트 저장
                            face_centers_3d = {}  # face별 중심 3D 좌표 저장
                            face_normals_3d = {}  # face별  법선 벡터 좌표 저장

                            # 박스의 중심 구하기
                            for face_name, points in corrected_faces.items():
                                if points is None:
                                    face_3d_points[face_name] = None
                                    face_centers_3d[face_name] = None
                                    continue

                                points_3d = []
                                for pt in points:
                                    if pt is None:
                                        points_3d.append(None)
                                        continue

                                    kx, ky = pt
                                    depth = get_average_depth(depth_image, kx, ky,depth_scale, window_size=2)

                                    if depth == 0.0:
                                        points_3d.append(None)
                                        continue

                                    point_3d = deproject_pixel_to_point(depth_intrinsics, (kx, ky), depth)
                                    #print(depth)
                                    point_3d = np.array(point_3d) * 100  # meter → cm 변환
                                    points_3d.append(point_3d)
                                    #print(point_3d)

                                face_3d_points[face_name] = points_3d
                                face_centers_3d[face_name] = compute_center_3d(points_3d)
                                

                                face_normals_3d[face_name] = compute_normal_from_points(points_3d)

                            ### QR 인식 파트
                            '''found_text = False
                            display_texts = []
                            # scale = 1.0
                            recognized_text = None

                            for face_name, points in corrected_faces.items():
                                if face_name in ["left", "right"] and points is not None:
                                    if found_text:
                                        break

                                    points_3d = face_3d_points.get(face_name)
                                    if points_3d is None or any(p is None for p in points_3d):
                                        continue

                                    src_pts_2d = np.array([
                                        project_point_to_pixel(depth_intrinsics, p / 100.0)  # cm -> m 변환
                                        for p in points_3d
                                    ], dtype=np.float32)

                                    def euclidean(p1, p2):
                                        return np.linalg.norm(np.array(p1) - np.array(p2))

                                    width_3d = euclidean(points_3d[0], points_3d[1])
                                    height_3d = euclidean(points_3d[0], points_3d[3])
                                    scale = 300.0 / max(width_3d, height_3d)
                                    width_px = int(width_3d * scale)
                                    height_px = int(height_3d * scale)

                                    dst_pts_2d = np.array([
                                        [0, 0],
                                        [width_px, 0],
                                        [width_px, height_px],
                                        [0, height_px]
                                    ], dtype=np.float32)

                                    # Homography 및 warp
                                    M = cv2.getPerspectiveTransform(src_pts_2d, dst_pts_2d)
                                    warped = cv2.warpPerspective(color_image, M, (width_px, height_px))

                                    # gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                                    # gray_eq = cv2.equalizeHist(gray)
                                    # gray_eq = cv2.resize(gray_eq, None, fx=2.0, fy=2.0)
                                    warped = cv2.resize(warped, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)

                                    # _, thresh = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                                    # padded = cv2.copyMakeBorder(gray_eq, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)

                                    # QR 코드 인식
                                    retval, decoded_info, qr_points, _ = detector.detectAndDecodeMulti(warped)
                                    if retval:
                                        for data in decoded_info:
                                            display_texts.append(data)
                                            recognized_text = data
                                            print(f"[QR:{face_name}] {data}")
                                            found_text = True
                                            #cv2.imshow(f"{face_name}_roi", warped)
                                            key = cv2.waitKey(1)
                                            break

                                    if not found_text:
                                        decoded = decode(warped)
                                        for obj in decoded:
                                            data = obj.data.decode('utf-8')
                                            if data:
                                                display_texts.append(data)
                                                recognized_text = data
                                                print(f"[QR:{face_name}] {data}")
                                                found_text = True
                                                #cv2.imshow(f"{face_name}_roi", warped)
                                                key = cv2.waitKey(1)
                                                break

                            if found_text:
                                cv2.putText(color_image, recognized_text, (20, 50 + 40 * idx),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)'''

                            ### 박스에 면의 중점 및 박스의 중심 표시
                            for face_name, points in corrected_faces.items():
                                if points is None:
                                    continue

                                # 기본 색상
                                color = (0, 255, 0) if face_name == "top" else (255, 0, 0) if face_name == "left" else (
                                    0, 0, 255)

                                center_3d = face_centers_3d.get(face_name)
                                center_pixel = None
                                exclude_face_from_box_center = False

                                # 중심점 3D → 2D 변환
                                if center_3d is not None:
                                    center_pixel = project_point_to_pixel(depth_intrinsics, center_3d / 100.0)

                                # top 면이면, 실측 depth 비교해서 색상 변경
                                if face_name == "top" and center_3d is not None and center_pixel is not None:
                                    cx, cy = center_pixel
                                    actual_depth = get_average_depth(depth_image, cx, cy,depth_scale, window_size=2)
                                    center_depth = center_3d[2] / 100.0  # cm → m

                                    if actual_depth > center_depth + 0.01:  # 1cm 차이 허용
                                        color = (255, 255, 255)
                                        center_3d = None
                                        exclude_face_from_box_center = True

                                if exclude_face_from_box_center:
                                    face_normals_3d[face_name] = None

                                points_valid = [pt for pt in points if pt is not None]

                                # 외곽선 그리기
                                if len(points_valid) >= 2:
                                    for j in range(len(points_valid)):
                                        pt1 = points_valid[j]
                                        pt2 = points_valid[(j + 1) % len(points_valid)]
                                        cv2.line(color_image, pt1, pt2, color, 2)

                                # 중심점 그리기
                                if center_3d is not None and center_pixel is not None:
                                    cv2.circle(color_image, center_pixel, 5, (0, 255, 255), -1)

                                    label = f"{face_name}\nX:{center_3d[0]:.1f} Y:{center_3d[1]:.1f} Z:{center_3d[2]:.1f}"
                                    for i, line in enumerate(label.split("\n")):
                                        cv2.putText(color_image, line, (center_pixel[0], center_pixel[1] + i * 15),
                                                    cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
                            box_center_3d = compute_box_center_from_faces(face_centers_3d, face_normals_3d)
                            # 박스 중심점 및 x축 표시
                            if box_center_3d is not None:

                                x_unit = None

                                if face_3d_points.get("left") and face_3d_points.get("right"):
                                    pts_left = face_3d_points["left"]
                                    pts_right = face_3d_points["right"]

                                    pt0, pt1 = pts_left[0], pts_left[1]
                                    pt2, pt3 = pts_right[0], pts_right[1]

                                    if pt0 is not None and pt1 is not None:
                                        left_length = np.linalg.norm(pt1 - pt0)
                                        #print(f"[info] Left side (0-1) length:  {left_length:.2f} cm")
                                    else:
                                        left_length = None
                                        print("[warn] left face keypoints 0 or 1 is None → 거리 계산 skip")

                                    if pt2 is not None and pt3 is not None:
                                        right_length = np.linalg.norm(pt3 - pt2)
                                        #print(f"[info] Right side (1-2) length: {right_length:.2f} cm")
                                    else:
                                        right_length = None
                                        print("[warn] right face keypoints 1 or 2 is None → 거리 계산 skip")

                                if (face_centers_3d.get("left") is not None and
                                        face_centers_3d.get("right") is not None and
                                        left_length is not None and
                                        right_length is not None):

                                    left_center = face_centers_3d["left"]
                                    right_center = face_centers_3d["right"]

                                    if left_length < right_length:
                                        # 왼쪽이 더 가까우면 → 오른쪽 방향(x축은 box → right)
                                        x_dir = np.array(box_center_3d) - np.array(right_center)
                                    else:
                                        # 오른쪽이 더 가까우면 → 왼쪽 방향(x축은 box → left)
                                        x_dir = np.array(box_center_3d) - np.array(left_center)

                                    x_unit = x_dir / (np.linalg.norm(x_dir) + 1e-6)

                                    previous_x_unit = x_unit
                                    previous_box_center_3d = box_center_3d

                                else:
                                    print("[warn] 거리 측정 실패 -> 이전 x축 벡터 사용")

                                    if 'previous_x_unit' in globals() and 'previous_box_center_3d' in globals():
                                        x_unit = previous_x_unit
                                        box_center_3d = previous_box_center_3d
                                    else:
                                        x_unit = None

                                box_center_pixel = project_point_to_pixel(depth_intrinsics, box_center_3d / 100.0)
                                cv2.circle(color_image, box_center_pixel, 5, (0, 255, 255), -1)

                                if x_unit is not None:
                                    origin_px = project_point_to_pixel(depth_intrinsics, box_center_3d / 100.0)
                                    x_arrow_end = project_point_to_pixel(depth_intrinsics, (
                                                box_center_3d + x_unit * 10) / 100.0)  # 길이 10cm

                                    cv2.arrowedLine(color_image, origin_px, x_arrow_end, (0, 0, 255), 2)  # 빨강: X축

                                    label = (
                                        f"Box Center\n"
                                        f"X:{box_center_3d[0]:.1f} Y:{box_center_3d[1]:.1f} Z:{box_center_3d[2]:.1f}\n"
                                        f"x_dir: ({x_unit[0]:.2f}, {x_unit[1]:.2f}, {x_unit[2]:.2f})"
                                    )
                                    print(box_center_3d)
                                    for i, line in enumerate(label.split("\n")):
                                        cv2.putText(color_image, line,
                                                    (origin_px[0] + 10, origin_px[1] + i * 15),
                                                    cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)


            # Create message: 1.0 + 7 dummy float values
            #yolo_result = [1.0] +list(box_center_3d)+ [0.0] * 4
            yolo_result = [1.0] + [0.0] * 7
            packed_msg = struct.pack("8f", *yolo_result)

            # Send to server and wait for reply
            req_socket.send(packed_msg)
            ack = req_socket.recv()
            print(f"[Client] Sent YOLO info, got: {ack.decode()}")
            #print(box_center_3d)

            if color_image is not None:
                cv2.imshow("Client View", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                   break
        except Exception as e:
            print(f"[Client] Exception: {e}")
            traceback.print_exc()
            break

    req_socket.close()
    sub_socket.close()
    context.term()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_client(server_ip="192.168.123.164")