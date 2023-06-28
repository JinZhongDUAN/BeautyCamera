import cv2
import numpy as np
import math
import dlib

predictor_path = 'face_det/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def landmark_dec_dlib_fun(img_src):
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    land_marks = []

    rects = detector(img_gray, 0)
    for i in range(len(rects)):
        landmarks = predictor(img_gray, rects[i])
        land_marks_node = np.array([[p.x, p.y] for p in landmarks.parts()])
        land_marks.append(land_marks_node)

    return land_marks


def bilinear_interpolation(img, vector_u, c):
    ux, uy = vector_u
    x1, x2 = int(ux), int(ux + 1)
    y1, y2 = int(uy), int(uy + 1)

    f_x_y1 = (x2 - ux) * img[y1, x1, c] + (ux - x1) * img[y1, x2, c]
    f_x_y2 = (x2 - ux) * img[y2, x1, c] + (ux - x1) * img[y2, x2, c]

    f_x_y = (y2 - uy) * f_x_y1 + (uy - y1) * f_x_y2
    return int(f_x_y)


def local_scaling_warps(img, cx, cy, r_max, a):
    img1 = np.copy(img)
    for y in range(cy - r_max, cy + r_max + 1):
        d = int(math.sqrt(r_max ** 2 - (y - cy) ** 2))
        x0 = cx - d
        x1 = cx + d
        for x in range(x0, x1 + 1):
            r = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            for c in range(3):
                vector_c = np.array([cx, cy])
                vector_r = np.array([x, y]) - vector_c
                f_s = (1 - ((r / r_max - 1) ** 2) * a)
                vector_u = vector_c + f_s * vector_r
                img1[y, x, c] = bilinear_interpolation(img, vector_u, c)
    return img1


def big_eye(img, r_max, a, left_eye_pos=None, right_eye_pos=None):
    img0 = img.copy()
    # 如果没有提供left_eye_pos或right_eye_pos，则使用landmark_dec_dlib_fun检测关键点
    if left_eye_pos is None or right_eye_pos is None:
        landmarks = landmark_dec_dlib_fun(img0)
        # 如果没有检测到关键点，则返回原始图像
        if len(landmarks) == 0:
            return img
        landmarks_node = landmarks[0]  # 假设只检测到了一个人脸
        # 通过计算左眼和右眼对应的关键点的平均位置来确定眼睛的位置
        left_eye_pos = np.mean(landmarks_node[36:41], axis=0).astype(int)
        right_eye_pos = np.mean(landmarks_node[42:47], axis=0).astype(int)
    # 在原始图像的副本 (img0) 上绘制圆圈，标记左眼和右眼的位置。
    img0 = cv2.circle(img0, tuple(left_eye_pos), radius=10, color=(0, 0, 255))
    img0 = cv2.circle(img0, tuple(right_eye_pos), radius=10, color=(0, 0, 255))
    # 对图像进行局部缩放变换。r_max参数确定缩放效果的最大半径，a参数控制缩放因子。
    img = local_scaling_warps(img, left_eye_pos[0], left_eye_pos[1], r_max=r_max, a=a)
    img = local_scaling_warps(img, right_eye_pos[0], right_eye_pos[1], r_max=r_max, a=a)
    return img
