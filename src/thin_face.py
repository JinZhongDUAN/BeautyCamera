import dlib
import cv2
import numpy as np
import math

predictor_path = 'face_det/shape_predictor_68_face_landmarks.dat'
# 加载预训练的人脸检测器
detector = dlib.get_frontal_face_detector()
# 加载预训练的人脸关键点检测器
predictor = dlib.shape_predictor(predictor_path)


def landmark_dec_dlib_fun(img_src):
    # 转灰度图
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    land_marks = []
    # 检测人脸
    rects = detector(img_gray, 0)
    # 使用dlib的预测器查找人脸关键点
    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])
        land_marks.append(land_marks_node)
    return land_marks

'''
方法： Interactive Image Warping 局部平移算法
'''


def localTranslationWarp(srcImg, startX, startY, endX, endY, radius):
    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()

    # 计算公式中的|m-c|^2
    ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)
    H, W, C = srcImg.shape
    for i in range(W):
        for j in range(H):
            # 计算该点是否在形变圆的范围之内
            # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中
            if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                continue

            distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)

            if (distance < ddradius):
                # 计算出（i,j）坐标的原坐标
                # 计算公式中右边平方号里的部分
                ratio = (ddradius - distance) / (ddradius - distance + ddmc)
                ratio = ratio * ratio

                # 映射原位置
                UX = i - ratio * (endX - startX)
                UY = j - ratio * (endY - startY)

                # 根据双线性插值法得到UX，UY的值
                value = BilinearInsert(srcImg, UX, UY)
                # 改变当前 i ，j的值
                copyImg[j, i] = value

    return copyImg


def BilinearInsert(src, ux, uy):
    w, h, c = src.shape
    if c == 3:
        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1

        part1 = src[y1, x1].astype(np.float64) * (float(x2) - ux) * (float(y2) - uy)
        part2 = src[y1, x2].astype(np.float64) * (ux - float(x1)) * (float(y2) - uy)
        part3 = src[y2, x1].astype(np.float64) * (float(x2) - ux) * (uy - float(y1))
        part4 = src[y2, x2].astype(np.float64) * (ux - float(x1)) * (uy - float(y1))

        insertValue = part1 + part2 + part3 + part4

        return insertValue.astype(np.int8)


def face_thin_auto(src, strength):
    # 68个关键点二维数组
    landmarks = landmark_dec_dlib_fun(src)

    # 如果未检测到人脸关键点，就不进行瘦脸
    if len(landmarks) == 0:
        return

    for landmarks_node in landmarks:
        # 左第4个特征点
        left_landmark = landmarks_node[3]
        # 左第6个特征点
        left_landmark_down = landmarks_node[5]
        # 右第14个特征点
        right_landmark = landmarks_node[13]
        # 右第16个特征点
        right_landmark_down = landmarks_node[15]
        # 第31个特征点鼻尖
        endPt = landmarks_node[30]
        # 计算第4个特征点到第6个特征点的距离作为瘦脸距离，并乘以瘦脸强度
        r_left = strength * math.sqrt(
            (left_landmark[0, 0] - left_landmark_down[0, 0]) ** 2 +
            (left_landmark[0, 1] - left_landmark_down[0, 1]) ** 2)
        # 计算第14个特征点到第16个特征点的距离作为瘦脸距离，并乘以瘦脸强度
        r_right = strength * math.sqrt(
            (right_landmark[0, 0] - right_landmark_down[0, 0]) ** 2 +
            (right_landmark[0, 1] - right_landmark_down[0, 1]) ** 2)
        # 瘦左边脸
        thin_image = localTranslationWarp(src, left_landmark[0, 0],
                                          left_landmark[0, 1], endPt[0, 0], endPt[0, 1], r_left)
        # 瘦右边脸
        thin_image = localTranslationWarp(thin_image, right_landmark[0, 0],
                                          right_landmark[0, 1], endPt[0, 0], endPt[0, 1], r_right)
    return thin_image
