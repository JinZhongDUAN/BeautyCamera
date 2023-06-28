from PyQt5 import QtWidgets, QtGui
import sys
from PyQt5.QtWidgets import QFileDialog, QGraphicsScene, QMessageBox, QWidget
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import dlib
import numpy as np
from matplotlib import pyplot as plt
import GUI
from src import camera_window
from src.thin_face import face_thin_auto
from src.enlarge_eyes import big_eye

class MainWindow():
    def __init__(self):
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        self.raw_image = None
        self.current_img = None
        self.last_image = None
        self.faces = None
        self.ui = GUI.Ui_MainWindow()
        self.ui.setupUi(MainWindow)
        self.action_connect()
        self.scale_factor=1.0
        MainWindow.show()
        sys.exit(app.exec_())

    # 信号槽绑定
    def action_connect(self):
        self.ui.action.triggered.connect(self.open_file)
        self.ui.action_2.triggered.connect(self.save_file)
        self.ui.action_5.triggered.connect(self.recover_img)
        self.ui.action_8.triggered.connect(self.enlarge_img)
        self.ui.action_10.triggered.connect(self.reduce_img)
        # 打开摄像头
        self.ui.action_12.triggered.connect(self.new_camera)
        # 标记人脸位置
        self.ui.action_12.triggered.connect(self.mark_face)
        # 撤销操作
        self.ui.action_13.triggered.connect(self.face_detec)
        self.ui.action_14.triggered.connect(self.revocat_img)
        self.ui.buttonBox.accepted.connect(self.add_text)
        self.ui.buttonBox.rejected.connect(self.reset_text)
        self.ui.buttonBox_2.accepted.connect(self.add_paster)
        self.ui.buttonBox_2.rejected.connect(self.reset_paster)
        # 涂鸦
        self.ui.pushButton.clicked.connect(self.draw_doodle)
        # 饱和度
        self.ui.horizontalSlider.valueChanged.connect(self.slider_change)
        self.ui.horizontalSlider.sliderReleased.connect(self.show_histogram)

        # 亮度
        self.ui.horizontalSlider_4.valueChanged.connect(self.slider_change)
        self.ui.horizontalSlider_4.sliderReleased.connect(self.show_histogram)

        # 人脸美白
        self.ui.horizontalSlider_8.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_8.sliderReleased.connect(self.show_histogram)

        # 皮肤美白
        self.ui.horizontalSlider_13.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_13.sliderReleased.connect(self.show_histogram)

        # 瘦脸
        self.ui.horizontalSlider_14.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_14.sliderReleased.connect(self.show_histogram)

        # 磨皮
        self.ui.horizontalSlider_11.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_11.sliderReleased.connect(self.show_histogram)

        # 对比度
        self.ui.horizontalSlider_5.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_5.sliderReleased.connect(self.show_histogram)

        # 人脸识别和皮肤检测
        self.ui.tabWidget.tabBarClicked.connect(self.calculate)

        # 木刻滤镜
        self.ui.horizontalSlider_9.sliderReleased.connect(self.woodcut)
        self.ui.horizontalSlider_9.sliderReleased.connect(self.show_histogram)

        # 灰色铅笔画
        self.ui.horizontalSlider_7.sliderReleased.connect(self.pencil_gray)
        self.ui.horizontalSlider_7.sliderReleased.connect(self.show_histogram)

        # 怀旧滤镜
        self.ui.horizontalSlider_10.sliderReleased.connect(self.reminiscene)
        self.ui.horizontalSlider_10.sliderReleased.connect(self.show_histogram)

        # 铅笔画滤镜
        self.ui.horizontalSlider_12.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_12.sliderReleased.connect(self.show_histogram)

        # 风格化
        self.ui.horizontalSlider_2.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_2.sliderReleased.connect(self.show_histogram)

        # 哈哈镜
        self.ui.horizontalSlider_6.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_6.sliderReleased.connect(self.show_histogram)

        # 细节增强
        self.ui.horizontalSlider_23.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_23.sliderReleased.connect(self.show_histogram)

        # 色温
        self.ui.horizontalSlider_3.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_3.sliderReleased.connect(self.show_histogram)

        # 马赛克
        self.ui.horizontalSlider_15.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_15.sliderReleased.connect(self.show_histogram)

        # 锐化
        self.ui.horizontalSlider_21.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_21.sliderReleased.connect(self.show_histogram)

        # 边缘保持
        self.ui.horizontalSlider_24.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_24.sliderReleased.connect(self.show_histogram)

        # 大眼
        self.ui.horizontalSlider_22.sliderReleased.connect(self.slider_change)
        self.ui.horizontalSlider_22.sliderReleased.connect(self.show_histogram)

    # 显示图片
    def show_image(self):
        img_cv = cv2.cvtColor(self.current_img, cv2.COLOR_RGB2BGR)
        img_width, img_height, a = img_cv.shape
        ratio_img = img_width / img_height
        ratio_scene = self.ui.graphicsView.width() / self.ui.graphicsView.height()
        if ratio_img > ratio_scene:
            width = int(self.ui.graphicsView.width())
            height = int(self.ui.graphicsView.width() / ratio_img)
        else:
            width = int(self.ui.graphicsView.height() * ratio_img)
            height = int(self.ui.graphicsView.height())
        img_resize = cv2.resize(img_cv, (height - 5, width - 5), interpolation=cv2.INTER_AREA)
        h, w, c = img_resize.shape
        bytesPerLine = w * 3
        qimg = QImage(img_resize.data, w, h, bytesPerLine, QImage.Format_RGB888)
        self.scene = QGraphicsScene()
        pix = QPixmap(qimg)
        self.scene.addPixmap(pix)
        self.ui.graphicsView.setScene(self.scene)

    # 显示灰度图像
    def show_grayimage(self):
        img_cv = self.gray_image
        img_width, img_height = img_cv.shape
        ratio_img = img_width / img_height
        ratio_scene = self.ui.graphicsView.width() / self.ui.graphicsView.height()
        if ratio_img > ratio_scene:
            width = int(self.ui.graphicsView.width())
            height = int(self.ui.graphicsView.width() / ratio_img)
        else:
            width = int(self.ui.graphicsView.height() * ratio_img)
            height = int(self.ui.graphicsView.height())
        img_resize = cv2.resize(img_cv, (height - 5, width - 5), interpolation=cv2.INTER_AREA)
        h, w = img_resize.shape
        qimg = QImage(img_resize.data, w, h, w, QImage.Format_Grayscale8)
        self.scene = QGraphicsScene()
        pix = QPixmap(qimg)
        self.scene.addPixmap(pix)
        self.ui.graphicsView.setScene(self.scene)

    # 显示直方图
    def show_histogram(self):
        if self.raw_image is None:
            return 0
        img = self.current_img
        plt.figure(figsize=((self.ui.tab_3.width() - 10) / 100, (self.ui.tab_3.width() - 60) / 100), frameon=False)
        plt.hist(img.ravel(), bins=256, range=[0, 256])
        plt.axes().get_yaxis().set_visible(False)
        # plt.axes().get_xaxis().set_visible(False)
        ax = plt.axes()
        # 隐藏坐标系的外围框线
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.savefig('Hist.png', bbox_inches="tight", transparent=True, dpi=100)
        pix = QPixmap("Hist.png")
        self.ui.label.setPixmap(pix)
        self.ui.label_2.setPixmap(pix)
        self.ui.label_3.setPixmap(pix)
        self.ui.label_4.setPixmap(pix)
        self.ui.label_5.setPixmap(pix)
        self.ui.label_26.setPixmap(pix)
        self.ui.label_27.setPixmap(pix)
        self.ui.label_32.setPixmap(pix)

    # 保存图片
    def save_file(self):
        fname = QFileDialog.getSaveFileName(None, '打开文件', './', ("Images (*.png *.xpm *.jpg)"))
        if fname[0]:
            cv2.imwrite(fname[0], self.current_img)

    # 打开图片
    def open_file(self):
        fname = QFileDialog.getOpenFileName(None, '打开文件', './', ("Images (*.png *.xpm *.jpg)"))
        if fname[0]:
            img_cv = cv2.imdecode(np.fromfile(fname[0], dtype=np.uint8), -1)  # 注意这里读取的是RGB空间的
            self.raw_image = img_cv
            self.last_image = img_cv
            self.current_img = img_cv
            self.show_image()
            self.show_histogram()
            self.imgskin = np.zeros(self.raw_image.shape)
        self.intial_value()

    # 还原图片
    def recover_img(self):
        self.current_img = self.raw_image
        global scale_factor
        scale_factor = 1.0
        self.show_image()
        self.show_histogram()
        self.intial_value()

    # 撤销
    def revocat_img(self):
        if self.current_img is None:
            QMessageBox.warning(None, "警告", "当前图像为空！", QMessageBox.Cancel)
        else:
            self.current_img = self.last_image
            self.show_image()
            self.show_histogram()

    # 饱和度
    def change_saturation(self):
        if self.raw_image is None:
            return 0

        value = self.ui.horizontalSlider.value()
        img_hsv = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2HLS)
        if value > 2:
            img_hsv[:, :, 2] = np.log(img_hsv[:, :, 2] / 255 * (value - 1) + 1) / np.log(value + 1) * 255
        if value < 0:
            img_hsv[:, :, 2] = np.uint8(img_hsv[:, :, 2] / np.log(- value + np.e))
        self.current_img = cv2.cvtColor(img_hsv, cv2.COLOR_HLS2BGR)

    # 明度调节
    def change_darker(self):
        if self.raw_image is None:
            return 0
        value = self.ui.horizontalSlider_4.value()
        img_hsv = cv2.cvtColor(self.current_img, cv2.COLOR_BGR2HLS)
        if value > 3:
            img_hsv[:, :, 1] = np.log(img_hsv[:, :, 1] / 255 * (value - 1) + 1) / np.log(value + 1) * 255
        if value < 0:
            img_hsv[:, :, 1] = np.uint8(img_hsv[:, :, 1] / np.log(- value + np.e))
        self.last_image = self.current_img
        self.current_img = cv2.cvtColor(img_hsv, cv2.COLOR_HLS2BGR)

    #人脸关键特征点检测
    def face_detec(self):
        if self.current_img is None:
            QMessageBox.warning(None, "警告", "当前图像为空！", QMessageBox.Cancel)
        else:
            self.faces = self.detect_face()
            self.mark_face()
            # 加载dlib的人脸检测器和人脸关键点检测器
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor('face_det/shape_predictor_68_face_landmarks.dat')
            # 读取输入图像
            image = self.current_img
            self.last_image = self.current_img
            # 转换图像为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 使用人脸检测器检测图像中的人脸
            faces = detector(gray)
            # 遍历每个检测到的人脸
            for face in faces:
                # 使用人脸关键点检测器检测人脸关键点
                landmarks = predictor(gray, face)

                # 遍历每个关键点，并在图像上绘制圆圈
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            self.current_img = image
            self.show_image()

    # 人脸识别
    def detect_face(self):
        # 获取图片
        img = self.raw_image
        # 加载人脸检测模型
        face_cascade = cv2.CascadeClassifier('face_det/haarcascade_frontalface_default.xml')
        # 转换为灰度图像，可以更好地识别人脸
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 识别图片中的人脸
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        return faces

    # 皮肤识别
    def detect_skin(self):
        img = self.raw_image
        rows, cols, channals = img.shape
        for r in range(rows):
            for c in range(cols):
                B = img.item(r, c, 0)
                G = img.item(r, c, 1)
                R = img.item(r, c, 2)
                if (abs(R - G) > 15) and (R > G) and (R > B):
                    if (R > 95) and (G > 40) and (B > 20) and (max(R, G, B) - min(R, G, B) > 15):
                        self.imgskin[r, c] = (1, 1, 1)
                    elif (R > 220) and (G > 210) and (B > 170):
                        self.imgskin[r, c] = (1, 1, 1)

    # 皮肤磨皮
    def dermabrasion(self, value=0):
        img = self.current_img
        value = self.ui.horizontalSlider_11.value()
        blur_img = cv2.bilateralFilter(img, value, 75, 75) # 双边滤波
        # 图像融合
        result_img = cv2.addWeighted(img, 0.3, blur_img, 0.7, 0)
        # 锐度调节
        result_pil = Image.fromarray(result_img)
        enh_img = ImageEnhance.Sharpness(result_pil)
        image_sharpened = enh_img.enhance(1.5)
        # 对比度调节
        con_img = ImageEnhance.Contrast(image_sharpened)
        result = con_img.enhance(1.15)
        # 转换回OpenCV图像格式
        result_img = np.array(result)
        self.last_image = self.current_img
        self.current_img = result_img

    # 皮肤美白
    def whitening_skin(self, value=30):
        # value = 30
        value = self.ui.horizontalSlider_13.value()
        img = self.current_img
        imgw = np.zeros(img.shape, dtype='uint8')
        imgw = img.copy()
        # 设置增益
        midtones_add = np.zeros(256)

        for i in range(256):
            midtones_add[i] = 0.667 * (1 - ((i - 127.0) / 127) * ((i - 127.0) / 127))
        # 构建映射关系数组lookup；value表示美白程度
        lookup = np.zeros(256, dtype="uint8")

        for i in range(256):
            red = i
            red += np.uint8(value * midtones_add[red])
            red = max(0, min(0xff, red))
            lookup[i] = np.uint8(red)

        rows, cols, channals = img.shape
        for r in range(rows):
            for c in range(cols):

                if self.imgskin[r, c, 0] == 1:
                    imgw[r, c, 0] = lookup[imgw[r, c, 0]]
                    imgw[r, c, 1] = lookup[imgw[r, c, 1]]
                    imgw[r, c, 2] = lookup[imgw[r, c, 2]]
        self.current_img = imgw

    # 人脸美白
    def whitening_face(self, value=30):
        # value = 30
        value = self.ui.horizontalSlider_8.value()
        img = self.current_img
        imgw = np.zeros(img.shape, dtype='uint8')
        imgw = img.copy()
        midtones_add = np.zeros(256)

        for i in range(256):
            midtones_add[i] = 0.667 * (1 - ((i - 127.0) / 127) * ((i - 127.0) / 127))

        lookup = np.zeros(256, dtype="uint8")

        for i in range(256):
            red = i
            red += np.uint8(value * midtones_add[red])
            red = max(0, min(0xff, red))
            lookup[i] = np.uint8(red)

        # faces可全局变量
        faces = self.faces

        if faces == ():
            rows, cols, channals = img.shape
            for r in range(rows):
                for c in range(cols):
                    imgw[r, c, 0] = lookup[imgw[r, c, 0]]
                    imgw[r, c, 1] = lookup[imgw[r, c, 1]]
                    imgw[r, c, 2] = lookup[imgw[r, c, 2]]

        else:
            x, y, w, h = faces[0]
            rows, cols, channals = img.shape
            x = max(x - (w * np.sqrt(2) - w) / 2, 0)
            y = max(y - (h * np.sqrt(2) - h) / 2, 0)
            w = w * np.sqrt(2)
            h = h * np.sqrt(2)
            rows = min(rows, y + h)
            cols = min(cols, x + w)
            for r in range(int(y), int(rows)):
                for c in range(int(x), int(cols)):
                    imgw[r, c, 0] = lookup[imgw[r, c, 0]]
                    imgw[r, c, 1] = lookup[imgw[r, c, 1]]
                    imgw[r, c, 2] = lookup[imgw[r, c, 2]]

            processWidth = int(max(min(rows - y, cols - 1) / 8, 2))
            for i in range(1, processWidth):
                alpha = (i - 1) / processWidth
                for r in range(int(y), int(rows)):
                    imgw[r, int(x) + i - 1] = np.uint8(
                        imgw[r, int(x) + i - 1] * alpha + img[r, int(x) + i - 1] * (1 - alpha))
                    imgw[r, int(cols) - i] = np.uint8(
                        imgw[r, int(cols) - i] * alpha + img[r, int(cols) - i] * (1 - alpha))
                for c in range(int(x) + processWidth, int(cols) - processWidth):
                    imgw[int(y) + i - 1, c] = np.uint8(
                        imgw[int(y) + i - 1, c] * alpha + img[int(y) + i - 1, c] * (1 - alpha))
                    imgw[int(rows) - i, c] = np.uint8(
                        imgw[int(rows) - i, c] * alpha + img[int(rows) - i, c] * (1 - alpha))
        self.current_img = imgw

    # Gamma矫正(对比度)
    def gamma_trans(self):
        gamma = (self.ui.horizontalSlider_5.value() + 10) / 10
        img = self.current_img
        gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
        gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
        self.last_image = self.current_img
        self.current_img = cv2.LUT(img, gamma_table)
        self.show_image()
        self.show_histogram()

    # 响应滑动条的变化
    def slider_change(self):
        if self.raw_image is None:
            return 0

        self.current_img = self.raw_image

        # 对比度
        if self.ui.horizontalSlider_5.value() != 0:
            self.gamma_trans()

        # 饱和度
        if self.ui.horizontalSlider.value() != 0:
            self.change_saturation()

        if self.ui.horizontalSlider_2.value() != 0:
            pass

        # 色温
        if self.ui.horizontalSlider_3.value() != 0:
            self.adjust_image_temperature()

        # 亮度
        if self.ui.horizontalSlider_4.value() != 0:
            self.change_darker()

        # 人脸美白
        if self.ui.horizontalSlider_8.value() != 0:
            self.whitening_face()

        # 皮肤美白
        if self.ui.horizontalSlider_13.value() != 0:
            self.whitening_skin()

        # 磨皮
        if self.ui.horizontalSlider_11.value() != 0:
            self.dermabrasion()

        # 瘦脸
        if self.ui.horizontalSlider_14.value() != 0:
            self.thin_face()

        # 大眼
        if self.ui.horizontalSlider_22.value() != 0:
            self.enlarge_eyes()

        # 风格化
        if self.ui.horizontalSlider_2.value() != 0:
            self.stylize()

        # 哈哈镜
        if self.ui.horizontalSlider_6.value() != 0:
            self.hahaha_filter()

        # 细节增强
        if self.ui.horizontalSlider_23.value() != 0:
            self.detail_enhance()

        # 边缘保持
        if self.ui.horizontalSlider_24.value() != 0:
            self.edge_preserve()

        # 锐化
        if self.ui.horizontalSlider_21.value() != 0:
            self.sharpen_image()

        # 铅笔画
        if self.ui.horizontalSlider_12.value() != 0:
            self.pencil_color()

        # 马赛克
        if self.ui.horizontalSlider_15.value() != 0:
            self.drawMosaic()

        self.show_image()

    # 计算人脸识别和皮肤识别的基本参数
    def calculate(self):
        if self.raw_image is None:
            return 0
        if self.calculated is False:
            self.faces = self.detect_face()
            if self.faces != ():
                self.detect_skin()
            self.calculated = True

    # 怀旧滤镜
    def reminiscene(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_10.value() == 0:
            self.current_img = self.raw_image
            self.show_image()
            return 0
        img = self.raw_image.copy()
        rows, cols, channals = img.shape
        for r in range(rows):
            for c in range(cols):
                B = img.item(r, c, 0)
                G = img.item(r, c, 1)
                R = img.item(r, c, 2)
                img[r, c, 0] = np.uint8(min(max(0.272 * R + 0.534 * G + 0.131 * B, 0), 255))
                img[r, c, 1] = np.uint8(min(max(0.349 * R + 0.686 * G + 0.168 * B, 0), 255))
                img[r, c, 2] = np.uint8(min(max(0.393 * R + 0.769 * G + 0.189 * B, 0), 255))
        self.current_img = img
        self.show_image()

    # 木刻滤镜
    def woodcut(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_9.value() == 0:
            self.current_img = self.raw_image
            self.show_image()
            return 0
        self.gray_image = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2GRAY)
        gray = self.gray_image
        value = 70 + self.ui.horizontalSlider_9.value()
        rows, cols = gray.shape
        for r in range(rows):
            for c in range(cols):
                if gray[r, c] > value:
                    gray[r, c] = 255
                else:
                    gray[r, c] = 0
        self.gray_image = gray
        self.show_grayimage()

    # 铅笔画(灰度)
    def pencil_gray(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_7.value() == 0:
            # self.current_img = self.raw_image
            self.show_image()
            return 0
        value = self.ui.horizontalSlider_7.value() * 0.05
        dst1_gray, dst1_color = cv2.pencilSketch(self.current_img, sigma_s=50, sigma_r=value, shade_factor=0.04)
        self.gray_image = dst1_gray
        self.current_img = dst1_gray
        self.show_grayimage()

    # 铅笔画(彩色)
    def pencil_color(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_12.value() == 0:
            self.current_img = self.raw_image
            self.show_image()
            return 0
        value = self.ui.horizontalSlider_12.value() * 0.05
        dst1_gray, dst1_color = cv2.pencilSketch(self.current_img, sigma_s=50, sigma_r=value, shade_factor=0.04)
        self.current_img = dst1_color

    # 风格化
    def stylize(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_2.value() == 0:
            self.current_img = self.raw_image
            self.show_image()
            return 0
        value = self.ui.horizontalSlider_2.value() * 0.05
        self.current_img = cv2.stylization(self.current_img, sigma_s=50, sigma_r=value)

    # 细节增强
    def detail_enhance(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_23.value() == 0:
            self.current_img = self.raw_image
            self.show_image()
            return 0
        value = self.ui.horizontalSlider_23.value() * 0.05
        self.current_img = cv2.detailEnhance(self.current_img, sigma_s=50, sigma_r=value)

    # 边缘保持
    def edge_preserve(self):
        if self.raw_image is None:
            return 0
        if self.ui.horizontalSlider_3.value() == 0:
            self.current_img = self.raw_image
            self.show_image()
            return 0
        value = self.ui.horizontalSlider_3.value() * 0.05
        self.current_img = cv2.edgePreservingFilter(self.current_img, flags=1, sigma_s=50, sigma_r=value)

    # 显示摄像照片
    def show_camera(self):
        flag, self.camera_image = self.cap.read()
        show = cv2.resize(self.image, (640, 480))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))

    # 初始化
    def intial_value(self):
        self.calculated = False
        self.ui.horizontalSlider.setValue(0)
        self.ui.horizontalSlider_2.setValue(0)
        self.ui.horizontalSlider_3.setValue(0)
        self.ui.horizontalSlider_4.setValue(0)
        self.ui.horizontalSlider_5.setValue(0)
        self.ui.horizontalSlider_6.setValue(0)
        self.ui.horizontalSlider_7.setValue(0)
        self.ui.horizontalSlider_8.setValue(0)
        self.ui.horizontalSlider_9.setValue(0)
        self.ui.horizontalSlider_10.setValue(0)
        self.ui.horizontalSlider_11.setValue(0)
        self.ui.horizontalSlider_12.setValue(0)
        self.ui.horizontalSlider_13.setValue(0)
        self.ui.horizontalSlider_14.setValue(0)
        self.ui.horizontalSlider_15.setValue(0)
        self.ui.horizontalSlider_16.setValue(0)
        self.ui.horizontalSlider_17.setValue(0)
        self.ui.horizontalSlider_18.setValue(0)
        self.ui.horizontalSlider_19.setValue(0)
        self.ui.horizontalSlider_20.setValue(0)
        self.ui.horizontalSlider_21.setValue(0)
        self.ui.horizontalSlider_22.setValue(0)
        self.ui.horizontalSlider_23.setValue(0)
        self.ui.horizontalSlider_24.setValue(0)
        self.ui.horizontalSlider_25.setValue(0)
        self.ui.horizontalSlider_26.setValue(0)
        self.ui.horizontalSlider_29.setValue(0)

    # 调用摄像头窗口
    def new_camera(self):
        Dialog = QtWidgets.QDialog()
        self.ui_2 = camera_window.Ui_Form()
        self.ui_2.setupUi(Dialog)
        Dialog.show()
        self.ui_2.pushButton_2.clicked.connect(self.get_image)
        Dialog.exec_()
        if self.ui_2.cap.isOpened():
            self.ui_2.cap.release()
        if self.ui_2.timer_camera.isActive():
            self.ui_2.timer_camera.stop()

    # 获取摄像头的图片
    def get_image(self):
        if self.ui_2.captured_image is not None:
            self.raw_image = self.ui_2.captured_image
            self.current_img = self.ui_2.captured_image
            self.show_image()
            self.show_histogram()
            self.imgskin = np.zeros(self.raw_image.shape)
            self.intial_value()

    # 显示人脸识别
    def mark_face(self):
        if self.raw_image is None:
            return 0
        if self.calculated == False:
            self.calculate()
        for (x, y, w, h) in self.faces:
            self.current_img = cv2.rectangle(self.current_img.copy(), (x, y), (x + w, y + h), (255, 0, 0), 1)
        self.show_image()

    # 马赛克
    def drawMosaic(self):
        if self.current_img is None:
            QMessageBox.warning(None, "警告", "当前图像为空！", QMessageBox.Cancel)
        else:
            image = self.current_img
            self.last_image = self.current_img
            scale = self.ui.horizontalSlider_15.value()
            image = Image.fromarray(image)
            image.thumbnail((401-40*scale, 401-40*scale), Image.LANCZOS)
            image = np.array(image)
            self.current_img = image

    # 瘦脸
    def thin_face(self, scale_factor=0):
        scale_factor = self.ui.horizontalSlider_14.value()
        image = self.current_img
        self.last_image = self.current_img
        self.current_img = face_thin_auto(image, scale_factor * 0.2)

    # 大眼
    def enlarge_eyes(self, scale_factor=0):
        scale_factor = self.ui.horizontalSlider_22.value()
        image = self.current_img
        self.last_image = self.current_img
        self.current_img = big_eye(image, r_max=40, a=scale_factor * 0.1)

    # 放大图像
    def enlarge_img(self):
        if self.current_img is None:
            QMessageBox.warning(None, "警告", "当前图像为空！", QMessageBox.Cancel)
        else:
            # global scale_factor
            self.scale_factor *= 1.2
            image = self.current_img
            image = Image.fromarray(image)
            # 计算放大后的尺寸
            width = int(image.width * self.scale_factor)
            height = int(image.height * self.scale_factor)

            # 使用双线性插值进行图像的放大
            enlarged_image = image.resize((width, height), Image.BILINEAR)
            enlarged_array = np.array(enlarged_image)
            self.current_img = enlarged_array
            # 获取图像属性
            height, width, channel = enlarged_array.shape
            bytes_per_line = width * channel

            # 创建 QImage 并从 numpy 数组中复制图像数据
            image_array_rgb = cv2.cvtColor(enlarged_array, cv2.COLOR_BGR2RGB)
            qimage = QImage(image_array_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.scene.clear()
            self.scene.addPixmap(pixmap)
            self.ui.graphicsView.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # 缩小图像
    def reduce_img(self):
        if self.current_img is None:
            QMessageBox.warning(None, "警告", "当前图像为空！", QMessageBox.Cancel)
        else:
            # global scale_factor
            self.scale_factor /= 1.2
            image = self.current_img
            image = Image.fromarray(image)
            # 计算放大后的尺寸
            width = int(image.width * self.scale_factor)
            height = int(image.height * self.scale_factor)

            # 使用双线性插值进行图像的放大
            reduced_image = image.resize((width, height), Image.BILINEAR)
            reduced_array = np.array(reduced_image)

            self.current_img = reduced_array
            # 获取图像属性
            height, width, channel = reduced_array.shape
            bytes_per_line = width * channel

            # 创建 QImage 并从 numpy 数组中复制图像数据
            image_array_rgb = cv2.cvtColor(reduced_array, cv2.COLOR_BGR2RGB)
            qimage = QImage(image_array_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            self.scene.clear()
            self.scene.addPixmap(pixmap)

            # 缩放图像以适应视图，并将其居中显示
            self.ui.graphicsView.fitInView(self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    # 添加文本
    def add_text(self):
        if self.current_img is None:
            QMessageBox.warning(None, "警告", "当前图像为空！", QMessageBox.Cancel)
        else:
            image = self.current_img
            self.last_image = self.current_img
            image = Image.fromarray(image)
            base = image.convert("RGBA")
            x = self.ui.horizontalSlider_17.value()
            y = self.ui.horizontalSlider_18.value()
            height, width, _ = self.current_img.shape
            if x > width:
                x = width
            if y > height:
                y = height
            txt = Image.new("RGBA", base.size, (255, 255, 255, 0))
            text = self.ui.lineEdit.text()
            # get a font
            fnt = ImageFont.truetype(r"face_det/微软雅黑.ttf", 40)
            if text is not None:
                # get a drawing context
                d = ImageDraw.Draw(txt)
                # draw text, half opacity
                d.text((x, y), text, font=fnt, fill=(255, 0, 0, 255))
                out = Image.alpha_composite(base, txt)
                out_array = np.array(out)
                self.current_img = out_array
                self.show_image()

    # 恢复
    def reset_text(self):
        self.ui.horizontalSlider_17.setValue(0)
        self.ui.horizontalSlider_18.setValue(0)
        self.ui.lineEdit.clear()

    # 贴纸
    def add_paster(self):
        if self.current_img is None:
            QMessageBox.warning(None, "警告", "当前图像为空！", QMessageBox.Cancel)
        else:
            image = self.current_img
            self.last_image = self.current_img
            background = Image.fromarray(image)
            x = self.ui.horizontalSlider_19.value()
            y = self.ui.horizontalSlider_20.value()
            height, width, _ = image.shape
            if x > width:
                x = width-90
            if y > height:
                y = height-90
            if self.ui.radioButton.isChecked():
                image_path = 'icon/1.png'
            elif self.ui.radioButton_2.isChecked():
                image_path = 'icon/3.png'
            else:
                return 0
            if image_path is None:
                return 0
            else:
                # 确定贴纸在背景图像上的位置
                mark = Image.open(image_path)
                mark = mark.resize((90, 90))
                layer = Image.new('RGBA', background.size, (0, 0, 0, 0))
                layer.paste(mark, (x, y))
                out = Image.composite(layer, background, layer)
                img = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGBA2RGB)
                self.current_img = img
                self.show_image()

    # 恢复
    def reset_paster(self):
        self.ui.horizontalSlider_19.setValue(0)
        self.ui.horizontalSlider_20.setValue(0)

    # 哈哈镜
    def hahaha_filter(self):
        image = self.current_img
        self.last_image = self.current_img
        strength = self.ui.horizontalSlider_6.value()
        # 获取图像尺寸
        height, width, _ = image.shape
        # 计算图像中心坐标
        center_x = width / 2
        center_y = height / 2
        # 创建输出图像
        output = np.zeros_like(image)

        # 循环遍历图像的每个像素
        for y in range(height):
            for x in range(width):

                # 计算当前像素相对于图像中心的偏移量
                dx = x - center_x
                dy = y - center_y

                # 计算球面坐标
                theta = np.arctan2(dy, dx)
                radius = np.sqrt(dx ** 2 + dy ** 2)

                # 根据球面坐标计算新的坐标位置
                new_radius = radius + strength * np.sin(theta * 4)
                new_x = int(center_x + new_radius * np.cos(theta))
                new_y = int(center_y + new_radius * np.sin(theta))

                # 检查新的坐标是否在图像范围内
                if 0 <= new_x < width and 0 <= new_y < height:
                    # 将像素值从原始图像复制到输出图像
                    output[y, x] = image[new_y, new_x]
        self.current_img = output

    # 锐化
    def sharpen_image(self):
        image = self.current_img
        self.last_image = self.current_img
        weight = self.ui.horizontalSlider_21.value()
        # 定义Laplace算子
        laplacian_kernel = np.array([[1, 1, 1],
                                    [1, -8, 1],
                                    [1, 1, 1]], dtype=np.float32)

        # 对图像应用Laplace算子
        laplacian = cv2.filter2D(image, cv2.CV_32F, laplacian_kernel)

        # 对应用Laplace算子的结果进行图像锐化
        sharpened_image = image - weight * laplacian

        # 将像素值限制在0-255范围内
        sharpened_image = np.clip(sharpened_image, 0, 255)

        # 将图像转换为8位无符号整型
        sharpened_image = sharpened_image.astype(np.uint8)
        self.current_img = sharpened_image

    # 色温
    def adjust_image_temperature(self):
        image = self.current_img
        self.last_image = self.current_img
        temperature = self.ui.horizontalSlider_3.value() * 0.1
        # 色温调整
        adjusted_image = np.zeros(image.shape, dtype=np.float32)
        matrix = np.zeros((3, 3))

        # 计算颜色校正矩阵
        matrix[0, 0] = 1.0 / temperature  # 红色通道校正系数
        matrix[1, 1] = 1.0  # 绿色通道校正系数
        matrix[2, 2] = 1.0 * temperature  # 蓝色通道校正系数

        # 应用颜色校正矩阵
        adjusted_image = cv2.transform(image, matrix)

        # 将图像转换回整数数据类型（uint8）
        adjusted_image = adjusted_image.astype(np.uint8)
        self.current_img = adjusted_image

    # 涂鸦
    def draw_doodle(self):
        image = self.current_img
        self.last_image = self.current_img
        if image is None:
            QMessageBox.warning(None, "警告", "当前图像为空！", QMessageBox.Cancel)
        else:
            x1 = self.ui.horizontalSlider_25.value()
            x2 = self.ui.horizontalSlider_29.value()
            y1 = self.ui.horizontalSlider_16.value()
            y2 = self.ui.horizontalSlider_26.value()
            height, width, _ = image.shape
            if x1 > width:
                x1 = width-100
            if y1 > height:
                y1 = height-100
            # 设置椭圆的中心坐标
            center = (x1, y1)
            # 设置椭圆的轴长（长轴长度、短轴长度）
            axes_length = (100, 50)
            # 设置椭圆的旋转角度（逆时针为正）
            angle = 0
            # 设置椭圆的起始角度和结束角度（逆时针方向，单位为度）
            start_angle = 0
            end_angle = 360
            # 设置颜色（BGR格式）
            color = (0, 0, 255)
            # 设置粗细
            thickness = 2

            if x2 > width:
                x2 = width-100
            if y2 > height:
                y2 = height-100
            # 设置矩形的左上角和右下角坐标
            top_left = (x2-45, y2-30)
            bottom_right = (x2+45, y2+30)

            if (x1 == 0 and y1 == 0) and (x2 == 0 and y2 == 0):
                QMessageBox.warning(None, "警告", "请正确选择起点和终点！", QMessageBox.Cancel)
            if x1 != 0 or y1 != 0:
                # 在图像上绘制椭圆
                cv2.ellipse(image, center, axes_length, angle, start_angle, end_angle, color, thickness)
            if x2 != 0 or y2 != 0:
                # 在图像上绘制矩形
                cv2.rectangle(image, top_left, bottom_right, color, thickness)
        self.current_img = image
        self.show_image()
        self.show_histogram()


if __name__ == "__main__":
    MainWindow()
