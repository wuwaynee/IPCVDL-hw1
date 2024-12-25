from typing import Container
import numpy as np
import cv2 as cv
import math

from fileinput import filename
from this import d
from tkinter import NONE
from PyQt5 import QtCore, QtGui, QtWidgets
from itertools import product 
from numpy import zeros, dot, exp, mgrid, pi, ravel, square, uint8, int16


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1002, 527)
        self.load_image = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(1))
        self.load_image.setGeometry(QtCore.QRect(50, 240, 221, 51))
        self.load_image.setObjectName("load_image")
        self.gaussian_blur = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(2))
        self.gaussian_blur.setGeometry(QtCore.QRect(380, 130, 221, 51))
        self.gaussian_blur.setObjectName("gaussian_blur")
        self.sobel_x = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(3))
        self.sobel_x.setGeometry(QtCore.QRect(380, 220, 221, 51))
        self.sobel_x.setObjectName("sobel_x")
        self.sobel_y = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(4))
        self.sobel_y.setGeometry(QtCore.QRect(380, 310, 221, 51))
        self.sobel_y.setObjectName("sobel_y")
        self.magnitude = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(5))
        self.magnitude.setGeometry(QtCore.QRect(380, 400, 221, 51))
        self.magnitude.setObjectName("magnitude")
        self.resize = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(6))
        self.resize.setGeometry(QtCore.QRect(720, 130, 221, 51))
        self.resize.setObjectName("resize")
        self.translation = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(7))
        self.translation.setGeometry(QtCore.QRect(720, 220, 221, 51))
        self.translation.setObjectName("translation")
        self.rotation_scaling = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(8))
        self.rotation_scaling.setGeometry(QtCore.QRect(720, 310, 221, 51))
        self.rotation_scaling.setObjectName("rotation_scaling")
        self.shearing = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(9))
        self.shearing.setGeometry(QtCore.QRect(720, 400, 221, 51))
        self.shearing.setObjectName("shearing")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(380, 60, 191, 41))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(720, 60, 191, 41))
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.load_image.setText(_translate("Form", "Load Image"))
        self.gaussian_blur.setText(_translate("Form", "3.1 Gaussian Blur"))
        self.sobel_x.setText(_translate("Form", "3.2 Sobel X"))
        self.sobel_y.setText(_translate("Form", "3.3 Sobel Y"))
        self.magnitude.setText(_translate("Form", "3.4 Magnitude"))
        self.resize.setText(_translate("Form", "4.1 Resize"))
        self.translation.setText(_translate("Form", "4.2 Translation"))
        self.rotation_scaling.setText(_translate("Form", "4.3 Rotation, Scaling"))
        self.shearing.setText(_translate("Form", "4.4 Shearing"))
        self.label.setText(_translate("Form", "3. Edge Detection"))
        self.label_2.setText(_translate("Form", "4. Transformation"))


    def load(self):
        self.filename, self.filetype = QtWidgets.QFileDialog.getOpenFileName(None, "Open file",".\\") 

    # 3
    def gaussian(self):
        img = cv.imread(self.filename)  
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

        height, width = gray.shape[0], gray.shape[1]
        dst_height = height - 3 + 1
        dst_width = width - 3 + 1

        image_array = zeros((dst_height * dst_width, 3 * 3))
        row = 0
        for i, j in product(range(dst_height), range(dst_width)):
            window = ravel(gray[i : i + 3, j : j + 3])
            image_array[row, :] = window
            row += 1

        center = 3 // 2
        x, y = mgrid[0 - center : 3 - center, 0 - center : 3 - center]
        sigma = math.sqrt(0.5)
        gaussian_kernal = 1 / (2 * pi * square(sigma)) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
        filter_array = ravel(gaussian_kernal)
        dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)
        cv.imshow("gaussian", dst)


    def sobelx(self):
        img = cv.imread(self.filename)  
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

        height, width = gray.shape[0], gray.shape[1]
        dst_height = height - 3 + 1
        dst_width = width - 3 + 1

        image_array = zeros((dst_height * dst_width, 3 * 3))
        row = 0
        for i, j in product(range(dst_height), range(dst_width)):
            window = ravel(gray[i : i + 3, j : j + 3])
            image_array[row, :] = window
            row += 1

        center = 3 // 2
        x, y = mgrid[0 - center : 3 - center, 0 - center : 3 - center]
        sigma = math.sqrt(0.5)
        gaussian_kernal = 1 / (2 * pi * square(sigma)) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
        filter_array = ravel(gaussian_kernal)
        dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

        # sobelX
        container = np.copy(dst)
        size = container.shape
        row = size[0]
        col = size[1]
        
        sobel = np.zeros((row, col), dtype = np.int16)  # avoid overflow
        
        # the given matrix
        for i in range(row):
            for j in range(col):
                    if i == 0 or i == row - 1 or j == 0 or j == col - 1:
                        sobel[i][j] = 0
                    else:
                        # test all int, all float, int with zero, float with zero
                        sobel[i][j] = (-2.0) * container[i][j - 1] + 0.0 * container[i][j] + 2.0 * container[i][j + 1] + (-1.0) * container[i - 1][j - 1] + 0.0 * container[i - 1][j] + 1.0 * container[i - 1][j + 1] + (-1.0) * container[i + 1][j - 1] + 0.0 * container[i + 1][j] + 1.0 * container[i + 1][j + 1]
 
                # gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
                # gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
                # container[i][j] = min(255, np.sqrt(gx**2 + gy**2))

        sobel = cv.convertScaleAbs(sobel)        
        cv.imshow("sobel X", sobel)

        # Gx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        # Gy = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        # [rows, columns] = np.shape(dst)
        # sobel_filtered_image = np.zeros(shape = (rows, columns))
        
        # for i in range(rows - 2):
        #     for j in range(columns - 2):
        #         gx = np.sum(np.multiply(Gx, dst[i : i + 3, j : j + 3]))
        #         gy = np.sum(np.multiply(Gy, dst[i : i + 3, j : j + 3]))
        #         sobel_filtered_image[i + 1, j + 1] = np.sqrt(gx ** 2 + gy ** 2)

        # cv.imshow("sobel", sobel_filtered_image)


    def sobely(self):
        img = cv.imread(self.filename)  
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

        height, width = gray.shape[0], gray.shape[1]
        dst_height = height - 3 + 1
        dst_width = width - 3 + 1

        image_array = zeros((dst_height * dst_width, 3 * 3))
        row = 0
        for i, j in product(range(dst_height), range(dst_width)):
            window = ravel(gray[i : i + 3, j : j + 3])
            image_array[row, :] = window
            row += 1

        center = 3 // 2
        x, y = mgrid[0 - center : 3 - center, 0 - center : 3 - center]
        sigma = math.sqrt(0.5)
        gaussian_kernal = 1 / (2 * pi * square(sigma)) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
        filter_array = ravel(gaussian_kernal)
        dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

        # sobelY
        container = np.copy(dst)
        size = container.shape
        row = size[0]
        col = size[1]
        
        sobel = np.zeros((row, col), dtype = np.int16)  # avoid overflow
        
        # the given matrix
        for i in range(row):
            for j in range(col):
                    if i == 0 or i == row - 1 or j == 0 or j == col - 1:
                        sobel[i][j] = 0
                    else:
                        # test all int, all float, int with zero, float with zero
                        sobel[i][j] = (-2.0) * container[i + 1][j] + 0.0 * container[i][j] + 2.0 * container[i - 1][j] + (-1.0) * container[i + 1][j - 1] + 0.0 * container[i][j - 1] + 1.0 * container[i - 1][j - 1] + (-1.0) * container[i + 1][j + 1] + 0.0 * container[i][j + 1] + 1.0 * container[i - 1][j + 1]
 
                # gx = (img[i - 1][j - 1] + 2*img[i][j - 1] + img[i + 1][j - 1]) - (img[i - 1][j + 1] + 2*img[i][j + 1] + img[i + 1][j + 1])
                # gy = (img[i - 1][j - 1] + 2*img[i - 1][j] + img[i - 1][j + 1]) - (img[i + 1][j - 1] + 2*img[i + 1][j] + img[i + 1][j + 1])
                # container[i][j] = min(255, np.sqrt(gx**2 + gy**2))

        sobel = cv.convertScaleAbs(sobel)        
        cv.imshow("sobel Y", sobel)


    def magni(self):
        img = cv.imread(self.filename)  
        # gaussian
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

        height, width = gray.shape[0], gray.shape[1]
        dst_height = height - 3 + 1
        dst_width = width - 3 + 1

        image_array = zeros((dst_height * dst_width, 3 * 3))
        row = 0
        for i, j in product(range(dst_height), range(dst_width)):
            window = ravel(gray[i : i + 3, j : j + 3])
            image_array[row, :] = window
            row += 1

        center = 3 // 2
        x, y = mgrid[0 - center : 3 - center, 0 - center : 3 - center]
        sigma = math.sqrt(0.5)
        gaussian_kernal = 1 / (2 * pi * square(sigma)) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
        filter_array = ravel(gaussian_kernal)
        dst = dot(image_array, filter_array).reshape(dst_height, dst_width).astype(uint8)

        # sobel X
        container1 = np.copy(dst)
        size = container1.shape
        row = size[0]
        col = size[1]
        
        sobel = np.zeros((row, col), dtype = np.int16)  # avoid overflow
        
        # the given matrix
        for i in range(row):
            for j in range(col):
                    if i == 0 or i == row - 1 or j == 0 or j == col - 1:
                        sobel[i][j] = 0
                    else:
                        # test all int, all float, int with zero, float with zero
                        sobel[i][j] = (-2.0) * container1[i][j - 1] + 0.0 * container1[i][j] + 2.0 * container1[i][j + 1] + (-1.0) * container1[i - 1][j - 1] + 0.0 * container1[i - 1][j] + 1.0 * container1[i - 1][j + 1] + (-1.0) * container1[i + 1][j - 1] + 0.0 * container1[i + 1][j] + 1.0 * container1[i + 1][j + 1]
 
        sobelx = cv.convertScaleAbs(sobel)  

        # sobel Y
        container2 = np.copy(dst)
        size = container2.shape
        row = size[0]
        col = size[1]
        
        sobel = np.zeros((row, col), dtype = np.int16)  # avoid overflow
        # the given matrix
        for i in range(row):
            for j in range(col):
                    if i == 0 or i == row - 1 or j == 0 or j == col - 1:
                        sobel[i][j] = 0
                    else:
                        # test all int, all float, int with zero, float with zero
                        sobel[i][j] = (-2.0) * container2[i + 1][j] + 0.0 * container2[i][j] + 2.0 * container2[i - 1][j] + (-1.0) * container2[i + 1][j - 1] + 0.0 * container2[i][j - 1] + 1.0 * container2[i - 1][j - 1] + (-1.0) * container2[i + 1][j + 1] + 0.0 * container2[i][j + 1] + 1.0 * container2[i - 1][j + 1]

        sobely = cv.convertScaleAbs(sobel)        

        # magnitude
        container = np.copy(dst)
        size = container.shape
        row = size[0]
        col = size[1]

        m = np.zeros((row, col), dtype = np.uint16)

        for i in range(row):
            for j in range(col):
                m[i][j] = min(255, np.sqrt(sobelx[i][j] ** 2 + sobely[i][j] ** 2))

        m = cv.convertScaleAbs(m)
        cv.imshow("magnitude", m)


    # 4
    def resized(self):
        img = cv.imread(self.filename)  
        i = cv.resize(img, (215, 215))

        M1 = np.float32([[1, 0, 0], [0, 1, 0]])
        i1 = cv.warpAffine(i, M1, (430, 430))

        cv.imshow("Resized", i1)
        cv.imwrite("Resized.png", i1)


    def translate(self):
        img = cv.imread(self.filename)
        i = cv.resize(img, (215, 215))

        M1 = np.float32([[1, 0, 0], [0, 1, 0]])
        i1 = cv.warpAffine(i, M1, (430, 430))

        M2 = np.float32([[1, 0, 215], [0, 1, 215]])
        i2 = cv.warpAffine(i, M2, (430, 430))

        i3 = cv.addWeighted(i1, 1, i2, 1, 0)
        cv.imshow("Translated", i3)
        cv.imwrite("Translated.png", i3)


    def rotate_scale(self):
        img = cv.imread(self.filename)
        i = cv.resize(img, (215, 215))

        M1 = np.float32([[1, 0, 0], [0, 1, 0]])
        i1 = cv.warpAffine(i, M1, (430, 430))
        M2 = np.float32([[1, 0, 215], [0, 1, 215]])
        i2 = cv.warpAffine(i, M2, (430, 430))
        i3 = cv.addWeighted(i1, 1, i2, 1, 0)

        M3 = cv.getRotationMatrix2D((215, 215), 45, 0.5)
        i4 = cv.warpAffine(i3, M3, (430, 430))
        cv.imshow("Rotated", i4)
        cv.imwrite("Rotated.png", i4)


    def shear(self):
        img = cv.imread(self.filename)    
        i = cv.resize(img, (215, 215))
        M1 = np.float32([[1, 0, 0], [0, 1, 0]])
        i1 = cv.warpAffine(i, M1, (430, 430))
        M2 = np.float32([[1, 0, 215], [0, 1, 215]])
        i2 = cv.warpAffine(i, M2, (430, 430))
        i3 = cv.addWeighted(i1, 1, i2, 1, 0)
        M3 = cv.getRotationMatrix2D((215, 215), 45, 0.5)
        i4 = cv.warpAffine(i3, M3, (430, 430)) 

        p1 = np.float32([[50, 50], [200, 50], [50, 200]])
        p2 = np.float32([[10, 100], [100, 50], [100, 250]])
        M4 = cv.getAffineTransform(p1, p2)
        i5 = cv.warpAffine(i4, M4, (430, 430))
        cv.imshow("sheared", i5)
        cv.imwrite("sheared.png", i5)

    # press the button function
    def press_it(self, data):
        if (data == 1):
            self.load()

        elif(data == 2):
            self.gaussian()

        elif(data == 3):
            self.sobelx()

        elif(data == 4):
            self.sobely()

        elif(data == 5):
            self.magni()

        elif(data == 6):
            self.resized()

        elif(data == 7):
            self.translate()

        elif(data == 8):
            self.rotate_scale()

        elif(data == 9):
            self.shear()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
