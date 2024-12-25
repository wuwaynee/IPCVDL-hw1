from fileinput import filename
from this import d
from tkinter import NONE
from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np
import cv2 as cv

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(850, 492)
        self.load_image_1 = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(1))
        self.load_image_1.setGeometry(QtCore.QRect(30, 110, 201, 41))
        self.load_image_1.setObjectName("load_image_1")
        self.load_image_2 = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(2))
        self.load_image_2.setGeometry(QtCore.QRect(30, 260, 201, 41))
        self.load_image_2.setObjectName("load_image_2")
        self.color_separation = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(3))
        self.color_separation.setGeometry(QtCore.QRect(320, 110, 201, 41))
        self.color_separation.setObjectName("color_separation")
        self.color_transformation = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(4))
        self.color_transformation.setGeometry(QtCore.QRect(320, 200, 201, 41))
        self.color_transformation.setObjectName("color_transformation")
        self.color_detection = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(5))
        self.color_detection.setGeometry(QtCore.QRect(320, 300, 201, 41))
        self.color_detection.setObjectName("color_detection")
        self.blending = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(6))
        self.blending.setGeometry(QtCore.QRect(320, 390, 201, 41))
        self.blending.setObjectName("blending")
        self.gaussian_blur = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(7))
        self.gaussian_blur.setGeometry(QtCore.QRect(600, 110, 201, 41))
        self.gaussian_blur.setObjectName("gaussian_blur")
        self.bilateral_filter = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(8))
        self.bilateral_filter.setGeometry(QtCore.QRect(600, 250, 201, 41))
        self.bilateral_filter.setObjectName("bilateral_filter")
        self.median_filter = QtWidgets.QPushButton(Form, clicked = lambda : self.press_it(9))
        self.median_filter.setGeometry(QtCore.QRect(600, 390, 201, 41))
        self.median_filter.setObjectName("median_filter")
        self.label_1 = QtWidgets.QLabel(Form)
        self.label_1.setGeometry(QtCore.QRect(320, 60, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(16)
        self.label_1.setFont(font)
        self.label_1.setObjectName("label_1")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(600, 60, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.load_image_1.setText(_translate("Form", "Load Image 1"))
        self.load_image_2.setText(_translate("Form", "Load Image 2"))
        self.color_separation.setText(_translate("Form", "1.1 color separation"))
        self.color_transformation.setText(_translate("Form", "1.2 color transformation"))
        self.color_detection.setText(_translate("Form", "1.3 color detection"))
        self.blending.setText(_translate("Form", "1.4 blending"))
        self.gaussian_blur.setText(_translate("Form", "2.1 Gaussian blur"))
        self.bilateral_filter.setText(_translate("Form", "2.2 Bilateral filter"))
        self.median_filter.setText(_translate("Form", "2.3 Median filter"))
        self.label_1.setText(_translate("Form", "1. Image Processing"))
        self.label_2.setText(_translate("Form", "2. Image Smoothing"))


    def load1(self):
        self.filename, self.filetype = QtWidgets.QFileDialog.getOpenFileName(None, "Open file",".\\") 

    def load2(self):
        self.filename1, self.filetype1 = QtWidgets.QFileDialog.getOpenFileName(None, "Open file",".\\") 

    def separate(self):
        img = cv.imread(self.filename)   

        blank = np.zeros(img.shape[:2], dtype='uint8')

        blue, green, red = cv.split(img)
        blueimg = cv.merge([blue, blank, blank])
        greenimg = cv.merge([blank, green, blank])
        redimg = cv.merge([blank, blank, red])

        cv.imshow('Bchannel', blueimg)
        cv.imshow('Gchannel', greenimg)
        cv.imshow('Rchannel', redimg)

        cv.imshow('img', img)
        cv.waitKey(0)

    def transform(self):
        img = cv.imread(self.filename)

        i1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
        
        blue, green, red = cv.split(img)
        # average weighted
        i2 = blue/3+green/3+red/3
        i2 = i2.astype('uint8')
        
        cv.imshow('OpenCV function', i1)
        cv.imshow('Average weighted', i2)

        cv.imshow('img', img)
        cv.waitKey(0)
    
    def detect(self):
        img = cv.imread(self.filename)

        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        g_lower = np.array([40, 50, 20])
        g_upper = np.array([80, 255, 255])
        w_lower = np.array([0, 0, 200])
        w_upper = np.array([180, 20, 255])

        g = cv.inRange(hsv, g_lower, g_upper)
        w = cv.inRange(hsv, w_lower, w_upper)

        result1 = cv.bitwise_and(img, img, mask = g)
        result2 = cv.bitwise_and(img, img, mask = w)

        cv.imshow('Green', result1)
        cv.imshow('White', result2)
        cv.waitKey(0)

    def blend(self,x):
        img1 = cv.imread(self.filename)
        img2 = cv.imread(self.filename1)

        num = cv.getTrackbarPos('Blend ', 'Blend')
        img = cv.addWeighted(img1, (255.0-num)/255.0, img2, num / 255.0, 0)
        cv.imshow('Blend', img)

    def gaussian(self,x):
        img = cv.imread(self.filename)

        m = cv.getTrackbarPos('magnitude', 'gaussian_blur')
        output_gau = cv.GaussianBlur(img, (2 * m + 1, 2 * m + 1), 0)
        cv.imshow('gaussian_blur', output_gau)

    def bilateral(self,x):
        img = cv.imread(self.filename)

        m = cv.getTrackbarPos('magnitude', 'bilateral_filter')
        output_bil = cv.bilateralFilter(img, 2 * m + 1, 90, 90)
        cv.imshow('bilateral_filter', output_bil)

    def median(self,x):
        img = cv.imread(self.filename)
        
        m = cv.getTrackbarPos('magnitude', 'median_filter')
        output_med = cv.medianBlur(img, 2 * m + 1)
        cv.imshow('median_filter', output_med)

    
    # press the button function
    def press_it(self, data):
        if (data == 1):
            self.load1()

        elif(data == 2):
            self.load2()

        elif(data == 3):
            self.separate()

        elif(data == 4):
            self.transform()

        elif(data == 5):
            self.detect()

        elif(data == 6):
            img1 = cv.imread(self.filename)

            cv.imshow('Blend',img1)
            cv.createTrackbar('Blend ', 'Blend', 0, 255, self.blend)
            
            # while True:
                
            #     num = cv.getTrackbarPos('Blend ', 'Blend')
            #     img = cv.addWeighted(img1, (255.0-num)/255.0, img2, num / 255.0, 0)
            #     cv.imshow('Blend', img)
            #     cv.waitKey(1)

        elif(data == 7):
            img = cv.imread(self.filename)

            cv.imshow('gaussian_blur', img)
            cv.createTrackbar('magnitude', 'gaussian_blur', 0, 10, self.gaussian)

            # while True:
            #     m = cv.getTrackbarPos('magnitude', 'gaussian_blur')
            #     output_gau = cv.GaussianBlur(img, (2 * m + 1, 2 * m + 1), 0)
                
            #     cv.imshow('gaussian_blur', output_gau)
            #     cv.waitKey(1)

        elif(data == 8):
            img = cv.imread(self.filename)

            cv.imshow('bilateral_filter', img)
            cv.createTrackbar('magnitude', 'bilateral_filter', 0, 10, self.bilateral)

            # while True:
            #     m = cv.getTrackbarPos('magnitude', 'bilateral_filter')
            #     output_bil = cv.bilateralFilter(img, 2 * m + 1, 90, 90)
                
            #     cv.imshow('bilateral_filter', output_bil)
            #     cv.waitKey(1)

        elif(data == 9):
            img = cv.imread(self.filename)

            cv.imshow('median_filter', img)
            cv.createTrackbar('magnitude', 'median_filter', 0, 10, self.median)

            # while True:
            #     m = cv.getTrackbarPos('magnitude', 'median_filter')
            #     output_med = cv.medianBlur(img, 2 * m + 1)
                
            #     cv.imshow('median_filter', output_med)
            #     cv.waitKey(1)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
