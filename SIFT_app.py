#!/usr/bin/env python

from PyQt5 import QtCore, QtGui, QtWidgets
from python_qt_binding import loadUi

import cv2
import sys
import numpy as np

class My_App(QtWidgets.QMainWindow):

    def __init__(self):
        super(My_App, self).__init__()
        loadUi("./SIFT_app.ui", self)

        self._cam_id = 0
        self._cam_fps = 10
        self._is_cam_enabled = False
        self._is_template_loaded = False

        self.browse_button.clicked.connect(self.SLOT_browse_button)
        self.toggle_cam_button.clicked.connect(self.SLOT_toggle_camera)

        self._camera_device = cv2.VideoCapture(self._cam_id)
        self._camera_device.set(3, 320)
        self._camera_device.set(4, 240)

        # Timer used to trigger the camera
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.SLOT_query_camera)
        self._timer.setInterval(1000 / self._cam_fps)

    def SLOT_browse_button(self):
        dlg = QtWidgets.QFileDialog()
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)

        if dlg.exec_():
            self.template_path = dlg.selectedFiles()[0]

        pixmap = QtGui.QPixmap(self.template_path)
        self.template_label.setPixmap(pixmap)

        print("Loaded template image file: " + self.template_path)

    # Source: stackoverflow.com/questions/34232632/
    def convert_cv_to_pixmap(self, cv_img):
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        height, width, channel = cv_img.shape
        bytesPerLine = channel * width
        q_img = QtGui.QImage(cv_img.data, width, height, 
                    bytesPerLine, QtGui.QImage.Format_RGB888)
        return QtGui.QPixmap.fromImage(q_img)

    def SLOT_query_camera(self):
        ret, frame = self._camera_device.read()
        #TODO run SIFT on the captured frame
        
        # The code is based on the following tutorial: 
        # https://pysource.com/2018/06/05/object-tracking-using-homography-opencv-3-4-with-python-3-tutorial-34/
        
        image = cv2.imread(self.template_path, cv2.IMREAD_GRAYSCALE)
        # Getting image features
        sift = cv2.xfeatures2d.SIFT_create()
        kp_image, desc_image = sift.detectAndCompute(image, None)
        image = cv2.drawKeypoints(image, kp_image, image)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        kp_frame_gray, desc_frame_gray = sift.detectAndCompute(frame_gray, None)

        # Matching features
        index_params = dict(algorithm=0, trees=5)
        search_params = dict()
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc_image, desc_frame_gray, k=2)
        
        # Finding good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good_matches.append(m)

        image_match = cv2.drawMatches(image, kp_image, frame_gray, kp_frame_gray, good_matches, frame_gray)

        # Homography
        if len(good_matches) > 12: 
            image_points = np.float32([kp_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            live_points = np.float32([kp_frame_gray[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            matrix, mask = cv2.findHomography(image_points, live_points, cv2.RANSAC, 5.0)

            # Perspective transform

            height, width, c = image.shape
            points = np.float32([[0, 0], [0, height], [width, height], [width, 0]]).reshape(-1, 1 ,2)
            dst = cv2.perspectiveTransform(points, matrix)

            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            homography_pixmap = self.convert_cv_to_pixmap(homography)
            self.live_image_label.setPixmap(homography_pixmap)


        else: 
            matched_pixmap = self.convert_cv_to_pixmap(image_match)    
            image_pixmap = self.convert_cv_to_pixmap(image)
            live_pixmap = self.convert_cv_to_pixmap(frame)
            self.live_image_label.setPixmap(live_pixmap)
            #self.live_image_label.setPixmap(image_pixmap)

    def SLOT_toggle_camera(self):
        if self._is_cam_enabled:
            self._timer.stop()
            self._is_cam_enabled = False
            self.toggle_cam_button.setText("&Enable camera")
        else:
            self._timer.start()
            self._is_cam_enabled = True
            self.toggle_cam_button.setText("&Disable camera")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myApp = My_App()
    myApp.show()
    sys.exit(app.exec_())
