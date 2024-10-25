'''
vision module test
'''
# from PyQt5 import QtCore, QtGui, QtWidgets
# from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
# from PyQt5.QtWidgets import QScrollArea, QTextEdit, QHeaderView, QAbstractItemView, QTableWidget, QTableWidgetItem
# from PyQt5.QtCore import QTimer, Qt
# from PyQt5.QtGui import QPixmap,QImage

import os, io, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# from datetime import datetime
# import threading
# import time
# import random
# import sqlite3

# import ui.ui_winqt as ui_winqt
# import database.sqlite as sqlite

import src.algorithm.zbar_v as zbar_v
import src.algorithm.K_means_5 as K_means_5
from scipy import signal

def y_edge(img, y_threshold):
    mask = np.zeros(img.shape, dtype=bool)
    h, w = img.shape
    mask[np.abs(img) > y_threshold] = 1
    return mask

def x_edge(img, x_threshold):
    mask = np.zeros(img.shape, dtype=bool)
    h, w = img.shape
    mask[np.abs(img) > x_threshold] = 1
    return mask

def edge_detection(img, y_threshold, x_threshold):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_normalize = cv2.normalize(img_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    d  = np.array([1,0,-1],ndmin=2)
    fx = signal.convolve2d(img_gray_normalize, d, mode='same', boundary='symm')
    fy = signal.convolve2d(img_gray_normalize, d.T, mode='same', boundary='symm')
    plt.imsave('img/fx.png', fx)
    plt.imsave('img/fy.png', fy)
    fy_mask = y_edge(fy, y_threshold)
    fx_mask = x_edge(fx, x_threshold)
    plt.imsave('img/fx_mask.png', fx_mask)
    plt.imsave('img/fy_mask.png', fy_mask)
    return fy_mask, fx_mask

def find_ROI_boundary(fy_mask, fx_mask, x_cutting, y_cutting):
    fy_row = np.array(np.sum(fy_mask, axis=1, keepdims=0))
    # 计算二维数组 fy_mask 中每一行元素的和
    up = np.argmax(fy_row[0:len(fy_row)//2]) + y_cutting
    down = np.argmax(fy_row[len(fy_row)//2:]) + len(fy_row)//2 - y_cutting

    fx_coloum = np.array(np.sum(fx_mask[up:down, :], axis=0))
    left_edge = np.argmax(fx_coloum[0:len(fx_coloum)//2])
    right_edge = np.argmax(fx_coloum[len(fx_coloum)//2:]) + len(fx_coloum)//2
    
    # 在左边界和右边界之间找出 fx_coloum 中最长的连续小于 5 的区间
    start, length = find_max_zero(fx_coloum, left_edge, right_edge)
    left = start + x_cutting
    right = start + length - 1 - x_cutting

    return right, left, up, down

def find_max_zero(fx_coloum, left, right):
    start = 0
    max_start = 0
    max = 0
    l = 0
    for i in range(left,right):
        if fx_coloum[i] < 5:
            if l==0:
                start=i
            l += 1
        else:
            if l>max:
                max = l
                max_start = start
            l = 0
    return max_start, max

# import motor.motor_control as motor_control
# from motor.motor_control import stop_event, batch
# from motor.m_ModBusTcp import photo_event, rotate_event

datanumber = 2

def get_one_frame():
# def get_one_frame(self):
        while True:
            # photo_event.wait()
            # logger.info("Photo event triggered")

            # realtime video detection
            # 从摄像头捕获一帧图像
            # ret, frame = self.capture.read()

            test_file_path = "data/data_first/{}-A.png".format(datanumber)
            image = cv2.imread(test_file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # height, width, channel = image.shape
            # bytes_per_line = 3 * width
            print(np.shape(image))
            
            image = image.astype(np.uint8)
            barcode_data = zbar_v.get_qrcode_result(image)
            print(f"Barcode: {barcode_data}")
            # image.save(image, format='PNG')
            
            ## 翻转机械夹
            test_file_path = "data/data_first/{}-B.png".format(datanumber)
            image = cv2.imread(test_file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.uint8)

            plt.imsave('img/cropped.png', image[500:1300, 800:1200, :])
            img = image[500:1300, 800:1200, :]
            fy_mask, fx_mask = edge_detection(img,25,25)
            right,left,up,down = find_ROI_boundary(fy_mask,fx_mask,10,20)
            plt.imshow(img, cmap="gray", alpha=0.5)
            plt.plot([0,len(img[0,:,0])-1],[up,up],color='r')
            plt.plot([0,len(img[0,:,0])-1],[down,down],color='r')
            plt.plot([left,left],[0,len(img[:,0,0])-1],color='b')
            plt.plot([right,right],[0,len(img[:,0,0])-1],color='b')
            plt.show()



            break
            # # if ret:
            # #     # 将图像转换为 RGB 格式（OpenCV 默认使用 BGR 格式）
            #     # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #     # 将图像转换为 Qt 的图像格式
            #     height, width, channel = image.shape
            #     bytes_per_line = 3 * width
            #     self.frame_height = height
            #     self.frame_width = width
            #     q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            #     # 将图像转换为 QPixmap 并设置给 QLabel
            #     pixmap = QPixmap.fromImage(q_image)
            #     scaled_pixmap = pixmap.scaled(self.photoLB.size(), QtCore.Qt.KeepAspectRatio)
            #     self.photoLB.clear()
            #     self.photoLB.setPixmap(scaled_pixmap)

            #     # KNN预测
            #     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #     pred_class = knnsp.predict_single_image(image)
            #     pred_class = None
            #     self.pred_class = pred_class


            #     #yolov5
            #     start_time = time.time()
            #     filename = f"tmp/{self.detect_number}.png"
            #     cv2.imwrite(filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            #     self.detect_number += 1
            #     # detect.run(weights=r'last.pt', source=r'tmp')
            #     print(f"time:{time.time()-start_time}")
                
            #     folder_path = 'tmp'
            #     for filename in os.listdir(folder_path):
            #         file_path = os.path.join(folder_path, filename)
            #         try:
            #             if os.path.isfile(file_path) or os.path.islink(file_path):
            #                 os.unlink(file_path)
            #         except Exception as e:
            #             print(f"Failed to delete {file_path}. Reason: {e}")

                
            #     # 条形码检测
            #     barcode_data, binary_value, decode_time,number = zbar_v.get_qrcode_result(frame_rgb)
            #     print(f"Barcode: {barcode_data}")
            #     print(f"time:{time.time()-start_time}")

            #     capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            #     self.capture_time = capture_time
            #     image_bytes = io.BytesIO()
            #     image.save(image_bytes, format='PNG')
            #     image_data = image_bytes.getvalue()
            #     self.image_data = image_data

            #     self.detectionLB.setText(f"capture_time:\n {capture_time}\nPredicted class:\n {pred_class}\n")
            #     print("set text")

            # else:
            #     self.detectionLB.setText("cant photo")

            # self.sample_lineEdit.clear()
            # self.heigt_lineEdit.clear()

            # ### offline database detection
            # if self.kk == 1:
            #     self.id_lineEdit.setText(f"{self.detect_number}")
            #     test_list = [2,3,5,6,7,30,32,35,37,38,10,11,12,14,16,18,19,24,27,28,240,241,242,243,244,245,246,247,248,249]
            #     number_k = random.choice(test_list)
            #     # number_k = self.pick_numbers(self.test_list)
            #     self.datanumber = number_k
            #     test_file_path = "./data_first/{}-A.png".format(self.datanumber)
            #     image = cv2.imread(test_file_path)
            #     pixmap = QPixmap(test_file_path)

            #     if not pixmap.isNull():
            #         self.photoLB.setPixmap(pixmap.scaled(self.photoLB.size(), Qt.KeepAspectRatio))
            #     self.barcode_data= zbar_v.get_qrcode_result(image,binary_step = 8)
            #     self.bar_lineEdit.setText(f"{self.barcode_data}")

            #     if self.datanumber <= 200:
            #         self.tube_lineEdit.setText(f"无盖试管")
            #         self.tube_type = '无盖试管'
            #     else:
            #         self.tube_lineEdit.setText(f"紫色EDTA抗凝管")
            #         self.tube_type = '紫色EDTA抗凝管'


            #     self.kk = 2
            # else:
            #     test_file_path = "./data_first/{}-B.png".format(self.datanumber)
            #     pixmap = QPixmap(test_file_path)
            #     # if not pixmap.isNull():
            #     #     self.photoLB.setPixmap(pixmap.scaled(self.photoLB.size(), Qt.KeepAspectRatio))
            #     # else:
            #     #     self.detectionLB.setText("Image not found")
            #     self.photoLB.setPixmap(pixmap.scaled(self.photoLB.size(), Qt.KeepAspectRatio))
            #     draw,segmented_image,img_draw,self.blood_layer_info,self.blood_height = K_means_5.test(number_k)

            #     test_file_path = "./dataset2/{}-seg.png".format(self.datanumber)
            #     pixmap = QPixmap(test_file_path)
            #     if not pixmap.isNull():
            #         self.photoLB.setPixmap(pixmap.scaled(self.photoLB.size(), Qt.KeepAspectRatio))

            #     time.sleep(1)
                
            #     if draw:
            #         test_file_path = "./dataset2/{}-draw.png".format(self.datanumber)
            #         pixmap = QPixmap(test_file_path)
            #         if not pixmap.isNull():
            #             self.photoLB.setPixmap(pixmap.scaled(self.photoLB.size(), Qt.KeepAspectRatio))
            #             self.heigt_lineEdit.setText(f"{self.blood_height}")
            #     else:
            #         self.heigt_lineEdit.setText(f"无分层")
            #         self.blood_height = 0
            #     self.sample_lineEdit.setText(self.blood_layer_info)


            #     self.data_event.set()
            #     self.kk = 1
            #     self.detect_number += 1

            # # 模拟分析时间 并观察是否真的停下来拍照完成检测后再翻转
            # time.sleep(5)

            # photo_event.clear()
            # rotate_event.set()


if __name__ == "__main__":
    get_one_frame()