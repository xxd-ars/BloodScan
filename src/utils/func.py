'''
    完整的后端程序可以与 PyQt5 生成的界面文件（Ui_winqt）一起使用，以实现复杂的图像处理和数据管理应用。

    
日志配置: 配置了日志记录器以记录调试信息和错误。记录在app.log中，线程变化的过程，用于调试。

线程: 使用线程处理电机初始化、视频显示、图像捕获和数据添加，以确保界面不会因为这些操作而卡顿。


视频捕获: 使用 OpenCV 从摄像头捕获视频帧，并在 QLabel 上显示。
函数：video_show

定时器: 使用 QTimer 定时更新一些信息，如帧数和批次等。
函数：timer_start&update_info

数据库操作: 使用 SQLite 存储和读取采样数据。
函数：view_datasql&load_table_data

条形码检测和 K-means 分层检测: 使用 zbar 和 K-means 进行条形码识别和血液分层检测，并将结果显示在界面上。
函数：get_one_frame
get_one_frame里面的注释部分是真实环境下用于拍照测试的，未注释的是调用本地照片用于展示识别结果的

用户界面交互: 界面元素的更新和事件处理，如开始检测、停止检测、继续检测等。
函数：detect_start&detect_stop&detect_continue&detect_end

    

'''
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QTextEdit,QHeaderView, QAbstractItemView, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap,QImage
import Ui_winqt #   Qt界面
import cv2
from PIL import Image
import sqlite3 #数据库
from datetime import datetime
import io
import motor_control
from motor_control import stop_event,batch
import threading
from m_ModBusTcp import photo_event,rotate_event
import time
# import detect
import os
import zbar_v
import K_means_5
import random
import sqlite
import logging
import logging.config

# 配置日志
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

logger = logging.getLogger(__name__)

class MyMainWindow(QtWidgets.QMainWindow, Ui_winqt.Ui_MainWindow):
    def __init__(self):
        super(MyMainWindow, self).__init__()
        self.setupUi(self)
        logger.info("UI setup complete")

        self.capture = cv2.VideoCapture('v4l2src device=/dev/video0 ! image/jpeg, width=2048, height=1536, framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER)
        if not self.capture.isOpened():
            logger.error("Could not open video device")
            raise Exception("Could not open video device")
        logger.info("Video device opened successfully")

        self.datanumber = 1
        self.image_data = 0
        self.pred_class = 0
        self.capture_time = 0
        self.frame_height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_count = 0
        self.photoLB.setAlignment(Qt.AlignCenter)  # Center align the QLabel
        self.timer_start()

        self.detect_number = 1
        self.start_time = time.time()

        self.motor_init_thread = threading.Thread(target=self.motor_init_wrapper)
        self.motor_init_thread.daemon = True
        self.motor_init_thread.start()
       

        self.motor_control_thread = threading.Thread(target=self.detect_all_wrapper)
        self.motor_control_thread.daemon = True
        stop_event.set()

        self.photo_thread = threading.Thread(target=self.get_one_frame)
        self.photo_thread.daemon = True
        self.photo_thread.start()
        self.kk = 1

        self.data_thread = threading.Thread(target=self.add_data)
        self.data_thread.daemon = True
        self.data_event = threading.Event()
        self.data_thread.start()

        self.barcode_data = None
        self.blood_layer_info = None
        self.blood_height = None
        self.collection_batch = batch
        self.tube_type = None

        self.test_list = [2, 3, 5, 6, 7, 30, 32, 35, 37, 38, 10, 11, 12, 14, 16, 18, 19, 24, 27, 28, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249]
        logger.info("Initialization complete")

    def motor_init_wrapper(self):
        logger.info("Motor initialization started")
        try:
            motor_control.motor_init()
            logger.info("Motor initialization completed successfully")
        except Exception as e:
            logger.error(f"Motor initialization failed: {e}")

    def detect_all_wrapper(self):
        logger.info("Detection started")
        try:
            motor_control.detect_all()
            logger.info("Detection completed successfully")
        except Exception as e:
            logger.error(f"Detection failed: {e}")
        

    def video_show(self):
        while True:
            ret, frame = self.capture.read()
            self.frame_count += 1
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                # qImg = QPixmap.fromImage(QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888))
                qImg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                # scaled_pixmap = qImg.scaled(self.videoLB.size(), QtCore.Qt.KeepAspectRatio)
                scaled_pixmap = QtGui.QPixmap.fromImage(qImg).scaled(self.videoLB.size(), QtCore.Qt.KeepAspectRatio)
                self.videoLB.setPixmap(scaled_pixmap)
            else:
                logger.warning("Failed to capture video frame")
            time.sleep(0.1)

    def update_info(self):
        # 更新文本框中的信息
        global batch
        elapsed_time = time.time() - self.start_time
        formatted_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        self.collection_batch = self.detect_number//10
        self.frame_lineEdit.setText(f"{self.frame_count}")
        self.batch_lineEdit.setText(f"{self.collection_batch}")
        self.time_lineEdit.setText(f"{formatted_time}")
        self.frame_count = 0

    def timer_start(self):
    # # 启动视频捕捉
        
        # 启动显示帧数和像素的定时器
        self.info_timer = QTimer(self)
        self.info_timer.timeout.connect(self.update_info)
        self.info_timer.start(1000)  # 每秒更新一次信息
        logger.info("Info timer started")


         # 将视频显示函数放入单独的线程中执行
        self.video_thread = threading.Thread(target=self.video_show)
        self.video_thread.daemon = True  
        self.video_thread.start()
        logger.info("Video thread started")

    def add_data(self):
        while True:
            self.data_event.wait()
            try:
                sqlite.insert_sample_data(
                    barcode=self.barcode_data,
                    tube_type=self.tube_type,
                    blood_layer_info=self.blood_layer_info,
                    blood_height=self.blood_height,
                    collection_batch=self.collection_batch,
                    collection_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                )
                logger.info("Data added to the database successfully")
            except Exception as e:
                logger.error(f"Failed to add data to the database: {e}")
            self.data_event.clear()

    def get_one_frame(self):
        while True:
            photo_event.wait()
            logger.info("Photo event triggered")
            # print("photo")
            # # 从摄像头捕获一帧图像
            # ret, frame = self.capture.read()
            # if ret:
            #     # 将图像转换为 RGB 格式（OpenCV 默认使用 BGR 格式）
            #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #     # 将图像转换为 Qt 的图像格式
            #     height, width, channel = frame_rgb.shape
            #     bytes_per_line = 3 * width
            #     self.frame_height = height
            #     self.frame_width = width
            #     q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
            #     # 将图像转换为 QPixmap 并设置给 QLabel
            #     pixmap = QPixmap.fromImage(q_image)
            #     scaled_pixmap = pixmap.scaled(self.photoLB.size(), QtCore.Qt.KeepAspectRatio)
            #     self.photoLB.clear()
            #     self.photoLB.setPixmap(scaled_pixmap)

                #KNN预测
                # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                # pred_class = knnsp.predict_single_image(image)
                # pred_class = None
                # self.pred_class = pred_class


                # #yolov5
                # start_time = time.time()
                # filename = f"tmp/{self.detect_number}.png"
                # cv2.imwrite(filename, cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                # self.detect_number += 1
                # # detect.run(weights=r'last.pt', source=r'tmp')
                # print(f"time:{time.time()-start_time}")
                
                # folder_path = 'tmp'
                # for filename in os.listdir(folder_path):
                #     file_path = os.path.join(folder_path, filename)
                #     try:
                #         if os.path.isfile(file_path) or os.path.islink(file_path):
                #             os.unlink(file_path)
                #     except Exception as e:
                #         print(f"Failed to delete {file_path}. Reason: {e}")

                
                #条形码检测
                # barcode_data, binary_value, decode_time,number = zbar_v.get_qrcode_result(frame_rgb)
                # print(f"Barcode: {barcode_data}")
                # print(f"time:{time.time()-start_time}")

                # capture_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # self.capture_time = capture_time
                # image_bytes = io.BytesIO()
                # image.save(image_bytes, format='PNG')
                # image_data = image_bytes.getvalue()
                # self.image_data = image_data

                # self.detectionLB.setText(f"capture_time:\n {capture_time}\nPredicted class:\n {pred_class}\n")
                # print("set text")

            # else:
            #     self.detectionLB.setText("cant photo")

            self.sample_lineEdit.clear()
            self.heigt_lineEdit.clear()

            
            if self.kk ==1:
                self.id_lineEdit.setText(f"{self.detect_number}")
                test_list = [2,3,5,6,7,30,32,35,37,38,10,11,12,14,16,18,19,24,27,28,240,241,242,243,244,245,246,247,248,249]
                number_k = random.choice(test_list)
                # number_k = self.pick_numbers(self.test_list)
                self.datanumber = number_k
                test_file_path = "./data_first/{}-A.png".format(self.datanumber)
                image = cv2.imread(test_file_path)
                pixmap = QPixmap(test_file_path)
                if not pixmap.isNull():
                    self.photoLB.setPixmap(pixmap.scaled(self.photoLB.size(), Qt.KeepAspectRatio))
                self.barcode_data= zbar_v.get_qrcode_result(image,binary_step = 8)
                self.bar_lineEdit.setText(f"{self.barcode_data}")
                if self.datanumber <= 200:
                    self.tube_lineEdit.setText(f"无盖试管")
                    self.tube_type = '无盖试管'
                else:
                    self.tube_lineEdit.setText(f"紫色EDTA抗凝管")
                    self.tube_type = '紫色EDTA抗凝管'


                self.kk = 2
            else:
                test_file_path = "./data_first/{}-B.png".format(self.datanumber)
                pixmap = QPixmap(test_file_path)
                # if not pixmap.isNull():
                #     self.photoLB.setPixmap(pixmap.scaled(self.photoLB.size(), Qt.KeepAspectRatio))
                # else:
                #     self.detectionLB.setText("Image not found")
                self.photoLB.setPixmap(pixmap.scaled(self.photoLB.size(), Qt.KeepAspectRatio))
                draw,segmented_image,img_draw,self.blood_layer_info,self.blood_height = K_means_5.test(number_k)

                test_file_path = "./dataset2/{}-seg.png".format(self.datanumber)
                pixmap = QPixmap(test_file_path)
                if not pixmap.isNull():
                    self.photoLB.setPixmap(pixmap.scaled(self.photoLB.size(), Qt.KeepAspectRatio))

                time.sleep(1)
                
                if draw:
                    test_file_path = "./dataset2/{}-draw.png".format(self.datanumber)
                    pixmap = QPixmap(test_file_path)
                    if not pixmap.isNull():
                        self.photoLB.setPixmap(pixmap.scaled(self.photoLB.size(), Qt.KeepAspectRatio))
                        self.heigt_lineEdit.setText(f"{self.blood_height}")
                else:
                    self.heigt_lineEdit.setText(f"无分层")
                    self.blood_height = 0
                self.sample_lineEdit.setText(self.blood_layer_info)


                self.data_event.set()
                self.kk = 1
                self.detect_number += 1

            photo_event.clear()
            rotate_event.set()


    def load_table_data(self):
        conn = sqlite3.connect('sample_data.db')  # 连接到SQLite数据库
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM SampleData")  # 查询SampleData表中的所有数据
        rows = cursor.fetchall()
        col_names = [description[0] for description in cursor.description]

        self.tableWidget.setColumnCount(len(col_names))
        self.tableWidget.setRowCount(len(rows))
        self.tableWidget.setHorizontalHeaderLabels(col_names)

        for row_index, row_data in enumerate(rows):
            for col_index, col_data in enumerate(row_data):
                self.tableWidget.setItem(row_index, col_index, QTableWidgetItem(str(col_data)))

        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)  # 使表格只读

        cursor.close()
        conn.close()
        logger.info("Table data loaded successfully")

    def view_datasql(self):
        self.datanumber = 1
        self.load_table_data()
        logger.info("Viewing data from SQL")

    def detect_start(self):
        self.motor_control_thread.start()
        # logger.info("Detection started")

    def detect_stop(self):
        stop_event.clear()
        logger.info("Detection stopped")

    def detect_continue(self):
        stop_event.set()
        logger.info("Detection continued")

    def detect_end(self):
        self.capture.release()
        self.video_thread.end()
        motor_control.motor_close()
        self.close()
        logger.info("Detection ended and application closed")