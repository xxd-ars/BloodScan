'''该脚本使用 PyQt5 和 OpenCV 实现一个简单的摄像头测试应用。它创建一个窗口，实时显示从摄像头捕获的视频流。此脚本适用于使用 GStreamer 管道捕获视频的情况。

功能
初始化摄像头

使用 GStreamer 管道从指定设备捕获视频流。
显示视频流

在 PyQt5 界面中显示视频流。
实时更新帧，以确保视频流的流畅性。
关闭事件

在关闭窗口时释放摄像头资源。

注意事项
设备路径:

默认设备路径为 /dev/video0, 可以根据需要更改设备路径。
分辨率和帧率:

默认分辨率为 640x480, 帧率为 30fps。可以根据需要调整 GStreamer 管道中的参数。
关闭事件:

在关闭窗口时，确保正确释放摄像头资源，以避免资源占用问题。'''
import sys
import cv2
import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets

class VideoWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.capture = cv2.VideoCapture('v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, framerate=30/1 ! videoconvert ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER)
        
        if not self.capture.isOpened():
            raise Exception("Could not open video device")

        self.videoLB = QtWidgets.QLabel(self)
        self.videoLB.resize(640, 480)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # 每30毫秒更新一次，即大约每秒33帧

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            scaled_pixmap = QtGui.QPixmap.fromImage(qImg).scaled(self.videoLB.size(), QtCore.Qt.KeepAspectRatio)
            self.videoLB.setPixmap(scaled_pixmap)
        else:
            print("Frame capture failed")

    def closeEvent(self, event):
        self.capture.release()
        event.accept()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    video_widget = VideoWidget()
    video_widget.show()
    sys.exit(app.exec_())
