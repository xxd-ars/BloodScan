import cv2
import requests
from PIL import Image
import paramiko
import os, time
import matplotlib.pyplot as plt
import numpy as np
import random

from upload import uploadImage, uploadImage2Flask

def get_one_frame(test_mode = True):
    start_time = time.time()
    # while True:
    i = 0
    while i == 0:
        # photo_event.wait()
        # logger.info("Photo event triggered")
        if not test_mode:
            # cap = cv2.VideoCapture('v4l2src device=/dev/video0 ! image/jpeg, width=2048, height=1536, framerate=30/1 ! jpegdec ! videoconvert ! video/x-raw, format=BGR ! appsink', 
            #                     cv2.CAP_GSTREAMER)
            
            image, frame = cap.read()
            if image:
                timestamp = time.strftime("%Y%m%d%H%M%S")
                filename = "test_tube" + timestamp + ".jpg"
                plt.imsave("data/data_collection/" + filename, image)
                # cap.release()
            else:
                print("Failed to capture video frame")
                # logger.warning("Failed to capture video frame")
        else:
            test_list = [25, 79, 80, 95, 96, 180, 181]
            datanumber = random.choice(test_list)
            ipath = "tests/datasets/data_first_valid/{}-B.png".format(datanumber)
            image = Image.open(ipath)
            timestamp = time.strftime("%Y%m%d%H%M%S")
            filename = "test_tube" + timestamp + ".jpg"
            # cv2.imwrite(filename, np.array(image.convert("RGB")))
            image = np.array(image.convert("RGB"))
            # plt.imshow(image)
            # plt.show()
            plt.imsave("data/data_collection/" + filename, image)
        i += 1

        # filename = 'bus.jpg'
        # res = uploadImage(filename)
        res = uploadImage2Flask(filename)
        plt.imshow(Image.open(res))
        plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Full processing took {elapsed_time:.2f} seconds.")
    
if __name__ == "__main__":
    get_one_frame()

# 优化方法 2：将模型部署为服务，可以有效减少模型加载时间，并支持高效的推理流程。以下是详细步骤：