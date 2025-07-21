from ultralytics import YOLO
import matplotlib.pyplot as plt
import os, io, sys
from PIL import Image
import numpy as np
import cv2
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # for datanumber in range(2, 6):
# for datanumber in range(2, 3):
#     ipath = "tests/datasets/data_first/{}-B.png".format(datanumber)
#     image = Image.open(ipath) # 1800*1200 'tests/segment/datasets/truck.jpg'
#     # image = np.array(image.convert("RGB"))
#     # plt.imshow(image)
#     # plt.show()

# Load a model
# model = YOLO("tests/yolo_seg/weights/best_augmented5.pt")
model = YOLO("tests/yolo_seg/runs/segment/train3_640/weights/best.pt")

# Predict with the model
ipath = "./tests/yolo_seg/datasets/Blood-Scan-Cropped-1/test/images/2022-03-28_103204_1_T5_2348_bmp_jpg.rf.e015458a82d1291596e001037efab85e.jpg"
image = Image.open(ipath)
results = model(source = image, # imgsz = [768, 1024], 
                device="cuda:0",
                visualize = False, show = False, 
                save = True, save_txt = True, save_conf = True, save_crop = True,
                show_labels = True, show_conf = True, show_boxes = True, line_width = 2)

results[0].plot(labels = False, boxes = False, masks = True, probs = True,
                show = True, save = False, filename = None, color_mode = 'class')
# print(results[0].masks)
# 'class', 'instance'

# annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
# plt.imshow(annotated_img)
# plt.show()