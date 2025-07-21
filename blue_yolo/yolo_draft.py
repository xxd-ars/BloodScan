from ultralytics import YOLO
import matplotlib.pyplot as plt
import os, io, sys
from PIL import Image
import numpy as np
import cv2
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# for datanumber in range(2, 6):
for datanumber in range(2, 3):
    ipath = "tests/datasets/data_first/{}-B.png".format(datanumber)
    image = Image.open(ipath) # 1800*1200 'tests/segment/datasets/truck.jpg'
    # image = np.array(image.convert("RGB"))
    # plt.imshow(image)
    # plt.show()

# Load a model
model = YOLO("tests/yolo_seg/weights/yolo11x-seg.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model(source = image, show = False, save = True)
annotated_img = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
plt.imshow(annotated_img)
plt.show()

# yolo segment predict model=yolo11n-seg.pt source='https://media.roboflow.com/notebooks/examples/dog.jpeg' save=true
