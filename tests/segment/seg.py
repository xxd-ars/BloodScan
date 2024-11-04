from ultralytics import YOLO
import matplotlib.pyplot as plt
import os, io, sys
import cv2
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

datanumber = 2
test_file_path = "data/data_first/{}-B.png".format(datanumber)
image = cv2.imread(test_file_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()

# Load a model
model = YOLO("tests/yolo/weights/yolo11x-seg.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model

# Predict with the model
results = model(source = image, show = True, save = True)
annotated_img = results[0].plot()
plt.imshow(annotated_img)
plt.show()
# print(results[0].save_dir)
# print(len(results))