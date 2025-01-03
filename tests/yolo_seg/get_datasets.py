import os
from roboflow import Roboflow

dataset_version = 6
location = r"./tests/yolo_seg/datasets/Blood-Scan-" + str(dataset_version)

rf = Roboflow(api_key="0Tt9VAmQ4YXp42tYyNLA")
project = rf.workspace("bloodscan").project("blood-scan-lyzhe")
version = project.version(dataset_version)
dataset = version.download(model_format = "yolov11",
                           location = location)