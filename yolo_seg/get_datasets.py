from roboflow import Roboflow

dataset_version = 1
location = r"./tests/yolo_seg/datasets/Blue-Rawdata-1504-500" + "-" + str(dataset_version)
# rf = Roboflow(api_key="0Tt9VAmQ4YXp42tYyNLA")
# project = rf.workspace("bloodscan").project("blood-scan-lyzhe")
# rf = Roboflow(api_key="C99pXEebpnE3dSAP3tuQ")
# project = rf.workspace("bloodscancropped").project("blood-scan-cropped")

rf = Roboflow(api_key="Q1tiegdnCfdGzkY3zduG")
project = rf.workspace("bloodscancropped-atnul").project("coannotation-bloodscan")

version = project.version(dataset_version)
dataset = version.download(model_format = "yolov11",
                           location = location)