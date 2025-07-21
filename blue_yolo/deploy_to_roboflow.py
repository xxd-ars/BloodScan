import roboflow

rf = roboflow.Roboflow(api_key='0Tt9VAmQ4YXp42tYyNLA')
project = rf.workspace().project("blood-scan-lyzhe")

#can specify weights_filename, default is "weights/best.pt"
version = project.version(6)
version.deploy("yolov11-seg", "tests/yolo_seg/weights", "best_augmented5.pt")