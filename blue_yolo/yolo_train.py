from ultralytics import YOLO
import os

if __name__ == '__main__':
    dataset_version = 1
    project_directory = "/home/xin99/BloodScan/tests/yolo_seg"
    # os.chdir(target_directory)

    model = YOLO(project_directory + "/weights/yolo11x-seg.pt")
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")

    # dataset = project_directory + "/datasets/Blood-Scan-Cropped-" + str(dataset_version) + '/data.yaml'
    dataset = project_directory + "/datasets/Blue-Rawdata-1504-500-" + str(dataset_version) + '/data.yaml'
    
    results = model.train(data = dataset, 
                          # device="cuda", 
                          device="0,1,2,3",
                          batch = 8, 
                          epochs = 200, 
                          imgsz = 1504, 
                          plots = True)