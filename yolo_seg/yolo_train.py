from ultralytics import YOLO
import os

if __name__ == '__main__':
    dataset_version = 6
    project_directory = "/Users/ASUS/Documents/SJTU M2/Graduation Project/BloodScan/tests/yolo_seg"
    # os.chdir(target_directory)

    model = YOLO(project_directory + "/weights/yolo11x-seg.pt")
    # model = YOLO("yolo11n.yaml").load("yolo11n.pt")

    # dataset = project_directory + "/datasets/Blood-Scan-" + str(dataset_version) + '/data.yaml'
    dataset = project_directory + "/datasets/Blue-Rawdata-1504-500-" + str(dataset_version) + '/data.yaml'
    
    results = model.train(data = dataset, 
                          device="cuda", 
                          batch = 8, 
                          epochs = 100, 
                          imgsz = 1024, 
                          plots = True)