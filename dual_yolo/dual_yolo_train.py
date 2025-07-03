import sys
import os
# 确保使用本地修改的ultralytics代码
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ultralytics import YOLO

if __name__ == '__main__':
    dataset_version = 1
    project_directory = "/Users/ASUS/Documents/SJTU M2/Graduation Project/BloodScan/"
    # os.chdir(target_directory)

    model = YOLO('./dual_yolo/models/yolo11x-dseg-concat.yaml').load('./dual_yolo/weights/dual_yolo11x.pt')
    model.info(verbose=True)

    # dataset = project_directory + "/datasets/Blood-Scan-" + str(dataset_version) + '/data.yaml'
    dataset = project_directory + "/datasets/Blue-Rawdata-1504-500-" + str(dataset_version) + '/data.yaml'
    
    # results = model.train(data = dataset, 
    #                     #   device="cuda", 
    #                       device="0,1,2,3",
    #                       batch = 4, 
    #                       epochs = 10, 
    #                       imgsz = 1504, 
    #                       plots = True)
    

    results = model.train(
        data='datasets/Blue-Rawdata-6ch/data.yaml',
        epochs=1,
        imgsz=1504,
        batch=1,
        name='6ch_dual_modal'
    )