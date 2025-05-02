import os, json, cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def crop_yolo_segment_labels(
    image_path,
    label_path,
    output_image_path,
    output_label_path: str,
    crop_left: int,
    crop_top: int,
    crop_width: int,
    crop_height: int
):
    """
    对单张图像及其对应的 YOLO Segment 标签做裁剪，并保存。
    如果需要更复杂的裁剪逻辑（如多边形部分越界处理），可在此函数内扩展。
    """
    # ---- 1. 打开原图并获取宽高 ----
    if isinstance(image_path, list):
        try:
            img = Image.open(image_path[0])
            output_image_path = output_image_path[0]
        except:
            img = Image.open(image_path[1])
            output_image_path = output_image_path[1]

    elif isinstance(image_path, str):
        img = Image.open(image_path)

    orig_w, orig_h = img.size
    
    # ---- 2. 进行图像裁剪并保存 ----
    # crop(box)中的box = (left, upper, right, lower)
    cropped_img = img.crop((
        crop_left,
        crop_top,
        crop_left + crop_width,
        crop_top + crop_height
    ))
    cropped_img = cropped_img.resize((max(crop_width, crop_height), max(crop_width, crop_height)), Image.Resampling.LANCZOS)
    cropped_img.save(output_image_path)
    # plt.imshow(cropped_img)
    # plt.show()
    # ---- 3. 如果没有标签文件，直接返回（有些数据可能没有对应txt）----
    if not os.path.exists(label_path):
        return
    else:
        return
    with open(label_path, 'r') as f:
        data = json.load(f)
    cropped_img = cv2.imread(output_image_path)
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
    
    for id, shape in enumerate(data["shapes"]):
        label = shape["label"]
        true_point = shape["points"][0]
        x, y = int(true_point[0]), int(true_point[1])
        x, y = int((x - crop_left) * crop_height/crop_width), int((y - crop_top) * 1)

        cv2.circle(cropped_img, (x, y), radius=8, color=(255, 0, 0), thickness=-1)  # Red circle
        cv2.putText(cropped_img, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # Green text
    plt.imshow(cropped_img)
    plt.show()

def main():
    class_name = "class1"
    file_dir_in  = f'./data/rawdata/{class_name}'
    file_dir_out = f'./data/rawdata_cropped_white/{class_name}'

    os.makedirs(file_dir_out, exist_ok=True)
    # os.makedirs(output_label_dir, exist_ok=True)

    # 假设原始图像是 2448 x 2048，想要裁剪掉周围不需要的区域。
    # 这里举个例子：从 (left=400, top=300) 开始裁 1648 x 1448，具体数值根据实际需求来定。
    crop_left = 800
    crop_top = 250
    crop_width = 1216
    crop_height = 1504 #1448

    json_files = [file for file in os.listdir(file_dir_in) if file.endswith('.json')]
    for json_name in json_files:
        
        with open(os.path.join(file_dir_in, json_name), 'r') as f:
            data = json.load(f)
        if class_name == 'class1':
            whit_image_name = [data["imagePath"].replace("T5", "T3").rsplit("_", 1)[0] + f"_{int(data["imagePath"].rsplit('_', 1)[1].split('.')[0]) - 2}.bmp",
                               data["imagePath"]]
            blue_image_name = data["imagePath"]
        elif class_name == 'class2':
            whit_image_name = data["imagePath"]
            blue_image_name = data["imagePath"].replace("T3", "T5").rsplit("_", 1)[0] + f"_{int(data["imagePath"].rsplit('_', 1)[1].split('.')[0]) + 2}.bmp"
        
        image_name = whit_image_name

        if isinstance(image_name, list):
            image_path = [os.path.join(file_dir_in, image_name[0]), os.path.join(file_dir_in, image_name[1])]
        elif isinstance(image_name, str):
            image_path = os.path.join(file_dir_in, image_name)
        json_path  = os.path.join(file_dir_in, json_name)
        
        # img = Image.open(image_path)
        # plt.imshow(img)
        # plt.show()
        # break
        # 输出的图像、标签路径
        if isinstance(image_name, list):
            output_image_path = [os.path.join(file_dir_out, image_name[0]), os.path.join(file_dir_out, image_name[1])]
            output_label_path = os.path.join(file_dir_out, json_name)
        elif isinstance(image_name, str):
            output_image_path = os.path.join(file_dir_out, image_name)
            output_label_path = os.path.join(file_dir_out, json_name)
            
        # 调用裁剪处理函数
        crop_yolo_segment_labels(
            image_path, json_path,
            output_image_path, output_label_path,
            crop_left, crop_top, crop_width, crop_height
        )
        # break
    # print("Done! 所有图像与标签已完成裁剪并输出。")

if __name__ == "__main__":
    main()