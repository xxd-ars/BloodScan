import os
from PIL import Image

def crop_yolo_segment_labels(
    image_path: str,
    label_path: str,
    output_image_path: str,
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
    cropped_img.save(output_image_path)
    
    # ---- 3. 如果没有标签文件，直接返回（有些数据可能没有对应txt）----
    if not os.path.exists(label_path):
        return
    
    # ---- 4. 读取原标签并进行多边形坐标转换 ----
    new_label_lines = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 例： "0 0.508986928 ... 0.3427734375"
            parts = line.split()
            class_id = parts[0]
            # 余下的坐标对
            coords = parts[1:]
            
            # 将 coords 拆分成 (x_i, y_i) 对，注意它们都是字符串
            xy_pairs = list(zip(coords[0::2], coords[1::2]))  # [(x1, y1), (x2, y2), ...]
            
            new_xy_pairs = []
            for x_str, y_str in xy_pairs:
                x_norm = float(x_str)
                y_norm = float(y_str)
                
                # (1) 先从归一化坐标 -> 原图像素坐标
                x_abs = x_norm * orig_w
                y_abs = y_norm * orig_h
                
                # (2) 做裁剪平移 (减去 crop_left, crop_top)，得到在裁剪后图中的绝对坐标
                x_cropped_abs = x_abs - crop_left
                y_cropped_abs = y_abs - crop_top
                
                # 如果该点已完全落在裁剪区域之外（例如 x_cropped_abs < 0 或 > crop_width），
                # 在此可以选择直接跳过，或者进行边界裁剪(clamp)。
                # 下面演示简单的 clamp 操作（若要直接丢弃可自行修改）
                if x_cropped_abs < 0: 
                    x_cropped_abs = 0
                elif x_cropped_abs > crop_width:
                    x_cropped_abs = crop_width
                if y_cropped_abs < 0:
                    y_cropped_abs = 0
                elif y_cropped_abs > crop_height:
                    y_cropped_abs = crop_height
                
                # (3) 转回新的归一化坐标 [0,1]
                new_x_norm = x_cropped_abs / crop_width
                new_y_norm = y_cropped_abs / crop_height
                
                # 若希望严格过滤完全不在图内的点，可以在此判断 new_x_norm, new_y_norm
                # 是否在 [0,1] 范围之外来进行剔除，本示例保留且做了 clamp。
                
                new_xy_pairs.append((new_x_norm, new_y_norm))
            
            # 将新的坐标对重新写回 line
            new_line = [class_id]
            for (nx, ny) in new_xy_pairs:
                new_line.append(f"{nx}")
                new_line.append(f"{ny}")
            
            new_label_lines.append(" ".join(new_line) + "\n")
    
    # ---- 5. 将转换好的新的标签写入到新目录的 label 文件中 ----
    with open(output_label_path, 'w') as f:
        f.writelines(new_label_lines)


def main():
    # ============ 需要根据自己的需求来修改的参数 ============
    # 下面以 “test” 为示例，train / valid 的处理同理
    file_name = 'valid'
    file_dir = './tests/yolo_seg/datasets/Blood-Scan-8/' + file_name
    file_dir_ = './tests/yolo_seg/datasets/Blood-Scan-8-cropped/' + file_name
    
    image_dir = os.path.join(file_dir, 'images')
    label_dir = os.path.join(file_dir, 'labels')
    output_image_dir = os.path.join(file_dir_, 'images')
    output_label_dir = os.path.join(file_dir_, 'labels')
    
    # 如果这些输出文件夹不存在，先创建它们
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 假设原始图像是 2448 x 2048，想要裁剪掉周围不需要的区域。
    # 这里举个例子：从 (left=400, top=300) 开始裁 1648 x 1448，具体数值根据实际需求来定。
    crop_left = 800
    crop_top = 250
    crop_width = 1216
    crop_height = 1504 #1448
    
    # ============= 批量处理 =============
    # 遍历 test/images 下所有 jpg 图片
    for img_name in os.listdir(image_dir):
        if not img_name.lower().endswith('.jpg'):
            continue
        
        image_path = os.path.join(image_dir, img_name)
        
        # 对应的 label 文件同名但后缀为 .txt
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)
        
        # 输出的图像、标签路径
        output_image_path = os.path.join(output_image_dir, img_name)
        output_label_path = os.path.join(output_label_dir, label_name)
        
        # 调用裁剪处理函数
        crop_yolo_segment_labels(
            image_path, label_path,
            output_image_path, output_label_path,
            crop_left, crop_top, crop_width, crop_height
        )
    
    print("Done! 所有图像与标签已完成裁剪并输出。")

if __name__ == "__main__":
    main()