import os
import cv2
import numpy as np
from pathlib import Path
import shutil

def create_6ch_dataset(dataset_path, output_path):
    """将蓝光+白光图片合并为6通道数据集"""
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # 创建输出目录结构
    for split in ['train', 'valid', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'valid', 'test']:
        blue_dir = dataset_path / split / 'images'
        white_dir = dataset_path / split / 'images_w'
        label_dir = dataset_path / split / 'labels'
        
        output_img_dir = output_path / split / 'images'
        output_label_dir = output_path / split / 'labels'
        
        if not blue_dir.exists():
            continue
            
        print(f"处理 {split} 数据...")
        
        # 获取蓝光图片列表
        blue_files = list(blue_dir.glob('*.bmp')) + list(blue_dir.glob('*.jpg')) + list(blue_dir.glob('*.png'))
        
        for blue_file in blue_files:
            # 根据文件名模式匹配白光图片
            # 蓝光: 2022-03-28_103204_3_T5_2356.bmp
            # 白光: 2022-03-28_103204_3_T3_2354.bmp
            
            # 提取基础名称（到T5之前的部分）
            base_name = blue_file.name.split('_T5_')[0]
            
            # 在白光目录中查找匹配的文件
            white_file = None
            for wf in white_dir.glob(f"{base_name}_T3_*{blue_file.suffix}"):
                white_file = wf
                break
            
            if white_file is None:
                print(f"警告: 找不到匹配的白光图片 {base_name}_T3_*")
                continue
            
            # 读取图片
            blue_img = cv2.imread(str(blue_file))
            white_img = cv2.imread(str(white_file))
            
            if blue_img is None or white_img is None:
                print(f"警告: 无法读取图片 {blue_file} 或 {white_file}")
                continue
            
            # 合并为6通道 (B,G,R,B,G,R)
            merged_img = np.concatenate([blue_img, white_img], axis=2)
            
            # 保存为.npy格式
            output_file = output_img_dir / f"{blue_file.stem}.npy"
            np.save(output_file, merged_img)
            
            # 复制对应的标注文件
            label_file = label_dir / f"{blue_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, output_label_dir / f"{blue_file.stem}.txt")
        
        print(f"{split} 完成，处理了 {len(blue_files)} 张图片")

if __name__ == "__main__":
    dataset_path = "datasets/Blue-Rawdata-1504-500-1"
    output_path = "datasets/Blue-Rawdata-6ch"
    
    create_6ch_dataset(dataset_path, output_path)
    print("6通道数据集创建完成！") 