"""
双模态数据预处理脚本
将蓝光和白光图像拼接为6通道numpy文件，供YOLO训练使用
"""

import os
import numpy as np
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm
project_root = Path(__file__).parent.parent

def load_and_concat_images(blue_path, white_path, target_size=(1504, 1504)):
    """加载并拼接蓝光和白光图像为6通道张量"""
    # 加载蓝光图像
    blue_img = cv2.imread(str(blue_path))
    blue_img = cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB)
    blue_img = cv2.resize(blue_img, target_size)
    
    # 加载白光图像
    white_img = cv2.imread(str(white_path))
    white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2RGB)
    white_img = cv2.resize(white_img, target_size)
    
    # 拼接为6通道 (蓝光3通道 + 白光3通道)
    dual_img = np.concatenate([blue_img, white_img], axis=2)  # Shape: (H, W, 6)
    
    return dual_img

def create_6ch_dataset(dataset_root):
    """创建6通道数据集"""
    # 数据集路径
    output_root = dataset_root + '-6ch'
    dataset_root = project_root / 'datasets' / dataset_root
    output_root = project_root / 'datasets' / output_root
    
    # 创建输出目录
    for split in ['train', 'valid', 'test']:
        (output_root / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_root / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # 处理每个数据分割
    for split in ['train', 'valid', 'test']:
        print(f"处理 {split} 数据集...")
        
        blue_dir = dataset_root / split / 'images_b'
        white_dir = dataset_root / split / 'images_w'
        labels_dir = dataset_root / split / 'labels'
        
        output_img_dir = output_root / split / 'images'
        output_label_dir = output_root / split / 'labels'
        
        # 获取蓝光图像列表
        blue_images = sorted(list(blue_dir.glob('*.jpg')) + list(blue_dir.glob('*.png')))
        
        processed_count = 0
        
        for blue_path in tqdm(blue_images, desc=f"Processing {split}"):
            # 构建对应的白光图像路径
            base_name = blue_path.stem
            
            # 寻找对应的白光图像
            white_path = None
            for white_candidate in white_dir.glob('*.jpg'):
                # 提取编号部分进行匹配
                blue_parts = base_name.split('_')
                white_parts = white_candidate.stem.split('_')
                
                # 检查是否是对应的白光图像（相同时间戳和编号）
                if (len(blue_parts) >= 6 and len(white_parts) >= 6 and 
                    blue_parts[0] == white_parts[0] and  # 日期
                    blue_parts[1] == white_parts[1] and  # 时间
                    blue_parts[2] == white_parts[2] and  # 编号1
                    blue_parts[5] == white_parts[5]):    # 编号2（最后的数字）
                    white_path = white_candidate
                    break
            
            if white_path is None:
                print(f"Warning: 找不到对应的白光图像 for {blue_path.name}")
                continue
            
            try:
                # 加载并拼接图像
                dual_img = load_and_concat_images(blue_path, white_path)
                
                # 保存6通道numpy文件
                output_path = output_img_dir / f"{base_name}.npy"
                np.save(output_path, dual_img)
                
                # 复制对应的标签文件
                label_path = labels_dir / f"{base_name}.txt"
                if label_path.exists():
                    shutil.copy2(label_path, output_label_dir / f"{base_name}.txt")
                else:
                    print(f"Warning: 找不到标签文件 {label_path}")
                
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing {blue_path.name}: {e}")
        
        print(f"{split} 数据集处理完成: {processed_count} 个样本")
    
    # 创建数据配置文件
    data_config = """# 6通道双模态YOLO数据配置
# 蓝光+白光拼接为6通道numpy文件

# 数据集路径
train: train/images
val: valid/images  
test: test/images

# 类别信息
nc: 3
names: ['0', '1', '2']

# 双模态配置
channels: 6  # 6通道 (3蓝光 + 3白光)
data_format: 'npy'  # numpy格式
img_size: 1504
"""
    
    with open(output_root / 'data.yaml', 'w') as f:
        f.write(data_config)
    
    print(f"✅ 6通道数据集创建完成！")
    print(f"输出目录: {output_root}")
    print(f"数据配置: {output_root / 'data.yaml'}")
    
    return output_root

if __name__ == "__main__":
    output_dir = create_6ch_dataset('Dual-Modal-1504-500-1')