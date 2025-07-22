#!/usr/bin/env python3
"""
双模态数据集数据增强脚本
功能：对数据集进行多种数据增强变换，包括旋转、亮度、曝光、模糊等
"""

import os
import cv2
import numpy as np
import math
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import shutil
from tqdm import tqdm

class DataAugmenter:
    def __init__(self, config):
        self.config = config
        self.test_dir = str(config.target_dataset)
        self.augmented_dir = config.augmented_dataset
        self.create_output_dirs()
        
    def create_output_dirs(self):
        """创建输出目录结构"""
        for subdir in ['images_b', 'images_w', 'labels']:
            os.makedirs(os.path.join(self.augmented_dir, subdir), exist_ok=True)
            

    def adjust_brightness(self, image, factor):
        """调整图像亮度"""
        if isinstance(image, np.ndarray):
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] *= factor
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        else:
            return ImageEnhance.Brightness(image).enhance(factor)
    
    def adjust_exposure(self, image, factor):
        """调整图像曝光"""
        if isinstance(image, np.ndarray):
            gamma = 1.0 / factor if factor > 1 else 2.0 - factor
            gamma_table = np.array([((i / 255.0) ** gamma) * 255 
                                  for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, gamma_table)
        else:
            return ImageEnhance.Contrast(image).enhance(factor)
    
    def apply_blur(self, image, blur_factor):
        """应用模糊效果"""
        if isinstance(image, np.ndarray):
            kernel_size = int(blur_factor * 2) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            return image.filter(ImageFilter.GaussianBlur(radius=blur_factor * 1.5))
    
    def parse_label_file(self, label_path, img_width, img_height):
        """解析标签文件"""
        annotations = []
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    if not (line := line.strip()):
                        continue
                    
                    parts = line.split()
                    class_id = int(parts[0])
                    
                    coords = [(float(parts[i]) * img_width, float(parts[i + 1]) * img_height) 
                             for i in range(1, len(parts), 2) if i + 1 < len(parts)]
                    
                    annotations.append((class_id, coords))
        except Exception as e:
            print(f"解析标签文件失败: {e}")
            return []
        
        return annotations
    
    def save_label_file(self, annotations, output_path, img_width, img_height):
        """保存标签文件"""
        try:
            with open(output_path, 'w') as f:
                for class_id, coords in annotations:
                    line_parts = [str(class_id)]
                    for x, y in coords:
                        line_parts.extend([f"{x/img_width:.6f}", f"{y/img_height:.6f}"])
                    f.write(' '.join(line_parts) + '\n')
        except Exception as e:
            print(f"保存标签文件失败: {e}")
    
    def augment_single_image(self, blue_img_path, white_img_path, label_path, 
                           augmentation_params, output_suffix):
        """对单张图像应用增强"""
        try:
            # 读取图像
            blue_img = cv2.imread(blue_img_path)
            white_img = cv2.imread(white_img_path)
            
            if blue_img is None or white_img is None:
                return False
            
            # 调整白光图像尺寸以匹配蓝光图像
            blue_height, blue_width = blue_img.shape[:2]
            white_height, white_width = white_img.shape[:2]
            
            if (blue_height, blue_width) != (white_height, white_width):
                white_img = cv2.resize(white_img, (blue_width, blue_height))
            
            img_height, img_width = blue_height, blue_width
            
            # 解析标签
            annotations = self.parse_label_file(label_path, img_width, img_height)
            
            # 应用增强变换
            augmented_blue = blue_img.copy()
            augmented_white = white_img.copy()
            augmented_annotations = annotations.copy()
            
            # 应用旋转
            if 'rotation' in augmentation_params:
                angle = augmentation_params['rotation']
                center = (img_width // 2, img_height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                
                augmented_blue = cv2.warpAffine(augmented_blue, rotation_matrix, (img_width, img_height))
                augmented_white = cv2.warpAffine(augmented_white, rotation_matrix, (img_width, img_height))
                
                # 应用旋转变换到标注点
                augmented_annotations = []
                for class_id, coords in annotations:
                    if coords:
                        points_array = np.array(coords, dtype=np.float32)
                        ones = np.ones((points_array.shape[0], 1), dtype=np.float32)
                        homogeneous_points = np.hstack([points_array, ones])
                        
                        rotated_homogeneous = rotation_matrix.dot(homogeneous_points.T).T
                        
                        rotated_coords = [(max(0, min(img_width - 1, pt[0])), 
                                         max(0, min(img_height - 1, pt[1]))) 
                                        for pt in rotated_homogeneous]
                        
                        augmented_annotations.append((class_id, rotated_coords))
                    else:
                        augmented_annotations.append((class_id, coords))
            
            # 应用其他变换
            for param_name in ['brightness', 'exposure', 'blur']:
                if param_name in augmentation_params:
                    factor = augmentation_params[param_name]
                    transform_func = getattr(self, f"adjust_{param_name}" if param_name != 'blur' else 'apply_blur')
                    augmented_blue = transform_func(augmented_blue, factor)
                    augmented_white = transform_func(augmented_white, factor)
            
            # 保存增强后的图像和标签
            base_name = os.path.splitext(os.path.basename(blue_img_path))[0]
            
            # 保存图像
            blue_output = os.path.join(self.augmented_dir, 'images_b', f"{base_name}_{output_suffix}.jpg")
            white_output = os.path.join(self.augmented_dir, 'images_w', 
                                      f"{os.path.splitext(os.path.basename(white_img_path))[0]}_{output_suffix}.jpg")
            
            cv2.imwrite(blue_output, augmented_blue)
            success = cv2.imwrite(white_output, augmented_white)
            
            if not success or not os.path.exists(white_output):
                return False
            
            # 保存标签
            label_output = os.path.join(self.augmented_dir, 'labels', f"{base_name}_{output_suffix}.txt")
            self.save_label_file(augmented_annotations, label_output, img_width, img_height)
            
            return True
            
        except Exception:
            return False
    
    def get_augmentation_strategies(self):
        """定义数据增强策略"""
        return self.config.strategies if self.config.strategies else {'0': {}}
    
    def get_corresponding_files(self, blue_image_path):
        """获取对应的白光图片和标签文件"""
        blue_filename = os.path.basename(blue_image_path)
        
        import re
        pattern = r'(\d{4}-\d{2}-\d{2}_\d{6})_(\d+)_([A-Z]\d+)_(\d+)'
        base_name = blue_filename.replace('.jpg', '')
        match = re.match(pattern, base_name)
        
        if match:
            timestamp, id_part = match.group(1), match.group(2)
            
            white_images_dir = os.path.join(self.test_dir, 'images_w')
            if os.path.exists(white_images_dir):
                white_pattern = f"{timestamp}_{id_part}_*.jpg"
                import glob
                white_candidates = glob.glob(os.path.join(white_images_dir, white_pattern))
                
                if white_candidates:
                    white_image_path = white_candidates[0]
                else:
                    return None, None
            else:
                return None, None
        else:
            return None, None
        
        # 构建标签文件路径
        label_filename = blue_filename.replace('.jpg', '.txt')
        label_path = os.path.join(self.test_dir, 'labels', label_filename)
        
        return white_image_path, label_path
    
    def augment_dataset(self):
        """对整个数据集进行增强"""
        # 获取所有蓝光图片
        images_dir = os.path.join(self.test_dir, 'images_b')
        blue_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        
        if not blue_images:
            print("没有找到图片文件")
            return
        
        strategies = self.get_augmentation_strategies()
        
        total_success = 0
        total_attempted = 0
        
        # 创建嵌套进度条
        with tqdm(blue_images, desc="Processing images") as pbar:
            for img_filename in pbar:
                blue_img_path = os.path.join(images_dir, img_filename)
                white_img_path, label_path = self.get_corresponding_files(blue_img_path)
                
                if not white_img_path or not label_path:
                    continue
                    
                if not os.path.exists(white_img_path) or not os.path.exists(label_path):
                    continue
                
                pbar.set_postfix({"Current": img_filename[:20] + "..."})
                
                strategy_success = 0
                for strategy_name, params in strategies.items():
                    total_attempted += 1
                    success = self.augment_single_image(
                        blue_img_path, white_img_path, label_path, 
                        params, strategy_name
                    )
                    
                    if success:
                        total_success += 1
                        strategy_success += 1
                
                pbar.set_postfix({
                    "Current": img_filename[:15] + "...",
                    "Strategies": f"{strategy_success}/{len(strategies)}"
                })
        
        print(f"\n数据增强完成! 成功: {total_success}/{total_attempted}")
        print(f"输出目录: {self.augmented_dir}")

if __name__ == "__main__":
    from d_dataset_config import DatasetConfig
    
    config = DatasetConfig(version=1, split="test")
    augmenter = DataAugmenter(config)
    augmenter.augment_dataset()