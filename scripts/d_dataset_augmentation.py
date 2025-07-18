#!/usr/bin/env python3
"""
双模态数据集数据增强脚本
功能：对test数据集进行多种数据增强变换，包括旋转、亮度、曝光、模糊等
"""

import os
import cv2
import numpy as np
import math
from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import shutil

class DataAugmenter:
    def __init__(self, dataset_root):
        self.test_dir = dataset_root
        self.augmented_dir = dataset_root + f'_augmented_{len(self.get_augmentation_strategies())}'
        
        # 创建增强数据输出目录
        self.create_output_dirs()
        
    def create_output_dirs(self):
        """创建输出目录结构"""
        dirs_to_create = [
            os.path.join(self.augmented_dir, 'images_b'),
            os.path.join(self.augmented_dir, 'images_w'),
            os.path.join(self.augmented_dir, 'labels')
        ]
        
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
            

    def adjust_brightness(self, image, factor):
        """调整图像亮度"""
        if isinstance(image, np.ndarray):
            # OpenCV格式
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] *= factor  # 调整V通道（亮度）
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        else:
            # PIL格式
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
    
    def adjust_exposure(self, image, factor):
        """调整图像曝光"""
        if isinstance(image, np.ndarray):
            # OpenCV格式 - 通过gamma校正模拟曝光调整
            gamma = 1.0 / factor if factor > 1 else 2.0 - factor
            gamma_table = np.array([((i / 255.0) ** gamma) * 255 
                                  for i in np.arange(0, 256)]).astype("uint8")
            return cv2.LUT(image, gamma_table)
        else:
            # PIL格式 - 通过对比度调整模拟曝光
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
    
    def apply_blur(self, image, blur_factor):
        """应用模糊效果"""
        if isinstance(image, np.ndarray):
            # OpenCV格式
            kernel_size = int(blur_factor * 2) + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        else:
            # PIL格式
            radius = blur_factor * 1.5
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def parse_label_file(self, label_path, img_width, img_height):
        """解析标签文件"""
        annotations = []
        
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    class_id = int(parts[0])
                    
                    # 提取坐标点并转换为像素坐标
                    coords = []
                    for i in range(1, len(parts), 2):
                        if i + 1 < len(parts):
                            x_norm = float(parts[i])
                            y_norm = float(parts[i + 1])
                            x_pixel = x_norm * img_width
                            y_pixel = y_norm * img_height
                            coords.append((x_pixel, y_pixel))
                    
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
                    
                    # 转换回归一化坐标
                    for x, y in coords:
                        x_norm = x / img_width
                        y_norm = y / img_height
                        line_parts.extend([f"{x_norm:.6f}", f"{y_norm:.6f}"])
                    
                    f.write(' '.join(line_parts) + '\n')
                    
        except Exception as e:
            print(f"保存标签文件失败: {e}")
    
    def augment_single_image(self, blue_img_path, white_img_path, label_path, 
                           augmentation_params, output_suffix):
        """对单张图像应用增强"""
        
        # 读取图像
        blue_img = cv2.imread(blue_img_path)
        white_img = cv2.imread(white_img_path)
        
        if blue_img is None:
            print(f"无法读取蓝光图像: {blue_img_path}")
            return False
        
        if white_img is None:
            print(f"无法读取白光图像: {white_img_path}")
            return False
        
        # 检查图像尺寸
        blue_height, blue_width = blue_img.shape[:2]
        white_height, white_width = white_img.shape[:2]
        
        # 如果尺寸不匹配，调整白光图片尺寸
        if (blue_height, blue_width) != (white_height, white_width):
            print(f"  警告: 图像尺寸不匹配 - 蓝光:{blue_width}x{blue_height}, 白光:{white_width}x{white_height}")
            print(f"  正在调整白光图片尺寸...")
            white_img = cv2.resize(white_img, (blue_width, blue_height))
            print(f"  白光图片已调整为: {blue_width}x{blue_height}")
        
        img_height, img_width = blue_height, blue_width
        
        # 解析标签
        annotations = self.parse_label_file(label_path, img_width, img_height)
        
        # 应用增强变换
        augmented_blue = blue_img.copy()
        augmented_white = white_img.copy()
        augmented_annotations = []
        
        # 应用旋转
        if 'rotation' in augmentation_params:
            angle = augmentation_params['rotation']
            
            # 验证两个图像尺寸一致
            blue_h, blue_w = augmented_blue.shape[:2]
            white_h, white_w = augmented_white.shape[:2]
            
            if (blue_h, blue_w) != (white_h, white_w):
                print(f"    错误: 旋转前图像尺寸不匹配! 蓝光:{blue_w}x{blue_h}, 白光:{white_w}x{white_h}")
                return False
            
            # 计算旋转矩阵（一次性计算）
            height, width = blue_h, blue_w
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # 旋转图像
            augmented_blue = cv2.warpAffine(augmented_blue, rotation_matrix, (width, height), 
                                          borderValue=(0, 0, 0))
            augmented_white = cv2.warpAffine(augmented_white, rotation_matrix, (width, height), 
                                           borderValue=(0, 0, 0))
            
            # 验证旋转后的图像
            if augmented_blue is None:
                print(f"    错误: 蓝光图像旋转失败!")
                return False
            if augmented_white is None:
                print(f"    错误: 白光图像旋转失败!")
                return False
                
            print(f"    ✓ 蓝光和白光图像旋转完成")
            
            # 对所有标注点应用相同的旋转变换
            augmented_annotations = []
            for class_id, coords in annotations:
                if coords:
                    # 将点转换为齐次坐标
                    points_array = np.array(coords, dtype=np.float32)
                    ones = np.ones((points_array.shape[0], 1), dtype=np.float32)
                    homogeneous_points = np.hstack([points_array, ones])
                    
                    # 应用旋转变换
                    rotated_homogeneous = rotation_matrix.dot(homogeneous_points.T).T
                    
                    # 确保坐标在图像范围内
                    rotated_coords = []
                    for i in range(rotated_homogeneous.shape[0]):
                        x = max(0, min(width - 1, rotated_homogeneous[i, 0]))
                        y = max(0, min(height - 1, rotated_homogeneous[i, 1]))
                        rotated_coords.append((x, y))
                    
                    augmented_annotations.append((class_id, rotated_coords))
                else:
                    augmented_annotations.append((class_id, coords))
        else:
            augmented_annotations = annotations.copy()
        
        # 应用亮度调整
        if 'brightness' in augmentation_params:
            factor = augmentation_params['brightness']
            augmented_blue = self.adjust_brightness(augmented_blue, factor)
            augmented_white = self.adjust_brightness(augmented_white, factor)
        
        # 应用曝光调整
        if 'exposure' in augmentation_params:
            factor = augmentation_params['exposure']
            augmented_blue = self.adjust_exposure(augmented_blue, factor)
            augmented_white = self.adjust_exposure(augmented_white, factor)
        
        # 应用模糊
        if 'blur' in augmentation_params:
            factor = augmentation_params['blur']
            augmented_blue = self.apply_blur(augmented_blue, factor)
            augmented_white = self.apply_blur(augmented_white, factor)
        
        # 保存增强后的图像和标签
        base_name = os.path.splitext(os.path.basename(blue_img_path))[0]
        
        # 保存蓝光图像
        blue_output_path = os.path.join(self.augmented_dir, 'images_b', 
                                       f"{base_name}_{output_suffix}.jpg")
        cv2.imwrite(blue_output_path, augmented_blue)
        
        # 保存白光图像
        white_base_name = os.path.splitext(os.path.basename(white_img_path))[0]
        white_output_path = os.path.join(self.augmented_dir, 'images_w', 
                                        f"{white_base_name}_{output_suffix}.jpg")
        
        # 验证白光图像是否为空
        if augmented_white is None:
            print(f"    错误: 白光图像为空，无法保存!")
            return False
            
        white_save_success = cv2.imwrite(white_output_path, augmented_white)
        
        if not white_save_success:
            print(f"    错误: 白光图像保存失败! 路径: {white_output_path}")
            return False
        
        # 验证保存的文件确实存在
        if not os.path.exists(white_output_path):
            print(f"    错误: 白光图像文件未找到! 路径: {white_output_path}")
            return False
        
        # 保存标签
        label_output_path = os.path.join(self.augmented_dir, 'labels', 
                                        f"{base_name}_{output_suffix}.txt")
        self.save_label_file(augmented_annotations, label_output_path, img_width, img_height)
        
        return True
    
    def get_augmentation_strategies(self):
        """定义数据增强策略"""
        strategies = {
            # 原图（复制）
            '0': {},
            # 二元组合 1-4
            '1': {'rotation': 5,    'blur': 1.5},
            '2': {'rotation': -5,   'blur': 1.5},
            '3': {'rotation': 5,    'exposure': 0.9},
            '4': {'rotation': -5,   'exposure': 1.1},
            # 三元组合 5-8
            '5': {'rotation': 10,   'brightness': 1.15, 'blur': 1.2},
            '6': {'rotation': -10,  'brightness': 0.85, 'blur': 1.2},
            '7': {'rotation': 5,    'exposure': 0.9,    'blur': 1.2},
            '8': {'rotation': -5,   'exposure': 1.1,    'blur': 1.2},
            }
        strategies_old = {
            # 原图（复制）
            'original': {},
            
            # 单一变换
            'rot_neg10': {'rotation': -10},
            'rot_neg5': {'rotation': -5},
            'rot_pos5': {'rotation': 5},
            'rot_pos10': {'rotation': 10},
            
            'bright_low': {'brightness': 0.85},  # -15%
            'bright_high': {'brightness': 1.15},  # +15%
            
            'exp_low': {'exposure': 0.9},   # -10%
            'exp_high': {'exposure': 1.1},  # +10%
            
            'blur_light': {'blur': 1.2},
            'blur_medium': {'blur': 1.5},
            'blur_heavy': {'blur': 2.0},
            
            # 二元组合 1-4
            'rot5_blur_medium': {'rotation': 5, 'blur': 1.5},
            'rot_neg5_blur_medium': {'rotation': -5, 'blur': 1.5},
            'rot5_exp_low': {'rotation': 5, 'exposure': 0.9},
            'rot_neg5_exp_high': {'rotation': -5, 'exposure': 1.1},
            
            # 'bright_low_blur': {'brightness': 0.85, 'blur': 1.2},
            # 'bright_high_blur': {'brightness': 1.15, 'blur': 1.2},
            # 'exp_low_blur': {'exposure': 0.9, 'blur': 1.2},
            # 'exp_high_blur': {'exposure': 1.1, 'blur': 1.2},
            
            # 三元组合 5-8
            'rot10_bright_blur': {'rotation': 10, 'brightness': 1.15, 'blur': 1.2},
            'rot_neg10_bright_blur': {'rotation': -10, 'brightness': 0.85, 'blur': 1.2},
            'rot5_exp_blur': {'rotation': 5, 'exposure': 0.9, 'blur': 1.2},
            'rot_neg5_exp_blur': {'rotation': -5, 'exposure': 1.1, 'blur': 1.2},
            
            # 强化组合
            'heavy_aug1': {'rotation': 10, 'brightness': 1.15, 'exposure': 1.1, 'blur': 1.5},
            'heavy_aug2': {'rotation': -10, 'brightness': 0.85, 'exposure': 0.9, 'blur': 1.5},
        }
        return strategies
    
    def get_corresponding_files(self, blue_image_path):
        """获取对应的白光图片和标签文件"""
        blue_filename = os.path.basename(blue_image_path)
        
        # 解析新格式的文件名：yyyy-mm-dd-HHMMSS_id_xx_xxxx.jpg
        import re
        pattern = r'(\d{4}-\d{2}-\d{2}_\d{6})_(\d+)_([A-Z]\d+)_(\d+)'
        base_name = blue_filename.replace('.jpg', '')
        match = re.match(pattern, base_name)
        
        if match:
            timestamp = match.group(1)  # yyyy_mm_dd_HHMMSS
            id_part = match.group(2)    # id
            type_part = match.group(3)  # xx (like T5)
            number_part = match.group(4) # xxxx
            
            # 在白光图片目录中查找具有相同时间戳和id的文件
            white_images_dir = os.path.join(self.test_dir, 'images_w')
            if os.path.exists(white_images_dir):
                # 查找匹配的白光图片文件
                white_pattern = f"{timestamp}_{id_part}_*.jpg"
                import glob
                white_candidates = glob.glob(os.path.join(white_images_dir, white_pattern))
                
                if white_candidates:
                    # 选择第一个匹配的文件
                    white_image_path = white_candidates[0]
                else:
                    print(f"  警告: 未找到匹配的白光图片 {white_pattern}")
                    return None, None
            else:
                print(f"  警告: 白光图片目录不存在: {white_images_dir}")
                return None, None
        else:
            print(f"  警告: 无法解析蓝光图片文件名格式: {blue_filename}")
            return None, None
        
        # 构建标签文件路径
        label_filename = blue_filename.replace('.jpg', '.txt')
        label_path = os.path.join(self.test_dir, 'labels', label_filename)
        
        return white_image_path, label_path
    
    def augment_dataset(self):
        """对整个数据集进行增强"""
        print("开始数据增强...")
        
        # 获取所有蓝光图片
        images_dir = os.path.join(self.test_dir, 'images_b')
        blue_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
        
        if not blue_images:
            print("没有找到图片文件")
            return
        
        strategies = self.get_augmentation_strategies()
        
        print(f"找到 {len(blue_images)} 张原始图片")
        print(f"将应用 {len(strategies)} 种增强策略")
        print(f"预计生成 {len(blue_images) * len(strategies)} 张增强图片")
        
        total_success = 0
        total_attempted = 0
        
        for img_filename in blue_images:
            blue_img_path = os.path.join(images_dir, img_filename)
            white_img_path, label_path = self.get_corresponding_files(blue_img_path)
            
            if not white_img_path or not label_path:
                print(f"跳过 {img_filename}: 无法找到对应文件")
                continue
                
            if not os.path.exists(white_img_path) or not os.path.exists(label_path):
                print(f"跳过 {img_filename}: 对应文件不存在")
                continue
            
            print(f"\n处理图片: {img_filename}")
            
            for strategy_name, params in strategies.items():
                total_attempted += 1
                success = self.augment_single_image(
                    blue_img_path, white_img_path, label_path, 
                    params, strategy_name
                )
                
                if success:
                    total_success += 1
                else:
                    print(f"  ✗ {strategy_name} 失败")
        
        print(f"\n数据增强完成!")
        print(f"成功: {total_success}/{total_attempted}")
        print(f"输出目录: {self.augmented_dir}")

if __name__ == "__main__":
    """主函数"""
    project_root = Path(__file__).parent.parent
    dataset_root = project_root / "datasets" / "Dual-Modal-1504-500-0-mac" / "test"
    
    print("=" * 60)
    print("双模态数据集数据增强工具")
    print("=" * 60)
    print(f"数据集路径: {dataset_root}")
    
    # 创建数据增强器
    augmenter = DataAugmenter(str(dataset_root))
    augmenter.augment_dataset()