#!/usr/bin/env python3
"""
蓝光白光图片匹配脚本
功能：将数据集中的蓝光图片与rawdata_cropped_white中的白光图片进行匹配，并转换为JPG格式
"""

import os
import shutil
import re
from pathlib import Path
from PIL import Image

def parse_blue_filename(filename):
    """
    解析蓝光图片文件名，提取关键信息
    例如：2022-03-28_103204_10_T5_2384_bmp.rf.hash.jpg
    返回：(timestamp, sequence, number)
    """
    # 移除.rf.hash.jpg后缀
    base_name = filename.replace('.jpg', '').split('.rf.')[0]
    
    # 解析模式：YYYY-MM-DD_HHMMSS_SEQ_T5_NUM_bmp
    pattern = r'(\d{4}-\d{2}-\d{2}_\d{6})_(\d+)_T5_(\d+)_bmp'
    match = re.match(pattern, base_name)
    
    if match:
        timestamp = match.group(1)
        sequence = match.group(2)
        number = int(match.group(3))
        return timestamp, sequence, number
    
    return None, None, None

def generate_white_filename(timestamp, sequence, blue_number, output_format='jpg'):
    """
    根据蓝光图片信息生成对应的白光图片文件名
    蓝光数字比白光大2
    """
    white_number = blue_number - 2
    if output_format == 'bmp':
        return f"{timestamp}_{sequence}_T3_{white_number}.bmp"
    else:
        return f"{timestamp}_{sequence}_T3_{white_number}.jpg"

def find_matching_white_image(white_dir, white_filename):
    """
    在白光目录中查找匹配的文件
    """
    white_path = os.path.join(white_dir, white_filename)
    return white_path if os.path.exists(white_path) else None

def convert_bmp_to_jpg(bmp_path, jpg_path, quality=95):
    """
    将BMP图片转换为JPG格式
    """
    try:
        with Image.open(bmp_path) as img:
            # 如果图片是RGBA模式，转换为RGB
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 保存为JPG格式
            img.save(jpg_path, 'JPEG', quality=quality)
            return True
    except Exception as e:
        print(f"    转换失败：{e}")
        return False

def process_dataset_directory(dataset_dir, white_source_dir):
    """
    处理数据集目录（train/valid/test）
    """
    images_dir = os.path.join(dataset_dir, 'images')
    images_w_dir = os.path.join(dataset_dir, 'images_w')
    
    if not os.path.exists(images_dir):
        print(f"跳过 {dataset_dir}：images目录不存在")
        return
    
    # 创建images_w目录
    os.makedirs(images_w_dir, exist_ok=True)
    print(f"创建目录：{images_w_dir}")
    
    # 统计变量
    total_blue = 0
    matched = 0
    failed = 0
    
    # 处理所有蓝光图片
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg'):
            total_blue += 1
            
            # 解析蓝光图片文件名
            timestamp, sequence, blue_number = parse_blue_filename(filename)
            
            if timestamp is None:
                print(f"  警告：无法解析文件名 {filename}")
                failed += 1
                continue
            
            # 生成对应的白光图片文件名（BMP格式用于查找源文件）
            white_bmp_filename = generate_white_filename(timestamp, sequence, blue_number, 'bmp')
            # 生成输出的JPG文件名
            white_jpg_filename = generate_white_filename(timestamp, sequence, blue_number, 'jpg')
            
            # 查找白光图片
            white_path = find_matching_white_image(white_source_dir, white_bmp_filename)
            
            if white_path:
                # 转换BMP为JPG并保存到images_w目录
                dest_path = os.path.join(images_w_dir, white_jpg_filename)
                if convert_bmp_to_jpg(white_path, dest_path):
                    print(f"  匹配成功：{filename} -> {white_jpg_filename} (BMP转JPG)")
                    matched += 1
                else:
                    print(f"  转换失败：{filename} -> {white_jpg_filename} (BMP转JPG失败)")
                    failed += 1
            else:
                print(f"  匹配失败：{filename} -> {white_bmp_filename} (白光图片未找到)")
                failed += 1
    
    print(f"  处理完成：总计{total_blue}张蓝光图片，成功匹配{matched}张，失败{failed}张")

def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent
    dataset_root = project_root / "datasets" / "Dual-Modal-1504-500-0-mac"
    white_source_dir = project_root / "data" / "rawdata_cropped_white" / "class1"
    
    print("蓝光白光图片匹配脚本开始运行...")
    print(f"数据集目录：{dataset_root}")
    print(f"白光图片源目录：{white_source_dir}")
    
    # 检查目录是否存在
    if not dataset_root.exists():
        print(f"错误：数据集目录不存在 {dataset_root}")
        return
    
    if not white_source_dir.exists():
        print(f"错误：白光图片源目录不存在 {white_source_dir}")
        return
    
    # 处理各个数据集子目录
    subdirs = ['train', 'valid', 'test']
    for subdir in subdirs:
        subdir_path = dataset_root / subdir
        if subdir_path.exists():
            print(f"\n处理 {subdir} 目录...")
            process_dataset_directory(str(subdir_path), str(white_source_dir))
        else:
            print(f"跳过 {subdir}：目录不存在")
    
    print("\n脚本执行完成！")

if __name__ == "__main__":
    main() 