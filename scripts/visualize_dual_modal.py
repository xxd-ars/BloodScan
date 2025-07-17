#!/usr/bin/env python3
"""
双模态数据集可视化脚本
功能：随机选择test数据集中的图片，显示蓝光和白光图像及其标注
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['PingFang HK', 'PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'SimSong', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

target_dataset = "test"

def load_image(image_path):
    """加载图像"""
    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        print(f"加载图像失败: {e}")
        return None

def parse_yolo_segmentation(label_file, img_width, img_height):
    """
    解析YOLO分割格式的标签文件
    返回：[(class_id, [(x1,y1), (x2,y2), ...]), ...]
    """
    annotations = []
    
    try:
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                class_id = int(parts[0])
                
                # 提取坐标点（每两个数字一对）
                coords = []
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        x_norm = float(parts[i])
                        y_norm = float(parts[i + 1])
                        # 转换为像素坐标
                        x_pixel = int(x_norm * img_width)
                        y_pixel = int(y_norm * img_height)
                        coords.append((x_pixel, y_pixel))
                
                if coords:
                    annotations.append((class_id, coords))
                    
    except Exception as e:
        print(f"解析标签文件失败: {e}")
        return []
    
    return annotations

def draw_annotations(ax, img, annotations, title):
    """在图像上绘制标注"""
    ax.imshow(img)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # 定义类别颜色
    colors = {
        0: 'red',      # 类别0 - 红色
        1: 'green',    # 类别1 - 绿色  
        2: 'blue'      # 类别2 - 蓝色
    }
    
    class_names = {
        0: '目标物质0',
        1: '目标物质1', 
        2: '目标物质2'
    }
    
    # 绘制每个标注
    for class_id, coords in annotations:
        if len(coords) < 3:  # 至少需要3个点构成多边形
            continue
            
        color = colors.get(class_id, 'yellow')
        
        # 创建多边形
        polygon = patches.Polygon(coords, linewidth=2, edgecolor=color, 
                                facecolor=color, alpha=0.3, label=class_names[class_id])
        ax.add_patch(polygon)
        
        # 在多边形中心添加类别标签
        centroid_x = sum(x for x, y in coords) / len(coords)
        centroid_y = sum(y for x, y in coords) / len(coords)
        ax.text(centroid_x, centroid_y, str(class_id), 
               fontsize=12, fontweight='bold', color='white',
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.8))

def get_corresponding_files(blue_image_path, dataset_root):
    """根据蓝光图片路径获取对应的白光图片和标签文件路径"""
    blue_filename = os.path.basename(blue_image_path)
    
    # 检查是否为增强数据（包含增强后缀）
    augmentation_suffix = ""
    base_blue_filename = blue_filename
    
    # 定义可能的增强策略后缀
    augmentation_patterns = [
        '_original', '_rot_neg10', '_rot_neg5', '_rot_pos5', '_rot_pos10',
        '_bright_low', '_bright_high', '_exp_low', '_exp_high',
        '_blur_light', '_blur_medium', '_blur_heavy',
        '_rot5_bright_low', '_rot5_bright_high', '_rot_neg5_exp_low', '_rot_neg5_exp_high',
        '_bright_low_blur', '_bright_high_blur', '_exp_low_blur', '_exp_high_blur',
        '_rot5_bright_blur', '_rot_neg5_bright_blur', '_rot5_exp_blur', '_bright_exp_blur',
        '_heavy_aug1', '_heavy_aug2'
    ]
    
    # 检查文件名是否包含增强后缀
    for suffix in augmentation_patterns:
        if blue_filename.endswith(f"{suffix}.jpg"):
            augmentation_suffix = suffix
            base_blue_filename = blue_filename.replace(f"{suffix}.jpg", ".jpg")
            break
    
    # 从基础文件名中提取关键信息
    base_name = base_blue_filename.replace('.jpg', '').split('.rf.')[0]
    import re
    pattern = r'(\d{4}-\d{2}-\d{2}_\d{6})_(\d+)_T5_(\d+)_bmp'
    match = re.match(pattern, base_name)
    
    if match:
        timestamp = match.group(1)
        sequence = match.group(2)
        blue_number = int(match.group(3))
        white_number = blue_number - 2
        
        # 生成白光图片文件名（包含增强后缀）
        if augmentation_suffix:
            white_filename = f"{timestamp}_{sequence}_T3_{white_number}{augmentation_suffix}.jpg"
        else:
            white_filename = f"{timestamp}_{sequence}_T3_{white_number}.jpg"
    else:
        print(f"无法解析蓝光图片文件名: {blue_filename}")
        print(f"  基础文件名: {base_blue_filename}")
        print(f"  增强后缀: {augmentation_suffix}")
        return None, None
    
    # 构建文件路径
    white_image_path = os.path.join(dataset_root, target_dataset, 'images_w', white_filename)
    label_filename = blue_filename.replace('.jpg', '.txt')
    label_path = os.path.join(dataset_root, target_dataset, 'labels', label_filename)
    
    return white_image_path, label_path

def get_dataset_statistics(dataset_root):
    """获取数据集统计信息"""
    test_images_dir = os.path.join(dataset_root, target_dataset, 'images')
    
    if not os.path.exists(test_images_dir):
        return None
    
    blue_images = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]
    
    # 统计增强策略
    augmentation_stats = {}
    original_count = 0
    
    for img in blue_images:
        if '_original.jpg' in img:
            original_count += 1
        elif any(suffix in img for suffix in ['_rot_', '_bright_', '_exp_', '_blur_', '_heavy_aug']):
            # 提取增强策略
            for suffix in ['_rot_neg10', '_rot_neg5', '_rot_pos5', '_rot_pos10',
                          '_bright_low', '_bright_high', '_exp_low', '_exp_high',
                          '_blur_light', '_blur_medium', '_blur_heavy',
                          '_rot5_bright_low', '_rot5_bright_high', '_rot_neg5_exp_low', '_rot_neg5_exp_high',
                          '_bright_low_blur', '_bright_high_blur', '_exp_low_blur', '_exp_high_blur',
                          '_rot5_bright_blur', '_rot_neg5_bright_blur', '_rot5_exp_blur', '_bright_exp_blur',
                          '_heavy_aug1', '_heavy_aug2']:
                if suffix in img:
                    augmentation_stats[suffix] = augmentation_stats.get(suffix, 0) + 1
                    break
        else:
            original_count += 1
    
    return {
        'total_images': len(blue_images),
        'original_count': original_count,
        'augmentation_stats': augmentation_stats
    }

def visualize_random_sample(dataset_root):
    """随机选择一个样本进行可视化"""
    test_images_dir = os.path.join(dataset_root, target_dataset, 'images')
    
    # 获取所有蓝光图片
    blue_images = [f for f in os.listdir(test_images_dir) if f.endswith('.jpg')]
    
    if not blue_images:
        print(f"{target_dataset}/images目录中没有找到图片文件")
        return
    
    # 随机选择一张图片
    selected_image = random.choice(blue_images)
    blue_image_path = os.path.join(test_images_dir, selected_image)
    
    print(f"选择的图片: {selected_image}")
    
    # 检查是否为增强数据
    is_augmented = any(suffix in selected_image for suffix in [
        '_original', '_rot_', '_bright_', '_exp_', '_blur_', '_heavy_aug'
    ])
    if is_augmented:
        print(f"  这是增强数据样本")
    
    # 获取对应的白光图片和标签文件
    white_image_path, label_path = get_corresponding_files(blue_image_path, dataset_root)
    
    if white_image_path:
        print(f"对应的白光图片: {os.path.basename(white_image_path)}")
    if label_path:
        print(f"标签文件: {os.path.basename(label_path)}")
    
    if not white_image_path or not label_path:
        print("无法获取对应的白光图片或标签文件路径")
        return
    
    # 检查文件是否存在
    if not os.path.exists(blue_image_path):
        print(f"蓝光图片不存在: {blue_image_path}")
        return
    
    if not os.path.exists(white_image_path):
        print(f"白光图片不存在: {white_image_path}")
        return
        
    if not os.path.exists(label_path):
        print(f"标签文件不存在: {label_path}")
        return
    
    # 加载图像
    blue_img = load_image(blue_image_path)
    white_img = load_image(white_image_path)
    
    if blue_img is None or white_img is None:
        print("图像加载失败")
        return
    
    # 获取图像尺寸
    img_height, img_width = blue_img.shape[:2]
    
    # 解析标注
    annotations = parse_yolo_segmentation(label_path, img_width, img_height)
    
    if not annotations:
        print("没有找到有效的标注数据")
        return
    
    print(f"找到 {len(annotations)} 个标注对象")
    for i, (class_id, coords) in enumerate(annotations):
        print(f"  对象 {i+1}: 类别 {class_id}, {len(coords)} 个顶点")
    
    # 创建可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 绘制蓝光图像和标注
    draw_annotations(ax1, blue_img, annotations, f'蓝光图像 (T5)\n{selected_image}')
    
    # 绘制白光图像和标注
    white_filename = os.path.basename(white_image_path)
    draw_annotations(ax2, white_img, annotations, f'白光图像 (T3)\n{white_filename}')
    
    # 添加图例
    colors_legend = {'red': '目标物质0', 'green': '目标物质1', 'blue': '目标物质2'}
    legend_elements = [patches.Patch(color=color, label=label) for color, label in colors_legend.items()]
    fig.legend(legend_elements, colors_legend.values(), loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def main():
    """主函数"""
    # 设置数据集路径
    project_root = Path(__file__).parent.parent
    dataset_root = project_root / "datasets" / "Dual-Modal-1504-500-0-mac"
    
    print("=" * 50)
    print("双模态数据集可视化工具")
    print("=" * 50)
    print(f"数据集路径: {dataset_root}")
    print("功能说明:")
    print(f"- 随机选择{target_dataset}数据集中的图片")
    print("- 自动识别原始图片和增强图片")
    print("- 左右对比显示蓝光(T5)和白光(T3)图像")
    print("- 用不同颜色显示三种目标物质的分割标注")
    print("- 红色: 目标物质0, 绿色: 目标物质1, 蓝色: 目标物质2")
    print("- 支持所有数据增强策略的可视化")
    print("=" * 50)
    # 2022-03-28_140655_23_T5_2419_bmp.rf.f661f4e144a581c91f644226b96e93f0_rot_neg5_exp_low.txt
    if not dataset_root.exists():
        print(f"错误: 数据集目录不存在 {dataset_root}")
        return
    
    # 显示数据集统计信息
    stats = get_dataset_statistics(str(dataset_root))
    if stats:
        print(f"\n数据集统计:")
        print(f"- 总图片数量: {stats['total_images']}")
        print(f"- 原始图片数量: {stats['original_count']}")
        if stats['augmentation_stats']:
            print(f"- 增强图片数量: {stats['total_images'] - stats['original_count']}")
            print("- 增强策略分布:")
            for strategy, count in sorted(stats['augmentation_stats'].items()):
                print(f"  {strategy}: {count}")
        print("=" * 50)
    
    # 可视化随机样本循环
    sample_count = 0
    try:
        while True:
            sample_count += 1
            print(f"\n第 {sample_count} 个样本:")
            print("-" * 30)
            
            visualize_random_sample(str(dataset_root))
            
            # 询问用户是否继续
            user_input = input("\n按 Enter 查看下一个样本，输入 'q' 退出: ").strip().lower()
            if user_input == 'q':
                print("程序退出")
                break
                
    except KeyboardInterrupt:
        print("\n用户中断程序")
    except Exception as e:
        print(f"程序执行出错: {e}")

if __name__ == "__main__":
    main()