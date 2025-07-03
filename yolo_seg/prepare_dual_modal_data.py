import os
import argparse
import glob
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import yaml
import re
import random


def parse_args():
    parser = argparse.ArgumentParser(description="准备双模态图像数据集")
    parser.add_argument('--white-src', type=str, required=True, help='白光图像源目录')
    parser.add_argument('--blue-src', type=str, required=True, help='蓝光图像源目录')
    parser.add_argument('--labels-src', type=str, required=True, help='标签源目录')
    parser.add_argument('--output-dir', type=str, default='datasets', help='输出目录')
    parser.add_argument('--split', type=float, nargs=3, default=[0.8, 0.1, 0.1], 
                        help='训练、验证、测试集比例')
    parser.add_argument('--img-size', type=int, default=1024, help='调整后的图像大小')
    parser.add_argument('--pair-file', type=str, default=None, help='白光和蓝光图像配对文件')
    return parser.parse_args()


def create_pair_file(white_src, blue_src, output_path):
    """
    创建白光和蓝光图像配对文件
    Args:
        white_src: 白光图像源目录
        blue_src: 蓝光图像源目录
        output_path: 输出配对文件路径
    """
    white_imgs = sorted(glob.glob(os.path.join(white_src, '*.jpg')) + 
                        glob.glob(os.path.join(white_src, '*.png')))
    blue_imgs = sorted(glob.glob(os.path.join(blue_src, '*.jpg')) + 
                       glob.glob(os.path.join(blue_src, '*.png')))
    
    # 根据文件名匹配
    pairs = []
    white_names = [os.path.splitext(os.path.basename(f))[0] for f in white_imgs]
    blue_names = [os.path.splitext(os.path.basename(f))[0] for f in blue_imgs]
    
    # 尝试匹配白光和蓝光图像
    for i, white_name in enumerate(white_names):
        # 尝试提取时间戳或ID
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}_\d+)', white_name)
        if timestamp_match:
            # 使用时间戳匹配
            timestamp = timestamp_match.group(1)
            blue_matches = [j for j, name in enumerate(blue_names) if timestamp in name]
            if blue_matches:
                pairs.append((white_imgs[i], blue_imgs[blue_matches[0]]))
        else:
            # 使用完全相同的文件名匹配
            blue_matches = [j for j, name in enumerate(blue_names) if name == white_name]
            if blue_matches:
                pairs.append((white_imgs[i], blue_imgs[blue_matches[0]]))
    
    # 将配对结果写入文件
    with open(output_path, 'w') as f:
        for white_img, blue_img in pairs:
            f.write(f"{white_img},{blue_img}\n")
    
    print(f"创建了 {len(pairs)} 对图像配对，已保存到 {output_path}")
    return pairs


def resize_and_save_image(src_path, dst_path, img_size):
    """
    调整图像大小并保存
    Args:
        src_path: 源图像路径
        dst_path: 目标图像路径
        img_size: 目标图像大小
    """
    # 读取图像
    img = cv2.imread(src_path)
    if img is None:
        print(f"警告：无法读取图像 {src_path}")
        return False
    
    # 调整大小
    img = cv2.resize(img, (img_size, img_size))
    
    # 保存图像
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    cv2.imwrite(dst_path, img)
    return True


def prepare_dataset(args):
    """
    准备双模态数据集
    Args:
        args: 命令行参数
    """
    # 创建输出目录
    output_dir = Path(args.output_dir)
    white_dir = output_dir / 'white'
    blue_dir = output_dir / 'blue'
    
    for split in ['train', 'val', 'test']:
        os.makedirs(white_dir / split / 'images', exist_ok=True)
        os.makedirs(blue_dir / split / 'images', exist_ok=True)
        if split != 'test':
            os.makedirs(white_dir / split / 'labels', exist_ok=True)
    
    # 如果提供了配对文件，则加载配对
    if args.pair_file and os.path.exists(args.pair_file):
        pairs = []
        with open(args.pair_file, 'r') as f:
            for line in f:
                if line.strip():
                    white_img, blue_img = line.strip().split(',')
                    pairs.append((white_img, blue_img))
    else:
        # 创建配对文件
        pair_file = output_dir / 'image_pairs.txt'
        pairs = create_pair_file(args.white_src, args.blue_src, pair_file)
    
    # 随机打乱并分割数据集
    random.seed(42)
    random.shuffle(pairs)
    
    train_size = int(len(pairs) * args.split[0])
    val_size = int(len(pairs) * args.split[1])
    
    train_pairs = pairs[:train_size]
    val_pairs = pairs[train_size:train_size + val_size]
    test_pairs = pairs[train_size + val_size:]
    
    print(f"训练集: {len(train_pairs)} 对图像")
    print(f"验证集: {len(val_pairs)} 对图像")
    print(f"测试集: {len(test_pairs)} 对图像")
    
    # 处理训练集
    print("处理训练集...")
    for white_img, blue_img in tqdm(train_pairs):
        # 获取文件名（不含扩展名）
        white_name = os.path.splitext(os.path.basename(white_img))[0]
        blue_name = os.path.splitext(os.path.basename(blue_img))[0]
        
        # 复制并调整白光图像
        white_dst = white_dir / 'train' / 'images' / f"{white_name}.jpg"
        resize_and_save_image(white_img, white_dst, args.img_size)
        
        # 复制并调整蓝光图像
        blue_dst = blue_dir / 'train' / 'images' / f"{blue_name}.jpg"
        resize_and_save_image(blue_img, blue_dst, args.img_size)
        
        # 复制标签（如果存在）
        label_src = os.path.join(args.labels_src, f"{white_name}.txt")
        if os.path.exists(label_src):
            label_dst = white_dir / 'train' / 'labels' / f"{white_name}.txt"
            shutil.copy(label_src, label_dst)
    
    # 处理验证集
    print("处理验证集...")
    for white_img, blue_img in tqdm(val_pairs):
        white_name = os.path.splitext(os.path.basename(white_img))[0]
        blue_name = os.path.splitext(os.path.basename(blue_img))[0]
        
        white_dst = white_dir / 'val' / 'images' / f"{white_name}.jpg"
        resize_and_save_image(white_img, white_dst, args.img_size)
        
        blue_dst = blue_dir / 'val' / 'images' / f"{blue_name}.jpg"
        resize_and_save_image(blue_img, blue_dst, args.img_size)
        
        label_src = os.path.join(args.labels_src, f"{white_name}.txt")
        if os.path.exists(label_src):
            label_dst = white_dir / 'val' / 'labels' / f"{white_name}.txt"
            shutil.copy(label_src, label_dst)
    
    # 处理测试集
    print("处理测试集...")
    for white_img, blue_img in tqdm(test_pairs):
        white_name = os.path.splitext(os.path.basename(white_img))[0]
        blue_name = os.path.splitext(os.path.basename(blue_img))[0]
        
        white_dst = white_dir / 'test' / 'images' / f"{white_name}.jpg"
        resize_and_save_image(white_img, white_dst, args.img_size)
        
        blue_dst = blue_dir / 'test' / 'images' / f"{blue_name}.jpg"
        resize_and_save_image(blue_img, blue_dst, args.img_size)
    
    # 创建数据集配置文件
    data_yaml = {
        'train': '../train/images',
        'val': '../val/images',
        'test': '../test/images',
        'nc': 3,
        'names': ['background', 'bloodzone', 'other']
    }
    
    with open(output_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    
    print(f"数据集准备完成，保存在 {output_dir}")


def main():
    args = parse_args()
    prepare_dataset(args)


if __name__ == "__main__":
    main() 