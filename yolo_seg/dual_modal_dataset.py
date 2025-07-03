import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
import random


class DualModalDataset(Dataset):
    """
    双模态数据集类，用于加载白光和蓝光图像对和对应的标签
    """
    def __init__(self, 
                 white_img_dir, 
                 blue_img_dir, 
                 label_dir=None, 
                 img_size=1024, 
                 transform=None, 
                 augment=True,
                 pairs_file=None):
        """
        初始化双模态数据集
        Args:
            white_img_dir: 白光图像目录
            blue_img_dir: 蓝光图像目录
            label_dir: 标签目录，若为None则为测试集
            img_size: 图像大小
            transform: 数据转换
            augment: 是否进行数据增强
            pairs_file: 白光和蓝光图像配对文件，若为None则自动配对
        """
        self.white_img_dir = Path(white_img_dir)
        self.blue_img_dir = Path(blue_img_dir)
        self.label_dir = Path(label_dir) if label_dir else None
        self.img_size = img_size
        self.transform = transform
        self.augment = augment
        
        # 获取图像对
        if pairs_file and os.path.exists(pairs_file):
            # 从文件中加载图像对
            self.img_pairs = self._load_img_pairs(pairs_file)
        else:
            # 自动配对图像
            self.img_pairs = self._auto_pair_images()
        
        print(f"加载了 {len(self.img_pairs)} 对白光-蓝光图像")
    
    def __len__(self):
        return len(self.img_pairs)
    
    def __getitem__(self, idx):
        """
        获取一对白光-蓝光图像和对应的标签
        Args:
            idx: 索引
        Returns:
            white_img: 白光图像
            blue_img: 蓝光图像
            labels: 标签 (如果有)
        """
        white_img_path, blue_img_path = self.img_pairs[idx]
        
        # 加载图像
        white_img = cv2.imread(white_img_path)
        white_img = cv2.cvtColor(white_img, cv2.COLOR_BGR2RGB)
        
        blue_img = cv2.imread(blue_img_path)
        blue_img = cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB)
        
        # 缩放图像到指定大小
        white_img = cv2.resize(white_img, (self.img_size, self.img_size))
        blue_img = cv2.resize(blue_img, (self.img_size, self.img_size))
        
        # 转换为张量
        white_img = torch.from_numpy(white_img.transpose(2, 0, 1)).float() / 255.0
        blue_img = torch.from_numpy(blue_img.transpose(2, 0, 1)).float() / 255.0
        
        # 如果有标签，加载标签
        if self.label_dir:
            # 从白光图像路径推断标签路径
            label_path = self._get_label_path(white_img_path)
            if os.path.exists(label_path):
                labels = self._load_labels(label_path)
                return white_img, blue_img, labels
            else:
                # 如果标签不存在，返回空标签
                return white_img, blue_img, torch.zeros((0, 6))
        
        # 如果是测试集，只返回图像
        return white_img, blue_img
    
    def _auto_pair_images(self):
        """
        自动配对白光和蓝光图像
        策略：
        1. 首先尝试通过文件名匹配
        2. 如果无法匹配，则尝试通过序号或时间戳匹配
        Returns:
            img_pairs: 配对的图像路径列表 [(white_path, blue_path), ...]
        """
        white_imgs = sorted(glob.glob(str(self.white_img_dir / '*.jpg')) + 
                           glob.glob(str(self.white_img_dir / '*.png')))
        blue_imgs = sorted(glob.glob(str(self.blue_img_dir / '*.jpg')) + 
                          glob.glob(str(self.blue_img_dir / '*.png')))
        
        if len(white_imgs) != len(blue_imgs):
            print(f"警告：白光图像 ({len(white_imgs)}) 和蓝光图像 ({len(blue_imgs)}) 数量不匹配")
        
        # 如果文件数量相同，直接按顺序配对
        img_pairs = []
        for i in range(min(len(white_imgs), len(blue_imgs))):
            img_pairs.append((white_imgs[i], blue_imgs[i]))
        
        return img_pairs
    
    def _load_img_pairs(self, pairs_file):
        """
        从文件中加载图像对
        Args:
            pairs_file: 图像对文件，每行格式为 "white_img_path,blue_img_path"
        Returns:
            img_pairs: 配对的图像路径列表
        """
        img_pairs = []
        with open(pairs_file, 'r') as f:
            for line in f:
                if line.strip():
                    white_path, blue_path = line.strip().split(',')
                    img_pairs.append((white_path, blue_path))
        return img_pairs
    
    def _get_label_path(self, img_path):
        """
        根据图像路径获取对应的标签路径
        Args:
            img_path: 图像路径
        Returns:
            label_path: 标签路径
        """
        if not self.label_dir:
            return None
        
        # 从图像路径提取文件名（不含扩展名）
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        # 构建标签路径
        return str(self.label_dir / f"{img_name}.txt")
    
    def _load_labels(self, label_path):
        """
        加载YOLO格式的标签
        Args:
            label_path: 标签路径
        Returns:
            labels: 标签张量，形状为(n, 6), [class, x, y, w, h, segment_id]
        """
        if not os.path.exists(label_path):
            return torch.zeros((0, 6))
        
        try:
            # 加载标签
            labels = np.loadtxt(label_path).reshape(-1, 5)  # class, x, y, w, h
            # 添加一个全0的segment_id列
            labels = np.column_stack((labels, np.zeros(len(labels))))
            return torch.from_numpy(labels).float()
        except Exception as e:
            print(f"加载标签 {label_path} 时出错: {e}")
            return torch.zeros((0, 6))


def create_dataloader(white_img_dir, blue_img_dir, label_dir=None, batch_size=8, img_size=1024, 
                      augment=True, shuffle=True, num_workers=4):
    """
    创建双模态数据加载器
    Args:
        white_img_dir: 白光图像目录
        blue_img_dir: 蓝光图像目录
        label_dir: 标签目录
        batch_size: 批次大小
        img_size: 图像大小
        augment: 是否进行数据增强
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
    Returns:
        dataloader: 数据加载器
    """
    dataset = DualModalDataset(
        white_img_dir=white_img_dir,
        blue_img_dir=blue_img_dir,
        label_dir=label_dir,
        img_size=img_size,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return dataloader


def collate_fn(batch):
    """
    数据批次整理函数
    处理不同样本的标签，使它们可以被批处理
    Args:
        batch: 批次数据
    Returns:
        整理后的批次
    """
    # 如果批次中只包含图像
    if len(batch[0]) == 2:
        white_imgs, blue_imgs = zip(*batch)
        return torch.stack(white_imgs), torch.stack(blue_imgs)
    
    # 如果批次中包含图像和标签
    white_imgs, blue_imgs, labels = zip(*batch)
    
    # 堆叠图像
    white_imgs = torch.stack(white_imgs)
    blue_imgs = torch.stack(blue_imgs)
    
    # 处理标签
    # 为每个样本添加样本索引
    for i, label in enumerate(labels):
        if label.shape[0] > 0:
            # 在标签前添加样本索引
            label = torch.cat([torch.ones(label.shape[0], 1) * i, label], dim=1)
        
    # 过滤掉空标签
    labels = [label for label in labels if label.shape[0] > 0]
    
    # 如果所有标签都为空，返回空张量
    if not labels:
        return white_imgs, blue_imgs, torch.zeros((0, 7))
    
    # 拼接所有标签
    labels = torch.cat(labels, dim=0)
    
    return white_imgs, blue_imgs, labels


if __name__ == "__main__":
    # 测试数据集
    white_dir = "path/to/white/images"
    blue_dir = "path/to/blue/images"
    label_dir = "path/to/labels"
    
    # 创建数据集
    dataset = DualModalDataset(white_dir, blue_dir, label_dir)
    
    # 测试获取一个样本
    white_img, blue_img, label = dataset[0]
    print(f"白光图像形状: {white_img.shape}")
    print(f"蓝光图像形状: {blue_img.shape}")
    print(f"标签形状: {label.shape}")
    
    # 创建数据加载器
    dataloader = create_dataloader(white_dir, blue_dir, label_dir)
    
    # 测试批次加载
    for white_imgs, blue_imgs, labels in dataloader:
        print(f"批次白光图像形状: {white_imgs.shape}")
        print(f"批次蓝光图像形状: {blue_imgs.shape}")
        print(f"批次标签形状: {labels.shape}")
        break 