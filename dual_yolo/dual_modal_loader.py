"""
Dual-Modal Dataset Loader for YOLO Training
Handles blue light and white light image pairs for dual-modal YOLO segmentation training.
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob
import random
import yaml
from collections import defaultdict


class DualModalYOLODataset(Dataset):
    """
    双模态YOLO数据集类，用于训练双模态YOLO分割模型
    返回6通道合并张量 (3通道蓝光 + 3通道白光)
    """
    
    def __init__(self, 
                 data_config_path,
                 split='train', 
                 img_size=1504,
                 augment=True,
                 hyp=None):
        """
        初始化双模态YOLO数据集
        
        Args:
            data_config_path: 数据配置文件路径 (data_dual.yaml)
            split: 数据集分割 ('train', 'val', 'test')
            img_size: 图像大小
            augment: 是否进行数据增强
            hyp: 超参数字典
        """
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp or {}
        
        # 加载数据集配置
        with open(data_config_path, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)
        
        # 设置路径
        self.dataset_root = Path(data_config_path).parent
        self._setup_paths()
        
        # 获取图像对和标签
        self.samples = self._load_samples()
        
        print(f"Loaded {len(self.samples)} dual-modal samples for {split}")
        
    def _setup_paths(self):
        """设置数据集路径"""
        if self.split == 'train':
            self.images_b_dir = self.dataset_root / self.data_config['train_images_b']
            self.images_w_dir = self.dataset_root / self.data_config['train_images_w']
            self.labels_dir = self.dataset_root / self.data_config['train_labels']
        elif self.split == 'val':
            self.images_b_dir = self.dataset_root / self.data_config['val_images_b']
            self.images_w_dir = self.dataset_root / self.data_config['val_images_w']
            self.labels_dir = self.dataset_root / self.data_config['val_labels']
        elif self.split == 'test':
            self.images_b_dir = self.dataset_root / self.data_config['test_images_b']
            self.images_w_dir = self.dataset_root / self.data_config['test_images_w']
            self.labels_dir = self.dataset_root / self.data_config['test_labels']
        else:
            raise ValueError(f"Invalid split: {self.split}")
    
    def _load_samples(self):
        """加载样本列表"""
        # 获取蓝光图像列表
        blue_images = sorted(glob.glob(str(self.images_b_dir / '*.jpg')) + 
                           glob.glob(str(self.images_b_dir / '*.png')))
        
        # 获取白光图像列表
        white_images = sorted(glob.glob(str(self.images_w_dir / '*.jpg')) + 
                            glob.glob(str(self.images_w_dir / '*.png')))
        
        # 创建图像配对映射
        blue_map = {self._get_base_name(p): p for p in blue_images}
        white_map = {self._get_base_name(p): p for p in white_images}
        
        # 找到匹配的图像对
        samples = []
        for base_name in blue_map:
            if base_name in white_map:
                blue_path = blue_map[base_name]
                white_path = white_map[base_name]
                label_path = self.labels_dir / f"{base_name}.txt"
                
                samples.append({
                    'blue_path': blue_path,
                    'white_path': white_path,
                    'label_path': str(label_path) if label_path.exists() else None,
                    'base_name': base_name
                })
        
        return samples
    
    def _get_base_name(self, path):
        """从路径中提取基础文件名（不含扩展名）"""
        return Path(path).stem
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        """
        获取一个样本
        
        Returns:
            dict: 包含图像和标签信息的字典
        """
        sample = self.samples[index]
        
        # 加载图像
        blue_img = self._load_image(sample['blue_path'])
        white_img = self._load_image(sample['white_path'])
        
        # 确保图像大小一致
        blue_img = cv2.resize(blue_img, (self.img_size, self.img_size))
        white_img = cv2.resize(white_img, (self.img_size, self.img_size))
        
        # 合并为6通道张量
        dual_img = np.concatenate([blue_img, white_img], axis=2)  # (H, W, 6)
        
        # 转换为张量格式 (C, H, W)
        dual_img = dual_img.transpose(2, 0, 1)  # (6, H, W)
        dual_img = dual_img.astype(np.float32) / 255.0
        
        # 加载标签
        labels = self._load_labels(sample['label_path']) if sample['label_path'] else np.zeros((0, 5))
        
        # 构建样本字典（YOLO格式）
        sample_dict = {
            'img': dual_img,
            'cls': labels[:, 0:1] if len(labels) > 0 else np.zeros((0, 1)),  # 类别
            'bboxes': labels[:, 1:5] if len(labels) > 0 else np.zeros((0, 4)),  # 边界框
            'im_file': sample['blue_path'],  # 主要图像路径
            'ori_shape': (self.img_size, self.img_size),
            'resized_shape': (self.img_size, self.img_size),
            'ratio_pad': (1.0, 1.0, 0.0, 0.0),  # 无填充
            'batch_idx': np.array([index]),  # 批次索引
        }
        
        # 数据增强
        if self.augment and self.split == 'train':
            sample_dict = self._apply_augmentations(sample_dict)
        
        return sample_dict
    
    def _load_image(self, path):
        """加载图像"""
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_labels(self, label_path):
        """
        加载YOLO格式的标签
        
        Args:
            label_path: 标签文件路径
            
        Returns:
            labels: 标签数组，形状为(n, 5) [class, x_center, y_center, width, height]
        """
        if not label_path or not os.path.exists(label_path):
            return np.zeros((0, 5))
        
        try:
            labels = np.loadtxt(label_path).reshape(-1, 5)
            return labels
        except Exception as e:
            print(f"Warning: Failed to load labels from {label_path}: {e}")
            return np.zeros((0, 5))
    
    def _apply_augmentations(self, sample_dict):
        """
        应用数据增强
        
        Args:
            sample_dict: 样本字典
            
        Returns:
            增强后的样本字典
        """
        # 这里可以添加各种数据增强技术
        # 如旋转、翻转、颜色变换等
        # 为了简化，目前只是返回原样本
        
        # 随机水平翻转
        if random.random() < 0.5:
            sample_dict = self._horizontal_flip(sample_dict)
        
        return sample_dict
    
    def _horizontal_flip(self, sample_dict):
        """水平翻转增强"""
        img = sample_dict['img']
        bboxes = sample_dict['bboxes']
        
        # 翻转图像
        img = np.flip(img, axis=2).copy()  # 沿宽度轴翻转
        
        # 翻转边界框
        if len(bboxes) > 0:
            bboxes[:, 1] = 1.0 - bboxes[:, 1]  # x_center = 1 - x_center
        
        sample_dict['img'] = img
        sample_dict['bboxes'] = bboxes
        
        return sample_dict


def create_dual_modal_dataloader(data_config_path, 
                                split='train',
                                batch_size=2,
                                img_size=1504,
                                augment=True,
                                shuffle=True,
                                num_workers=4,
                                hyp=None):
    """
    创建双模态数据加载器
    
    Args:
        data_config_path: 数据配置文件路径
        split: 数据集分割
        batch_size: 批次大小
        img_size: 图像大小
        augment: 是否数据增强
        shuffle: 是否打乱数据
        num_workers: 工作线程数
        hyp: 超参数
        
    Returns:
        DataLoader: 数据加载器
    """
    from torch.utils.data import DataLoader
    
    dataset = DualModalYOLODataset(
        data_config_path=data_config_path,
        split=split,
        img_size=img_size,
        augment=augment,
        hyp=hyp
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=split == 'train'  # 训练时丢弃最后一个不完整批次
    )
    
    return dataloader


def collate_fn(batch):
    """
    自定义批次整理函数，适用于YOLO训练
    
    Args:
        batch: 批次数据列表
        
    Returns:
        整理后的批次字典
    """
    # 收集批次数据
    imgs = []
    cls = []
    bboxes = []
    batch_indices = []
    
    for i, sample in enumerate(batch):
        imgs.append(torch.from_numpy(sample['img']))
        
        if len(sample['cls']) > 0:
            # 添加批次索引
            batch_idx = np.full((len(sample['cls']), 1), i)
            cls.append(np.column_stack([batch_idx, sample['cls']]))
            bboxes.append(sample['bboxes'])
    
    # 转换为张量
    imgs = torch.stack(imgs)
    
    if cls:
        cls = np.vstack(cls)
        bboxes = np.vstack(bboxes)
        # 合并类别和边界框
        targets = np.column_stack([cls, bboxes])  # [batch_idx, class, x, y, w, h]
    else:
        targets = np.zeros((0, 6))
    
    return {
        'img': imgs,
        'cls': torch.from_numpy(targets[:, 1:2]).long() if len(targets) > 0 else torch.zeros((0, 1), dtype=torch.long),
        'bboxes': torch.from_numpy(targets[:, 2:6]).float() if len(targets) > 0 else torch.zeros((0, 4)),
        'batch_idx': torch.from_numpy(targets[:, 0:1]).long() if len(targets) > 0 else torch.zeros((0, 1), dtype=torch.long),
    }


if __name__ == "__main__":
    # 测试数据加载器
    data_config = "/Users/xin99/Documents/BloodScan/datasets/Dual-Modal-1504-500-1/data_dual.yaml"
    
    # 创建训练数据加载器
    train_loader = create_dual_modal_dataloader(
        data_config_path=data_config,
        split='train',
        batch_size=2,
        img_size=1504,
        augment=True,
        shuffle=True
    )
    
    # 测试加载一个批次
    for batch in train_loader:
        print(f"Images shape: {batch['img'].shape}")  # 应该是 [batch_size, 6, 1504, 1504]
        print(f"Classes shape: {batch['cls'].shape}")
        print(f"Bboxes shape: {batch['bboxes'].shape}")
        print(f"Batch indices shape: {batch['batch_idx'].shape}")
        break