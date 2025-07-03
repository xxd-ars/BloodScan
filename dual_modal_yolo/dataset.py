import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DualModalDataset(Dataset):
    """
    双模态数据集，同时加载白光和蓝光图像及其标签
    """
    def __init__(self, 
                 white_img_dir, 
                 blue_img_dir, 
                 annotation_dir=None, 
                 img_size=640,
                 transform=None,
                 phase='train',
                 pair_mode='filename',  # 配对模式: 'filename'(同名文件)或'index'(索引对应)
                 img_formats=('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp'),
                 prefix_white='',  # 白光图像文件前缀
                 prefix_blue='',   # 蓝光图像文件前缀
                 suffix_white='',  # 白光图像文件后缀
                 suffix_blue='',   # 蓝光图像文件后缀
                 augment=True):
        super().__init__()
        self.white_img_dir = white_img_dir
        self.blue_img_dir = blue_img_dir
        self.annotation_dir = annotation_dir
        self.img_size = img_size
        self.phase = phase
        self.augment = augment and phase == 'train'
        self.pair_mode = pair_mode
        self.prefix_white = prefix_white
        self.prefix_blue = prefix_blue
        self.suffix_white = suffix_white 
        self.suffix_blue = suffix_blue
        
        # 创建转换器
        if transform:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()
        
        # 获取图像文件列表
        white_files = []
        for img_format in img_formats:
            white_files.extend(glob.glob(os.path.join(white_img_dir, f"**/*.{img_format}"), recursive=True))
        self.white_files = sorted([f for f in white_files if os.path.isfile(f)])
        
        if pair_mode == 'filename':
            # 通过文件名匹配配对
            self.pairs = self._pair_by_filename()
        else:
            # 通过索引匹配配对
            blue_files = []
            for img_format in img_formats:
                blue_files.extend(glob.glob(os.path.join(blue_img_dir, f"**/*.{img_format}"), recursive=True))
            self.blue_files = sorted([f for f in blue_files if os.path.isfile(f)])
            
            # 确保两个目录中的图像数量相同
            assert len(self.white_files) == len(self.blue_files), \
                f"白光({len(self.white_files)})和蓝光({len(self.blue_files)})图像数量不匹配!"
            
            self.pairs = list(zip(self.white_files, self.blue_files))
        
        print(f"找到{len(self.pairs)}对双模态图像")
    
    def _pair_by_filename(self):
        """
        通过文件名匹配白光和蓝光图像对
        """
        pairs = []
        white_name_to_path = {}
        
        # 创建白光图像的文件名到路径映射
        for white_path in self.white_files:
            filename = os.path.basename(white_path)
            # 去除前缀和后缀
            if self.prefix_white and filename.startswith(self.prefix_white):
                filename = filename[len(self.prefix_white):]
            if self.suffix_white and filename.endswith(self.suffix_white + os.path.splitext(filename)[1]):
                ext_len = len(os.path.splitext(filename)[1])
                filename = filename[:-ext_len - len(self.suffix_white)] + filename[-ext_len:]
            
            white_name_to_path[filename] = white_path
        
        # 寻找蓝光图像对应项
        for root, _, files in os.walk(self.blue_img_dir):
            for file in files:
                if file.split('.')[-1].lower() in ('bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp'):
                    # 处理蓝光图像文件名
                    blue_filename = file
                    if self.prefix_blue and blue_filename.startswith(self.prefix_blue):
                        blue_filename = blue_filename[len(self.prefix_blue):]
                    if self.suffix_blue and blue_filename.endswith(self.suffix_blue + os.path.splitext(blue_filename)[1]):
                        ext_len = len(os.path.splitext(blue_filename)[1])
                        blue_filename = blue_filename[:-ext_len - len(self.suffix_blue)] + blue_filename[-ext_len:]
                    
                    # 尝试匹配白光图像
                    if blue_filename in white_name_to_path:
                        white_path = white_name_to_path[blue_filename]
                        blue_path = os.path.join(root, file)
                        pairs.append((white_path, blue_path))
        
        return pairs
    
    def _get_default_transforms(self):
        """
        获取默认的数据转换
        """
        if self.phase == 'train' and self.augment:
            # 训练阶段的数据增强
            transform = A.Compose([
                A.RandomResizedCrop(height=self.img_size, width=self.img_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            # 验证/测试阶段的数据处理
            transform = A.Compose([
                A.Resize(height=self.img_size, width=self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        
        return transform
    
    def _load_image(self, image_path):
        """
        加载图像
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def _load_annotation(self, img_path):
        """
        加载标注文件
        """
        if self.annotation_dir is None:
            return [], []
        
        # 从图像路径生成标注文件路径
        img_stem = Path(img_path).stem
        label_path = os.path.join(self.annotation_dir, f"{img_stem}.txt")
        
        boxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    data = line.strip().split()
                    if len(data) >= 5:  # 类别 + bbox坐标
                        class_id = int(data[0])
                        x_center, y_center, width, height = map(float, data[1:5])
                        
                        # YOLO格式 [x_center, y_center, width, height]
                        boxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        return boxes, class_labels
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        # 获取图像路径对
        white_path, blue_path = self.pairs[idx]
        
        # 加载图像
        white_img = self._load_image(white_path)
        blue_img = self._load_image(blue_path)
        
        # 加载标注信息 (使用白光图像的标注)
        boxes, class_labels = self._load_annotation(white_path)
        
        # 应用数据转换
        if boxes and self.transform:
            # 对两个图像应用相同的几何变换
            transformed = self.transform(
                image=white_img,
                image1=blue_img,
                bboxes=boxes,
                class_labels=class_labels
            )
            
            white_img = transformed['image']
            blue_img = transformed['image1']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        elif self.transform:
            # 没有边界框，只转换图像
            transformed_white = self.transform(image=white_img)
            transformed_blue = self.transform(image=blue_img)
            
            white_img = transformed_white['image']
            blue_img = transformed_blue['image']
            
        # 转换为Tensor
        if not isinstance(white_img, torch.Tensor):
            white_img = torch.from_numpy(white_img).permute(2, 0, 1).float() / 255.0
            blue_img = torch.from_numpy(blue_img).permute(2, 0, 1).float() / 255.0
        
        # 构建标签tensor
        if boxes:
            labels = torch.zeros((len(boxes), 5))
            for i, (box, cls) in enumerate(zip(boxes, class_labels)):
                labels[i, 0] = cls
                labels[i, 1:] = torch.tensor(box)
        else:
            labels = torch.zeros((0, 5))
        
        return {
            'white_img': white_img,
            'blue_img': blue_img,
            'labels': labels,
            'white_path': white_path,
            'blue_path': blue_path
        }


def create_dataloader(white_dir, blue_dir, annotation_dir=None, batch_size=16, img_size=640, 
                     workers=4, phase='train', augment=True, prefix_white='', prefix_blue='',
                     suffix_white='', suffix_blue='', pair_mode='filename'):
    """
    创建双模态数据加载器
    
    Args:
        white_dir: 白光图像目录
        blue_dir: 蓝光图像目录
        annotation_dir: 标注目录
        batch_size: 批量大小
        img_size: 图像大小
        workers: 数据加载工作线程数
        phase: 阶段 ('train', 'val', 'test')
        augment: 是否应用数据增强
        prefix_white/blue: 白光/蓝光图像前缀
        suffix_white/blue: 白光/蓝光图像后缀
        pair_mode: 配对模式 ('filename'或'index')
    
    Returns:
        数据加载器
    """
    dataset = DualModalDataset(
        white_img_dir=white_dir,
        blue_img_dir=blue_dir,
        annotation_dir=annotation_dir,
        img_size=img_size,
        phase=phase,
        augment=augment,
        prefix_white=prefix_white,
        prefix_blue=prefix_blue,
        suffix_white=suffix_white,
        suffix_blue=suffix_blue,
        pair_mode=pair_mode
    )
    
    batch_size = min(batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(phase == 'train'),
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return loader


def collate_fn(batch):
    """
    自定义batch收集函数，处理不同尺寸的标签
    """
    white_imgs = []
    blue_imgs = []
    labels = []
    white_paths = []
    blue_paths = []
    
    for item in batch:
        white_imgs.append(item['white_img'])
        blue_imgs.append(item['blue_img'])
        labels.append(item['labels'])
        white_paths.append(item['white_path'])
        blue_paths.append(item['blue_path'])
    
    # 堆叠图像
    white_imgs = torch.stack(white_imgs)
    blue_imgs = torch.stack(blue_imgs)
    
    return {
        'white_img': white_imgs,
        'blue_img': blue_imgs,
        'labels': labels,
        'white_path': white_paths,
        'blue_path': blue_paths
    }


if __name__ == "__main__":
    # 测试代码
    import matplotlib.pyplot as plt
    
    # 测试数据集
    dataset = DualModalDataset(
        white_img_dir='data/blood_samples/white',
        blue_img_dir='data/blood_samples/blue',
        annotation_dir='data/blood_samples/labels',
        img_size=640,
        augment=True
    )
    
    # 获取一个样本
    sample = dataset[0]
    white_img = sample['white_img']
    blue_img = sample['blue_img']
    labels = sample['labels']
    
    print(f"白光图像形状: {white_img.shape}")
    print(f"蓝光图像形状: {blue_img.shape}")
    print(f"标签形状: {labels.shape}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # 转换回NumPy并显示
    white_img_np = white_img.permute(1, 2, 0).numpy()
    blue_img_np = blue_img.permute(1, 2, 0).numpy()
    
    axes[0].imshow(white_img_np)
    axes[0].set_title('White Light')
    axes[0].axis('off')
    
    axes[1].imshow(blue_img_np)
    axes[1].set_title('Blue Light')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('dual_modal_sample.png')
    plt.close()
    
    print('样本可视化已保存为 dual_modal_sample.png') 