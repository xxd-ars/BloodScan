import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from pathlib import Path
import yaml
import time
from tqdm import tqdm
import cv2

from model import create_dual_yolo

class DualModalityDataset(Dataset):
    """双模态数据集类，用于加载蓝光和白光图像对"""
    def __init__(self, root_dir, blue_dir="blue", white_dir="white", 
                 annotation_file=None, transform=None, img_size=640):
        """
        Args:
            root_dir: 数据根目录
            blue_dir: 蓝光图像子目录
            white_dir: 白光图像子目录
            annotation_file: 标注文件路径
            transform: 图像预处理转换
            img_size: 图像大小调整
        """
        self.root_dir = Path(root_dir)
        self.blue_dir = self.root_dir / blue_dir
        self.white_dir = self.root_dir / white_dir
        self.annotation_file = annotation_file
        self.transform = transform
        self.img_size = img_size
        
        # 加载标注数据
        if self.annotation_file and os.path.exists(self.annotation_file):
            with open(self.annotation_file, 'r') as f:
                self.annotations = yaml.safe_load(f)
        else:
            self.annotations = None
        
        # 获取所有蓝光图像文件
        self.blue_images = sorted([img for img in self.blue_dir.glob("*.jpg") or self.blue_dir.glob("*.png")])
        
        # 确保每个蓝光图像都有对应的白光图像
        self.white_images = []
        for blue_img in self.blue_images:
            white_img = self.white_dir / blue_img.name
            if white_img.exists():
                self.white_images.append(white_img)
            else:
                raise FileNotFoundError(f"在{self.white_dir}中未找到对应的白光图像: {blue_img.name}")
                
    def __len__(self):
        return len(self.blue_images)
    
    def __getitem__(self, idx):
        # 读取蓝光和白光图像
        blue_img_path = self.blue_images[idx]
        white_img_path = self.white_images[idx]
        
        blue_img = Image.open(blue_img_path).convert("RGB")
        white_img = Image.open(white_img_path).convert("RGB")
        
        # 应用变换
        if self.transform:
            blue_img = self.transform(blue_img)
            white_img = self.transform(white_img)
        
        # 获取标注
        if self.annotations:
            img_id = blue_img_path.stem
            if img_id in self.annotations:
                labels = self.annotations[img_id]
                # 处理标注数据，转换为模型需要的格式
                # 这里需要根据实际标注格式进行调整
            else:
                labels = []
        else:
            labels = []
            
        return {
            'blue_img': blue_img,
            'white_img': white_img,
            'labels': labels,
            'img_id': blue_img_path.stem
        }


def train(args):
    """训练函数
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_dual_yolo(
        fusion_type=args.fusion_type,
        channels=args.channels,
        dim=args.dim, 
        num_heads=args.heads
    )
    model = model.to(device)
    print(f"使用融合类型: {args.fusion_type}")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    train_dataset = DualModalityDataset(
        root_dir=args.data_dir,
        blue_dir=args.blue_dir,
        white_dir=args.white_dir,
        annotation_file=args.annotation_file,
        transform=transform,
        img_size=args.img_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练循环
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            blue_imgs = batch['blue_img'].to(device)
            white_imgs = batch['white_img'].to(device)
            labels = batch['labels']  # 根据标注格式可能需要特殊处理
            
            # 前向传播
            outputs = model(blue_imgs, white_imgs)
            
            # 计算损失（需要根据YOLO11的损失函数进行调整）
            loss = outputs[0]  # YOLO输出的第一个元素通常是总损失
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 更新进度条
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': epoch_loss / (progress_bar.n + 1)})
        
        # 更新学习率
        scheduler.step()
        
        # 打印统计信息
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")
        
        # 保存模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, f"{args.output_dir}/dual_yolo_{args.fusion_type}_best.pt")
            print(f"保存最佳模型, Loss: {best_loss:.4f}")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"{args.output_dir}/dual_yolo_{args.fusion_type}_epoch_{epoch+1}.pt")
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, f"{args.output_dir}/dual_yolo_{args.fusion_type}_final.pt")
    
    print(f"训练完成，总用时: {(time.time() - start_time) / 60:.2f} 分钟")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DualYOLO模型训练")
    parser.add_argument('--data_dir', type=str, required=True, help='数据根目录')
    parser.add_argument('--blue_dir', type=str, default='blue', help='蓝光图像子目录')
    parser.add_argument('--white_dir', type=str, default='white', help='白光图像子目录')
    parser.add_argument('--annotation_file', type=str, help='标注文件路径')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--fusion_type', type=str, default='ctr', choices=['add', 'cat', 'ctr'], 
                        help='特征融合类型: add, cat, 或 ctr')
    parser.add_argument('--batch_size', type=int, default=16, help='批量大小')
    parser.add_argument('--img_size', type=int, default=640, help='图像大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数')
    parser.add_argument('--save_interval', type=int, default=10, help='模型保存间隔（轮数）')
    parser.add_argument('--channels', type=int, default=256, help='特征通道数')
    parser.add_argument('--dim', type=int, default=128, help='XFormerFusion内部维度')
    parser.add_argument('--heads', type=int, default=4, help='XFormerFusion注意力头数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 开始训练
    train(args)
