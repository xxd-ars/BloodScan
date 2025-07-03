import os
import argparse
import yaml
import torch
import torch.optim as optim
from ultralytics import YOLO
from pathlib import Path

from dual_modal_yolo import DualModalYOLO
from dual_modal_dataset import create_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description="训练双模态YOLO11分割模型")
    parser.add_argument('--white-dir', type=str, default='datasets/white', help='白光图像目录')
    parser.add_argument('--blue-dir', type=str, default='datasets/blue', help='蓝光图像目录')
    parser.add_argument('--data', type=str, default='datasets/data.yaml', help='数据集配置文件')
    parser.add_argument('--img-size', type=int, default=1024, help='训练图像大小')
    parser.add_argument('--batch-size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮次')
    parser.add_argument('--device', type=str, default='cuda:0', help='训练设备')
    parser.add_argument('--workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--fusion-type', type=str, default='transformer', 
                        choices=['add', 'concat', 'transformer'], help='特征融合类型')
    parser.add_argument('--white-model', type=str, default='yolo11x-seg.pt', help='白光模型权重')
    parser.add_argument('--blue-model', type=str, default='yolo11x-seg.pt', help='蓝光模型权重')
    parser.add_argument('--save-dir', type=str, default='runs/dual_modal', help='保存路径')
    parser.add_argument('--name', type=str, default='exp', help='实验名称')
    return parser.parse_args()


def main(args):
    # 创建保存目录
    save_dir = Path(args.save_dir) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    with open(save_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)
    
    # 读取数据集配置
    with open(args.data, 'r') as f:
        data_dict = yaml.safe_load(f)
    
    # 提取相关路径
    train_white_dir = os.path.join(args.white_dir, 'train/images')
    train_blue_dir = os.path.join(args.blue_dir, 'train/images')
    val_white_dir = os.path.join(args.white_dir, 'val/images')
    val_blue_dir = os.path.join(args.blue_dir, 'val/images')
    
    train_label_dir = os.path.join(args.white_dir, 'train/labels')
    val_label_dir = os.path.join(args.white_dir, 'val/labels')
    
    # 创建训练和验证数据加载器
    train_loader = create_dataloader(
        white_img_dir=train_white_dir,
        blue_img_dir=train_blue_dir,
        label_dir=train_label_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=True,
        shuffle=True,
        num_workers=args.workers
    )
    
    val_loader = create_dataloader(
        white_img_dir=val_white_dir,
        blue_img_dir=val_blue_dir,
        label_dir=val_label_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        augment=False,
        shuffle=False,
        num_workers=args.workers
    )
    
    # 创建双模态模型
    model = DualModalYOLO(
        white_model_path=args.white_model,
        blue_model_path=args.blue_model,
        fusion_type=args.fusion_type
    )
    
    # 移动模型到指定设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 定义优化器
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练
    print(f"开始训练双模态YOLO11分割模型...")
    print(f"训练设备: {device}")
    print(f"训练轮次: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"图像大小: {args.img_size}")
    print(f"融合类型: {args.fusion_type}")
    
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # 训练
        model.train()
        train_loss = 0.0
        
        for batch_i, (white_imgs, blue_imgs, targets) in enumerate(train_loader):
            white_imgs = white_imgs.to(device)
            blue_imgs = blue_imgs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            loss_dict = model(white_imgs, blue_imgs, targets=targets)
            
            # 计算总损失
            loss = sum(loss for loss in loss_dict.values())
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 累积损失
            train_loss += loss.item()
            
            # 打印批次信息
            if batch_i % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_i}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        
        # 验证
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_i, (white_imgs, blue_imgs, targets) in enumerate(val_loader):
                white_imgs = white_imgs.to(device)
                blue_imgs = blue_imgs.to(device)
                targets = targets.to(device)
                
                # 前向传播
                loss_dict = model(white_imgs, blue_imgs, targets=targets)
                
                # 计算总损失
                loss = sum(loss for loss in loss_dict.values())
                
                # 累积损失
                val_loss += loss.item()
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        
        # 更新学习率
        scheduler.step()
        
        # 打印轮次信息
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 保存完整模型
            torch.save(model.state_dict(), save_dir / 'best.pt')
            print(f"保存最佳模型到 {save_dir / 'best.pt'}")
        
        # 每10轮保存一次检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss
            }, save_dir / f'checkpoint_epoch{epoch+1}.pt')
    
    # 保存最终模型
    torch.save(model.state_dict(), save_dir / 'last.pt')
    print(f"保存最终模型到 {save_dir / 'last.pt'}")
    
    print("训练完成!")


if __name__ == "__main__":
    args = parse_args()
    main(args) 