import argparse
import os
import torch
import cv2
import numpy as np
from PIL import Image
import time
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from model import create_dual_yolo

def process_image(blue_img_path, white_img_path, transform, device):
    """处理图像对，转换为适合模型输入的格式
    
    Args:
        blue_img_path: 蓝光图像路径
        white_img_path: 白光图像路径
        transform: 图像转换
        device: 设备
        
    Returns:
        blue_tensor: 转换后的蓝光图像张量
        white_tensor: 转换后的白光图像张量
        original_size: 原始图像尺寸 (w, h)
    """
    # 读取图像
    blue_img = Image.open(blue_img_path).convert("RGB")
    white_img = Image.open(white_img_path).convert("RGB")
    
    # 保存原始尺寸
    original_size = blue_img.size
    
    # 应用变换
    blue_tensor = transform(blue_img).unsqueeze(0).to(device)
    white_tensor = transform(white_img).unsqueeze(0).to(device)
    
    return blue_tensor, white_tensor, original_size

def draw_results(image_path, detections, output_path=None, conf_threshold=0.25, mask_alpha=0.3):
    """在图像上绘制检测结果
    
    Args:
        image_path: 原始图像路径
        detections: 检测结果
        output_path: 输出图像路径，如果为None则显示图像
        conf_threshold: 置信度阈值
        mask_alpha: 分割掩膜透明度
    """
    # 读取原始图像
    original_img = cv2.imread(str(image_path))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # 创建画布
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(original_img)
    
    # 解析并绘制检测结果
    if len(detections) > 0:
        # 对于YOLO检测结果的处理，根据实际输出格式调整
        # 通常包括边界框、类别标签、置信度和分割掩膜
        
        boxes = detections[0].boxes  # 假设已经有边界框信息
        
        # 获取边界框、置信度和类别
        for i in range(len(boxes)):
            box = boxes[i].xyxy[0].cpu().numpy()  # 获取边界框坐标 (x1, y1, x2, y2)
            conf = boxes[i].conf[0].cpu().numpy()  # 获取置信度
            cls = int(boxes[i].cls[0].cpu().numpy())  # 获取类别索引
            
            # 筛选高置信度检测结果
            if conf >= conf_threshold:
                # 绘制边界框
                rect = patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                    linewidth=2, edgecolor='r', facecolor='none'
                )
                ax.add_patch(rect)
                
                # 添加标签
                class_name = f"Class {cls}"  # 替换为实际类别名称
                label = f"{class_name}: {conf:.2f}"
                ax.text(box[0], box[1] - 5, label, color='white', fontsize=10,
                        bbox=dict(facecolor='red', alpha=0.5))
                
                # 如果有分割掩膜，则绘制掩膜
                if hasattr(detections[0], 'masks') and detections[0].masks is not None:
                    mask = detections[0].masks.data[i].cpu().numpy()
                    colored_mask = np.zeros_like(original_img)
                    colored_mask[mask > 0.5] = [255, 0, 0]  # 红色掩膜
                    ax.imshow(colored_mask, alpha=mask_alpha)
    
    # 去除坐标轴
    ax.axis('off')
    
    # 保存或显示图像
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def detect(args):
    """检测函数
    
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
    
    # 加载模型权重
    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型权重: {args.weights}")
    else:
        raise FileNotFoundError(f"未找到模型权重文件: {args.weights}")
    
    # 设置为评估模式
    model.eval()
    model = model.to(device)
    
    # 图像转换
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 蓝光图像目录
    blue_dir = Path(args.blue_dir)
    if not blue_dir.exists() or not blue_dir.is_dir():
        raise ValueError(f"无效的蓝光图像目录: {args.blue_dir}")
    
    # 白光图像目录
    white_dir = Path(args.white_dir)
    if not white_dir.exists() or not white_dir.is_dir():
        raise ValueError(f"无效的白光图像目录: {args.white_dir}")
    
    # 获取所有蓝光图像
    blue_images = sorted([img for img in blue_dir.glob("*.jpg") or blue_dir.glob("*.png")])
    if not blue_images:
        raise ValueError(f"在{args.blue_dir}中未找到图像")
    
    # 执行检测
    total_time = 0
    processed_images = 0
    
    for blue_img_path in tqdm(blue_images, desc="处理图像"):
        # 构建对应的白光图像路径
        white_img_path = white_dir / blue_img_path.name
        if not white_img_path.exists():
            print(f"警告: 在{white_dir}中未找到对应的白光图像: {blue_img_path.name}，跳过")
            continue
        
        # 处理图像
        blue_tensor, white_tensor, original_size = process_image(
            blue_img_path, white_img_path, transform, device
        )
        
        # 推理
        start_time = time.time()
        with torch.no_grad():
            detections = model(blue_tensor, white_tensor)
        inference_time = time.time() - start_time
        
        total_time += inference_time
        processed_images += 1
        
        # 输出结果
        output_path = Path(args.output_dir) / f"{blue_img_path.stem}_result.jpg"
        draw_results(blue_img_path, detections, output_path, args.conf_threshold)
        
        # 输出推理时间
        print(f"图像: {blue_img_path.name}, 推理时间: {inference_time:.4f}秒")
    
    # 计算平均FPS
    if processed_images > 0:
        avg_time = total_time / processed_images
        fps = 1.0 / avg_time
        print(f"平均推理时间: {avg_time:.4f}秒, FPS: {fps:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DualYOLO模型推理")
    parser.add_argument('--weights', type=str, required=True, help='模型权重文件路径')
    parser.add_argument('--blue_dir', type=str, required=True, help='蓝光图像目录')
    parser.add_argument('--white_dir', type=str, required=True, help='白光图像目录')
    parser.add_argument('--output_dir', type=str, default='results', help='输出目录')
    parser.add_argument('--fusion_type', type=str, default='ctr', choices=['add', 'cat', 'ctr'], 
                        help='特征融合类型: add, cat, 或 ctr')
    parser.add_argument('--img_size', type=int, default=640, help='图像大小')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--channels', type=int, default=256, help='特征通道数')
    parser.add_argument('--dim', type=int, default=128, help='XFormerFusion内部维度')
    parser.add_argument('--heads', type=int, default=4, help='XFormerFusion注意力头数')
    
    args = parser.parse_args()
    
    # 执行检测
    detect(args)
