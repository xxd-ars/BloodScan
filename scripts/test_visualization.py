"""
测试6通道数据中蓝光图像的可视化功能
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import os

def test_blue_channel_visualization():
    """测试蓝光通道可视化"""
    
    # 数据路径
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / 'datasets' / 'Dual-Modal-1504-500-1-6ch'
    test_images = dataset_path / 'test' / 'images'
    
    # 获取第一个测试文件
    npy_files = [f for f in os.listdir(test_images) if f.endswith('_0.npy')]
    if not npy_files:
        print("未找到测试文件")
        return
    
    test_file = npy_files[0]
    print(f"测试文件: {test_file}")
    
    # 加载数据
    dual_tensor = np.load(test_images / test_file)
    print(f"原始数据形状: {dual_tensor.shape}")
    print(f"数据类型: {dual_tensor.dtype}")
    print(f"数据范围: [{dual_tensor.min():.3f}, {dual_tensor.max():.3f}]")
    
    # 确保数据格式为 (6, H, W)
    if dual_tensor.shape[-1] == 6:
        dual_tensor = dual_tensor.transpose(2, 0, 1)
    print(f"转换后形状: {dual_tensor.shape}")
    
    # 提取前3个通道（蓝光）和后3个通道（白光）
    blue_channels = dual_tensor[:3, :, :]  # 前3通道 - 蓝光
    white_channels = dual_tensor[3:, :, :] # 后3通道 - 白光
    
    print(f"蓝光通道形状: {blue_channels.shape}")
    print(f"白光通道形状: {white_channels.shape}")
    print(f"蓝光通道范围: [{blue_channels.min():.3f}, {blue_channels.max():.3f}]")
    print(f"白光通道范围: [{white_channels.min():.3f}, {white_channels.max():.3f}]")
    
    # 创建可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 方法1: 直接显示蓝光通道 (HWC格式)
    blue_image_method1 = blue_channels.transpose(1, 2, 0)  # CHW -> HWC
    # 检查数据范围并适当缩放
    if blue_image_method1.max() <= 1.0:
        blue_image_method1 = np.clip(blue_image_method1 * 255, 0, 255).astype(np.uint8)
    else:
        blue_image_method1 = np.clip(blue_image_method1, 0, 255).astype(np.uint8)
    axes[0, 0].imshow(blue_image_method1)
    axes[0, 0].set_title('Method 1: Blue Channels (Direct)')
    axes[0, 0].axis('off')
    
    # 方法2: 使用torch转换 (不乘以255)
    torch_tensor = torch.from_numpy(dual_tensor).unsqueeze(0).float()
    blue_image_method2 = torch_tensor[0, :3, :, :].permute(1, 2, 0).numpy()
    # 直接使用归一化数据 [0,1] 范围
    blue_image_method2 = np.clip(blue_image_method2, 0, 1)
    axes[0, 1].imshow(blue_image_method2)
    axes[0, 1].set_title('Method 2: Blue via Torch')
    axes[0, 1].axis('off')
    
    # 方法3: 直接使用原始数据但正确处理
    blue_image_method3 = blue_channels.transpose(1, 2, 0)  # CHW -> HWC
    # 如果数据范围不在[0,1]，进行归一化
    blue_image_method3 = (blue_image_method3 - blue_image_method3.min()) / (blue_image_method3.max() - blue_image_method3.min())
    axes[0, 2].imshow(blue_image_method3)
    axes[0, 2].set_title('Method 3: Blue with Channel Flip')
    axes[0, 2].axis('off')
    
    # 方法4: 白光通道对比 (修复底片反色问题)
    white_image = white_channels.transpose(1, 2, 0)
    # 使用与蓝光相同的处理方法
    if white_image.max() <= 1.0:
        white_image = np.clip(white_image * 255, 0, 255).astype(np.uint8)
    else:
        # 如果数据范围异常，使用归一化
        white_image = (white_image - white_image.min()) / (white_image.max() - white_image.min())
        white_image = (white_image * 255).astype(np.uint8)
    axes[0, 3].imshow(white_image)
    axes[0, 3].set_title('White Channels (Comparison)')
    axes[0, 3].axis('off')
    
    # 显示各个单独通道
    channel_names = ['Blue R', 'Blue G', 'Blue B', 'White R']
    for i in range(4):
        if i < 3:
            channel_data = blue_channels[i, :, :]
        else:
            channel_data = white_channels[0, :, :]
        
        axes[1, i].imshow(channel_data, cmap='gray')
        axes[1, i].set_title(f'{channel_names[i]} Channel')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # 保存测试结果
    save_path = project_root / 'dual_yolo' / 'test_visualization_result.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n测试结果已保存到: {save_path}")
    
    # 分析哪种方法正确
    print("\n=== 分析结果 ===")
    print("如果蓝光图像应该显示为蓝色调:")
    print("- Method 1: 应该显示原始蓝光图像")
    print("- Method 2: 通过torch处理的结果")  
    print("- Method 3: BGR->RGB转换后的结果")
    print("- 对比白光通道来验证差异")
    
    return blue_image_method1, blue_image_method2, blue_image_method3

if __name__ == '__main__':
    test_blue_channel_visualization()