import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# import torchviz
from torch.autograd import Variable
import argparse
import os
import numpy as np
from pathlib import Path
import matplotlib
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from model import create_dual_yolo, AddFusion, CatFusion, XFormerFusion


def test_fusion_modules(channels=256, dim=128, heads=4):
    """测试三种融合模块的输出维度一致性
    
    Args:
        channels: 特征通道数
        dim: XFormerFusion内部维度
        heads: XFormerFusion注意力头数
    """
    print("测试融合模块输出维度一致性...")
    
    # 创建融合模块
    add_fusion = AddFusion(channels)
    cat_fusion = CatFusion(channels)
    ctr_fusion = XFormerFusion(channels, dim, heads)
    
    # 准备测试输入
    batch_size = 2
    height, width = 64, 64
    f_b = torch.rand(batch_size, channels, height, width)
    f_w = torch.rand(batch_size, channels, height, width)
    
    # 执行前向传播
    out_add = add_fusion(f_b, f_w)
    out_cat = cat_fusion(f_b, f_w)
    out_ctr = ctr_fusion(f_b, f_w)
    
    # 检查输出维度
    print(f"输入维度: {f_b.shape}")
    print(f"AddFusion 输出维度: {out_add.shape}")
    print(f"CatFusion 输出维度: {out_cat.shape}")
    print(f"XFormerFusion 输出维度: {out_ctr.shape}")
    
    # 验证维度一致性
    assert out_add.shape == out_cat.shape == out_ctr.shape == f_b.shape, "融合模块输出维度不一致"
    print("通过: 所有融合模块输出维度一致")


def visualize_model_architecture(fusion_type='ctr', output_dir='outputs', img_size=640, channels=256, dim=128, heads=4):
    """生成模型架构示意图
    
    Args:
        fusion_type: 融合类型, 'add', 'cat', 或 'ctr'
        output_dir: 输出目录
        img_size: 输入图像大小
        channels: 特征通道数
        dim: XFormerFusion内部维度
        heads: XFormerFusion注意力头数
    """
    print(f"生成模型架构示意图 (fusion_type={fusion_type})...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建模型
    model = create_dual_yolo(fusion_type, channels, dim, heads)
    
    # 准备输入数据
    blue_img = torch.rand(1, 3, img_size, img_size)
    white_img = torch.rand(1, 3, img_size, img_size)
    
    # 生成模型架构图
    dot = make_dot(model, params=dict(list(model.named_parameters())))
    dot.render(os.path.join(output_dir, f'dual_yolo_{fusion_type}_architecture'), format='png')
    
    print(f"模型架构图已保存到 {output_dir}/dual_yolo_{fusion_type}_architecture.png")


def make_dot(model, inputs=None, params=None, show_attrs=False, show_saved=False):
    """生成模型架构图的辅助函数
    
    Args:
        model: PyTorch模型
        inputs: 模型输入
        params: 模型参数字典
        show_attrs: 是否显示属性
        show_saved: 是否显示保存的变量
        
    Returns:
        graphviz.Digraph对象
    """
    try:
        from graphviz import Digraph
    except ImportError:
        raise ImportError("需要安装graphviz包: pip install graphviz")
    
    if params is None:
        params = dict(model.named_parameters())
    
    # 创建有向图
    node_attr = dict(style='filled', shape='box', align='left', fontsize='12', ranksep='0.1', height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    
    # 添加模型结构
    def add_nodes(model, prefix=''):
        for name, module in model.named_children():
            node_name = prefix + name
            dot.node(node_name, str(module))
            add_nodes(module, node_name + '.')
    
    add_nodes(model)
    
    # 添加连接
    def add_edges(model, prefix=''):
        for name, module in model.named_children():
            node_name = prefix + name
            if isinstance(module, nn.Sequential):
                # 连接Sequential模块内部节点
                prev_node = None
                for i, submodule in enumerate(module):
                    subnode_name = f"{node_name}.{i}"
                    if prev_node is not None:
                        dot.edge(prev_node, subnode_name)
                    prev_node = subnode_name
            # 对于有明确层级关系的模块，添加相应连接
            if hasattr(module, 'children'):
                add_edges(module, node_name + '.')
    
    add_edges(model)
    
    return dot


def visualize_fusion_modules(channels=256, dim=128, heads=4, output_dir='outputs'):
    """生成融合模块的可视化对比图
    
    Args:
        channels: 特征通道数
        dim: XFormerFusion内部维度
        heads: XFormerFusion注意力头数
        output_dir: 输出目录
    """
    print("生成融合模块可视化对比图...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建融合模块
    add_fusion = AddFusion(channels)
    cat_fusion = CatFusion(channels)
    ctr_fusion = XFormerFusion(channels, dim, heads)
    
    # 生成融合模块架构图
    dot_add = make_dot(add_fusion)
    dot_add.render(os.path.join(output_dir, 'add_fusion_architecture'), format='png')
    
    dot_cat = make_dot(cat_fusion)
    dot_cat.render(os.path.join(output_dir, 'cat_fusion_architecture'), format='png')
    
    dot_ctr = make_dot(ctr_fusion)
    dot_ctr.render(os.path.join(output_dir, 'ctr_fusion_architecture'), format='png')
    
    print(f"融合模块架构图已保存到 {output_dir}/")
    
    # 创建融合模块对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 获取架构图文件
    add_img = plt.imread(os.path.join(output_dir, 'add_fusion_architecture.png'))
    cat_img = plt.imread(os.path.join(output_dir, 'cat_fusion_architecture.png'))
    ctr_img = plt.imread(os.path.join(output_dir, 'ctr_fusion_architecture.png'))
    
    # 显示架构图
    axes[0].imshow(add_img)
    axes[0].set_title('AddFusion')
    axes[0].axis('off')
    
    axes[1].imshow(cat_img)
    axes[1].set_title('CatFusion')
    axes[1].axis('off')
    
    axes[2].imshow(ctr_img)
    axes[2].set_title('XFormerFusion')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fusion_modules_comparison.png'))
    print(f"融合模块对比图已保存到 {output_dir}/fusion_modules_comparison.png")


def curved_arrow(ax, start, end, radius=0.1, direction=1, color='black', linewidth=1.5, alpha=1.0, arrow_style='-|>', connection_style='arc3,rad=0.3'):
    """绘制曲线箭头
    
    Args:
        ax: matplotlib轴对象
        start: 起点坐标(x, y)
        end: 终点坐标(x, y)
        radius: 曲线半径
        direction: 曲线方向 (1 顺时针, -1 逆时针)
        color: 线条颜色
        linewidth: 线条宽度
        alpha: 透明度
        arrow_style: 箭头样式
        connection_style: 连接样式
    """
    arrow = FancyArrowPatch(
        start, end, 
        connectionstyle=connection_style, 
        arrowstyle=arrow_style,
        linewidth=linewidth,
        color=color,
        alpha=alpha
    )
    ax.add_patch(arrow)


def draw_dual_yolo_structure(output_dir='outputs'):
    """Draw enhanced DualYOLO model structure diagram in English
    """
    print("Generating DualYOLO model structure diagram...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use a clean, modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set up figure with higher resolution
    fig, ax = plt.subplots(figsize=(14, 10), dpi=150)
    fig.patch.set_facecolor('white')
    
    # Define color palette
    colors = {
        'blue_backbone': '#4285F4',  # Google Blue
        'white_backbone': '#34A853',  # Google Green
        'fusion': '#FBBC05',         # Google Yellow
        'neck': '#4ECDC4',           # Teal
        'head': '#FF6B6B',           # Coral Red
        'text': '#202124',           # Dark Grey
        'arrow': '#5F6368',          # Grey
        'background': '#FFFFFF'      # White
    }
    
    # Set text color
    matplotlib.rcParams['text.color'] = colors['text']
    
    # Define module positions
    x_start_blue = 0.2
    x_start_white = 0.8
    y_inputs = 0.9
    y_backbones = 0.75
    y_features = 0.55
    y_fusion = 0.38
    y_neck = 0.25
    y_head = 0.12
    box_height = 0.1
    box_width_backbone = 0.22
    box_width_fusion = 0.35
    box_width_neck = 0.28
    box_width_head = 0.32

    
    # Draw inputs
    ax.text(x_start_blue, y_inputs, "Blue Light\nImage", fontsize=13, ha='center', va='center', fontweight='bold')
    ax.text(x_start_white, y_inputs, "White Light\nImage", fontsize=13, ha='center', va='center', fontweight='bold')
    
    # Draw backbones
    draw_box(ax, x_start_blue - box_width_backbone/2, y_backbones - box_height/2, box_width_backbone, box_height, 
             "Backbone\n(Blue Light)", alpha=0.9, color=colors['blue_backbone'])
    draw_box(ax, x_start_white - box_width_backbone/2, y_backbones - box_height/2, box_width_backbone, box_height, 
             "Backbone\n(White Light)", alpha=0.9, color=colors['white_backbone'])
    
    # Draw feature extraction arrows
    curved_arrow(ax, (x_start_blue, y_inputs - 0.05), (x_start_blue, y_backbones + box_height/2), 
                 color=colors['arrow'], linewidth=2, connection_style='arc3,rad=0')
    curved_arrow(ax, (x_start_white, y_inputs - 0.05), (x_start_white, y_backbones + box_height/2), 
                 color=colors['arrow'], linewidth=2, connection_style='arc3,rad=0')
    
    # Define P3, P4, P5 positions
    feature_offset_x = 0.07
    blue_features = [
        (x_start_blue - feature_offset_x, "P3"),
        (x_start_blue, "P4"),
        (x_start_blue + feature_offset_x, "P5")
    ]
    
    white_features = [
        (x_start_white - feature_offset_x, "P3"),
        (x_start_white, "P4"),
        (x_start_white + feature_offset_x, "P5")
    ]
    
    # Draw feature points and arrows
    for x, label in blue_features:
        curved_arrow(ax, (x, y_backbones - box_height/2), (x, y_features + 0.03), 
                   color=colors['arrow'], linewidth=1.5, connection_style='arc3,rad=0')
        ax.text(x, y_features, label, fontsize=11, ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor=colors['blue_backbone'], alpha=0.4, 
                        edgecolor=colors['blue_backbone']))
    
    for x, label in white_features:
        curved_arrow(ax, (x, y_backbones - box_height/2), (x, y_features + 0.03), 
                   color=colors['arrow'], linewidth=1.5, connection_style='arc3,rad=0')
        ax.text(x, y_features, label, fontsize=11, ha='center', va='center', 
               bbox=dict(boxstyle="round,pad=0.2", facecolor=colors['white_backbone'], alpha=0.4, 
                        edgecolor=colors['white_backbone']))
    
    # Draw fusion module
    x_fusion_center = 0.5
    draw_box(ax, x_fusion_center - box_width_fusion/2, y_fusion - box_height/2, box_width_fusion, box_height, 
             "Feature Fusion Module\n{Add | Cat | XFormer}", alpha=0.9, color=colors['fusion'])
    
    # Draw fusion arrows
    for x, _ in blue_features:
        curved_arrow(ax, (x, y_features - 0.03), (x_fusion_center - 0.05, y_fusion + box_height/2), # Target slightly left of center for blue
                    color=colors['arrow'], linewidth=1.5, connection_style=f"arc3,rad={0.2 if x < x_fusion_center else -0.2}")
    
    for x, _ in white_features:
        curved_arrow(ax, (x, y_features - 0.03), (x_fusion_center + 0.05, y_fusion + box_height/2), # Target slightly right of center for white
                    color=colors['arrow'], linewidth=1.5, connection_style=f"arc3,rad={0.2 if x < x_fusion_center else -0.2}")
    
    # Draw neck
    draw_box(ax, x_fusion_center - box_width_neck/2, y_neck - box_height/2, box_width_neck, box_height, 
             "Neck (FPN)", alpha=0.9, color=colors['neck'])
    
    # Connect fusion to neck
    curved_arrow(ax, (x_fusion_center, y_fusion - box_height/2), (x_fusion_center, y_neck + box_height/2), 
                color=colors['arrow'], linewidth=2, connection_style='arc3,rad=0')
    
    # Draw detection head
    draw_box(ax, x_fusion_center - box_width_head/2, y_head - box_height/2, box_width_head, box_height, 
             "Segmentation Head", alpha=0.9, color=colors['head'])
    
    # Connect neck to head
    curved_arrow(ax, (x_fusion_center, y_neck - box_height/2), (x_fusion_center, y_head + box_height/2), 
                color=colors['arrow'], linewidth=2, connection_style='arc3,rad=0')
    
    # Set figure properties
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("DualYOLO - Dual-Backbone Feature Fusion Architecture", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Draw a subtle grid
    ax.grid(False)
    
    # Save figure with high quality
    plt.tight_layout(pad=1.5) # Add some padding to tight_layout
    plt.savefig(os.path.join(output_dir, 'dual_yolo_structure.png'), dpi=300, bbox_inches='tight', facecolor='white')
    print(f"DualYOLO model structure diagram saved to {output_dir}/dual_yolo_structure.png")


def draw_box(ax, x, y, width, height, label, alpha=0.5, color='skyblue'):
    """在轴上绘制带标签的框
    
    Args:
        ax: matplotlib轴对象
        x, y: 框的左上角坐标
        width, height: 框的宽度和高度
        label: 文本标签
        alpha: 透明度
        color: 框的颜色
    """
    # Define color palette
    colors = {
        'blue_backbone': '#4285F4',  # Google Blue
        'white_backbone': '#34A853',  # Google Green
        'fusion': '#FBBC05',         # Google Yellow
        'neck': '#4ECDC4',           # Teal
        'head': '#FF6B6B',           # Coral Red
        'text': '#202124',           # Dark Grey
        'arrow': '#5F6368',          # Grey
        'background': '#FFFFFF'      # White
    }
    rect = FancyBboxPatch((x, y), width, height, 
                              boxstyle=matplotlib.patches.BoxStyle("Round", pad=0.15), # Reduced pad for tighter boxes
                              linewidth=1.5, # Slightly thinner lines
                              edgecolor=matplotlib.colors.to_rgba(color, 1.0), # Full opacity edge
                              facecolor=matplotlib.colors.to_rgba(color, alpha), # Use specified alpha for face
                              antialiased=True)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', fontsize=11, fontweight='normal', color=colors['text'])

# Define color palette
    colors = {
        'blue_backbone': '#4285F4',  # Google Blue
        'white_backbone': '#34A853',  # Google Green
        'fusion': '#FBBC05',         # Google Yellow
        'neck': '#4ECDC4',           # Teal
        'head': '#FF6B6B',           # Coral Red
        'text': '#202124',           # Dark Grey
        'arrow': '#5F6368',          # Grey
        'background': '#FFFFFF'      # White
    }
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DualYOLO模型测试和可视化")
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--channels', type=int, default=256, help='特征通道数')
    parser.add_argument('--dim', type=int, default=128, help='XFormerFusion内部维度')
    parser.add_argument('--heads', type=int, default=4, help='XFormerFusion注意力头数')
    parser.add_argument('--img_size', type=int, default=640, help='图像大小')
    parser.add_argument('--test_fusion', action='store_true', help='测试融合模块输出维度一致性')
    parser.add_argument('--visualize_architecture', action='store_true', help='生成模型架构图')
    parser.add_argument('--visualize_fusion', action='store_true', help='生成融合模块对比图')
    parser.add_argument('--draw_structure', action='store_true', help='绘制DualYOLO模型结构示意图')
    parser.add_argument('--all', action='store_true', help='执行所有测试和可视化')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行选定的功能
    if args.test_fusion or args.all:
        test_fusion_modules(args.channels, args.dim, args.heads)
        print()
    
    if args.visualize_architecture or args.all:
        for fusion_type in ['add', 'cat', 'ctr']:
            visualize_model_architecture(fusion_type, args.output_dir, args.img_size, args.channels, args.dim, args.heads)
        print()
    
    if args.visualize_fusion or args.all:
        visualize_fusion_modules(args.channels, args.dim, args.heads, args.output_dir)
        print()
    
    if args.draw_structure or args.all:
        draw_dual_yolo_structure(args.output_dir)
        print()
    
    # 如果没有指定任何操作，则默认执行绘制结构图
    if not (args.test_fusion or args.visualize_architecture or args.visualize_fusion or args.draw_structure or args.all):
        draw_dual_yolo_structure(args.output_dir)
        print("提示: 使用 --all 参数执行所有测试和可视化功能")
