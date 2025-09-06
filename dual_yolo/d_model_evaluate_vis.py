#!/usr/bin/env python3
"""
双模态YOLO模型评估结果可视化脚本
从保存的metrics JSON文件生成可视化图表
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_metrics_from_file(metrics_file):
    """从JSON文件加载metrics数据"""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def generate_evaluation_chart_from_file(metrics_file, output_path=None):
    """从保存的metrics文件生成评估图表"""
    # 加载数据
    metrics = load_metrics_from_file(metrics_file)
    fusion_name = Path(metrics_file).stem.replace('metrics_', '')
    
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 提取数据 (包括标准差)
    original_data = []
    augmented_data = []
    original_stds = []
    augmented_stds = []
    
    for key in ['original', 'augmented']:
        group_metrics = metrics[key]
        detection_rate = group_metrics['detection_rate']
        iou_mean = group_metrics['iou_mean']
        iou_std = group_metrics['iou_std']
        upper_diff = group_metrics['upper_diff_mean']
        upper_diff_std = group_metrics['upper_diff_std']
        lower_diff = group_metrics['lower_diff_mean']
        lower_diff_std = group_metrics['lower_diff_std']
        
        if key == 'original':
            original_data = [detection_rate, iou_mean, upper_diff, lower_diff]
            original_stds = [0, iou_std, upper_diff_std, lower_diff_std]  # 检测率无标准差
        else:
            augmented_data = [detection_rate, iou_mean, upper_diff, lower_diff]
            augmented_stds = [0, iou_std, upper_diff_std, lower_diff_std]
    
    # 设置x轴位置和宽度
    metrics_labels = ['Detection Rate', 'IoU', 'Upper Surface', 'Lower Surface']
    x_pos = np.arange(len(metrics_labels))
    width = 0.35
    
    # 定义颜色 - 4种不同的基础颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红
    
    # 设置左侧y轴 (比例类指标)
    ax1.set_ylabel('Ratio', fontsize=12)
    ax1.set_ylim([0, 1.01])
    ax1.tick_params(axis='y', labelcolor='black')
    
    # 创建右侧y轴 (像素差异指标)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Pixel Difference', fontsize=12)
    ax2.set_ylim([0, 12.01])  # 设置右侧y轴范围为0-6
    ax2.tick_params(axis='y', labelcolor='black')
    
    # 绘制所有4个指标的柱状图 (包含标准差)
    for i, (metric_name, color) in enumerate(zip(metrics_labels, colors)):
        x_position = x_pos[i]
        
        if i < 2:  # Detection Rate 和 IoU 使用左侧y轴
            # Original数据：不透明
            ax1.bar(x_position - width/2, original_data[i], width, 
                   yerr=original_stds[i] if original_stds[i] > 0 else None,
                   color=color, alpha=0.9, label='Original' if i == 0 else "",
                   capsize=3)
            # Augmented数据：透明
            ax1.bar(x_position + width/2, augmented_data[i], width, 
                   yerr=augmented_stds[i] if augmented_stds[i] > 0 else None,
                   color=color, alpha=0.6, label='Augmented' if i == 0 else "",
                   capsize=3)
        else:  # Upper Surface 和 Lower Surface 使用右侧y轴
            # Original数据：不透明
            ax2.bar(x_position - width/2, original_data[i], width, 
                   yerr=original_stds[i] if original_stds[i] > 0 else None,
                   color=color, alpha=0.9, capsize=3)
            # Augmented数据：透明
            ax2.bar(x_position + width/2, augmented_data[i], width, 
                   yerr=augmented_stds[i] if augmented_stds[i] > 0 else None,
                   color=color, alpha=0.6, capsize=3)
    
    # 设置x轴
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(metrics_labels, fontsize=11)
    ax1.set_xlabel('Metrics', fontsize=12)
    
    # 添加网格
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 添加图例
    ax1.legend(loc='upper left', fontsize=10)
    
    # 设置标题
    plt.title(f'Dual-Modal YOLO Evaluation Results - {fusion_name}', fontsize=14, pad=20)
    
    # 调整布局并保存
    plt.tight_layout()
    
    if output_path is None:
        output_path = Path(metrics_file).parent / f'evaluation_chart_{fusion_name}.png'
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"可视化图表已保存到: {output_path}")
    return output_path


def print_metrics_summary(metrics_file):
    """打印metrics摘要信息"""
    metrics = load_metrics_from_file(metrics_file)
    fusion_name = Path(metrics_file).stem.replace('metrics_', '')
    
    print(f"\n=== {fusion_name} 模型评估结果 ===")
    
    for group_key in ['original', 'augmented']:
        group_metrics = metrics[group_key]
        group_name = "原始数据" if group_key == 'original' else "增强数据集"
        
        print(f"\n【{group_name}】 (共{group_metrics['total_count']}张图像)")
        print(f"  检测率: {group_metrics['detection_rate']*100:.2f}%")
        
        if group_metrics['iou_mean'] > 0:
            print(f"  平均IoU: {group_metrics['iou_mean']:.4f} ± {group_metrics['iou_std']:.4f}")
            print(f"  上表面差异: {group_metrics['upper_diff_mean']:.2f} ± {group_metrics['upper_diff_std']:.2f} 像素")
            print(f"  下表面差异: {group_metrics['lower_diff_mean']:.2f} ± {group_metrics['lower_diff_std']:.2f} 像素")
        else:
            print("  没有成功检测到任何目标")


def batch_visualize(results_dir, fusion_names=None):
    """批量生成可视化图表"""
    results_path = Path(results_dir)
    
    if fusion_names is None:
        # 自动查找所有metrics文件
        metrics_files = list(results_path.glob('*/metrics_*.json'))
    else:
        # 根据指定的fusion_names查找文件
        metrics_files = []
        for fusion_name in fusion_names:
            pattern = f'{fusion_name}/metrics_{fusion_name}.json'
            metrics_files.extend(results_path.glob(pattern))
    
    if not metrics_files:
        print("未找到任何metrics文件")
        return
    
    print(f"找到 {len(metrics_files)} 个metrics文件")
    
    for metrics_file in metrics_files:
        print(f"\n处理: {metrics_file}")
        try:
            print_metrics_summary(metrics_file)
            generate_evaluation_chart_from_file(metrics_file)
        except Exception as e:
            print(f"处理失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='生成双模态YOLO评估结果可视化')
    parser.add_argument('--metrics_file', '-f', type=str, 
                       help='单个metrics JSON文件路径')
    parser.add_argument('--results_dir', '-d', type=str, 
                       default='./evaluation_results_aug',
                       help='评估结果目录路径')
    parser.add_argument('--fusion_names', '-n', nargs='+', 
                       help='指定要处理的fusion策略名称')
    parser.add_argument('--output', '-o', type=str, 
                       help='输出图片路径 (仅在单文件模式下有效)')
    
    args = parser.parse_args()
    
    if args.metrics_file:
        # 单文件模式
        if not Path(args.metrics_file).exists():
            print(f"文件不存在: {args.metrics_file}")
            return
        
        print_metrics_summary(args.metrics_file)
        generate_evaluation_chart_from_file(args.metrics_file, args.output)
    else:
        # 批量模式
        batch_visualize(args.results_dir, args.fusion_names)


if __name__ == '__main__':
    main()