"""
双模态YOLO模型评估脚本
使用医生标注的JSON数据进行精度评估，计算IoU和高度差异指标
"""

import torch
import sys
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO


def calculate_iou(mask1, mask2):
    """计算两个二值掩码的IoU"""
    m1 = (mask1 > 0)
    m2 = (mask2 > 0)
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return intersection / union if union > 0 else 1.0


def sort_points_by_angle(points):
    """根据点相对质心的极角排序"""
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def find_json_annotation(npy_filename, json_dirs=['./data/rawdata/class1/', './data/rawdata/class2/']):
    """根据npy文件名查找匹配的JSON标注文件"""
    parts = npy_filename.split('_')
    if len(parts) < 4:
        return None
    
    # 提取时间戳和ID前缀
    json_prefix = '_'.join(parts[:3]) + '_'
    
    for class_dir in json_dirs:
        if not os.path.exists(class_dir):
            continue
        for json_filename in os.listdir(class_dir):
            if json_filename.endswith('.json') and json_filename.startswith(json_prefix.replace('_', '_', 2)):
                try:
                    with open(class_dir + json_filename, 'r') as f:
                        return json.load(f)
                except Exception:
                    continue
    return None


def extract_annotation_points(json_data):
    """从JSON数据中提取医生标注的4个白膜层点位"""
    true_points = []
    for id, shape in enumerate(json_data.get("shapes", [])):
        if id >= 2 and id <= 5:  # 白膜层的4个点位
            true_point = shape["points"][0]
            x, y = int(true_point[0]), int(true_point[1])
            # 坐标转换
            x, y = int((x - 800) * 1504/1216), int((y - 250) * 1)
            true_points.append([x, y])
    
    if len(true_points) >= 4:
        return sort_points_by_angle(np.array(true_points, dtype=np.int32))
    return None


def visualize_results(annotated_image, pred_points=None, true_points=None, save_path=None):
    """可视化预测结果和真实标注"""
    # 注意：cv2使用BGR格式，但matplotlib使用RGB格式
    if pred_points is not None:
        # 绿色轮廓线 (BGR格式: B=0, G=255, R=0)
        cv2.polylines(annotated_image, pred_points.reshape((-1, 1, 2)), 
                     isClosed=True, color=(0, 255, 0), thickness=5)
    
    if true_points is not None:
        # 红色点位 (BGR格式: B=0, G=0, R=255)
        for point in true_points:
            cv2.circle(annotated_image, tuple(point), radius=5, color=(0, 0, 255), thickness=-1)
    
    if save_path:
        # 保存时转换为RGB格式
        rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        plt.imsave(save_path, rgb_image)


def evaluate_dual_yolo_model(fusion_name = 'crossattn'):
    """主评估函数"""
    # 配置参数
    project_root = Path(__file__).parent.parent
    model_yaml = project_root / 'dual_yolo' / 'models' / f'yolo11x-dseg-{fusion_name}.yaml'
    model_pt = project_root / 'dual_yolo' / 'runs' / 'segment' / f'dual_modal_train_{fusion_name}' / 'weights' / 'best.pt'
    
    dataset_path = project_root / 'datasets' / 'Dual-Modal-1504-500-1-6ch'
    test_images = dataset_path / 'test' / 'images'
    
    eval_results_dir = project_root / 'dual_yolo' / 'evaluation_results' / f'{fusion_name}'
    eval_results_dir.mkdir(exist_ok=True)
    
    # 加载模型
    print("加载双模态YOLO模型...")
    model = YOLO(model_yaml).load(model_pt)
    
    # 获取测试文件列表（只评估_0后缀的原始图像）
    npy_files = sorted([f for f in os.listdir(test_images) 
                       if f.endswith('_0.npy')])
    
    print(f"评估图像数量: {len(npy_files)}")
    
    # 评估指标
    metrics = {
        'iou_list': [],
        'height_upper_diff': [],
        'height_lower_diff': [],
        'height_upper_diff_percent': [],
        'height_lower_diff_percent': [],
        'detected_count': 0
    }
    
    # 逐个评估图像
    for npy_file in npy_files:
        success = process_single_image(
            npy_file, test_images, model, eval_results_dir, metrics
        )
        if success:
            metrics['detected_count'] += 1
    
    # 打印和保存结果
    print_evaluation_results(metrics, len(npy_files))
    generate_evaluation_chart(metrics, len(npy_files), eval_results_dir, fusion_name)
    
    print(f"\n评估完成！结果保存在 {eval_results_dir}")


def process_single_image(npy_file, test_images_dir, model, results_dir, metrics):
    """处理单个图像的评估"""
    try:
        # 加载数据 (完全使用numpy处理，与测试脚本保持一致)
        dual_tensor = np.load(test_images_dir / npy_file)
        if dual_tensor.shape[-1] == 6:
            dual_tensor = dual_tensor.transpose(2, 0, 1)
        
        # 获取可视化图像 (与测试脚本完全相同的Method 1)
        blue_channels = dual_tensor[:3, :, :]  # 前3个通道是蓝光
        # 尝试BGR通道顺序 (如果显示红色，说明可能需要RGB->BGR转换)
        blue_channels = blue_channels[[2, 1, 0], :, :]  # RGB -> BGR
        blue_image = blue_channels.transpose(1, 2, 0)  # CHW -> HWC
        
        # 检查数据范围并适当缩放 (与测试脚本完全相同)
        if blue_image.max() <= 1.0:
            annotated_image = np.clip(blue_image * 255, 0, 255).astype(np.uint8)
        else:
            annotated_image = np.clip(blue_image, 0, 255).astype(np.uint8)
        
        # 为模型推理准备torch tensor
        model_input = torch.from_numpy(dual_tensor).unsqueeze(0).float()
        
        # 查找JSON标注
        json_data = find_json_annotation(npy_file)
        if not json_data:
            print(f"未找到JSON标注: {npy_file}")
            return False
        
        # 提取真实标注点
        true_points = extract_annotation_points(json_data)
        if true_points is None:
            print(f"标注点不足: {npy_file}")
            return False
        
        # 模型预测
        results = model(model_input, imgsz=1504, device="cuda:0", verbose=False)
        
        # 查找血液区域检测结果
        bloodzone_detections = [i for i, cls_id in enumerate(results[0].boxes.cls.cpu().numpy()) 
                               if cls_id == 1]
        
        if bloodzone_detections:
            # 提取预测掩码和点位
            pred_points = extract_prediction_points(results[0], bloodzone_detections)
            pred_mask = get_prediction_mask(results[0], bloodzone_detections)
            
            # 计算评估指标
            calculate_metrics(true_points, pred_points, pred_mask, metrics)
            
            # 可视化并保存
            base_filename = npy_file.replace('.npy', '')
            save_path = results_dir / f'{base_filename}_evaluation.jpg'
            visualize_results(annotated_image, pred_points, true_points, save_path)
            
            print(f'检测成功: {npy_file}')
            return True
        else:
            # 未检测到的情况
            base_filename = npy_file.replace('.npy', '')
            save_path = results_dir / f'{base_filename}_no_detection.jpg'
            visualize_results(annotated_image, None, true_points, save_path)
            
            print(f"未检测到: {npy_file}")
            return False
            
    except Exception as e:
        print(f"处理失败 {npy_file}: {e}")
        return False


def extract_prediction_points(result, bloodzone_detections):
    """提取预测的点位信息"""
    points_list = []
    for i in bloodzone_detections:
        points = result[i].masks.xyn[0]
        points[:, 0] *= result.orig_shape[1]  # width
        points[:, 1] *= result.orig_shape[0]  # height
        points_list.append(points)
    
    return np.vstack(points_list).astype(np.int32)


def get_prediction_mask(result, bloodzone_detections):
    """获取预测掩码"""
    mask = result[bloodzone_detections[0]].masks.data.cpu().numpy()[0]
    return cv2.resize(mask, (1504, 1504), interpolation=cv2.INTER_NEAREST)


def calculate_metrics(true_points, pred_points, pred_mask, metrics):
    """计算评估指标"""
    # 高度计算
    pred_height_upper = np.mean(np.sort(pred_points[:, 1])[:2])
    pred_height_lower = np.mean(np.sort(pred_points[:, 1])[-2:])
    true_height_upper = np.mean(np.sort(true_points[:, 1])[:1])
    true_height_lower = np.mean(np.sort(true_points[:, 1])[-1:])
    
    # 构建真实掩码并计算IoU
    true_mask = np.zeros((1504, 1504), dtype=np.uint8)
    cv2.fillPoly(true_mask, [true_points], 255)
    iou = calculate_iou(true_mask, pred_mask)
    
    # 记录指标
    metrics['iou_list'].append(iou)
    metrics['height_upper_diff'].append(abs(true_height_upper - pred_height_upper))
    metrics['height_lower_diff'].append(abs(true_height_lower - pred_height_lower))
    metrics['height_upper_diff_percent'].append(abs((true_height_upper - pred_height_upper) / true_height_upper))
    metrics['height_lower_diff_percent'].append(abs((true_height_lower - pred_height_lower) / true_height_lower))


def print_evaluation_results(metrics, total_images):
    """打印评估结果"""
    print('\n=== 双模态YOLO模型评估结果 ===')
    print(f'检测率: {round((metrics["detected_count"]/total_images)*100, 2)}%')
    print(f'平均IoU: {np.mean(metrics["iou_list"]):.4f}')
    print(f'平均上表面差异: {np.mean(metrics["height_upper_diff"]):.2f} 像素')
    print(f'平均下表面差异: {np.mean(metrics["height_lower_diff"]):.2f} 像素')
    print(f'上表面差异百分比: {round(np.mean(metrics["height_upper_diff_percent"])*100, 2)}%')
    print(f'下表面差异百分比: {round(np.mean(metrics["height_lower_diff_percent"])*100, 2)}%')


def generate_evaluation_chart(metrics, total_images, save_dir, fusion_name):
    """生成评估结果图表"""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    x_positions = [0.5, 1.5, 2.5, 3.5]
    bar_width = 0.4
    
    # 检测率和IoU
    ax1.bar(x_positions[0], metrics['detected_count']/total_images, 
           width=bar_width, color='C0', alpha=0.9)
    ax1.bar(x_positions[1], np.mean(metrics['iou_list']), 
           yerr=np.std(metrics['iou_list']), width=bar_width, color='C1', alpha=0.9)
    ax1.set_ylabel('Ratio')
    ax1.set_ylim([0, 1])
    
    # 高度差异
    ax2 = ax1.twinx()
    ax2.bar(x_positions[2], np.mean(metrics['height_upper_diff']), 
           yerr=np.std(metrics['height_upper_diff']), width=bar_width, color='C2', alpha=0.9)
    ax2.bar(x_positions[3], np.mean(metrics['height_lower_diff']), 
           yerr=np.std(metrics['height_lower_diff']), width=bar_width, color='C3', alpha=0.9)
    ax2.set_ylabel('Pixel Difference')
    
    plt.xticks(x_positions, ['Detection Rate', 'IoU', 'Upper Diff', 'Lower Diff'])
    plt.title('Dual-Modal YOLO Evaluation Results', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(save_dir / f'evaluation_chart_{fusion_name}.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    fusion_name = 'crossattn' # 'crossattn', 'id', 'concat_compress', 'weighted_fusion'
    evaluate_dual_yolo_model(fusion_name = fusion_name)