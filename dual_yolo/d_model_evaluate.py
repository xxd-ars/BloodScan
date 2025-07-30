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
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO


# 增强参数映射表
AUGMENTATION_STRATEGIES = {
    '0': {},  # 原始图像
    '1': {'rotation': 5, 'blur': 1.5},
    '2': {'rotation': -5, 'blur': 1.5},
    '3': {'rotation': 5, 'exposure': 0.9},
    '4': {'rotation': -5, 'exposure': 1.1},
    '5': {'rotation': 10, 'brightness': 1.15, 'blur': 1.2},
    '6': {'rotation': -10, 'brightness': 0.85, 'blur': 1.2},
    '7': {'rotation': 5, 'exposure': 0.9, 'blur': 1.2},
    '8': {'rotation': -5, 'exposure': 1.1, 'blur': 1.2},
}


def get_augmentation_params(filename):
    """根据文件名获取增强参数"""
    suffix = filename.split('_')[-1].replace('.npy', '')
    return AUGMENTATION_STRATEGIES.get(suffix, {})


def apply_rotation_to_points(points, angle, img_size=(1504, 1504)):
    """对点位应用旋转变换"""
    if angle == 0:
        return points
    
    center = (img_size[0] // 2, img_size[1] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 转换为齐次坐标
    points_array = np.array(points, dtype=np.float32)
    ones = np.ones((points_array.shape[0], 1), dtype=np.float32)
    homogeneous_points = np.hstack([points_array, ones])
    
    # 应用旋转
    rotated_homogeneous = rotation_matrix.dot(homogeneous_points.T).T
    
    # 裁剪到图像边界
    rotated_points = []
    for pt in rotated_homogeneous:
        x = max(0, min(img_size[0] - 1, pt[0]))
        y = max(0, min(img_size[1] - 1, pt[1]))
        rotated_points.append([x, y])
    
    return np.array(rotated_points, dtype=np.float32)  # 保持浮点精度


def calculate_iou(mask1, mask2):
    """计算IoU"""
    m1, m2 = mask1 > 0, mask2 > 0
    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    return intersection / union if union > 0 else 1.0


def sort_points_by_angle(points):
    """按极角排序点位"""
    center = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles)]


def find_json_annotation(npy_filename, json_dirs=['./data/rawdata/class1/', './data/rawdata/class2/']):
    """查找JSON标注文件"""
    parts = npy_filename.split('_')
    if len(parts) < 4:
        return None
    
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
    """提取医生标注点位"""
    true_points = []
    for id, shape in enumerate(json_data.get("shapes", [])):
        if 2 <= id <= 5:  # 白膜层4个点
            x, y = shape["points"][0]
            x, y = int((x - 800) * 1504/1216), int(y - 250)
            true_points.append([x, y])
    
    return sort_points_by_angle(np.array(true_points, dtype=np.int32)) if len(true_points) >= 4 else None


def visualize_results(annotated_image, pred_points=None, true_points=None, save_path=None):
    """可视化结果"""
    # 确保图像格式正确
    if not isinstance(annotated_image, np.ndarray):
        annotated_image = np.array(annotated_image)
    if annotated_image.dtype != np.uint8:
        annotated_image = annotated_image.astype(np.uint8)
    annotated_image = np.ascontiguousarray(annotated_image)
    
    # 绘制真实点位（红色圆点）
    if true_points is not None:
        for point in true_points:
            cv2.circle(annotated_image, tuple(map(int, point)), 5, (0, 0, 255), -1)

    # 绘制预测点位（绿色十字）
    if pred_points is not None:
        pred_line = np.array(pred_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_image, [pred_line], True, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        for point in pred_points:
            x, y = int(point[0]), int(point[1])
            cv2.line(annotated_image, (x-2, y), (x+2, y), (0, 255, 0), 2)
            cv2.line(annotated_image, (x, y-2), (x, y+2), (0, 255, 0), 2)
    
    # 保存图像
    if save_path:
        rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        plt.imsave(save_path, rgb_image)


def evaluate_dual_yolo_model(fusion_name, debug=False, include_augmented=True):
    """主评估函数"""
    # 配置参数
    project_root = Path(__file__).parent.parent
    if fusion_name:
        model_yaml = project_root / 'dual_yolo' / 'models' / f'yolo11x-dseg-{fusion_name}.yaml'
        if fusion_name == 'crossattn-30epoch':
            model_yaml = project_root / 'dual_yolo' / 'models' / f'yolo11x-dseg-crossattn.yaml'
        model_pt = project_root / 'dual_yolo' / 'runs' / 'segment' / f'dual_modal_train_{fusion_name}' / 'weights' / 'best.pt'
    else:
        model_yaml = project_root / 'dual_yolo' / 'models' / f'yolo11x-seg.yaml'
        model_pt = project_root / 'dual_yolo' / 'weights' / 'yolo11x-seg-blue.pt'
    
    dataset_path = project_root / 'datasets' / 'Dual-Modal-1504-500-1-6ch'
    test_images = dataset_path / 'test' / 'images'
    
    eval_results_dir = project_root / 'dual_yolo' / 'evaluation_results_aug' / f'{fusion_name}'
    os.makedirs(eval_results_dir, exist_ok=True)
    
    # 检查路径是否存在
    if debug:
        print(f"🔍 调试信息：")
        print(f"  模型配置文件: {model_yaml} ({'✅存在' if model_yaml.exists() else '❌不存在'})")
        print(f"  模型权重文件: {model_pt} ({'✅存在' if model_pt.exists() else '❌不存在'})")
        print(f"  测试图像目录: {test_images} ({'✅存在' if test_images.exists() else '❌不存在'})")
    
    # 加载模型
    print("加载双模态YOLO模型...")
    try:
        model = YOLO(model_yaml).load(model_pt)
        if debug:
            print(f"✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 获取测试文件列表
    if include_augmented:
        # 包含所有增强数据 _0-8
        npy_files = sorted([f for f in os.listdir(test_images) 
                           if f.endswith('.npy') and f.split('_')[-1].replace('.npy', '') in AUGMENTATION_STRATEGIES])
        print(f"评估图像数量: {len(npy_files)} (包含增强数据)")
    else:
        # 只评估原始图像 _0
        npy_files = sorted([f for f in os.listdir(test_images) 
                           if f.endswith('_0.npy')])
        print(f"评估图像数量: {len(npy_files)} (仅原始数据)")
    
    # 评估指标 - 分组统计
    metrics = {
        'original': {
            'iou_list': [],
            'height_upper_diff': [],
            'height_lower_diff': [],
            'height_upper_diff_percent': [],
            'height_lower_diff_percent': [],
            'detected_count': 0,
            'total_count': 0
        },
        'augmented': {
            'iou_list': [],
            'height_upper_diff': [],
            'height_lower_diff': [],
            'height_upper_diff_percent': [],
            'height_lower_diff_percent': [],
            'detected_count': 0,
            'total_count': 0
        }
    }
    
    # 逐个评估图像
    for npy_file in tqdm(npy_files, desc="评估进度"):
        # 确定是原始数据还是增强数据
        suffix = npy_file.split('_')[-1].replace('.npy', '')
        is_original = (suffix == '0')
        
        # original: 仅_0文件
        if is_original:
            metrics['original']['total_count'] += 1
            success = process_single_image(npy_file, test_images, model, eval_results_dir, metrics, 'original')
            if success:
                metrics['original']['detected_count'] += 1
        
        # augmented: 所有_0-8文件 (包含原始数据)
        metrics['augmented']['total_count'] += 1
        success_aug = process_single_image(npy_file, test_images, model, eval_results_dir, metrics, 'augmented')
        if success_aug:
            metrics['augmented']['detected_count'] += 1
    
    # 打印和保存结果
    print_evaluation_results(metrics, include_augmented)
    if any(metrics[key]["iou_list"] for key in metrics):  # 只有在有检测结果时才生成图表
        generate_evaluation_chart(metrics, eval_results_dir, fusion_name)
    
    print(f"\n评估完成！结果保存在 {eval_results_dir}")


def process_single_image(npy_file, test_images_dir, model, results_dir, metrics, group_key):
    """处理单个图像评估"""
    try:
        # 加载并预处理数据
        dual_tensor = np.load(test_images_dir / npy_file)
        if dual_tensor.shape[-1] == 6:
            dual_tensor = dual_tensor.transpose(2, 0, 1)
        
        # 准备可视化图像, [0, 1, 2]原始RGB通道顺序, [2, 1, 0]BGR通道顺序
        blue_channels = dual_tensor[:3, :, :][[2, 1, 0], :, :]  # BGR通道顺序
        blue_image = blue_channels.transpose(1, 2, 0)
        annotated_image = np.clip(blue_image * 255 if blue_image.max() <= 1.0 else blue_image, 0, 255).astype(np.uint8)
        
        # 准备模型输入
        model_tensor = dual_tensor / 255.0 if dual_tensor.max() > 1.0 else dual_tensor
        model_input = torch.from_numpy(model_tensor).unsqueeze(0).float()
        
        # 获取增强参数
        augmentation_params = get_augmentation_params(npy_file)
        rotation_angle = augmentation_params.get('rotation', 0)
        
        # 获取标注数据
        json_data = find_json_annotation(npy_file)
        if not json_data:
            return False
        
        # 提取原始标注点
        original_true_points = extract_annotation_points(json_data)
        if original_true_points is None:
            return False
        
        # 为可视化准备旋转后的标注点
        rotated_true_points = apply_rotation_to_points(original_true_points, rotation_angle)
        
        # 模型推理
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        results = model(model_input, imgsz=1504, device=device, verbose=False)
        
        # 检查模型输出
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            if len(results[0].boxes) > 0:
                all_classes = results[0].boxes.cls.cpu().numpy()
                bloodzone_detections = [i for i, cls_id in enumerate(all_classes) if cls_id == 1]
            else:
                bloodzone_detections = []
        else:
            bloodzone_detections = []
        
        base_filename = npy_file.replace('.npy', '')
        
        if bloodzone_detections:
            # 提取预测结果
            pred_points = extract_prediction_points(results[0], bloodzone_detections)
            pred_mask = get_prediction_mask(results[0], bloodzone_detections)
            
            # 计算指标：使用反向旋转方法
            calculate_metrics_with_rotation(
                original_true_points, pred_points, pred_mask, 
                rotation_angle, metrics, group_key
            )
            
            # 保存可视化结果：使用旋转后的标注点
            save_path = results_dir / f'{base_filename}_evaluation.jpg'
            visualize_results(annotated_image, pred_points, rotated_true_points, save_path)
            return True
        else:
            # 未检测到的情况：使用旋转后的标注点进行可视化
            save_path = results_dir / f'{base_filename}_no_detection.jpg'
            visualize_results(annotated_image, None, rotated_true_points, save_path)
            return False
            
    except Exception as e:
        print(f"处理失败 {npy_file}: {e}")
        return False


def extract_prediction_points(result, bloodzone_detections):
    """提取预测点位"""
    points_list = []
    for i in bloodzone_detections:
        points = result[i].masks.xyn[0]
        points[:, 0] *= result.orig_shape[1]
        points[:, 1] *= result.orig_shape[0]
        points_list.append(points)
    return np.vstack(points_list).astype(np.float32)  # 保持float32精度


def get_prediction_mask(result, bloodzone_detections):
    """获取预测掩码"""
    mask = result[bloodzone_detections[0]].masks.data.cpu().numpy()[0]
    return cv2.resize(mask, (1504, 1504), interpolation=cv2.INTER_NEAREST)


def calculate_metrics_with_rotation(original_true_points, pred_points, pred_mask, rotation_angle, metrics, group_key):
    """使用反向旋转方法计算评估指标"""
    # 将预测点反向旋转到原始坐标系
    pred_points_original = apply_rotation_to_points(pred_points, -rotation_angle)
    
    # 在原始坐标系中计算高度差异
    pred_heights = np.sort(pred_points_original[:, 1])
    true_heights = np.sort(original_true_points[:, 1])
    # 预测点数量多（十几个），取前后2个均值；真实点只有4个，取首末1个
    pred_upper, pred_lower = np.mean(pred_heights[:2]), np.mean(pred_heights[-2:])
    true_upper, true_lower = np.mean(true_heights[:1]), np.mean(true_heights[-1:])
    
    # 计算IoU：在原始坐标系中
    true_mask = np.zeros((1504, 1504), dtype=np.uint8)
    cv2.fillPoly(true_mask, [original_true_points.astype(np.int32)], 255)
    
    pred_mask_original = np.zeros((1504, 1504), dtype=np.uint8)
    cv2.fillPoly(pred_mask_original, [pred_points_original.astype(np.int32)], 255)
    
    iou = calculate_iou(true_mask, pred_mask_original)
    
    # 记录指标到对应分组
    metrics[group_key]['iou_list'].append(iou)
    metrics[group_key]['height_upper_diff'].append(abs(true_upper - pred_upper))
    metrics[group_key]['height_lower_diff'].append(abs(true_lower - pred_lower))
    metrics[group_key]['height_upper_diff_percent'].append(abs((true_upper - pred_upper) / true_upper))
    metrics[group_key]['height_lower_diff_percent'].append(abs((true_lower - pred_lower) / true_lower))


def calculate_metrics(true_points, pred_points, pred_mask, metrics):
    """计算评估指标 - 保留用于向后兼容"""
    # 计算高度
    pred_heights = np.sort(pred_points[:, 1])
    true_heights = np.sort(true_points[:, 1])
    pred_upper, pred_lower = np.mean(pred_heights[:2]), np.mean(pred_heights[-2:])
    true_upper, true_lower = np.mean(true_heights[:1]), np.mean(true_heights[-1:])
    
    # 计算IoU
    true_mask = np.zeros((1504, 1504), dtype=np.uint8)
    cv2.fillPoly(true_mask, [true_points], 255)
    iou = calculate_iou(true_mask, pred_mask)
    
    # 记录指标
    metrics['iou_list'].append(iou)
    metrics['height_upper_diff'].append(abs(true_upper - pred_upper))
    metrics['height_lower_diff'].append(abs(true_lower - pred_lower))
    metrics['height_upper_diff_percent'].append(abs((true_upper - pred_upper) / true_upper))
    metrics['height_lower_diff_percent'].append(abs((true_lower - pred_lower) / true_lower))


def print_evaluation_results(metrics, include_augmented):
    """打印分组评估结果"""
    print('\n=== 双模态YOLO评估结果 ===')
    
    # 打印原始数据结果
    original_metrics = metrics['original']
    if original_metrics['total_count'] > 0:
        print(f'\n【原始数据】 (共{original_metrics["total_count"]}张图像)')
        detection_rate = (original_metrics['detected_count'] / original_metrics['total_count']) * 100
        print(f'  检测率: {detection_rate:.2f}%')
        
        if original_metrics["iou_list"]:
            print(f'  平均IoU: {np.mean(original_metrics["iou_list"]):.4f}')
            print(f'  上表面差异: {np.mean(original_metrics["height_upper_diff"]):.2f} 像素 ({np.mean(original_metrics["height_upper_diff_percent"])*100:.2f}%)')
            print(f'  下表面差异: {np.mean(original_metrics["height_lower_diff"]):.2f} 像素 ({np.mean(original_metrics["height_lower_diff_percent"])*100:.2f}%)')
        else:
            print('  没有成功检测到任何目标')
    
    # 打印增强数据结果
    if include_augmented:
        augmented_metrics = metrics['augmented']
        if augmented_metrics['total_count'] > 0:
            print(f'\n【增强数据集】 (共{augmented_metrics["total_count"]}张图像，包含原始数据)')
            detection_rate = (augmented_metrics['detected_count'] / augmented_metrics['total_count']) * 100
            print(f'  检测率: {detection_rate:.2f}%')
            
            if augmented_metrics["iou_list"]:
                print(f'  平均IoU: {np.mean(augmented_metrics["iou_list"]):.4f}')
                print(f'  上表面差异: {np.mean(augmented_metrics["height_upper_diff"]):.2f} 像素 ({np.mean(augmented_metrics["height_upper_diff_percent"])*100:.2f}%)')
                print(f'  下表面差异: {np.mean(augmented_metrics["height_lower_diff"]):.2f} 像素 ({np.mean(augmented_metrics["height_lower_diff_percent"])*100:.2f}%)')
            else:
                print('  没有成功检测到任何目标')
    
    # 故障排除提示
    if not any(metrics[key]["iou_list"] for key in metrics):
        print('\n建议检查：')
        print('- 模型文件是否存在且正确')
        print('- 数据格式是否匹配')
        print('- JSON标注文件是否找到')
        print('- 模型输入数据范围和格式')


def generate_evaluation_chart(metrics, save_dir, fusion_name):
    """生成合并的评估对比图表"""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 准备数据
    original_metrics = metrics['original']
    augmented_metrics = metrics['augmented']
    
    # 提取数据
    original_data = []
    augmented_data = []
    
    for key in ['original', 'augmented']:
        group_metrics = metrics[key]
        if group_metrics['total_count'] > 0:
            detection_rate = group_metrics['detected_count'] / group_metrics['total_count']
            if group_metrics['iou_list']:
                iou_mean = np.mean(group_metrics['iou_list'])
                upper_diff = np.mean(group_metrics['height_upper_diff'])
                lower_diff = np.mean(group_metrics['height_lower_diff'])
            else:
                iou_mean = 0
                upper_diff = 0
                lower_diff = 0
        else:
            detection_rate = 0
            iou_mean = 0
            upper_diff = 0
            lower_diff = 0
        
        if key == 'original':
            original_data = [detection_rate, iou_mean, upper_diff, lower_diff]
        else:
            augmented_data = [detection_rate, iou_mean, upper_diff, lower_diff]
    
    # 设置x轴位置和宽度
    metrics_labels = ['Detection Rate', 'IoU', 'Upper Surface', 'Lower Surface']
    x_pos = np.arange(len(metrics_labels))
    width = 0.35
    
    # 定义颜色 - 4种不同的基础颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红
    
    # 分别处理比例指标和像素差异指标
    
    # 设置左侧y轴 (比例类指标)
    ax1.set_ylabel('Ratio', fontsize=12)
    ax1.set_ylim([0, 1.01])
    ax1.tick_params(axis='y', labelcolor='black')
    
    # 创建右侧y轴 (像素差异指标)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Pixel Difference', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='black')
    
    # 绘制所有4个指标的柱状图
    for i, (metric_name, color) in enumerate(zip(metrics_labels, colors)):
        x_position = x_pos[i]
        
        if i < 2:  # Detection Rate 和 IoU 使用左侧y轴
            # Original数据：不透明
            ax1.bar(x_position - width/2, original_data[i], width, 
                   color=color, alpha=0.9, label='Original' if i == 0 else "")
            # Augmented数据：透明
            ax1.bar(x_position + width/2, augmented_data[i], width, 
                   color=color, alpha=0.6, label='Augmented' if i == 0 else "")
        else:  # Upper Surface 和 Lower Surface 使用右侧y轴
            # Original数据：不透明
            ax2.bar(x_position - width/2, original_data[i], width, 
                   color=color, alpha=0.9)
            # Augmented数据：透明
            ax2.bar(x_position + width/2, augmented_data[i], width, 
                   color=color, alpha=0.6)
    
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
    plt.savefig(save_dir / f'evaluation_chart_{fusion_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    for fusion_name in ['crossattn', ]: #'crossattn-30epoch', 'id', 'concat-compress', 'weighted-fusion']: 
        evaluate_dual_yolo_model(fusion_name=fusion_name, debug=True, include_augmented=True)