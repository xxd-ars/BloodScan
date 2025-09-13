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
import json

# 导入数据增强策略配置
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'dual_dataset'))
from d_dataset_config import DatasetConfig

# 获取增强参数映射表
_config = DatasetConfig()
AUGMENTATION_STRATEGIES = _config.strategies

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
    """提取医生标注点位 - 保持向后兼容，默认提取Class 1"""
    return extract_annotation_points_multiclass(json_data, class_id=1)


def extract_annotation_points_multiclass(json_data, class_id):
    """提取指定类别的医生标注点位
    
    Args:
        json_data: JSON标注数据
        class_id: 类别ID (0=血清层, 1=白膜层, 2=血浆层)
    
    Returns:
        np.array: 排序后的标注点位，如果点数不足返回None
    """
    true_points = []
    shapes = json_data.get("shapes", [])
    
    if class_id == 0:
        # Class 0 (血清层): 使用点位 0,1,2,3
        point_indices = [0, 1, 2, 3]
        min_points = 4
    elif class_id == 1:
        # Class 1 (白膜层): 使用点位 2,3,4,5 - 保持现有逻辑
        point_indices = [2, 3, 4, 5]
        min_points = 4
    elif class_id == 2:
        # Class 2 (血浆层): 使用点位 4,5,6
        point_indices = [4, 5, 6]
        min_points = 3
    else:
        return None
    
    # 提取指定索引的点位
    for idx in point_indices:
        if idx < len(shapes):
            x, y = shapes[idx]["points"][0]
            # 应用坐标变换
            x, y = int((x - 800) * 1504/1216), int(y - 250)
            true_points.append([x, y])
    
    # 检查点数是否严格匹配
    if len(true_points) == min_points:
        return sort_points_by_angle(np.array(true_points, dtype=np.int32))
    else:
        return None


def visualize_results(annotated_image, all_class_data, save_path=None):
    """可视化多类别结果
    
    Args:
        annotated_image: 标注图像
        all_class_data: 所有类别数据字典 {class_id: {'pred_points': points, 'true_points': points}}
        save_path: 保存路径
    """
    # 确保图像格式正确
    if not isinstance(annotated_image, np.ndarray):
        annotated_image = np.array(annotated_image)
    if annotated_image.dtype != np.uint8:
        annotated_image = annotated_image.astype(np.uint8)
    annotated_image = np.ascontiguousarray(annotated_image)
    
    # 预测点颜色方案 (BGR格式)
    pred_colors = {
        0: (0, 255, 255),    # Class 0: 黄色十字
        1: (0, 255, 0),      # Class 1: 绿色十字 (保持现有)
        2: (255, 0, 0)       # Class 2: 蓝色十字
    }
    
    # 先绘制所有真实标注点 (红色圆点)
    for class_id, class_data in all_class_data.items():
        true_points = class_data.get('true_points')
        if true_points is not None:
            for point in true_points:
                cv2.circle(annotated_image, tuple(map(int, point)), 5, (0, 0, 255), -1)
    
    # 再按类别绘制预测结果
    for class_id, class_data in all_class_data.items():
        pred_points = class_data.get('pred_points')
        if pred_points is not None:
            pred_color = pred_colors.get(class_id, (0, 255, 0))  # 默认绿色
            
            # 绘制预测线连接 (白色细线)
            pred_line = np.array(pred_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_image, [pred_line], True, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            
            # 绘制预测点十字
            for point in pred_points:
                x, y = int(point[0]), int(point[1])
                cv2.line(annotated_image, (x-2, y), (x+2, y), pred_color, 2)
                cv2.line(annotated_image, (x, y-2), (x, y+2), pred_color, 2)
    
    # 保存图像
    if save_path:
        rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        plt.imsave(save_path, rgb_image)


def visualize_results_single(annotated_image, pred_points=None, true_points=None, save_path=None, class_id=1):
    """单类别可视化结果 - 保持向后兼容"""
    all_class_data = {}
    if true_points is not None or pred_points is not None:
        all_class_data[class_id] = {
            'true_points': true_points,
            'pred_points': pred_points
        }
    visualize_results(annotated_image, all_class_data, save_path)


def evaluate_dual_yolo_model(fusion_name, debug=False, include_augmented=True, 
                            evaluate_classes=[1], conf_threshold=0.5):
    """主评估函数
    
    Args:
        fusion_name: 融合策略名称
        debug: 是否启用调试模式
        include_augmented: 是否包含增强数据
        evaluate_classes: 要评估的类别列表，默认[1]保持向后兼容
        conf_threshold: YOLO推理置信度阈值，默认0.5，直接在模型推理时应用过滤
    """
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
    
    eval_results_name = 'evaluation_results_aug' if include_augmented else 'evaluation_results'
    for class_id in evaluate_classes:
        eval_results_name += f'_{class_id}'

    eval_results_dir = project_root / 'dual_yolo' / eval_results_name / f'{fusion_name}'
    os.makedirs(eval_results_dir, exist_ok=True)
    
    # 检查路径是否存在
    if debug:
        print(f"🔍 调试信息：")
        print(f"  模型配置文件: {model_yaml} ({'✅存在' if model_yaml.exists() else '❌不存在'})")
        print(f"  模型权重文件: {model_pt} ({'✅存在' if model_pt.exists() else '❌不存在'})")
        print(f"  测试图像目录: {test_images} ({'✅存在' if test_images.exists() else '❌不存在'})")
    
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
        # print(f"评估图像数量: {len(npy_files)} (包含增强数据)")
    else:
        # 只评估原始图像 _0
        npy_files = sorted([f for f in os.listdir(test_images) 
                           if f.endswith('_0.npy')])
        # print(f"评估图像数量: {len(npy_files)} (仅原始数据)")
    
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
        
        # Process for augmented group (always runs)
        metrics['augmented']['total_count'] += 1
        success_aug = process_single_image(npy_file, test_images, model, eval_results_dir, metrics, 'augmented', 
                                        evaluate_classes, conf_threshold)
        if success_aug:
            metrics['augmented']['detected_count'] += 1

        # Process for original group (only for _0 files)
        if is_original:
            metrics['original']['total_count'] += 1
            # We need to re-process for the 'original' metrics group, even if it's redundant,
            # to ensure metrics are stored in the correct dictionary key.
            success_orig = process_single_image(npy_file, test_images, model, eval_results_dir, metrics, 'original', 
                                         evaluate_classes, conf_threshold)
            if success_orig:
                metrics['original']['detected_count'] += 1
    
    # 打印和保存结果
    print_evaluation_results(metrics, include_augmented, fusion_name)
    
    # 保存metrics数据到JSON文件
    save_metrics_to_file(metrics, eval_results_dir, fusion_name)
    
    if any(metrics[key]["iou_list"] for key in metrics):  # 只有在有检测结果时才生成图表
        generate_evaluation_chart(metrics, eval_results_dir, fusion_name)
    
    print(f"\n评估完成！结果保存在 {eval_results_dir}")


def process_single_image(npy_file, test_images_dir, model, results_dir, metrics, group_key, 
                        evaluate_classes=[1], conf_threshold=0.5):
    """处理单个图像评估
    
    Args:
        evaluate_classes: 要评估的类别列表
        conf_threshold: 置信度阈值，直接在模型推理时应用
    """
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
        
        # 模型推理 - 直接在推理时设置置信度阈值
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        results = model(model_input, imgsz=1504,
                        device=device, verbose=False, conf=conf_threshold)
        
        # 检查模型输出并按类别分组（置信度已在推理时过滤）
        detected_classes = {}
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            if len(results[0].boxes) > 0:
                all_classes = results[0].boxes.cls.cpu().numpy()
                
                # 按类别分组检测结果
                for i, cls_id in enumerate(all_classes):
                    if int(cls_id) in evaluate_classes:
                        if int(cls_id) not in detected_classes:
                            detected_classes[int(cls_id)] = []
                        detected_classes[int(cls_id)].append(i)
        
        base_filename = npy_file.replace('.npy', '')
        # --- 新的精确检测计数逻辑 ---
        is_successful_detection = True
        # 检查每个期望的类别是否都只被检测到了一次
        for class_id_to_check in evaluate_classes:
            if detected_classes.get(class_id_to_check) is None or len(detected_classes[class_id_to_check]) != 1:
                is_successful_detection = False
                break # 只要有一个类别不满足条件，就判定为失败

        # --- 可视化和指标计算逻辑 ---
        all_class_data = {}
        
        # 步骤1: 始终准备可视化数据，无论检测是否成功
        # 这样我们总能看到模型到底预测了什么
        for class_id in evaluate_classes:
            original_true_points_class = extract_annotation_points_multiclass(json_data, class_id)
            if original_true_points_class is None:
                continue
            
            rotated_true_points_class = apply_rotation_to_points(original_true_points_class, rotation_angle)
            
            pred_points_for_vis = None
            if class_id in detected_classes:
                # 即使检测失败（比如检测到2个），我们依然提取点位用于可视化
                detections = detected_classes[class_id]
                pred_points_for_vis = extract_prediction_points(results[0], detections)

            all_class_data[class_id] = {
                'true_points': rotated_true_points_class,
                'pred_points': pred_points_for_vis
            }

        # 步骤2: 仅在检测成功时才计算指标
        if is_successful_detection:
            for class_id in evaluate_classes:
                # 我们知道每个类别肯定都在detected_classes里，并且只有一个实例
                detections = detected_classes[class_id]
                pred_points = extract_prediction_points(results[0], detections)
                pred_mask = get_prediction_mask(results[0], detections)
                original_true_points_class = extract_annotation_points_multiclass(json_data, class_id)

                if original_true_points_class is not None:
                    calculate_metrics_multiclass(
                        original_true_points_class, pred_points, pred_mask, 
                        rotation_angle, metrics, group_key, class_id
                    )

        # 步骤3: 根据成功与否决定文件名并进行可视化
        if is_successful_detection:
            save_path = results_dir / f'{base_filename}_evaluation.jpg'
        else:
            save_path = results_dir / f'{base_filename}_no_detection.jpg'
        
        # 执行可视化
        if len(evaluate_classes) == 1:
            class_id = evaluate_classes[0]
            class_data = all_class_data.get(class_id, {})
            visualize_results_single(annotated_image, 
                                   class_data.get('pred_points'), 
                                   class_data.get('true_points'), 
                                   save_path, class_id)
        else:
            visualize_results(annotated_image, all_class_data, save_path)
        
        return is_successful_detection
            
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
    """使用反向旋转方法计算评估指标 - 保持向后兼容，默认为Class 1"""
    calculate_metrics_multiclass(original_true_points, pred_points, pred_mask, rotation_angle, 
                                metrics, group_key, class_id=1)


def calculate_metrics_multiclass(original_true_points, pred_points, pred_mask, rotation_angle, 
                                metrics, group_key, class_id):
    """使用反向旋转方法计算多类别评估指标
    
    Args:
        original_true_points: 原始标注点位
        pred_points: 预测点位
        pred_mask: 预测掩码（暂时保留，用于IoU计算）
        rotation_angle: 旋转角度
        metrics: 指标字典
        group_key: 分组键名
        class_id: 类别ID
    """
    # 将预测点反向旋转到原始坐标系
    pred_points_original = apply_rotation_to_points(pred_points, -rotation_angle)
    
    # 在原始坐标系中计算高度差异
    pred_heights = np.sort(pred_points_original[:, 1])
    true_heights = np.sort(original_true_points[:, 1])
    
    # 根据类别采用不同的高度计算策略
    if class_id == 0:
        # Class 0 (血清层): 4个点，取前后2个均值
        pred_upper, pred_lower = np.mean(pred_heights[:2]), np.mean(pred_heights[-2:])
        true_upper, true_lower = np.mean(true_heights[:2]), np.mean(true_heights[-2:])
    elif class_id == 1:
        # Class 1 (白膜层): 4个点，预测点数量多（十几个），取前后2个均值；真实点只有4个，取首末1个
        pred_upper, pred_lower = np.mean(pred_heights[:2]), np.mean(pred_heights[-2:])
        true_upper, true_lower = np.mean(true_heights[:2]), np.mean(true_heights[-2:])
    elif class_id == 2:
        # Class 2 (血浆层): 3-4个点，取前后2个均值
        pred_upper, pred_lower = np.mean(pred_heights[:2]), np.mean(pred_heights[-2:])
        true_upper, true_lower = np.mean(true_heights[:2]), np.mean(true_heights[-1:])
    else:
        return  # 未知类别，跳过
    
    # 计算IoU：只对Class 0和Class 1计算，Class 2跳过（三角形IoU无意义）
    if class_id in [0, 1]:
        true_mask = np.zeros((1504, 1504), dtype=np.uint8)
        cv2.fillPoly(true_mask, [original_true_points.astype(np.int32)], 255)
        
        pred_mask_original = np.zeros((1504, 1504), dtype=np.uint8)
        cv2.fillPoly(pred_mask_original, [pred_points_original.astype(np.int32)], 255)
        
        iou = calculate_iou(true_mask, pred_mask_original)
        
        # 记录IoU到对应分组
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


def print_evaluation_results(metrics, include_augmented, fusion_name):
    """打印分组评估结果"""
    print(f'\n=== {fusion_name}评估结果 ===')
    
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


def save_metrics_to_file(metrics, save_dir, fusion_name):
    """保存metrics数据到JSON文件"""
    # 计算统计数据
    metrics_summary = {}
    
    for group_key in ['original', 'augmented']:
        group_metrics = metrics[group_key]
        if group_metrics['total_count'] > 0:
            detection_rate = group_metrics['detected_count'] / group_metrics['total_count']
            
            if group_metrics['iou_list']:
                iou_mean = np.mean(group_metrics['iou_list'])
                iou_std = np.std(group_metrics['iou_list'])
                upper_diff_mean = np.mean(group_metrics['height_upper_diff'])
                upper_diff_std = np.std(group_metrics['height_upper_diff'])
                lower_diff_mean = np.mean(group_metrics['height_lower_diff'])
                lower_diff_std = np.std(group_metrics['height_lower_diff'])
                upper_diff_percent_mean = np.mean(group_metrics['height_upper_diff_percent'])
                lower_diff_percent_mean = np.mean(group_metrics['height_lower_diff_percent'])
            else:
                iou_mean = iou_std = 0
                upper_diff_mean = upper_diff_std = 0
                lower_diff_mean = lower_diff_std = 0
                upper_diff_percent_mean = lower_diff_percent_mean = 0
            
            metrics_summary[group_key] = {
                'total_count': group_metrics['total_count'],
                'detected_count': group_metrics['detected_count'],
                'detection_rate': detection_rate,
                'iou_mean': float(iou_mean),
                'iou_std': float(iou_std),
                'upper_diff_mean': float(upper_diff_mean),
                'upper_diff_std': float(upper_diff_std),
                'lower_diff_mean': float(lower_diff_mean), 
                'lower_diff_std': float(lower_diff_std),
                'upper_diff_percent_mean': float(upper_diff_percent_mean),
                'lower_diff_percent_mean': float(lower_diff_percent_mean),
                # 保存原始数据列表
                'raw_data': {
                    'iou_list': [float(x) for x in group_metrics['iou_list']],
                    'height_upper_diff': [float(x) for x in group_metrics['height_upper_diff']],
                    'height_lower_diff': [float(x) for x in group_metrics['height_lower_diff']],
                    'height_upper_diff_percent': [float(x) for x in group_metrics['height_upper_diff_percent']],
                    'height_lower_diff_percent': [float(x) for x in group_metrics['height_lower_diff_percent']]
                }
            }
        else:
            metrics_summary[group_key] = {
                'total_count': 0,
                'detected_count': 0,
                'detection_rate': 0,
                'iou_mean': 0, 'iou_std': 0,
                'upper_diff_mean': 0, 'upper_diff_std': 0,
                'lower_diff_mean': 0, 'lower_diff_std': 0,
                'upper_diff_percent_mean': 0, 'lower_diff_percent_mean': 0,
                'raw_data': {'iou_list': [], 'height_upper_diff': [], 'height_lower_diff': [], 
                           'height_upper_diff_percent': [], 'height_lower_diff_percent': []}
            }
    
    # 保存到JSON文件
    metrics_file = save_dir / f'metrics_{fusion_name}.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    print(f"Metrics数据已保存到: {metrics_file}")


def generate_evaluation_chart(metrics, save_dir, fusion_name):
    """生成合并的评估对比图表"""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # 准备数据
    original_metrics = metrics['original']
    augmented_metrics = metrics['augmented']
    
    # 提取数据 (包括标准差)
    original_data = []
    augmented_data = []
    original_stds = []
    augmented_stds = []
    
    for key in ['original', 'augmented']:
        group_metrics = metrics[key]
        if group_metrics['total_count'] > 0:
            detection_rate = group_metrics['detected_count'] / group_metrics['total_count']
            if group_metrics['iou_list']:
                iou_mean = np.mean(group_metrics['iou_list'])
                iou_std = np.std(group_metrics['iou_list'])
                upper_diff = np.mean(group_metrics['height_upper_diff'])
                upper_diff_std = np.std(group_metrics['height_upper_diff'])
                lower_diff = np.mean(group_metrics['height_lower_diff'])
                lower_diff_std = np.std(group_metrics['height_lower_diff'])
            else:
                iou_mean = iou_std = 0
                upper_diff = upper_diff_std = 0
                lower_diff = lower_diff_std = 0
        else:
            detection_rate = 0
            iou_mean = iou_std = 0
            upper_diff = upper_diff_std = 0
            lower_diff = lower_diff_std = 0
        
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
    
    # 分别处理比例指标和像素差异指标
    
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
    plt.savefig(save_dir / f'evaluation_chart_{fusion_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    fusion_names = ['crossattn-precise']
    # fusion_names = ['id', 'crossattn', 'crossattn-30epoch', 'weighted-fusion', 'concat-compress']
    for fusion_name in fusion_names:
        evaluate_dual_yolo_model(fusion_name=fusion_name, 
                             debug=True, 
                             include_augmented=True, 
                             evaluate_classes=[0, 1, 2], 
                             conf_threshold=0.70)