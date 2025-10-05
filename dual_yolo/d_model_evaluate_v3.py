"""
双模态YOLO模型评估脚本V3 - 重构版
- 按类别独立评估（class 0, 1, 2分开统计）
- 医学指标：Detection Rate, IoU, Surface Diff（固定conf阈值）
- 学术指标：mAP@0.5, mAP@0.5:0.95, Precision, Recall（遍历所有conf）
"""

import torch
import sys
import os
import json
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO
from dual_dataset.d_dataset_config import DatasetConfig
from ultralytics.utils.metrics import ap_per_class


class mAPCalculator:
    """基于Ultralytics的mAP计算器"""

    def __init__(self, class_config):
        self.class_config = class_config
        self.iou_thresholds = np.linspace(0.5, 0.95, 10)

        # 累积数据（跨所有测试图像）
        self.all_tp_box = []
        self.all_tp_mask = []
        self.all_conf = []
        self.all_pred_cls = []
        self.all_target_cls = []

    def collect(self, tp_box, tp_mask, conf, pred_cls, target_cls):
        """收集单张图像的检测数据"""
        if len(tp_box) > 0:
            self.all_tp_box.append(tp_box)
            self.all_tp_mask.append(tp_mask)
            self.all_conf.append(conf)
            self.all_pred_cls.append(pred_cls)

        if len(target_cls) > 0:
            self.all_target_cls.append(target_cls)

    def compute(self):
        """计算mAP和相关指标"""
        if not self.all_tp_box:
            return None

        # 合并所有数据
        tp_box = np.vstack(self.all_tp_box)
        tp_mask = np.vstack(self.all_tp_mask)
        conf = np.concatenate(self.all_conf)
        pred_cls = np.concatenate(self.all_pred_cls)
        target_cls = np.concatenate(self.all_target_cls)

        # 计算Box AP
        results_box = ap_per_class(
            tp_box, conf, pred_cls, target_cls,
            plot=False, save_dir=Path(), names={}
        )

        # 计算Mask AP
        results_mask = ap_per_class(
            tp_mask, conf, pred_cls, target_cls,
            plot=False, save_dir=Path(), names={}
        )

        # 解析结果：(tp, fp, p, r, f1, ap, unique_classes, ...)
        _, _, p_box, r_box, f1_box, ap_box, unique_classes, *_ = results_box
        _, _, p_mask, r_mask, f1_mask, ap_mask, _, *_ = results_mask

        # 按类别组织结果
        per_class_results = {}
        for i, class_id in enumerate(unique_classes):
            class_id = int(class_id)
            per_class_results[class_id] = {
                'box_ap50': float(ap_box[i, 0]) if len(ap_box.shape) > 1 else 0.0,
                'box_ap75': float(ap_box[i, 5]) if len(ap_box.shape) > 1 else 0.0,
                'box_ap50_95': float(ap_box[i].mean()) if len(ap_box.shape) > 1 else 0.0,
                'mask_ap50': float(ap_mask[i, 0]) if len(ap_mask.shape) > 1 else 0.0,
                'mask_ap75': float(ap_mask[i, 5]) if len(ap_mask.shape) > 1 else 0.0,
                'mask_ap50_95': float(ap_mask[i].mean()) if len(ap_mask.shape) > 1 else 0.0,
                'precision': float(p_box[i]) if len(p_box) > 0 else 0.0,
                'recall': float(r_box[i]) if len(r_box) > 0 else 0.0,
                'f1_score': float(f1_box[i]) if len(f1_box) > 0 else 0.0
            }

        return per_class_results


class DualYOLOEvaluatorV3:
    def __init__(self, fusion_name, train_mode, conf_threshold=0.5):
        self.fusion_name = fusion_name
        self.conf_threshold = conf_threshold
        self.project_root = Path(__file__).parent.parent
        self.model = None
        self.train_mode = train_mode
        self.config = DatasetConfig()

        # 类别配置
        self.class_config = {
            0: {'name': 'serum', 'display_name': '血清层'},
            1: {'name': 'buffy_coat', 'display_name': '白膜层'},
            2: {'name': 'plasma', 'display_name': '血浆层'}
        }

        # 结果保存路径
        self.results_dir = self.project_root / 'dual_yolo' / 'evaluation_results_v3' / f'conf_{conf_threshold}' / fusion_name
        self.vis_dir = self.results_dir / 'sample_visualizations'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        # 按类别独立存储metrics
        self.per_class_metrics = {
            0: self._create_class_metric_structure(),
            1: self._create_class_metric_structure(),
            2: self._create_class_metric_structure()
        }

        # mAP计算器
        self.map_calculator = mAPCalculator(class_config=self.class_config)

        # 总体统计
        self.total_images = 0
        self.overall_success_count = 0  # 所有类别都正确检测的图像数

    def _create_class_metric_structure(self):
        """创建单个类别的metric结构"""
        return {
            'medical_metrics': {
                'total_samples': 0,
                'detected_samples': 0,
                'detection_rate': 0.0,
                'iou_scores': [],
                'upper_surface_diffs': [],
                'lower_surface_diffs': []
            },
            'raw_data': {
                'iou_scores': [],
                'upper_surface_diffs': [],
                'lower_surface_diffs': []
            }
        }

    def load_model(self):
        """加载模型"""
        if self.fusion_name:
            # 处理特殊情况：crossattn-30epoch使用crossattn的模型架构
            if self.fusion_name == 'crossattn-30epoch':
                model_yaml = self.project_root / 'dual_yolo' / 'models' / f'yolo11x-dseg-crossattn.yaml'
            else:
                model_yaml = self.project_root / 'dual_yolo' / 'models' / f'yolo11x-dseg-{self.fusion_name}.yaml'

            model_pt = self.project_root / 'dual_yolo' / 'runs' / 'segment' / f'dual_modal_{self.train_mode}_{self.fusion_name}' / 'weights' / 'best.pt'

        try:
            self.model = YOLO(model_yaml).load(model_pt)
            print(f"✅ 模型加载成功: {self.fusion_name}")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

    def parse_yolo_label(self, label_file):
        """解析YOLO标签文件"""
        if not os.path.exists(label_file):
            return {}

        class_masks = {}
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                class_id = int(parts[0])
                coords = list(map(float, parts[1:]))

                # 转换归一化坐标到像素坐标
                points = []
                for i in range(0, len(coords), 2):
                    x = coords[i] * 1504
                    y = coords[i + 1] * 1504
                    points.append([x, y])

                class_masks[class_id] = np.array(points, dtype=np.float32)

        return class_masks

    def calculate_iou(self, pred_points, true_points):
        """计算Mask IoU"""
        # 创建预测mask
        pred_mask = np.zeros((1504, 1504), dtype=np.uint8)
        cv2.fillPoly(pred_mask, [pred_points.astype(np.int32)], 255)

        # 创建真实mask
        true_mask = np.zeros((1504, 1504), dtype=np.uint8)
        cv2.fillPoly(true_mask, [true_points.astype(np.int32)], 255)

        # 计算IoU
        intersection = np.logical_and(pred_mask > 0, true_mask > 0).sum()
        union = np.logical_or(pred_mask > 0, true_mask > 0).sum()

        return intersection / union if union > 0 else 0.0

    def calculate_box_iou(self, box1, box2):
        """计算Box IoU (xyxy格式)"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def calculate_surface_diff(self, pred_points, true_points, rotation_angle):
        """计算上下表面差异"""
        # 将预测点和真实点都反向旋转到原始坐标系进行比较
        pred_original = self.apply_rotation(pred_points, -rotation_angle)
        true_original = self.apply_rotation(true_points, -rotation_angle)

        # 获取y坐标并排序
        pred_y = np.sort(pred_original[:, 1])
        true_y = np.sort(true_original[:, 1])

        # 计算上下表面（取前3个和后3个点的均值）
        pred_upper = np.mean(pred_y[:3])
        pred_lower = np.mean(pred_y[-3:])
        true_upper = np.mean(true_y[:3])
        true_lower = np.mean(true_y[-3:])

        upper_diff = abs(pred_upper - true_upper)
        lower_diff = abs(pred_lower - true_lower)

        return upper_diff, lower_diff

    def apply_rotation(self, points, angle, img_size=(1504, 1504)):
        """应用旋转变换"""
        if angle == 0:
            return points

        center = (img_size[0] // 2, img_size[1] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 转换为齐次坐标
        points_array = np.array(points, dtype=np.float32)
        ones = np.ones((points_array.shape[0], 1), dtype=np.float32)
        homogeneous_points = np.hstack([points_array, ones])

        # 应用旋转
        rotated_points = rotation_matrix.dot(homogeneous_points.T).T

        # 裁剪到图像边界
        rotated_points[:, 0] = np.clip(rotated_points[:, 0], 0, img_size[0] - 1)
        rotated_points[:, 1] = np.clip(rotated_points[:, 1], 0, img_size[1] - 1)

        return rotated_points

    def visualize_sample(self, npy_file, dual_tensor, detected_classes, true_masks, results, rotation_angle, is_successful_detection=True):
        """可视化单个样本"""
        # 准备可视化图像（使用蓝光通道）
        blue_channels = dual_tensor[:3, :, :][[2, 1, 0], :, :]  # BGR通道顺序
        blue_image = blue_channels.transpose(1, 2, 0)
        vis_image = np.clip(blue_image * 255 if blue_image.max() <= 1.0 else blue_image, 0, 255).astype(np.uint8)
        vis_image = np.ascontiguousarray(vis_image)

        # 预测点颜色方案 (BGR格式)
        pred_colors = {0: (0, 255, 255), 1: (0, 255, 0), 2: (255, 0, 0)}  # 黄、绿、蓝

        # 绘制真实标注点 (红色圆点)
        for class_id, true_points in true_masks.items():
            for point in true_points:
                cv2.circle(vis_image, tuple(map(int, point)), 5, (0, 0, 255), -1)

        # 绘制预测结果
        for class_id, detection_indices in detected_classes.items():
            if class_id in pred_colors:
                pred_color = pred_colors[class_id]
                for detection_idx in detection_indices:
                    # 提取预测点
                    pred_points = results[0][detection_idx].masks.xyn[0].copy()
                    pred_points[:, 0] *= 1504
                    pred_points[:, 1] *= 1504

                    # 绘制预测线连接 (白色细线)
                    pred_line = np.array(pred_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis_image, [pred_line], True, (255, 255, 255), 1, lineType=cv2.LINE_AA)

                    # 绘制预测点十字
                    for point in pred_points:
                        x, y = int(point[0]), int(point[1])
                        cv2.line(vis_image, (x-3, y), (x+3, y), pred_color, 2)
                        cv2.line(vis_image, (x, y-3), (x, y+3), pred_color, 2)

        # 保存可视化图像
        base_filename = npy_file.replace('.npy', '')
        suffix = '_eval.jpg' if is_successful_detection else '_lack_detection.jpg'
        save_path = self.vis_dir / f'{base_filename}{suffix}'

        cv2.imwrite(str(save_path), vis_image)

    def evaluate_single_image(self, npy_file, test_images_dir, test_labels_dir):
        """评估单张图像"""
        try:
            # 加载图像数据
            dual_tensor = np.load(test_images_dir / npy_file)
            if dual_tensor.shape[-1] == 6:
                dual_tensor = dual_tensor.transpose(2, 0, 1)

            # 准备模型输入
            model_tensor = dual_tensor / 255.0 if dual_tensor.max() > 1.0 else dual_tensor
            model_input = torch.from_numpy(model_tensor).unsqueeze(0).float()

            # 获取旋转角度
            suffix = npy_file.split('_')[-1].replace('.npy', '')
            rotation_angle = self.config.strategies.get(suffix, {}).get('rotation', 0)

            # 加载真实标签
            label_file = test_labels_dir / npy_file.replace('.npy', '.txt')
            true_masks = self.parse_yolo_label(label_file)

            if not true_masks:
                return False

            # 模型推理
            device = "cuda:3" if torch.cuda.is_available() else "cpu"
            results = self.model(model_input, imgsz=1504, device=device, verbose=False, conf=self.conf_threshold)

            # 检测结果分组
            detected_classes = {}
            if hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
                all_classes = results[0].boxes.cls.cpu().numpy()
                for i, cls_id in enumerate(all_classes):
                    cls_id = int(cls_id)
                    if cls_id not in detected_classes:
                        detected_classes[cls_id] = []
                    detected_classes[cls_id].append(i)

            # 按类别独立评估（医学指标）
            class_detection_status = {}
            for class_id, true_points in true_masks.items():
                is_detected = self._evaluate_single_class(
                    class_id,
                    detected_classes.get(class_id, []),
                    true_points,
                    results[0],
                    rotation_angle
                )
                class_detection_status[class_id] = is_detected

            # 收集mAP数据（学术指标，不受检测次数限制）
            self._collect_map_data(results[0], true_masks, rotation_angle)

            # 判断整体成功：所有有GT的类别都恰好检测一次
            overall_success = all(class_detection_status.values())
            if overall_success:
                self.overall_success_count += 1

            # 可视化
            self.visualize_sample(npy_file, dual_tensor, detected_classes, true_masks, results, rotation_angle, overall_success)

            return overall_success

        except Exception as e:
            print(f"处理失败 {npy_file}: {e}")
            return False

    def _evaluate_single_class(self, class_id, detection_indices, true_points, result, rotation_angle):
        """
        评估单个类别的检测结果（医学严格标准）
        返回: 是否正确检测（恰好检测一次）
        """
        metrics = self.per_class_metrics[class_id]['medical_metrics']
        metrics['total_samples'] += 1

        # 检查是否恰好检测一次
        if len(detection_indices) != 1:
            return False

        # 提取预测结果
        detection_idx = detection_indices[0]
        pred_points = result[detection_idx].masks.xyn[0].copy()
        pred_points[:, 0] *= 1504
        pred_points[:, 1] *= 1504

        # 计算IoU
        iou = self.calculate_iou(pred_points, true_points)

        # 计算表面差异
        upper_diff, lower_diff = self.calculate_surface_diff(pred_points, true_points, rotation_angle)

        # 记录数据
        metrics['detected_samples'] += 1
        metrics['iou_scores'].append(iou)
        metrics['upper_surface_diffs'].append(upper_diff)
        metrics['lower_surface_diffs'].append(lower_diff)

        self.per_class_metrics[class_id]['raw_data']['iou_scores'].append(iou)
        self.per_class_metrics[class_id]['raw_data']['upper_surface_diffs'].append(upper_diff)
        self.per_class_metrics[class_id]['raw_data']['lower_surface_diffs'].append(lower_diff)

        return True

    def _collect_map_data(self, result, true_masks, rotation_angle):
        """收集mAP计算所需数据（允许多检测）"""
        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            # 无检测但有GT，需要记录target_cls
            if true_masks:
                target_cls = np.array(list(true_masks.keys()))
                self.map_calculator.collect(
                    np.array([]).reshape(0, 10),  # 空tp_box
                    np.array([]).reshape(0, 10),  # 空tp_mask
                    np.array([]),  # 空conf
                    np.array([]),  # 空pred_cls
                    target_cls
                )
            return

        n_pred = len(result.boxes)

        # 提取预测信息
        pred_boxes = result.boxes.xyxy.cpu().numpy()  # (n_pred, 4)
        pred_confs = result.boxes.conf.cpu().numpy()  # (n_pred,)
        pred_classes = result.boxes.cls.cpu().numpy().astype(int)  # (n_pred,)

        # 提取预测masks
        pred_masks = []
        for i in range(n_pred):
            mask_points = result[i].masks.xyn[0].copy()
            mask_points[:, 0] *= 1504
            mask_points[:, 1] *= 1504
            pred_masks.append(mask_points)

        # 提取GT信息
        gt_boxes = []
        gt_masks = []
        gt_classes = []
        for class_id, true_points in true_masks.items():
            # 从多边形提取bbox
            x_coords = true_points[:, 0]
            y_coords = true_points[:, 1]
            bbox = [x_coords.min(), y_coords.min(), x_coords.max(), y_coords.max()]

            gt_boxes.append(bbox)
            gt_masks.append(true_points)
            gt_classes.append(class_id)

        gt_boxes = np.array(gt_boxes)  # (n_gt, 4)
        gt_classes = np.array(gt_classes)  # (n_gt,)

        # 计算TP矩阵（10个IoU阈值）
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        tp_box = np.zeros((n_pred, 10))
        tp_mask = np.zeros((n_pred, 10))

        # 按置信度排序
        sorted_indices = np.argsort(-pred_confs)

        for threshold_idx, iou_threshold in enumerate(iou_thresholds):
            matched_gts = set()

            for pred_idx in sorted_indices:
                pred_class = pred_classes[pred_idx]

                # 找到同类别的GT
                best_iou_box = 0
                best_iou_mask = 0
                best_gt_idx = -1

                for gt_idx in range(len(gt_classes)):
                    if gt_classes[gt_idx] != pred_class:
                        continue
                    if gt_idx in matched_gts:
                        continue

                    # 计算Box IoU
                    box_iou = self.calculate_box_iou(pred_boxes[pred_idx], gt_boxes[gt_idx])

                    # 计算Mask IoU
                    mask_iou = self.calculate_iou(pred_masks[pred_idx], gt_masks[gt_idx])

                    if mask_iou > best_iou_mask:
                        best_iou_box = box_iou
                        best_iou_mask = mask_iou
                        best_gt_idx = gt_idx

                # 判定TP（每个阈值独立判定）
                if best_gt_idx >= 0 and best_iou_box >= iou_threshold:
                    tp_box[pred_idx, threshold_idx] = 1
                    if threshold_idx == 0:  # 只在第一个阈值时匹配GT
                        matched_gts.add(best_gt_idx)

                if best_gt_idx >= 0 and best_iou_mask >= iou_threshold:
                    tp_mask[pred_idx, threshold_idx] = 1

        # 收集到mAP计算器
        self.map_calculator.collect(tp_box, tp_mask, pred_confs, pred_classes, gt_classes)

    def run_evaluation(self):
        """运行完整评估"""
        if not self.load_model():
            return

        # 数据路径
        dataset_path = self.project_root / 'datasets' / 'Dual-Modal-1504-500-1-6ch'
        test_images = dataset_path / 'test' / 'images'
        test_labels = dataset_path / 'test' / 'labels'

        # 获取所有测试文件
        npy_files = sorted([f for f in os.listdir(test_images)
                           if f.endswith('.npy') and f.split('_')[-1].replace('.npy', '') in self.config.strategies])

        self.total_images = len(npy_files)
        print(f"开始评估 {self.total_images} 张图像，置信度阈值: {self.conf_threshold}")

        # 逐个评估
        for npy_file in tqdm(npy_files, desc="评估进度"):
            self.evaluate_single_image(npy_file, test_images, test_labels)

        # 计算最终检测率
        for class_id in self.per_class_metrics:
            metrics = self.per_class_metrics[class_id]['medical_metrics']
            if metrics['total_samples'] > 0:
                metrics['detection_rate'] = metrics['detected_samples'] / metrics['total_samples']

        # 保存结果
        self.save_results()
        self.print_results()

    def save_results(self):
        """保存评估结果"""
        # 计算统计量
        for class_id in self.per_class_metrics:
            metrics = self.per_class_metrics[class_id]['medical_metrics']
            if metrics['iou_scores']:
                metrics['iou_mean'] = float(np.mean(metrics['iou_scores']))
                metrics['iou_std'] = float(np.std(metrics['iou_scores']))
                metrics['upper_diff_mean'] = float(np.mean(metrics['upper_surface_diffs']))
                metrics['upper_diff_std'] = float(np.std(metrics['upper_surface_diffs']))
                metrics['lower_diff_mean'] = float(np.mean(metrics['lower_surface_diffs']))
                metrics['lower_diff_std'] = float(np.std(metrics['lower_surface_diffs']))

        # 计算mAP
        map_results = self.map_calculator.compute()

        # 构建保存数据
        save_data = {
            'evaluation_metadata': {
                'model': self.fusion_name,
                'conf_threshold': float(self.conf_threshold),
                'total_images': int(self.total_images),
                'dataset': 'Dual-Modal-1504-500-1-6ch/test'
            },
            'per_class_metrics': {},
            'aggregate_metrics': {
                'medical': {
                    'overall_detection_rate': (
                        self.overall_success_count / self.total_images if self.total_images > 0 else 0.0
                    ),
                    'overall_detection_rate_description': '所有类别都正确检测的样本比例'
                },
                'academic': {}
            }
        }

        # 保存每个类别的结果
        for class_id, config in self.class_config.items():
            class_key = f"class_{class_id}_{config['name']}"

            metrics = self.per_class_metrics[class_id]
            class_data = {
                'class_id': int(class_id),
                'class_name': config['name'],
                'display_name': config['display_name'],
                'total_samples': int(metrics['medical_metrics']['total_samples']),
                'medical_metrics': {
                    k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                    for k, v in metrics['medical_metrics'].items()
                    if k not in ['iou_scores', 'upper_surface_diffs', 'lower_surface_diffs']
                },
                'raw_data': {
                    k: [float(x) for x in v]
                    for k, v in metrics['raw_data'].items()
                }
            }

            # 添加学术指标
            if map_results and class_id in map_results:
                class_data['academic_metrics'] = map_results[class_id]

            save_data['per_class_metrics'][class_key] = class_data

        # 计算aggregate academic metrics
        if map_results:
            all_classes_ap = list(map_results.values())
            save_data['aggregate_metrics']['academic'] = {
                'mean_box_ap50': float(np.mean([c['box_ap50'] for c in all_classes_ap])),
                'mean_box_ap50_95': float(np.mean([c['box_ap50_95'] for c in all_classes_ap])),
                'mean_mask_ap50': float(np.mean([c['mask_ap50'] for c in all_classes_ap])),
                'mean_mask_ap50_95': float(np.mean([c['mask_ap50_95'] for c in all_classes_ap])),
                'mean_precision': float(np.mean([c['precision'] for c in all_classes_ap])),
                'mean_recall': float(np.mean([c['recall'] for c in all_classes_ap]))
            }

        # 保存JSON
        metrics_file = self.results_dir / f'metrics_{self.fusion_name}.json'
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 结果已保存到: {metrics_file}")

    def print_results(self):
        """打印评估结果"""
        print(f'\n{"="*70}')
        print(f'  {self.fusion_name} 评估结果 (置信度: {self.conf_threshold})')
        print(f'{"="*70}')
        print(f'总测试图像: {self.total_images}\n')

        # 按类别打印
        for class_id, config in self.class_config.items():
            metrics = self.per_class_metrics[class_id]['medical_metrics']

            print(f'┌{"─"*68}┐')
            print(f'│ Class {class_id}: {config["name"]:<10} ({config["display_name"]})' + ' ' * (68 - 20 - len(config['name']) - len(config['display_name'])) + '│')
            print(f'├{"─"*68}┤')

            # 医学指标
            print(f'│ 医学指标 (Medical Metrics - conf={self.conf_threshold})' + ' ' * 18 + '│')
            print(f'│   总样本数: {metrics["total_samples"]:<6}                                                   │')
            detected = metrics['detected_samples']
            total = metrics['total_samples']
            rate = metrics['detection_rate']
            print(f'│   检测率: {rate:.2%} ({detected}/{total})' + ' ' * (68 - 23 - len(str(detected)) - len(str(total))) + '│')

            if metrics.get('iou_mean') is not None:
                print(f'│   平均IoU: {metrics["iou_mean"]:.4f} ± {metrics["iou_std"]:.4f}' + ' ' * 33 + '│')
                print(f'│   上表面差异: {metrics["upper_diff_mean"]:.2f} ± {metrics["upper_diff_std"]:.2f} 像素' + ' ' * 30 + '│')
                print(f'│   下表面差异: {metrics["lower_diff_mean"]:.2f} ± {metrics["lower_diff_std"]:.2f} 像素' + ' ' * 30 + '│')

            # 学术指标
            map_results = self.map_calculator.compute()
            if map_results and class_id in map_results:
                print(f'│' + ' ' * 68 + '│')
                print(f'│ 学术指标 (Academic Metrics - YOLO Standard)' + ' ' * 24 + '│')
                ap = map_results[class_id]
                print(f'│   Box  mAP@0.5: {ap["box_ap50"]:.4f}  mAP@0.75: {ap["box_ap75"]:.4f}  mAP@0.5:0.95: {ap["box_ap50_95"]:.4f}' + ' ' * 7 + '│')
                print(f'│   Mask mAP@0.5: {ap["mask_ap50"]:.4f}  mAP@0.75: {ap["mask_ap75"]:.4f}  mAP@0.5:0.95: {ap["mask_ap50_95"]:.4f}' + ' ' * 7 + '│')
                print(f'│   Precision: {ap["precision"]:.4f}  Recall: {ap["recall"]:.4f}  F1: {ap["f1_score"]:.4f}' + ' ' * 15 + '│')

            print(f'└{"─"*68}┘\n')

        # 总体指标
        print(f'┌{"─"*68}┐')
        print(f'│ 总体指标 (Overall Metrics)' + ' ' * 42 + '│')
        print(f'├{"─"*68}┤')

        overall_rate = self.overall_success_count / self.total_images if self.total_images > 0 else 0.0
        print(f'│ 医学严格标准                                                          │')
        print(f'│   所有类别正确检测率: {overall_rate:.2%} ({self.overall_success_count}/{self.total_images})' + ' ' * (68 - 34 - len(str(self.overall_success_count)) - len(str(self.total_images))) + '│')

        map_results = self.map_calculator.compute()
        if map_results:
            all_classes_ap = list(map_results.values())
            print(f'│                                                                      │')
            print(f'│ 学术标准 (跨类别平均)                                                 │')
            mean_box_ap50 = np.mean([c['box_ap50'] for c in all_classes_ap])
            mean_box_ap50_95 = np.mean([c['box_ap50_95'] for c in all_classes_ap])
            print(f'│   Mean Box  mAP@0.5: {mean_box_ap50:.4f}  mAP@0.5:0.95: {mean_box_ap50_95:.4f}' + ' ' * 20 + '│')
            mean_mask_ap50 = np.mean([c['mask_ap50'] for c in all_classes_ap])
            mean_mask_ap50_95 = np.mean([c['mask_ap50_95'] for c in all_classes_ap])
            print(f'│   Mean Mask mAP@0.5: {mean_mask_ap50:.4f}  mAP@0.5:0.95: {mean_mask_ap50_95:.4f}' + ' ' * 20 + '│')
            mean_prec = np.mean([c['precision'] for c in all_classes_ap])
            mean_rec = np.mean([c['recall'] for c in all_classes_ap])
            print(f'│   Mean Precision: {mean_prec:.4f}  Mean Recall: {mean_rec:.4f}' + ' ' * 23 + '│')

        print(f'└{"─"*68}┘')


def main():
    """主函数"""
    fusion_names = ['crossattn-precise']
    # fusion_names = ['crossattn', 'crossattn-precise', 'weighted-fusion', 'concat-compress'] # 'id-white', 'id-blue', 
    conf_thresholds = [0.7, 0.75, 0.8]
    train_mode = 'scratch'  # 'scratch', 'pretrained', 'freeze_backbone'

    for fusion_name in fusion_names:
        for conf_threshold in conf_thresholds:
            print(f"\n{'='*70}")
            print(f"开始评估: {fusion_name}, 置信度: {conf_threshold}")
            print(f"{'='*70}")
            evaluator = DualYOLOEvaluatorV3(fusion_name, train_mode, conf_threshold)
            evaluator.run_evaluation()


if __name__ == '__main__':
    main()