"""
双模态YOLO模型评估脚本V2
基于YOLO训练标签进行精度评估，分组计算血清/血浆层和白膜层效果
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
from dual_dataset.d_dataset_config import DatasetConfig


class DualYOLOEvaluatorV2:
    def __init__(self, fusion_name, conf_threshold=0.5):
        self.fusion_name = fusion_name
        self.conf_threshold = conf_threshold
        self.project_root = Path(__file__).parent.parent
        self.model = None
        self.config = DatasetConfig()

        # 结果保存路径
        self.results_dir = self.project_root / 'dual_yolo' / 'evaluation_results_v2' / f'conf_{conf_threshold}' / fusion_name
        self.vis_dir = self.results_dir / 'sample_visualizations'
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        # 初始化metrics
        self.metrics = {
            'serum_plasma': {  # class 0, 2
                'detection_rate': 0.0,
                'iou_scores': [],
                'upper_surface_diffs': [],
                'lower_surface_diffs': [],
                'total_samples': 0,
                'detected_samples': 0,
                'class_data': {
                    'class0': {'iou_scores': [], 'upper_surface_diffs': [], 'lower_surface_diffs': []},
                    'class2': {'iou_scores': [], 'upper_surface_diffs': [], 'lower_surface_diffs': []}
                }
            },
            'buffy_coat': {   # class 1
                'detection_rate': 0.0,
                'iou_scores': [],
                'upper_surface_diffs': [],
                'lower_surface_diffs': [],
                'total_samples': 0,
                'detected_samples': 0,
                'class_data': {
                    'class1': {'iou_scores': [], 'upper_surface_diffs': [], 'lower_surface_diffs': []}
                }
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

            model_pt = self.project_root / 'dual_yolo' / 'runs' / 'segment' / f'dual_modal_train_{self.fusion_name}' / 'weights' / 'best.pt'

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
        """计算IoU"""
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
        # 注意：true_masks已经是旋转后的坐标，无需再次旋转
        for class_id, true_points in true_masks.items():
            for point in true_points:
                cv2.circle(vis_image, tuple(map(int, point)), 5, (0, 0, 255), -1)

        # 绘制预测结果
        for class_id, detection_indices in detected_classes.items():
            if class_id in pred_colors:
                pred_color = pred_colors[class_id]
                for detection_idx in detection_indices:
                    # 提取预测点（模型输出的坐标已经是基于旋转后图像的坐标）
                    pred_points = results[0][detection_idx].masks.xyn[0].copy()
                    pred_points[:, 0] *= 1504
                    pred_points[:, 1] *= 1504

                    # 预测点无需额外旋转，因为模型是在旋转后的图像上训练和预测的

                    # 绘制预测线连接 (白色细线)
                    pred_line = np.array(pred_points, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis_image, [pred_line], True, (255, 255, 255), 1, lineType=cv2.LINE_AA)

                    # 绘制预测点十字
                    for point in pred_points:
                        x, y = int(point[0]), int(point[1])
                        cv2.line(vis_image, (x-3, y), (x+3, y), pred_color, 2)
                        cv2.line(vis_image, (x, y-3), (x, y+3), pred_color, 2)

        # 保存可视化图像，根据检测成功与否使用不同后缀
        base_filename = npy_file.replace('.npy', '')
        if is_successful_detection:
            save_path = self.vis_dir / f'{base_filename}_eval.jpg'
        else:
            save_path = self.vis_dir / f'{base_filename}_lack_detection.jpg'

        # 确保保存路径存在
        os.makedirs(self.vis_dir, exist_ok=True)

        # 使用cv2.imwrite保存图像 (vis_image已经是BGR格式)
        success = cv2.imwrite(str(save_path), vis_image)
        if not success:
            print(f"警告: 无法保存可视化图像到 {save_path}")
        else:
            # print(f"已保存可视化图像: {save_path.name}")
            pass

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
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

            # 评估血清/血浆层组合 (class 0, 2)
            serum_plasma_detected = self._evaluate_group([0, 2], detected_classes, true_masks, results[0], rotation_angle, 'serum_plasma')

            # 评估白膜层 (class 1)
            buffy_detected = self._evaluate_group([1], detected_classes, true_masks, results[0], rotation_angle, 'buffy_coat')

            # 判断整体检测是否成功
            overall_success = serum_plasma_detected or buffy_detected

            # 可视化所有样本
            try:
                # print(f"正在保存可视化样本: {npy_file}")
                self.visualize_sample(npy_file, dual_tensor, detected_classes, true_masks, results, rotation_angle, overall_success)
            except Exception as vis_error:
                # 可视化失败不影响评估，但要记录错误
                print(f"可视化失败 {npy_file}: {vis_error}")

            return overall_success

        except Exception as e:
            print(f"处理失败 {npy_file}: {e}")
            return False

    def _evaluate_group(self, class_ids, detected_classes, true_masks, result, rotation_angle, group_key):
        """评估指定类别组"""
        group_detected = True

        # 检查该组是否有真实标签
        true_classes_in_group = [cid for cid in class_ids if cid in true_masks]
        if not true_classes_in_group:
            return False

        self.metrics[group_key]['total_samples'] += 1

        # 检查每个类别是否被正确检测到恰好一次
        for class_id in true_classes_in_group:
            if class_id not in detected_classes or len(detected_classes[class_id]) != 1:
                group_detected = False
                break

        if group_detected:
            self.metrics[group_key]['detected_samples'] += 1

            # 计算指标
            for class_id in true_classes_in_group:
                detection_idx = detected_classes[class_id][0]

                # 提取预测点（模型输出已经是基于旋转后图像的坐标）
                pred_points = result[detection_idx].masks.xyn[0].copy()
                pred_points[:, 0] *= 1504
                pred_points[:, 1] *= 1504

                # 预测点和真实点都在同一坐标系（旋转后），直接计算
                true_points = true_masks[class_id]

                # 计算IoU（都在旋转后坐标系）
                iou = self.calculate_iou(pred_points, true_points)
                self.metrics[group_key]['iou_scores'].append(iou)

                # 计算表面差异（需要转回原始坐标系进行比较）
                upper_diff, lower_diff = self.calculate_surface_diff(pred_points, true_points, rotation_angle)
                self.metrics[group_key]['upper_surface_diffs'].append(upper_diff)
                self.metrics[group_key]['lower_surface_diffs'].append(lower_diff)

                # 按class分开存储raw_data
                class_key = f'class{class_id}'
                if class_key in self.metrics[group_key]['class_data']:
                    self.metrics[group_key]['class_data'][class_key]['iou_scores'].append(iou)
                    self.metrics[group_key]['class_data'][class_key]['upper_surface_diffs'].append(upper_diff)
                    self.metrics[group_key]['class_data'][class_key]['lower_surface_diffs'].append(lower_diff)

        return group_detected

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

        print(f"开始评估 {len(npy_files)} 张图像，置信度阈值: {self.conf_threshold}")

        # 逐个评估
        for npy_file in tqdm(npy_files, desc="评估进度"):
            self.evaluate_single_image(npy_file, test_images, test_labels)

        # 计算最终检测率
        for group_key in ['serum_plasma', 'buffy_coat']:
            if self.metrics[group_key]['total_samples'] > 0:
                self.metrics[group_key]['detection_rate'] = (
                    self.metrics[group_key]['detected_samples'] / self.metrics[group_key]['total_samples']
                )

        # 保存结果
        self.save_results()
        self.print_results()

    def save_results(self):
        """保存评估结果"""
        # 准备保存数据
        save_data = {}
        for group_key, group_metrics in self.metrics.items():
            # 准备按class分开的raw_data
            raw_data = {}
            for class_key, class_data in group_metrics['class_data'].items():
                raw_data[class_key] = {
                    'iou_scores': [float(x) for x in class_data['iou_scores']],
                    'upper_surface_diffs': [float(x) for x in class_data['upper_surface_diffs']],
                    'lower_surface_diffs': [float(x) for x in class_data['lower_surface_diffs']]
                }

            save_data[group_key] = {
                'detection_rate': group_metrics['detection_rate'],
                'total_samples': group_metrics['total_samples'],
                'detected_samples': group_metrics['detected_samples'],
                'iou_mean': np.mean(group_metrics['iou_scores']) if group_metrics['iou_scores'] else 0,
                'iou_std': np.std(group_metrics['iou_scores']) if group_metrics['iou_scores'] else 0,
                'upper_diff_mean': np.mean(group_metrics['upper_surface_diffs']) if group_metrics['upper_surface_diffs'] else 0,
                'upper_diff_std': np.std(group_metrics['upper_surface_diffs']) if group_metrics['upper_surface_diffs'] else 0,
                'lower_diff_mean': np.mean(group_metrics['lower_surface_diffs']) if group_metrics['lower_surface_diffs'] else 0,
                'lower_diff_std': np.std(group_metrics['lower_surface_diffs']) if group_metrics['lower_surface_diffs'] else 0,
                'raw_data': raw_data
            }

        # 保存JSON
        metrics_file = self.results_dir / f'metrics_{self.fusion_name}.json'
        with open(metrics_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        # 生成图表
        self.generate_chart()

        print(f"结果已保存到: {self.results_dir}")

    def generate_chart(self):
        """生成评估图表"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

        groups = ['Serum/Plasma', 'Buffy Coat']
        group_keys = ['serum_plasma', 'buffy_coat']

        # 检测率
        detection_rates = [self.metrics[key]['detection_rate'] for key in group_keys]
        ax1.bar(groups, detection_rates, color=['skyblue', 'lightcoral'])
        ax1.set_title('Detection Rate')
        ax1.set_ylabel('Rate')
        ax1.set_ylim(0, 1)

        # IoU
        iou_means = [np.mean(self.metrics[key]['iou_scores']) if self.metrics[key]['iou_scores'] else 0
                     for key in group_keys]
        ax2.bar(groups, iou_means, color=['skyblue', 'lightcoral'])
        ax2.set_title('Mean IoU')
        ax2.set_ylabel('IoU')
        ax2.set_ylim(0, 1)

        # 上表面差异
        upper_diffs = [np.mean(self.metrics[key]['upper_surface_diffs']) if self.metrics[key]['upper_surface_diffs'] else 0
                       for key in group_keys]
        ax3.bar(groups, upper_diffs, color=['skyblue', 'lightcoral'])
        ax3.set_title('Upper Surface Difference (pixels)')
        ax3.set_ylabel('Pixels')

        # 下表面差异
        lower_diffs = [np.mean(self.metrics[key]['lower_surface_diffs']) if self.metrics[key]['lower_surface_diffs'] else 0
                       for key in group_keys]
        ax4.bar(groups, lower_diffs, color=['skyblue', 'lightcoral'])
        ax4.set_title('Lower Surface Difference (pixels)')
        ax4.set_ylabel('Pixels')

        plt.suptitle(f'Dual-Modal YOLO Evaluation Results - {self.fusion_name} (conf={self.conf_threshold})')
        plt.tight_layout()
        plt.savefig(self.results_dir / f'evaluation_chart_{self.fusion_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def print_results(self):
        """打印评估结果"""
        print(f'\n=== {self.fusion_name} 评估结果 (置信度: {self.conf_threshold}) ===')

        for group_name, group_key in [('血清/血浆层', 'serum_plasma'), ('白膜层', 'buffy_coat')]:
            metrics = self.metrics[group_key]
            print(f'\n【{group_name}】')
            print(f'  总样本数: {metrics["total_samples"]}')
            print(f'  检测率: {metrics["detection_rate"]:.2%}')

            if metrics['iou_scores']:
                print(f'  平均IoU: {np.mean(metrics["iou_scores"]):.4f}')
                print(f'  上表面差异: {np.mean(metrics["upper_surface_diffs"]):.2f} 像素')
                print(f'  下表面差异: {np.mean(metrics["lower_surface_diffs"]):.2f} 像素')
            else:
                print('  无成功检测样本')


def main():
    """主函数"""
    fusion_names = ['crossattn-precise', 'crossattn-30epoch', 'weighted-fusion', 'concat-compress', 'id']
    conf_thresholds = [0.65, 0.75, 0.8]

    for fusion_name in fusion_names:
        for conf_threshold in conf_thresholds:
            print(f"\n开始评估: {fusion_name}, 置信度: {conf_threshold}")
            evaluator = DualYOLOEvaluatorV2(fusion_name, conf_threshold)
            evaluator.run_evaluation()


if __name__ == '__main__':
    main()