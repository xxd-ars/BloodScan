"""
双模态YOLO评估脚本 V4 - 简化重构版
- 学术指标: AP50, AP75, AP50-95, Precision, Recall, F1 @ conf=0.001
- 医学指标: Detection_Rate, IoU, Surface_Diff @ IoU≥0.5, conf=可调
- 按类别计算 + 总体平均
"""

import torch
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dual_dataset.d_dataset_config import DatasetConfig


class EvaluatorV4:
    def __init__(self, model_name, train_mode='pretrained', conf_medical=0.5):
        self.model_name = model_name
        self.train_mode = train_mode
        self.conf_medical = conf_medical
        self.conf_academic = 0.001

        # 路径配置
        root = Path(__file__).parent.parent
        self.model_yaml = root / 'dual_yolo' / 'models' / f'yolo11x-dseg-{model_name}.yaml'
        self.model_pt = root / 'dual_yolo' / 'runs' / 'segment' / f'dual_modal_{train_mode}_{model_name}' / 'weights' / 'best.pt'
        self.dataset = root / 'datasets' / 'Dual-Modal-1504-500-1-6ch'
        self.save_dir = root / 'dual_yolo' / 'evaluation_results_v4' / f'conf_{conf_medical}' / model_name
        self.vis_dir = self.save_dir / 'visualizations'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        # 数据配置
        self.config = DatasetConfig()
        self.classes = {0: 'serum', 1: 'buffy_coat', 2: 'plasma'}

        # 学术指标累积器 (conf=0.001)
        self.academic_data = {'tp': [], 'conf': [], 'pred_cls': [], 'target_cls': []}

        # 医学指标累积器 (conf=conf_medical)
        self.medical_data = {cls_id: {
            'detected_count': 0, 'total_count': 0,
            'iou_list': [], 'upper_diff_list': [], 'lower_diff_list': [],
            'raw_samples': []  # 存储每个样本的详细数据
        } for cls_id in self.classes}

        self.overall_success_count = 0  # 所有类别都正确检测的图像数

        self.model = None
        self.device = "cuda:3" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """加载模型"""
        # 特殊处理: crossattn-30epoch 使用 crossattn 配置
        if self.model_name == 'crossattn-30epoch':
            yaml_path = self.model_yaml.parent / 'yolo11x-dseg-crossattn.yaml'
        else:
            yaml_path = self.model_yaml

        self.model = YOLO(yaml_path).load(self.model_pt)
        print(f"✅ 模型加载: {self.model_name}")

    def load_gt(self, label_file):
        """加载GT标签 -> {class_id: np.array([[x,y], ...])"""
        gt = {}
        if not label_file.exists():
            return gt

        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                cls_id = int(parts[0])
                coords = np.array(parts[1:], dtype=float).reshape(-1, 2) * 1504
                gt[cls_id] = coords
        return gt

    def calc_iou(self, pred_pts, gt_pts):
        """计算Mask IoU"""
        pred_mask = np.zeros((1504, 1504), np.uint8)
        gt_mask = np.zeros((1504, 1504), np.uint8)
        cv2.fillPoly(pred_mask, [pred_pts.astype(np.int32)], 1)
        cv2.fillPoly(gt_mask, [gt_pts.astype(np.int32)], 1)

        inter = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        return inter / union if union > 0 else 0.0

    def calc_surface_diff(self, pred_pts, gt_pts, angle):
        """计算上下表面差异"""
        if angle != 0:
            center = (752, 752)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            pred_pts = cv2.transform(pred_pts.reshape(1, -1, 2), M).reshape(-1, 2)
            gt_pts = cv2.transform(gt_pts.reshape(1, -1, 2), M).reshape(-1, 2)

        pred_y = np.sort(pred_pts[:, 1])
        gt_y = np.sort(gt_pts[:, 1])

        upper_diff = abs(pred_y[:3].mean() - gt_y[:3].mean())
        lower_diff = abs(pred_y[-3:].mean() - gt_y[-3:].mean())
        return upper_diff, lower_diff

    def eval_image(self, npy_file):
        """评估单张图像"""
        # 加载图像
        img_path = self.dataset / 'test' / 'images' / npy_file
        label_path = self.dataset / 'test' / 'labels' / npy_file.replace('.npy', '.txt')

        img_data = np.load(img_path)
        if img_data.shape[-1] == 6:
            img_data = img_data.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_data / 255.0).unsqueeze(0).float()

        # 获取旋转角度
        suffix = npy_file.split('_')[-1].replace('.npy', '')
        angle = self.config.strategies.get(suffix, {}).get('rotation', 0)

        # 加载GT
        gt_masks = self.load_gt(Path(label_path))
        if not gt_masks:
            return

        # 推理两次: 学术 (conf=0.001) + 医学 (conf=conf_medical)
        results_academic = self.model(img_tensor, imgsz=1504, device=self.device,
                                       conf=self.conf_academic, verbose=False)
        results_medical = self.model(img_tensor, imgsz=1504, device=self.device,
                                      conf=self.conf_medical, verbose=False)

        # 收集学术指标数据
        self._collect_academic(results_academic[0], gt_masks)

        # 收集医学指标数据并可视化
        class_success = self._collect_medical(results_medical[0], gt_masks, angle, npy_file)

        # 判断整体成功
        if len(class_success) == len(gt_masks) and all(class_success.values()):
            self.overall_success_count += 1

        # 可视化
        self._visualize(npy_file, img_data, results_medical[0], gt_masks, class_success)

    def _collect_academic(self, result, gt_masks):
        """收集学术指标数据 (用于计算mAP)"""
        gt_classes = np.array(list(gt_masks.keys()))

        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            # 无检测，记录GT
            self.academic_data['tp'].append(np.zeros((0, 10)))
            self.academic_data['conf'].append(np.array([]))
            self.academic_data['pred_cls'].append(np.array([]))
            self.academic_data['target_cls'].append(gt_classes)
            return

        n_pred = len(result.boxes)
        pred_confs = result.boxes.conf.cpu().numpy()
        pred_classes = result.boxes.cls.cpu().numpy().astype(int)

        # 提取预测masks
        pred_masks = []
        for i in range(n_pred):
            pts = result[i].masks.xyn[0].copy() * 1504
            pred_masks.append(pts)

        # 提取GT masks
        gt_masks_list = [gt_masks[cls_id] for cls_id in gt_classes]

        # 计算TP矩阵 (10个IoU阈值: 0.5, 0.55, ..., 0.95)
        iou_thresholds = np.linspace(0.5, 0.95, 10)
        tp_matrix = np.zeros((n_pred, 10))

        sorted_idx = np.argsort(-pred_confs)

        for iou_idx, iou_thr in enumerate(iou_thresholds):
            matched_gt = set()
            for pred_idx in sorted_idx:
                pred_cls = pred_classes[pred_idx]
                best_iou = 0
                best_gt_idx = -1

                for gt_idx, gt_cls in enumerate(gt_classes):
                    if gt_cls != pred_cls or gt_idx in matched_gt:
                        continue
                    iou = self.calc_iou(pred_masks[pred_idx], gt_masks_list[gt_idx])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_gt_idx >= 0 and best_iou >= iou_thr:
                    tp_matrix[pred_idx, iou_idx] = 1
                    if iou_idx == 0:
                        matched_gt.add(best_gt_idx)

        self.academic_data['tp'].append(tp_matrix)
        self.academic_data['conf'].append(pred_confs)
        self.academic_data['pred_cls'].append(pred_classes)
        self.academic_data['target_cls'].append(gt_classes)

    def _collect_medical(self, result, gt_masks, angle, npy_file):
        """收集医学指标数据 (检测率 + IoU + Surface Diff @ IoU≥0.5)
        返回: {cls_id: success} 字典
        """
        # 统计每个类别的检测次数
        detected_classes = {}
        if hasattr(result, 'boxes') and result.boxes is not None:
            for i, cls_id in enumerate(result.boxes.cls.cpu().numpy().astype(int)):
                detected_classes.setdefault(cls_id, []).append(i)

        class_success = {}

        # 按类别评估
        for cls_id, gt_pts in gt_masks.items():
            metrics = self.medical_data[cls_id]
            metrics['total_count'] += 1

            # 检查是否恰好检测1次
            if detected_classes.get(cls_id, []) != [detected_classes.get(cls_id, [None])[0]] or \
               len(detected_classes.get(cls_id, [])) != 1:
                class_success[cls_id] = False
                continue

            # 提取预测结果
            det_idx = detected_classes[cls_id][0]
            pred_pts = result[det_idx].masks.xyn[0].copy() * 1504

            # 计算IoU (用于判断是否≥0.5)
            iou = self.calc_iou(pred_pts, gt_pts)
            if iou < 0.5:
                class_success[cls_id] = False
                continue

            # 计算表面差异
            upper_diff, lower_diff = self.calc_surface_diff(pred_pts, gt_pts, angle)

            # 通过检测 (IoU≥0.5 且恰好1次)
            metrics['detected_count'] += 1
            metrics['iou_list'].append(iou)
            metrics['upper_diff_list'].append(upper_diff)
            metrics['lower_diff_list'].append(lower_diff)

            # 记录原始样本数据
            metrics['raw_samples'].append({
                'filename': npy_file,
                'iou': float(iou),
                'upper_diff': float(upper_diff),
                'lower_diff': float(lower_diff)
            })

            class_success[cls_id] = True

        return class_success

    def _visualize(self, npy_file, img_data, result, gt_masks, class_success):
        """可视化检测结果"""
        # 使用蓝光通道 (channels 0-2)
        blue_img = img_data[:3][[2, 1, 0], :, :]  # BGR
        vis_img = blue_img.transpose(1, 2, 0)
        vis_img = np.clip(vis_img * 255 if vis_img.max() <= 1.0 else vis_img, 0, 255).astype(np.uint8)
        vis_img = np.ascontiguousarray(vis_img)

        # 颜色方案
        colors = {0: (0, 255, 255), 1: (0, 255, 0), 2: (255, 0, 0)}  # 黄绿蓝

        # 绘制GT (红色点)
        for cls_id, gt_pts in gt_masks.items():
            for pt in gt_pts:
                cv2.circle(vis_img, tuple(map(int, pt)), 5, (0, 0, 255), -1)

        # 绘制预测结果
        if hasattr(result, 'boxes') and result.boxes is not None:
            for i, cls_id in enumerate(result.boxes.cls.cpu().numpy().astype(int)):
                if cls_id in colors:
                    pred_pts = result[i].masks.xyn[0].copy() * 1504
                    pred_pts_int = pred_pts.astype(np.int32).reshape((-1, 1, 2))

                    # 绘制预测多边形
                    cv2.polylines(vis_img, [pred_pts_int], True, (255, 255, 255), 1, cv2.LINE_AA)

                    # 绘制预测点十字
                    for pt in pred_pts:
                        x, y = int(pt[0]), int(pt[1])
                        cv2.line(vis_img, (x-3, y), (x+3, y), colors[cls_id], 2)
                        cv2.line(vis_img, (x, y-3), (x, y+3), colors[cls_id], 2)

        # 保存
        overall_success = len(class_success) == len(gt_masks) and all(class_success.values())
        suffix = '_success.jpg' if overall_success else '_fail.jpg'
        save_path = self.vis_dir / f"{npy_file.replace('.npy', '')}{suffix}"
        cv2.imwrite(str(save_path), vis_img)

    def compute_metrics(self):
        """计算最终指标"""
        results = {
            'model': self.model_name,
            'conf_academic': self.conf_academic,
            'conf_medical': self.conf_medical,
            'per_class_metrics': {},
            'aggregate_metrics': {}
        }

        # 计算学术指标 (mAP)
        tp = np.vstack(self.academic_data['tp'])
        conf = np.concatenate(self.academic_data['conf'])
        pred_cls = np.concatenate(self.academic_data['pred_cls'])
        target_cls = np.concatenate(self.academic_data['target_cls'])

        _, _, p, r, f1, ap, unique_cls, *_ = ap_per_class(
            tp, conf, pred_cls, target_cls, plot=False, save_dir=Path(), names={}
        )

        # 按类别组织结果
        for i, cls_id in enumerate(unique_cls):
            cls_id = int(cls_id)
            cls_name = self.classes[cls_id]

            # 学术指标
            academic = {
                'AP50': float(ap[i, 0]),
                'AP75': float(ap[i, 5]),
                'AP50_95': float(ap[i].mean()),
                'Precision_IoU0.5': float(p[i]),
                'Recall_IoU0.5': float(r[i]),
                'F1_IoU0.5': float(f1[i])
            }

            # 医学指标
            med = self.medical_data[cls_id]
            medical = {
                'Detection_Rate_IoU0.5_conf': med['detected_count'] / med['total_count'] if med['total_count'] > 0 else 0,
                'IoU_mean_IoU0.5_conf': float(np.mean(med['iou_list'])) if med['iou_list'] else 0,
                'IoU_std_IoU0.5_conf': float(np.std(med['iou_list'])) if med['iou_list'] else 0,
                'Upper_Diff_mean_IoU0.5_conf': float(np.mean(med['upper_diff_list'])) if med['upper_diff_list'] else 0,
                'Upper_Diff_std_IoU0.5_conf': float(np.std(med['upper_diff_list'])) if med['upper_diff_list'] else 0,
                'Lower_Diff_mean_IoU0.5_conf': float(np.mean(med['lower_diff_list'])) if med['lower_diff_list'] else 0,
                'Lower_Diff_std_IoU0.5_conf': float(np.std(med['lower_diff_list'])) if med['lower_diff_list'] else 0,
                'raw_samples': med['raw_samples']  # 包含每个样本的详细数据
            }

            results['per_class_metrics'][f'class_{cls_id}_{cls_name}'] = {
                'academic': academic,
                'medical': medical
            }

        # 总体指标
        all_academic = [v['academic'] for v in results['per_class_metrics'].values()]
        all_medical = [v['medical'] for v in results['per_class_metrics'].values()]

        # 计算总图像数
        total_images = self.medical_data[0]['total_count']

        results['aggregate_metrics'] = {
            'academic': {
                'mAP50': float(np.mean([a['AP50'] for a in all_academic])),
                'mAP75': float(np.mean([a['AP75'] for a in all_academic])),
                'mAP50_95': float(np.mean([a['AP50_95'] for a in all_academic])),
                'mean_Precision': float(np.mean([a['Precision_IoU0.5'] for a in all_academic])),
                'mean_Recall': float(np.mean([a['Recall_IoU0.5'] for a in all_academic])),
                'mean_F1': float(np.mean([a['F1_IoU0.5'] for a in all_academic]))
            },
            'medical': {
                'mean_Detection_Rate': float(np.mean([m['Detection_Rate_IoU0.5_conf'] for m in all_medical])),
                'mean_IoU': float(np.mean([m['IoU_mean_IoU0.5_conf'] for m in all_medical])),
                'mean_Upper_Diff': float(np.mean([m['Upper_Diff_mean_IoU0.5_conf'] for m in all_medical])),
                'mean_Lower_Diff': float(np.mean([m['Lower_Diff_mean_IoU0.5_conf'] for m in all_medical])),
                'Overall_Success_Rate': float(self.overall_success_count / total_images) if total_images > 0 else 0.0,
                'Overall_Success_Count': int(self.overall_success_count),
                'Total_Images': int(total_images)
            }
        }

        return results

    def save_results(self, results):
        """保存结果"""
        save_path = self.save_dir / f'metrics_{self.model_name}.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✅ 结果保存: {save_path}")

    def print_results(self, results):
        """打印结果"""
        print(f"\n{'='*80}")
        print(f"  {self.model_name} 评估结果")
        print(f"  学术conf={self.conf_academic}, 医学conf={self.conf_medical}")
        print(f"{'='*80}\n")

        # 按类别打印
        for key, metrics in results['per_class_metrics'].items():
            cls_name = key.split('_')[-1]
            print(f"【{cls_name}】")
            print(f"  学术: AP50={metrics['academic']['AP50']:.4f} "
                  f"AP50-95={metrics['academic']['AP50_95']:.4f} "
                  f"Recall={metrics['academic']['Recall_IoU0.5']:.4f}")
            print(f"  医学: DR={metrics['medical']['Detection_Rate_IoU0.5_conf']:.2%} "
                  f"IoU={metrics['medical']['IoU_mean_IoU0.5_conf']:.4f}±{metrics['medical']['IoU_std_IoU0.5_conf']:.4f} "
                  f"UD={metrics['medical']['Upper_Diff_mean_IoU0.5_conf']:.1f}±{metrics['medical']['Upper_Diff_std_IoU0.5_conf']:.1f}")

        # 总体指标
        agg = results['aggregate_metrics']
        print(f"\n【总体】")
        print(f"  学术: mAP50={agg['academic']['mAP50']:.4f} "
              f"mAP50-95={agg['academic']['mAP50_95']:.4f} "
              f"Recall={agg['academic']['mean_Recall']:.4f}")
        print(f"  医学: DR={agg['medical']['mean_Detection_Rate']:.2%} "
              f"IoU={agg['medical']['mean_IoU']:.4f}")
        print(f"{'='*80}\n")

    def run(self):
        """运行评估"""
        self.load_model()

        # 获取测试文件
        test_dir = self.dataset / 'test' / 'images'
        npy_files = sorted([f for f in test_dir.glob('*.npy')
                           if f.stem.split('_')[-1] in self.config.strategies])

        print(f"开始评估 {len(npy_files)} 张图像...")

        for npy_file in tqdm(npy_files):
            self.eval_image(npy_file.name)

        # 计算并保存结果
        results = self.compute_metrics()
        self.save_results(results)
        self.print_results(results)


def main():
    models = ['id-blue', 'id-white', 'crossattn', 'crossattn-precise',
              'weighted-fusion', 'concat-compress']
    # 推荐使用较低conf以鼓励检测，提高召回率
    conf_thresholds = [0.25, 0.3, 0.35, 0.4, 0.5]
    train_mode = 'pretrained'

    for model in models:
        for conf in conf_thresholds:
            print(f"\n{'='*80}")
            print(f"评估: {model} @ conf={conf}")
            print(f"{'='*80}")

            evaluator = EvaluatorV4(model, train_mode, conf)
            evaluator.run()


if __name__ == '__main__':
    main()
