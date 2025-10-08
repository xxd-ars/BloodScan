"""
双模态YOLO评估脚本 V4 - 多GPU并行优化版
- 图像级并行: 4张GPU同时处理不同图像子集
- 理论加速比: ~4倍
- 学术指标: AP50, AP75, AP50-95, Precision, Recall, F1 @ conf=0.001
- 医学指标: Detection_Rate, IoU, Surface_Diff @ IoU≥0.5, conf=可调
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
import multiprocessing as mp
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dual_dataset.d_dataset_config import DatasetConfig


class EvaluatorV4:
    def __init__(self, model_name, train_mode='pretrained', conf_medical=0.5, gpu_id=0):
        self.model_name = model_name
        self.train_mode = train_mode
        self.conf_medical = conf_medical
        self.conf_academic = 0.001
        self.gpu_id = gpu_id

        # 路径配置
        root = Path(__file__).parent.parent
        self.model_yaml = root / 'dual_yolo' / 'models' / f'yolo11x-dseg-{model_name}.yaml'
        self.model_pt = root / 'dual_yolo' / 'runs' / 'segment' / f'dual_modal_{train_mode}_{model_name}' / 'weights' / 'best.pt'
        self.dataset = root / 'datasets' / 'Dual-Modal-1504-500-1-6ch'
        self.save_dir = root / 'dual_yolo' / 'evaluation_results_v4' / f'conf_{conf_medical}' / model_name
        self.vis_dir = self.save_dir / 'visualizations'  # 移除GPU子文件夹
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
            'raw_samples': []
        } for cls_id in self.classes}

        self.overall_success_count = 0

        self.model = None
        self.device = f"cuda:{gpu_id}"

    def load_model(self):
        """加载模型"""
        # 特殊处理: 提取基础架构名称
        if self.model_name == 'crossattn-30epoch':
            yaml_path = self.model_yaml.parent / 'yolo11x-dseg-crossattn.yaml'
        elif self.model_name.startswith('id-blue-') and self.model_name[8:].isdigit():
            # id-blue-数字 系列使用 id-blue 架构
            yaml_path = self.model_yaml.parent / 'yolo11x-dseg-id-blue.yaml'
        elif self.model_name.startswith('id-white-') and self.model_name[9:].isdigit():
            # id-white-数字 系列使用 id-white 架构
            yaml_path = self.model_yaml.parent / 'yolo11x-dseg-id-white.yaml'
        else:
            yaml_path = self.model_yaml

        # 加载模型（在CUDA_VISIBLE_DEVICES隔离环境下，模型会自动加载到逻辑GPU 0）
        self.model = YOLO(yaml_path).load(self.model_pt)
        print(f"✅ GPU {self.gpu_id}: 模型加载完成 (架构={yaml_path.name})")

    def load_gt(self, label_file):
        """加载GT标签"""
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
        # 加载图像和标签
        img_path = self.dataset / 'test' / 'images' / npy_file
        label_path = self.dataset / 'test' / 'labels' / npy_file.replace('.npy', '.txt')

        img_data = np.load(img_path)
        img_data = img_data.transpose(2, 0, 1) if img_data.shape[-1] == 6 else img_data
        img_tensor = torch.from_numpy(img_data / 255.0).unsqueeze(0).float()

        # 获取旋转角度
        suffix = npy_file.split('_')[-1].replace('.npy', '')
        angle = self.config.strategies.get(suffix, {}).get('rotation', 0)

        # 加载GT
        gt_masks = self.load_gt(Path(label_path))
        if not gt_masks:
            return

        # 推理两次: 学术 (conf=0.001) + 医学 (conf=conf_medical)
        # ✅ 传入device参数让YOLO自动处理设备分配
        res_academic = self.model(img_tensor, imgsz=1504, device=self.device, conf=self.conf_academic, verbose=False)[0]
        res_medical = self.model(img_tensor, imgsz=1504, device=self.device, conf=self.conf_medical, verbose=False)[0]

        # 收集指标
        self._collect_academic(res_academic, gt_masks)
        class_success = self._collect_medical(res_medical, gt_masks, angle, npy_file)

        # 判断整体成功
        if len(class_success) == len(gt_masks) and all(class_success.values()):
            self.overall_success_count += 1

        # 可视化
        self._visualize(npy_file, img_data, res_medical, gt_masks, class_success)

    def _collect_academic(self, result, gt_masks):
        """收集学术指标数据"""
        gt_classes = np.array(list(gt_masks.keys()))

        if not hasattr(result, 'boxes') or result.boxes is None or len(result.boxes) == 0:
            self.academic_data['tp'].append(np.zeros((0, 10)))
            self.academic_data['conf'].append(np.array([]))
            self.academic_data['pred_cls'].append(np.array([]))
            self.academic_data['target_cls'].append(gt_classes)
            return

        n_pred = len(result.boxes)
        pred_confs = result.boxes.conf.cpu().numpy()
        pred_classes = result.boxes.cls.cpu().numpy().astype(int)

        pred_masks = []
        for i in range(n_pred):
            pts = result[i].masks.xyn[0].copy() * 1504
            pred_masks.append(pts)

        gt_masks_list = [gt_masks[cls_id] for cls_id in gt_classes]

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
        """收集医学指标数据"""
        detected_classes = {}
        if hasattr(result, 'boxes') and result.boxes is not None:
            for i, cls_id in enumerate(result.boxes.cls.cpu().numpy().astype(int)):
                detected_classes.setdefault(cls_id, []).append(i)

        class_success = {}

        for cls_id, gt_pts in gt_masks.items():
            metrics = self.medical_data[cls_id]
            metrics['total_count'] += 1

            # 检查是否恰好检测1次
            detections = detected_classes.get(cls_id, [])
            if len(detections) != 1:
                class_success[cls_id] = False
                continue

            # 提取预测并计算IoU
            pred_pts = result[detections[0]].masks.xyn[0].copy() * 1504
            iou = self.calc_iou(pred_pts, gt_pts)
            if iou < 0.5:
                class_success[cls_id] = False
                continue

            # 计算表面差异并记录
            upper_diff, lower_diff = self.calc_surface_diff(pred_pts, gt_pts, angle)
            metrics['detected_count'] += 1
            metrics['iou_list'].append(iou)
            metrics['upper_diff_list'].append(upper_diff)
            metrics['lower_diff_list'].append(lower_diff)
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
        blue_img = img_data[:3][[2, 1, 0], :, :]
        vis_img = blue_img.transpose(1, 2, 0)
        vis_img = np.clip(vis_img * 255 if vis_img.max() <= 1.0 else vis_img, 0, 255).astype(np.uint8)
        vis_img = np.ascontiguousarray(vis_img)

        colors = {0: (0, 255, 255), 1: (0, 255, 0), 2: (255, 0, 0)}

        for cls_id, gt_pts in gt_masks.items():
            for pt in gt_pts:
                cv2.circle(vis_img, tuple(map(int, pt)), 5, (0, 0, 255), -1)

        if hasattr(result, 'boxes') and result.boxes is not None:
            for i, cls_id in enumerate(result.boxes.cls.cpu().numpy().astype(int)):
                if cls_id in colors:
                    pred_pts = result[i].masks.xyn[0].copy() * 1504
                    pred_pts_int = pred_pts.astype(np.int32).reshape((-1, 1, 2))

                    cv2.polylines(vis_img, [pred_pts_int], True, (255, 255, 255), 1, cv2.LINE_AA)

                    for pt in pred_pts:
                        x, y = int(pt[0]), int(pt[1])
                        cv2.line(vis_img, (x-3, y), (x+3, y), colors[cls_id], 2)
                        cv2.line(vis_img, (x, y-3), (x, y+3), colors[cls_id], 2)

        overall_success = len(class_success) == len(gt_masks) and all(class_success.values())
        suffix = '_success.jpg' if overall_success else '_fail.jpg'
        save_path = self.vis_dir / f"{npy_file.replace('.npy', '')}{suffix}"
        cv2.imwrite(str(save_path), vis_img)

    def get_results_dict(self):
        """返回当前GPU的原始评估数据（用于跨GPU聚合）"""
        return {
            'academic_data': self.academic_data,
            'medical_data': self.medical_data,
            'overall_success_count': self.overall_success_count
        }


def worker_process(gpu_id, model_name, train_mode, conf_medical, npy_files_subset):
    """单个GPU的工作进程"""
    import os

    print(f"[进程 {os.getpid()}] 启动，使用物理GPU {gpu_id}")

    # 创建评估器并加载模型
    evaluator = EvaluatorV4(model_name, train_mode, conf_medical, gpu_id)
    # 保持device为cuda:{gpu_id}，直接使用物理GPU编号
    evaluator.load_model()

    print(f"[GPU {gpu_id}] 开始处理 {len(npy_files_subset)} 张图像...")

    # 逐张评估
    for npy_file in tqdm(npy_files_subset, desc=f"GPU {gpu_id}", position=gpu_id):
        evaluator.eval_image(npy_file)

    torch.cuda.empty_cache()
    return evaluator.get_results_dict()


def aggregate_results(gpu_results, model_name, conf_medical, classes):
    """聚合多个GPU的评估结果"""
    # 聚合学术指标
    all_tp, all_conf, all_pred_cls, all_target_cls = [], [], [], []
    for res in gpu_results:
        acad = res['academic_data']
        all_tp.extend(acad['tp'])
        all_conf.extend(acad['conf'])
        all_pred_cls.extend(acad['pred_cls'])
        all_target_cls.extend(acad['target_cls'])

    # 聚合医学指标
    agg_medical = {cls_id: {
        'detected_count': 0, 'total_count': 0,
        'iou_list': [], 'upper_diff_list': [], 'lower_diff_list': [], 'raw_samples': []
    } for cls_id in classes}

    overall_success_count = sum(res['overall_success_count'] for res in gpu_results)

    for res in gpu_results:
        for cls_id in classes:
            src = res['medical_data'][cls_id]
            dst = agg_medical[cls_id]
            dst['detected_count'] += src['detected_count']
            dst['total_count'] += src['total_count']
            dst['iou_list'].extend(src['iou_list'])
            dst['upper_diff_list'].extend(src['upper_diff_list'])
            dst['lower_diff_list'].extend(src['lower_diff_list'])
            dst['raw_samples'].extend(src['raw_samples'])

    # 计算学术指标
    tp = np.vstack(all_tp)
    conf = np.concatenate(all_conf)
    pred_cls = np.concatenate(all_pred_cls)
    target_cls = np.concatenate(all_target_cls)

    _, _, p, r, f1, ap, unique_cls, *_ = ap_per_class(
        tp, conf, pred_cls, target_cls, plot=False, save_dir=Path(), names={}
    )

    # 组织结果
    results = {
        'model': model_name,
        'conf_academic': 0.001,
        'conf_medical': conf_medical,
        'per_class_metrics': {},
        'aggregate_metrics': {}
    }

    for i, cls_id in enumerate(unique_cls):
        cls_id = int(cls_id)
        cls_name = classes[cls_id]

        academic = {
            'AP50': float(ap[i, 0]),
            'AP75': float(ap[i, 5]),
            'AP50_95': float(ap[i].mean()),
            'Precision_IoU0.5': float(p[i]),
            'Recall_IoU0.5': float(r[i]),
            'F1_IoU0.5': float(f1[i])
        }

        med = agg_medical[cls_id]
        medical = {
            'Detection_Rate_IoU0.5_conf': med['detected_count'] / med['total_count'] if med['total_count'] else 0,
            'IoU_mean_IoU0.5_conf': float(np.mean(med['iou_list'])) if med['iou_list'] else 0,
            'IoU_std_IoU0.5_conf': float(np.std(med['iou_list'])) if med['iou_list'] else 0,
            'Upper_Diff_mean_IoU0.5_conf': float(np.mean(med['upper_diff_list'])) if med['upper_diff_list'] else 0,
            'Upper_Diff_std_IoU0.5_conf': float(np.std(med['upper_diff_list'])) if med['upper_diff_list'] else 0,
            'Lower_Diff_mean_IoU0.5_conf': float(np.mean(med['lower_diff_list'])) if med['lower_diff_list'] else 0,
            'Lower_Diff_std_IoU0.5_conf': float(np.std(med['lower_diff_list'])) if med['lower_diff_list'] else 0,
            'raw_samples': med['raw_samples']
        }

        results['per_class_metrics'][f'class_{cls_id}_{cls_name}'] = {
            'academic': academic,
            'medical': medical
        }

    # 总体指标
    all_academic = [v['academic'] for v in results['per_class_metrics'].values()]
    all_medical = [v['medical'] for v in results['per_class_metrics'].values()]
    total_images = agg_medical[0]['total_count']

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
            'Overall_Success_Rate': float(overall_success_count / total_images) if total_images > 0 else 0.0,
            'Overall_Success_Count': int(overall_success_count),
            'Total_Images': int(total_images)
        }
    }

    return results


def run_parallel_evaluation(model_name, train_mode='pretrained', conf_medical=0.5, num_gpus=4):
    """
    并行评估主函数
    Args:
        model_name: 模型名称
        train_mode: 训练模式
        conf_medical: 医学指标置信度阈值
        num_gpus: 使用的GPU数量
    """
    root = Path(__file__).parent.parent
    dataset = root / 'datasets' / 'Dual-Modal-1504-500-1-6ch'
    save_dir = root / 'dual_yolo' / 'evaluation_results_v4' / f'conf_{conf_medical}' / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config = DatasetConfig()
    classes = {0: 'serum', 1: 'buffy_coat', 2: 'plasma'}

    # 获取测试文件
    test_dir = dataset / 'test' / 'images'
    npy_files = sorted([f.name for f in test_dir.glob('*.npy')
                       if f.stem.split('_')[-1] in config.strategies])

    print(f"\n{'='*80}")
    print(f"  多GPU并行评估: {model_name}")
    print(f"  总图像数: {len(npy_files)}")
    print(f"  使用GPU数: {num_gpus}")
    print(f"  每GPU处理: ~{len(npy_files)//num_gpus} 张")
    print(f"{'='*80}\n")

    # 图像分片
    chunk_size = len(npy_files) // num_gpus
    file_chunks = [
        npy_files[i*chunk_size:(i+1)*chunk_size if i < num_gpus-1 else len(npy_files)]
        for i in range(num_gpus)
    ]

    # 启动多进程
    start_time = time.time()

    with mp.Pool(processes=num_gpus) as pool:
        # 构造完整参数列表: (gpu_id, model_name, train_mode, conf_medical, npy_files_subset)
        worker_args = [
            (i, model_name, train_mode, conf_medical, file_chunks[i])
            for i in range(num_gpus)
        ]

        gpu_results = pool.starmap(worker_process, worker_args)

    elapsed_time = time.time() - start_time

    # 聚合结果
    print("\n正在聚合多GPU结果...")
    results = aggregate_results(gpu_results, model_name, conf_medical, classes)

    # 保存结果
    save_path = save_dir / f'metrics_{model_name}.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ 结果保存: {save_path}")

    # 打印结果
    print(f"\n{'='*80}")
    print(f"  {model_name} 评估结果")
    print(f"  学术conf=0.001, 医学conf={conf_medical}")
    print(f"  总耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)")
    print(f"  平均每张: {elapsed_time/len(npy_files):.3f}秒")
    print(f"{'='*80}\n")

    for key, metrics in results['per_class_metrics'].items():
        cls_name = key.split('_')[-1]
        print(f"【{cls_name}】")
        print(f"  学术: AP50={metrics['academic']['AP50']:.4f} "
              f"AP50-95={metrics['academic']['AP50_95']:.4f} "
              f"Recall={metrics['academic']['Recall_IoU0.5']:.4f}")
        print(f"  医学: DR={metrics['medical']['Detection_Rate_IoU0.5_conf']:.2%} "
              f"IoU={metrics['medical']['IoU_mean_IoU0.5_conf']:.4f}±{metrics['medical']['IoU_std_IoU0.5_conf']:.4f} "
              f"UD={metrics['medical']['Upper_Diff_mean_IoU0.5_conf']:.1f}±{metrics['medical']['Upper_Diff_std_IoU0.5_conf']:.1f}")

    agg = results['aggregate_metrics']
    print(f"\n【总体】")
    print(f"  学术: mAP50={agg['academic']['mAP50']:.4f} "
          f"mAP50-95={agg['academic']['mAP50_95']:.4f} "
          f"Recall={agg['academic']['mean_Recall']:.4f}")
    print(f"  医学: DR={agg['medical']['mean_Detection_Rate']:.2%} "
          f"IoU={agg['medical']['mean_IoU']:.4f}")
    print(f"  整体成功率: {agg['medical']['Overall_Success_Rate']:.2%} "
          f"({agg['medical']['Overall_Success_Count']}/{agg['medical']['Total_Images']})")
    print(f"{'='*80}\n")


def main():
    """批量评估入口"""
    models = ['id-blue-2', 'id-white-2', 'id-blue-3', 'id-white-3']
    # models = ['id-blue', 'id-white', 'crossattn', 'crossattn-precise', 'weighted-fusion', 'concat-compress']
    conf_thresholds = [0.25, 0.3, 0.4, 0.5]
    train_mode = 'pretrained'
    num_gpus = 4  # 使用4张3090

    for model in models:
        for conf in conf_thresholds:
            run_parallel_evaluation(model, train_mode, conf, num_gpus)


if __name__ == '__main__':
    # 设置multiprocessing启动方式为spawn（PyTorch + CUDA必需）
    mp.set_start_method('spawn', force=True)
    main()