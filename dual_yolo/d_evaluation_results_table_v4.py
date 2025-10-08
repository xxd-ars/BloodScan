"""
读取evaluation_results_v4_novis文件夹中的结果数据并在终端打印表格
简化版：只打印表格，不生成文件
"""

import json
import numpy as np
from pathlib import Path


class ResultsTableV4:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)

        # 固定方法顺序
        self.method_order = [
            'id-blue',
            'id-white',
            'concat-compress',
            'weighted-fusion',
            'crossattn',
            'crossattn-precise'
        ]

        # 方法名映射
        self.method_names = {
            'id-blue': 'Yolo11-Blue',
            'id-white': 'Yolo11-White',
            'concat-compress': 'Dual Yolo Concat',
            'weighted-fusion': 'Dual Yolo Weighted',
            'crossattn': 'Dual Yolo CrossAttn',
            'crossattn-precise': 'Dual Yolo CrossAttn Precise'
        }

    def load_metrics(self, method, conf):
        """加载单个方法在指定conf下的metrics"""
        metrics_file = self.results_dir / f'conf_{conf}' / method / f'metrics_{method}.json'
        if not metrics_file.exists():
            return None

        with open(metrics_file) as f:
            return json.load(f)

    def extract_metrics(self, metrics):
        """提取关键指标"""
        per_class = metrics.get('per_class_metrics', {})

        # Serum (class 0) + Plasma (class 2) 平均
        serum = per_class.get('class_0_serum', {})
        plasma = per_class.get('class_2_plasma', {})

        serum_ac = serum.get('academic', {})
        plasma_ac = plasma.get('academic', {})
        serum_med = serum.get('medical', {})
        plasma_med = plasma.get('medical', {})

        sp_ap50 = (serum_ac.get('AP50', 0) + plasma_ac.get('AP50', 0)) / 2 * 100
        sp_recall = (serum_ac.get('Recall_IoU0.5', 0) + plasma_ac.get('Recall_IoU0.5', 0)) / 2 * 100
        sp_iou_mean = (serum_med.get('IoU_mean_IoU0.5_conf', 0) + plasma_med.get('IoU_mean_IoU0.5_conf', 0)) / 2
        sp_iou_std = np.sqrt((serum_med.get('IoU_std_IoU0.5_conf', 0)**2 + plasma_med.get('IoU_std_IoU0.5_conf', 0)**2) / 2)
        sp_diff_up = (serum_med.get('Upper_Diff_mean_IoU0.5_conf', 0) + plasma_med.get('Upper_Diff_mean_IoU0.5_conf', 0)) / 2
        sp_diff_low = (serum_med.get('Lower_Diff_mean_IoU0.5_conf', 0) + plasma_med.get('Lower_Diff_mean_IoU0.5_conf', 0)) / 2

        # Buffy Coat (class 1)
        buffy = per_class.get('class_1_buffy_coat', {})
        buffy_ac = buffy.get('academic', {})
        buffy_med = buffy.get('medical', {})

        bc_ap50 = buffy_ac.get('AP50', 0) * 100
        bc_recall = buffy_ac.get('Recall_IoU0.5', 0) * 100
        bc_iou_mean = buffy_med.get('IoU_mean_IoU0.5_conf', 0)
        bc_iou_std = buffy_med.get('IoU_std_IoU0.5_conf', 0)
        bc_diff_up = buffy_med.get('Upper_Diff_mean_IoU0.5_conf', 0)
        bc_diff_low = buffy_med.get('Lower_Diff_mean_IoU0.5_conf', 0)

        return {
            'sp_ap50': sp_ap50,
            'sp_recall': sp_recall,
            'sp_iou': f"{sp_iou_mean:.2f}±{sp_iou_std:.2f}",
            'sp_diff_up': f"{sp_diff_up:.1f}",
            'sp_diff_low': f"{sp_diff_low:.1f}",
            'bc_ap50': bc_ap50,
            'bc_recall': bc_recall,
            'bc_iou': f"{bc_iou_mean:.2f}±{bc_iou_std:.2f}",
            'bc_diff_up': f"{bc_diff_up:.1f}",
            'bc_diff_low': f"{bc_diff_low:.1f}"
        }

    def print_table(self, conf):
        """打印指定conf的表格"""
        # 列宽定义
        col_method = 30
        col_num = 10
        col_iou = 13

        print(f"\n{'='*155}")
        print(f"Evaluation Results - conf={conf} (Academic: conf=0.001, Medical: conf={conf})")
        print(f"{'='*155}")

        # 表头 - 所有列左对齐
        header = (
            f"{'Method':<{col_method}} | "
            f"{'SP_AP50':<{col_num}} {'SP_Recall':<{col_num}} {'SP_IoU':<{col_iou}} "
            f"{'SP_DiffUp':<{col_num}} {'SP_DiffLow':<{col_num}} | "
            f"{'BC_AP50':<{col_num}} {'BC_Recall':<{col_num}} {'BC_IoU':<{col_iou}} "
            f"{'BC_DiffUp':<{col_num}} {'BC_DiffLow':<{col_num}}"
        )
        print(header)
        print("-" * 155)

        # 按固定顺序打印每个方法 - 所有列左对齐
        for method in self.method_order:
            metrics = self.load_metrics(method, conf)
            if not metrics:
                continue

            data = self.extract_metrics(metrics)
            display_name = self.method_names[method]

            row = (
                f"{display_name:<{col_method}} | "
                f"{data['sp_ap50']:<{col_num}.2f} {data['sp_recall']:<{col_num}.2f} {data['sp_iou']:<{col_iou}} "
                f"{data['sp_diff_up']:<{col_num}} {data['sp_diff_low']:<{col_num}} | "
                f"{data['bc_ap50']:<{col_num}.2f} {data['bc_recall']:<{col_num}.2f} {data['bc_iou']:<{col_iou}} "
                f"{data['bc_diff_up']:<{col_num}} {data['bc_diff_low']:<{col_num}}"
            )
            print(row)

        print("=" * 155)

    def run(self):
        """运行所有conf的表格打印"""
        # 获取所有conf文件夹并按数值排序
        conf_dirs = [d for d in self.results_dir.glob('conf_*') if d.is_dir()]
        conf_values = sorted([float(d.name.replace('conf_', '')) for d in conf_dirs])

        if not conf_values:
            print("❌ 未找到任何conf结果文件夹")
            return

        print(f"\n找到 {len(conf_values)} 个conf阈值: {conf_values}")

        # 依次打印每个conf的表格
        for conf in conf_values:
            self.print_table(str(conf))


def main():
    results_dir = Path(__file__).parent / 'evaluation_results_v4_novis'

    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return

    generator = ResultsTableV4(results_dir)
    generator.run()


if __name__ == '__main__':
    main()
