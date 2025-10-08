"""
读取evaluation_results_v5文件夹中的结果数据并在终端打印表格
兼容V5优化推理版本的结果
"""

import json
import numpy as np
from pathlib import Path


class ResultsTableV5:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)

        # 固定方法顺序
        self.base_method_order = [
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

    def get_display_name(self, method):
        """获取方法的显示名称"""
        if method in self.method_names:
            return self.method_names[method]

        if method.startswith('id-blue-') and method[8:].isdigit():
            epoch = method[8:]
            return f'Yolo11-Blue-{epoch}'

        if method.startswith('id-white-') and method[9:].isdigit():
            epoch = method[9:]
            return f'Yolo11-White-{epoch}'

        return method

    def discover_methods(self, conf):
        """自动发现指定conf下的所有模型"""
        conf_dir = self.results_dir / f'conf_{conf}'
        if not conf_dir.exists():
            return []

        all_methods = []

        for method_dir in conf_dir.iterdir():
            if method_dir.is_dir():
                method = method_dir.name
                if (method_dir / f'metrics_{method}.json').exists():
                    all_methods.append(method)

        # 排序
        sorted_methods = []

        for base in self.base_method_order:
            if base in all_methods:
                sorted_methods.append(base)

            variants = []

            if base == 'id-blue':
                variants = [m for m in all_methods if m.startswith('id-blue-') and m[8:].isdigit()]
                variants.sort(key=lambda x: int(x[8:]))
            elif base == 'id-white':
                variants = [m for m in all_methods if m.startswith('id-white-') and m[9:].isdigit()]
                variants.sort(key=lambda x: int(x[9:]))

            sorted_methods.extend(variants)

        remaining = [m for m in all_methods if m not in sorted_methods]
        sorted_methods.extend(sorted(remaining))

        return sorted_methods

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

        sp_ap5095 = (serum_ac.get('AP50_95', 0) + plasma_ac.get('AP50_95', 0)) / 2 * 100
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

        bc_ap5095 = buffy_ac.get('AP50_95', 0) * 100
        bc_ap50 = buffy_ac.get('AP50', 0) * 100
        bc_recall = buffy_ac.get('Recall_IoU0.5', 0) * 100
        bc_iou_mean = buffy_med.get('IoU_mean_IoU0.5_conf', 0)
        bc_iou_std = buffy_med.get('IoU_std_IoU0.5_conf', 0)
        bc_diff_up = buffy_med.get('Upper_Diff_mean_IoU0.5_conf', 0)
        bc_diff_low = buffy_med.get('Lower_Diff_mean_IoU0.5_conf', 0)

        return {
            'sp_ap5095': sp_ap5095,
            'sp_ap50': sp_ap50,
            'sp_recall': sp_recall,
            'sp_iou': f"{sp_iou_mean:.2f}±{sp_iou_std:.2f}",
            'sp_diff_up': f"{sp_diff_up:.1f}",
            'sp_diff_low': f"{sp_diff_low:.1f}",
            'bc_ap5095': bc_ap5095,
            'bc_ap50': bc_ap50,
            'bc_recall': bc_recall,
            'bc_iou': f"{bc_iou_mean:.2f}±{bc_iou_std:.2f}",
            'bc_diff_up': f"{bc_diff_up:.1f}",
            'bc_diff_low': f"{bc_diff_low:.1f}"
        }

    def print_table(self, conf):
        """打印指定conf的表格"""
        methods = self.discover_methods(conf)

        if not methods:
            print(f"\n❌ conf={conf} 未找到任何模型结果")
            return

        # 列宽定义
        col_method = 30
        col_num = 10
        col_iou = 13

        print(f"\n{'='*175}")
        print(f"Evaluation Results V5 - conf={conf} (Academic: conf=0.001, Medical: conf={conf})")
        print(f"Found {len(methods)} models")
        print(f"{'='*175}")

        # 表头 - AP50-95优先，AP50第二
        header = (
            f"{'Method':<{col_method}} | "
            f"{'SP_AP5095':<{col_num}} {'SP_AP50':<{col_num}} {'SP_Recall':<{col_num}} {'SP_IoU':<{col_iou}} "
            f"{'SP_DiffUp':<{col_num}} {'SP_DiffLow':<{col_num}} | "
            f"{'BC_AP5095':<{col_num}} {'BC_AP50':<{col_num}} {'BC_Recall':<{col_num}} {'BC_IoU':<{col_iou}} "
            f"{'BC_DiffUp':<{col_num}} {'BC_DiffLow':<{col_num}}"
        )
        print(header)
        print("-" * 175)

        # 打印每个方法
        for method in methods:
            metrics = self.load_metrics(method, conf)
            if not metrics:
                continue

            data = self.extract_metrics(metrics)
            display_name = self.get_display_name(method)

            row = (
                f"{display_name:<{col_method}} | "
                f"{data['sp_ap5095']:<{col_num}.2f} {data['sp_ap50']:<{col_num}.2f} {data['sp_recall']:<{col_num}.2f} {data['sp_iou']:<{col_iou}} "
                f"{data['sp_diff_up']:<{col_num}} {data['sp_diff_low']:<{col_num}} | "
                f"{data['bc_ap5095']:<{col_num}.2f} {data['bc_ap50']:<{col_num}.2f} {data['bc_recall']:<{col_num}.2f} {data['bc_iou']:<{col_iou}} "
                f"{data['bc_diff_up']:<{col_num}} {data['bc_diff_low']:<{col_num}}"
            )
            print(row)

        print("=" * 175)

    def run(self):
        """运行所有conf的表格打印"""
        conf_dirs = [d for d in self.results_dir.glob('conf_*') if d.is_dir()]
        conf_values = sorted([float(d.name.replace('conf_', '')) for d in conf_dirs])

        if not conf_values:
            print("❌ 未找到任何conf结果文件夹")
            return

        print(f"\n找到 {len(conf_values)} 个conf阈值: {conf_values}")

        for conf in conf_values:
            self.print_table(str(conf))


def main():
    results_dir = Path(__file__).parent / 'evaluation_results_v5_novis'

    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return

    generator = ResultsTableV5(results_dir)
    generator.run()


if __name__ == '__main__':
    main()