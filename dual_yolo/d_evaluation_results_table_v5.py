"""
读取evaluation_results_v5文件夹中的结果数据并在终端打印表格
V5版本：分离学术指标和医学指标表格
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

    def print_academic_table(self, conf):
        """打印学术指标表格（不依赖conf，使用conf=0.001的结果）"""
        methods = self.discover_methods(conf)

        if not methods:
            print(f"\n❌ 未找到任何模型结果")
            return

        # 列宽定义
        col_method = 30
        col_num = 8

        print(f"\n{'='*175}")
        print(f"Table 1: Academic Metrics (conf=0.001, IoU@0.5)")
        print(f"Found {len(methods)} models")
        print(f"{'='*175}")

        # 表头
        header = (
            f"{'Method':<{col_method}} | "
            f"{'S_AP5095':<{col_num}} {'S_AP75':<{col_num}} {'S_AP50':<{col_num}} {'S_F1':<{col_num}} {'S_Prec':<{col_num}} {'S_Rec':<{col_num}} | "
            f"{'B_AP5095':<{col_num}} {'B_AP75':<{col_num}} {'B_AP50':<{col_num}} {'B_F1':<{col_num}} {'B_Prec':<{col_num}} {'B_Rec':<{col_num}} | "
            f"{'P_AP5095':<{col_num}} {'P_AP75':<{col_num}} {'P_AP50':<{col_num}} {'P_F1':<{col_num}} {'P_Prec':<{col_num}} {'P_Rec':<{col_num}}"
        )
        print(header)
        print("-" * 175)

        # 打印每个方法
        for method in methods:
            metrics = self.load_metrics(method, conf)
            if not metrics:
                continue

            per_class = metrics.get('per_class_metrics', {})

            # Serum (class 0)
            serum = per_class.get('class_0_serum', {}).get('academic', {})
            s_ap5095 = serum.get('AP50_95', 0) * 100
            s_ap75 = serum.get('AP75', 0) * 100
            s_ap50 = serum.get('AP50', 0) * 100
            s_f1 = serum.get('F1_IoU0.5', 0) * 100
            s_prec = serum.get('Precision_IoU0.5', 0) * 100
            s_rec = serum.get('Recall_IoU0.5', 0) * 100

            # Buffy Coat (class 1)
            buffy = per_class.get('class_1_buffy_coat', {}).get('academic', {})
            b_ap5095 = buffy.get('AP50_95', 0) * 100
            b_ap75 = buffy.get('AP75', 0) * 100
            b_ap50 = buffy.get('AP50', 0) * 100
            b_f1 = buffy.get('F1_IoU0.5', 0) * 100
            b_prec = buffy.get('Precision_IoU0.5', 0) * 100
            b_rec = buffy.get('Recall_IoU0.5', 0) * 100

            # Plasma (class 2)
            plasma = per_class.get('class_2_plasma', {}).get('academic', {})
            p_ap5095 = plasma.get('AP50_95', 0) * 100
            p_ap75 = plasma.get('AP75', 0) * 100
            p_ap50 = plasma.get('AP50', 0) * 100
            p_f1 = plasma.get('F1_IoU0.5', 0) * 100
            p_prec = plasma.get('Precision_IoU0.5', 0) * 100
            p_rec = plasma.get('Recall_IoU0.5', 0) * 100

            display_name = self.get_display_name(method)

            row = (
                f"{display_name:<{col_method}} | "
                f"{s_ap5095:<{col_num}.2f} {s_ap75:<{col_num}.2f} {s_ap50:<{col_num}.2f} {s_f1:<{col_num}.2f} {s_prec:<{col_num}.2f} {s_rec:<{col_num}.2f} | "
                f"{b_ap5095:<{col_num}.2f} {b_ap75:<{col_num}.2f} {b_ap50:<{col_num}.2f} {b_f1:<{col_num}.2f} {b_prec:<{col_num}.2f} {b_rec:<{col_num}.2f} | "
                f"{p_ap5095:<{col_num}.2f} {p_ap75:<{col_num}.2f} {p_ap50:<{col_num}.2f} {p_f1:<{col_num}.2f} {p_prec:<{col_num}.2f} {p_rec:<{col_num}.2f}"
            )
            print(row)

        print("=" * 175)

    def print_medical_table(self, conf):
        """打印医学指标表格（依赖conf阈值）"""
        methods = self.discover_methods(conf)

        if not methods:
            print(f"\n❌ conf={conf} 未找到任何模型结果")
            return

        # 列宽定义
        col_method = 30
        col_dr = 8
        col_iou = 13
        col_diff = 13

        print(f"\n{'='*185}")
        print(f"Table 2: Medical Metrics at conf={conf} (IoU>=0.5)")
        print(f"Found {len(methods)} models")
        print(f"{'='*185}")

        # 表头
        header = (
            f"{'Method':<{col_method}} | "
            f"{'S_DR':<{col_dr}} {'S_IoU':<{col_iou}} {'S_DiffUp':<{col_diff}} {'S_DiffLow':<{col_diff}} | "
            f"{'B_DR':<{col_dr}} {'B_IoU':<{col_iou}} {'B_DiffUp':<{col_diff}} {'B_DiffLow':<{col_diff}} | "
            f"{'P_DR':<{col_dr}} {'P_IoU':<{col_iou}} {'P_DiffUp':<{col_diff}} {'P_DiffLow':<{col_diff}}"
        )
        print(header)
        print("-" * 185)

        # 打印每个方法
        for method in methods:
            metrics = self.load_metrics(method, conf)
            if not metrics:
                continue

            per_class = metrics.get('per_class_metrics', {})

            # Serum (class 0)
            serum_med = per_class.get('class_0_serum', {}).get('medical', {})
            s_dr = serum_med.get('Detection_Rate_IoU0.5_conf', 0) * 100
            s_iou_mean = serum_med.get('IoU_mean_IoU0.5_conf', 0)
            s_iou_std = serum_med.get('IoU_std_IoU0.5_conf', 0)
            s_diff_up_mean = serum_med.get('Upper_Diff_mean_IoU0.5_conf', 0)
            s_diff_up_std = serum_med.get('Upper_Diff_std_IoU0.5_conf', 0)
            s_diff_low_mean = serum_med.get('Lower_Diff_mean_IoU0.5_conf', 0)
            s_diff_low_std = serum_med.get('Lower_Diff_std_IoU0.5_conf', 0)

            # Buffy Coat (class 1)
            buffy_med = per_class.get('class_1_buffy_coat', {}).get('medical', {})
            b_dr = buffy_med.get('Detection_Rate_IoU0.5_conf', 0) * 100
            b_iou_mean = buffy_med.get('IoU_mean_IoU0.5_conf', 0)
            b_iou_std = buffy_med.get('IoU_std_IoU0.5_conf', 0)
            b_diff_up_mean = buffy_med.get('Upper_Diff_mean_IoU0.5_conf', 0)
            b_diff_up_std = buffy_med.get('Upper_Diff_std_IoU0.5_conf', 0)
            b_diff_low_mean = buffy_med.get('Lower_Diff_mean_IoU0.5_conf', 0)
            b_diff_low_std = buffy_med.get('Lower_Diff_std_IoU0.5_conf', 0)

            # Plasma (class 2)
            plasma_med = per_class.get('class_2_plasma', {}).get('medical', {})
            p_dr = plasma_med.get('Detection_Rate_IoU0.5_conf', 0) * 100
            p_iou_mean = plasma_med.get('IoU_mean_IoU0.5_conf', 0)
            p_iou_std = plasma_med.get('IoU_std_IoU0.5_conf', 0)
            p_diff_up_mean = plasma_med.get('Upper_Diff_mean_IoU0.5_conf', 0)
            p_diff_up_std = plasma_med.get('Upper_Diff_std_IoU0.5_conf', 0)
            p_diff_low_mean = plasma_med.get('Lower_Diff_mean_IoU0.5_conf', 0)
            p_diff_low_std = plasma_med.get('Lower_Diff_std_IoU0.5_conf', 0)

            display_name = self.get_display_name(method)

            # 格式化IoU、DiffUp和DiffLow字符串，确保与表头对齐
            s_iou_str = f"{s_iou_mean:.2f}±{s_iou_std:.2f}"
            s_diff_up_str = f"{s_diff_up_mean:.1f}±{s_diff_up_std:.1f}"
            s_diff_low_str = f"{s_diff_low_mean:.1f}±{s_diff_low_std:.1f}"
            b_iou_str = f"{b_iou_mean:.2f}±{b_iou_std:.2f}"
            b_diff_up_str = f"{b_diff_up_mean:.1f}±{b_diff_up_std:.1f}"
            b_diff_low_str = f"{b_diff_low_mean:.1f}±{b_diff_low_std:.1f}"
            p_iou_str = f"{p_iou_mean:.2f}±{p_iou_std:.2f}"
            p_diff_up_str = f"{p_diff_up_mean:.1f}±{p_diff_up_std:.1f}"
            p_diff_low_str = f"{p_diff_low_mean:.1f}±{p_diff_low_std:.1f}"

            row = (
                f"{display_name:<{col_method}} | "
                f"{s_dr:<{col_dr}.2f} {s_iou_str:<{col_iou}} {s_diff_up_str:<{col_diff}} {s_diff_low_str:<{col_diff}} | "
                f"{b_dr:<{col_dr}.2f} {b_iou_str:<{col_iou}} {b_diff_up_str:<{col_diff}} {b_diff_low_str:<{col_diff}} | "
                f"{p_dr:<{col_dr}.2f} {p_iou_str:<{col_iou}} {p_diff_up_str:<{col_diff}} {p_diff_low_str:<{col_diff}}"
            )
            print(row)

        print("=" * 185)

    def run(self):
        """运行所有表格打印"""
        conf_dirs = [d for d in self.results_dir.glob('conf_*') if d.is_dir()]
        conf_values = sorted([float(d.name.replace('conf_', '')) for d in conf_dirs])

        if not conf_values:
            print("❌ 未找到任何conf结果文件夹")
            return

        print(f"\n找到 {len(conf_values)} 个conf阈值: {conf_values}")

        # 打印学术指标表格（只打印一次，使用第一个conf的数据）
        self.print_academic_table(str(conf_values[0]))

        # 为每个conf打印医学指标表格
        for conf in conf_values:
            self.print_medical_table(str(conf))


def main():
    results_dir = Path(__file__).parent / 'evaluation_results'
    # results_dir = Path(__file__).parent / 'evaluation_results_v5_novis'

    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return

    generator = ResultsTableV5(results_dir)
    generator.run()


if __name__ == '__main__':
    main()