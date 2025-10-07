"""
读取evaluation_results_v4文件夹中的结果数据并生成表格可视化
适配V4版本的JSON结构（双推理：学术conf=0.001 + 医学conf=可调）
"""

import json
import numpy as np
from pathlib import Path


class ResultsTableGeneratorV4:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.data = []

    def load_all_results(self):
        """加载所有结果数据"""
        for conf_dir in self.results_dir.glob('conf_*'):
            conf_threshold = conf_dir.name.replace('conf_', '')

            for method_dir in conf_dir.iterdir():
                if method_dir.is_dir():
                    method_name = method_dir.name
                    metrics_file = method_dir / f'metrics_{method_name}.json'

                    if metrics_file.exists():
                        with open(metrics_file, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)

                        self.data.append({
                            'method': method_name,
                            'conf': conf_threshold,
                            'metrics': metrics
                        })

    def extract_class_metrics(self, per_class_metrics, class_key):
        """从per_class_metrics中提取指定类别的数据（V4格式）"""
        class_data = per_class_metrics.get(class_key, {})
        academic = class_data.get('academic', {})
        medical = class_data.get('medical', {})

        return {
            # 学术指标
            'ap50': academic.get('AP50', 0) * 100,
            'ap50_95': academic.get('AP50_95', 0) * 100,
            'recall': academic.get('Recall_IoU0.5', 0) * 100,
            'precision': academic.get('Precision_IoU0.5', 0) * 100,
            # 医学指标
            'detection_rate': medical.get('Detection_Rate_IoU0.5_conf', 0) * 100,
            'iou_mean': medical.get('IoU_mean_IoU0.5_conf', 0),
            'iou_std': medical.get('IoU_std_IoU0.5_conf', 0),
            'upper_diff_mean': medical.get('Upper_Diff_mean_IoU0.5_conf', 0),
            'upper_diff_std': medical.get('Upper_Diff_std_IoU0.5_conf', 0),
            'lower_diff_mean': medical.get('Lower_Diff_mean_IoU0.5_conf', 0),
            'lower_diff_std': medical.get('Lower_Diff_std_IoU0.5_conf', 0)
        }

    def format_value_with_std(self, mean, std, decimals=2, format_type='latex'):
        """格式化均值±标准差"""
        if std == 0 or std < 0.01:
            return f"{mean:.{decimals}f}"
        if format_type == 'latex':
            return f"{mean:.{decimals}f}$\\pm${std:.{decimals}f}"
        else:
            return f"{mean:.{decimals}f}±{std:.{decimals}f}"

    def generate_table(self, conf_threshold='0.3'):
        """生成指定置信度的结果表格"""
        conf_data = [d for d in self.data if d['conf'] == conf_threshold]

        if not conf_data:
            print(f"未找到conf={conf_threshold}的数据")
            return

        # 方法名映射
        method_names = {
            'id': 'Yolo11',
            'id-blue': 'Yolo11-Blue',
            'id-white': 'Yolo11-White',
            'id-blue-30': 'Yolo11-Blue-30',
            'id-white-30': 'Yolo11-White-30',
            'concat-compress': 'Dual Yolo Concat',
            'weighted-fusion': 'Dual Yolo Weighted',
            'crossattn': 'Dual Yolo CrossAttn',
            'crossattn-30epoch': 'Dual Yolo CrossAttn (30 Epochs)',
            'crossattn-precise': 'Dual Yolo (Our Best)'
        }

        # 收集表格数据
        table_data = []
        for item in conf_data:
            method = item['method']
            metrics = item['metrics']
            per_class = metrics.get('per_class_metrics', {})

            # 提取Serum/Plasma (合并class 0和2)
            serum_metrics = self.extract_class_metrics(per_class, 'class_0_serum')
            plasma_metrics = self.extract_class_metrics(per_class, 'class_2_plasma')

            # 合并serum和plasma的指标（取平均或RMS）
            sp_ap50 = (serum_metrics['ap50'] + plasma_metrics['ap50']) / 2
            sp_ap50_95 = (serum_metrics['ap50_95'] + plasma_metrics['ap50_95']) / 2
            sp_recall = (serum_metrics['recall'] + plasma_metrics['recall']) / 2
            sp_iou = self.format_value_with_std(
                (serum_metrics['iou_mean'] + plasma_metrics['iou_mean']) / 2,
                np.sqrt((serum_metrics['iou_std']**2 + plasma_metrics['iou_std']**2) / 2)
            )
            sp_diff_up = self.format_value_with_std(
                (serum_metrics['upper_diff_mean'] + plasma_metrics['upper_diff_mean']) / 2,
                np.sqrt((serum_metrics['upper_diff_std']**2 + plasma_metrics['upper_diff_std']**2) / 2),
                decimals=1
            )
            sp_diff_low = self.format_value_with_std(
                (serum_metrics['lower_diff_mean'] + plasma_metrics['lower_diff_mean']) / 2,
                np.sqrt((serum_metrics['lower_diff_std']**2 + plasma_metrics['lower_diff_std']**2) / 2),
                decimals=1
            )

            # 提取Buffy Coat (class 1)
            bc_metrics = self.extract_class_metrics(per_class, 'class_1_buffy_coat')
            bc_ap50 = bc_metrics['ap50']
            bc_ap50_95 = bc_metrics['ap50_95']
            bc_recall = bc_metrics['recall']
            bc_iou = self.format_value_with_std(bc_metrics['iou_mean'], bc_metrics['iou_std'])
            bc_diff_up = self.format_value_with_std(bc_metrics['upper_diff_mean'], bc_metrics['upper_diff_std'], decimals=1)
            bc_diff_low = self.format_value_with_std(bc_metrics['lower_diff_mean'], bc_metrics['lower_diff_std'], decimals=1)

            table_data.append({
                'Method': method_names.get(method, method),
                'Dual': '$\\checkmark$' if 'dual' in method.lower() or method in ['concat-compress', 'weighted-fusion', 'crossattn', 'crossattn-30epoch', 'crossattn-precise'] else '$\\times$',
                'SP_AP50': f"{sp_ap50:.2f}",
                'SP_AP50_95': f"{sp_ap50_95:.2f}",
                'SP_Recall': f"{sp_recall:.2f}",
                'SP_IOU': sp_iou,
                'SP_Diff_up': sp_diff_up,
                'SP_Diff_low': sp_diff_low,
                'BC_AP50': f"{bc_ap50:.2f}",
                'BC_AP50_95': f"{bc_ap50_95:.2f}",
                'BC_Recall': f"{bc_recall:.2f}",
                'BC_IOU': bc_iou,
                'BC_Diff_up': bc_diff_up,
                'BC_Diff_low': bc_diff_low
            })

        # 找出最佳和次佳值
        numeric_cols = ['SP_AP50', 'SP_AP50_95', 'SP_Recall', 'BC_AP50', 'BC_AP50_95', 'BC_Recall']
        best_values = {}
        second_best_values = {}

        for col in numeric_cols:
            values = [float(row[col]) for row in table_data]
            sorted_vals = sorted(values, reverse=True)
            if len(sorted_vals) >= 2:
                best_values[col] = sorted_vals[0]
                second_best_values[col] = sorted_vals[1]

        # 生成LaTeX表格
        latex_lines = self._generate_latex(table_data, best_values, second_best_values, conf_threshold)

        # 生成Markdown表格
        md_lines = self._generate_markdown(table_data, best_values, second_best_values, conf_threshold)

        # 保存文件
        save_dir = self.results_dir / f'conf_{conf_threshold}'
        save_dir.mkdir(exist_ok=True)

        latex_file = save_dir / f'results_table_conf_{conf_threshold}.tex'
        md_file = save_dir / f'results_table_conf_{conf_threshold}.md'

        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(latex_lines))

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(md_lines))

        print(f"✅ LaTeX表格: {latex_file}")
        print(f"✅ Markdown表格: {md_file}")

    def _generate_latex(self, table_data, best_vals, second_best_vals, conf):
        """生成LaTeX表格"""
        lines = [f"% Evaluation Results V4 (Academic conf=0.001, Medical conf={conf})"]

        for row in table_data:
            values = []
            values.append(row['Method'])
            values.append(row['Dual'])

            # Serum/Plasma
            for key in ['SP_AP50', 'SP_AP50_95', 'SP_Recall']:
                val = float(row[key])
                formatted = row[key]
                if key in best_vals and abs(val - best_vals[key]) < 0.01:
                    formatted = f"\\textbf{{{formatted}}}"
                elif key in second_best_vals and abs(val - second_best_vals[key]) < 0.01:
                    formatted = f"\\underline{{{formatted}}}"
                values.append(formatted)

            values.extend([row['SP_IOU'], row['SP_Diff_up'], row['SP_Diff_low']])

            # Buffy Coat
            for key in ['BC_AP50', 'BC_AP50_95', 'BC_Recall']:
                val = float(row[key])
                formatted = row[key]
                if key in best_vals and abs(val - best_vals[key]) < 0.01:
                    formatted = f"\\textbf{{{formatted}}}"
                elif key in second_best_vals and abs(val - second_best_vals[key]) < 0.01:
                    formatted = f"\\underline{{{formatted}}}"
                values.append(formatted)

            values.extend([row['BC_IOU'], row['BC_Diff_up'], row['BC_Diff_low']])

            # 特殊处理最后一行
            if row['Method'] == 'Dual Yolo (Our Best)':
                lines.append('\\hline')

            line = ' & '.join(values) + '\\\\'
            lines.append(line)

            if row['Method'] == 'Dual Yolo (Our Best)':
                lines.append('\\hline')

        return lines

    def _generate_markdown(self, table_data, best_vals, second_best_vals, conf):
        """生成Markdown表格"""
        lines = [
            f"# Evaluation Results V4 (Academic conf=0.001, Medical conf={conf})",
            "",
            "| Method | Dual | Serum/Plasma | | | | | | Buffy Coat | | | | | |",
            "|--------|------|--------------|---|---|---|---|---|------------|---|---|---|---|---|",
            "| | | AP50 | AP50-95 | Recall | IOU | Diff_up | Diff_low | AP50 | AP50-95 | Recall | IOU | Diff_up | Diff_low |"
        ]

        for row in table_data:
            cells = [row['Method']]
            cells.append('✓' if '$\\checkmark$' in row['Dual'] else '✗')

            # Serum/Plasma
            for key in ['SP_AP50', 'SP_AP50_95', 'SP_Recall']:
                val = float(row[key])
                formatted = row[key]
                if key in best_vals and abs(val - best_vals[key]) < 0.01:
                    formatted = f"**{formatted}**"
                elif key in second_best_vals and abs(val - second_best_vals[key]) < 0.01:
                    formatted = f"_{formatted}_"
                cells.append(formatted)

            cells.extend([row['SP_IOU'], row['SP_Diff_up'], row['SP_Diff_low']])

            # Buffy Coat
            for key in ['BC_AP50', 'BC_AP50_95', 'BC_Recall']:
                val = float(row[key])
                formatted = row[key]
                if key in best_vals and abs(val - best_vals[key]) < 0.01:
                    formatted = f"**{formatted}**"
                elif key in second_best_vals and abs(val - second_best_vals[key]) < 0.01:
                    formatted = f"_{formatted}_"
                cells.append(formatted)

            cells.extend([row['BC_IOU'], row['BC_Diff_up'], row['BC_Diff_low']])

            lines.append('| ' + ' | '.join(cells) + ' |')

        lines.append("")
        lines.append("**Note**: **Bold** = Best, _Italic_ = Second Best")

        return lines


def main():
    results_dir = Path(__file__).parent / 'evaluation_results_v4'

    if not results_dir.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return

    generator = ResultsTableGeneratorV4(results_dir)
    generator.load_all_results()

    print(f"加载了 {len(generator.data)} 个结果文件")

    # 生成所有conf阈值的表格
    conf_values = ['0.25', '0.3', '0.4', '0.5']
    for conf in conf_values:
        print(f"\n生成conf={conf}的表格...")
        generator.generate_table(conf)


if __name__ == '__main__':
    main()
