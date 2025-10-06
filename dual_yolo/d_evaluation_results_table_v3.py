"""
读取evaluation_results_v3_novis文件夹中的结果数据并生成表格可视化
适配v3版本的JSON结构（优化版：以Mask mAP为主）
"""

import json
from pathlib import Path


class ResultsTableGeneratorV3:
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

                    # 尝试多种metrics文件名模式
                    possible_files = [
                        method_dir / f'metrics_{method_name}.json',
                        *list(method_dir.glob('metrics_*.json'))
                    ]

                    metrics_file = None
                    for f in possible_files:
                        if f.exists():
                            metrics_file = f
                            break

                    if metrics_file:
                        with open(metrics_file, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)

                        self.data.append({
                            'method': method_name,
                            'conf': conf_threshold,
                            'metrics': metrics
                        })

    def format_value_with_std(self, mean, std, decimals=2, format_type='latex'):
        """格式化均值±标准差"""
        if std == 0:
            return f"{mean:.{decimals}f}"
        else:
            if format_type == 'latex':
                return f"{mean:.{decimals}f}$\\pm${std:.{decimals}f}"
            else:
                return f"{mean:.{decimals}f}±{std:.{decimals}f}"

    def extract_class_metrics(self, per_class_metrics, class_key):
        """从per_class_metrics中提取指定类别的数据"""
        class_data = per_class_metrics.get(class_key, {})
        medical = class_data.get('medical_metrics', {})
        academic = class_data.get('academic_metrics', {})

        return {
            'detection_rate': medical.get('detection_rate', 0) * 100,
            'iou_mean': medical.get('iou_mean', 0),
            'iou_std': medical.get('iou_std', 0),
            'upper_diff_mean': medical.get('upper_diff_mean', 0),
            'upper_diff_std': medical.get('upper_diff_std', 0),
            'lower_diff_mean': medical.get('lower_diff_mean', 0),
            'lower_diff_std': medical.get('lower_diff_std', 0),
            'mask_ap50': academic.get('mask_ap50', 0),
            'mask_ap50_95': academic.get('mask_ap50_95', 0),
            'recall': academic.get('recall', 0),
            'precision': academic.get('precision', 0)
        }

    def generate_table(self, conf_threshold='0.5'):
        """生成指定置信度的结果表格"""
        # 筛选指定置信度的数据
        conf_data = [d for d in self.data if d['conf'] == conf_threshold]

        if not conf_data:
            print(f"没有找到置信度为{conf_threshold}的数据")
            return None

        # 构建表格数据
        table_data = []
        for item in conf_data:
            method = item['method']
            metrics = item['metrics']

            # 获取per_class_metrics
            per_class = metrics.get('per_class_metrics', {})

            # 提取血清层（class 0）和血浆层（class 2）数据并合并
            serum_data = self.extract_class_metrics(per_class, 'class_0_serum')
            plasma_data = self.extract_class_metrics(per_class, 'class_2_plasma')

            # 合并class 0和class 2的数据（取平均）
            if serum_data['detection_rate'] > 0 and plasma_data['detection_rate'] > 0:
                sp_det_rate = (serum_data['detection_rate'] + plasma_data['detection_rate']) / 2
                sp_iou_mean = (serum_data['iou_mean'] + plasma_data['iou_mean']) / 2
                sp_iou_std = ((serum_data['iou_std']**2 + plasma_data['iou_std']**2) / 2) ** 0.5
                sp_up_mean = (serum_data['upper_diff_mean'] + plasma_data['upper_diff_mean']) / 2
                sp_up_std = ((serum_data['upper_diff_std']**2 + plasma_data['upper_diff_std']**2) / 2) ** 0.5
                sp_low_mean = (serum_data['lower_diff_mean'] + plasma_data['lower_diff_mean']) / 2
                sp_low_std = ((serum_data['lower_diff_std']**2 + plasma_data['lower_diff_std']**2) / 2) ** 0.5
                sp_map50 = (serum_data['mask_ap50'] + plasma_data['mask_ap50']) / 2 * 100
                sp_recall = (serum_data['recall'] + plasma_data['recall']) / 2 * 100
            elif serum_data['detection_rate'] > 0:
                sp_det_rate = serum_data['detection_rate']
                sp_iou_mean, sp_iou_std = serum_data['iou_mean'], serum_data['iou_std']
                sp_up_mean, sp_up_std = serum_data['upper_diff_mean'], serum_data['upper_diff_std']
                sp_low_mean, sp_low_std = serum_data['lower_diff_mean'], serum_data['lower_diff_std']
                sp_map50 = serum_data['mask_ap50'] * 100
                sp_recall = serum_data['recall'] * 100
            elif plasma_data['detection_rate'] > 0:
                sp_det_rate = plasma_data['detection_rate']
                sp_iou_mean, sp_iou_std = plasma_data['iou_mean'], plasma_data['iou_std']
                sp_up_mean, sp_up_std = plasma_data['upper_diff_mean'], plasma_data['upper_diff_std']
                sp_low_mean, sp_low_std = plasma_data['lower_diff_mean'], plasma_data['lower_diff_std']
                sp_map50 = plasma_data['mask_ap50'] * 100
                sp_recall = plasma_data['recall'] * 100
            else:
                sp_det_rate = 0
                sp_iou_mean, sp_iou_std = 0, 0
                sp_up_mean, sp_up_std = 0, 0
                sp_low_mean, sp_low_std = 0, 0
                sp_map50 = 0
                sp_recall = 0

            # 格式化血清/血浆层数据
            sp_iou = self.format_value_with_std(sp_iou_mean, sp_iou_std, 2, 'latex')
            sp_diff_up = self.format_value_with_std(sp_up_mean, sp_up_std, 1, 'latex')
            sp_diff_low = self.format_value_with_std(sp_low_mean, sp_low_std, 1, 'latex')

            # 提取白膜层（class 1）数据
            bc_data = self.extract_class_metrics(per_class, 'class_1_buffy_coat')
            bc_det_rate = bc_data['detection_rate']
            bc_iou = self.format_value_with_std(bc_data['iou_mean'], bc_data['iou_std'], 2, 'latex')
            bc_diff_up = self.format_value_with_std(bc_data['upper_diff_mean'], bc_data['upper_diff_std'], 1, 'latex')
            bc_diff_low = self.format_value_with_std(bc_data['lower_diff_mean'], bc_data['lower_diff_std'], 1, 'latex')
            bc_map50 = bc_data['mask_ap50'] * 100
            bc_recall = bc_data['recall'] * 100

            table_data.append({
                'Method': method,
                'SP_mAP50': f"{sp_map50:.2f}",
                'SP_Recall': f"{sp_recall:.2f}",
                'SP_IOU': sp_iou,
                'SP_Diff_up': sp_diff_up,
                'SP_Diff_low': sp_diff_low,
                'BC_mAP50': f"{bc_map50:.2f}",
                'BC_Recall': f"{bc_recall:.2f}",
                'BC_IOU': bc_iou,
                'BC_Diff_up': bc_diff_up,
                'BC_Diff_low': bc_diff_low
            })

        # 应用格式化
        formatted_data = self.apply_formatting(table_data, format_type='latex')

        # 按指定顺序排序
        method_order = [
            'Yolo11', 'Yolo11-Blue', 'Yolo11-White', 'Yolo11-Blue-30', 'Yolo11-White-30',
            'Dual Yolo Concat', 'Dual Yolo Weighted',
            'Dual Yolo CrossAttn', 'Dual Yolo CrossAttn (30 Epochs)', 'Dual Yolo (Our Best)'
        ]

        method_map = {
            'id-blue-30': 'Yolo11-Blue-30',
            'id-white-30': 'Yolo11-White-30',
            'id-blue': 'Yolo11-Blue',
            'id-white': 'Yolo11-White',
            'id_blue': 'Yolo11-Blue',
            'id_white': 'Yolo11-White',
            'id': 'Yolo11',
            'concat-compress': 'Dual Yolo Concat',
            'weighted-fusion': 'Dual Yolo Weighted',
            'crossattn': 'Dual Yolo CrossAttn',
            'crossattn-precise': 'Dual Yolo (Our Best)',
            'crossattn-30epoch': 'Dual Yolo CrossAttn (30 Epochs)'
        }

        # 添加排序键
        for row in formatted_data:
            display_name = method_map.get(row['Method'], row['Method'])
            if display_name in method_order:
                row['sort_key'] = method_order.index(display_name)
            else:
                row['sort_key'] = 999

        formatted_data.sort(key=lambda x: x['sort_key'])

        for row in formatted_data:
            row.pop('sort_key', None)

        return formatted_data

    def apply_formatting(self, table_data, format_type='latex'):
        """应用最优值加粗和次优值下划线格式"""
        if not table_data:
            return table_data

        # 需要格式化的数值列（值越高越好）
        higher_better = ['SP_mAP50', 'SP_Recall', 'SP_IOU', 'BC_mAP50', 'BC_Recall', 'BC_IOU']
        # 值越低越好
        lower_better = ['SP_Diff_up', 'SP_Diff_low', 'BC_Diff_up', 'BC_Diff_low']

        def extract_value(val):
            if isinstance(val, str):
                clean_val = val.replace('**', '').replace('_', '').replace('\\textbf{', '').replace('}', '').replace('\\underline{', '')
                if '$\\pm$' in clean_val:
                    return float(clean_val.split('$\\pm$')[0])
                elif '±' in clean_val:
                    return float(clean_val.split('±')[0])
                return float(clean_val)
            return float(val)

        for col in higher_better + lower_better:
            values = [extract_value(row[col]) for row in table_data]

            if col in higher_better:
                sorted_indices = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
            else:
                sorted_indices = sorted(range(len(values)), key=lambda i: values[i])

            for i, idx in enumerate(sorted_indices):
                if format_type == 'latex':
                    if i == 0:
                        table_data[idx][col] = f"\\textbf{{{table_data[idx][col]}}}"
                    elif i == 1:
                        table_data[idx][col] = f"\\underline{{{table_data[idx][col]}}}"
                else:
                    if i == 0:
                        table_data[idx][col] = f"**{table_data[idx][col]}**"
                    elif i == 1:
                        table_data[idx][col] = f"_{table_data[idx][col]}_"

        return table_data

    def save_table(self, table_data, conf_threshold='0.5'):
        """保存表格"""
        if not table_data:
            return

        # 保存LaTeX格式
        latex_file = self.results_dir / f'results_table_conf_{conf_threshold}.tex'

        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(f"% Evaluation Results (Confidence = {conf_threshold})\n")
            for i, row in enumerate(table_data):
                method_map = {
                    'id-blue-30': 'Yolo11-Blue-30',
                    'id-white-30': 'Yolo11-White-30',
                    'id-blue': 'Yolo11-Blue',
                    'id-white': 'Yolo11-White',
                    'id_blue': 'Yolo11-Blue',
                    'id_white': 'Yolo11-White',
                    'id': 'Yolo11',
                    'concat-compress': 'Dual Yolo Concat',
                    'weighted-fusion': 'Dual Yolo Weighted',
                    'crossattn': 'Dual Yolo CrossAttn',
                    'crossattn-precise': 'Dual Yolo (Our Best)',
                    'crossattn-30epoch': 'Dual Yolo CrossAttn (30 Epochs)'
                }

                method_name = method_map.get(row['Method'], row['Method'])

                # 判断是否为双模态方法
                dual_methods = ['concat-compress', 'weighted-fusion', 'crossattn', 'crossattn-precise', 'crossattn-30epoch']
                is_dual = row['Method'] in dual_methods
                dual_symbol = '$\\checkmark$' if is_dual else '$\\times$'

                # 处理数据缺失
                sp_map50 = row['SP_mAP50'] if row['SP_mAP50'] != '0.00' else '--'
                sp_recall = row['SP_Recall'] if row['SP_Recall'] != '0.00' else '--'
                sp_iou = row['SP_IOU'] if row['SP_IOU'] != '0.00' else '--'
                sp_up = row['SP_Diff_up'] if row['SP_Diff_up'] != '0.0' else '--'
                sp_low = row['SP_Diff_low'] if row['SP_Diff_low'] != '0.0' else '--'

                bc_map50 = row['BC_mAP50'] if row['BC_mAP50'] != '0.00' else '--'
                bc_recall = row['BC_Recall'] if row['BC_Recall'] != '0.00' else '--'

                # Our Best 特殊处理
                is_our_best = row['Method'] == 'crossattn-precise'

                if is_our_best and i > 0:
                    f.write("\\hline\n")

                if is_our_best:
                    method_display = f"\\textbf{{{method_name}}}"
                else:
                    method_display = method_name

                # 输出LaTeX行
                f.write(f"{method_display} & {dual_symbol} & {sp_map50} & {sp_recall} & {sp_iou} & {sp_up} & {sp_low} & {bc_map50} & {bc_recall} & {row['BC_IOU']} & {row['BC_Diff_up']} & {row['BC_Diff_low']}\\\\")

                if is_our_best:
                    f.write("\n\\hline")

                f.write("\n")

        # 保存markdown格式
        md_file = self.results_dir / f'results_table_conf_{conf_threshold}.md'

        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(f"# Evaluation Results (Confidence = {conf_threshold})\n\n")

            # 写入表头
            f.write("| Method | Serum/Plasma | | | | | Buffy Coat | | | | |\n")
            f.write("|--------|--------------|---|---|---|---|-----------|----|---|---|---|\n")
            f.write("| | mAP@0.5 | Recall | IOU | Diff_up | Diff_low | mAP@0.5 | Recall | IOU | Diff_up | Diff_low |\n")
            f.write("|--------|---------|--------|-----|---------|----------|---------|--------|-----|---------|----------|\n")

            # 写入数据行
            for row in table_data:
                f.write(f"| {row['Method']} | {row['SP_mAP50']} | {row['SP_Recall']} | {row['SP_IOU']} | {row['SP_Diff_up']} | {row['SP_Diff_low']} | {row['BC_mAP50']} | {row['BC_Recall']} | {row['BC_IOU']} | {row['BC_Diff_up']} | {row['BC_Diff_low']} |\n")

            f.write(f"\n\n**Note**: **Bold** = Best, _Italic_ = Second Best\n")

        print(f"LaTeX表格已保存到: {latex_file}")
        print(f"Markdown表格已保存到: {md_file}")
        return latex_file, md_file

    def print_table(self, table_data, conf_threshold='0.5'):
        """打印LaTeX表格到控制台"""
        if not table_data:
            return

        print(f"\n=== LaTeX Table (Confidence = {conf_threshold}) ===")

        for i, row in enumerate(table_data):
            method_map = {
                'id-blue-30': 'Yolo11-Blue-30',
                'id-white-30': 'Yolo11-White-30',
                'id-blue': 'Yolo11-Blue',
                'id-white': 'Yolo11-White',
                'id_blue': 'Yolo11-Blue',
                'id_white': 'Yolo11-White',
                'id': 'Yolo11',
                'concat-compress': 'Dual Yolo Concat',
                'weighted-fusion': 'Dual Yolo Weighted',
                'crossattn': 'Dual Yolo CrossAttn',
                'crossattn-precise': 'Dual Yolo (Our Best)',
                'crossattn-30epoch': 'Dual Yolo CrossAttn (30 Epochs)'
            }

            method_name = method_map.get(row['Method'], row['Method'])

            dual_methods = ['concat-compress', 'weighted-fusion', 'crossattn', 'crossattn-precise', 'crossattn-30epoch']
            is_dual = row['Method'] in dual_methods
            dual_symbol = '$\\checkmark$' if is_dual else '$\\times$'

            sp_map50 = row['SP_mAP50'] if row['SP_mAP50'] != '0.00' else '--'
            sp_recall = row['SP_Recall'] if row['SP_Recall'] != '0.00' else '--'
            sp_iou = row['SP_IOU'] if row['SP_IOU'] != '0.00' else '--'
            sp_up = row['SP_Diff_up'] if row['SP_Diff_up'] != '0.0' else '--'
            sp_low = row['SP_Diff_low'] if row['SP_Diff_low'] != '0.0' else '--'

            bc_map50 = row['BC_mAP50'] if row['BC_mAP50'] != '0.00' else '--'
            bc_recall = row['BC_Recall'] if row['BC_Recall'] != '0.00' else '--'

            is_our_best = row['Method'] == 'crossattn-precise'

            if is_our_best and i > 0:
                print("\\hline")

            if is_our_best:
                method_display = f"\\textbf{{{method_name}}}"
            else:
                method_display = method_name

            print(f"{method_display} & {dual_symbol} & {sp_map50} & {sp_recall} & {sp_iou} & {sp_up} & {sp_low} & {bc_map50} & {bc_recall} & {row['BC_IOU']} & {row['BC_Diff_up']} & {row['BC_Diff_low']}\\\\")

            if is_our_best:
                print("\\hline")

        print("\n% Note: \\textbf{} = Best, \\underline{} = Second Best")

    def generate_all_tables(self):
        """生成所有置信度的表格"""
        conf_thresholds = list(set([d['conf'] for d in self.data]))

        for conf in sorted(conf_thresholds):
            table_data = self.generate_table(conf)
            if table_data is not None:
                self.save_table(table_data, conf)
                self.print_table(table_data, conf)
                print("\n" + "="*80)


def main():
    """主函数"""
    # 设置结果目录路径
    results_dir = Path(__file__).parent / 'evaluation_results_v3_novis'

    if not results_dir.exists():
        print(f"结果目录不存在: {results_dir}")
        return

    # 创建表格生成器
    generator = ResultsTableGeneratorV3(results_dir)

    # 加载数据
    generator.load_all_results()

    if not generator.data:
        print("没有找到任何评估结果数据")
        return

    # 生成所有置信度的表格
    generator.generate_all_tables()


if __name__ == '__main__':
    main()
