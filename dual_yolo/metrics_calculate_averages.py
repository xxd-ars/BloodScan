"""
计算每个方法在三个类别（Serum, Buffy Coat, Plasma）上的平均指标
"""

# conf=0.25的DR和IoU数据
medical_data = {
    'Yolo11-White': {
        'DR': [52.50, 8.06, 51.39],
        'IoU': [0.81, 0.69, 0.88]
    },
    'Yolo11-Blue': {
        'DR': [24.72, 38.06, 2.22],
        'IoU': [0.94, 0.75, 0.96]
    },
    'Dual Yolo Channel Concat': {
        'DR': [86.94, 67.78, 49.72],
        'IoU': [0.97, 0.75, 0.98]
    },
    'Dual Yolo Adaptive Weighted': {
        'DR': [99.72, 75.56, 99.44],
        'IoU': [0.97, 0.73, 0.98]
    },
    'Dual Yolo Cross Attention': {
        'DR': [100.00, 96.39, 100.00],
        'IoU': [0.97, 0.76, 0.98]
    }
}

# Academic Metrics数据
academic_data = {
    'Yolo11-White': {
        'AP50-95': [48.17, 7.15, 55.75],
        'AP75': [47.56, 1.84, 50.49],
        'AP50': [65.85, 17.18, 73.35],
        'F1': [62.55, 18.82, 71.68],
        'Precision': [56.35, 44.39, 84.53],
        'Recall': [70.28, 11.94, 62.23]
    },
    'Yolo11-Blue': {
        'AP50-95': [87.52, 53.17, 93.24],
        'AP75': [97.33, 47.35, 98.57],
        'AP50': [96.90, 97.38, 99.50],
        'F1': [96.20, 90.34, 73.09],
        'Precision': [94.87, 95.69, 57.60],
        'Recall': [97.57, 85.56, 100.00]
    },
    'Dual Yolo Channel Concat': {
        'AP50-95': [98.80, 55.22, 98.52],
        'AP75': [99.65, 57.77, 98.53],
        'AP50': [99.39, 90.86, 98.49],
        'F1': [94.88, 84.17, 91.69],
        'Precision': [90.71, 87.79, 86.09],
        'Recall': [99.44, 80.83, 98.06]
    },
    'Dual Yolo Adaptive Weighted': {
        'AP50-95': [99.36, 50.31, 99.68],
        'AP75': [99.95, 38.62, 99.50],
        'AP50': [99.50, 95.06, 99.50],
        'F1': [99.96, 88.83, 99.39],
        'Precision': [100.00, 87.70, 98.78],
        'Recall': [99.92, 90.00, 100.00]
    },
    'Dual Yolo Cross Attention': {
        'AP50-95': [99.57, 55.71, 99.50],
        'AP75': [100.00, 54.53, 99.50],
        'AP50': [99.50, 99.50, 99.50],
        'F1': [99.63, 98.67, 99.89],
        'Precision': [99.25, 100.00, 99.78],
        'Recall': [100.00, 97.37, 100.00]
    }
}

print("="*120)
print("Average Metrics Across Three Classes (Serum, Buffy Coat, Plasma)")
print("="*120)
print(f"{'Method':<35} {'Avg DR':<10} {'Avg IoU':<10} {'Avg mAP':<10} {'Avg AP75':<10} {'Avg AP50':<10} {'Avg F1':<10} {'Avg Prec':<10} {'Avg Rec':<10}")
print("-"*120)

for method in medical_data.keys():
    # 计算平均值
    avg_dr = sum(medical_data[method]['DR']) / 3
    avg_iou = sum(medical_data[method]['IoU']) / 3
    avg_map = sum(academic_data[method]['AP50-95']) / 3
    avg_ap75 = sum(academic_data[method]['AP75']) / 3
    avg_ap50 = sum(academic_data[method]['AP50']) / 3
    avg_f1 = sum(academic_data[method]['F1']) / 3
    avg_prec = sum(academic_data[method]['Precision']) / 3
    avg_rec = sum(academic_data[method]['Recall']) / 3

    print(f"{method:<35} {avg_dr:<10.2f} {avg_iou:<10.3f} {avg_map:<10.2f} {avg_ap75:<10.2f} {avg_ap50:<10.2f} {avg_f1:<10.2f} {avg_prec:<10.2f} {avg_rec:<10.2f}")

print("="*120)
print("\nNotes:")
print("- DR (Detection Rate) and IoU are from conf=0.25")
print("- mAP (AP50-95), AP75, AP50, F1, Precision, Recall are from conf=0.001")
print("- All values are averages across Serum, Buffy Coat, and Plasma classes")
