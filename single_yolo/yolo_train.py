import sys, os, shutil, yaml, platform, json
from pathlib import Path
from ultralytics import YOLO

# 项目根目录
project_root = Path(__file__).parent.parent

# ==================== 单模态训练配置 ====================
MODALITY = 'blue'        # 'blue', 'white'
USE_PRETRAINED = False   # True: 加载蓝光权重, False: 从头训练

if platform.system() == 'Windows':
    TRAIN_CONFIG = {
        'epochs': 30,
        'batch': 1,
        'imgsz': 1504,
        
        'amp': True,
        'device': 0,
    }
elif platform.system() == 'Linux':
    TRAIN_CONFIG = {
        'epochs': 30,
        'batch': 8,
        'imgsz': 1504,

        'amp': True,
        'device': [0, 1, 2, 3],
    }
# ========================================================
print(json.dumps(TRAIN_CONFIG, indent=4, ensure_ascii=False))

def setup_model(use_pretrained):
    """设置模型：选择是否使用预训练权重"""
    if use_pretrained:
        # 加载已训练的蓝光权重
        pretrained_path = project_root / 'single_yolo' / 'runs' / 'segment' / 'train_blue_rawdata_1504_500_10epoch' / 'weights' / 'best.pt'
        if os.path.exists(pretrained_path):
            model = YOLO('yolo11x-seg.yaml').load(str(pretrained_path))
            print(f"✅ 加载预训练权重: {pretrained_path}")
        else:
            print(f"❌ 预训练权重不存在，使用默认权重")
            model = YOLO(project_root / 'single_yolo' / 'weights' / 'yolo11x-seg.pt')
    else:
        # 使用ImageNet预训练权重
        model = YOLO(project_root / 'single_yolo' / 'weights' / 'yolo11x-seg.pt')
        print("📦 使用ImageNet预训练权重")

    return model

def setup_images_for_training(modality):
    """根据模态设置训练图像：复制指定模态的图像到images目录"""
    dataset_root = project_root / 'datasets' / 'Dual-Modal-1504-500-1'

    for split in ['train', 'valid', 'test']:
        images_target = dataset_root / 'images' / split
        labels_source = dataset_root / 'labels' / split

        # 清空目标图像目录
        if images_target.exists():
            shutil.rmtree(images_target)
        images_target.mkdir(parents=True, exist_ok=True)

        if not labels_source.exists():
            print(f"⚠️ 标签目录不存在: {labels_source}")
            continue

        if modality == 'white':
            # 白光模式：根据标签文件找到对应的白光图像并重命名
            images_source = dataset_root / 'images_w' / split

            if not images_source.exists():
                print(f"⚠️ 白光图像目录不存在: {images_source}")
                continue

            # 遍历所有标签文件，为每个标签找到对应的白光图像并重命名
            for label_file in labels_source.glob('*.txt'):
                # 标签: 2022-03-28_103204_17_T5_2412_0.txt
                parts = label_file.stem.split('_')
                if len(parts) < 5:
                    continue

                # 前三个下划线之前: 2022-03-28_103204_17_
                prefix = '_'.join(parts[:3]) + '_'
                # 最后一个下划线及之后: _0
                suffix = '_' + parts[-1]

                # 查找匹配的白光图像: 2022-03-28_103204_17_*_0.jpg
                # 排除蓝光图像（T5），只匹配白光图像
                pattern = f"{prefix}*{suffix}.jpg"
                matching_white = list(images_source.glob(pattern))

                if matching_white:
                    # 将白光图像重命名为标签文件对应的名称并复制到images目录
                    # 原白光: 2022-03-28_103204_17_T3_2410_0.jpg
                    # 重命名为: 2022-03-28_103204_17_T5_2412_0.jpg (匹配标签文件名)
                    target_img_path = images_target / (label_file.stem + '.jpg')
                    shutil.copy2(matching_white[0], target_img_path)

        elif modality == 'blue':
            # 蓝光模式：直接复制蓝光图像
            images_source = dataset_root / 'images_b' / split

            if not images_source.exists():
                print(f"⚠️ 蓝光图像目录不存在: {images_source}")
                continue

            # 直接复制所有蓝光图像
            for img_file in images_source.glob('*.jpg'):
                target_img_path = images_target / img_file.name
                shutil.copy2(img_file, target_img_path)

    print(f"✅ 已设置 {modality} 模态的训练图像")

def prepare_dataset_for_training(modality):
    """为指定模态准备数据集"""
    # 设置图像目录
    setup_images_for_training(modality)

    # 返回标准data.yaml路径
    return project_root / 'datasets' / 'Dual-Modal-1504-500-1' / 'data.yaml'

def cleanup():
    """清理临时文件"""
    # 可选：清空images目录
    dataset_root = project_root / 'datasets' / 'Dual-Modal-1504-500-1'
    images_dir = dataset_root / 'images'
    if images_dir.exists():
        for split_dir in ['train', 'valid', 'test']:
            split_path = images_dir / split_dir
            if split_path.exists():
                shutil.rmtree(split_path)
                split_path.mkdir(exist_ok=True)

def main():
    """主训练函数"""
    print(f"🔧 训练模态: {MODALITY}")
    print(f"🔧 使用预训练: {USE_PRETRAINED}")

    # 准备数据集
    data_config = prepare_dataset_for_training(MODALITY)

    # 检查数据集是否存在
    if not os.path.exists(data_config):
        print(f"❌ 数据集配置不存在: {data_config}")
        return

    # 设置模型
    model = setup_model(USE_PRETRAINED)

    # 开始训练
    print("🚀 开始训练单模态YOLO模型...")
    results = model.train(
        data=str(data_config),
        name=f'single_{MODALITY}_{"pretrained" if USE_PRETRAINED else "scratch"}',
        project=project_root / 'single_yolo' / 'runs' / 'segment',
        **TRAIN_CONFIG
    )

    print("✅ 训练完成！")
    print(f"📊 训练结果: {results}")

    # cleanup()  # 注释掉自动清理

if __name__ == "__main__":
    main()