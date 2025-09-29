import torch, sys, os
from pathlib import Path
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 确保使用本地修改的ultralytics代码
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO

# 项目根目录
project_root = Path(__file__).parent.parent

# ==================== 训练配置 ====================
# 训练模式选择
TRAIN_MODE = 'scratch'  # 'scratch', 'pretrained', 'freeze_backbone'

# 融合策略选择
# FUSION_NAME = 'crossattn'  # 支持所有融合策略
FUSION_NAMES = ['crossattn-precise', 'crossattn', 'concat-compress', 'weighted-fusion']

# 训练参数
TRAIN_CONFIG = {
    'epochs': 30,
    'batch': 4,
    'imgsz': 1504,
    
    'amp': False,
    'device': [0, 1, 2, 3],
}
# =====================================================

# 融合策略字典
fusion_dict = {
    'concat-compress':  'yolo11x-dseg-concat-compress.yaml',
    'weighted-fusion':  'yolo11x-dseg-weighted-fusion.yaml',
    'crossattn':        'yolo11x-dseg-crossattn.yaml',
    'crossattn-precise':'yolo11x-dseg-crossattn-precise.yaml',
    'id-blue':          'yolo11x-dseg-id-blue.yaml',
    'id-white':         'yolo11x-dseg-id-white.yaml',
}

def freeze_backbone_layers(model):
    """冻结backbone层(0-21)，只训练head层"""
    frozen_count = 0
    total_count = 0

    for name, param in model.model.named_parameters():
        total_count += 1
        if 'model.' in name:
            try:
                layer_num = int(name.split('.')[1])
                # 冻结backbone_b (0-10) 和 backbone_w (11-21)
                if 0 <= layer_num <= 21:
                    param.requires_grad = False
                    frozen_count += 1
            except (ValueError, IndexError):
                pass

    print(f"✅ 已冻结 {frozen_count}/{total_count} 个参数 (backbone layers)")
    return model

def setup_model(mode, fusion_name):
    """根据训练模式设置模型"""
    model_yaml = project_root / 'dual_yolo' / 'models' / fusion_dict[fusion_name]
    model_pt = project_root / 'dual_yolo' / 'weights' / 'dual_yolo11x_bw.pt'

    print(f"* 训练模式: {mode}")
    print(f"* 融合策略: {fusion_name}")

    if mode == 'scratch':
        print("* 从零开始训练")
        model = YOLO(model_yaml)

    elif mode == 'pretrained':
        print("* 使用预训练权重")
        if not os.path.exists(model_pt):
            print(f"* 预训练权重不存在: {model_pt}")
            return None
        model = YOLO(model_yaml).load(model_pt)

    elif mode == 'freeze_backbone':
        print("* 使用预训练权重 + 冻结backbone")
        if not os.path.exists(model_pt):
            print(f"* 预训练权重不存在: {model_pt}")
            return None
        model = YOLO(model_yaml).load(model_pt)
        model = freeze_backbone_layers(model)

    return model

def main():
    for FUSION_NAME in FUSION_NAMES:
        """主训练函数"""
        # 数据配置
        data_config = project_root / 'datasets' / 'Dual-Modal-1504-500-1-6ch' / 'data.yaml'

        # 设置模型
        model = setup_model(TRAIN_MODE, FUSION_NAME)
        model.info(verbose=True)

        # 开始训练
        print("\n* 开始训练双模态YOLO模型...")
        results = model.train(
            data=str(data_config),
            name=f'dual_modal_{TRAIN_MODE}_{FUSION_NAME}',
            project=project_root / 'dual_yolo' / 'runs' / 'segment',
            **TRAIN_CONFIG
        )

        print(f"* 训练结果: {results}")

if __name__ == "__main__":
    main()