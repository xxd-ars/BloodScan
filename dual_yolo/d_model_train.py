import torch, sys, os
from pathlib import Path

# 确保使用本地修改的ultralytics代码
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO

# 项目根目录
project_root = Path(__file__).parent.parent

# 融合策略字典（和 d_model_test.py 相同）
fusion_dict = {
    'concat_compress': 'yolo11x-dseg-concat-compress.yaml',
    'weighted_fusion': 'yolo11x-dseg-weighted-fusion.yaml',
    'cross_attn': 'yolo11x-dseg-crossattn.yaml',
    'id': 'yolo11x-dseg-id.yaml'
}

# 模型配置（和 d_model_test.py 相同）
model_yaml = project_root / 'dual_yolo' / 'models' / fusion_dict['cross_attn']  # 使用交叉注意力融合
model_pt = project_root / 'dual_yolo' / 'weights' / 'dual_yolo11x.pt'

# 数据配置（使用6通道拼接数据）
data_config = project_root / 'datasets' / 'Dual-Modal-1504-500-1-6ch' / 'data.yaml'

# 加载模型（和 d_model_test.py 相同）
model_dual = YOLO(model_yaml).load(model_pt)
model_dual.info(verbose=True)

print("开始训练双模态YOLO模型...")
results = model_dual.train(
    data=str(data_config),
    device="cuda:0",
    epochs=10,
    imgsz=1504,
    batch=4,
    name='d_modal_train',
)

print("训练完成！")
print(f"训练结果: {results}")