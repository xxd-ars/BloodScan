import torch, sys, os
from torchvision import transforms
from PIL import Image
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO

project_root = Path(__file__).parent.parent
fusion_dict = {
    'concat_compress': 'yolo11x-dseg-concat-compress.yaml',
    'crossattn': 'yolo11x-dseg-crossattn.yaml',
    'crossattn-precise': 'yolo11x-dseg-crossattn-precise.yaml',
    'weighted-fusion': 'yolo11x-dseg-weighted-fusion.yaml',
    'id-blue': 'yolo11x-dseg-id-blue.yaml',
    'id-white': 'yolo11x-dseg-id-white.yaml',
}
fusion_name = 'crossattn-precise'  # 'concat_compress', 'weighted-fusion', 'crossattn', 'id-blue', 'id-white'
model_yaml = project_root / 'dual_yolo' / 'models' / fusion_dict[fusion_name]
model_pt = project_root / 'dual_yolo' / 'runs' / 'segment' / f'dual_modal_train_{fusion_name}' / 'weights' / 'best.pt'

model_dual = YOLO(model_yaml).load(model_pt)
model_dual.info(verbose=True)

# 加载蓝光和白光图像
project_root = Path(__file__).parent.parent
img_path_b = project_root / 'datasets' / 'Dual-Modal-1504-500-1' / 'images_b' / 'test' / '2022-03-28_103204_17_T5_2412_0.jpg'
img_path_w = project_root / 'datasets' / 'Dual-Modal-1504-500-1' / 'images_w' / 'test' / '2022-03-28_103204_17_T3_2410_0.jpg'
image_b = Image.open(img_path_b).convert('RGB')
image_w = Image.open(img_path_w).convert('RGB')
print(f"原图尺寸 (白光): {image_w.size}")

# 准备双模态输入张量
transform = transforms.Compose([transforms.ToTensor()])
blue_tensor = transform(image_b).unsqueeze(0)   # [1, 3, H, W]
white_tensor = transform(image_w).unsqueeze(0)  # [1, 3, H, W]
dual_tensor = torch.cat([blue_tensor, white_tensor], dim=1)  # [1, 6, H, W]

print(f"蓝光图像张量: {blue_tensor.shape}")
print(f"白光图像张量: {white_tensor.shape}")
print(f"双模态拼接张量: {dual_tensor.shape}")

model_dual.predict(dual_tensor) #, visualize = True)

## 输出结果
# Transferred 1575/1575 items from pretrained weights
# YOLO11x-dseg-id summary: 557 layers, 90,901,049 parameters, 90,901,033 gradients
# 原图尺寸 (白光): (1504, 1504)
# 蓝光图像张量: torch.Size([1, 3, 1504, 1504])
# 白光图像张量: torch.Size([1, 3, 1504, 1504])
# 双模态拼接张量: torch.Size([1, 6, 1504, 1504])

# 0: 1504x1504 1 0, 1 1, 1 2, 4974.1ms
# Speed: 0.0ms preprocess, 4974.1ms inference, 16.8ms postprocess per image at shape (1, 6, 1504, 1504)