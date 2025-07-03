# DualYOLO: 蓝-白光双模态血液分层检测

本项目实现了一个基于 YOLOv11 架构的双骨干网络，用于处理蓝光和白光下的血液样本图像，旨在通过特征融合提升血液分层检测的准确性。

## 特点

- **双模态输入**: 同时处理蓝光和白光图像，利用不同光照条件下的特征。
- **YOLOv11 骨干**: 采用两条独立的 YOLOv11Backbone 提取特征。
- **可切换特征融合**: 支持三种特征融合策略：
    - `AddFusion`: 基于注意力的加权融合。
    - `CatFusion`: 特征拼接与压缩。
    - `XFormerFusion`: 基于 Cross-Transformer 的特征交互。
- **端到端训练**: 所有模块参数均可联合训练。
- **分割任务**: 目标是进行像素级的血液分层分割。

## 环境要求

- Python 3.12+
- PyTorch 2.7.0+
- ultralytics (用于 YOLOv11 组件)

## 文件结构

```
dual_yolo/
├── model.py         # 网络模型定义 (DualYOLO, Fusion modules)
├── train.py         # 训练脚本
├── test.py          # 测试与验证脚本 (包含模型结构输出)
├── README.md        # 项目说明
└── (其他可能的配置文件或数据目录)
```

## 输入格式

- 蓝光图像: `image_blue.png` (例如: 3 x 640 x 640)
- 白光图像: `image_white.png` (例如: 3 x 640 x 640)
- 两幅图像需要来自同一场景，具有相同的分辨率，并且已经对齐。

## 使用示例

### 1. 安装依赖

```bash
pip install torch torchvision torchaudio ultralytics
# 其他可能的依赖
```

### 2. 训练模型

```bash
python dual_yolo/train.py --data <path_to_data_config.yaml> --fusion_type add --epochs 100 --batch_size 8
python dual_yolo/train.py --data <path_to_data_config.yaml> --fusion_type cat --epochs 100 --batch_size 8
python dual_yolo/train.py --data <path_to_data_config.yaml> --fusion_type ctr --epochs 100 --batch_size 8
```
*(具体参数请根据 `train.py` 中的定义调整)*

### 3. 进行推理/检测

```bash
python dual_yolo/test.py --weights <path_to_trained_weights.pt> --source_blue <path_to_blue_image> --source_white <path_to_white_image> --fusion_type ctr
```
*(具体参数请根据 `test.py` 中的定义调整)*

## 基准测试 (Ablation Study)

| 融合策略 (`fusion_type`) | mAP@0.5 (Seg) | FPS (GPU) | FPS (CPU) | 参数量 (M) |
|---------------------------|---------------|-----------|-----------|------------|
| `AddFusion`               | (待填充)      | (待填充)  | (待填充)  | (待填充)   |
| `CatFusion`               | (待填充)      | (待填充)  | (待填充)  | (待填充)   |
| `XFormerFusion` (`ctr`)   | (待填充)      | (待填充)  | (待填充)  | (待填充)   |
| 单模态 (蓝光) - 基线     | (待填充)      | (待填充)  | (待填充)  | (待填充)   |
| 单模态 (白光) - 基线     | (待填充)      | (待填充)  | (待填充)  | (待填充)   |

## 模型架构示意图

(将由 `test.py` 中的脚本生成并嵌入)

---

*此 README 为初步版本，后续将根据实际开发情况更新。*
