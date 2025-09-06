# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

BloodScan是一个基于计算机视觉的血液试管检测和分类项目，主要包含以下几个核心模块：

- **blue_yolo**: 单模态蓝光通道YOLO模型，用于血液分层检测和分割
- **dual_yolo**: 双模态（蓝光+白光）特征融合YOLO模型，实现更精确的检测
- **src**: 控制采血管流水线的机械臂、摄像头等外设控制代码
- **ultralytics**: 专门为dual_yolo设计的自定义ultralytics库，包含跨模态注意力机制
- **dual_dataset**: 双模态数据集创建和预处理工具
- **scripts**: 图像处理和远程服务脚本
- **cv**: 传统计算机视觉方法和SAM相关实验代码

## 常用命令

### 环境和依赖

```bash
# 安装项目依赖
pip install -r requirements.txt

# 核心依赖（如需单独安装）
pip install torch>=2.4.1+cu124 torchvision>=0.19.1+cu124
pip install ultralytics>=8.3.31 opencv-python>=4.10.0
pip install PyQt5>=5.15.10 pyserial>=3.5
pip install numpy>=1.26.0 matplotlib>=3.9.2

# 可选：SAM2依赖（用于cv/sam_class/）
# 注意：requirements.txt中包含SAM2的editable安装路径
```

### 训练模型

```bash
# 单模态蓝光训练
cd blue_yolo
python yolo_train.py

# 双模态训练 (支持多GPU: device=[0,1,2,3])
cd dual_yolo
python d_model_train.py

# 权重迁移 (单模态→双模态)
cd dual_yolo  
python d_weights_transfer.py
```

### 模型评估与测试

```bash
# 双模态模型测试
cd dual_yolo
python d_model_test.py

# 完整评估系统 (医学标注对比)
cd dual_yolo
python d_model_evaluate.py

# 注意力可视化
cd dual_yolo
python d_model_attention_vis.py
```

### 数据集处理

```bash
# 双模态数据集创建和预处理
cd dual_dataset
python d_dataset_main.py

# 6通道数据生成 (.npy格式)
cd dual_dataset
python d_dataset_concat_6ch.py

# 数据增强处理
cd dual_dataset
python d_dataset_augmentation.py
```

### 运行主控制程序

```bash
# 启动PyQt5界面和硬件控制
cd src
python main.py

# 图像旋转处理
python scripts/image_rotate.py

# 远程YOLO服务
python scripts/yolo_service.py
python scripts/yolo_valid_remote.py
```

## 核心架构

### 双模态YOLO架构
项目的核心创新在于dual_yolo模块，实现了白光和蓝光图像的特征融合：

1. **双Backbone设计**: 分别处理白光(T3)和蓝光(T5)输入
2. **特征融合策略**: 
   - 加权融合 (weighted_fusion)
   - 拼接压缩融合 (concat_compress) 
   - 跨模态注意力融合 (cross_attn) - 推荐方案

3. **跨模态注意力机制**:
   - 位于 `ultralytics/nn/modules/fusion.py`
   - 实现了Token级别的局部注意力
   - 蓝光主导的单向查询设计
   - 大幅降低计算复杂度（相比全局注意力减少91%）

### 模型配置文件
- `dual_yolo/models/yolo11x-dseg-crossattn-precise.yaml`: **推荐**跨模态注意力配置
- `dual_yolo/models/yolo11x-dseg-crossattn.yaml`: 标准跨模态注意力配置
- `dual_yolo/models/yolo11x-dseg-weighted-fusion.yaml`: 加权融合配置
- `dual_yolo/models/yolo11x-dseg-concat-compress.yaml`: 拼接融合配置
- `dual_yolo/models/yolo11x-dseg-id.yaml`: 单模态基线配置

### 数据流
```
白光输入 → 白光Backbone → 特征1, 特征2, 特征3
                              ↓      ↓      ↓
                          CrossTrans CrossTrans CrossTrans
                              ↑      ↑      ↑
蓝光输入 → 蓝光Backbone → 特征1, 特征2, 特征3
                              ↓
                            Neck
                              ↓
                           检测头
```

## 数据集结构

### 单模态数据集 (blue_yolo)
```
datasets/Blue-Rawdata-1504-500-1/
├── train/images/
├── valid/images/ 
├── test/images/
├── train/labels/
├── valid/labels/
└── test/labels/
```

### 双模态数据集 (dual_yolo)
```
datasets/Dual-Modal-1504-500-1/
├── train/
│   ├── images_b/ (蓝光图像)
│   ├── images_w/ (白光图像)
│   └── labels/
├── valid/
│   ├── images_b/
│   ├── images_w/
│   └── labels/
└── test/
    ├── images_b/
    ├── images_w/
    └── labels/
```

### 6通道数据集 (高效训练)
```
datasets/Dual-Modal-1504-500-1-6ch/
├── train/
│   ├── images/ (*.npy files, 6通道拼接)
│   └── labels/
├── valid/
└── test/
```

### 数据增强策略
- **旋转增强**: 9种角度变化（文件后缀_0到_8）
- **颜色增强**: 亮度、对比度、曝光调整
- **模糊增强**: 适度模糊模拟成像条件
- **文件命名规范**: `YYYY-MM-DD_HHMMSS_序号_通道_编号_增强ID`

## 硬件控制架构

### 主要组件 (src/)
- **motor/**: 机械臂和步进电机控制
  - `motor_control.py`: 高层运动控制接口
  - `m_ModBusTcp.py`: ModBus TCP通信
  - `m_RS485.py`: RS485串口通信
  
- **algorithm/**: 视觉算法
  - `K_means_5.py`: K-means聚类算法
  - `zbar_v.py`: QR码识别

- **ui/**: PyQt5用户界面
  - `ui_winqt.py`: 主界面

- **database/**: SQLite数据库操作
  - `sqlite.py`: 数据库接口

## 关键文件说明

### 训练相关
- `blue_yolo/yolo_train.py`: 单模态训练脚本
- `dual_yolo/d_model_train.py`: 双模态训练脚本 (支持5种融合策略)
- `dual_yolo/d_model_test.py`: 模型推理测试脚本
- `dual_yolo/d_model_evaluate.py`: **核心**医学评估系统 (严格检测标准)
- `dual_yolo/d_weights_transfer.py`: 权重迁移工具

### 数据处理
- `dual_dataset/d_dataset_main.py`: 双模态数据集构建主程序
- `dual_dataset/d_dataset_creation.py`: 数据集创建逻辑
- `dual_dataset/d_dataset_concat_6ch.py`: **重要**6通道数据生成
- `dual_dataset/d_dataset_augmentation.py`: 数据增强处理
- `dual_dataset/rawdata_crop.py`: 原始数据预处理

### 可视化工具
- `dual_yolo/d_model_attention_vis.py`: 跨模态注意力可视化
- `dual_yolo/d_model_evaluate_vis.py`: 评估结果可视化
- `dual_yolo/d_model_evaluate_vis.ipynb`: 交互式评估分析

### 自定义库修改
- `ultralytics/nn/modules/fusion.py`: **核心**CrossModalAttention实现
- `ultralytics/nn/tasks.py`: 模型解析器修改 (支持双模态)
- `ultralytics/data/utils.py`: 双模态数据加载支持

### 工具和脚本
- `scripts/image_rotate.py`: 图像旋转处理工具
- `scripts/yolo_service.py`: YOLO模型远程服务
- `scripts/upload.py`/`upload2flask.py`: 数据上传工具
- `llm/llm_service.py`: LLM服务接口

## 训练配置参数

### 推荐训练配置
```python
# 双模态训练关键参数
epochs=30                    # 跨模态注意力需要充分训练
imgsz=1504                   # 高分辨率保证检测精度
batch=4                      # 受GPU内存限制
amp=False                    # 关闭混合精度，保证注意力计算精度
workers=4                    # 数据加载并行度
device=[0,1,2,3]            # 多GPU训练支持
```

### 融合策略选择
- **crossattn-precise**: **推荐**最佳性能 (93.06%检测率)
- **crossattn**: 标准跨模态注意力
- **weighted-fusion**: 简单加权融合 (**警告**: 训练不稳定)
- **concat-compress**: 通道拼接融合
- **id**: 单模态基线对比

## 评估系统说明

### 严格检测标准
项目采用"精确一次检测"标准：
- 每个血液分层类别必须被检测到**恰好一次**
- 不允许漏检 (影响召回率)
- 不允许多检 (影响精确率)  
- 检测成功率 = 完全符合标准的样本数 / 总样本数

### 多类别标注体系
- **血清层 (Class 0)**: 点位 0,1,2,3 (矩形区域)
- **白膜层 (Class 1)**: 点位 2,3,4,5 (重叠边界设计)
- **血浆层 (Class 2)**: 点位 4,5,6 (三角形区域)

## 开发注意事项

1. **依赖管理**: 项目使用了大量深度学习和硬件控制库，注意CUDA版本兼容性
2. **GPU内存要求**: 双模态训练需要大容量GPU内存，推荐RTX 3090或以上 (24GB+)
3. **数据对齐**: 双模态训练要求白光和蓝光图像严格对应，文件名规范为核心
4. **模型融合**: 跨模态注意力机制是项目核心创新，修改时需考虑计算效率
5. **文件名规范**: 必须遵循 `日期_时间_序号_通道_编号_增强ID` 格式

## 常见问题

1. **CUDA内存不足**: 降低batch_size到2或1，或使用gradient checkpointing
2. **数据加载错误**: 检查双模态数据集中白光和蓝光图像的文件名对应关系
3. **训练不收敛**: 建议先用加权融合预训练，再切换到跨模态注意力
4. **硬件通信异常**: 检查串口权限和设备连接状态
5. **WeightedFusion失效**: 该策略训练不稳定(7.78%检测率)，避免使用
6. **注意力可视化失败**: 确保模型配置包含CrossModalAttention模块

## 模型权重

- `weights/yolo11x-seg.pt`: 预训练分割权重
- `dual_yolo/weights/dual_yolo11x.pt`: 双模态预训练权重
- `dual_yolo/runs/segment/dual_modal_train_*/weights/`: 训练结果权重

## 项目特色和创新点

### 跨模态注意力机制
项目的核心创新是CrossModalAttention模块，实现了：
- **单向查询设计**: 蓝光特征作为Query查询白光特征
- **Token级局部注意力**: 相比全局注意力减少91%计算复杂度
- **精确边界检测**: 通过token_size和neighbor_size参数优化边界检测精度
- **可视化支持**: 支持注意力权重的空间映射和可视化

### 医学评估标准
采用严格的"精确一次检测"医学标准：
- 每个血液分层类别必须被检测恰好一次
- 不允许漏检或多检
- 基于医学标注的严格评估体系

### 多模态数据增强
- **旋转增强**: 9种角度变化 (0-8)
- **颜色增强**: 适合医学成像的亮度/对比度调整  
- **文件命名规范**: `YYYY-MM-DD_HHMMSS_序号_通道_编号_增强ID`

## 重要路径说明

### 数据集路径
```
datasets/
├── Dual-Modal-1504-500-1/          # 主要双模态数据集
├── Dual-Modal-1504-500-1-6ch/      # 6通道预处理数据集（训练加速）
└── Blue-Rawdata-1504-500-1/        # 单模态蓝光数据集
```

### 配置文件路径
```
dual_yolo/models/
├── yolo11x-dseg-crossattn-precise.yaml  # 推荐配置
├── yolo11x-dseg-crossattn.yaml          # 标准跨模态注意力
├── yolo11x-dseg-weighted-fusion.yaml    # 加权融合（不推荐）
└── yolo11x-dseg-id.yaml                 # 单模态基线
```

### 评估结果路径
```
dual_yolo/evaluation_results_aug/        # 数据增强评估结果
├── crossattn-precise/                   # 最佳模型评估
├── crossattn/                          # 标准注意力评估
└── metrics_*.json                      # 量化指标文件
```

## 开发和调试指南

### 常见调试步骤
1. **数据对齐检查**: 验证白光和蓝光图像文件名对应关系
2. **内存监控**: 双模态训练需要大量GPU内存，建议使用`nvidia-smi`监控
3. **注意力可视化**: 使用`d_model_attention_vis.py`检查注意力学习效果
4. **评估分析**: 运行`d_model_evaluate.py`获得严格的医学评估指标

### 性能优化建议
1. **使用6通道数据集**: `Dual-Modal-1504-500-1-6ch/`可显著加速训练
2. **梯度检查点**: 启用`gradient_checkpointing`减少显存占用
3. **多GPU训练**: 配置`device=[0,1,2,3]`充分利用硬件资源
4. **混合精度**: 注意力计算建议关闭AMP保证精度