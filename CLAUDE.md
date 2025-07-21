# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

BloodScan是一个基于计算机视觉的血液试管检测和分类项目，主要包含以下几个核心模块：

- **blue_yolo**: 单模态蓝光通道YOLO模型，用于血液分层检测和分割
- **dual_yolo**: 双模态（蓝光+白光）特征融合YOLO模型，实现更精确的检测
- **src**: 控制采血管流水线的机械臂、摄像头等外设控制代码
- **ultralytics**: 专门为dual_yolo设计的自定义ultralytics库，包含跨模态注意力机制

## 常用命令

### 训练模型

```bash
# 单模态蓝光训练
cd blue_yolo
python yolo_train.py

# 双模态训练
cd dual_yolo
python d_model_train.py
```

### 模型测试

```bash
# 双模态模型测试
cd dual_yolo
python d_model_test.py
```

### 数据集处理

```bash
# 双模态数据集创建和预处理
cd dual_dataset
python d_dataset_main.py
```

### 运行主控制程序

```bash
# 启动PyQt5界面和硬件控制
cd src
python main.py
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
- `dual_yolo/models/yolo11x-dseg-crossattn.yaml`: 跨模态注意力配置
- `dual_yolo/models/yolo11x-dseg-weighted-fusion.yaml`: 加权融合配置
- `dual_yolo/models/yolo11x-dseg-concat-compress.yaml`: 拼接融合配置

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
- `dual_yolo/d_model_train.py`: 双模态训练脚本  
- `dual_yolo/d_model_test.py`: 模型测试脚本

### 数据处理
- `dual_dataset/d_dataset_main.py`: 双模态数据集构建主程序
- `dual_dataset/d_dataset_creation.py`: 数据集创建逻辑
- `dual_dataset/rawdata_crop.py`: 原始数据预处理

### 自定义库修改
- `ultralytics/nn/modules/fusion.py`: CrossModalAttention实现
- `ultralytics/nn/tasks.py`: 模型解析器修改
- `ultralytics/data/utils.py`: 双模态数据加载支持

## 开发注意事项

1. **依赖管理**: 项目使用了大量深度学习和硬件控制库，注意CUDA版本兼容性
2. **路径配置**: 训练脚本中的路径配置需要根据实际部署环境调整
3. **硬件要求**: 双模态训练需要大容量GPU内存，推荐RTX 3090或以上
4. **数据对齐**: 双模态训练要求白光和蓝光图像严格对应，文件名规范为核心
5. **模型融合**: 跨模态注意力机制是项目核心创新，修改时需考虑计算效率

## 常见问题

1. **CUDA内存不足**: 降低batch_size或使用gradient checkpointing
2. **数据加载错误**: 检查双模态数据集中白光和蓝光图像的文件名对应关系
3. **训练不收敛**: 建议先用加权融合预训练，再切换到跨模态注意力
4. **硬件通信异常**: 检查串口权限和设备连接状态

## 模型权重

- `weights/yolo11x-seg.pt`: 预训练分割权重
- `dual_yolo/weights/dual_yolo11x.pt`: 双模态预训练权重