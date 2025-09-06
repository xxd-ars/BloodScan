# CrossModalAttention 注意力可视化功能

## 功能概述

本实现为双模态YOLO模型添加了跨模态注意力可视化功能，可以直观展示蓝光特征如何通过注意力机制从白光特征中获取信息。

## 核心功能

1. **多层级注意力捕获**: 自动识别并捕获模型中所有CrossModalAttention层的注意力权重
2. **空间热力图生成**: 将Token级注意力权重映射到原始图像像素空间
3. **综合可视化展示**: 在单一图像中展示原图、预测结果和注意力热力图
4. **智能网格布局**: 自动排列多个可视化结果

## 使用方法

### 配置参数

直接编辑 `d_model_attention_vis.py` 文件中的参数配置区域：

```python
# ===== 参数配置区域 =====
# 模型配置
MODEL_PATH = "dual_yolo/runs/segment/dual_modal_train_crossattn-30epoch/weights/best.pt"
MODEL_YAML = "dual_yolo/models/yolo11x-dseg-crossattn.yaml"

# 输入数据（二选一）
# 方式1: 使用.npy双模态文件（推荐）
NPY_PATH = "datasets/Dual-Modal-1504-500-1-6ch/test/images/2022-03-28_103204_17_T5_2412_0.npy"

# 方式2: 使用分离的蓝光/白光图像（向后兼容）
BLUE_IMG = None  # "datasets/Dual-Modal-1504-500-1/test/images_b/xxx.jpg"
WHITE_IMG = None  # "datasets/Dual-Modal-1504-500-1/test/images_w/yyy.jpg"

# 输出配置
OUTPUT_PATH = "attention_visualization_result.png"
```

### 运行方式

```bash
python d_model_attention_vis.py
```

### 参数说明

- `MODEL_PATH`: 训练好的CrossModalAttention模型权重文件路径
- `MODEL_YAML`: 模型配置文件路径
- `NPY_PATH`: .npy双模态文件路径（优先选择）
- `BLUE_IMG/WHITE_IMG`: 分离的蓝光/白光图像路径（向后兼容）
- `OUTPUT_PATH`: 输出可视化结果路径

### 测试功能

运行测试脚本验证环境配置：

```bash
python test_attention_vis.py
```

## 输出结果解读

### 可视化网格布局

生成的结果图包含以下内容：

1. **Blue Light**: 原始蓝光图像
2. **White Light**: 原始白光图像
3. **Predictions**: 模型预测结果（检测框+分割掩码）
4. **Attention-P3/P4/P5**: 各层级注意力热力图叠加

### 注意力热力图含义

- **红色区域**: 高注意力权重，表示蓝光在这些位置heavily依赖白光信息
- **蓝色区域**: 低注意力权重，表示蓝光在这些位置较少使用白光信息
- **渐变过渡**: 显示注意力权重的空间分布模式

### 预期注意力模式

在血液分层检测中，理想的注意力模式应该：

1. **集中在分界面**: 注意力应主要集中在血液分层边界区域
2. **P3精细化**: 高分辨率P3层显示精细的边界细节关注
3. **P5全局化**: 低分辨率P5层显示整体结构关注
4. **空间一致性**: 注意力分布应与血液分层结构相关

## 技术实现细节

### 修改的文件

1. **ultralytics/nn/modules/fusion.py**: 
   - 添加注意力权重保存功能
   - 实现Token到像素空间的映射
   - 提供可视化控制接口

2. **dual_yolo/d_model_attention_vis.py**: 
   - 主要可视化脚本
   - 自动识别注意力模块
   - 生成综合可视化结果

### 关键技术点

1. **注意力权重捕获**: 在`_local_attention_vectorized`方法中保存Softmax后的注意力权重
2. **空间映射**: 通过双线性插值将Token级权重上采样到像素级
3. **多层融合**: 支持同时可视化P3/P4/P5三个层级的注意力
4. **内存优化**: 使用`.detach().cpu()`避免梯度计算和GPU内存占用

## 调试和分析

### 常见问题

1. **注意力权重全零**: 检查模型是否正确加载CrossModalAttention模块
2. **热力图不清晰**: 调整`alpha`参数控制叠加透明度
3. **尺寸不匹配**: 确保输入图像尺寸与训练时一致

### 性能分析工具

可以通过修改`visualize`方法添加性能分析：

```python
# 在attention_maps字典中查看各层注意力统计
for name, attn_map in attention_maps.items():
    print(f"{name}: min={attn_map.min():.4f}, max={attn_map.max():.4f}, mean={attn_map.mean():.4f}")
```

## 扩展应用

### 模型调试

- 检查注意力是否集中在预期区域
- 分析不同层级的注意力模式差异
- 验证跨模态融合的有效性

### 算法优化

- 根据注意力分布调整token_size和neighbor_size参数
- 分析注意力稀疏性指导模型压缩
- 评估不同融合策略的注意力模式

### 医学分析

- 可视化模型对血液分层边界的关注程度
- 分析模型在不同血液样本上的注意力一致性
- 为医生提供模型决策过程的可解释性

## 注意事项

1. **模型兼容性**: 仅支持包含CrossModalAttention的模型
2. **内存使用**: 大图像可能占用较多内存，建议在GPU环境运行
3. **可视化质量**: 热力图质量取决于注意力权重的分布特性
4. **实时性**: 当前版本主要用于离线分析，不适合实时应用