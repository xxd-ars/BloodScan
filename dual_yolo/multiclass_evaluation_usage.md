# 多类别YOLO评估使用指南

## 功能简介

新的多类别评估功能支持同时评估血液试管的三个层级：
- **Class 0**: 血清层 (使用JSON标注点位0,1,2,3)
- **Class 1**: 白膜层 (使用JSON标注点位2,3,4,5) - 原有功能
- **Class 2**: 血浆层 (使用JSON标注点位4,5,6+底部参考点)

## 基本用法

### 1. 单类别评估（向后兼容）
```python
# 保持原有行为，只评估Class 1
evaluate_dual_yolo_model(fusion_name='weighted-fusion', debug=True)
```

### 2. 多类别评估
```python
# 评估所有三个类别
evaluate_dual_yolo_model(
    fusion_name='weighted-fusion', 
    debug=True, 
    include_augmented=True,
    evaluate_classes=[0, 1, 2],  # 评估所有类别
    conf_threshold=0.3           # 降低置信度阈值
)
```

### 3. 自定义类别组合
```python
# 只评估血清层和白膜层
evaluate_dual_yolo_model(
    fusion_name='crossattn',
    evaluate_classes=[0, 1],
    conf_threshold=0.5
)
```

## 参数说明

- **evaluate_classes**: 要评估的类别列表，默认`[1]`保持向后兼容
- **conf_threshold**: YOLO推理置信度阈值，默认`0.5`
- **其他参数**: 与原版本保持一致

## 输出文件

### 统一输出文件
- `{filename}_evaluation.jpg`: 检测成功的可视化（包含所有评估类别）
- `{filename}_no_detection.jpg`: 未检测到任何类别的可视化

## 可视化颜色方案

### 多类别模式 (evaluate_classes包含多个类别)
- **所有真实标注点**: 红色圆点
- **Class 0预测点**: 黄色十字
- **Class 1预测点**: 绿色十字 (保持原有)
- **Class 2预测点**: 蓝色十字
- **预测连接线**: 白色细线

### 单类别模式 (向后兼容)
- **真实标注点**: 红色圆点
- **预测点**: 绿色十字
- **预测连接线**: 白色细线

## 注意事项

1. **向后兼容**: 不传入新参数时，行为与原版本完全一致
2. **JSON标注**: 确保JSON文件包含足够的标注点位(至少7个)
3. **置信度调节**: 多类别评估时可能需要调低confidence阈值
4. **性能影响**: 多类别评估会增加计算时间，但影响较小

## 故障排除

### 常见问题
1. **标注点不足**: 检查JSON文件是否包含完整的7个标注点
2. **检测率低**: 尝试降低conf_threshold参数
3. **类别0/2无检测**: 确认模型是否训练了对应类别

### 调试模式
```python
# 启用详细调试信息
evaluate_dual_yolo_model(
    fusion_name='weighted-fusion',
    debug=True,  # 显示路径检查和模型加载信息
    evaluate_classes=[0, 1, 2]
)
```