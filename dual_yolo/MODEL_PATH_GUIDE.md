# 模型路径配置指南

本文档说明评估脚本中不同模型名称对应的文件路径逻辑。

## 路径生成规则

评估脚本使用以下规则生成模型路径：

```python
model_yaml = f'dual_yolo/models/yolo11x-dseg-{model_name}.yaml'
model_pt = f'dual_yolo/runs/segment/dual_modal_{train_mode}_{model_name}/weights/best.pt'
save_dir = f'dual_yolo/evaluation_results_v4/conf_{conf_medical}/{model_name}/'
```

## 常用模型示例

### 1. 标准模型：`crossattn-precise`

**模型架构文件**:
```
dual_yolo/models/yolo11x-dseg-crossattn-precise.yaml
```

**模型权重文件** (train_mode='pretrained'):
```
dual_yolo/runs/segment/dual_modal_pretrained_crossattn-precise/weights/best.pt
```

**评估结果保存路径** (conf_medical=0.5):
```
dual_yolo/evaluation_results_v4/conf_0.5/crossattn-precise/
├── metrics_crossattn-precise.json
└── visualizations/
    ├── 2024-01-01_120000_001_b_0_0_success.jpg
    └── ...
```

---

### 2. 单模态模型：`id-blue`

**模型架构文件**:
```
dual_yolo/models/yolo11x-dseg-id-blue.yaml
```

**模型权重文件**:
```
dual_yolo/runs/segment/dual_modal_pretrained_id-blue/weights/best.pt
```

**评估结果保存路径**:
```
dual_yolo/evaluation_results_v4/conf_0.5/id-blue/
├── metrics_id-blue.json
└── visualizations/
```

---

### 3. 变体模型：`id-blue-5`

**模型架构文件**:
```
dual_yolo/models/yolo11x-dseg-id-blue-5.yaml
```

**模型权重文件**:
```
dual_yolo/runs/segment/dual_modal_pretrained_id-blue-5/weights/best.pt
```

**评估结果保存路径**:
```
dual_yolo/evaluation_results_v4/conf_0.5/id-blue-5/
├── metrics_id-blue-5.json
└── visualizations/
```

---

### 4. 其他融合策略

#### `weighted-fusion`
```
架构: dual_yolo/models/yolo11x-dseg-weighted-fusion.yaml
权重: dual_yolo/runs/segment/dual_modal_pretrained_weighted-fusion/weights/best.pt
结果: dual_yolo/evaluation_results_v4/conf_0.5/weighted-fusion/
```

#### `concat-compress`
```
架构: dual_yolo/models/yolo11x-dseg-concat-compress.yaml
权重: dual_yolo/runs/segment/dual_modal_pretrained_concat-compress/weights/best.pt
结果: dual_yolo/evaluation_results_v4/conf_0.5/concat-compress/
```

## 特殊情况处理

### `crossattn-30epoch` 模型

这是一个特殊的模型变体，使用 `crossattn` 的架构但训练了30个epoch：

```python
if model_name == 'crossattn-30epoch':
    yaml_path = 'dual_yolo/models/yolo11x-dseg-crossattn.yaml'  # 使用crossattn架构
else:
    yaml_path = f'dual_yolo/models/yolo11x-dseg-{model_name}.yaml'
```

**路径映射**:
- 架构文件: `dual_yolo/models/yolo11x-dseg-crossattn.yaml` (注意不是 `-30epoch`)
- 权重文件: `dual_yolo/runs/segment/dual_modal_pretrained_crossattn-30epoch/weights/best.pt`
- 结果路径: `dual_yolo/evaluation_results_v4/conf_0.5/crossattn-30epoch/`

## 不同训练模式

### `train_mode='pretrained'` (默认)
```
dual_yolo/runs/segment/dual_modal_pretrained_{model_name}/weights/best.pt
```

### `train_mode='scratch'`
```
dual_yolo/runs/segment/dual_modal_scratch_{model_name}/weights/best.pt
```

## 检查模型是否存在

在运行评估前，确保以下文件存在：

```bash
# 1. 检查模型架构文件
ls dual_yolo/models/yolo11x-dseg-{model_name}.yaml

# 2. 检查模型权重文件
ls dual_yolo/runs/segment/dual_modal_pretrained_{model_name}/weights/best.pt

# 示例: 检查 id-blue-5
ls dual_yolo/models/yolo11x-dseg-id-blue-5.yaml
ls dual_yolo/runs/segment/dual_modal_pretrained_id-blue-5/weights/best.pt
```

## 添加新模型

要评估一个新模型（如 `id-blue-5`），需要：

1. **创建模型架构文件**:
   ```bash
   dual_yolo/models/yolo11x-dseg-id-blue-5.yaml
   ```

2. **训练模型并保存权重**:
   ```bash
   # 训练后权重自动保存到：
   dual_yolo/runs/segment/dual_modal_pretrained_id-blue-5/weights/best.pt
   ```

3. **在评估脚本中添加模型名**:
   ```python
   models = ['id-blue', 'id-blue-5', 'id-white', ...]
   ```

4. **运行评估**:
   ```bash
   python d_model_evaluate_v4_parallel.py
   ```

5. **查看结果**:
   ```bash
   cat dual_yolo/evaluation_results_v4/conf_0.5/id-blue-5/metrics_id-blue-5.json
   ```

## 路径总结表

| 模型名称 | 架构文件 | 权重文件 | 结果目录 |
|---------|---------|---------|---------|
| `id-blue` | `yolo11x-dseg-id-blue.yaml` | `dual_modal_pretrained_id-blue/weights/best.pt` | `conf_0.5/id-blue/` |
| `id-blue-5` | `yolo11x-dseg-id-blue-5.yaml` | `dual_modal_pretrained_id-blue-5/weights/best.pt` | `conf_0.5/id-blue-5/` |
| `id-white` | `yolo11x-dseg-id-white.yaml` | `dual_modal_pretrained_id-white/weights/best.pt` | `conf_0.5/id-white/` |
| `crossattn` | `yolo11x-dseg-crossattn.yaml` | `dual_modal_pretrained_crossattn/weights/best.pt` | `conf_0.5/crossattn/` |
| `crossattn-precise` | `yolo11x-dseg-crossattn-precise.yaml` | `dual_modal_pretrained_crossattn-precise/weights/best.pt` | `conf_0.5/crossattn-precise/` |
| `weighted-fusion` | `yolo11x-dseg-weighted-fusion.yaml` | `dual_modal_pretrained_weighted-fusion/weights/best.pt` | `conf_0.5/weighted-fusion/` |
| `concat-compress` | `yolo11x-dseg-concat-compress.yaml` | `dual_modal_pretrained_concat-compress/weights/best.pt` | `conf_0.5/concat-compress/` |
| `crossattn-30epoch` | `yolo11x-dseg-crossattn.yaml` ⚠️ | `dual_modal_pretrained_crossattn-30epoch/weights/best.pt` | `conf_0.5/crossattn-30epoch/` |

⚠️ 注意: `crossattn-30epoch` 使用 `crossattn` 的架构文件，这是唯一的特殊情况。
