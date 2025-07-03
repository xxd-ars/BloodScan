# DualYOLO：蓝白光双骨干分割检测模型

DualYOLO是一个专为血液试管分层检测设计的双模态计算机视觉模型，利用蓝光和白光下的图像特性，通过三种可切换的特征融合机制提高分割精度。

## 模型架构

![DualYOLO架构图](outputs/dual_yolo_structure.png)

DualYOLO基于YOLO v11的基础架构，包含以下关键组件：

1. **双骨干网络**：分别处理蓝光和白光图像
2. **三种特征融合模式**：
   - AddFusion：带空间和通道感知的权重加法融合
   - CatFusion：通道拼接再压缩融合
   - XFormerFusion：基于Cross-Transformer的跨模态融合
3. **颈部网络**：与YOLO v11相同的FPN结构
4. **分割检测头**：实现边界框检测和实例分割

## 特征融合模块对比

![融合模块对比](outputs/fusion_modules_comparison.png)

### AddFusion
空间+通道双感知权重融合：
```python
α = Sigmoid(Conv3x3(ReLU(BN(Conv1x1(concat(F_b,F_w))))))
F_add = α ⊙ F_b + (1-α) ⊙ F_w
```

### CatFusion
通道拼接再压缩：
```python
F_cat = Conv1x1(concat(F_b, F_w))  # 2C → C
```

### XFormerFusion
双向Cross-Attention交互：
```python
Q_b,K_b,V_b = Conv1x1(F_b) → split (3d)  
Q_w,K_w,V_w = Conv1x1(F_w) → split (3d)  
F_b←w = Softmax(Q_b·K_wᵀ/√d) · V_w  
F_w←b = Softmax(Q_w·K_bᵀ/√d) · V_b  
F_ctr = φ(F_b + F_b←w) + φ(F_w + F_w←b)  
```
其中 φ = LayerNorm

## 环境要求

- Python 3.12+
- PyTorch 2.7.0+
- ultralytics (YOLO v11)
- 其他依赖见requirements.txt

## 安装

```bash
# 克隆仓库
git clone https://github.com/username/DualYOLO.git
cd DualYOLO

# 安装依赖
pip install -r requirements.txt
```

## 数据准备

数据目录结构应如下：
```
data/
  ├── blue/       # 蓝光图像
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  ├── white/      # 白光图像（文件名与蓝光图像匹配）
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  └── annotations.yaml  # 标注文件
```

## 使用方法

### 训练

```bash
python train.py --data_dir /path/to/data \
                --blue_dir blue \
                --white_dir white \
                --annotation_file /path/to/annotations.yaml \
                --fusion_type ctr \  # 选择融合类型：add, cat, 或 ctr
                --batch_size 8 \
                --epochs 100 \
                --learning_rate 0.001
```

### 推理

```bash
python detect.py --weights outputs/dual_yolo_ctr_best.pt \
                 --blue_dir /path/to/test/blue \
                 --white_dir /path/to/test/white \
                 --fusion_type ctr \
                 --conf_threshold 0.25
```

### 可视化与测试

```bash
# 生成模型架构图和执行单元测试
python test.py --all

# 或选择性执行
python test.py --test_fusion  # 测试融合模块输出维度一致性
python test.py --draw_structure  # 生成结构图
```

## 性能基准

### 不同融合模式对比 (mAP@0.5)

| 融合类型      | mAP@0.5 | mAP@0.5:0.95 | 分割mAP@0.5 | FPS |
|-------------|---------|--------------|------------|-----|
| AddFusion   | TBD     | TBD          | TBD        | TBD |
| CatFusion   | TBD     | TBD          | TBD        | TBD |
| XFormerFusion | TBD   | TBD          | TBD        | TBD |

### 单模态与双模态对比

| 模型         | mAP@0.5 | 分割mAP@0.5 | FPS |
|-------------|---------|-----------|-----|
| 仅蓝光（YOLO v11） | TBD  | TBD      | TBD |
| 仅白光（YOLO v11） | TBD  | TBD      | TBD |
| DualYOLO (XFormer) | TBD | TBD    | TBD |

## 结果示例

待添加检测结果图像示例。

## 引用

如果您在研究中使用了本项目，请引用：

```
待添加引用信息
```

## 许可证

MIT

## 联系方式

- 作者：待添加
- 邮箱：待添加
