# Dual-Modal Dataset Processing Scripts

这个文档描述了双模态数据集创建、增强和可视化的完整流程。

## 概述

该脚本集合用于将单模态血液扫描数据转换为双模态（蓝光+白光）训练数据集，并进行数据增强处理。

## 核心组件

### 1. DatasetConfig (`d_dataset_config.py`)
集中配置管理器，负责所有路径和参数的统一管理。

**主要功能：**
- 管理项目根目录、版本号、数据集分割
- 定义源数据集和目标数据集路径
- 配置数据增强策略
- 自动生成增强数据集目录路径

**关键参数：**
- `version`: 数据集版本号（默认1）
- `split`: 数据集分割（train/valid/test）
- `strategies`: 数据增强策略字典

### 2. DualModalDatasetCreator (`d_dataset_creation.py`)
负责将单模态数据转换为双模态数据集。

**处理流程：**
1. 重命名标签文件（去除文件名中的冗余部分）
2. 复制标签文件到目标目录，文件名添加`_0`后缀
3. 读取源数据集中的JPG图像
4. 根据文件名前缀匹配BMP格式的蓝光和白光图像
5. 转换BMP为JPG格式并保存到目标目录，文件名添加`_0`后缀

**输出目录结构：**
```
{split}_augmented_9/
├── images_b/    # 蓝光图像（_0后缀）
├── images_w/    # 白光图像（_0后缀）
└── labels/      # 标签文件（_0后缀）
```

### 3. DataAugmenter (`d_dataset_augmentation.py`)
对双模态数据进行数据增强处理。

**增强策略：**
- `'0'`: 原图（由DualModalDatasetCreator创建，跳过处理）
- `'1'`: 旋转5° + 模糊1.5
- `'2'`: 旋转-5° + 模糊1.5
- `'3'`: 旋转5° + 曝光0.9
- `'4'`: 旋转-5° + 曝光1.1
- `'5'`: 旋转10° + 亮度1.15 + 模糊1.2
- `'6'`: 旋转-10° + 亮度0.85 + 模糊1.2
- `'7'`: 旋转5° + 曝光0.9 + 模糊1.2
- `'8'`: 旋转-5° + 曝光1.1 + 模糊1.2

**处理细节：**
- 自动跳过策略'0'（避免重复保存）
- 同时处理蓝光和白光图像
- 对标签中的多边形坐标应用相同变换
- 确保图像尺寸一致性

### 4. DualModalVisualizer (`d_dataset_visulize.py`)
可视化双模态数据集和标注信息。

**功能：**
- 随机选择图像样本进行显示
- 并排显示蓝光和白光图像
- 在图像上绘制多边形标注
- 支持交互式浏览

## 文件命名规则

所有输出文件都遵循统一的命名规则：

### 原始文件名格式
```
yyyy-mm-dd_HHMMSS_id_type_number.jpg
例如：2022-03-28_103204_17_T5_2412.jpg
```

### 输出文件名格式
```
{原始文件名}_{策略编号}.{扩展名}
例如：2022-03-28_103204_17_T5_2412_0.jpg（原图）
     2022-03-28_103204_17_T5_2412_1.jpg（策略1增强）
```

**重要说明：**
- `_0`后缀表示原图（未经变换的转换版本）
- `_1`到`_8`后缀表示不同的增强策略
- 蓝光、白光图像和标签文件使用相同的命名规则

## 使用方法

### 基本使用
```bash
cd scripts
python d_dataset_main.py
```

### 单独使用各组件
```python
from d_dataset_config import DatasetConfig
from d_dataset_creation import DualModalDatasetCreator
from d_dataset_augmentation import DataAugmenter
from d_dataset_visulize import DualModalVisualizer

# 配置
config = DatasetConfig(version=1, split="train")

# 创建双模态数据
creator = DualModalDatasetCreator(config)
creator.run()

# 数据增强
augmenter = DataAugmenter(config)
augmenter.augment_dataset()

# 可视化（可选）
visualizer = DualModalVisualizer(config)
visualizer.run()
```

## 目录结构

### 输入结构
```
datasets/Dual-Modal-1504-500-0/
├── train/
│   ├── images/    # 原始JPG图像
│   └── labels/    # YOLO格式标签
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

data/
├── rawdata_cropped/class1/       # 蓝光BMP文件
└── rawdata_cropped_white/class1/ # 白光BMP文件
```

### 输出结构
```
datasets/Dual-Modal-1504-500-1/
├── train_augmented_9/
│   ├── images_b/  # 蓝光图像（9个策略 × N张图）
│   ├── images_w/  # 白光图像（9个策略 × N张图）
│   └── labels/    # 标签文件（9个策略 × N张图）
├── valid_augmented_9/
│   ├── images_b/
│   ├── images_w/
│   └── labels/
└── test_augmented_9/
    ├── images_b/
    ├── images_w/
    └── labels/
```

## 关键设计决策

### 1. 统一文件名后缀
所有输出文件都添加策略后缀（`_0`到`_8`），确保命名一致性和可追溯性。

### 2. 跳过策略'0'重复处理
DualModalDatasetCreator直接生成`_0`文件，DataAugmenter检测并跳过策略'0'，避免重复保存。

### 3. 直接输出到最终目录
target_dataset直接指向增强目录，避免中间非增强数据的冗余存储。

### 4. 集中配置管理
通过DatasetConfig实现路径和参数的统一管理，便于维护和修改。

## 故障排除

### 常见问题

1. **找不到匹配的BMP文件**
   - 检查rawdata目录路径是否正确
   - 确认文件名前缀匹配逻辑

2. **图像尺寸不匹配**
   - 脚本会自动调整白光图像尺寸以匹配蓝光图像

3. **标签文件缺失**
   - 确认源数据集labels目录存在
   - 检查标签文件命名是否与图像文件对应

### 性能考虑

- 大数据集处理可能需要较长时间
- 建议先在小数据集上测试
- 磁盘空间需求：原始数据大小 × 9（策略数量）

## 扩展和自定义

### 添加新的增强策略
在DatasetConfig的strategies字典中添加新策略：
```python
strategies = {
    # 现有策略...
    '9': {'rotation': 15, 'brightness': 1.2, 'blur': 2.0},
}
```

### 修改文件命名规则
修改DualModalDatasetCreator和DataAugmenter中的文件名生成逻辑。

### 添加新的增强操作
在DataAugmenter中添加新的图像处理函数，并在augment_single_image中调用。