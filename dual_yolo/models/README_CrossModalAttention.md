# CrossModalAttention 跨模态注意力机制

## 设计理念

基于血液分层检测的具体场景特性，我们设计了一种高效的跨模态注意力机制：

### 核心原则
1. **蓝光主导**：蓝光特征作为主体信号载体，白光提供辅助信息
2. **单向查询**：仅蓝光查询白光，避免对称计算的冗余
3. **局部约束**：基于血液样本空间对齐的先验知识，限制注意力搜索范围
4. **Token级别**：使用细粒度Token来平衡计算效率与空间灵活性

### 算法流程
```
输入：蓝光特征图 [B, C, H, W]、白光特征图 [B, C, H, W]
1. Token化：将特征图分割为 token_size×token_size 的空间块
2. 投影：生成Query(蓝光)、Key/Value(白光)
3. 局部注意力：每个蓝光Token只查询邻近区域的白光Token
4. 特征增强：用注意力加权的白光信息增强蓝光特征
5. 重组：将增强后的Token重组为特征图
输出：增强后的蓝光特征图 [B, C, H, W]
```

## 技术特点

### 高效性
- **局部约束**：neighbor_size 限制搜索范围，大幅减少计算量
- **Token机制**：避免像素级全局注意力的 O(N²) 复杂度
- **单向设计**：相比双向注意力减少50%计算量

### 针对性
- **领域先验**：利用血液检测中白膜层位置一致的特点
- **多尺度适配**：支持不同特征层级的不同Token粒度
- **空间感知**：保持局部空间感受野处理微小对齐误差

## 参数说明

### CrossModalAttention参数
```python
CrossModalAttention(c1, c2=None, token_size=4, neighbor_size=3)
```

- `c1`: 输入通道数
- `c2`: 输出通道数（默认与c1相同）
- `token_size`: Token空间大小（如4表示4×4像素块）
- `neighbor_size`: 邻近搜索范围（如3表示3×3邻近Token）

### 建议配置
```yaml
# P3层级：高分辨率，需要精细交互
token_size: 4, neighbor_size: 3

# P4层级：中等分辨率
token_size: 6, neighbor_size: 3

# P5层级：低分辨率，可用粗粒度
token_size: 8, neighbor_size: 5
```

## 使用示例

### 1. 独立模块测试
```python
from ultralytics.nn.modules.fusion import CrossModalAttention

# 创建模块
cross_attn = CrossModalAttention(c1=256, c2=256, token_size=4, neighbor_size=3)

# 前向传播
blue_feat = torch.randn(2, 256, 32, 32)
white_feat = torch.randn(2, 256, 32, 32)
enhanced_feat = cross_attn([blue_feat, white_feat])
```

### 2. 在YAML配置中使用
```yaml
head:
  # 跨模态注意力融合层
  - [[blue_layer_idx, white_layer_idx], 1, CrossModalAttention, [channels, channels, token_size, neighbor_size]]
```

### 3. 完整模型配置
参考 `yolo11n-dseg-crossattn.yaml` 获取完整的模型配置示例。

## 计算复杂度分析

### 传统全局注意力
- 复杂度：O(N²) where N = H×W
- 内存：O(N²×C)

### 我们的局部Token注意力
- 复杂度：O(N×K) where K = neighbor_size²×token_size²
- 内存：O(N×K×C)
- 当 K << N 时，大幅降低计算量

### 实际效果
以 320×320 输入为例：
- P3层(40×40): 传统需要 1600² 注意力计算，我们只需 1600×(3²×4²) = 1600×144
- 计算量减少约 91%

## 与现有方案对比

| 方案 | 计算复杂度 | 空间灵活性 | 实现难度 | 适用场景 |
|------|------------|------------|----------|----------|
| 空间对齐+变形 | 高 | 高 | 高 | 大视角差异 |
| 全局跨注意力 | 极高 | 高 | 中 | 通用场景 |
| 简单拼接融合 | 低 | 低 | 低 | 完美对齐 |
| **我们的方案** | **中** | **中高** | **中** | **血液检测** |

## 训练建议

1. **渐进训练**：先用简单融合预训练，再切换到跨模态注意力
2. **学习率调整**：注意力模块使用较小学习率避免不稳定
3. **数据增强**：保持空间对齐性，避免过度几何变换
4. **消融实验**：对比不同token_size和neighbor_size的效果 