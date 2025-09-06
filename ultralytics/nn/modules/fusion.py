# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Dual-modal fusion modules for YOLO."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("ConcatCompress", "WeightedFusion", "CrossModalAttention", "MultiLayerCrossModalAttention")


class ConcatCompress(nn.Module):
    """
    Concatenation followed by compression fusion module.
    
    This module concatenates features from two modalities and then compresses
    them back to the original channel dimension using 1x1 convolution.
    """

    def __init__(self, c1, c2=None):
        """
        Initialize ConcatCompress module.
        
        Args:
            c1 (int): Number of input channels (for each modality).
            c2 (int, optional): Number of output channels. If None, defaults to c1.
        """
        super().__init__()
        if c2 is None:
            c2 = c1
        self.c1 = c1
        self.c2 = c2
        # 使用1x1卷积将通道数从c1*2压缩到c2
        # c1*2: 输入通道数(两个模态特征拼接后的通道数)
        # c2: 输出通道数(压缩后的通道数)
        # kernel_size=1: 1x1卷积
        # stride=1: 步长为1
        self.compress = Conv(c1 + c2, (c1 + c2)//2, 1, 1)
    
    def forward(self, x):
        """
        Forward pass of ConcatCompress.
        
        Args:
            x (list): List containing two feature tensors [blue_feat, white_feat].
            
        Returns:
            torch.Tensor: Fused feature tensor.
        """
        # Concatenate along channel dimension
        x = torch.cat(x, 1)
        # Compress back to original channels
        return self.compress(x)


class WeightedFusion(nn.Module):
    """
    Spatial and content-aware weighted fusion module.
    
    This module learns adaptive weights based on both spatial patterns and global
    content to fuse features from two modalities intelligently.
    """

    def __init__(self, c1, c2=None):
        """
        Initialize WeightedFusion module.
        
        Args:
            c1 (int): Number of input channels (for each modality).
            c2 (int, optional): Number of output channels. If None, defaults to c1.
        """
        super().__init__()
        if c2 is None:
            c2 = c1
        self.c1 = c1
        self.c2 = c2
        
        # Spatial weight prediction network using standard Conv modules
        self.spatial_conv1 = Conv(c1 * 2, c1 // 4, 3, 1)
        self.spatial_conv2 = Conv(c1 // 4, c1 // 8, 3, 1)
        self.spatial_out = nn.Sequential(
            nn.Conv2d(c1 // 8, 1, 1),
            nn.Sigmoid()
        )
        
        # 全局上下文预测器 - 使用自适应平均池化将特征图压缩为1x1
        # AdaptiveAvgPool2d(1)会自动计算池化窗口大小,将任意输入尺寸的特征图池化为1x1
        # 这样可以获取全局的特征信息,用于预测全局权重
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 使用1x1卷积处理全局特征,不使用BN避免训练问题
        # 输入通道数为c1*2(拼接的双模态特征),输出通道数为c1//4 
        self.global_conv = nn.Sequential(
            nn.Conv2d(c1 * 2, c1 // 4, 1),  # 1x1卷积降维
            nn.SiLU()  # SiLU激活函数
        )
        
        # 最后输出单通道的全局权重,使用Sigmoid归一化到0-1
        self.global_out = nn.Sequential(
            nn.Conv2d(c1 // 4, 1, 1),  # 1x1卷积得到权重
            nn.Sigmoid()  # Sigmoid归一化
        )
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Output projection if needed
        if c1 != c2:
            self.proj = Conv(c1, c2, 1, 1)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x):
        """
        Forward pass of WeightedFusion.
        
        Args:
            x (list): List containing two feature tensors [blue_feat, white_feat].
            
        Returns:
            torch.Tensor: Fused feature tensor with adaptive weighting.
        """
        blue_feat, white_feat = x
        
        # Concatenate features for weight prediction
        concat_feat = torch.cat([blue_feat, white_feat], dim=1)
        
        # Predict spatial-aware weights
        spatial_weight = self.spatial_conv1(concat_feat)
        spatial_weight = self.spatial_conv2(spatial_weight)
        spatial_weight = self.spatial_out(spatial_weight)
        
        # Predict global content-aware weight
        global_feat = self.global_pool(concat_feat)
        global_weight = self.global_conv(global_feat)
        global_weight = self.global_out(global_weight)
        
        # Combine spatial and global weights with temperature scaling
        final_weight = spatial_weight * global_weight * self.temperature
        
        # Apply weighted fusion: w * blue + (1-w) * white
        fused_feat = final_weight * blue_feat + (1 - final_weight) * white_feat
        
        # Apply output projection
        return self.proj(fused_feat)


class CrossModalAttention(nn.Module):
    """
    跨模态注意力模块：蓝光特征查询白光特征进行自我增强
    Cross-modal attention module for blue light querying white light features.
    
    核心思想：蓝光特征作为"问题"(Query)，去白光特征中寻找"答案"(Key-Value)，
    通过注意力权重来选择性地融合白光信息，从而增强蓝光特征。
    Blue light features act as queries to enhance themselves using white light features.
    Uses token-based local attention for efficiency.
    
    使用基于Token的局部注意力来提高计算效率。
    """

    def __init__(self, c1, c2=None, token_size=4, neighbor_size=3):
        """
        Initialize CrossModalAttention module.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int, optional): Number of output channels. If None, defaults to c1.
            token_size (int): Token size for spatial partitioning. (default 4x4 pixels)
            neighbor_size (int): Neighbor search range for local attention. (default 3x3 neighbors)
        """
        super().__init__()
        if c2 is None:
            c2 = c1
        self.c1 = c1
        self.c2 = c2
        self.token_size = token_size
        self.neighbor_size = neighbor_size
        
        # 可视化相关属性
        self.enable_visualization = False
        self.attention_maps = None
        self.last_input_size = None
        
        # Projection layers: 将特征投影到Query、Key、Value空间
        self.q_proj = nn.Conv2d(c1, c1, 1) # Blue  light -> Query
        self.k_proj = nn.Conv2d(c1, c1, 1) # White light -> Key
        self.v_proj = nn.Conv2d(c1, c1, 1) # White light -> Value

        ## Scale factor
        # self.scale = (c1 // (token_size * token_size)) ** -0.5
        
        # Corrected scale factor
        self.scale = (token_size * token_size) ** -0.5
        
        # Output projection
        self.out_proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        """
        前向传播：实现跨模态注意力特征增强
        Forward pass of CrossModalAttention.
        
        Args:
            x (list): List containing [blue_feat, white_feat] tensors.
            
        Returns:
            torch.Tensor: Enhanced blue features.
        """
        blue_feat, white_feat = x
        B, C, H, W = blue_feat.shape
        
        # 保存输入尺寸用于可视化
        if self.enable_visualization:
            self.last_input_size = (H, W)
        
        # ============================================================================
        # 步骤1：特征投影 - 生成Query、Key、Value
        # ============================================================================
        # Q：蓝光特征的"问题" - "我需要什么样的信息？"
        # K：白光特征的"索引" - "我有什么样的信息？"  
        # V：白光特征的"内容" - "我的具体信息是什么？"
        Q = self.q_proj(blue_feat)   # [B, C, H, W] - 蓝光的需求
        K = self.k_proj(white_feat)  # [B, C, H, W] - 白光的索引
        V = self.v_proj(white_feat)  # [B, C, H, W] - 白光的内容
        
        # ============================================================================
        # 步骤2：填充处理 - 确保尺寸能被token_size整除
        # ============================================================================
        pad_h = (self.token_size - H % self.token_size) % self.token_size
        pad_w = (self.token_size - W % self.token_size) % self.token_size
        if pad_h > 0 or pad_w > 0:
            Q = F.pad(Q, (0, pad_w, 0, pad_h))
            K = F.pad(K, (0, pad_w, 0, pad_h))
            V = F.pad(V, (0, pad_w, 0, pad_h))
            H_pad, W_pad = Q.shape[2], Q.shape[3]
        else:
            H_pad, W_pad = H, W
        
        # ============================================================================
        # 步骤3：特征Token化 - 将特征图分割成小块进行处理
        # ============================================================================
        # 目的：降低计算复杂度，从像素级别的O(N^2)降到Token级别的O(M^2)
        # 其中N=H*W（像素数），M=num_tokens（Token数），通常M << N
        
        num_h, num_w = H_pad // self.token_size, W_pad // self.token_size  # Token网格大小
        token_dim = self.token_size * self.token_size  # 每个Token的特征维度
        
        # 重塑操作详解：[B, C, H, W] -> [B, num_h, num_w, C, token_dim]
        # 1. view: 将H维度分割为(num_h, token_size)，W维度分割为(num_w, token_size)
        # 2. permute: 重新排列维度，将token的空间维度移到最后
        # 3. reshape: 将token内的空间维度展平为一维
        Q_tokens = Q.view(B, C, num_h, self.token_size, num_w, self.token_size).permute(0, 2, 4, 1, 3, 5).reshape(B, num_h, num_w, C, token_dim)
        K_tokens = K.view(B, C, num_h, self.token_size, num_w, self.token_size).permute(0, 2, 4, 1, 3, 5).reshape(B, num_h, num_w, C, token_dim)
        V_tokens = V.view(B, C, num_h, self.token_size, num_w, self.token_size).permute(0, 2, 4, 1, 3, 5).reshape(B, num_h, num_w, C, token_dim)

        # ============================================================================
        # 步骤4：核心注意力计算 - 这里实现特征增强的关键逻辑
        # ============================================================================
        enhanced_tokens = self._local_attention_vectorized(Q_tokens, K_tokens, V_tokens)
        
        # ============================================================================
        # 步骤5：特征重组 - 将增强后的Token重新组装成特征图
        # ============================================================================
        # 逆向操作：[B, num_h, num_w, C, token_dim] -> [B, C, H, W]
        
        # 5a. 恢复Token内的空间结构：token_dim -> (token_size, token_size)
        enhanced_tokens = enhanced_tokens.reshape(B, num_h, num_w, C, self.token_size, self.token_size)
        
        # 5b. 重新排列维度：将通道维度移到前面，为最终重塑做准备
        enhanced_tokens = enhanced_tokens.permute(0, 3, 1, 4, 2, 5)  # [B, C, num_h, token_size, num_w, token_size]
        
        # 5c. 合并Token网格和Token内部空间维度，重建完整特征图
        enhanced_feat = enhanced_tokens.reshape(B, C, H_pad, W_pad)
        
        # 5d. 去除之前添加的填充
        if pad_h > 0 or pad_w > 0:
            enhanced_feat = enhanced_feat[:, :, :H, :W]
        
        # ============================================================================
        # 步骤6：残差连接 - 特征增强的最终实现
        # ============================================================================
        # 关键：我们不是替换原始特征，而是在其基础上添加注意力增强的信息
        # enhanced_feat：从白光中学到的、经过注意力筛选的有用信息
        # blue_feat：原始蓝光特征
        # 两者相加：原始信息 + 跨模态增强信息 = 增强后的特征
        return self.out_proj(blue_feat + enhanced_feat)
    
    def _local_attention_vectorized(self, Q, K, V):
        """
        使用向量化实现局部注意力计算
        
        这是整个模块的核心：实现"用注意力加权的白光信息增强蓝光特征"
        
        Args:
            Q: [B, num_h, num_w, C, token_dim] - 蓝光Query tokens
            K: [B, num_h, num_w, C, token_dim] - 白光Key tokens  
            V: [B, num_h, num_w, C, token_dim] - 白光Value tokens
            
        Returns:
            [B, num_h, num_w, C, token_dim] - 增强后的tokens
        """
        B, num_h, num_w, C, token_dim = Q.shape
        num_tokens = num_h * num_w
        
        # ========================================================================
        # 子步骤1：提取局部邻域 - 实现"局部约束"的关键
        # ========================================================================
        # 使用unfold高效提取每个位置的邻域信息，避免Python循环
        kH = kW = self.neighbor_size  # 邻域大小（如3x3）
        padding = self.neighbor_size // 2  # 边界填充
        
        # 重塑K和V为适合unfold的格式：[B, C*token_dim, num_h, num_w]
        K_reshaped = K.permute(0, 3, 4, 1, 2).reshape(B, C * token_dim, num_h, num_w)
        V_reshaped = V.permute(0, 3, 4, 1, 2).reshape(B, C * token_dim, num_h, num_w)

        # unfold操作：为每个位置提取其邻域窗口
        # 结果形状：[B, C*token_dim*neighbor_size^2, num_tokens]
        K_unfolded = F.unfold(K_reshaped, kernel_size=(kH, kW), padding=padding)
        V_unfolded = F.unfold(V_reshaped, kernel_size=(kH, kW), padding=padding)
        
        # 重塑为方便后续计算的格式
        num_neighbors = kH * kW  # 邻域内的token数量（如3x3=9）
        K_neighbors = K_unfolded.reshape(B, C, token_dim, num_neighbors, num_tokens).permute(0, 4, 3, 1, 2)
        V_neighbors = V_unfolded.reshape(B, C, token_dim, num_neighbors, num_tokens).permute(0, 4, 3, 1, 2)
        # 最终形状：[B, num_tokens, num_neighbors, C, token_dim]
        
        # 将Q重塑为与neighbors匹配的格式
        Q_flat = Q.reshape(B, num_tokens, 1, C, token_dim)  # [B, num_tokens, 1, C, token_dim]
        
        # ========================================================================
        # 子步骤2：计算注意力相似度 - "蓝光问题"与"白光索引"的匹配程度
        # ========================================================================
        # 这里回答您的问题："计算得到相似度之后我们做什么？"
        
        # 2a. 计算点积相似度
        # einsum解释：对于每个蓝光token的query，计算它与所有邻近白光token的key的相似度
        # 'BLqCD, BLnCD -> BLCn' 表示：
        # - B: batch维度
        # - L: token位置维度  
        # - q: query维度(1)，n: neighbor维度
        # - C: 通道维度，D: token特征维度
        # 结果：每个蓝光token对其邻域内每个白光token的相似度分数
        attn = torch.einsum('BLqCD, BLnCD -> BLCn', Q_flat, K_neighbors) * self.scale
        
        # 2b. Softmax归一化：将相似度转换为概率分布
        # 含义：对于每个蓝光token，它的邻域白光tokens的重要性权重总和为1
        attn = F.softmax(attn, dim=-1)  # [B, num_tokens, C, num_neighbors]
        
        # 保存注意力权重用于可视化
        if self.enable_visualization:
            self.attention_maps = attn.detach().cpu()
        
        # ========================================================================
        # 子步骤3：加权求和 - 实现特征增强的核心操作
        # ========================================================================
        # 这一步回答了"如何增强原有特征"：
        
        # 3a. 用注意力权重对白光Value进行加权求和
        # einsum解释：
        # - attn[B,L,C,n]：每个位置L、通道C对邻域n的注意力权重
        # - V_neighbors[B,L,n,C,D]：邻域白光token的具体特征内容
        # - 结果[B,L,C,D]：加权融合后的特征，包含了来自白光的有用信息
        enhanced_token = torch.einsum('BLCn, BLnCD -> BLCD', attn, V_neighbors)
        
        # 3b. 特征增强的直观解释：
        # - 如果某个白光邻域token与当前蓝光token相似度高(attn大)，
        #   那么这个白光token的特征(V)会在最终结果中占更大比重
        # - 如果相似度低(attn小)，则该白光特征的贡献很小
        # - 最终结果是所有邻域白光特征的"智能混合"，混合权重由相似度决定
        
        # 重塑回原始Token网格格式
        return enhanced_token.reshape(B, num_h, num_w, C, token_dim)
    
    def get_attention_spatial_map(self, target_size=None):
        """
        将Token级注意力权重映射到原始图像空间
        
        Args:
            target_size (tuple): 目标图像尺寸 (H, W)，默认使用last_input_size
            
        Returns:
            torch.Tensor: 空间注意力图 [B, H, W] 或 None（如果未启用可视化）
        """
        if not self.enable_visualization or self.attention_maps is None:
            return None
        
        if target_size is None:
            target_size = self.last_input_size
        
        if target_size is None:
            return None
        
        H, W = target_size
        B, num_tokens, C, num_neighbors = self.attention_maps.shape
        
        # 计算Token网格尺寸
        pad_h = (self.token_size - H % self.token_size) % self.token_size
        pad_w = (self.token_size - W % self.token_size) % self.token_size
        H_pad, W_pad = H + pad_h, W + pad_w
        num_h, num_w = H_pad // self.token_size, W_pad // self.token_size
        
        # 对注意力权重进行平均聚合：[B, num_tokens, C, num_neighbors] -> [B, num_tokens]
        # 聚合策略：对通道和邻域维度求平均，得到每个token位置的整体注意力强度
        attn_avg = self.attention_maps.mean(dim=(2, 3))  # [B, num_tokens]
        
        # 重塑为Token网格：[B, num_tokens] -> [B, num_h, num_w]
        attn_grid = attn_avg.reshape(B, num_h, num_w)
        
        # 上采样到像素空间：[B, num_h, num_w] -> [B, H_pad, W_pad]
        attn_spatial = F.interpolate(
            attn_grid.unsqueeze(1),  # 添加通道维度 [B, 1, num_h, num_w]
            size=(H_pad, W_pad),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # 移除通道维度 [B, H_pad, W_pad]
        
        # 裁剪到原始尺寸
        if pad_h > 0 or pad_w > 0:
            attn_spatial = attn_spatial[:, :H, :W]
        
        return attn_spatial
    
    def enable_attention_visualization(self, enable=True):
        """启用/禁用注意力可视化"""
        self.enable_visualization = enable
        if not enable:
            self.attention_maps = None
            self.last_input_size = None


class MultiLayerCrossModalAttention(nn.Module):
    """
    多层跨模态注意力模块：通过多层注意力实现渐进式特征增强
    Multi-layer cross-modal attention module for progressive feature enhancement.
    
    核心设计理念：
    1. 第一层：建立基础的蓝光-白光特征对应关系
    2. 后续层：在前一层基础上进一步精化和增强特征
    3. 残差连接：确保信息流通和梯度传播
    4. 层间特征融合：每层都能获得来自白光的新信息
    
    相比单层注意力的优势：
    - 渐进式特征精化：逐层提升特征质量
    - 更强的表达能力：多层非线性变换
    - 层次化信息融合：不同层级捕获不同尺度的模态关系
    """

    def __init__(self, c1, c2=None, token_size=4, neighbor_size=3, num_layers=2):
        """
        Initialize MultiLayerCrossModalAttention module.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int, optional): Number of output channels. If None, defaults to c1.
            token_size (int): Token size for spatial partitioning. (default 4x4 pixels)
            neighbor_size (int): Neighbor search range for local attention. (default 3x3 neighbors)
            num_layers (int): Number of attention layers. (default 2)
        """
        super().__init__()
        if c2 is None:
            c2 = c1
        self.c1 = c1
        self.c2 = c2
        self.token_size = token_size
        self.neighbor_size = neighbor_size
        self.num_layers = num_layers
        
        # 创建多层注意力模块
        self.attention_layers = nn.ModuleList()
        for i in range(num_layers):
            # 每一层都有独立的Q、K、V投影
            layer = nn.ModuleDict({
                'q_proj': nn.Conv2d(c1, c1, 1),
                'k_proj': nn.Conv2d(c1, c1, 1), 
                'v_proj': nn.Conv2d(c1, c1, 1),
                'norm': nn.LayerNorm(c1),  # 层归一化稳定训练
            })
            self.attention_layers.append(layer)
        
        # Scale factor
        self.scale = (token_size * token_size) ** -0.5
        
        # 中间层的特征融合权重（学习如何组合不同层的信息）
        if num_layers > 1:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Output projection
        self.out_proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        """
        前向传播：实现多层跨模态注意力特征增强
        
        Args:
            x (list): List containing [blue_feat, white_feat] tensors.
            
        Returns:
            torch.Tensor: Enhanced blue features through multi-layer attention.
        """
        blue_feat, white_feat = x
        B, C, H, W = blue_feat.shape
        
        # 记录每一层的输出，用于最终的加权融合
        layer_outputs = []
        
        # 当前处理的蓝光特征（会在每层更新）
        current_blue = blue_feat
        
        # 逐层处理
        for layer_idx, layer_modules in enumerate(self.attention_layers):
            # ================================================================
            # 第一层：使用原始蓝光和白光特征
            # 后续层：使用增强后的蓝光特征查询原始白光特征
            # ================================================================
            
            # 特征投影
            Q = layer_modules['q_proj'](current_blue)  # 查询：当前蓝光特征
            K = layer_modules['k_proj'](white_feat)    # 键：始终使用原始白光特征
            V = layer_modules['v_proj'](white_feat)    # 值：始终使用原始白光特征
            
            # 执行单层注意力计算
            enhanced_feat = self._single_layer_attention(Q, K, V, H, W)
            
            # 层归一化（在通道维度上）
            # 需要将特征reshape为 [B, H*W, C] 格式进行LayerNorm
            B_norm, C_norm, H_norm, W_norm = enhanced_feat.shape
            enhanced_feat_norm = enhanced_feat.permute(0, 2, 3, 1).reshape(B_norm, H_norm * W_norm, C_norm)
            enhanced_feat_norm = layer_modules['norm'](enhanced_feat_norm)
            enhanced_feat = enhanced_feat_norm.reshape(B_norm, H_norm, W_norm, C_norm).permute(0, 3, 1, 2)
            
            # 残差连接：当前蓝光特征 + 本层增强特征
            current_blue = current_blue + enhanced_feat
            
            # 保存本层输出用于最终融合
            layer_outputs.append(enhanced_feat)
        
        # ================================================================
        # 多层信息融合：加权组合不同层的增强信息
        # ================================================================
        if self.num_layers > 1:
            # 使用可学习权重融合不同层的输出
            # 理念：早期层捕获基础对应，深层捕获复杂关系
            weighted_enhancement = torch.zeros_like(layer_outputs[0])
            for i, layer_output in enumerate(layer_outputs):
                weighted_enhancement += self.layer_weights[i] * layer_output
            
            # 最终结果：原始蓝光 + 加权融合的多层增强信息
            final_output = blue_feat + weighted_enhancement
        else:
            # 单层情况：直接使用当前蓝光特征
            final_output = current_blue
        
        return self.out_proj(final_output)
    
    def _single_layer_attention(self, Q, K, V, H, W):
        """
        单层注意力计算（复用CrossModalAttention的逻辑）
        
        Args:
            Q: Query features [B, C, H, W]
            K: Key features [B, C, H, W]  
            V: Value features [B, C, H, W]
            H, W: Original height and width
            
        Returns:
            torch.Tensor: Enhanced features from this attention layer
        """
        B, C = Q.shape[:2]
        
        # 填充处理
        pad_h = (self.token_size - H % self.token_size) % self.token_size
        pad_w = (self.token_size - W % self.token_size) % self.token_size
        if pad_h > 0 or pad_w > 0:
            Q = F.pad(Q, (0, pad_w, 0, pad_h))
            K = F.pad(K, (0, pad_w, 0, pad_h))
            V = F.pad(V, (0, pad_w, 0, pad_h))
            H_pad, W_pad = Q.shape[2], Q.shape[3]
        else:
            H_pad, W_pad = H, W
        
        # Token化
        num_h, num_w = H_pad // self.token_size, W_pad // self.token_size
        token_dim = self.token_size * self.token_size
        
        Q_tokens = Q.view(B, C, num_h, self.token_size, num_w, self.token_size).permute(0, 2, 4, 1, 3, 5).reshape(B, num_h, num_w, C, token_dim)
        K_tokens = K.view(B, C, num_h, self.token_size, num_w, self.token_size).permute(0, 2, 4, 1, 3, 5).reshape(B, num_h, num_w, C, token_dim)
        V_tokens = V.view(B, C, num_h, self.token_size, num_w, self.token_size).permute(0, 2, 4, 1, 3, 5).reshape(B, num_h, num_w, C, token_dim)

        # 注意力计算
        enhanced_tokens = self._local_attention_vectorized(Q_tokens, K_tokens, V_tokens)
        
        # 特征重组
        enhanced_tokens = enhanced_tokens.reshape(B, num_h, num_w, C, self.token_size, self.token_size)
        enhanced_tokens = enhanced_tokens.permute(0, 3, 1, 4, 2, 5)
        enhanced_feat = enhanced_tokens.reshape(B, C, H_pad, W_pad)
        
        # 去除填充
        if pad_h > 0 or pad_w > 0:
            enhanced_feat = enhanced_feat[:, :, :H, :W]
        
        return enhanced_feat
    
    def _local_attention_vectorized(self, Q, K, V):
        """
        局部注意力计算（与CrossModalAttention相同的实现）
        """
        B, num_h, num_w, C, token_dim = Q.shape
        num_tokens = num_h * num_w
        
        # 提取局部邻域
        kH = kW = self.neighbor_size
        padding = self.neighbor_size // 2
        
        K_reshaped = K.permute(0, 3, 4, 1, 2).reshape(B, C * token_dim, num_h, num_w)
        V_reshaped = V.permute(0, 3, 4, 1, 2).reshape(B, C * token_dim, num_h, num_w)

        K_unfolded = F.unfold(K_reshaped, kernel_size=(kH, kW), padding=padding)
        V_unfolded = F.unfold(V_reshaped, kernel_size=(kH, kW), padding=padding)
        
        num_neighbors = kH * kW
        K_neighbors = K_unfolded.reshape(B, C, token_dim, num_neighbors, num_tokens).permute(0, 4, 3, 1, 2)
        V_neighbors = V_unfolded.reshape(B, C, token_dim, num_neighbors, num_tokens).permute(0, 4, 3, 1, 2)
        
        Q_flat = Q.reshape(B, num_tokens, 1, C, token_dim)
        
        # 计算注意力
        attn = torch.einsum('BLqCD, BLnCD -> BLCn', Q_flat, K_neighbors) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # 加权求和
        enhanced_token = torch.einsum('BLCn, BLnCD -> BLCD', attn, V_neighbors)
        
        return enhanced_token.reshape(B, num_h, num_w, C, token_dim) 