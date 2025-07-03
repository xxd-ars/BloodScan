# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Dual-modal fusion modules for YOLO."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("ConcatCompress", "WeightedFusion", "CrossModalAttention")


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
    Cross-modal attention module for blue light querying white light features.
    
    Blue light features act as queries to enhance themselves using white light features.
    Uses token-based local attention for efficiency.
    """

    def __init__(self, c1, c2=None, token_size=4, neighbor_size=3):
        """
        Initialize CrossModalAttention module.
        
        Args:
            c1 (int): Number of input channels.
            c2 (int, optional): Number of output channels. If None, defaults to c1.
            token_size (int): Token size for spatial partitioning.
            neighbor_size (int): Neighbor search range for local attention.
        """
        super().__init__()
        if c2 is None:
            c2 = c1
        self.c1 = c1
        self.c2 = c2
        self.token_size = token_size
        self.neighbor_size = neighbor_size
        
        # Projection layers
        self.q_proj = nn.Conv2d(c1, c1, 1)
        self.k_proj = nn.Conv2d(c1, c1, 1)
        self.v_proj = nn.Conv2d(c1, c1, 1)
        
        # Scale factor
        self.scale = (c1 // (token_size * token_size)) ** -0.5
        
        # Output projection
        self.out_proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()

    def forward(self, x):
        """
        Forward pass of CrossModalAttention.
        
        Args:
            x (list): List containing [blue_feat, white_feat] tensors.
            
        Returns:
            torch.Tensor: Enhanced blue features.
        """
        blue_feat, white_feat = x
        B, C, H, W = blue_feat.shape
        
        # Project features
        Q = self.q_proj(blue_feat)
        K = self.k_proj(white_feat) 
        V = self.v_proj(white_feat)
        
        # Ensure dimensions are divisible by token_size
        pad_h = (self.token_size - H % self.token_size) % self.token_size
        pad_w = (self.token_size - W % self.token_size) % self.token_size
        if pad_h > 0 or pad_w > 0:
            Q = F.pad(Q, (0, pad_w, 0, pad_h))
            K = F.pad(K, (0, pad_w, 0, pad_h))
            V = F.pad(V, (0, pad_w, 0, pad_h))
            H_pad, W_pad = Q.shape[2], Q.shape[3]
        else:
            H_pad, W_pad = H, W
        
        # Tokenize: reshape to [B, C, num_h, num_w, token_size^2]
        num_h, num_w = H_pad // self.token_size, W_pad // self.token_size
        
        Q_tokens = Q.reshape(B, C, num_h, self.token_size, num_w, self.token_size)
        Q_tokens = Q_tokens.permute(0, 2, 4, 1, 3, 5).contiguous()
        Q_tokens = Q_tokens.reshape(B, num_h, num_w, C, -1)
        
        K_tokens = K.reshape(B, C, num_h, self.token_size, num_w, self.token_size)
        K_tokens = K_tokens.permute(0, 2, 4, 1, 3, 5).contiguous()
        K_tokens = K_tokens.reshape(B, num_h, num_w, C, -1)
        
        V_tokens = V.reshape(B, C, num_h, self.token_size, num_w, self.token_size)
        V_tokens = V_tokens.permute(0, 2, 4, 1, 3, 5).contiguous()
        V_tokens = V_tokens.reshape(B, num_h, num_w, C, -1)
        
        # Apply local cross-modal attention
        enhanced_tokens = self._local_attention(Q_tokens, K_tokens, V_tokens)
        
        # Reshape back to feature map
        enhanced_tokens = enhanced_tokens.reshape(B, num_h, num_w, C, self.token_size, self.token_size)
        enhanced_tokens = enhanced_tokens.permute(0, 3, 1, 4, 2, 5).contiguous()
        enhanced_feat = enhanced_tokens.reshape(B, C, H_pad, W_pad)
        
        # Remove padding if added
        if pad_h > 0 or pad_w > 0:
            enhanced_feat = enhanced_feat[:, :, :H, :W]
        
        return self.out_proj(enhanced_feat)
    
    def _local_attention(self, Q, K, V):
        """Apply local attention with neighbor constraints."""
        B, num_h, num_w, C, token_dim = Q.shape
        enhanced = torch.zeros_like(Q)
        
        for i in range(num_h):
            for j in range(num_w):
                # Define neighbor range
                i_start = max(0, i - self.neighbor_size // 2)
                i_end = min(num_h, i + self.neighbor_size // 2 + 1)
                j_start = max(0, j - self.neighbor_size // 2)
                j_end = min(num_w, j + self.neighbor_size // 2 + 1)
                
                # Get query token
                q = Q[:, i, j]  # [B, C, token_dim]
                
                # Get neighbor keys and values
                k_neighbors = K[:, i_start:i_end, j_start:j_end]  # [B, nh, nw, C, token_dim]
                v_neighbors = V[:, i_start:i_end, j_start:j_end]
                
                nh, nw = k_neighbors.shape[1], k_neighbors.shape[2]
                k_neighbors = k_neighbors.reshape(B, nh * nw, C, token_dim)
                v_neighbors = v_neighbors.reshape(B, nh * nw, C, token_dim)
                
                # Compute attention scores
                attn = torch.einsum('bct,bnct->bcn', q, k_neighbors) * self.scale
                attn = F.softmax(attn, dim=-1)
                
                # Apply attention to values
                enhanced_token = torch.einsum('bcn,bnct->bct', attn, v_neighbors)
                enhanced[:, i, j] = enhanced_token
        
        return enhanced 