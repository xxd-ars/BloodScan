import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from typing import Dict, List, Tuple, Optional

class AddFusion(nn.Module):
    """空间+通道双感知权重融合模块"""
    def __init__(self, channels: int):
        super().__init__()
        self.weight_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, f_b: torch.Tensor, f_w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_b: 蓝光特征, shape (B, C, H, W)
            f_w: 白光特征, shape (B, C, H, W)
        
        Returns:
            融合后的特征, shape (B, C, H, W)
        """
        # 连接特征用于生成空间权重
        concat_feat = torch.cat([f_b, f_w], dim=1)
        alpha = self.weight_net(concat_feat)
        
        # 加权融合
        out = alpha * f_b + (1 - alpha) * f_w
        return out


class CatFusion(nn.Module):
    """通道拼接再压缩融合模块"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=1)
    
    def forward(self, f_b: torch.Tensor, f_w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_b: 蓝光特征, shape (B, C, H, W)
            f_w: 白光特征, shape (B, C, H, W)
        
        Returns:
            融合后的特征, shape (B, C, H, W)
        """
        concat_feat = torch.cat([f_b, f_w], dim=1)
        out = self.conv(concat_feat)
        return out


class XFormerFusion(nn.Module):
    """基于交叉注意力的特征融合模块"""
    def __init__(self, channels: int, dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.dim = dim
        self.num_heads = num_heads
        
        # 投影层: 将通道维度投影到dim*3用于Q,K,V
        self.proj_b = nn.Conv2d(channels, dim * 3, kernel_size=1)
        self.proj_w = nn.Conv2d(channels, dim * 3, kernel_size=1)
        
        # 输出投影层
        self.proj_out_b = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.LayerNorm([channels, 1, 1])
        )
        self.proj_out_w = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.LayerNorm([channels, 1, 1])
        )
        
        self.scale = dim ** -0.5
    
    def forward(self, f_b: torch.Tensor, f_w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_b: 蓝光特征, shape (B, C, H, W)
            f_w: 白光特征, shape (B, C, H, W)
        
        Returns:
            融合后的特征, shape (B, C, H, W)
        """
        B, C, H, W = f_b.shape
        
        # 生成Q,K,V
        q_b, k_b, v_b = self.proj_b(f_b).chunk(3, dim=1)
        q_w, k_w, v_w = self.proj_w(f_w).chunk(3, dim=1)
        
        # 重塑张量适应多头注意力
        q_b = q_b.view(B, self.num_heads, self.dim // self.num_heads, H * W).permute(0, 1, 3, 2)  # B, heads, HW, dim/heads
        k_b = k_b.view(B, self.num_heads, self.dim // self.num_heads, H * W).permute(0, 1, 2, 3)  # B, heads, dim/heads, HW
        v_b = v_b.view(B, self.num_heads, self.dim // self.num_heads, H * W).permute(0, 1, 3, 2)  # B, heads, HW, dim/heads
        
        q_w = q_w.view(B, self.num_heads, self.dim // self.num_heads, H * W).permute(0, 1, 3, 2)  # B, heads, HW, dim/heads
        k_w = k_w.view(B, self.num_heads, self.dim // self.num_heads, H * W).permute(0, 1, 2, 3)  # B, heads, dim/heads, HW
        v_w = v_w.view(B, self.num_heads, self.dim // self.num_heads, H * W).permute(0, 1, 3, 2)  # B, heads, HW, dim/heads
        
        # 交叉注意力计算
        attn_bw = torch.matmul(q_b, k_w) * self.scale
        attn_bw = F.softmax(attn_bw, dim=-1)
        f_b_enhanced = torch.matmul(attn_bw, v_w)  # B, heads, HW, dim/heads
        
        attn_wb = torch.matmul(q_w, k_b) * self.scale
        attn_wb = F.softmax(attn_wb, dim=-1)
        f_w_enhanced = torch.matmul(attn_wb, v_b)  # B, heads, HW, dim/heads
        
        # 重塑回原始维度
        f_b_enhanced = f_b_enhanced.permute(0, 1, 3, 2).reshape(B, self.dim, H, W)
        f_w_enhanced = f_w_enhanced.permute(0, 1, 3, 2).reshape(B, self.dim, H, W)
        
        # 恢复通道维度并确保与输入特征通道一致
        if self.dim != C:
            f_b_enhanced = f_b_enhanced[:, :C, :, :]
            f_w_enhanced = f_w_enhanced[:, :C, :, :]
        
        # 创建适配2D输入的LayerNorm
        class LayerNorm2d(nn.Module):
            def __init__(self, num_channels):
                super().__init__()
                self.norm = nn.LayerNorm(num_channels)
                
            def forward(self, x):
                # (B, C, H, W) -> (B, H, W, C)
                x = x.permute(0, 2, 3, 1)
                x = self.norm(x)
                # (B, H, W, C) -> (B, C, H, W)
                return x.permute(0, 3, 1, 2)
        
        # 融合增强后的特征
        layer_norm = LayerNorm2d(C)
        enhanced_b = f_b + f_b_enhanced
        enhanced_b = layer_norm(enhanced_b)
        
        enhanced_w = f_w + f_w_enhanced
        enhanced_w = layer_norm(enhanced_w)
        
        return enhanced_b + enhanced_w


class DualYOLO(nn.Module):
    """双模态YOLO模型，包含蓝光和白光两个骨干网络"""
    def __init__(self, fusion_type: str = 'ctr', channels: int = 256, dim: int = 128, num_heads: int = 4):
        """
        Args:
            fusion_type: 融合类型，'add', 'cat', 或 'ctr'
            channels: 特征通道数
            dim: XFormerFusion模块内部维度
            num_heads: XFormerFusion模块头数
        """
        super().__init__()
        self.fusion_type = fusion_type
        self.channels = channels
        
        # 加载预训练的YOLO骨干网络
        model_b = YOLO('./weights/yolo11x-seg.pt')
        model_w = YOLO('./weights/yolo11x-seg.pt')
        
        # 提取骨干网络
        self.backbone_b = model_b.model.model[0:10]  # 骨干网络部分
        self.backbone_w = model_w.model.model[0:10]  # 骨干网络部分
        
        # 提取颈部和头部
        self.neck = model_b.model.model[10:16]  # 颈部网络
        self.head = model_b.model.model[16:]    # 头部网络
        
        # 根据融合类型创建融合模块
        if fusion_type == 'add':
            self.fusion_p3 = AddFusion(channels)
            self.fusion_p4 = AddFusion(channels)
            self.fusion_p5 = AddFusion(channels)
        elif fusion_type == 'cat':
            self.fusion_p3 = CatFusion(channels)
            self.fusion_p4 = CatFusion(channels)
            self.fusion_p5 = CatFusion(channels)
        elif fusion_type == 'ctr':
            self.fusion_p3 = XFormerFusion(channels, dim, num_heads)
            self.fusion_p4 = XFormerFusion(channels, dim, num_heads)
            self.fusion_p5 = XFormerFusion(channels, dim, num_heads)
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}，请选择 'add', 'cat', 或 'ctr'")
    
    def forward(self, x_b: torch.Tensor, x_w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_b: 蓝光图像，shape (B, 3, H, W)
            x_w: 白光图像，shape (B, 3, H, W)
        
        Returns:
            model output
        """
        # 通过骨干网络提取特征
        features_b = self.backbone_b(x_b)
        features_w = self.backbone_w(x_w)
        
        # 提取P3、P4和P5特征
        f_b_p3, f_b_p4, f_b_p5 = features_b[0], features_b[1], features_b[2]
        f_w_p3, f_w_p4, f_w_p5 = features_w[0], features_w[1], features_w[2]
        
        # 特征融合
        fused_p3 = self.fusion_p3(f_b_p3, f_w_p3)
        fused_p4 = self.fusion_p4(f_b_p4, f_w_p4)
        fused_p5 = self.fusion_p5(f_b_p5, f_w_p5)
        
        # 通过颈部和头部网络
        neck_out = self.neck([fused_p3, fused_p4, fused_p5])
        output = self.head(neck_out)
        
        return output


# 创建模型辅助函数
def create_dual_yolo(fusion_type: str = 'ctr', channels: int = 256, dim: int = 128, num_heads: int = 4) -> DualYOLO:
    """创建DualYOLO模型实例
    
    Args:
        fusion_type: 融合类型，'add', 'cat', 或 'ctr'
        channels: 特征通道数
        dim: XFormerFusion模块内部维度
        num_heads: XFormerFusion模块头数
    
    Returns:
        DualYOLO模型实例
    """
    model = DualYOLO(fusion_type=fusion_type, channels=channels, dim=dim, num_heads=num_heads)
    return model

