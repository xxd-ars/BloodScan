import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules import C3, Conv, SPPF
from ultralytics import YOLO
import math
import numpy as np
from typing import Optional, List, Tuple, Dict


class SpatialTransformer(nn.Module):
    """
    空间变换网络，用于特征对齐
    """
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, src, flow, mode='bilinear'):
        # 获取特征图的形状
        shape = flow.shape[2:]
        
        # 创建参考网格
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x
        grid = torch.unsqueeze(grid, 0)  # 添加batch维度
        grid = grid.type(torch.FloatTensor).to(flow.device)
        
        # 应用变形
        new_locs = grid + flow
        
        # 归一化到[-1, 1]范围
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        
        # 调整网格维度顺序以适配grid_sample
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]  # x, y顺序
        
        # 使用grid_sample进行采样
        return F.grid_sample(src, new_locs, mode=mode, align_corners=True)


class CrossAttention(nn.Module):
    """
    跨模态注意力模块，实现两个模态之间的特征交互
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 对于查询，键和值的线性映射
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, context):
        """
        前向传播函数
        Args:
            x: 主特征，shape (B, N, C)
            context: 上下文特征，shape (B, M, C)
        Returns:
            输出特征，shape (B, N, C)
        """
        B, N, C = x.shape
        _, M, _ = context.shape
        
        # 计算查询，键和值
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, num_heads, N, head_dim)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # (B, num_heads, M, head_dim)
        
        # 计算注意力得分
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, M)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 注意力加权和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class DeformableCrossTransformer(nn.Module):
    """
    变形跨模态Transformer模块，结合空间变换和注意力机制实现特征融合
    """
    def __init__(self, dim, num_heads=8, hidden_dim=64, kernel_size=3, mlp_ratio=4., 
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0., 
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(DeformableCrossTransformer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # 规范化层
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # 跨模态注意力
        self.cross_attn = CrossAttention(
            dim=dim, num_heads=num_heads, 
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        
        # 变形偏移预测网络
        self.offset_net = nn.Sequential(
            nn.Conv2d(dim * 2, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.GroupNorm(hidden_dim // 16, hidden_dim),
            act_layer(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1)
        )
        
        # 空间变换器
        self.transformer = SpatialTransformer()
        
        # 多层感知机
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
        # 残差连接的dropout
        self.drop_path = nn.Identity()  # 可替换为DropPath
        
    def _img2seq(self, x):
        """将特征图转换为序列形式"""
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        return x, (H, W)
    
    def _seq2img(self, x, size):
        """将序列转换回特征图形式"""
        H, W = size
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x
    
    def forward(self, x1, x2):
        """
        前向传播函数
        Args:
            x1: 第一个模态的特征，shape (B, C, H, W)
            x2: 第二个模态的特征，shape (B, C, H, W)
        Returns:
            融合后的特征，shape (B, C, H, W)
        """
        # 保存输入形状
        B, C, H, W = x1.shape
        
        # 1. 特征对齐(变形)
        # 预测偏移
        concat = torch.cat([x1, x2], dim=1)
        offsets = self.offset_net(concat)
        
        # 应用空间变换
        x2_aligned = self.transformer(x2, offsets)
        
        # 2. 跨模态注意力
        # 转换为序列形式
        x1_seq, hw = self._img2seq(x1)
        x2_seq, _ = self._img2seq(x2_aligned)
        
        # 应用注意力
        # x1通过注意力从x2获取信息
        x1_norm = self.norm1(x1_seq)
        x2_norm = self.norm1(x2_seq)
        attn_out = self.cross_attn(x1_norm, x2_norm)
        x1_seq = x1_seq + self.drop_path(attn_out)
        
        # MLP
        x1_seq = x1_seq + self.drop_path(self.mlp(self.norm2(x1_seq)))
        
        # 转换回特征图形式
        x_out = self._seq2img(x1_seq, hw)
        
        return x_out


class CrossModalFusion(nn.Module):
    """
    跨模态特征融合模块，用于融合两个backbone的特征
    """
    def __init__(self, dim, fusion_type='transformer', num_heads=8):
        super(CrossModalFusion, self).__init__()
        self.dim = dim
        self.fusion_type = fusion_type
        
        if fusion_type == 'add':
            # 简单加权求和
            self.weight1 = nn.Parameter(torch.ones(1))
            self.weight2 = nn.Parameter(torch.ones(1))
        
        elif fusion_type == 'concat':
            # 通道拼接加1x1卷积
            self.fusion_conv = nn.Conv2d(dim * 2, dim, kernel_size=1)
        
        elif fusion_type == 'transformer':
            # 变形跨模态Transformer
            self.transformer1 = DeformableCrossTransformer(dim, num_heads=num_heads)
            self.transformer2 = DeformableCrossTransformer(dim, num_heads=num_heads)
            
            # 组合权重
            self.combine_weights = nn.Parameter(torch.ones(2))
            self.softmax = nn.Softmax(dim=0)
    
    def forward(self, feat1, feat2):
        """
        前向传播函数
        Args:
            feat1: 第一个模态的特征，shape (B, C, H, W)
            feat2: 第二个模态的特征，shape (B, C, H, W)
        Returns:
            融合后的特征，shape (B, C, H, W)
        """
        if self.fusion_type == 'add':
            # 加权求和
            weights = torch.softmax(torch.stack([self.weight1, self.weight2]), dim=0)
            return weights[0] * feat1 + weights[1] * feat2
        
        elif self.fusion_type == 'concat':
            # 通道拼接加1x1卷积
            concat_feat = torch.cat([feat1, feat2], dim=1)
            return self.fusion_conv(concat_feat)
        
        elif self.fusion_type == 'transformer':
            # 变形跨模态Transformer (双向)
            feat1_enhanced = self.transformer1(feat1, feat2)  # feat1从feat2获取信息
            feat2_enhanced = self.transformer2(feat2, feat1)  # feat2从feat1获取信息
            
            # 组合融合特征
            weights = self.softmax(self.combine_weights)
            fused_feat = weights[0] * feat1_enhanced + weights[1] * feat2_enhanced
            
            return fused_feat
        
        else:
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")


class DualModalYOLO(nn.Module):
    """
    双模态YOLO模型，包含两个backbone和融合机制
    """
    def __init__(self, model_cfg="yolov8n.yaml", white_weights=None, blue_weights=None, 
                 fusion_type='transformer', num_heads=8):
        super(DualModalYOLO, self).__init__()
        
        # 初始化两个backbone (从预训练模型或配置加载)
        if isinstance(model_cfg, str) and model_cfg.endswith('.pt'):
            # 从预训练权重加载
            self.white_model = YOLO(model_cfg)
            self.blue_model = YOLO(model_cfg)
        else:
            # 从配置创建
            self.white_model = YOLO(model_cfg)
            self.blue_model = YOLO(model_cfg)
        
        # 加载预训练权重(如果提供)
        if white_weights:
            self.white_model.load(white_weights)
        if blue_weights:
            self.blue_model.load(blue_weights)
        
        # 获取backbone和head
        self.white_backbone = self.white_model.model.model[:10]  # 假设前10层是backbone
        self.blue_backbone = self.blue_model.model.model[:10]
        
        # 获取neck和head (假设共享)
        self.neck = self.white_model.model.model[10:17]
        self.head = self.white_model.model.model[17:]
        
        # 确定融合层的维度 (从backbone的输出特征获取)
        # P3/8 (第4层), P4/16 (第6层), P5/32 (第9层)
        self.backbone_outputs = [4, 6, 9]  # 特征输出层的索引
        
        # 获取各层特征维度
        with torch.no_grad():
            # 创建一个虚拟输入
            dummy_input = torch.zeros(1, 3, 640, 640)
            
            # 通过backbone获取特征
            feat_dims = []
            x = dummy_input
            for i, m in enumerate(self.white_backbone):
                x = m(x)
                if i in self.backbone_outputs:
                    feat_dims.append(x.shape[1])  # 添加通道数
        
        # 创建特征融合模块
        self.fusion_modules = nn.ModuleList([
            CrossModalFusion(dim=dim, fusion_type=fusion_type, num_heads=num_heads)
            for dim in feat_dims
        ])
        
        # 特征索引到融合模块的映射
        self.output_idx_to_fusion = {idx: i for i, idx in enumerate(self.backbone_outputs)}
    
    def forward(self, white_img, blue_img):
        """
        前向传播函数
        Args:
            white_img: 白光图像，shape (B, 3, H, W)
            blue_img: 蓝光图像，shape (B, 3, H, W)
        Returns:
            模型输出
        """
        # 特征储存
        white_features = []
        blue_features = []
        fused_features = []
        
        # 1. 提取白光特征
        x_white = white_img
        for i, m in enumerate(self.white_backbone):
            x_white = m(x_white)
            if i in self.backbone_outputs:
                white_features.append(x_white)
        
        # 2. 提取蓝光特征
        x_blue = blue_img
        for i, m in enumerate(self.blue_backbone):
            x_blue = m(x_blue)
            if i in self.backbone_outputs:
                blue_features.append(x_blue)
        
        # 3. 融合特征
        for i, (w_feat, b_feat) in enumerate(zip(white_features, blue_features)):
            fusion_module = self.fusion_modules[i]
            fused_feat = fusion_module(w_feat, b_feat)
            fused_features.append(fused_feat)
        
        # 4. 将融合的特征送入neck和head
        # 这里需要根据YOLO架构调整，将融合特征正确送入neck
        # 通常YOLO的neck需要P3, P4, P5三个尺度的特征
        
        # 假设neck的第一层接收P5特征
        x = fused_features[-1]  # 使用最后一个融合特征(P5)
        
        # 将特征通过neck
        for i, m in enumerate(self.neck):
            if isinstance(m, nn.Upsample):
                # 上采样层通常与之前的特征拼接
                x = m(x)
            elif isinstance(m, C3) and i in [3, 7]:
                # 这些C3层通常在concat之后
                x = m(x)
            elif isinstance(m, Conv) and i in [4, 8]:
                # 这些Conv层通常在C3之后
                x = m(x)
            else:
                if hasattr(m, 'f') and m.f != -1:
                    # 如果当前层需要之前层的特征
                    # 这里需要修改为使用正确的融合特征
                    layer_idx = m.f
                    if isinstance(layer_idx, int):
                        if layer_idx == -1:
                            feature = x
                        else:
                            # 找到这个索引对应的融合特征
                            fusion_idx = self.output_idx_to_fusion.get(layer_idx)
                            if fusion_idx is not None:
                                feature = fused_features[fusion_idx]
                            else:
                                feature = x  # 默认使用当前特征
                    else:
                        # 处理多个层的拼接
                        feature = [
                            fused_features[self.output_idx_to_fusion.get(idx, 0)] 
                            if idx in self.output_idx_to_fusion else x 
                            for idx in layer_idx
                        ]
                    
                    x = m(feature)
                else:
                    x = m(x)
        
        # 5. 将neck的输出送入head
        output = self.head(x)
        
        return output
    
    def load_weights(self, white_weights=None, blue_weights=None):
        """
        加载预训练权重
        """
        if white_weights:
            state_dict = torch.load(white_weights, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            # 仅加载白光backbone的权重
            white_backbone_dict = {k: v for k, v in state_dict.items() 
                                 if k.startswith('model.0') and int(k.split('.')[1]) < 10}
            missing, unexpected = self.white_backbone.load_state_dict(white_backbone_dict, strict=False)
            print(f"白光backbone加载: missing={len(missing)}, unexpected={len(unexpected)}")
        
        if blue_weights:
            state_dict = torch.load(blue_weights, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            # 仅加载蓝光backbone的权重
            blue_backbone_dict = {k: v for k, v in state_dict.items() 
                                if k.startswith('model.0') and int(k.split('.')[1]) < 10}
            missing, unexpected = self.blue_backbone.load_state_dict(blue_backbone_dict, strict=False)
            print(f"蓝光backbone加载: missing={len(missing)}, unexpected={len(unexpected)}")
    
    def freeze_backbones(self, freeze=True):
        """
        冻结或解冻backbone参数
        """
        for param in self.white_backbone.parameters():
            param.requires_grad = not freeze
        
        for param in self.blue_backbone.parameters():
            param.requires_grad = not freeze

# 示例用法
if __name__ == "__main__":
    # 创建双模态YOLO模型
    model = DualModalYOLO(model_cfg="yolov8n.yaml", fusion_type='transformer')
    
    # 输入示例
    white_img = torch.randn(1, 3, 640, 640)
    blue_img = torch.randn(1, 3, 640, 640)
    
    # 前向传播
    outputs = model(white_img, blue_img)
    
    print(f"Model outputs: {len(outputs)}") 