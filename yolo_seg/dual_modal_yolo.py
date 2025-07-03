from ultralytics import YOLO
import torch
import torch.nn as nn
import os
import yaml
from pathlib import Path
import copy

class DualModalYOLO(nn.Module):
    def __init__(self, white_model_path, blue_model_path, fusion_type='transformer'):
        """
        双模态YOLO11分割模型
        Args:
            white_model_path: 白光模型权重路径
            blue_model_path: 蓝光模型权重路径
            fusion_type: 特征融合类型 ['add', 'concat', 'transformer']
        """
        super(DualModalYOLO, self).__init__()
        
        # 加载两个单模态模型
        self.white_model = YOLO(white_model_path)
        self.blue_model = YOLO(blue_model_path)
        
        # 融合方式
        self.fusion_type = fusion_type
        
        # 提取两个模型的backbone部分
        self.white_backbone = self.white_model.model.model[0:10]  # 根据YOLO11模型结构调整索引
        self.blue_backbone = self.blue_model.model.model[0:10]
        
        # 提取neck部分
        self.neck = self.white_model.model.model[10:17]  # 根据YOLO11模型结构调整索引
        
        # 提取检测头
        self.head = self.white_model.model.model[17:]
        
        # 如果使用Transformer融合
        if fusion_type == 'transformer':
            # 跨模态特征交互Transformer层
            self.cross_attention = CrossModalTransformer(
                dim=512,  # 根据backbone输出特征的维度调整
                num_heads=8,
                dropout=0.1
            )
        
    def forward(self, white_img, blue_img):
        """
        前向传播
        Args:
            white_img: 白光图像，shape=(B, 3, H, W)
            blue_img: 蓝光图像，shape=(B, 3, H, W)
        Returns:
            预测结果
        """
        # 提取两个模态的特征
        white_features = self.white_backbone(white_img)
        blue_features = self.blue_backbone(blue_img)
        
        # 融合特征
        if self.fusion_type == 'add':
            # 简单加权叠加融合
            fused_features = white_features * 0.5 + blue_features * 0.5
        elif self.fusion_type == 'concat':
            # 通道拼接后通过1x1卷积融合
            fused_features = torch.cat([white_features, blue_features], dim=1)
            fused_features = nn.Conv2d(fused_features.size(1), white_features.size(1), kernel_size=1).to(white_features.device)(fused_features)
        elif self.fusion_type == 'transformer':
            # 使用Transformer进行特征融合
            fused_features = self.cross_attention(white_features, blue_features)
        else:
            raise ValueError(f"不支持的融合类型: {self.fusion_type}")
        
        # 通过neck和head得到最终结果
        x = self.neck(fused_features)
        output = self.head(x)
        
        return output
    
    @classmethod
    def create_from_yaml(cls, yaml_path, weights_path=None, fusion_type='transformer'):
        """
        从YAML配置文件创建双模态模型
        Args:
            yaml_path: YAML配置文件路径
            weights_path: 预训练权重路径
            fusion_type: 特征融合类型
        Returns:
            DualModalYOLO模型实例
        """
        # 加载基础模型
        base_model = YOLO('yolo11x-seg.pt')
        
        # 创建双模态模型
        model = cls(base_model, base_model, fusion_type)
        
        # 如果提供了权重，加载权重
        if weights_path and os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path))
        
        return model
    
    def train_model(self, data_yaml, white_dir, blue_dir, epochs=100, batch=8, imgsz=1024, device='cuda'):
        """
        训练双模态模型
        Args:
            data_yaml: 数据集配置文件路径
            white_dir: 白光图像文件夹
            blue_dir: 蓝光图像文件夹
            epochs: 训练轮次
            batch: 批量大小
            imgsz: 图像大小
            device: 训练设备
        """
        # 这里需要实现自定义训练逻辑...
        # 可能需要修改ultralytics的训练流程以支持双输入
        pass
        
    def predict(self, white_img_path, blue_img_path, conf=0.25, device='cuda'):
        """
        使用双模态模型进行预测
        Args:
            white_img_path: 白光图像路径
            blue_img_path: 蓝光图像路径
            conf: 置信度阈值
            device: 推理设备
        Returns:
            预测结果
        """
        # 加载图像
        white_img = self.white_model.predictor.setup_source(white_img_path)
        blue_img = self.blue_model.predictor.setup_source(blue_img_path)
        
        # 确保两个图像在同一设备上
        white_img = white_img.to(device)
        blue_img = blue_img.to(device)
        
        # 前向传播
        results = self.forward(white_img, blue_img)
        
        # 后处理结果
        # ...
        
        return results


class CrossModalTransformer(nn.Module):
    """
    跨模态特征交互Transformer模块
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super(CrossModalTransformer, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # 自注意力模块
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        
        # 跨模态注意力模块
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        
        # 前馈神经网络
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        
        # 规范化层
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # 激活函数
        self.activation = nn.GELU()
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
    def with_pos_embed(self, tensor, pos=None):
        """位置编码加特征"""
        return tensor if pos is None else tensor + pos
    
    def forward(self, white_feat, blue_feat):
        """
        跨模态特征交互
        Args:
            white_feat: 白光特征
            blue_feat: 蓝光特征
        Returns:
            融合后的特征
        """
        # 重塑特征以适应注意力机制
        b, c, h, w = white_feat.shape
        white_feat_flat = white_feat.flatten(2).permute(2, 0, 1)  # (h*w, b, c)
        blue_feat_flat = blue_feat.flatten(2).permute(2, 0, 1)  # (h*w, b, c)
        
        # 白光特征自注意力
        white_feat_norm = self.norm1(white_feat_flat)
        white_attn = self.self_attn(white_feat_norm, white_feat_norm, white_feat_norm)[0]
        white_feat_flat = white_feat_flat + self.dropout(white_attn)
        
        # 蓝光特征自注意力
        blue_feat_norm = self.norm1(blue_feat_flat)
        blue_attn = self.self_attn(blue_feat_norm, blue_feat_norm, blue_feat_norm)[0]
        blue_feat_flat = blue_feat_flat + self.dropout(blue_attn)
        
        # 白光对蓝光的跨模态注意力
        white_feat_norm = self.norm2(white_feat_flat)
        blue_feat_norm = self.norm2(blue_feat_flat)
        white2blue = self.cross_attn(white_feat_norm, blue_feat_norm, blue_feat_norm)[0]
        white_feat_flat = white_feat_flat + self.dropout(white2blue)
        
        # 蓝光对白光的跨模态注意力
        blue2white = self.cross_attn(blue_feat_norm, white_feat_norm, white_feat_norm)[0]
        blue_feat_flat = blue_feat_flat + self.dropout(blue2white)
        
        # 融合两个模态的特征
        fused_feat = white_feat_flat + blue_feat_flat
        
        # 前馈神经网络
        fused_feat_norm = self.norm3(fused_feat)
        fused_feat_ff = self.linear2(self.dropout(self.activation(self.linear1(fused_feat_norm))))
        fused_feat = fused_feat + self.dropout(fused_feat_ff)
        
        # 重塑回原始形状
        fused_feat = fused_feat.permute(1, 2, 0).reshape(b, c, h, w)
        
        return fused_feat


def create_dual_modal_yaml(output_path):
    """
    创建双模态YOLO11模型的YAML配置文件
    Args:
        output_path: 输出文件路径
    """
    config = {
        'nc': 3,  # 根据实际类别数调整
        'depth_multiple': 0.33,
        'width_multiple': 0.25,
        'anchors': [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]
        ],
        
        # 白光backbone
        'white_backbone': [
            # [from, number, module, args]
            [-1, 1, 'Conv', [64, 3, 2, 1, 1]],  # 0-P1/2
            [-1, 1, 'Conv', [128, 3, 2, 1, 1]],  # 1-P2/4
            [-1, 3, 'C3', [128, True, 1, 0.5]],
            [-1, 1, 'Conv', [256, 3, 2, 1, 1]],  # 3-P3/8
            [-1, 6, 'C3', [256, True, 1, 0.5]],
            [-1, 1, 'Conv', [512, 3, 2, 1, 1]],  # 5-P4/16
            [-1, 9, 'C3', [512, True, 1, 0.5]],
            [-1, 1, 'Conv', [1024, 3, 2, 1, 1]],  # 7-P5/32
            [-1, 3, 'C3', [1024, True, 1, 0.5]],
            [-1, 1, 'SPPF', [1024, 5]],  # 9
        ],
        
        # 蓝光backbone
        'blue_backbone': [
            # [from, number, module, args]
            [-1, 1, 'Conv', [64, 3, 2, 1, 1]],  # 0-P1/2
            [-1, 1, 'Conv', [128, 3, 2, 1, 1]],  # 1-P2/4
            [-1, 3, 'C3', [128, True, 1, 0.5]],
            [-1, 1, 'Conv', [256, 3, 2, 1, 1]],  # 3-P3/8
            [-1, 6, 'C3', [256, True, 1, 0.5]],
            [-1, 1, 'Conv', [512, 3, 2, 1, 1]],  # 5-P4/16
            [-1, 9, 'C3', [512, True, 1, 0.5]],
            [-1, 1, 'Conv', [1024, 3, 2, 1, 1]],  # 7-P5/32
            [-1, 3, 'C3', [1024, True, 1, 0.5]],
            [-1, 1, 'SPPF', [1024, 5]],  # 9
        ],
        
        # 特征融合层
        'fusion': [
            # [from, number, module, args]
            ['white_backbone.9', 'blue_backbone.9', 1, 'CrossModalFusion', [1024]],  # 融合P5特征
        ],
        
        # Neck
        'neck': [
            [-1, 1, 'Conv', [512, 1, 1, 1, 1]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 'white_backbone.6', 'blue_backbone.6'], 1, 'Concat', [1]],  # 使用Concat替代cat操作
            [-1, 3, 'C3', [512, False, 1, 0.5]],
            [-1, 1, 'Conv', [256, 1, 1, 1, 1]],
            [-1, 1, 'nn.Upsample', [None, 2, 'nearest']],
            [[-1, 'white_backbone.4', 'blue_backbone.4'], 1, 'Concat', [1]],
            [-1, 3, 'C3', [256, False, 1, 0.5]],  # 14 (P3/8)
            [-1, 1, 'Conv', [256, 3, 2, 1, 1]],
            [[-1, 14], 1, 'Concat', [1]],
            [-1, 3, 'C3', [512, False, 1, 0.5]],  # 17 (P4/16)
            [-1, 1, 'Conv', [512, 3, 2, 1, 1]],
            [[-1, 11], 1, 'Concat', [1]],
            [-1, 3, 'C3', [1024, False, 1, 0.5]],  # 20 (P5/32)
        ],
        
        # 分割头
        'head': [
            [[14, 17, 20], 1, 'Segment', [3, 32, 1]],  # Segment(P3, P4, P5)
        ]
    }
    
    # 保存YAML配置
    with open(output_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    print(f"双模态YOLO11配置已保存到: {output_path}")


if __name__ == "__main__":
    # 创建双模态配置文件
    create_dual_modal_yaml('yolo_seg/models/dual_modal_yolo11.yaml')
    
    # 创建双模态模型示例
    # model = DualModalYOLO('yolo11x-seg.pt', 'yolo11x-seg.pt', fusion_type='transformer')
    # print(model) 