import os
from pathlib import Path
import yaml


class Config:
    """配置类，包含模型和训练的各项参数"""
    
    def __init__(self):
        # 基本路径
        self.project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.data_dir = self.project_dir / 'data'
        self.weights_dir = self.project_dir / 'dual_modal_yolo' / 'weights'
        self.results_dir = self.project_dir / 'dual_modal_yolo' / 'results'
        
        # 确保目录存在
        self.weights_dir.mkdir(exist_ok=True, parents=True)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # 数据集配置
        self.train_white_dir = ''  # 训练集白光图像目录
        self.train_blue_dir = ''   # 训练集蓝光图像目录
        self.train_annotation_dir = ''  # 训练集标注目录
        
        self.val_white_dir = ''    # 验证集白光图像目录
        self.val_blue_dir = ''     # 验证集蓝光图像目录
        self.val_annotation_dir = ''  # 验证集标注目录
        
        self.test_white_dir = ''   # 测试集白光图像目录
        self.test_blue_dir = ''    # 测试集蓝光图像目录
        
        # 数据处理配置
        self.img_size = 640        # 输入图像大小
        self.batch_size = 16       # 批量大小
        self.num_workers = 4       # 数据加载线程数
        self.augment = True        # 是否使用数据增强
        self.prefix_white = ''     # 白光图像前缀
        self.prefix_blue = ''      # 蓝光图像前缀
        self.suffix_white = ''     # 白光图像后缀
        self.suffix_blue = ''      # 蓝光图像后缀
        self.pair_mode = 'filename'  # 图像配对模式
        
        # 模型配置
        self.model_cfg = 'yolov8s.yaml'  # 模型配置文件
        self.white_weights = ''    # 白光模型预训练权重
        self.blue_weights = ''     # 蓝光模型预训练权重
        self.fusion_type = 'transformer'  # 特征融合方式
        self.num_heads = 8         # 注意力头数量
        self.num_classes = 3       # 类别数量
        
        # 训练配置
        self.epochs = 100          # 训练轮次
        self.lr = 0.01             # 初始学习率
        self.weight_decay = 0.0005 # 权重衰减
        self.momentum = 0.937      # SGD动量
        self.freeze_backbone = True  # 是否冻结backbone
        self.amp = True            # 是否使用混合精度训练
        self.device = 'cuda'       # 训练设备
        
        # 优化器配置
        self.optimizer = 'SGD'     # 优化器类型
        self.lr_scheduler = 'cosine'  # 学习率调度器类型
        self.warmup_epochs = 3     # 预热轮次
        
        # 损失函数权重
        self.box_loss_weight = 0.05    # 边界框损失权重
        self.cls_loss_weight = 0.5     # 分类损失权重
        self.obj_loss_weight = 1.0     # 目标性损失权重
        self.seg_loss_weight = 0.2     # 分割损失权重
        
        # 推理配置
        self.conf_thres = 0.25     # 置信度阈值
        self.iou_thres = 0.45      # IoU阈值
        self.max_det = 300         # 最大检测框数量
        
    def update(self, **kwargs):
        """更新配置参数"""
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                raise ValueError(f"配置中不存在参数: {k}")
                
    def save(self, config_path):
        """保存配置到YAML文件"""
        config_dict = {k: str(v) if isinstance(v, Path) else v 
                      for k, v in self.__dict__.items()}
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, sort_keys=False)
            
    @classmethod
    def load(cls, config_path):
        """从YAML文件加载配置"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            
        config = cls()
        for k, v in config_dict.items():
            if hasattr(config, k):
                if k.endswith('_dir') and not k.startswith('pair_mode'):
                    setattr(config, k, Path(v))
                else:
                    setattr(config, k, v)
                    
        return config
    
    def __str__(self):
        """打印配置信息"""
        config_str = "双模态YOLO配置:\n"
        for k, v in self.__dict__.items():
            config_str += f"  {k}: {v}\n"
        return config_str


# 默认配置实例
default_config = Config()


def get_blood_sample_config():
    """获取血液样本数据集的配置"""
    config = Config()
    
    # 数据集路径配置
    config.train_white_dir = config.data_dir / 'blood_samples/train/white'
    config.train_blue_dir = config.data_dir / 'blood_samples/train/blue'
    config.train_annotation_dir = config.data_dir / 'blood_samples/train/labels'
    
    config.val_white_dir = config.data_dir / 'blood_samples/val/white'
    config.val_blue_dir = config.data_dir / 'blood_samples/val/blue'
    config.val_annotation_dir = config.data_dir / 'blood_samples/val/labels'
    
    config.test_white_dir = config.data_dir / 'blood_samples/test/white'
    config.test_blue_dir = config.data_dir / 'blood_samples/test/blue'
    
    # 模型配置
    config.model_cfg = 'yolov8s-seg.pt'  # 使用分割模型
    config.num_classes = 3  # 血液分层类别数
    
    # 训练配置
    config.batch_size = 8
    config.epochs = 50
    config.lr = 0.001
    
    return config


if __name__ == "__main__":
    # 测试配置类
    
    # 创建默认配置
    config = Config()
    print(config)
    
    # 更新配置
    config.update(
        batch_size=8,
        epochs=50,
        lr=0.001
    )
    print(f"更新后的批量大小: {config.batch_size}")
    
    # 保存配置
    config.save('dual_modal_yolo/config.yaml')
    print("配置已保存")
    
    # 加载配置
    loaded_config = Config.load('dual_modal_yolo/config.yaml')
    print(f"加载的配置 - 批量大小: {loaded_config.batch_size}")
    
    # 获取血液样本配置
    blood_config = get_blood_sample_config()
    print(f"血液样本配置 - 类别数: {blood_config.num_classes}") 