#!/usr/bin/env python3
"""
双模态YOLO模型注意力可视化脚本
实现跨模态注意力热力图可视化

用法:
python d_model_attention_vis.py --model_path best.pt --blue_img test_b.jpg --white_img test_w.jpg --output result.png
"""

import torch
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO


class AttentionVisualizer:
    """跨模态注意力可视化器"""
    
    def __init__(self, model_path, model_yaml=None):
        """
        初始化可视化器
        
        Args:
            model_path (str): 模型权重文件路径
            model_yaml (str, optional): 模型配置文件路径
        """
        self.project_root = Path(__file__).parent.parent
        
        # 设置默认模型配置
        if model_yaml is None:
            model_yaml = self.project_root / 'dual_yolo' / 'models' / 'yolo11x-dseg-crossattn.yaml'
        
        # 加载模型
        print(f"加载模型: {model_yaml}")
        print(f"加载权重: {model_path}")
        self.model = YOLO(model_yaml).load(model_path)
        
        # 启用所有CrossModalAttention模块的可视化
        self._enable_attention_visualization()
        
    def _enable_attention_visualization(self):
        """启用模型中所有CrossModalAttention模块的可视化"""
        attention_modules = []
        
        def find_attention_modules(module, name=""):
            """递归查找CrossModalAttention模块"""
            if hasattr(module, '__class__') and 'CrossModalAttention' in module.__class__.__name__:
                attention_modules.append((name, module))
                module.enable_attention_visualization(True)
            
            for child_name, child in module.named_children():
                find_attention_modules(child, f"{name}.{child_name}" if name else child_name)
        
        find_attention_modules(self.model.model)
        
        self.attention_modules = attention_modules
        print(f"找到 {len(attention_modules)} 个注意力模块:")
        for name, _ in attention_modules:
            print(f"  - {name}")
    
    def load_dual_modal_image(self, npy_path):
        """
        加载双模态图像（支持.npy和分离的.jpg格式）
        
        Args:
            npy_path (str): .npy文件路径，或蓝光图像路径（用于向后兼容）
            
        Returns:
            tuple: (dual_tensor, blue_image, white_image)
        """
        npy_path = Path(npy_path)
        
        if npy_path.suffix == '.npy':
            # 加载.npy格式的双模态数据
            print(f"加载双模态.npy文件: {npy_path}")
            dual_data = np.load(npy_path)
            
            # 处理数据维度
            if dual_data.shape[-1] == 6:  # [H, W, 6]
                dual_data = dual_data.transpose(2, 0, 1)  # [6, H, W]
            
            print(f"双模态数据形状: {dual_data.shape}")
            
            # 分离蓝光和白光通道
            blue_channels = dual_data[:3]  # [3, H, W]
            white_channels = dual_data[3:]  # [3, H, W]
            
            # 转换为显示格式 [H, W, 3]
            blue_image = blue_channels.transpose(1, 2, 0)
            white_image = white_channels.transpose(1, 2, 0)
            
            # 确保数值范围在[0, 255]
            if blue_image.max() <= 1.0:
                blue_image = (blue_image * 255).astype(np.uint8)
                white_image = (white_image * 255).astype(np.uint8)
            else:
                blue_image = blue_image.astype(np.uint8)
                white_image = white_image.astype(np.uint8)
            
            # 准备模型输入张量
            if dual_data.max() > 1.0:
                dual_tensor = torch.from_numpy(dual_data / 255.0).unsqueeze(0).float()
            else:
                dual_tensor = torch.from_numpy(dual_data).unsqueeze(0).float()
            
        else:
            # 向后兼容：处理分离的蓝光/白光图像
            print("使用分离图像模式（向后兼容）")
            # 这里需要传入white_path作为第二个参数，暂时不支持
            raise ValueError("对于分离图像，请使用load_separated_images方法")
        
        return dual_tensor, blue_image, white_image
    
    def load_separated_images(self, blue_path, white_path):
        """
        加载分离的蓝光和白光图像（向后兼容）
        
        Args:
            blue_path (str): 蓝光图像路径
            white_path (str): 白光图像路径
            
        Returns:
            tuple: (dual_tensor, blue_image, white_image)
        """
        # 加载图像
        blue_img = Image.open(blue_path).convert('RGB')
        white_img = Image.open(white_path).convert('RGB')
        
        print(f"蓝光图像尺寸: {blue_img.size}")
        print(f"白光图像尺寸: {white_img.size}")
        
        # 转换为张量
        blue_tensor = torch.from_numpy(np.array(blue_img)).permute(2, 0, 1).float() / 255.0
        white_tensor = torch.from_numpy(np.array(white_img)).permute(2, 0, 1).float() / 255.0
        
        # 拼接双模态输入
        dual_tensor = torch.cat([blue_tensor, white_tensor], dim=0).unsqueeze(0)  # [1, 6, H, W]
        
        return dual_tensor, np.array(blue_img), np.array(white_img)
    
    def run_inference_with_attention(self, dual_tensor):
        """
        执行推理并捕获注意力权重
        
        Args:
            dual_tensor (torch.Tensor): 双模态输入张量 [1, 6, H, W]
            
        Returns:
            tuple: (prediction_results, attention_maps)
        """
        print("执行模型推理...")
        
        # 执行推理
        with torch.no_grad():
            results = self.model(dual_tensor, verbose=False)
        
        # 收集注意力权重
        attention_maps = {}
        for name, module in self.attention_modules:
            spatial_map = module.get_attention_spatial_map()
            if spatial_map is not None:
                attention_maps[name] = spatial_map[0].numpy()  # 取第一个batch的结果
        
        print(f"捕获到 {len(attention_maps)} 个注意力热力图")
        
        return results, attention_maps
    
    def create_attention_heatmap(self, attention_map, base_image, alpha=0.5):
        """
        创建注意力热力图叠加
        
        Args:
            attention_map (np.ndarray): 注意力权重图 [H, W]
            base_image (np.ndarray): 基础图像 [H, W, 3]
            alpha (float): 热力图透明度
            
        Returns:
            np.ndarray: 叠加后的图像
        """
        # 归一化注意力图到0-1
        attn_norm = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
        # 应用颜色映射（红色表示高注意力，蓝色表示低注意力）
        heatmap = plt.cm.jet(attn_norm)[:, :, :3]  # 去除alpha通道
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # 确保base_image是uint8格式
        if base_image.dtype != np.uint8:
            base_image = (base_image * 255).astype(np.uint8)
        
        # 调整尺寸匹配
        if heatmap.shape[:2] != base_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (base_image.shape[1], base_image.shape[0]))
        
        # 叠加热力图
        overlay = cv2.addWeighted(base_image, 1-alpha, heatmap, alpha, 0)
        
        return overlay
    
    def visualize_prediction_results(self, results, base_image):
        """
        可视化预测结果（检测框和分割掩码）
        
        Args:
            results: YOLO预测结果
            base_image (np.ndarray): 基础图像
            
        Returns:
            np.ndarray: 带预测结果的图像
        """
        result_image = base_image.copy()
        
        if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                # 绘制检测框
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # 绘制边界框
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 添加标签
                    label = f"Class {cls}: {conf:.2f}"
                    cv2.putText(result_image, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 绘制分割掩码
        if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            
            for i, mask in enumerate(masks):
                # 调整掩码尺寸
                mask_resized = cv2.resize(mask, (result_image.shape[1], result_image.shape[0]))
                
                # 创建彩色掩码
                color = plt.cm.Set1(i % 10)[:3]  # 使用不同颜色
                colored_mask = np.zeros_like(result_image)
                colored_mask[mask_resized > 0.5] = [int(c * 255) for c in color]
                
                # 叠加掩码
                result_image = cv2.addWeighted(result_image, 0.8, colored_mask, 0.2, 0)
        
        return result_image
    
    def create_visualization_grid(self, blue_image, white_image, prediction_image, attention_maps):
        """
        创建可视化网格显示
        
        Args:
            blue_image (np.ndarray): 蓝光图像
            white_image (np.ndarray): 白光图像  
            prediction_image (np.ndarray): 带预测结果的图像
            attention_maps (dict): 注意力热力图字典
            
        Returns:
            np.ndarray: 网格化的可视化结果
        """
        H, W = blue_image.shape[:2]
        
        # 创建注意力热力图叠加
        attention_overlays = []
        for name, attn_map in attention_maps.items():
            overlay = self.create_attention_heatmap(attn_map, blue_image, alpha=0.6)
            attention_overlays.append((name, overlay))
        
        # 计算网格布局
        num_images = 3 + len(attention_overlays)  # 原图+预测+注意力图们
        cols = min(3, num_images)
        rows = (num_images + cols - 1) // cols
        
        # 创建网格画布
        grid_h, grid_w = rows * H, cols * W
        grid_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        # 放置图像
        images_to_place = [
            ("Blue Light", blue_image),
            ("White Light", white_image), 
            ("Predictions", prediction_image)
        ] + [(f"Attention-{name.split('.')[-1]}", overlay) for name, overlay in attention_overlays]
        
        for i, (title, img) in enumerate(images_to_place):
            row, col = i // cols, i % cols
            y_start, y_end = row * H, (row + 1) * H
            x_start, x_end = col * W, (col + 1) * W
            
            # 确保图像尺寸匹配
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H))
            
            grid_image[y_start:y_end, x_start:x_end] = img
            
            # 添加标题
            cv2.putText(grid_image, title, (x_start + 10, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return grid_image
    
    def visualize(self, npy_path=None, blue_path=None, white_path=None, output_path="attention_visualization.png"):
        """
        执行完整的注意力可视化流程
        
        Args:
            npy_path (str, optional): .npy文件路径（优先）
            blue_path (str, optional): 蓝光图像路径（向后兼容）
            white_path (str, optional): 白光图像路径（向后兼容）
            output_path (str): 输出图像路径
        """
        print("=== 开始注意力可视化 ===")
        
        # 1. 加载双模态图像
        if npy_path:
            dual_tensor, blue_image, white_image = self.load_dual_modal_image(npy_path)
        elif blue_path and white_path:
            dual_tensor, blue_image, white_image = self.load_separated_images(blue_path, white_path)
        else:
            raise ValueError("请提供npy_path或者(blue_path, white_path)参数")
        
        # 2. 执行推理并捕获注意力
        results, attention_maps = self.run_inference_with_attention(dual_tensor)
        
        # 3. 生成预测结果可视化
        prediction_image = self.visualize_prediction_results(results, blue_image)
        
        # 4. 创建综合可视化网格
        grid_image = self.create_visualization_grid(
            blue_image, white_image, prediction_image, attention_maps
        )
        
        # 5. 保存结果
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_path), cv2.cvtColor(grid_image, cv2.COLOR_RGB2BGR))
        print(f"可视化结果已保存至: {output_path}")
        
        # 6. 打印统计信息
        print("\n=== 可视化统计 ===")
        if results and len(results) > 0:
            result = results[0]
            boxes_count = len(result.boxes) if hasattr(result, 'boxes') and result.boxes is not None else 0
            masks_count = len(result.masks.data) if hasattr(result, 'masks') and result.masks is not None else 0
            
            print(f"检测结果数量: {boxes_count}")
            print(f"分割掩码数量: {masks_count}")
            
            # 详细检测信息
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    conf = box.conf.cpu().numpy()[0] if hasattr(box, 'conf') else 0
                    cls = int(box.cls.cpu().numpy()[0]) if hasattr(box, 'cls') else -1
                    print(f"  检测{i}: Class {cls}, Conf {conf:.3f}")
            else:
                print("  ❌ 未检测到任何目标")
        else:
            print("❌ 模型推理无结果")
            
        print(f"注意力层数量: {len(attention_maps)}")
        
        for name, attn_map in attention_maps.items():
            attn_min, attn_max = attn_map.min(), attn_map.max()
            attn_mean = attn_map.mean()
            print(f"  {name}: 范围 [{attn_min:.4f}, {attn_max:.4f}], 均值 {attn_mean:.4f}")


def main():
    """主函数 - 直接定义参数"""
    
    # ===== 参数配置区域 =====
    # 模型配置
    MODEL_PATH = "dual_yolo/runs/segment/dual_modal_train_crossattn-precise/weights/best.pt"
    MODEL_YAML = "dual_yolo/models/yolo11x-dseg-crossattn-precise.yaml"
    
    # 输入数据（二选一）
    # 方式1: 使用.npy双模态文件
    NPY_PATH = "datasets/Dual-Modal-1504-500-1-6ch/test/images/2022-03-28_103204_17_T5_2412_0.npy"
    
    # 方式2: 使用分离的蓝光/白光图像（如果使用方式1，这两个参数会被忽略）
    BLUE_IMG = None  # "datasets/Dual-Modal-1504-500-1/test/images_b/2022-03-28_103204_17_T5_2412_0.jpg"
    WHITE_IMG = None  # "datasets/Dual-Modal-1504-500-1/test/images_w/2022-03-28_103204_17_T3_2410_0.jpg"
    
    # 输出配置
    OUTPUT_PATH = "attention_visualization_result.png"
    
    # ===== 参数验证 =====
    project_root = Path(__file__).parent.parent
    
    # 转换为绝对路径
    model_path = project_root / MODEL_PATH if not Path(MODEL_PATH).is_absolute() else Path(MODEL_PATH)
    model_yaml = project_root / MODEL_YAML if not Path(MODEL_YAML).is_absolute() else Path(MODEL_YAML)
    
    print("=== 参数配置 ===")
    print(f"模型权重: {model_path}")
    print(f"模型配置: {model_yaml}")
    
    # 验证模型文件
    if not model_path.exists():
        print(f"❌ 模型权重文件不存在: {model_path}")
        return
    
    if not model_yaml.exists():
        print(f"❌ 模型配置文件不存在: {model_yaml}")
        return
    
    # 处理输入数据
    npy_path = None
    blue_path = None
    white_path = None
    
    if NPY_PATH:
        npy_path = project_root / NPY_PATH if not Path(NPY_PATH).is_absolute() else Path(NPY_PATH)
        if not npy_path.exists():
            print(f"❌ .npy文件不存在: {npy_path}")
            return
        print(f"输入数据: {npy_path} (.npy双模态)")
        
    elif BLUE_IMG and WHITE_IMG:
        blue_path = project_root / BLUE_IMG if not Path(BLUE_IMG).is_absolute() else Path(BLUE_IMG)
        white_path = project_root / WHITE_IMG if not Path(WHITE_IMG).is_absolute() else Path(WHITE_IMG)
        
        if not blue_path.exists():
            print(f"❌ 蓝光图像不存在: {blue_path}")
            return
        if not white_path.exists():
            print(f"❌ 白光图像不存在: {white_path}")
            return  
        print(f"输入数据: {blue_path} + {white_path} (分离图像)")
        
    else:
        print("❌ 请配置NPY_PATH或者(BLUE_IMG, WHITE_IMG)参数")
        return
    
    output_path = Path(OUTPUT_PATH)
    print(f"输出路径: {output_path.absolute()}")
    
    # ===== 执行可视化 =====
    try:
        print("\n=== 开始执行可视化 ===")
        visualizer = AttentionVisualizer(str(model_path), str(model_yaml))
        
        if npy_path:
            visualizer.visualize(npy_path=str(npy_path), output_path=str(output_path))
        else:
            visualizer.visualize(blue_path=str(blue_path), white_path=str(white_path), output_path=str(output_path))
            
        print(f"✅ 可视化完成！结果保存在: {output_path.absolute()}")
        
    except Exception as e:
        print(f"❌ 可视化过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()