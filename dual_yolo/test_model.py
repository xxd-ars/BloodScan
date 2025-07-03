import torch
import torch.nn.functional as F
# from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import os
# 确保使用本地修改的ultralytics代码
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ultralytics import YOLO

project_dir= "./yolo_seg"
# project_dir= "/home/xin99/BloodScan/yolo_seg"

epoch_num  = 10
model_name = f"train_blue_rawdata_1504_500_{epoch_num}epoch"
dataset_v  = "Blue-Rawdata-1504-500-1"
pt_path = project_dir + f"/runs/segment/{model_name}/weights/best.pt"

# model = YOLO("./dual_yolo/models/yolo11x-seg.yaml").load(pt_path)
# model.info(verbose=True)


fusion_dict = {
    'concat_compress': './dual_yolo/models/yolo11x-dseg-concat-compress.yaml',
    'weighted_fusion': './dual_yolo/models/yolo11x-dseg-weighted-fusion.yaml',
    'cross_attn': './dual_yolo/models/yolo11x-dseg-crossattn.yaml'
}
model_yaml = fusion_dict['concat_compress']
model_pt = './dual_yolo/weights/dual_yolo11x.pt'

model_dual = YOLO(model_yaml).load(model_pt)
model_dual.info(verbose=True)

# 加载蓝光和白光图像
img_path_b = 'data/rawdata_cropped/class1/2022-03-28_103204_1_T5_2348.jpg'
img_path_w = 'data/rawdata_cropped/class1/2022-03-28_103204_1_T3_2346.jpg'
image_b = Image.open(img_path_b).convert('RGB')
image_w = Image.open(img_path_w).convert('RGB')
print(f"原图尺寸 (白光): {image_w.size}")
transform = transforms.Compose([transforms.ToTensor()])

# 准备双模态输入张量
blue_tensor = transform(image_b).unsqueeze(0)   # [1, 3, H, W]
white_tensor = transform(image_w).unsqueeze(0)  # [1, 3, H, W]
dual_tensor = torch.cat([blue_tensor, white_tensor], dim=1)  # [1, 6, H, W]

print(f"蓝光图像张量: {blue_tensor.shape}")
print(f"白光图像张量: {white_tensor.shape}")
print(f"双模态拼接张量: {dual_tensor.shape}")

# model.predict(blue_tensor, visualize = True)
# 0: 1504x1504 1 0, 1 1, 1 2, 4991.7ms
# Speed: 0.0ms preprocess, 4991.7ms inference, 19.6ms postprocess per image at shape (1, 3, 1504, 1504)

model_dual.predict(dual_tensor) #, visualize = True)

# # 双模态推理
# model.model.eval()
# with torch.no_grad():
#     outputs = model.model(dual_tensor)  # 传入6通道拼接张量
#     # outputs = model.model(blue_tensor, white_tensor)  # 传入6通道拼接张量
#     print(f"\n推理成功！输出类型: {type(outputs)}")
    
#     # 解析模型输出
#     if isinstance(outputs, tuple) and len(outputs) >= 2:
#         # 检测输出 [1, 39, 46389] - 包含边界框、置信度、类别、掩膜系数
#         detection_output = outputs[0]  # [1, 39, 46389]
        
#         # 分割相关输出
#         seg_outputs = outputs[1]
#         proto_coeffs = seg_outputs[1]  # [1, 32, 46389] - 掩膜系数
#         proto_features = seg_outputs[2]  # [1, 32, 376, 376] - 掩膜原型特征图
        
#         print(f"\n=== 模型输出解析 ===")
#         print(f"检测输出形状: {detection_output.shape}")  # [1, 39, 46389]
#         print(f"掩膜系数形状: {proto_coeffs.shape}")      # [1, 32, 46389]
#         print(f"掩膜原型特征图形状: {proto_features.shape}")  # [1, 32, 376, 376]
        
#         # 解析检测结果
#         # 前4个值是边界框 (x_center, y_center, width, height)
#         # 接下来是类别置信度 (假设3个类别)
#         # 最后32个值是掩膜系数
#         detection_data = detection_output[0]  # [39, 46389]
        
#         # 分离不同的数据
#         boxes = detection_data[:4]  # [4, 46389] - 边界框
#         class_scores = detection_data[4:7]  # [3, 46389] - 类别置信度 (假设3个类别)
#         mask_coeffs = detection_data[7:]  # [32, 46389] - 掩膜系数
        
#         print(f"边界框数据形状: {boxes.shape}")
#         print(f"类别置信度形状: {class_scores.shape}")
#         print(f"掩膜系数形状: {mask_coeffs.shape}")
        
#         # 计算目标置信度 (最大类别置信度)
#         max_class_scores, predicted_classes = torch.max(class_scores, dim=0)  # [46389]
        
#         # 设置置信度阈值
#         conf_threshold = 0.25
#         valid_detections = max_class_scores > conf_threshold
        
#         print(f"\n=== 检测统计 ===")
#         print(f"总检测数: {valid_detections.sum().item()}")
#         print(f"置信度阈值: {conf_threshold}")
        
#         if valid_detections.sum() > 0:
#             # 获取有效检测
#             valid_boxes = boxes[:, valid_detections]  # [4, N_valid]
#             valid_scores = max_class_scores[valid_detections]  # [N_valid]
#             valid_classes = predicted_classes[valid_detections]  # [N_valid]
#             valid_mask_coeffs = mask_coeffs[:, valid_detections]  # [32, N_valid]
            
#             print(f"有效检测数: {valid_boxes.shape[1]}")
#             print(f"检测到的类别: {torch.unique(valid_classes).tolist()}")
            
#             # === 生成分割掩膜 ===
#             print(f"\n=== 生成分割掩膜 ===")
            
#             # 获取掩膜原型特征 [32, 376, 376]
#             proto_features_2d = proto_features[0]  # [32, 376, 376]
            
#             # 为每个有效检测生成掩膜
#             all_masks = []
#             for i in range(valid_boxes.shape[1]):
#                 # 获取当前检测的掩膜系数 [32]
#                 coeffs = valid_mask_coeffs[:, i]  # [32]
                
#                 # 计算掩膜: coeffs @ proto_features_2d
#                 # coeffs: [32], proto_features_2d: [32, 376, 376]
#                 mask = torch.sum(coeffs.unsqueeze(-1).unsqueeze(-1) * proto_features_2d, dim=0)  # [376, 376]
                
#                 # 应用sigmoid激活
#                 mask = torch.sigmoid(mask)  # [376, 376]
#                 all_masks.append(mask)
            
#             # 合并所有掩膜 (取最大值)
#             if all_masks:
#                 combined_mask = torch.stack(all_masks, dim=0)  # [N_valid, 376, 376]
#                 final_mask, _ = torch.max(combined_mask, dim=0)  # [376, 376]
                
#                 print(f"最终掩膜形状: {final_mask.shape}")
#                 print(f"掩膜值范围: [{final_mask.min().item():.4f}, {final_mask.max().item():.4f}]")
                
#                 # 转换为二值掩膜
#                 mask_threshold = 0.5
#                 binary_mask = (final_mask > mask_threshold).float()
                
#                 # 统计分割结果
#                 foreground_pixels = binary_mask.sum().item()
#                 total_pixels = binary_mask.numel()
#                 foreground_ratio = foreground_pixels / total_pixels
                
#                 print(f"前景像素数: {int(foreground_pixels)}")
#                 print(f"前景占比: {foreground_ratio:.2%}")
                
#                 if foreground_pixels > 0:
#                     print(f"\n🎯 成功检测到分割区域！")
                    
#                     # === 可视化结果 ===
#                     print(f"\n=== 开始可视化 ===")
                    
#                     # 转换为numpy
#                     binary_mask_np = binary_mask.cpu().numpy()
#                     final_mask_np = final_mask.cpu().numpy()
                    
#                     # 调整掩膜大小以匹配原图
#                     img_array = np.array(image_w)
#                     original_size = (img_array.shape[1], img_array.shape[0])  # (width, height)
                    
#                     # 调整掩膜大小
#                     mask_resized = cv2.resize(binary_mask_np, original_size, interpolation=cv2.INTER_NEAREST)
#                     mask_prob_resized = cv2.resize(final_mask_np, original_size, interpolation=cv2.INTER_LINEAR)
                    
#                     print(f"原图尺寸: {img_array.shape}")
#                     print(f"调整后掩膜尺寸: {mask_resized.shape}")
                    
#                     # 创建彩色掩膜 (红色表示检测到的区域)
#                     colored_mask = np.zeros((*mask_resized.shape, 3), dtype=np.uint8)
#                     colored_mask[mask_resized > 0] = [255, 0, 0]  # 红色
                    
#                     # 创建概率热力图
#                     heatmap = plt.cm.jet(mask_prob_resized)[:, :, :3] * 255
#                     heatmap = heatmap.astype(np.uint8)
                    
#                     # 叠加到原图
#                     alpha = 0.4
#                     overlay_binary = cv2.addWeighted(img_array, 1-alpha, colored_mask, alpha, 0)
#                     overlay_heatmap = cv2.addWeighted(img_array, 1-alpha, heatmap, alpha, 0)
                    
#                     # === 保存结果 ===
#                     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    
#                     # 原图 (白光)
#                     axes[0, 0].imshow(image_w)
#                     axes[0, 0].set_title('原图 (白光)', fontsize=14, fontweight='bold')
#                     axes[0, 0].axis('off')
                    
#                     # 概率掩膜
#                     im1 = axes[0, 1].imshow(mask_prob_resized, cmap='jet', vmin=0, vmax=1)
#                     axes[0, 1].set_title('分割概率图', fontsize=14, fontweight='bold')
#                     axes[0, 1].axis('off')
#                     plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
                    
#                     # 二值掩膜
#                     axes[0, 2].imshow(mask_resized, cmap='gray')
#                     axes[0, 2].set_title('二值分割掩膜', fontsize=14, fontweight='bold')
#                     axes[0, 2].axis('off')
                    
#                     # 彩色分割图
#                     axes[1, 0].imshow(colored_mask)
#                     axes[1, 0].set_title('彩色分割图', fontsize=14, fontweight='bold')
#                     axes[1, 0].axis('off')
                    
#                     # 概率热力图叠加
#                     axes[1, 1].imshow(overlay_heatmap)
#                     axes[1, 1].set_title('概率热力图叠加', fontsize=14, fontweight='bold')
#                     axes[1, 1].axis('off')
                    
#                     # 二值掩膜叠加
#                     axes[1, 2].imshow(overlay_binary)
#                     axes[1, 2].set_title('二值掩膜叠加', fontsize=14, fontweight='bold')
#                     axes[1, 2].axis('off')
                    
#                     plt.tight_layout()
#                     plt.savefig('dual_yolo/segmentation_results_corrected.png', dpi=300, bbox_inches='tight')
#                     print(f"✅ 分割结果已保存到: dual_yolo/segmentation_results_corrected.png")
                    
#                     # 保存单独的叠加图像
#                     overlay_pil = Image.fromarray(overlay_binary)
#                     overlay_pil.save('dual_yolo/overlay_corrected.png')
#                     print(f"✅ 叠加图像已保存到: dual_yolo/overlay_corrected.png")
                    
#                     # 输出检测框信息
#                     print(f"\n=== 检测框详细信息 ===")
#                     for i in range(valid_boxes.shape[1]):
#                         x_center, y_center, width, height = valid_boxes[:, i].cpu().numpy()
#                         score = valid_scores[i].cpu().item()
#                         class_id = valid_classes[i].cpu().item()
                        
#                         # 转换为原图坐标 (假设模型输入是640x640)
#                         img_h, img_w = img_array.shape[:2]
#                         x_center_orig = x_center * img_w / 640
#                         y_center_orig = y_center * img_h / 640
#                         width_orig = width * img_w / 640
#                         height_orig = height * img_h / 640
                        
#                         x1 = int(x_center_orig - width_orig / 2)
#                         y1 = int(y_center_orig - height_orig / 2)
#                         x2 = int(x_center_orig + width_orig / 2)
#                         y2 = int(y_center_orig + height_orig / 2)
                        
#                         print(f"检测 {i+1}:")
#                         print(f"  类别: {class_id}, 置信度: {score:.4f}")
#                         print(f"  边界框: ({x1}, {y1}) -> ({x2}, {y2})")
#                         print(f"  尺寸: {x2-x1} x {y2-y1}")
                        
#                 else:
#                     print(f"\n❌ 掩膜中未检测到前景区域")
#             else:
#                 print(f"\n❌ 未生成有效掩膜")
#         else:
#             print(f"\n❌ 未检测到满足置信度阈值的目标")
#     else:
#         print(f"❌ 模型输出格式不符合预期")
