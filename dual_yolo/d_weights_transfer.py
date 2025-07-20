#!/usr/bin/env python3
import torch, os
from pathlib import Path
from ultralytics import YOLO
project_root = Path(__file__).parent.parent

def load_and_inspect_pretrained():
    """加载预训练权重并检查结构"""
    pretrained_path = project_root / 'yolo_seg' / 'runs' / 'segment' / 'train_blue_rawdata_1504_500_10epoch' / 'weights' / 'best.pt'
    
    if not os.path.exists(pretrained_path):
        print(f"❌ 预训练文件不存在: {pretrained_path}")
        return None
    
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    state_dict = checkpoint['model'].state_dict()
    
    print(f"✅ 加载预训练权重: {pretrained_path}")
    print(f"总参数数量: {len(state_dict)}")
    
    # 按层编号分组显示
    layers = {}
    for name in state_dict.keys():
        if name.startswith('model.'):
            layer_num = int(name.split('.')[1])
            if layer_num not in layers:
                layers[layer_num] = []
            layers[layer_num].append(name)
    
    print(f"检测到层编号范围: {min(layers.keys())} - {max(layers.keys())}")
    print("前10层结构:")
    for i in range(min(10, max(layers.keys())+1)):
        if i in layers:
            print(f"  层{i}: {len(layers[i])}个参数")
    
    return state_dict, layers

def create_weight_mapping(original_layers):
    """创建权重映射关系"""
    mapping = {}
    
    # Backbone映射: 0-10 → 0-10 (backbone_b) 和 11-21 (backbone_w)
    for layer_num in range(11):  # 0-10
        if layer_num in original_layers:
            for param_name in original_layers[layer_num]:
                # 映射到backbone_b (保持原编号)
                mapping[param_name] = param_name
                # 映射到backbone_w (编号+11)
                new_layer_num = layer_num + 11
                new_param_name = param_name.replace(f'model.{layer_num}.', f'model.{new_layer_num}.')
                mapping[param_name + '_w'] = new_param_name
    
    # Head映射: 11-23 → 25-37 (跳过22-24的Identity层)
    head_mapping = {
        11: 25, 12: 26, 13: 27, 14: 28, 15: 29, 16: 30,
        17: 31, 18: 32, 19: 33, 20: 34, 21: 35, 22: 36, 23: 37
    }
    
    for old_layer, new_layer in head_mapping.items():
        if old_layer in original_layers:
            for param_name in original_layers[old_layer]:
                new_param_name = param_name.replace(f'model.{old_layer}.', f'model.{new_layer}.')
                mapping[param_name] = new_param_name
    
    return mapping

def transfer_weights():
    """执行权重迁移"""
    # 1. 加载预训练权重
    original_state_dict, original_layers = load_and_inspect_pretrained()
    if original_state_dict is None:
        return
    
    # 2. 创建新架构模型
    new_model = YOLO(project_root / 'dual_yolo' / 'models' / 'yolo11x-dseg.yaml')
    new_state_dict = new_model.model.state_dict()
    
    print(f"新模型参数数量: {len(new_state_dict)}")
    
    # 3. 创建映射关系
    mapping = create_weight_mapping(original_layers)
    
    print(f"创建映射关系: {len(mapping)}个映射")
    
    # 4. 执行权重复制
    transferred_count = 0
    skipped_count = 0
    
    for orig_key, new_key in mapping.items():
        if orig_key.endswith('_w'):
            # 处理白光backbone (复制蓝光backbone参数)
            source_key = orig_key[:-2]  # 移除'_w'后缀
            if source_key in original_state_dict and new_key in new_state_dict:
                if original_state_dict[source_key].shape == new_state_dict[new_key].shape:
                    new_state_dict[new_key] = original_state_dict[source_key].clone()
                    transferred_count += 1
                else:
                    print(f"⚠️  形状不匹配 {source_key}: {original_state_dict[source_key].shape} vs {new_key}: {new_state_dict[new_key].shape}")
                    skipped_count += 1
        else:
            # 处理蓝光backbone和head
            if orig_key in original_state_dict and new_key in new_state_dict:
                if original_state_dict[orig_key].shape == new_state_dict[new_key].shape:
                    new_state_dict[new_key] = original_state_dict[orig_key].clone()
                    transferred_count += 1
                else:
                    print(f"⚠️  形状不匹配 {orig_key}: {original_state_dict[orig_key].shape} vs {new_key}: {new_state_dict[new_key].shape}")
                    skipped_count += 1
    
    print(f"✅ 成功迁移: {transferred_count}个参数")
    print(f"⚠️  跳过: {skipped_count}个参数")
    
    # 5. 加载迁移后的权重到模型
    new_model.model.load_state_dict(new_state_dict)
    
    # 6. 保存新的权重文件
    output_path = project_root / 'dual_yolo' / 'weights' / 'dual_yolo_transferred.pt'
    os.makedirs(project_root / 'dual_yolo' / 'weights', exist_ok=True)
    
    # 构建完整的checkpoint
    checkpoint = {
        'model': new_model.model,
        'optimizer': None,
        'best_fitness': None,
        'epoch': 0,
        'date': None
    }
    
    torch.save(checkpoint, output_path)
    print(f"✅ 权重已保存到: {output_path}")
    
    return new_model, output_path

def verify_transfer():
    """验证迁移结果"""
    print("\n" + "="*50)
    print("🔍 验证权重迁移结果")
    print("="*50)
    
    # 加载迁移后的模型
    model = YOLO(project_root / 'dual_yolo' / 'models' / 'yolo11x-dseg.yaml').load(project_root / 'dual_yolo' / 'weights' / 'dual_yolo_transferred.pt')
    
    # 检查第一层参数
    first_params = []
    for name, param in model.model.named_parameters():
        if 'model.0.' in name or 'model.11.' in name:
            first_params.append((name, param.shape, param.mean().item()))
    
    print("前两层参数对比:")
    backbone_b_params = [p for p in first_params if 'model.0.' in p[0]]
    backbone_w_params = [p for p in first_params if 'model.11.' in p[0]]
    
    for (name_b, shape_b, mean_b), (name_w, shape_w, mean_w) in zip(backbone_b_params, backbone_w_params):
        print(f"  {name_b}: {mean_b:.6f}")
        print(f"  {name_w}: {mean_w:.6f}")
        if abs(mean_b - mean_w) < 1e-6:
            print("  ✅ 参数相同")
        else:
            print("  ❌ 参数不同")
        print()

if __name__ == "__main__":
    model, output_path = transfer_weights()
    verify_transfer()
    print(f"\n🎉 权重迁移完成！新模型已保存到: {output_path}") 