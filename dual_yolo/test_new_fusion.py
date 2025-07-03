from ultralytics import YOLO
import torch

def test_new_fusion():
    print("=== 测试新的融合策略 ===")
    
    # 测试新的配置
    try:
        model = YOLO('./dual_yolo/yolo11x-dseg.yaml')
        print("✅ 模型配置加载成功")
        model.info()
        
        # 创建测试输入
        batch_size = 1
        x = torch.randn(batch_size, 6, 640, 640)  # 6通道输入（蓝光3 + 白光3）
        
        print('\n=== 测试前向传播 ===')
        model.model.eval()
        with torch.no_grad():
            outputs = model.model(x)
            print('✅ 新配置前向传播成功！')
            print(f'输出类型: {type(outputs)}')
            if isinstance(outputs, tuple):
                print(f'输出数量: {len(outputs)}')
                for i, output in enumerate(outputs):
                    if torch.is_tensor(output):
                        print(f'输出{i} 形状: {output.shape}')
                    elif isinstance(output, list):
                        print(f'输出{i} 包含 {len(output)} 个张量')
                        for j, tensor in enumerate(output):
                            print(f'  张量{j} 形状: {tensor.shape}')
    except Exception as e:
        print(f'❌ 测试失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_new_fusion()