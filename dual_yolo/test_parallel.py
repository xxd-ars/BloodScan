"""
测试并行评估脚本 - 单模型快速验证
"""

import multiprocessing as mp
from d_model_evaluate_v4_parallel import run_parallel_evaluation

if __name__ == '__main__':
    # 设置multiprocessing启动方式为spawn（PyTorch + CUDA必需）
    mp.set_start_method('spawn', force=True)

    # 测试单个模型
    model_name = 'id-blue-2'  # 最佳模型
    train_mode = 'pretrained'
    conf_medical = 0.5
    num_gpus = 4

    print("=" * 80)
    print("开始并行评估测试")
    print(f"模型: {model_name}")
    print(f"使用GPU数: {num_gpus}")
    print("=" * 80)

    run_parallel_evaluation(
        model_name=model_name,
        train_mode=train_mode,
        conf_medical=conf_medical,
        num_gpus=num_gpus
    )

    print("\n✅ 并行评估测试完成！")
