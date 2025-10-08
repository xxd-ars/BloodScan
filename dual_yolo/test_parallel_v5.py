"""
测试V5并行评估脚本 - 单模型快速验证
"""

import multiprocessing as mp
from d_model_evaluate_v5_parallel import run_parallel_evaluation_v5

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # 测试单个模型
    model_name = 'id-blue-2'
    train_mode = 'pretrained'
    conf_thresholds = [0.25, 0.3, 0.4, 0.5]
    num_gpus = 4

    print("=" * 80)
    print("开始V5并行评估测试 - 终极优化版")
    print(f"模型: {model_name}")
    print(f"使用GPU数: {num_gpus}")
    print(f"conf阈值: {conf_thresholds}")
    print(f"推理策略: 1次推理 + CPU过滤 = 5x加速")
    print("=" * 80)

    run_parallel_evaluation_v5(
        model_name=model_name,
        train_mode=train_mode,
        conf_thresholds=conf_thresholds,
        num_gpus=num_gpus
    )

    print("\n✅ V5并行评估测试完成！")
