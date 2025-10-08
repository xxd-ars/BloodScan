#!/bin/bash
# 多GPU并行评估启动脚本
# 使用CUDA_VISIBLE_DEVICES在进程级隔离GPU

# 设置PyTorch CUDA内存分配器
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "启动多GPU并行评估..."
echo "提示：每个Python进程将独立启动并使用一张GPU"
echo "========================================="

# 在后台启动4个进程，每个使用不同的GPU
CUDA_VISIBLE_DEVICES=0 python -c "
from d_model_evaluate_v4_parallel import run_parallel_evaluation
run_parallel_evaluation('id-blue-2', 'pretrained', 0.5, num_gpus=1)
" &
PID1=$!

CUDA_VISIBLE_DEVICES=1 python -c "
from d_model_evaluate_v4_parallel import run_parallel_evaluation
run_parallel_evaluation('id-blue-2', 'pretrained', 0.5, num_gpus=1)
" &
PID2=$!

CUDA_VISIBLE_DEVICES=2 python -c "
from d_model_evaluate_v4_parallel import run_parallel_evaluation
run_parallel_evaluation('id-blue-2', 'pretrained', 0.5, num_gpus=1)
" &
PID3=$!

CUDA_VISIBLE_DEVICES=3 python -c "
from d_model_evaluate_v4_parallel import run_parallel_evaluation
run_parallel_evaluation('id-blue-2', 'pretrained', 0.5, num_gpus=1)
" &
PID4=$!

echo "已启动4个进程:"
echo "  进程 $PID1 → GPU 0"
echo "  进程 $PID2 → GPU 1"
echo "  进程 $PID3 → GPU 2"
echo "  进程 $PID4 → GPU 4"
echo ""
echo "等待所有进程完成..."

# 等待所有后台进程完成
wait $PID1
wait $PID2
wait $PID3
wait $PID4

echo ""
echo "✅ 所有GPU评估完成！"
