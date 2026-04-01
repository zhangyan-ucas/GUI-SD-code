#!/bin/bash
source /vlm-ssd/zhangyan/venv/opd/bin/activate
cd /mnt/vlm-ks3/zhangyan/nips_code/opd

# 修改 run_name 以示区别
run_name="v45_grpo"

nnodes=1
nproc_per_node=2

IMAGE_MAX_TOKEN_NUM=10000 \
NNODES=$nnodes \
NODE_RANK=${RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=6,7 \
swift rollout \
    --model "/vlm-ssd/FoundationModel/mllm/Qwen3-VL-8B-Instruct" \
    --vllm_data_parallel_size 2 \
    --host 127.0.0.1 \
    --port 12000 &

sleep 360

nnodes=1
nproc_per_node=6

# 2. CUDA_VISIBLE_DEVICES 包含 0-7 所有卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
IMAGE_MAX_TOKEN_NUM=10000 \
NNODES=$nnodes \
NODE_RANK=${RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=$nproc_per_node \
swift \
    rlhf \
    --rlhf_type grpo \
    --model "/vlm-ssd/FoundationModel/mllm/Qwen3-VL-8B-Instruct" \
    --reward_weights "0.8" "0.2" \
    --reward_funcs "ground-acc" "ground-format" \
    --external_plugins swift/cus_rewards/ground_reward.py \
    --attn_impl sdpa \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 12000 \
    --vllm_max_model_len 20000 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/ground_benchmark/venusbench.jsonl" \
    --load_from_cache_file false \
    --max_length 20000 \
    --max_completion_length 128 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 8 \
    --save_strategy steps \
    --save_steps 200 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --output_dir output/${run_name} \
    --warmup_ratio 0.05  \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --deepspeed zero3 \
    --log_completions true \
    --report_to none \
    --beta 0.04 \
    --top_p 0.9 \
    --top_k 50 \
    --num_iterations 1 \
    