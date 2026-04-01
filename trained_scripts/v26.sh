#!/bin/bash
source /vlm-ssd/zhangyan/venv/opd/bin/activate
cd /mnt/vlm-ks3/zhangyan/nips_code/opd

# 修改 run_name 以示区别
run_name="v26_opsd_venusbench"

nnodes=1
nproc_per_node=1

CUDA_VISIBLE_DEVICES=7 \
NNODES=$nnodes \
NODE_RANK=${RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=$nproc_per_node \
IMAGE_MAX_TOKEN_NUM=10000 \
swift rollout \
    --model_type "qwen3_vl" \
    --model "/vlm-ssd/FoundationModel/mllm/Qwen3-VL-8B-Instruct" \
    --vllm_gpu_memory_utilization 0.8 \
    --vllm_max_model_len 20000 \
    --vllm_data_parallel_size 1 \
    --host 127.0.0.1 \
    --port 8192 &

sleep 120

nnodes=1
nproc_per_node=7

NNODES=$nnodes \
NODE_RANK=${RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=$nproc_per_node \
IMAGE_MAX_TOKEN_NUM=10000 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
swift rlhf \
    --rlhf_type gkd \
    --use_opsd true \
    --opsd_mask_dir "/vlm-ssd/zhangyan/cache/opd_cache/260310" \
    --model "/vlm-ssd/FoundationModel/mllm/Qwen3-VL-8B-Instruct" \
    --model_type "qwen3_vl" \
    --teacher_model "/vlm-ssd/FoundationModel/mllm/Qwen3-VL-8B-Instruct" \
    --teacher_model_type "qwen3_vl" \
    --train_type full \
    --dataset "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/ground_benchmark/venusbench.jsonl" \
    --seq_kd false \
    --lmbda 1 \
    --beta 1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --save_steps 10 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --max_length 20000 \
    --max_completion_length 128 \
    --output_dir output/${run_name} \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --log_completions true \
    --dataloader_num_workers 64 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --teacher_deepspeed zero3 \
    --attn_impl flash_attn \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8192