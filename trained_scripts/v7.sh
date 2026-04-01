#!/bin/bash
source /vlm-ssd/zhangyan/venv/opd/bin/activate
cd /mnt/vlm-ks3/zhangyan/nips_code/opd

# 修改 run_name 以示区别
run_name="v7_venus_hyperdata"

nnodes=1
nproc_per_node=8

NNODES=$nnodes \
NODE_RANK=${RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=$nproc_per_node \
IMAGE_MAX_TOKEN_NUM=10000 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /mnt/vlm-ks3/zhangyan/pretrained_models/hf/inclusionAI/UI-Venus-1.5-2B \
    --model_type "qwen3_vl" \
    --train_type full \
    --dataset "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/hyper_data/all_venus.jsonl" \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --save_steps 1000 \
    --save_total_limit 100 \
    --logging_steps 1 \
    --max_length 20000 \
    --output_dir output/${run_name} \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 64 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --attn_impl flash_attn 