#!/bin/bash
source /vlm-ssd/zhangyan/venv/opd/bin/activate
cd /mnt/vlm-ks3/zhangyan/nips_code/opd

# 修改 run_name 以示区别
run_name="v20_sft_0310"

nnodes=1
nproc_per_node=8

IMAGE_MAX_TOKEN_NUM=10000 \
NNODES=$nnodes \
NODE_RANK=${RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model "/vlm-ssd/FoundationModel/mllm/Qwen3-VL-8B-Instruct" \
    --model_type "qwen3_vl" \
    --train_type full \
    --dataset "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/250310/250310_filter.jsonl" \
    --torch_dtype bfloat16 \
    --num_train_epochs 1.0 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --weight_decay 0 \
    --save_steps 100 \
    --save_total_limit 1000 \
    --logging_steps 1 \
    --max_length 20000 \
    --output_dir output/${run_name} \
    --save_only_model true \
    --dataloader_num_workers 64 \
    --dataset_num_proc 4 \
    --deepspeed zero3 \
    --attn_impl flash_attn