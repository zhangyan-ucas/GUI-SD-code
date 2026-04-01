#!/bin/bash
source /vlm-ssd/zhangyan/venv/opd/bin/activate
cd /mnt/vlm-ks3/zhangyan/nips_code/opd


ckpt="/vlm-ssd/FoundationModel/mllm/Qwen3-VL-8B-Instruct"
val_dataset="/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/screenspotpro/qwen3vl.jsonl"
output_path="/mnt/vlm-ks3/zhangyan/nips_code/opd/useful_output/useful_output/qwen3vl/screenspotpro/8b.jsonl"


nnodes=1
nproc_per_node=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=10000 \
NNODES=$nnodes \
NODE_RANK=${RANK} \
MASTER_PORT=${MASTER_PORT} \
MASTER_ADDR=${MASTER_ADDR} \
NPROC_PER_NODE=$nproc_per_node \
swift \
    infer \
    --model ${ckpt} \
    --stream false \
    --infer_backend pt \
    --max_length 20000 \
    --max_new_tokens 1024 \
    --result_path ${output_path} \
    --val_dataset ${val_dataset} \
    --max_batch_size 4 \
    --remove_unused_columns false \
    --dataset_shuffle false \
    --write_batch_size 800 \
    --model_type qwen3_vl
    

