#!/bin/bash
source /vlm-ssd/zhangyan/venv/opd/bin/activate
cd /mnt/vlm-ks3/zhangyan/nips_code/opd

run_name="${1}"
dataset_name="${2}"


# v* 目录（例如 v0-202...）
base_dir="/mnt/vlm-ks3/zhangyan/nips_code/opd/output/${run_name}"
vdir=$(ls -1dt ${base_dir}/v* | head -n 1)

echo "Using vdir: ${vdir}"

if [ "${dataset_name}" = "screenspotpro" ]; then
    # val_dataset="/mnt/vlm-ks3/zhangyan/datasets/swift_data/jigsaw_swift_data/screenspotpro/temp/venus15_ground.jsonl"
    val_dataset="/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/screenspotpro/qwen3vl.jsonl"

elif [ "${dataset_name}" = "ui-vision" ]; then
    val_dataset="/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/ground_benchmark/ui-vision.jsonl"

fi

# ====== 遍历所有 checkpoint ======


# 每个 checkpoint 各自保存 infer 结果
ckpt="/mnt/vlm-ks3/zhangyan/nips_code/opd/output/v8_qwen3vl_hyperdata/v0-20260312-092457/checkpoint-1543"
output_path="${ckpt}/infer_result/${dataset_name}.jsonl"
mkdir -p "$(dirname ${output_path})"

nnodes=1
nproc_per_node=1

CUDA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=10000 \
NNODES=$nnodes \
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
    

