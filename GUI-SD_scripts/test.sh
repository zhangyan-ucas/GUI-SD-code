run_name="${1}"
shift  # 移除第1个参数，剩余的都是 dataset 名称
dataset_names=("$@")

if [ ${#dataset_names[@]} -eq 0 ]; then
    echo "Usage: bash infer_node.sh <run_name> <dataset1> [dataset2] [dataset3] ..."
    echo "Supported datasets: screenspotpro, screenspotv2, uivision, mmbench, osworldg, osworldg_r"
    exit 1
fi

echo "Run name: ${run_name}"
echo "Datasets: ${dataset_names[@]}"

# dataset_name -> val_dataset 路径映射
declare -A DATASET_MAP
DATASET_MAP["screenspotpro"]="ground_benchmark/screenspotpro.jsonl"
DATASET_MAP["screenspotv2"]="ground_benchmark/screenspotv2.jsonl"
DATASET_MAP["uivision"]="ground_benchmark/ui-vision.jsonl"
DATASET_MAP["mmbench"]="ground_benchmark/mmbench.jsonl"
DATASET_MAP["osworldg"]="ground_benchmark/osworldg.jsonl"
DATASET_MAP["osworldg_r"]="ground_benchmark/osworldg_r.jsonl"

# v* 目录（例如 v0-202...）
base_dir="output/${run_name}"
vdir=$(ls -1dt ${base_dir}/v* | head -n 1)

echo "Using vdir: ${vdir}"

# ====== 遍历所有 checkpoint ======
for ckpt in $(ls -1d ${vdir}/checkpoint-* 2>/dev/null | sort -V); do

    echo "==============================="
    echo " Evaluating checkpoint: $ckpt"
    echo "==============================="

    # ====== 遍历所有 dataset ======
    for dataset_name in "${dataset_names[@]}"; do

        val_dataset="${DATASET_MAP[$dataset_name]}"
        if [ -z "${val_dataset}" ]; then
            echo "WARNING: unknown dataset '${dataset_name}', skipping."
            continue
        fi

        echo "--- Dataset: ${dataset_name} ---"

        output_path="${ckpt}/infer_result/${dataset_name}.jsonl"
        mkdir -p "$(dirname ${output_path})"

        # 如果结果文件已存在则先删除，避免追加写入导致数据重复
        if [ -f "${output_path}" ]; then
            echo "  Removing existing result: ${output_path}"
            rm -f "${output_path}"
        fi

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

        echo "save result at ${output_path}"
        echo ">>> ${ckpt} / ${dataset_name} finished"
    done
done

echo "All checkpoints in ${vdir} evaluated on datasets: ${dataset_names[@]}!"
