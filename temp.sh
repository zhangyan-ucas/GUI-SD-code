

#!/bin/bash

OUTPUT_DIR="/mnt/vlm-ks3/zhangyan/nips_code/opd/output/v51_opsd/v1-20260331-152348"  # 改成你的实际路径

for ckpt in checkpoint-5 checkpoint-15 checkpoint-20 checkpoint-25 checkpoint-30 checkpoint-35 checkpoint-40 checkpoint-45 checkpoint-50 checkpoint-55 checkpoint-60 checkpoint-64; do
    # 跳过已经 merge 过的
    if [ -d "${OUTPUT_DIR}/${ckpt}-merged" ]; then
        echo ">>> 跳过 ${ckpt}，已存在 merged 目录"
        continue
    fi

    echo ">>> 正在 merge ${ckpt} ..."
    CUDA_VISIBLE_DEVICES=0 swift export \
        --adapters "${OUTPUT_DIR}/${ckpt}" \
        --merge_lora true

    echo ">>> ${ckpt} merge 完成"
done

echo ">>> 全部完成！"