CUDA_VISIBLE_DEVICES=2 \
swift rollout \
    --model /mnt/vlm-ks3/zhangyan/pretrained_models/hf/Qwen/Qwen3-VL/Qwen3-VL-8B-Instruct \
    --vllm_max_model_len 24192 \
    --vllm_gpu_memory_utilization 0.8 \
    --port 8193 \
    --model_type "qwen3_vl"