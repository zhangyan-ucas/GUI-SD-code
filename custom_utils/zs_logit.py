from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
import json 
import os 
import base64
import torch # 记得导入 torch

model_name_or_path = "/mnt/vlm-ks3/zhangyan/pretrained_models/hf/Qwen/Qwen3-VL/Qwen3-VL-8B-Instruct"
model = AutoModelForImageTextToText.from_pretrained(
   model_name_or_path, dtype="auto", device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
processor = AutoProcessor.from_pretrained(model_name_or_path)


anno_path = "/vlm-ssd/FoundationModel/dataset/ScreenSpot-Pro/annotations.json"
with open(anno_path, 'r') as f:
    total_data = json.load(f)

# 设定你想要查看的 Top-K 数量
top_k_num = 1

for sample in total_data:
    image_path = os.path.join("/vlm-ssd/FoundationModel/dataset/ScreenSpot-Pro/images",sample['img_filename'])

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    with open("/mnt/vlm-ks3/zhangyan/nips_code/opd/synth_pipeline/prompts/qwen3vl_ground.txt", encoding='utf-8') as f:
        sys_prompt = f.read()

    instruction = sample['instruction']
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text", 
                    "text": sys_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                    "max_pixels": 8294400 
                },
                {
                    "type": "text", 
                    "text": instruction,
                }
            ],
        },
    ]

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    # 【改动点 1】: 添加 return_dict_in_generate 和 output_scores
    outputs = model.generate(
        **inputs, 
        max_new_tokens=128,
        return_dict_in_generate=True,
        output_scores=True
    )
    
    # 提取生成的序列
    generated_ids = outputs.sequences
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print(f"\n==========================================")
    print(f"最终输出文本: {output_text[0]}")
    print(f"==========================================\n")

    # 【改动点 2】: 遍历 outputs.scores 获取每一步的 Top-K
    # outputs.scores 是一个 tuple，长度为生成的 token 数量。
    # 每一个元素是一个 tensor，形状为 (batch_size, vocab_size)
    print(f"--- 每一个 Token 位置的 Top-{top_k_num} 预测结果 ---")
    
    for step_idx, step_logits in enumerate(outputs.scores):
        # 因为 batch_size 为 1，我们直接取索引 0
        logits = step_logits[0] 
        
        # 将 logits 转换为概率分布 (可选，但通常比看原始 logits 更直观)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # 获取 Top-K 的概率和对应的 Token ID
        topk_probs, topk_indices = torch.topk(probs, top_k_num)
        
        print(f"第 {step_idx + 1} 步 (生成第 {step_idx + 1} 个 Token):")
        for rank in range(top_k_num):
            token_id = topk_indices[rank].item()
            prob = topk_probs[rank].item()
            # 解码 Token ID 为字符串 (使用 repr 避免空格换行符导致打印错乱)
            token_str = repr(tokenizer.decode([token_id])) 
            
            print(f"  Top {rank+1}: {token_str:<10} (ID: {token_id:<6}) - 概率: {prob:.4%}")
    
    print("\n")
    # 如果只想测试一个样本看效果，可以在这里 break
    break