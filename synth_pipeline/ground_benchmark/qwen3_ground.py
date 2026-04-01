import json 
import argparse
from MMbench import mmbench_builer
from os_worldg import osworldg_builer
from uivision import uivision_builer
from venusbench import VenusBench_builer
from screenspotpro import screenspotpro_builder
from swift.custom_utils.ground_func import bboxreal2norm

def messages_builder(sample, sys_prompt):

    norm_bbox = bboxreal2norm(sample['bbox'], sample['additional_paras']['image_size'])
    norm_point = [int(sum(norm_bbox[::2]) // 2), int(sum(norm_bbox[1::2]) // 2)]

    ass_text = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [p_x, p_y]}}\n</tool_call>'
    ass_text = ass_text.replace("p_x", str(norm_point[0])).replace("p_y", str(norm_point[1]))
    
    messages = [
        {"content": sys_prompt, "role": "system"},
        {"content": sample['instruction'], "role": "user"},
        {"content": ass_text, "role": "assistant"},
    ]

    ret = {
        "solution": {'name': 'mobile_use', 'arguments': {'action': 'click', 'coordinate': sample['bbox']}},   
        "images": sample['images'], 
        "messages": messages, 
        "additional_paras": json.dumps(sample['additional_paras']),
    }

    return ret

def sample2files(total_samples, save_path):
    # 只需要在最开始读取一次 system prompt
    with open("/mnt/vlm-ks3/zhangyan/nips_code/opd/synth_pipeline/prompts/qwen3vl_ground.txt", encoding='utf-8') as f:
        sys_prompt = f.read()



    with open(save_path, 'w', encoding='utf-8') as f:
        for idx, sample in enumerate(total_samples):
            ret = messages_builder(sample, sys_prompt)
            ret['sample_id'] = idx
            json_line = json.dumps(ret, ensure_ascii=False)
            f.write(json_line + '\n')
            
    print(f"成功保存 {len(total_samples)} 条数据到 {save_path}")

def collect_samples(args):
    total_samples = []
    
    # 遍历所有传入的数据集名称
    for dataset_name in args.dataset_names:
        print(f"正在处理数据集: {dataset_name}...")
        if dataset_name == 'screenspotpro':
            samples = screenspotpro_builder()
        elif dataset_name == 'os_worldg':
            samples = osworldg_builer()
        elif dataset_name == 'mmbench':
            samples = mmbench_builer()
        elif dataset_name == 'ui-vision':
            samples = uivision_builer()
        elif dataset_name == 'venusbench':
            samples = VenusBench_builer()
        else:
            print(f"警告: 未知的数据集 {dataset_name}")
            continue
            
        # 将当前数据集的样本追加到总列表中
        total_samples.extend(samples)
        
    return total_samples

def parse_args():
    parser = argparse.ArgumentParser(description="处理多个数据集并汇总保存到同一个JSONL文件的脚本")
    
    # 使用 nargs='+' 允许接收多个用空格分隔的参数
    parser.add_argument(
        '--dataset_names', 
        nargs='+', 
        type=str, 
        default=['mmbench', ], 
        # default=['venusbench', 'os_worldg', 'mmbench', ], 
        choices=['screenspotpro', 'os_worldg', 'mmbench', 'ui-vision', 'venusbench'],
    )
    
    parser.add_argument(
        '--save_path', 
        type=str, 
        default="/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/ground_benchmark/mmbench.jsonl"
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    total_samples = collect_samples(args)
    sample2files(total_samples, args.save_path)