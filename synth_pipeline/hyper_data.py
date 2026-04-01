import json
from tqdm import tqdm 
import sys
import argparse
import random 
from swift.custom_utils.ground_func import (
    bboxreal2norm
)

def parse_args():
    parser = argparse.ArgumentParser(description="这是一个参数解析示例脚本")
    parser.add_argument('--output_path', type=str, 
        default="/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/hyper_data/all_qwen3vl.jsonl", help='输入文件的路径 (必须)')
    parser.add_argument('--model_name', type=str, default="qwen3vl", help='输入文件的路径 (必须)')
    parser.add_argument('--seed', type=int, default=42, )
    parser.add_argument('--num_sample', type=int, default=0, )

    args = parser.parse_args()
    return args


def qwen3vl_builer(dataset_name, sample):
    instruction = sample['instruction']
    
    gt_bbox = eval(sample['solution'])
    norm_bbox = bboxreal2norm(gt_bbox, [sample['width'], sample['height']])
    norm_point = [sum(norm_bbox[::2]) // 2, sum(norm_bbox[1::2]) // 2]

    with open("/mnt/vlm-ks3/zhangyan/cvpr_code/ms-swift-qwen3vl/synth_pipeline/prompts/qwen3vl_ground/report_mod.txt") as f:
        sys_prompt = f.read()

    ass_text = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [p_x, p_y]}}\n</tool_call>'
    ass_text = ass_text.replace("p_x", str(norm_point[0])).replace("p_y", str(norm_point[1]))

    messages = [
        {"role": "system", "content": sys_prompt},
        {"content": instruction, "role": "user"},
        {"content": ass_text, "role": "assistant"},
    ]

    ret = {
        "images": [sample['image_path']], 
        "messages": messages,  
        "additional_paras": {            
            "idx": f"{dataset_name}_{sample['id']}",
            "gt_bbox": eval(sample['solution']),
            "instruction": sample['instruction'],
            "image_size": [sample['width'], sample['height']]
        },
    }
    return ret 

def qwen25vl_builder(dataset_name, sample):
    instruction = sample['instruction']
    
    gt_bbox = eval(sample['solution'])
    norm_bbox = bboxreal2norm(gt_bbox, [sample['width'], sample['height']])
    norm_point = [sum(norm_bbox[::2]) // 2, sum(norm_bbox[1::2]) // 2]

    with open("/mnt/vlm-ks3/zhangyan/cvpr_code/ms-swift-qwen3vl/synth_pipeline/prompts/qwen3vl_ground/report_mod.txt") as f:
        sys_prompt = f.read()

    ass_text = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [p_x, p_y]}}\n</tool_call>'
    ass_text = ass_text.replace("p_x", str(norm_point[0])).replace("p_y", str(norm_point[1]))

    messages = [
        {"role": "system", "content": sys_prompt},
        {"content": instruction, "role": "user"},
        {"content": ass_text, "role": "assistant"},
    ]

    ret = {
        "images": [sample['image_path']], 
        "messages": messages,  
        "additional_paras": {            
            "idx": f"{dataset_name}_{sample['id']}",
            "gt_bbox": eval(sample['solution']),
            "instruction": sample['instruction'],
            "image_size": [sample['width'], sample['height']]
        },
    }
    return ret 


def venus15_builer(dataset_name, sample):
    instruction = sample['instruction']
    user_query = f"<image>Output the center point of the position corresponding to the following instruction: \n{instruction}. \n\nThe output should just be the coordinates of a point, in the format [x,y]. Additionally, if the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1]."

    gt_bbox = eval(sample['solution'])
    norm_bbox = bboxreal2norm(gt_bbox, [sample['width'], sample['height']])
    norm_point = [sum(norm_bbox[::2]) // 2, sum(norm_bbox[1::2]) // 2]

    messages = [
        {"content": user_query, "role": "user"},
        {"content": str(norm_point), "role": "assistant"},
    ]
    ret = {
        "images": [sample['image_path']], 
        "messages": messages,  
        "additional_paras": {            
            "idx": f"{dataset_name}_{sample['id']}",
            "gt_bbox": eval(sample['solution']),
            "instruction": sample['instruction'],
            "image_size": [sample['width'], sample['height']]
        },
    }
    return ret 

def main():
    args = parse_args()
    input_paths = {
        "omniact": "/mnt/vlm-ks3/zhangyan/datasets/hyper_data/omniact_4709_filter_1076_rollout_119.json",
        "widget_captioning": "/mnt/vlm-ks3/zhangyan/datasets/hyper_data/widget_captioning_128570_filter_35478_rollout_3672.json",
        "showui_web": "/mnt/vlm-ks3/zhangyan/datasets/hyper_data/showui_web_599186_filter_128483_rollout_19172.json",
        "ui_refexp": "/mnt/vlm-ks3/zhangyan/datasets/hyper_data/ui_refexp_16660_filter_2509_rollout_280.json",
        "os_atlas": "/mnt/vlm-ks3/zhangyan/datasets/hyper_data/os_atlas_583990_filter_116962_rollout_26114.json",
    }

    save_data = []
    for dataset_name, input_path in input_paths.items():

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for sample in tqdm(data, desc=f"正在合并文件 {dataset_name}"):
            if args.model_name == "venus":
                ret = venus15_builer(dataset_name, sample)
            else:
                ret = qwen3vl_builer(dataset_name, sample)
            save_data.append(ret)

    random.seed(args.seed)
    if args.num_sample != 0:
        save_data = random.sample(save_data, args.num_sample)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        for entry in save_data:
            json_str = json.dumps(entry, ensure_ascii=False)
            f.write(json_str + '\n')  
    print(f"合并完成！文件已保存至: {args.output_path}")

if __name__ == "__main__":
    main()