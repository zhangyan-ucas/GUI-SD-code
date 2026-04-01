from PIL import Image, ImageDraw, ImageFont
import re
import json
import os
from tqdm import tqdm 

def pointnorm2real(norm_point, image_size):
    width, height =  image_size
    point = [
        int(norm_point[0] / 1000 * width),
        int(norm_point[1] / 1000 * height),
    ]
    return point

def bboxnorm2real(norm_bbox, image_size):
    width, height =  image_size
    bbox = [
        int(norm_bbox[0] / 1000 * width),
        int(norm_bbox[1] / 1000 * height),
        int(norm_bbox[2] / 1000 * width),
        int(norm_bbox[3] / 1000 * height)
    ]
    return bbox

def extract_action(content):
    try:
        output_text = eval(content)
        return output_text
    except:
        return "no action"


def uivenus_sample(ret_data):
    instruction = ret_data['instruction']
    user_query = f"<image>Output the center point of the position corresponding to the following instruction: \n{instruction}. \n\nThe output should just be the coordinates of a point, in the format [x,y]. Additionally, if the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1]."

    messages = [
        {"content": user_query, "role": "user"},
    ]
    ret = {
        "images": [ret_data['image_path']], 
        "messages": messages,  
        "additional_paras": {
            "gt_bbox": ret_data['gt_bbox'],
            "instruction": ret_data['instruction'],
            "venus1.5-32b-pred": ret_data['pred']['0']['global_point'],
            "image_size": ret_data['image_size']
        },
    }
    return ret 



anno_path = "/mnt/vlm-ks3/zhangyan/cvpr_code/ms-swift-qwen3vl/useful_output/qwen3vl/32b_dual_atlas.jsonl"
output_path = "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/atlas/venus15_atlas.jsonl"

# 获取总行数用于 tqdm 进度条
with open(anno_path, 'r', encoding='utf-8') as f:
    total_lines = sum(1 for _ in f)

# 同时打开输入文件和输出文件
with open(anno_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
    # 用 tqdm 包装 enumerate(f_in)
    for i, line in tqdm(enumerate(f_in), total=total_lines, desc="Evaluating"):
        ret = json.loads(line)
        
        # 处理得到 swift_sample 字典
        swift_sample = uivenus_sample(ret)
        
        # 将字典转换为 JSON 字符串并写入，记得加上换行符 '\n'
        # ensure_ascii=False 可以保证数据中的中文等非 ASCII 字符正常显示
        f_out.write(json.dumps(swift_sample, ensure_ascii=False) + '\n')

