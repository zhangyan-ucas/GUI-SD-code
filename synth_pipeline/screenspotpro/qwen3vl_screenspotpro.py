import json
from tqdm import tqdm 
from multiprocessing import Pool
import os 
import re
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import io
import random
from glob import glob 



def qwen3vl_builder(image_dir, sample):
    # image 
    image_path = os.path.join(image_dir, sample['img_filename'])
    assert os.path.exists(image_path)
    img = Image.open(image_path)

    # anno 
    bbox = sample['bbox']
    action = {'name': 'mobile_use', 'arguments': {'action': 'click', 'coordinate': bbox}}

    # sample write to jsonl 
    with open("/mnt/vlm-ks3/zhangyan/cvpr_code/ms-swift-qwen3vl/synth_pipeline/prompts/qwen3vl_ground/report_mod.txt") as f:
        sys_prompt = f.read()
    
    # qwen3vl 
    user_text = sample['instruction']
    
    ret = {
        "images": [image_path], 
        "solution": action,   
        "messages":[
            {"content": sys_prompt, "role": "system"},
            {"content": user_text, "role": "user"},
        ], 
        "additional_paras": {
            "group": sample['group'], 
            "ui_type": sample['ui_type'],
            "image_size": img.size
        },
        "sample_id": sample['id']
    }

    return ret


def main():
    # init 
    anno_path = "/vlm-ssd/FoundationModel/dataset/ScreenSpot-Pro/annotations.json"
    image_dir =  "/vlm-ssd/FoundationModel/dataset/ScreenSpot-Pro/images"
    save_path = "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/screenspotpro/qwen3vl.jsonl"

    with open(anno_path, 'r') as f:
        data = json.load(f)

    # ---- 串行执行（替代多进程）----
    with open(save_path, 'w', encoding='utf-8') as f:
        for sample in data:
            ret = qwen3vl_builder(image_dir, sample)
            f.write(json.dumps(ret, ensure_ascii=False) + '\n')
            a = 1

if __name__ == "__main__":
    main()