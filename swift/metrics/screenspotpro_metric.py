# Copyright (c) Alibaba, Inc. and its affiliates.
import collections
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Literal

import numpy as np
from transformers.trainer_utils import EvalPrediction
from collections import defaultdict
import re
from tqdm import tqdm 
from PIL import Image, ImageFont, ImageDraw
from collections import defaultdict
import os 
from torchvision.ops.boxes import box_area
import torch
from tqdm import tqdm 
import json
import re
from glob import glob 
from swift.custom_utils.format_func import (
    extract_action, 
    extract_ground
)



from swift.custom_utils.ground_func import (
    pointreal2norm,
    get_scroll_direction,
    pointnorm2real,
    calculate_pred_norm_point,
    qwen25_get_resize,
)
from swift.utils import read_from_jsonl


def compute_screenspotpro(data_list, metric):
    corr_action = 0
    total_eval = {
        "Dev": { "text": list(), "icon": list()},
        "Creative": { "text": list(), "icon": list()},
        "CAD": { "text": list(), "icon": list()},
        "Scientific": { "text": list(), "icon": list()},
        "Office": { "text": list(), "icon": list()},
        "OS": { "text": list(), "icon": list()}
    }
    

    num_wrong_format = 0
    num_action = len(data_list)
    for data in tqdm(data_list, total=len(data_list)):
        label, pred = data['solution'], data['response']
        para, image_path = json.loads(data['additional_paras']), data["images"][0]['path']
        # para, image_path = data['additional_paras'], data["images"][0]['path']

        if 'ground_qwen3' in metric:
            pred_action = extract_ground(pred)
        elif 'navi_qwen3' in metric:
            pred_action = extract_action(pred)
        elif 'venus' in metric:
            try:
                pred_action = eval(pred)
            except:
                pred_action = [0,0]
        else:
            raise NotImplementedError

        gt_action = label

        if pred_action == "no action":
            num_wrong_format += 1
            total_eval[para['group']][para['ui_type']].append(0)
        elif 'ground' in metric:
            pred_point = calculate_pred_norm_point(para['image_size'], pred_action, "qwen3vl")
            pred_point = pointnorm2real(pred_point, para['image_size'])
            gt_bbox =  gt_action['arguments']['coordinate']
        
        elif 'venus' in metric:
            pred_point = calculate_pred_norm_point(para['image_size'], pred_action, "qwen3vl")
            pred_point = pointnorm2real(pred_point, para['image_size'])
            gt_bbox =  gt_action['arguments']['coordinate']
            if gt_bbox[0] < pred_point[0] < gt_bbox[2] and gt_bbox[1] < pred_point[1] < gt_bbox[3]:
                corr_action += 1
                total_eval[para['group']][para['ui_type']].append(1)
            else:
                total_eval[para['group']][para['ui_type']].append(0)

        elif 'navi_qwen3' in metric:
            try:
                pred_point = calculate_pred_norm_point(para['image_size'],  pred_action['arguments']['coordinate'], "qwen3vl")
                pred_point = pointnorm2real(pred_point, para['image_size'])
            except:
                pred_point = [0,0]

            gt_bbox =  gt_action['arguments']['coordinate']

            if gt_bbox[0] < pred_point[0] < gt_bbox[2] and gt_bbox[1] < pred_point[1] < gt_bbox[3]:
                corr_action += 1
                total_eval[para['group']][para['ui_type']].append(1)

            else:
                total_eval[para['group']][para['ui_type']].append(0)
    
    # === 计算每个 group 的 text/icon 准确率 ===
    metrics = {}
    for group, types in total_eval.items():
        group_metrics = {}
        for ui_type, results in types.items():
            if len(results) == 0:
                acc = 0.0
            else:
                acc = sum(results) / len(results) * 100
            group_metrics[ui_type] = f"{acc:.2f}"
        metrics[group] = group_metrics

    # === 计算总体指标 ===
    all_text = [x for g in total_eval.values() for x in g["text"]]
    all_icon = [x for g in total_eval.values() for x in g["icon"]]

    overall = {
        "text": f"{sum(all_text)/len(all_text)*100:.2f}" if all_text else "0.00",
        "icon": f"{sum(all_icon)/len(all_icon)*100:.2f}" if all_icon else "0.00",
        "Total Acc": f"{corr_action / num_action * 100:.2f}",
        "Total num": str(num_action),
        "Wrong format num": str(num_wrong_format),
    }

    print("\n=== Group-wise Accuracy ===")
    for group, vals in metrics.items():
        print(f"{group}: text={vals['text']}%, icon={vals['icon']}%")

    print("\n=== Overall ===")
    print(overall)


    return overall

def get_latest_vdir(run_name):
    base_dir = f"/mnt/vlm-ks3/zhangyan/cvpr_code/ms-swift-qwen3vl/output/{run_name}"
    # 列出所有以 v 开头的目录
    v_dirs = [d for d in os.listdir(base_dir) if d.startswith("v") and os.path.isdir(os.path.join(base_dir, d))]
    if not v_dirs:
        return None

    # 排序：v 后面的数字可能两位三位，所以按目录名整体排序
    v_dirs_sorted = sorted(v_dirs, key=lambda x: int(x.split("-")[0][1:]))  
    latest = v_dirs_sorted[-1]

    return os.path.join(base_dir, latest) + "/checkpoint-*"

if __name__ == "__main__":
    # 1. single ckpt 
    # # # zero shot
    # jsonl_path = "/mnt/vlm-ks3/zhangyan/nips_code/opd/useful_output/useful_output/qwen3vl/screenspotpro/8b.jsonl"

    # # 2
    # # train multiple ckpt 
    # for ckpt_num in range(5, 70, 5):
    #     jsonl_path = f"/mnt/vlm-ks3/zhangyan/nips_code/opd/output/v51_opsd/v1-20260331-152348/checkpoint-{ckpt_num}-merged/infer_result/screenspotpro.jsonl"
    #     metric = "screenspotpro_navi_qwen3"
    #     data_list = read_from_jsonl(jsonl_path)
    #     compute_screenspotpro(data_list, metric )

    

    jsonl_path = "/mnt/vlm-ks3/zhangyan/nips_code/opd/output/v53_sft/v0-20260331-205929/checkpoint-151/infer_result/screenspotpro.jsonl"
    metric = "screenspotpro_navi_qwen3"
    data_list = read_from_jsonl(jsonl_path)
    compute_screenspotpro(data_list, metric )