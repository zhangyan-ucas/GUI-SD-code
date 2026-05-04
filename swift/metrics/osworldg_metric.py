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
import os 
from torchvision.ops.boxes import box_area
import torch
import json
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


# ============ OSWorld-G 官方评测逻辑 (参考 GroundingEval) ============

def _is_point_in_rectangle(point, rect):
    """判断点是否在矩形内 rect=[x1, y1, x2, y2]"""
    return rect[0] <= point[0] <= rect[2] and rect[1] <= point[1] <= rect[3]


def _is_point_in_polygon(point, polygon):
    """射线法判断点是否在多边形内, polygon 为 [x0,y0,x1,y1,...] 的扁平列表"""
    x, y = point
    n = len(polygon) // 2
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i * 2], polygon[i * 2 + 1]
        xj, yj = polygon[j * 2], polygon[j * 2 + 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def osworldg_eval(pred_point, box_type, box_coordinates, image_size):
    """
    参考 OSWorld-G 官方 GroundingEval._eval 的评测逻辑。
    
    Args:
        pred_point: 预测的绝对坐标点 [x, y] (像素坐标)
        box_type: "bbox" | "polygon" | "refusal"
        box_coordinates: 
            - bbox: [x, y, w, h]
            - polygon: [x0,y0,x1,y1,...]  扁平列表
            - refusal: 任意
        image_size: [width, height]
    
    Returns:
        bool: 是否命中
    """
    center_point = pred_point  # 已经是 [x, y] 坐标

    if box_type == "bbox":
        # box_coordinates 格式: [x, y, w, h] -> 转换为 [x1, y1, x2, y2]
        gt_rect = [
            box_coordinates[0],
            box_coordinates[1],
            box_coordinates[0] + box_coordinates[2],
            box_coordinates[1] + box_coordinates[3],
        ]
        return _is_point_in_rectangle(center_point, gt_rect)
    
    elif box_type == "polygon":
        return _is_point_in_polygon(center_point, box_coordinates)
    
    elif box_type == "refusal":
        # refusal: 预测点坐标都为负数才算正确（即模型拒绝输出有效坐标）
        return all(center_point[i] < 0 for i in range(2))
    
    else:
        print(f"WARNING: unknown box_type '{box_type}', treated as wrong.")
        return False


# ============ 主评测函数 ============

def compute_osworldg(data_list, metric):
    """
    OSWorld-G 评测指标计算。
    
    数据中 additional_paras 包含:
        - id, GUI_types, image_size, box_type, category
    数据中 solution 包含:
        - {'name': 'mobile_use', 'arguments': {'action': 'click', 'coordinate': [bbox]}}
        - 其中 coordinate 即原始的 box_coordinates
    """
    corr_action = 0
    num_wrong_format = 0
    num_action = len(data_list)
    
    # 按 category / GUI_types / box_type 分组统计
    category_eval = defaultdict(list)
    gui_type_eval = defaultdict(list)
    box_type_eval = defaultdict(list)

    for data in tqdm(data_list, total=len(data_list)):
        label, pred = data['solution'], data['response']
        para = json.loads(data['additional_paras'])
        
        box_type = para['box_type']
        image_size = para['image_size']
        category = para.get('category', 'unknown')
        gui_types = para.get('GUI_types', ['unknown'])
        if isinstance(gui_types, str):
            gui_types = [gui_types]
        
        # box_coordinates 从原始数据中取（存在 solution 的 coordinate 中，但那个已经被 ground_benchmark 转成了 xyxy bbox）
        # 这里需要从原始标注中还原:
        #   - bbox 原始为 [x, y, w, h], 但 os_worldg.py 中 convert2swift 把 bbox 转成了 [x1,y1,x2,y2]
        #   - polygon 原始为扁平列表 [x0,y0,...], convert2swift 中直接存了原始值
        #   - refusal 也直接存了原始值
        # solution['arguments']['coordinate'] 存的是转换后的 bbox (xyxy) 或原始 polygon/refusal
        gt_coordinate = label['arguments']['coordinate']

        # ====== 解析预测 ======
        if 'ground_qwen3' in metric:
            pred_action = extract_ground(pred)
        elif 'navi_qwen3' in metric:
            pred_action = extract_action(pred)
        elif 'venus' in metric:
            try:
                pred_action = eval(pred)
            except:
                pred_action = [0, 0]
        else:
            raise NotImplementedError

        # ====== 处理格式错误 ======
        if pred_action == "no action":
            num_wrong_format += 1
            # 格式错误统一判为错误（不能将格式错误默认视为 refusal 正确）
            is_correct = False
        else:
            # ====== 计算预测的像素坐标 ======
            parse_failed = False
            try:
                if 'ground' in metric:
                    pred_point = calculate_pred_norm_point(image_size, pred_action, "qwen3vl")
                    pred_point = pointnorm2real(pred_point, image_size)
                elif 'navi_qwen3' in metric:
                    pred_point = calculate_pred_norm_point(image_size, pred_action['arguments']['coordinate'], "qwen3vl")
                    pred_point = pointnorm2real(pred_point, image_size)
                elif 'venus' in metric:
                    pred_point = calculate_pred_norm_point(image_size, pred_action, "qwen3vl")
                    pred_point = pointnorm2real(pred_point, image_size)
                else:
                    pred_point = [0, 0]
                    parse_failed = True
            except:
                pred_point = [0, 0]
                parse_failed = True

            if parse_failed:
                # 坐标解析失败，直接判错
                is_correct = False
            else:
                # ====== 按 box_type 调用官方评测逻辑 ======
                if box_type == "bbox":
                    # gt_coordinate 被 convert2swift 转成了 [x1, y1, x2, y2]
                    # 直接用 _is_point_in_rectangle 判定，无需再做 xywh 转换
                    is_correct = _is_point_in_rectangle(pred_point, gt_coordinate)
                
                elif box_type == "polygon":
                    is_correct = _is_point_in_polygon(pred_point, gt_coordinate)
                
                elif box_type == "refusal":
                    # 模型产生了有效坐标但应该 refusal → 判为错
                    # 官方逻辑：预测坐标都为负才算正确
                    is_correct = all(pred_point[i] < 0 for i in range(2))
                
                else:
                    is_correct = False

        # ====== 统计 ======
        if is_correct:
            corr_action += 1

        result_val = 1 if is_correct else 0
        category_eval[category].append(result_val)
        for gt in gui_types:
            gui_type_eval[gt].append(result_val)
        box_type_eval[box_type].append(result_val)

    # ====== 按 category 计算准确率 ======
    print("\n=== Category-wise Accuracy ===")
    category_metrics = {}
    
    # 按照指定顺序打印类别
    ordered_categories = [
        "Text Matching",
        "Element Recognition", 
        "Layout Understanding",
        "Fine-grained-Manipulation"
    ]
    
    for cat in ordered_categories:
        if cat in category_eval:
            results = category_eval[cat]
            acc = sum(results) / len(results) * 100 if results else 0.0
            category_metrics[cat] = f"{acc:.2f}"
            print(f"  {cat}: {acc:.2f}% ({sum(results)}/{len(results)})")
    
    # 打印其他未在指定顺序中的类别（按字母顺序）
    other_categories = [cat for cat in category_eval.keys() if cat not in ordered_categories]
    if other_categories:
        for cat in sorted(other_categories):
            results = category_eval[cat]
            acc = sum(results) / len(results) * 100 if results else 0.0
            category_metrics[cat] = f"{acc:.2f}"
            print(f"  {cat}: {acc:.2f}% ({sum(results)}/{len(results)})")

    # ====== 按 GUI_types 计算准确率 ======
    print("\n=== GUI-type-wise Accuracy ===")
    gui_metrics = {}
    for gt, results in sorted(gui_type_eval.items()):
        acc = sum(results) / len(results) * 100 if results else 0.0
        gui_metrics[gt] = f"{acc:.2f}"
        print(f"  {gt}: {acc:.2f}% ({sum(results)}/{len(results)})")

    # ====== 按 box_type 计算准确率 ======
    print("\n=== Box-type-wise Accuracy ===")
    box_metrics = {}
    for bt, results in sorted(box_type_eval.items()):
        acc = sum(results) / len(results) * 100 if results else 0.0
        box_metrics[bt] = f"{acc:.2f}"
        print(f"  {bt}: {acc:.2f}% ({sum(results)}/{len(results)})")

    # ====== 总体指标 ======
    overall = {
        "Total Acc": f"{corr_action / num_action * 100:.2f}" if num_action > 0 else "0.00",
        "Total num": str(num_action),
        "Correct num": str(corr_action),
        "Wrong format num": str(num_wrong_format),
    }

    print("\n=== Overall ===")
    print(overall)

    return overall


