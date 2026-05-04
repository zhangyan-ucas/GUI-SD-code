# Copyright (c) Alibaba, Inc. and its affiliates.
# MMBench-GUI L2 Element Grounding 评测指标
# 参考: https://github.com/open-compass/MMBench-GUI

import collections
from typing import Dict, List
from collections import defaultdict
import re
from tqdm import tqdm 
import os 
import json
import numpy as np

from swift.custom_utils.format_func import (
    extract_action, 
    extract_ground
)

from swift.custom_utils.ground_func import (
    pointreal2norm,
    pointnorm2real,
    calculate_pred_norm_point,
)
from swift.utils import read_from_jsonl


def compute_mmbench(data_list, metric):
    """
    MMBench-GUI L2 Element Grounding 评测。
    
    官方评测逻辑:
      1. 将预测点和GT bbox都归一化到 [0,1] 范围
      2. 判断预测点是否落在 GT bbox 内
      3. 按 grounding_type (basic/advanced) × platform (os_windows/os_mac/...) × data_type (icon/text) 分组统计
    
    数据中 additional_paras 包含:
        - id, app_name, image_size, data_type (icon/text), platform, grounding_type (basic/advanced)
    数据中 solution['arguments']['coordinate'] 为像素坐标的 bbox [x1, y1, x2, y2]
        (由 MMbench.py 中将归一化 bbox 乘以 image_size 得到)
    """
    corr_action = 0
    num_wrong_format = 0
    num_action = len(data_list)

    # 三层嵌套统计: stats[grounding_type][platform][data_type] = [1, 0, -1, ...]
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for data in tqdm(data_list, total=len(data_list)):
        label, pred = data['solution'], data['response']
        para = json.loads(data['additional_paras'])

        image_size = para['image_size']  # [width, height]
        data_type = para.get('data_type', 'unknown')     # icon / text
        platform = para.get('platform', 'unknown')        # os_windows / os_mac / ...
        grounding_type = para.get('grounding_type', 'basic')  # basic / advanced

        # GT bbox: 像素坐标 [x1, y1, x2, y2]
        gt_bbox = label['arguments']['coordinate']

        # 归一化 GT bbox 到 [0,1]
        gt_bbox_norm = [
            gt_bbox[0] / image_size[0],
            gt_bbox[1] / image_size[1],
            gt_bbox[2] / image_size[0],
            gt_bbox[3] / image_size[1],
        ]

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
            stats[grounding_type][platform][data_type].append(-1)
            continue

        # ====== 计算预测的归一化坐标 ======
        parse_failed = False
        try:
            if 'ground' in metric:
                pred_norm = calculate_pred_norm_point(image_size, pred_action, "qwen3vl")
            elif 'navi_qwen3' in metric:
                pred_norm = calculate_pred_norm_point(image_size, pred_action['arguments']['coordinate'], "qwen3vl")
            elif 'venus' in metric:
                pred_norm = calculate_pred_norm_point(image_size, pred_action, "qwen3vl")
            else:
                pred_norm = (0, 0)
                parse_failed = True
        except:
            pred_norm = (0, 0)
            parse_failed = True

        if parse_failed:
            stats[grounding_type][platform][data_type].append(-1)
            num_wrong_format += 1
            continue

        # 归一化预测点到 [0,1] (qwen3vl 输出是 0-1000 范围，需要除以 1000)
        pred_x = pred_norm[0] / 1000.0
        pred_y = pred_norm[1] / 1000.0

        # ====== 点击判定: 预测点是否落在 GT bbox 内 ======
        match = (gt_bbox_norm[0] <= pred_x <= gt_bbox_norm[2]) and \
                (gt_bbox_norm[1] <= pred_y <= gt_bbox_norm[3])

        if match:
            corr_action += 1
            stats[grounding_type][platform][data_type].append(1)
        else:
            stats[grounding_type][platform][data_type].append(0)

    # ====== 按 MMBench-GUI 官方格式计算分数 ======
    final_score_dict = {}

    for level, level_value in stats.items():
        final_score_dict[level] = {}

        for platform, platform_value in level_value.items():
            level_platform_total_num = sum(len(platform_value[t]) for t in platform_value)

            icon_results = platform_value.get("icon", [])
            text_results = platform_value.get("text", [])

            icon_num = len(icon_results)
            text_num = len(text_results)

            icon_correct = sum(1 for x in icon_results if x == 1)
            text_correct = sum(1 for x in text_results if x == 1)

            icon_wrong_fmt = sum(1 for x in icon_results if x == -1)
            text_wrong_fmt = sum(1 for x in text_results if x == -1)

            icon_acc = icon_correct / icon_num * 100 if icon_num > 0 else 0.0
            text_acc = text_correct / text_num * 100 if text_num > 0 else 0.0
            total_acc = (icon_correct + text_correct) / level_platform_total_num * 100 if level_platform_total_num > 0 else 0.0

            final_score_dict[level][platform] = {
                "Total num": level_platform_total_num,
                "Icon num": icon_num,
                "Text num": text_num,
                "Total accuracy": f"{total_acc:.2f}",
                "Icon accuracy": f"{icon_acc:.2f}",
                "Text accuracy": f"{text_acc:.2f}",
                "Correct num": icon_correct + text_correct,
                "Error format num": icon_wrong_fmt + text_wrong_fmt,
            }

    # ====== 计算 Basic / Advanced 加权平均 ======
    summary = {}
    for level in ["basic", "advanced"]:
        if level not in final_score_dict:
            summary[f"{level} accuracy"] = "0.00"
            continue
        nums = [v["Total num"] for v in final_score_dict[level].values()]
        accs = [float(v["Total accuracy"]) for v in final_score_dict[level].values()]
        total = sum(nums)
        if total > 0:
            weighted = sum(n * a for n, a in zip(nums, accs)) / total
        else:
            weighted = 0.0
        summary[f"{level} accuracy"] = f"{weighted:.2f}"

    basic_acc = float(summary.get("basic accuracy", 0))
    advanced_acc = float(summary.get("advanced accuracy", 0))
    summary["Average accuracy"] = f"{(basic_acc + advanced_acc) / 2:.2f}"

    # ====== 打印结果 ======
    # 按指定平台顺序打印，每个平台分别显示 basic 和 advanced
    PLATFORM_ORDER = [
        ("os_windows", "Windows"),
        ("os_mac",     "MacOS"),
        ("os_linux",   "Linux"),
        ("os_ios",     "iOS"),
        ("os_android", "Android"),
        ("os_web",     "Web"),
    ]

    print("\n" + "=" * 80)
    print("MMBench-GUI L2 Element Grounding Results")
    print("=" * 80)

    for platform_key, platform_name in PLATFORM_ORDER:
        print(f"\n--- {platform_name} ---")
        for level in ["basic", "advanced"]:
            if level in final_score_dict and platform_key in final_score_dict[level]:
                vals = final_score_dict[level][platform_key]
                print(f"  {level:<10}: total={vals['Total accuracy']}% "
                      f"(icon={vals['Icon accuracy']}%, text={vals['Text accuracy']}%) "
                      f"[{vals['Correct num']}/{vals['Total num']}]")
            else:
                print(f"  {level:<10}: N/A")

    print(f"\n--- SUMMARY ---")
    print(f"  Basic accuracy:    {summary['basic accuracy']}%")
    print(f"  Advanced accuracy: {summary['advanced accuracy']}%")
    print(f"  Average accuracy:  {summary['Average accuracy']}%")
    print(f"  Total num: {num_action}, Wrong format: {num_wrong_format}")

    overall = {
        "Average Acc": summary["Average accuracy"],
        "Basic Acc": summary["basic accuracy"],
        "Advanced Acc": summary["advanced accuracy"],
        "Total num": str(num_action),
        "Wrong format num": str(num_wrong_format),
    }
    overall.update(final_score_dict)

    return overall



