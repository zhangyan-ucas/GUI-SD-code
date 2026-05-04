# Copyright (c) Alibaba, Inc. and its affiliates.
# ScreenSpot-v2 评测指标
# 按 platform (desktop/mobile/web) × data_type (icon/text) 分组统计

from collections import defaultdict
from tqdm import tqdm
import os
import json

from swift.custom_utils.format_func import (
    extract_action,
    extract_ground
)

from swift.custom_utils.ground_func import (
    pointnorm2real,
    calculate_pred_norm_point,
)
from swift.utils import read_from_jsonl


def compute_screenspotv2(data_list, metric):
    corr_action = 0
    num_wrong_format = 0
    num_action = len(data_list)

    # 按 platform × data_type 分组: total_eval[platform][data_type] = [1, 0, ...]
    total_eval = defaultdict(lambda: defaultdict(list))

    for data in tqdm(data_list, total=len(data_list)):
        label, pred = data['solution'], data['response']
        para = json.loads(data['additional_paras'])

        image_size = para['image_size']
        platform = para.get('platform', 'unknown')      # desktop / mobile / web
        data_type = para.get('data_type', 'unknown')     # icon / text
        gt_bbox = label['arguments']['coordinate']       # [x1, y1, x2, y2]

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

        if pred_action == "no action":
            num_wrong_format += 1
            total_eval[platform][data_type].append(0)
            continue

        # ====== 计算预测像素坐标 ======
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
            num_wrong_format += 1
            total_eval[platform][data_type].append(0)
            continue

        # ====== 点击判定 ======
        if gt_bbox[0] < pred_point[0] < gt_bbox[2] and gt_bbox[1] < pred_point[1] < gt_bbox[3]:
            corr_action += 1
            total_eval[platform][data_type].append(1)
        else:
            total_eval[platform][data_type].append(0)

    # ====== 按 platform 计算准确率 ======
    print("\n=== Platform-wise Accuracy ===")
    metrics = {}
    for platform in ["mobile", "desktop", "web"]:
        types = total_eval.get(platform, {})
        platform_metrics = {}
        all_results = []
        for dt in ["text", "icon"]:
            results = types.get(dt, [])
            all_results.extend(results)
            acc = sum(results) / len(results) * 100 if results else 0.0
            platform_metrics[dt] = f"{acc:.2f}"
        overall_acc = sum(all_results) / len(all_results) * 100 if all_results else 0.0
        platform_metrics['overall'] = f"{overall_acc:.2f}"
        metrics[platform] = platform_metrics
        print(f"  {platform}: overall={platform_metrics['overall']}%, text={platform_metrics['text']}%, icon={platform_metrics['icon']}%")

    # ====== 总体指标 ======
    all_text = [x for p in total_eval.values() for x in p.get("text", [])]
    all_icon = [x for p in total_eval.values() for x in p.get("icon", [])]

    overall = {
        "text": f"{sum(all_text)/len(all_text)*100:.2f}" if all_text else "0.00",
        "icon": f"{sum(all_icon)/len(all_icon)*100:.2f}" if all_icon else "0.00",
        "Total Acc": f"{corr_action / num_action * 100:.2f}" if num_action > 0 else "0.00",
        "Total num": str(num_action),
        "Wrong format num": str(num_wrong_format),
    }

    print("\n=== Overall ===")
    print(overall)

    return overall



