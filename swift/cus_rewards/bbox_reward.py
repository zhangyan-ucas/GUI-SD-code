"""
BBox Reward: 参考 GUI-G1 (arXiv 2505.15810) 的 bbox reward 设计。

GUI-G1 使用三个 reward 的线性组合:
  R_total = R_hit + α * R_iou + β * R_box

  - R_hit:  预测中心点是否落在 GT bbox 内（0/1）
  - R_iou:  预测 bbox 与 GT bbox 的 IoU
  - R_box:  预测 bbox 四个坐标与 GT bbox 的归一化距离约束
            R_box = 4 / (d_x1 + d_x2 + d_y1 + d_y2)
            其中 d_xi = 1 / (1 - |pred_xi - gt_xi| / image_dim)

默认超参:  α=0.25, β=0.125

注意: 我们的 grounding 任务模型输出的是中心点坐标（不是 bbox），
所以我们将预测点构造为一个点 bbox (pred_x, pred_y, pred_x, pred_y)，
然后与 GT bbox 计算 IoU 和 Box reward。
由于点 bbox 面积为 0，IoU 为 0，所以实际只有 R_hit 和 R_box 有效。

因此我们做如下适配:
  - R_hit:  预测点是否在 GT bbox 内（同原始 ground-acc）
  - R_iou:  改为基于距离的 soft reward（高斯衰减，复用 g2_point_reward）
  - R_box:  基于预测点到 GT bbox 中心的归一化距离约束
"""

from typing import Dict, List, Union
import re
import json
import math

from swift.custom_utils.format_func import extract_action
from swift.custom_utils.ground_func import (
    pointnorm2real,
    calculate_pred_norm_point,
    bboxreal2norm,
)


class ORM:
    """Base class for synchronous outcome reward models (ORM)."""

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class BBoxReward(ORM):
    """
    GUI-G1 style bbox reward: R = R_hit + α * R_dist + β * R_box

    R_hit:  预测点在 GT bbox 内 → 1.0，否则 → 0.0
    R_dist: 基于预测点到 GT 中心的高斯距离衰减 (soft reward, 0~1)
    R_box:  坐标距离约束 R_box = 2 / (d_x + d_y)
            d_x = 1 / (1 - |pred_x - gt_cx| / img_w)
            d_y = 1 / (1 - |pred_y - gt_cy| / img_h)
            预测越接近 GT 中心，R_box 越大（最大=1.0）
    """

    def __init__(self, alpha=0.25, beta=0.125):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, completions, solution, additional_paras, **kwargs) -> List[float]:
        rewards = []
        for predict_str, ground_truth, para in zip(completions, solution, additional_paras):
            if isinstance(para, str):
                para = json.loads(para)
            image_size = para['image_size']
            reward = self.compute_reward(predict_str, ground_truth, image_size)
            rewards.append(reward)
        return rewards

    def compute_reward(self, predict_str: str, ground_truth: dict, image_size: list) -> float:
        try:
            pred_action = extract_action(predict_str)
            if pred_action == "no action":
                return 0.0

            # 预测归一化坐标
            pred_point = calculate_pred_norm_point(
                image_size, pred_action['arguments']['coordinate'], "qwen3vl"
            )
            pred_x, pred_y = pred_point

            # GT bbox 归一化到 0-1000
            gt_bbox = bboxreal2norm(ground_truth['arguments']['coordinate'], image_size)
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
            gt_cx = (gt_x1 + gt_x2) / 2
            gt_cy = (gt_y1 + gt_y2) / 2

            # ====== R_hit: 点在 bbox 内 ======
            r_hit = 1.0 if (gt_x1 <= pred_x <= gt_x2 and gt_y1 <= pred_y <= gt_y2) else 0.0

            # ====== R_dist: 高斯距离衰减 ======
            gt_w = max(gt_x2 - gt_x1, 1)
            gt_h = max(gt_y2 - gt_y1, 1)
            sigma_x = 0.5 * gt_w
            sigma_y = 0.5 * gt_h
            x_term = (pred_x - gt_cx) ** 2 / (sigma_x ** 2)
            y_term = (pred_y - gt_cy) ** 2 / (sigma_y ** 2)
            r_dist = math.exp(-0.5 * (x_term + y_term))

            # ====== R_box: 坐标距离约束 ======
            # 归一化距离到 [0, 1]（除以 1000，因为坐标归一化到 0-1000）
            dx = abs(pred_x - gt_cx) / 1000.0
            dy = abs(pred_y - gt_cy) / 1000.0
            # 防止除零
            d_x = 1.0 / max(1.0 - dx, 1e-6)
            d_y = 1.0 / max(1.0 - dy, 1e-6)
            r_box = 2.0 / (d_x + d_y)

            # ====== 总 reward ======
            reward = r_hit + self.alpha * r_dist + self.beta * r_box
            return round(reward, 4)

        except Exception:
            return 0.0


class BBoxFormat(ORM):
    """格式检查 reward"""

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for predict_str, ground_truth in zip(completions, solution):
            action = extract_action(predict_str)
            rewards.append(1.0 if action != "no action" else 0.0)
        return rewards
