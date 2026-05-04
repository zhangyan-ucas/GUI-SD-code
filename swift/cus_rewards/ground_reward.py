from typing import Dict, List, Union
import re
import json 
import os

from swift.custom_utils.format_func import (
    extract_action,
    extract_ground
)


from swift.custom_utils.ground_func import (
    pointreal2norm,
    get_scroll_direction,
    pointnorm2real,
    
    calculate_pred_norm_point,
    ground_reward_func,
    ground_reward_jigsaw_func
)


class ORM:
    """Base class for synchronous outcome reward models (ORM).

    Subclasses should implement the __call__ method to compute rewards.

    Example:
        class MyReward(ORM):
            def __call__(self, completions, **kwargs) -> List[float]:
                return [1.0 if len(c) > 100 else 0.0 for c in completions]
    """

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


# For additional reward functions, refer to swift/plugin/orm.py.
class GroundAcc(ORM):
    
        
    def __call__(self, completions, solution, additional_paras, **kwargs) -> List[float]:
        rewards = []
        for predict_str, ground_truth, para in zip(completions, solution, additional_paras):
            if isinstance(para, str):
                para = json.loads(para)
            image_size = para['image_size']

            accuracy = self.ground_acc_reward(predict_str, ground_truth, image_size)
            rewards.append(accuracy) 
        return rewards


    def ground_acc_reward(self, predict_str: str, ground_truth: str, image_size) -> float:
        """
        比较 predict_str 和 ground_truth 中的动作和参数是否一致。
        """
        try:
            # 提取 ground_truth 的动作和参数
            gt_action_type = ground_truth['arguments']['action']
            
            pred_action=extract_action(predict_str)
            
            if pred_action == "no action":
                return 0.0

            # pred to norm 
            pred_point = calculate_pred_norm_point(image_size, pred_action['arguments']['coordinate'], "qwen3vl")
            
            return ground_reward_func(
                pred_point, ground_truth, image_size
            )
            
        except Exception as e:
            return 0.0




class GroundFormat(ORM):
    

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
            检查 predict_str 是否动作空间 的格式。
        """
    
        rewards = []
        for predict_str, ground_truth in zip(completions, solution):
            
            format = self.ground_format_reward(predict_str)
            rewards.append(format) 
        return rewards

    def ground_format_reward(self, predict_str: str) -> float:
        action = extract_action(predict_str)
        if action == "no action":
            return 0.0
        else:
            return 1.0 




