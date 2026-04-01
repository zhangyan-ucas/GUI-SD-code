from PIL import Image, ImageDraw, ImageFont
import re
import json
import os
from tqdm import tqdm 

"""
    用的32b模型和gt都正确的数据，
    2b模型第一次推理错误，第二次推理正确
"""

from swift.custom_utils.ground_func import (
    pointnorm2real, 
    pointreal2norm, 
    bboxnorm2real, 
    bboxreal2norm, 
)

from custom_utils.vis_image import vis_func 

def extract_action(content):
    try:
        output_text = eval(content)
        return output_text
    except:
        return "no action"

def point_in_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2




def sft_sample(ret_data, idx):
    instruction = ret_data['instruction']
    user_query = f"<image>Output the center point of the position corresponding to the following instruction: \n{instruction}. \n\nThe output should just be the coordinates of a point, in the format [x,y]. Additionally, if the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1]."

    # supervised signal for qwen3vl 
    norm_bbox = bboxreal2norm(ret_data['gt_bbox'], ret_data['image_size'])
    norm_point = [sum(norm_bbox[::2]) // 2, sum(norm_bbox[1::2]) // 2]

    messages = [
        {"content": user_query, "role": "user"},
        {"content": str(norm_point), "role": "assistant"},
    ]
    ret = {
        "images": [ret_data['image_path']], 
        "messages": messages,  
        "additional_paras": {
            "idx": idx,
            "gt_bbox": ret_data['gt_bbox'],
            "instruction": ret_data['instruction'],
            "venus1.5-2b-pred-infer1": ret_data['pred']['0']['global_point'],
            "venus1.5-2b-pred-infer2": ret_data['pred']['1']['global_point'],
            "venus1.5-32b-pred": ret_data['pred']['32b'],
            "image_size": ret_data['image_size']
        },
    }
    return ret 



output_path = "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/atlas/temp.jsonl"
anno_path1 = "/mnt/vlm-ks3/zhangyan/cvpr_code/ms-swift-qwen3vl/useful_output/qwen3vl/32b_dual_atlas.jsonl"
anno_path2 = "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/atlas/venus15/2b_2infer_mask_wgt.jsonl"


# 辅助函数：提取用于唯一标识一条数据的 key (使用 tuple 以保证可哈希)
def get_unique_key(d):
    return (d.get('image_path'), str(d.get('gt_bbox')), d.get('instruction'))

# 1. 收集 anno_path1 中的有效标识 (使用 set 提高检索效率)
valid_keys = set()
pred_bboxes = {}
with open(anno_path1, 'r', encoding='utf-8') as f1:
    for line in f1:
        ret = json.loads(line)
        pred_bboxes[get_unique_key(ret)] = ret['pred']['0']['global_point']
        valid_keys.add(get_unique_key(ret))

# 2. 过滤 anno_path2 的数据
filter_data = []
with open(anno_path2, 'r', encoding='utf-8') as f2:
    # 你的注释提到了 tqdm，这里应用在读取 f2 时展示进度
    for line in tqdm(f2, desc="Filtering data"):
        ret = json.loads(line)
        
        # 安全获取预测点和标注框，避免使用裸 except
        try:
            pred1 = ret['pred']['0']['global_point']
            pred2 = ret['pred']['1']['global_point']
            gt_bbox = ret['gt_bbox']
        except KeyError:
            continue  # 如果缺少这些字段，直接跳过当前行
        
        # 核心判断逻辑
        if (not point_in_bbox(pred1, gt_bbox) and 
            point_in_bbox(pred2, gt_bbox) and 
            get_unique_key(ret) in valid_keys):
            ret['pred']['32b'] = pred_bboxes[get_unique_key(ret)]
            filter_data.append(ret)

 

# 同时打开输入文件和输出文件
with open(output_path, 'w', encoding='utf-8') as f_out:
    for idx, sample in enumerate(filter_data):
        swift_sample = sft_sample(sample, idx)
        vis_img = vis_func(
            org_image=swift_sample['images'][0], 
            user_text=swift_sample['additional_paras']['instruction'], 
            pred=swift_sample['additional_paras']['venus1.5-32b-pred'], 
            gt=swift_sample['additional_paras']['gt_bbox'], 
            padding=10, line_spacing=8
        )
        vis_img.save(f"/mnt/vlm-ks3/zhangyan/datasets/otlas/{os.path.basename(swift_sample['images'][0])}")
        f_out.write(json.dumps(swift_sample, ensure_ascii=False) + '\n')

        if idx == 100: break 
    print(f"the file saved in {output_path}")

