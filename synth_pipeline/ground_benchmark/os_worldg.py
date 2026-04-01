import json
import os 
from custom_utils.vis_image import vis_func 

def convert2swift(image_dir, sample):
    """
        只考虑了bbox_type = bbox的情况
    """

    image_path = os.path.join(image_dir, sample['image_path'])
    
    if sample['box_type'] != 'bbox':
        return None

    boxes_coordinate = sample["box_coordinates"][:2]
    boxes_size = sample["box_coordinates"][2:]
    boxes_coordinate = [
        boxes_coordinate[0],
        boxes_coordinate[1],
        boxes_coordinate[0] + boxes_size[0],
        boxes_coordinate[1] + boxes_size[1],
    ]

    ret = {
        "images": [image_path], 
        "instruction": sample['instruction'],
        "bbox": [int(i) for i in boxes_coordinate],
        "additional_paras": {
            "id": sample['id'],
            "GUI_types": sample['GUI_types'],
            "image_size": sample['image_size'],
        },
    }

    # vis_img = vis_func(
    #     org_image=ret['images'][0], 
    #     user_text=ret['instruction'], 
    #     pred=[0,0], 
    #     gt=ret['bbox'], 
    #     padding=10, line_spacing=8
    # )
    # vis_img.save(f"/mnt/vlm-ks3/zhangyan/datasets/vis/osworldg/{os.path.basename(ret['images'][0])}")

    return ret 

def osworldg_builer():
    image_dir = "/mnt/vlm-ks3/zhangyan/datasets/benchmark/OSWorld-G/benchmark/images"
    anno_path = "/mnt/vlm-ks3/zhangyan/datasets/benchmark/OSWorld-G/benchmark/OSWorld-G_refined.json"
    ret_samples = []

    with open(anno_path, 'r', encoding="utf-8") as f:
        total_data = json.load(f)

    for sample in total_data:
        ret = convert2swift(image_dir, sample)
        if ret:
            ret_samples.append(ret)
    
    return ret_samples


if __name__ == "__main__":
    osworldg_builer()

a = 1
