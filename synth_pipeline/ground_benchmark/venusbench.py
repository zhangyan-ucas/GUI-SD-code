import json
import os 
from glob import glob 
from PIL import Image
from custom_utils.vis_image import vis_func

def convert2swift(image_dir, sample):
    image_path = os.path.join(image_dir, sample['img_filename'])
    
    if 'bbox' not in sample:
        return None

    ret = {
        "images": [image_path], 
        "instruction": sample['instruction'],
        "bbox": sample['bbox'],
        "additional_paras": {
            "category":sample['category'],
            "image_size": Image.open(image_path).size
        },
    }
    # vis_img = vis_func(
    #     org_image=ret['images'][0], 
    #     user_text=ret['instruction'], 
    #     pred=[0,0], 
    #     gt=ret['bbox'], 
    #     padding=10, line_spacing=8
    # )
    # vis_img.save(f"/mnt/vlm-ks3/zhangyan/datasets/vis/venusbench/{os.path.basename(ret['images'][0])}")

    return ret 

def VenusBench_builer():
    image_dir = "/mnt/vlm-ks3/zhangyan/datasets/benchmark/VenusBench-GD/images"
    anno_paths = glob("/mnt/vlm-ks3/zhangyan/datasets/benchmark/VenusBench-GD/instruction/*.json")
    ret_samples = []

    for anno_path in anno_paths:
        with open(anno_path, 'r', encoding="utf-8") as f:
            total_data = json.load(f)

        for sample in total_data:
            ret = convert2swift(image_dir, sample)
            if ret:
                ret_samples.append(ret)
    
    return ret_samples

