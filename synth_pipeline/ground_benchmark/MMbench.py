import json
import os 
from custom_utils.vis_image import vis_func

def convert2swift(image_dir, sample):
    image_path = os.path.join(image_dir, sample['platform'], sample['image_path'])
    
    bbox = [
        int(sample['bbox'][0] * sample['image_size'][0]),
        int(sample['bbox'][1] * sample['image_size'][1]),
        int(sample['bbox'][2] * sample['image_size'][0]),
        int(sample['bbox'][3] * sample['image_size'][1]), 
    ]

    ret = {
        "images": [image_path], 
        "instruction": sample['instruction'],
        "bbox": bbox,
        "additional_paras": {
            "id": sample['index'],
            "app_name": sample['app_name'],
            "image_size": sample['image_size'],
            "data_type": sample['data_type'],
            "platform":sample['platform']
        },
    }

    vis_img = vis_func(
        org_image=ret['images'][0], 
        user_text=ret['instruction'], 
        pred=[0,0], 
        gt=ret['bbox'], 
        padding=10, line_spacing=8
    )
    vis_img.save(f"/mnt/vlm-ks3/zhangyan/datasets/vis/mmbench/{os.path.basename(ret['images'][0])}")


    return ret 

def mmbench_builer():
    image_dir = "/mnt/vlm-ks3/zhangyan/datasets/benchmark/MMBench-GUI/offline_images"
    anno_path = "/mnt/vlm-ks3/zhangyan/datasets/benchmark/MMBench-GUI/L2_annotations.json"
    ret_samples = []

    with open(anno_path, 'r', encoding="utf-8") as f:
        total_data = json.load(f)

    for sample in total_data:
        ret = convert2swift(image_dir, sample)
        ret_samples.append(ret)
    
    return ret_samples

if __name__ == "__main__":
    mmbench_builer()
