import json
import os 
from glob import glob 

def element2swift(image_dir, category, sample):

    image_path = os.path.join(image_dir, sample['image_path'])
    instruction = sample["prompt_to_evaluate"]

    ret = {
        "images": [image_path], 
        "instruction": instruction,
        "bbox": [int(i) for i in sample['bbox']],
        "additional_paras": {
            "platform": sample['platform'],
            "element_type": sample['element_type'],
            "category":sample['category'], 
            "image_size": sample['image_size'],
            "split": category
        },
    }
    return ret 




def uivision_builer():
    image_dir = "/mnt/vlm-ks3/zhangyan/datasets/benchmark/ui-vision/images"
    anno_paths = {
        "basic": '/mnt/vlm-ks3/zhangyan/datasets/benchmark/ui-vision/annotations/element_grounding/element_grounding_basic.json',
        "functional": '/mnt/vlm-ks3/zhangyan/datasets/benchmark/ui-vision/annotations/element_grounding/element_grounding_functional.json',
        "spatial": '/mnt/vlm-ks3/zhangyan/datasets/benchmark/ui-vision/annotations/element_grounding/element_grounding_spatial.json',
    }
    ret_samples = []

    for category, anno_path in anno_paths.items():
        with open(anno_path, 'r', encoding="utf-8") as f:
            total_data = json.load(f)

        for sample in total_data:
            ret = element2swift(image_dir, category, sample)
            ret_samples.append(ret)

    return ret_samples

if __name__ == "__main__":
    uivision_builer()
