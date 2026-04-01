import json
import os
from tqdm import tqdm 
import re
from custom_utils.vis_image import vis_func
from swift.custom_utils.ground_func import bboxnorm2real
from synth_pipeline.scalecua.scalecua_builder import (
    sample2files,
    messages_builder,
    convert_message
)


def main():
    # init 
    save_path = "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/250328/250328_sc.jsonl"

    with open('/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/meta.json', 'r') as f:
        meta_data = json.load(f)

    total_samples = []

    # Process each data source defined in meta.json
    for item in tqdm(meta_data.values(), desc="Processing meta.json sources"):
        root_path = os.path.join("/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/", item['root'])
        annotation_path = os.path.join("/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/", item['annotation'])
        

        anno2platform = {
             "/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/annotations/data_20250328_windows_internvl_grounding_20250425_20250722.jsonl" : "windows",
            # "/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/annotations/data_20250328_web_25k_internvl_grounding_20250409.jsonl": "web",             
        }
        if annotation_path not in anno2platform: continue

        # if annotation_path not in  [
        #             # point anno , 不选择
        #             # "/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/annotations/data_20250328_android_filter_action_grounding_20250405_202507011.jsonl",
                    
        #             # point anno   , 不选择
        #             # "/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/annotations/data_20250328_windows_action_grounding_20250409_202507011_20250722.jsonl",
                    
        #             # point anno   , 不选择
        #             # "/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/annotations/data_20250328_web_25k_action_grounding_20250404_202507011.jsonl",
                    
        #             # # bbox anno  , 选择
        #             # "/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/annotations/data_20250328_web_25k_internvl_grounding_20250409.jsonl",
                    
        #             # # bbox anno, android图像有问题, 不选择
        #             # "/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/annotations/data_20250328_android_internvl_grounding_20250409.jsonl",

        #              # # bbox anno 
        #             # "/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/annotations/data_20250328_windows_internvl_grounding_20250425_20250722.jsonl"
        #         ]:continue

        # if "grounding" not in annotation_path or "250328" not in annotation_path : continue

        with open(annotation_path, 'r') as f:
            # 核心修改：使用 readlines() 先读取所有行，让 tqdm 知道总数，从而显示具体剩余百分比
            lines = f.readlines()
            for idx, line in enumerate(tqdm(lines, desc=f"Reading {annotation_path}")):
               
                

                # Load a single JSON data sample
                sample = json.loads(line.strip())
                sample['instruction'], sample['gt']  = convert_message(sample)


                # Construct the full path to the image
                image_path = os.path.join(root_path, sample['image'])
                assert os.path.exists(image_path)
                sample['images'] = image_path
                sample['platform'] = anno2platform[annotation_path]
                
                # if len(sample['gt']) == 4:
                #     vis_img = vis_func(
                #         org_image=image_path, 
                #         user_text=sample['instruction'], 
                #         pred=[0,0], 
                #         gt=sample['gt'], 
                #         padding=10, line_spacing=8
                #     )
                #     vis_img.save(f"/mnt/vlm-ks3/zhangyan/datasets/vis/scalecua/{idx}_{os.path.basename((image_path))}")
                #     pass   

                total_samples.append(sample)

    sample2files(total_samples, save_path)

if __name__ == "__main__":
    main()