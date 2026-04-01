import json
import os
from tqdm import tqdm 
import re
from custom_utils.vis_image import vis_func


def convert_message(sample):
    instruciton = sample['conversations'][0]['value'].replace("<image>", "")  

    text = sample['conversations'][1]['value'].replace("<image>", "")
    match = re.search(r'x=([0-9.]+),\s*y=([0-9.]+)', text)
    x_val = float(match.group(1)) * sample['width']
    y_val = float(match.group(2)) * sample['height']
    gt_point = [x_val, y_val]

    return instruciton, gt_point


# Load the manifest file
with open('/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/meta.json', 'r') as f:
    meta_data = json.load(f)

total_sample = 0

# Process each data source defined in meta.json
for item in tqdm(meta_data.values()):
    root_path = os.path.join("/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/", item['root'])
    annotation_path = os.path.join("/mnt/vlm-ks3/zhangyan/datasets/source_pt/ScaleCUA-Data/", item['annotation'])

    if "grounding" not in annotation_path or "250310" not in annotation_path or "windows" not in annotation_path: continue

    
    # print(f"--- Loading data from root: {root_path} ---")

    # Open the corresponding annotation file
    with open(annotation_path, 'r') as f:
        for line in f:

            # Load a single JSON data sample
            sample = json.loads(line.strip())
            instruction, gt_point = convert_message(sample)

            # Construct the full path to the image
            image_path = os.path.join(root_path, sample['image'])

            # Get the conversations and image dimensions
            conversations = sample['conversations']
            width = sample['width']
            height = sample['height']


            # vis 
            vis_img = vis_func(
                org_image=image_path, 
                user_text=instruction, 
                pred=[0,0], 
                gt=gt_point, 
                padding=10, line_spacing=8
            )
            vis_img.save(f"/mnt/vlm-ks3/zhangyan/datasets/vis/scalecua/{total_sample}_{os.path.basename((image_path))}")


            total_sample += 1

print(total_sample)