import json
from tqdm import tqdm 
from multiprocessing import Pool
import os 
import re
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import io
import random
from glob import glob 



def convert2swift(image_dir, sample):
    # image 
    image_path = os.path.join(image_dir, sample['img_filename'])
   
    ret = {
        "images": [image_path], 
        "instruction":sample['instruction'], 
        "bbox":sample['bbox'],
        "additional_paras": {
            "group": sample['group'], 
            "ui_type": sample['ui_type'],
            "image_size": Image.open(image_path).size,
            'platform':sample['platform'],
            'application': sample['application']
        },   
    }

    return ret


def screenspotpro_builder():
    # init 
    anno_path = "/vlm-ssd/FoundationModel/dataset/ScreenSpot-Pro/annotations.json"
    image_dir =  "/vlm-ssd/FoundationModel/dataset/ScreenSpot-Pro/images"
    ret_samples = []

    with open(anno_path, 'r') as f:
        total_data = json.load(f)

    for sample in total_data:
        ret = convert2swift(image_dir, sample)
        ret_samples.append(ret)
    
    return ret_samples
if __name__ == "__main__":
    screenspotpro_builder()