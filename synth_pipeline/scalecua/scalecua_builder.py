import json
import os
from tqdm import tqdm 
import re
from custom_utils.vis_image import vis_func
from swift.custom_utils.ground_func import bboxnorm2real, pointreal2norm
from copy import deepcopy

def convert_message(sample):
    instruction = sample['conversations'][0]['value'].replace("<image>\n", "")  
    text = sample['conversations'][1]['value'].replace("<image>", "")

    if "<action>" in text:
        # point
        match = re.search(r'x=([0-9.]+),\s*y=([0-9.]+)', text)
        x_val = int(float(match.group(1)) * sample['width'])
        y_val = int(float(match.group(2)) * sample['height'])

        return instruction, [x_val, y_val]
    
    elif "<point>" in text:
        # point 
        match = re.search(r'<point>\[\[([0-9.]+),\s*([0-9.]+)\]\]</point>', text)
        x_val = match.group(1)
        y_val = match.group(2)

        return instruction, [x_val, y_val]

    elif "<ref>" in text:
        # bbox
        inst_match = re.search(r"<ref>(.*?)</ref>", instruction, flags=re.DOTALL)
        extracted_content = inst_match.group(1)

        match = re.search(r"<box>\[\[(.*?)\]\]</box>", text)
        coords_str = match.group(1)
        coords = [int(x.strip()) for x in coords_str.split(',')]
        coords = bboxnorm2real(coords, [sample['width'], sample['height']])

        return extracted_content, coords
    
    else:
        raise NotImplementedError()



def messages_builder(sample, sys_prompt):
    if len(sample['gt']) != 4: return None

    ass_text = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [p_x, p_y]}}\n</tool_call>'
    
    real_bbox = deepcopy(sample['gt'])
    norm_point = pointreal2norm([sum(real_bbox[::2]) // 2, sum(real_bbox[1::2]) // 2], [sample['width'], sample['height']])

    ass_text = ass_text.replace("p_x", str(norm_point[0])).replace("p_y", str(norm_point[1]))
    
    messages = [
        {"content": sys_prompt, "role": "system"},
        {"content": sample['instruction'], "role": "user"},
        {"content": ass_text, "role": "assistant"},
    ]
    
    ret = {
        "solution": {'name': 'mobile_use', 'arguments': {'action': 'click', 'coordinate': sample['gt']}},   
        "images": sample['images'], 
        "messages": messages, 
        "additional_paras": json.dumps(
            {
                "image_size": [sample['width'], sample['height']],
                "platform": sample['platform']
            }
        ),
    }

    return ret

def sample2files(total_samples, save_path):
    # 只需要在最开始读取一次 system prompt
    with open("/mnt/vlm-ks3/zhangyan/nips_code/opd/synth_pipeline/prompts/qwen3vl_ground.txt", encoding='utf-8') as f:
        sys_prompt = f.read()

    with open(save_path, 'w', encoding='utf-8') as f:
        # 这里顺便也加个进度条，保存时也能看到进度
        for idx, sample in enumerate(tqdm(total_samples, desc="Saving to JSONL")):
            ret = messages_builder(sample, sys_prompt)
            if ret != None: 
                ret['sample_id'] = idx
                json_line = json.dumps(ret, ensure_ascii=False)
                f.write(json_line + '\n')
            
    print(f"成功保存 {len(total_samples)} 条数据到 {save_path}")