import json
import os
import random
from collections import Counter

# 设置随机种子保证可复现
random.seed(42)

# ----------------- 路径与参数配置 -----------------
data_name = "250317"
base_dir = f"/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/{data_name}"

venus_pred_path  = f"{base_dir}/{data_name}_venuspred.jsonl"
venus_pred2_path = f"{base_dir}/{data_name}_venuspred2.jsonl"
inst_path        = f"{base_dir}/{data_name}_instmod.jsonl"
src_path         = f"{base_dir}/{data_name}_sc.jsonl"
category_path    = f"{base_dir}/{data_name}_category.jsonl"
qwen3vl_path     = f"{base_dir}/{data_name}_qwen3vl.jsonl"
teacher_path     = f"{base_dir}/{data_name}_qwen3vl_teacher.jsonl"
save_path        = f"{base_dir}/{data_name}_filter.jsonl"


# 辅助函数:判断点是否在框内
def is_in_box(point, bbox):
    return (bbox[0] < point[0] < bbox[2]) and (bbox[1] < point[1] < bbox[3])


# ----------------- 第一次循环:数据对齐、验证、标注 -----------------
stage1_samples = []
total_samples = 0
venus_both_correct_count = 0
teacher_correct_count = 0
teacher_wrong_count = 0

with open(venus_pred_path, 'r', encoding='utf-8') as f_v1, \
     open(venus_pred2_path, 'r', encoding='utf-8') as f_v2, \
     open(inst_path, 'r', encoding='utf-8') as f_inst, \
     open(src_path, 'r', encoding='utf-8') as f_src, \
     open(category_path, 'r', encoding='utf-8') as f_cat, \
     open(qwen3vl_path, 'r', encoding='utf-8') as f_qwen, \
     open(teacher_path, 'r', encoding='utf-8') as f_teacher:
     
    for line_v1, line_v2, line_inst, line_src, line_cat, line_qwen, line_teacher in zip(
            f_v1, f_v2, f_inst, f_src, f_cat, f_qwen, f_teacher):
        total_samples += 1
        
        sample_v1 = json.loads(line_v1.strip())
        sample_v2 = json.loads(line_v2.strip())
        sample_inst = json.loads(line_inst.strip())
        sample_src = json.loads(line_src.strip())
        sample_cat = json.loads(line_cat.strip())
        sample_qwen = json.loads(line_qwen.strip())
        sample_teacher = json.loads(line_teacher.strip())
        
        # 确保7个文件的样本是对齐的
        sample_id = sample_src['sample_id']
        assert sample_id == sample_v1['sample_id'] == sample_v2['sample_id'] == \
               sample_inst['sample_id'] == sample_cat['sample_id'] == \
               sample_qwen['sample_id'] == sample_teacher['sample_id']

        # 提取 GT BBox
        gt_bbox = sample_src['solution']['arguments']['coordinate']
        
        # 提取 venus 预测点
        pred_point1 = json.loads(sample_v1['additional_paras'])['venus_pred']
        pred_point2 = json.loads(sample_v2['additional_paras'])['venus_pred']
        
        # 判断 venus 是否都在 GT 内
        if is_in_box(pred_point1, gt_bbox) and is_in_box(pred_point2, gt_bbox):
            venus_both_correct_count += 1
            
            # 1. 修改 instruct
            mod_instruction = json.loads(sample_inst['additional_paras'])['mod_instruction']
            sample_src['messages'][1]['content'] = mod_instruction
            
            # 2. 提取 category
            ui_type = json.loads(sample_cat['additional_paras'])['ui_type']
            assert ui_type in ['icon', 'text']
            
            # 3. 判断 qwen3vl 是否做对
            qwen_pred = json.loads(sample_qwen['additional_paras'])['qwen3vl_pred']
            qwen_correct = is_in_box(qwen_pred, gt_bbox)
            
            # 4. 判断 teacher 是否做对
            teacher_pred = json.loads(sample_teacher['additional_paras'])['qwen3vl_pred']
            teacher_correct = is_in_box(teacher_pred, gt_bbox)
            
            # 统计 teacher 正确/错误数量(在 venus 通过的样本中)
            if teacher_correct:
                teacher_correct_count += 1
            else:
                teacher_wrong_count += 1
            
            # 将补充信息保存进 additional_paras
            additional_paras = json.loads(sample_src['additional_paras'])
            additional_paras['ui_type'] = ui_type
            additional_paras['qwen_correct'] = qwen_correct
            additional_paras['teacher_correct'] = teacher_correct
            sample_src['additional_paras'] = json.dumps(additional_paras)
            
            # 存入第一阶段列表
            stage1_samples.append(sample_src)

print(f"Stage 1 - Venus Strict Correct Accuracy: {venus_both_correct_count/total_samples*100:.2f}% ({venus_both_correct_count}/{total_samples})")
print(f"\n--- Teacher Statistics (among venus-passed samples) ---")
print(f"Teacher Correct: {teacher_correct_count} ({teacher_correct_count/venus_both_correct_count*100:.2f}%)")
print(f"Teacher Wrong  : {teacher_wrong_count} ({teacher_wrong_count/venus_both_correct_count*100:.2f}%)")


# ----------------- 第二次循环:过滤与清洗 -----------------
final_samples = []
image_counter = Counter()

for sample in stage1_samples:
    additional_paras = json.loads(sample['additional_paras'])
    qwen_correct = additional_paras['qwen_correct']
    teacher_correct = additional_paras['teacher_correct']
    
    # 条件0: teacher 必须做对,做错直接跳过
    if not teacher_correct:
        continue
    
    # 条件1:如果 qwen 做对了,仅保留 10% (即 90% 的概率被跳过);做错了则全保留
    if qwen_correct and random.random() >= 0.1:
        continue
        
    # 条件2:同一张图片出现不能超过 4 次
    img_path = sample['images']
    if image_counter[img_path] < 4:
        image_counter[img_path] += 1
        final_samples.append(sample)


# ----------------- 保存结果 -----------------
with open(save_path, 'w', encoding='utf-8') as f_save:
    for sample in final_samples:
        f_save.write(json.dumps(sample, ensure_ascii=False) + '\n')

print(f"\nStage 2 - Final filtered samples count: {len(final_samples)}")
print(f"File successfully saved in: {save_path}")


# ----------------- 数据统计:text与icon比例 -----------------
total = len(final_samples)

if total > 0:
    counts = Counter(json.loads(s['additional_paras']).get('ui_type') for s in final_samples)
    
    print("\n--- Final Category Statistics ---")
    print(f"Total Text : {counts['text']} ({counts['text'] / total * 100:.2f}%)")
    print(f"Total Icon : {counts['icon']} ({counts['icon'] / total * 100:.2f}%)")
else:
    print("\nNo samples left after filtering.")