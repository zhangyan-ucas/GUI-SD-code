import json
import os
import random
from collections import Counter

# 设置随机种子保证可复现
random.seed(42)

# ----------------- 路径与参数配置 -----------------
data_names = ["250310", "250317", "250407", "250414",
              "250428", "250505", "250526", "250630", "250707", "250714"]

base_root = "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua"


def is_in_box(point, bbox):
    return (bbox[0] < point[0] < bbox[2]) and (bbox[1] < point[1] < bbox[3])


def process_one(data_name):
    """处理单个 data_name，返回统计信息"""
    base_dir = f"{base_root}/{data_name}"

    venus_pred_path  = f"{base_dir}/{data_name}_venuspred.jsonl"
    venus_pred2_path = f"{base_dir}/{data_name}_venuspred2.jsonl"
    inst_path        = f"{base_dir}/{data_name}_instmod.jsonl"
    src_path         = f"{base_dir}/{data_name}_sc.jsonl"
    category_path    = f"{base_dir}/{data_name}_category.jsonl"
    qwen3vl_path     = f"{base_dir}/{data_name}_qwen3vl.jsonl"
    teacher_path     = f"{base_dir}/{data_name}_qwen3vl_teacher.jsonl"

    # ============ Stage 1: 数据对齐、验证、标注 ============
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

            sample_id = sample_src['sample_id']
            assert sample_id == sample_v1['sample_id'] == sample_v2['sample_id'] == \
                   sample_inst['sample_id'] == sample_cat['sample_id'] == \
                   sample_qwen['sample_id'] == sample_teacher['sample_id']

            gt_bbox = sample_src['solution']['arguments']['coordinate']
            pred_point1 = json.loads(sample_v1['additional_paras'])['venus_pred']
            pred_point2 = json.loads(sample_v2['additional_paras'])['venus_pred']

            if is_in_box(pred_point1, gt_bbox) and is_in_box(pred_point2, gt_bbox):
                venus_both_correct_count += 1

                mod_instruction = json.loads(sample_inst['additional_paras'])['mod_instruction']
                sample_src['messages'][1]['content'] = mod_instruction

                ui_type = json.loads(sample_cat['additional_paras'])['ui_type']
                assert ui_type in ['icon', 'text']

                qwen_pred = json.loads(sample_qwen['additional_paras'])['qwen3vl_pred']
                qwen_correct = is_in_box(qwen_pred, gt_bbox)

                teacher_pred = json.loads(sample_teacher['additional_paras'])['qwen3vl_pred']
                teacher_correct = is_in_box(teacher_pred, gt_bbox)

                if teacher_correct:
                    teacher_correct_count += 1
                else:
                    teacher_wrong_count += 1

                additional_paras = json.loads(sample_src['additional_paras'])
                additional_paras['ui_type'] = ui_type
                additional_paras['qwen_correct'] = qwen_correct
                additional_paras['teacher_correct'] = teacher_correct
                sample_src['additional_paras'] = json.dumps(additional_paras)

                stage1_samples.append(sample_src)

    print(f"\n{'='*60}")
    print(f"Data: {data_name}")
    print(f"{'='*60}")
    print(f"Stage 1 - Venus Strict Correct: {venus_both_correct_count/total_samples*100:.2f}% ({venus_both_correct_count}/{total_samples})")
    print(f"Teacher Correct: {teacher_correct_count} ({teacher_correct_count/venus_both_correct_count*100:.2f}%)")
    print(f"Teacher Wrong  : {teacher_wrong_count} ({teacher_wrong_count/venus_both_correct_count*100:.2f}%)")

    # ============ Stage 2: 统计分布，计算平衡保留率 ============
    # 先过滤掉 teacher 做错的
    teacher_passed = []
    for s in stage1_samples:
        ap = json.loads(s['additional_paras'])
        if ap['teacher_correct']:
            teacher_passed.append(s)

    # 统计四个象限: {ui_type} x {qwen_correct}
    counts = {
        'icon_qwen_wrong': 0,   # icon 且 qwen 做错 → 全部保留
        'icon_qwen_correct': 0, # icon 且 qwen 做对 → 按比例保留
        'text_qwen_wrong': 0,   # text 且 qwen 做错 → 全部保留
        'text_qwen_correct': 0, # text 且 qwen 做对 → 按比例保留
    }
    for s in teacher_passed:
        ap = json.loads(s['additional_paras'])
        key = f"{ap['ui_type']}_qwen_{'correct' if ap['qwen_correct'] else 'wrong'}"
        counts[key] += 1

    print(f"\n--- Distribution (teacher-passed) ---")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # 计算保留率，目标: icon_total == text_total
    # icon_total = icon_qwen_wrong + icon_qwen_correct * p_icon
    # text_total = text_qwen_wrong + text_qwen_correct * p_text
    #
    # 策略: 少数类的 qwen_correct 全保留 (p=1.0)，多数类降采样
    # 如果少数类全保留后仍然不够，则还需要对多数类的 qwen_wrong 也降采样

    iw = counts['icon_qwen_wrong']
    ic = counts['icon_qwen_correct']
    tw = counts['text_qwen_wrong']
    tc = counts['text_qwen_correct']

    # 假设两类的 qwen_correct 都全保留时的最大值
    icon_max = iw + ic
    text_max = tw + tc

    if icon_max == 0 and text_max == 0:
        print("No samples available!")
        return

    # 确定哪个是少数类
    if icon_max <= text_max:
        minority, majority = 'icon', 'text'
        min_w, min_c = iw, ic
        maj_w, maj_c = tw, tc
    else:
        minority, majority = 'text', 'icon'
        min_w, min_c = tw, tc
        maj_w, maj_c = iw, ic

    # 少数类全保留 → 少数类总量
    min_total = min_w + min_c
    p_minority = 1.0  # 少数类 qwen_correct 全保留

    # 多数类需要降到 min_total
    # maj_total = maj_w + maj_c * p_majority = min_total
    # → p_majority = (min_total - maj_w) / maj_c
    if maj_c > 0:
        p_majority = (min_total - maj_w) / maj_c
    else:
        # 没有 qwen_correct 样本可调，只能对 qwen_wrong 截断
        p_majority = 0.0

    # 处理边界情况
    p_majority_wrong = 1.0  # 默认不对 qwen_wrong 降采样
    if p_majority > 1.0:
        # 多数类即使全保留也不够 → 两边都全保留，无法完美 1:1
        p_majority = 1.0
        print(f"\n⚠ Warning: Cannot achieve perfect 1:1. "
              f"Minority({minority})={min_total}, Majority({majority})={maj_w + maj_c}. "
              f"Keeping all and accepting imbalance.")
    elif p_majority < 0:
        # 多数类的 qwen_wrong 本身就超过少数类总量 → 需要对 qwen_wrong 也降采样
        # 此时不保留任何 qwen_correct，并对 qwen_wrong 降采样
        p_majority = 0.0
        # 还需要对多数类 qwen_wrong 降采样
        p_majority_wrong = min_total / maj_w if maj_w > 0 else 0.0
        print(f"\n⚠ Note: Majority({majority}) qwen_wrong ({maj_w}) > minority total ({min_total}). "
              f"Will also downsample majority qwen_wrong to p={p_majority_wrong:.4f}")
    else:
        p_majority_wrong = 1.0  # 不需要对 qwen_wrong 降采样

    # 设置各类保留率
    if minority == 'icon':
        p_icon_correct = p_minority
        p_text_correct = p_majority
        p_icon_wrong = 1.0
        p_text_wrong = p_majority_wrong
    else:
        p_text_correct = p_minority
        p_icon_correct = p_majority
        p_text_wrong = 1.0
        p_icon_wrong = p_majority_wrong

    print(f"\n--- Retention Rates ---")
    print(f"  icon_qwen_correct: {p_icon_correct:.4f}  (keep {p_icon_correct*100:.1f}%)")
    print(f"  icon_qwen_wrong  : {p_icon_wrong:.4f}  (keep {p_icon_wrong*100:.1f}%)")
    print(f"  text_qwen_correct: {p_text_correct:.4f}  (keep {p_text_correct*100:.1f}%)")
    print(f"  text_qwen_wrong  : {p_text_wrong:.4f}  (keep {p_text_wrong*100:.1f}%)")

    expected_icon = iw * p_icon_wrong + ic * p_icon_correct
    expected_text = tw * p_text_wrong + tc * p_text_correct
    print(f"\n  Expected icon: {expected_icon:.0f}, Expected text: {expected_text:.0f}")

    # ============ Stage 3: 按计算出的比例过滤 ============
    final_samples = []
    image_counter = Counter()

    for sample in teacher_passed:
        ap = json.loads(sample['additional_paras'])
        ui_type = ap['ui_type']
        qwen_correct = ap['qwen_correct']

        # 根据类型和 qwen 正确性决定保留率
        if ui_type == 'icon':
            p = p_icon_correct if qwen_correct else p_icon_wrong
        else:
            p = p_text_correct if qwen_correct else p_text_wrong

        # 按概率采样
        if p < 1.0 and random.random() >= p:
            continue

        # 同一张图片不超过 4 次
        img_path = sample['images']
        if image_counter[img_path] < 4:
            image_counter[img_path] += 1
            final_samples.append(sample)

    total = len(final_samples)
    if total > 0:
        fc = Counter(json.loads(s['additional_paras']).get('ui_type') for s in final_samples)
        icon_n = fc.get('icon', 0)
        text_n = fc.get('text', 0)
        ratio = icon_n / text_n if text_n > 0 else float('inf')
        print(f"\n--- Final Result ---")
        print(f"  Total: {total}")
        print(f"  Icon : {icon_n} ({icon_n/total*100:.2f}%)")
        print(f"  Text : {text_n} ({text_n/total*100:.2f}%)")
        print(f"  Ratio icon:text = 1:{1/ratio:.2f}" if ratio > 0 else "  Ratio: N/A")
    else:
        print("\nNo samples left after filtering.")

    return {
        'data_name': data_name,
        'total': total,
        'icon': fc.get('icon', 0) if total > 0 else 0,
        'text': fc.get('text', 0) if total > 0 else 0,
        'samples': final_samples,
    }


# ----------------- 主流程 -----------------
all_stats = []
all_samples = []
for dn in data_names:
    stats = process_one(dn)
    if stats:
        all_stats.append(stats)
        all_samples.extend(stats['samples'])

# 汇总保存到最终路径
final_save_path = "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/version_data/v54_opsd.jsonl"
os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
with open(final_save_path, 'w', encoding='utf-8') as f_out:
    for sample in all_samples:
        f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

# 汇总
print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")
total_all = sum(s['total'] for s in all_stats)
total_icon = sum(s['icon'] for s in all_stats)
total_text = sum(s['text'] for s in all_stats)
print(f"{'Data':<10} {'Total':>8} {'Icon':>8} {'Text':>8} {'Ratio':>10}")
print("-" * 50)
for s in all_stats:
    r = f"1:{s['text']/s['icon']:.2f}" if s['icon'] > 0 else "N/A"
    print(f"{s['data_name']:<10} {s['total']:>8} {s['icon']:>8} {s['text']:>8} {r:>10}")
print("-" * 50)
r_all = f"1:{total_text/total_icon:.2f}" if total_icon > 0 else "N/A"
print(f"{'ALL':<10} {total_all:>8} {total_icon:>8} {total_text:>8} {r_all:>10}")
print(f"\nAll samples saved to: {final_save_path}")
print(f"Total samples: {len(all_samples)}")