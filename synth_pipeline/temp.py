import json

paths = [
    "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/250310/250310_filter.jsonl",
    "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/250317/250317_filter.jsonl",
    "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/250407/250407_filter.jsonl",
    "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/250414/250414_filter.jsonl",
    "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/250428/250428_filter.jsonl",
    "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/250505/250505_filter.jsonl",
    "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/250526/250526_filter.jsonl",
    "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/250630/250630_filter.jsonl",
    "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/250707/250707_filter.jsonl",
    "/mnt/vlm-ks3/zhangyan/datasets/nips_data/swift_data/scalecua/250714/250714_filter.jsonl",
]

total_text = 0
total_icon = 0

for path in paths:
    text_count = 0
    icon_count = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            ui_type = json.loads(sample["additional_paras"])['ui_type']
            if ui_type == 'text':
                text_count += 1
            elif ui_type == 'icon':
                icon_count += 1

    total = text_count + icon_count
    ratio = f"{text_count/icon_count:.2f}" if icon_count > 0 else "N/A"
    print(f"{path.split('/')[-2]:>8} | Text: {text_count:>6}, Icon: {icon_count:>6}, "
          f"Total: {total:>6}, Text/Icon: {ratio}")
    total_text += text_count
    total_icon += icon_count

print("-" * 80)
total = total_text + total_icon
ratio = f"{total_text/total_icon:.2f}" if total_icon > 0 else "N/A"
print(f"{'合计':>8} | Text: {total_text:>6}, Icon: {total_icon:>6}, "
      f"Total: {total:>6}, Text/Icon: {ratio}")