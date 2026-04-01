import random
import argparse
import json 

# 1. 设置命令行参数
parser = argparse.ArgumentParser(description="合并并打乱多个 JSONL 文件")
parser.add_argument("-i", "--inputs", nargs='+', required=True, help="输入需要合并的 jsonl 文件列表 (空格分隔)")
parser.add_argument("-o", "--save_path", required=True, help="输出的 jsonl 文件路径")
args = parser.parse_args()

# 2. 读取所有文件中的所有行
lines = [line for f in args.inputs for line in open(f, encoding='utf-8')]

# 3. 随机打乱
random.seed(42)
random.shuffle(lines)

# 4. 重新分配 sample_id 并写入新文件
with open(args.save_path, 'w', encoding='utf-8') as f:
    for idx, line in enumerate(lines):
        sample = json.loads(line)
        sample['sample_id'] = idx
        
        # 将修改后的字典转回 JSON 字符串，并写入文件
        # ensure_ascii=False 确保中文等非 ASCII 字符正常显示，不变成 \uXXXX
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')