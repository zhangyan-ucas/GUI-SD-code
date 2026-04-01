import json
import matplotlib.pyplot as plt

# path = "/mnt/vlm-ks3/zhangyan/nips_code/opd/output/v4_2infer/v0-20260310-094417/logging.jsonl"
path = "/mnt/vlm-ks3/zhangyan/nips_code/opd/output/v3_opd/v2-20260310-160823/logging.jsonl"

data = []

# 1. 读取数据
with open(path, 'r', encoding='utf-8') as f:
    for line in f:

        json_obj = json.loads(line.strip())
        data.append(json_obj)

# 2. 提取 loss 和 step
# 提取所有的 loss，同时做个安全检查防止某一行没有 'loss' 字段
losses = [item['loss'] for item in data if 'loss' in item]

# 如果你的 json_obj 里面记录了 step（比如 item['step']），建议提取出来作为 X 轴：
# steps = [item['step'] for item in data if 'loss' in item]
# 如果没有 step 字段，可以直接用数据点的序号作为 X 轴：
steps = range(len(losses))

# 3. 绘制可视化曲线
plt.figure(figsize=(10, 6))  # 设置画布大小
plt.plot(steps, losses, label='Training Loss', color='#1f77b4', linewidth=1.5)

# 4. 美化图表
plt.title('Training Loss Curve', fontsize=15)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
plt.legend(fontsize=12)                    # 显示图例

# 5. 保存并显示
save_path = "/mnt/vlm-ks3/zhangyan/nips_code/opd/useful_output/train_curves/loss_curve.png"
plt.savefig(save_path, dpi=300, bbox_inches='tight') # 高清保存
print(f"Loss曲线已成功保存至当前目录: {save_path}")

# 如果你是在本地 Jupyter Notebook 运行，可以解除下面这行的注释直接显示
# plt.show()