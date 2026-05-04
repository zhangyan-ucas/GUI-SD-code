"""
统一评测脚本：给定 run_name，自动遍历所有 checkpoint，对四个数据集分别计算评测指标。

用法:
    python swift/metrics/total_metric.py --run_name v58_g2_grpo
    python swift/metrics/total_metric.py --run_name v55_opsd --metric navi_qwen3
    python swift/metrics/total_metric.py --run_name v56_grpo --datasets screenspotpro uivision
"""

import os
import sys
import json
import argparse
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from swift.utils import read_from_jsonl
from swift.metrics.screenspotpro_metric import compute_screenspotpro
from swift.metrics.screenspotv2_metric import compute_screenspotv2
from swift.metrics.uivision_metric import compute_uivision
from swift.metrics.osworldg_metric import compute_osworldg
from swift.metrics.mmbench_metric import compute_mmbench


# 数据集名 -> (结果文件名, compute 函数, metric 前缀)
DATASET_REGISTRY = {
    "screenspotpro": ("screenspotpro.jsonl", compute_screenspotpro, "screenspotpro"),
    "screenspotv2":  ("screenspotv2.jsonl",  compute_screenspotv2,  "screenspotv2"),
    "uivision":      ("uivision.jsonl",      compute_uivision,      "uivision"),
    "osworldg":      ("osworldg.jsonl",       compute_osworldg,      "osworldg"),
    "osworldg_r":    ("osworldg_r.jsonl",     compute_osworldg,      "osworldg"),
    "mmbench":       ("mmbench.jsonl",        compute_mmbench,       "mmbench"),
}

OUTPUT_BASE = "./output"


def get_latest_vdir(run_name):
    base_dir = os.path.join(OUTPUT_BASE, run_name)
    if not os.path.exists(base_dir):
        print(f"ERROR: run_name '{run_name}' not found in {OUTPUT_BASE}")
        return None
    v_dirs = sorted(glob(os.path.join(base_dir, "v*")), key=os.path.getmtime, reverse=True)
    return v_dirs[0] if v_dirs else None


def get_checkpoints(vdir):
    ckpts = glob(os.path.join(vdir, "checkpoint-*"))
    ckpts = sorted(ckpts, key=lambda x: int(os.path.basename(x).split("-")[1]))
    return ckpts


def evaluate_checkpoint(ckpt_path, datasets, metric_suffix):
    """对单个 checkpoint 评测指定数据集，返回汇总结果。"""
    ckpt_name = os.path.basename(ckpt_path)
    results = {}

    for ds_name in datasets:
        if ds_name not in DATASET_REGISTRY:
            print(f"  WARNING: unknown dataset '{ds_name}', skipping.")
            continue

        filename, compute_fn, prefix = DATASET_REGISTRY[ds_name]
        jsonl_path = os.path.join(ckpt_path, "infer_result", filename)

        if not os.path.exists(jsonl_path):
            print(f"  [{ds_name}] SKIP - file not found: {jsonl_path}")
            results[ds_name] = None
            continue

        metric = f"{prefix}_{metric_suffix}"
        print(f"\n{'='*60}")
        print(f"  [{ckpt_name}] Evaluating {ds_name} (metric={metric})")
        print(f"{'='*60}")

        data_list = read_from_jsonl(jsonl_path)
        overall = compute_fn(data_list, metric)
        results[ds_name] = overall

    return results


def print_summary_table(all_results, datasets):
    """打印所有 checkpoint × 数据集的汇总表。"""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    # 表头
    header = f"{'Checkpoint':<25}"
    for ds in datasets:
        header += f"  {ds:>15}"
    print(header)
    print("-" * len(header))

    # 每行
    for ckpt_name, results in all_results.items():
        row = f"{ckpt_name:<25}"
        for ds in datasets:
            if ds not in results or results[ds] is None:
                row += f"  {'N/A':>15}"
            else:
                overall = results[ds]
                # 统一取 Total Acc 或 Average Acc
                acc = overall.get("Total Acc", overall.get("Average Acc", "N/A"))
                row += f"  {acc + '%':>15}"
        print(row)

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="统一评测: 给定 run_name 评测四个数据集")
    parser.add_argument("--run_name", type=str, required=True,
                        help="实验名称，如 v55_opsd, v56_grpo, v57_sft, v58_g2_grpo")
    parser.add_argument("--metric", type=str, default="navi_qwen3",
                        help="metric 后缀，如 navi_qwen3, ground_qwen3, venus (default: navi_qwen3)")
    parser.add_argument("--datasets", nargs="+", type=str,
                        default=["screenspotpro", "screenspotv2", "uivision", "osworldg", "osworldg_r", "mmbench"],
                        choices=list(DATASET_REGISTRY.keys()),
                        help="要评测的数据集列表 (default: 全部六个)")
    args = parser.parse_args()

    # 找到最新 vdir
    vdir = get_latest_vdir(args.run_name)
    if not vdir:
        print(f"ERROR: no vdir found for '{args.run_name}'")
        return
    print(f"Run: {args.run_name}")
    print(f"Vdir: {vdir}")
    print(f"Datasets: {args.datasets}")
    print(f"Metric suffix: {args.metric}")

    # 找到所有 checkpoint
    ckpts = get_checkpoints(vdir)
    if not ckpts:
        print(f"ERROR: no checkpoints found in {vdir}")
        return
    print(f"Found {len(ckpts)} checkpoint(s): {[os.path.basename(c) for c in ckpts]}")

    # 逐个 checkpoint 评测
    all_results = {}
    for ckpt in ckpts:
        ckpt_name = os.path.basename(ckpt)
        results = evaluate_checkpoint(ckpt, args.datasets, args.metric)
        all_results[ckpt_name] = results

    # 打印汇总表
    print_summary_table(all_results, args.datasets)

    # # 保存结果到 JSON
    # save_path = os.path.join(vdir, "eval_summary.json")
    # with open(save_path, "w", encoding="utf-8") as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=2)
    # print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    main()
