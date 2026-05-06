<h1 align="center"> Learn where to Click from Yourself: On-Policy Self-Distillation for GUI Grounding</h1>
<p align="center">
<h4 align="center">This is the official repository of the paper <a href="https://arxiv.org/abs/2605.00642">GUI-SD</a>.</h4>
<h5 align="center"><em><a href="https://scholar.google.com/citations?hl=zh-CN&user=IUNcUO0AAAAJ">Yan Zhang</a>, <a href="https://scholar.google.com/citations?user=EMhXDBMAAAAJ&hl=zh-CN">Daiqing Wu</a>, Huawen Shen, Can Ma, Yu Zhou </em></h5>



# News

***2025/04/30***

- We are the first to introduce on-policy self-distillation for GUI grounding.
- We release the code and training data.

# Installation

1. Install ms-swift framework following the installation. We recommend installing ms-swift==4.0.0-dev, so that you can directly start our training script.

2. Manually install the dependencies:

```python
conda create -n GUI-SD python=3.10 -y
conda activate GUI-SD
pip install -r requirements.txt
```

# Data preparation

1. Download our dataset from Hugging Face:

```bash
# Option 1: Using huggingface-cli (recommended)
pip install huggingface_hub
huggingface-cli download yankie123/GUI-SD-data --repo-type dataset --local-dir ./data

# Option 2: Using git lfs
git lfs install
git clone https://huggingface.co/datasets/yankie123/GUI-SD-data ./data
```

You can also browse and download the dataset directly from the [🤗 Hugging Face Dataset Page](https://huggingface.co/datasets/yankie123/GUI-SD-data).

2. After downloading, place the dataset file (e.g., `gui-sd.jsonl`) in your working directory. The training script loads data via the `--dataset` argument:

```bash
--dataset "gui-sd.jsonl"
```

# Training

```
sh ./GUI-SD_scripts/train.sh
```

The training script consists of two stages: (1) launching a vLLM rollout server for on-policy generation, and (2) running the OPSD (On-Policy Self-Distillation) trainer. Below are the key arguments we defined:

**Core OPSD Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_opsd` | `false` | Enable On-Policy Self-Distillation mode. Must be set to `true` to activate GUI-SD training. |
| `--rlhf_type` | - | Set to `gkd` to use Generalized Knowledge Distillation as the base framework. |
| `--lmbda` | `0.5` | On-policy probability. With probability λ, the student generates new responses for training; otherwise uses dataset responses. Set to `1` for fully on-policy. |
| `--beta` | - | KL divergence coefficient controlling the distillation strength. |
| `--seq_kd` | `false` | Whether to use sequence-level knowledge distillation. Set to `false` for token-level KD in GUI-SD. |

**Teacher Model Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--teacher_model` | - | Path or name of the teacher model. In GUI-SD, we use the same model as the student (self-distillation). |
| `--teacher_model_type` | - | Model type of the teacher (e.g., `qwen3_vl`). |
| `--teacher_deepspeed` | - | DeepSpeed strategy for the teacher model (e.g., `zero3`). |
| `--opsd_ema_decay` | `0.0` | EMA decay rate for updating the teacher model. `0`=fixed teacher (no update); `0.95`=moderate update speed. |

**Visual Hint (Mask) Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--opsd_mask_dir` | `cache/opd_cache` | Directory to cache the generated visual hint images for the teacher. |
| `--opsd_mask_mode` | `zoom_in` | Visual hint strategy for the teacher input. Options: `zoom_in` (crop center region), `adaptive` (adaptive bbox scaling), `gaussian` (Gaussian blur around target), `original` (draw bbox on original image), `no_mask` (no visual hint). |
| `--opsd_hint_mode` | `hint` | Text hint mode for the teacher. `none`=no hint; `hint`=add green box + hint text; `gt`=provide ground-truth coordinates directly. |

**Token Weight Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--opsd_token_weight_mode` | `linear` | Token-level weight strategy for KD loss. `uniform`=equal weight 1.0; `linear`=positional weight (hundreds>tens>ones); `uniform-entropy`=uniform × teacher confidence; `linear-entropy`=positional × teacher confidence. |
| `--opsd_non_digit_weight` | `1.0` | Weight assigned to non-digit tokens in the output. Set to `0.1` to focus distillation on coordinate digits. |
| `--opsd_max_digit_len` | `0` | Max effective digit length for linear weighting. `0`=unlimited; `3`=for norm-1000 coordinates (Qwen3-VL); `4`=for absolute pixel coordinates (Qwen2.5-VL). |

**vLLM Rollout Server Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_vllm` | `true` | Enable vLLM for on-policy response generation. |
| `--vllm_mode` | `server` | vLLM running mode. `server` means connecting to an external vLLM server. |
| `--vllm_server_host` | `127.0.0.1` | Host address of the vLLM rollout server. |
| `--vllm_server_port` | `8192` | Port of the vLLM rollout server. |

# Evaluate

```bash
sh ./GUI-SD_scripts/test.sh <run_name> <dataset1> [dataset2] [dataset3] ...
```

**Example:**

```bash
# Evaluate on a single benchmark
sh ./GUI-SD_scripts/test.sh gui-sd screenspotv2

# Evaluate on multiple benchmarks
sh ./GUI-SD_scripts/test.sh gui-sd screenspotpro screenspotv2 uivision
```

The script automatically iterates over all checkpoints under `output/<run_name>/v*/checkpoint-*` and evaluates each on the specified datasets.

**Supported Datasets:**

| Dataset Name | File Path | Description |
|-------------|-----------|-------------|
| `screenspotpro` | `ground_benchmark/screenspotpro.jsonl` | ScreenSpot-Pro benchmark |
| `screenspotv2` | `ground_benchmark/screenspotv2.jsonl` | ScreenSpot-v2 benchmark |
| `uivision` | `ground_benchmark/ui-vision.jsonl` | UI-Vision benchmark |
| `mmbench` | `ground_benchmark/mmbench.jsonl` | MMBench benchmark |
| `osworldg` | `ground_benchmark/osworldg.jsonl` | OSWorld-G benchmark |
| `osworldg_r` | `ground_benchmark/osworldg_r.jsonl` | OSWorld-G (relaxed) benchmark |

**Key Inference Arguments:**

| Argument | Value | Description |
|----------|-------|-------------|
| `--infer_backend` | `pt` | Use PyTorch backend for inference. |
| `--max_new_tokens` | `1024` | Maximum number of new tokens to generate. |
| `--max_batch_size` | `4` | Batch size for inference. |
| `--model_type` | `qwen3_vl` | Model architecture type. |

Results are saved to `<checkpoint>/infer_result/<dataset_name>.jsonl`.


# Statement

This project is for research purpose only. For any other questions please contact [zhangyan2022@iie.ac.cn](mailto:zhangyan2022@iie.ac.cn).

🔥 **We are actively seeking job opportunities (research/engineering positions) in the areas of Multimodal LLMs, GUI Agents, and Reinforcement Learning.** If you are interested in our work or have relevant openings, please feel free to reach out via email!

## Acknowledgements

This project is built upon [ms-swift](https://github.com/modelscope/ms-swift), an efficient and lightweight framework for LLM/VLM fine-tuning and inference. We sincerely thank the ms-swift team for their excellent open-source contribution.

We also thank the open-source dataset [ScaleQuA](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/scalequa) for providing high-quality GUI grounding data.