<h1 align="center"> Learn where to Click from Yourself: On-Policy Self-Distillation for GUI Grounding</h1>
<p align="center">
<h4 align="center">This is the official repository of the paper <a href="https://arxiv.org/abs/2605.00642">GUI-SD</a>.</h4>
<h5 align="center"><em><a href="https://scholar.google.com/citations?hl=zh-CN&user=IUNcUO0AAAAJ">Yan Zhang</a>, Daiqing Wu, Huawen Shen, Yu Zhou, Can Ma </em></h5>


# News

***2025/04/30***

- We are the first to introduce on-policy self-distillation for GUI grounding.
- We release the code and training data.

# Installation

```python
conda create -n GUI-SD python=3.10 -y
conda activate GUI-SD
pip install -r requirements.txt
```

# Training

```
sh ./GUI-SD_scripts/train.sh
```

# Evaluate

```
sh ./GUI-SD_scripts/test.sh
```


# Statement

This project is for research purpose only. For any other questions please contact [zhangyan2022@iie.ac.cn](mailto:zhangyan2022@iie.ac.cn).

## Acknowledgements

Thanks for providiong open-source dataset ScaleCuA.