## ASDQE — Acoustic-Specific Denoise Quality Evaluator

[简体中文](README_ZH.md) | English

---

### Introduction
ASDQE is a quality assessment tool for comparing the effectiveness of different acoustic image denoising methods. Based on a lightweight deep learning model, it takes a pair of images as input: a noisy low-quality image (LQ) and the corresponding high-quality/denoised image (GT), outputs a method score for comparison, and aggregates statistical metrics (mean, standard deviation, quantiles, etc.) for horizontal comparison across different methods.

This repository contains:
- `ASDQE_model.py`: Model structure (including feature extraction, simplified UNet enhancement, and regression head).
- `ASDQE_test.py`: Evaluation script that loads paired images and performs batch inference and statistical comparison on multiple denoising results.
- `weights/ASDQE.pth`: Pretrained weights file.

---

### Directory Structure
```text
ASDQE/
├─ ASDQE_model.py         # 模型定义 / Model definition
├─ ASDQE_test.py          # 评估脚本 / Evaluation script
└─ weights/
   └─ ASDQE.pth           # 预训练权重 / Pretrained weights
```

---

### Quick Start

1) After cloning or entering this directory, confirm that the weights file exists: `weights/ASDQE.pth`.

2) Configure data paths in `ASDQE_test.py`:
   - `base_lg_dir`: Low-quality (noisy) image directory
   - `denoise_dir`: Root directory of denoising results
   - `denoising_methods`: Mapping from method names to their respective denoising subdirectories (examples provided, can be added or removed as needed)

3) Run the evaluation script:
```bash
python ASDQE_test.py
```

The script will:
- Load the model and data
- Perform pairwise inference for each method
- Output statistical metrics for each method
- Generate `stats_transposed.csv` (rows: statistical items; columns: methods)

---

### Path and Data Organization
The evaluation script assumes:
- Low-quality image files under `base_lg_dir` (e.g., `0001.jpg`, `0002.jpg` ...)
- Each denoising method directory contains files with the same names as those in `base_lg_dir` (one-to-one file name correspondence), for example:
  - `${denoise_dir}/Teacher/0001.jpg`
  - `${denoise_dir}/Teacher/0002.jpg`

If the count or file names do not match, the script will raise an error to prompt data organization check.

---

### Result Interpretation
The script prints statistical summaries for each method (`mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`). Whether larger values are better depends on the training objective (the regression output range in this project is [-1, 1], and its absolute meaning needs to be combined with specific training definitions and downstream scales). It is recommended:
- To perform relative comparison across multiple methods on the same dataset
- To cross-validate with visual inspection or objective metrics

---

### Citation
If this project is helpful for your research or product, please cite and acknowledge in your documentation or paper: TBD after paper acceptance

---

