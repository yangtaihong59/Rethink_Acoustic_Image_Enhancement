# KDLAE: Knowledge-Distilled Lightweight Acoustic Enhancer

[简体中文](README_ZH.md) | English

---

## Introduction
- **KDLAE-T (Teacher)**: Teacher network modified based on Restormer, supporting controllable denoising strength and super-resolution for high-quality denoising.
- **KDLAE-S (Student)**: Lightweight multi-frame denoising network using 3D convolution for spatiotemporal feature fusion, targeting real-time/low-resource scenarios.

This directory contains model implementations, weights, and two example notebooks: `KDLAE_T.ipynb` and `KDLAE-S.ipynb`, which can be run directly for inference and comparison.

---

## Directory Structure
- `KDLAE_model.py`: Contains `KDLAE_teacher`, `KDLAE_student`, and basic `Restormer` implementation
- `weights/`: Pretrained weights, e.g., `KDLAE-S-FLS.pth`, `KDLAE-S-US.pth`
- `KDLAE_T.ipynb`: Teacher model inference/enhancement examples
- `KDLAE-S.ipynb`: Student model multi-frame denoising examples

---

## Pretrained Weights
Place weights under `weights/`:
- Ultrasound (US) example: `weights/KDLAE-S-US.pth`
- Sonar/laparoscopic, etc. (FLS) example: `weights/KDLAE-S-FLS.pth`

> If using custom weights, please modify the weight paths in the notebooks or scripts.

---

## Quick Start

### KDLAE-T Teacher Model Inference
In `KDLAE_T.ipynb`:
1) Prepare input images and optional `denoise_rate` (to control denoising strength)
2) Run inference, export `hq` (denoised) and optional `sr` (enhanced)

### KDLAE-S Student Model Inference
In `KDLAE-S.ipynb`:
1) Select input sequence (multi-frame PNG/JPG)
2) Load corresponding weights (US/FLS)
3) Run inference cells to save denoised results

---

## Training and Distillation
- Teacher model: Based on `KDLAE_teacher` and Restormer structure, supports `denoise_rate` to control strength.
- Student model: `KDLAE_student` uses 3D UNet-style encode-fusion-decode, suitable for distillation deployment.

Note: This repository currently provides inference/comparison examples. For complete training scripts and distillation pipelines, please extend data loading, loss functions, and schedulers based on this.

---

## Citation
If this project is helpful for your research or product, please cite and acknowledge in your documentation or paper: TBD after paper acceptance

