[简体中文](README_ZH.md) | English
---

## Project Description

This project is the open-source code repository of the paper, containing the denoising method KDLAE, acoustic image quality assessment ASDQE, sample data, and environment dependencies. This document is a quick start guide and navigation.

We provide the training dataset used. [Quantitative Simulated Forward-Looking Sonar Dataset](https://www.kaggle.com/datasets/taihongyang59/quantitative-simulated-forward-looking-sonar)


#### FLS Sample Video

[![Demo Video 1](https://img.youtube.com/vi/CkXvV9udJZE/0.jpg)](https://youtu.be/CkXvV9udJZE)

#### Ultrasound Sample Video

[![Demo Video 2](https://img.youtube.com/vi/uBQoOC9qfEc/0.jpg)](https://youtu.be/uBQoOC9qfEc)

The training code and acoustic image simulation tools are still being organized.

### Directory Structure

- `requirements.txt`: Python dependency list in this directory.
- `KDLAE/`: KDLAE denoising method.
  - `KDLAE_T.ipynb`: KDLAE teacher model.
  - `KDLAE-S.ipynb`: KDLAE student model.
  - `KDLAE_model.py`: KDLAE model implementation.

- `ASDQE/`: ASDQE acoustic reference-free image quality assessment.
  - `ASDQE_model.py`: ASDQE model implementation.
  - `ASDQE_test.py`: ASDQE usage script.

- `Sample/`: Sample data or configuration.

### Environment Setup

It is recommended to use a Python virtual environment (such as venv or conda).

```bash
# 1) Navigate to the project directory
cd Rethink_Acoustic_Image_Enhancement

# 2) (Optional) Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
```

If you encounter version compatibility issues with specific deep learning frameworks (e.g., PyTorch, CUDA, etc.), please install the appropriate precompiled packages based on your CUDA version.

### Quick Start

#### Download pretrained weights from Releases

| Weight Name | Download Link |
|-------------|----------------|
| KDLAE-T        | [KDLAE_T.pth](https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/KDLAE_T.pth)         |
| KDLAE-S-FLS    | [KDLAE-S-FLS.pth](https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/KDLAE-S-FLS.pth) |
| KDLAE-S-US     | [KDLAE-S-US.pth](https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/KDLAE-S-US.pth)   |
| ASDQE          | [ASDQE.pth](https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/ASDQE.pth)             |

#### 1) [KDLAE](./KDLAE/README_EN.md) Denoising

Run directly in Jupyter.
  1. Open `KDLAE/KDLAE_T.ipynb` or `KDLAE/KDLAE-S.ipynb`.
  2. In the first cell, configure the dataset path, output directory, and training/inference parameters as needed.
  3. Run all cells sequentially to complete training or inference and visualization.

#### 2) [ASDQE](./ASDQE/README_EN.md) Image Quality Assessment

`ASDQE/ASDQE_test.py` provides a test entry for reference-free quality assessment.

Please check the script header and function comments, and fill in or modify parameters such as input directory, file suffix, batch processing settings, etc., to evaluate objective metrics of original/denoised results.

## Citation

If this project is helpful for your research or product, please cite and acknowledge in your documentation or paper: TBD after paper acceptance

### License and Acknowledgments

Copyright 2025 Taihong Yang

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


