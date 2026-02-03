<!-- <div align="center"> -->

# 重新思考声学图像增强

### 从水下声纳与医学超声成像

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18q6odwhEs-xJtmEZ1VuEZ1b7Rp0LcqG6?usp=sharing)
[![GitHub stars](https://img.shields.io/github/stars/yangtaihong59/Rethink_Acoustic_Image_Enhancement.svg?style=social&label=Star)](https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=yangtaihong59.Rethink_Acoustic_Image_Enhancement)](https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement)

**简体中文** | [English](README.md)

<!-- </div> -->

---

## 项目说明

本项目是论文的预览开源代码库，包含去噪模型 KDLAE、声学图像质量评价模型 ASDQE 的可运行最小依赖的代码、示例数据等。本文档为中文快速上手指南与导航。

同时本文提供了使用的训练数据集 

<figure>
        <img src="https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/gif/SFLS.gif" alt ="Audio Art" >
        <figcaption>
            Datasets Download: <a href="https://www.kaggle.com/datasets/taihongyang59/quantitative-simulated-forward-looking-sonar/data">[SFLS-Q]</a>
        </figcaption>
</figure>


#### FLS 示例视频

[![Demo Video 1](https://img.youtube.com/vi/CkXvV9udJZE/0.jpg)](https://youtu.be/CkXvV9udJZE)

#### Ultrasound 示例视频

[![Demo Video 2](https://img.youtube.com/vi/uBQoOC9qfEc/0.jpg)](https://youtu.be/uBQoOC9qfEc)

声学影像仿真工具还在整理中

### 目录结构

- `requirements.txt`: 本目录下 Python 依赖清单。
- `KDLAE/`: KDLAE 去噪方法。
  - `KDLAE_T.ipynb`: KDLAE 教师模型。
  - `KDLAE-S.ipynb`: KDLAE 学生模型。
  - `KDLAE_model.py`: KDLAE 模型实现。

- `ASDQE/`: ASDQE 声学无参考图像质量评价。
  - `ASDQE_model.py`: ASDQE 模型实现。
  - `ASDQE_test.py`: ASDQE 使用脚本。

- `Sample/`: 示例数据或配置。

### 环境准备

建议使用 Python 虚拟环境（如 venv 或 conda）。

```bash
# 1) 进入项目目录
cd Rethink_Acoustic_Image_Enhancement

# 2) （可选）创建并激活虚拟环境
python -m venv .venv && source .venv/bin/activate

# 3) 安装依赖
pip install -r requirements.txt

```

如遇到特定深度学习框架（例如 PyTorch、CUDA 等）版本兼容问题，请根据本机 CUDA 版本选择合适的预编译包进行安装。

### 快速开始

#### 在 release 下载权重
| 权重名称         | 下载链接                                                                                                                | 详细信息                                              |
|----------------|----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| KDLAE-T        | [KDLAE_T.pth](https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/KDLAE_T.pth)         | 去噪教师模型$L1_{Shadow}$，可调整去噪率                             |
| KDLAE-T-Dice        | [KDLAE_T_L2Dice.pth](https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/KDLAE_T_L2Dice.pth)         | 去噪教师模型$L2_{Dice}$，可调整去噪率                             |
| KDLAE-S-FLS    | [KDLAE-S-FLS.pth](https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/KDLAE-S-FLS.pth) | 前视声纳蒸馏模型，可输入连续图像帧                     |
| KDLAE-S-US     | [KDLAE-S-US.pth](https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/KDLAE-S-US.pth)   | 超声蒸馏模型，可输入连续图像帧                         |
| ASDQE          | [ASDQE.pth](https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/weight/ASDQE.pth)             | 声学影像去噪评估模型                                 |

你可以直接运行脚本

```bash
# 1) 下载权重
python utils/download_weights.py
```

#### 1) [KDLAE](./KDLAE/README_ZH.md) 去噪

过jypyter直接运行。
  1. 打开 `KDLAE/KDLAE_T.ipynb` 或 `KDLAE/KDLAE-S.ipynb`。
  2. 在开头单元格中按需配置数据集路径、输出目录、训练/推理参数。
  3. 依次运行全部单元格，完成训练或推理与可视化。

#### 2) [ASDQE](./ASDQE/README_EN.md) 图像质量评价

`ASDQE/ASDQE_test.py` 提供无参考质量评价的测试入口。

请查看脚本头部与函数注释，填写或修改输入目录、文件后缀、批处理设置等参数，以评估原始/去噪结果的客观指标。

## 引用

若本项目对你的研究或产品有帮助，请在文档或论文中引用致谢： 论文录用后更新

### 许可与致谢

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



