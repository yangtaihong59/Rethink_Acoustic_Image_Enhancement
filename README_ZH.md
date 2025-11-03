简体中文 | [English](README.md)

---

## 项目说明

本项目是论文的开源代码库，包含去噪方法 KDLAE、声学图像质量评价 ASDQE、示例数据与环境依赖等。本文档为中文快速上手指南与导航。

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

若未在仓库中另行注明许可协议，则默认仅供学术研究交流使用。引用时请注明来源。如涉及第三方数据集与方法，请遵循其各自的许可条款与引用要求。


