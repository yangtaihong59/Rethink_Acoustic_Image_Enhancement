## ASDQE — Acoustic-Specific Denoise Quality Evaluator


简体中文 | [English](README_EN.md)

---

### 简介 
ASDQE 是一个用于比较不同声学图像去噪方法效果的质量评估工具。它基于一个轻量的深度模型，输入一对图像：含噪低质量图像 (LQ) 与对应的高质量/去噪图像 (GT)，输出一个用于比较的方法分数，并汇总统计指标（均值、标准差、分位数等），便于不同方法横向对比。

本仓库包含：
- `ASDQE_model.py`：模型结构（包含特征提取与简化 UNet 增强、回归头）。
- `ASDQE_test.py`：评估脚本，加载成对图像并对多种去噪结果进行批量推理与统计对比。
- `weights/ASDQE.pth`：预训练权重文件。


---

### 目录结构
```text
ASDQE/
├─ ASDQE_model.py         # 模型定义 / Model definition
├─ ASDQE_test.py          # 评估脚本 / Evaluation script
└─ weights/
   └─ ASDQE.pth           # 预训练权重 / Pretrained weights
```

---

### 快速开始

1) 克隆或进入本目录后，确认权重文件存在：`weights/ASDQE.pth`。

2) 在 `ASDQE_test.py` 内配置数据路径：
   - `base_lg_dir`：低质量(含噪)图像目录
   - `denoise_dir`：去噪结果根目录
   - `denoising_methods`：方法名到各自去噪子目录的映射（示例已给出，可自行增删）

3) 运行评估脚本:
```bash
python ASDQE_test.py
```

脚本将：
- 加载模型与数据
- 对每种方法进行逐对推理
- 输出每种方法的统计指标
- 生成 `stats_transposed.csv`（行：统计项；列：方法）
---

### 路径与数据组织
评估脚本假设：
- `base_lg_dir` 下是低质量图像文件（如 `0001.jpg`、`0002.jpg` ...）
- 每个去噪方法目录中有与 `base_lg_dir` 同名的文件（文件名一一对应），例如：
  - `${denoise_dir}/Teacher/0001.jpg`
  - `${denoise_dir}/Teacher/0002.jpg`

若数量或文件名不匹配，脚本会抛出错误以提醒检查数据组织。

---

### 结果解读
脚本打印每个方法的统计汇总（`mean`, `std`, `min`, `25%`, `50%`, `75%`, `max`）。数值越大是否更优取决于训练目标（本工程中的回归输出范围在 [-1, 1]，其绝对意义需结合具体训练定义与下游标尺）。建议：
- 在相同数据集上对多种方法进行相对比较
- 结合视觉检验或客观指标做交叉验证


---

### 引用
若本项目对你的研究或产品有帮助，请在文档或论文中致谢： 论文录用后更新

---


