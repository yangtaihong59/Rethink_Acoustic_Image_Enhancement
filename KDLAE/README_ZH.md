# KDLAE: Knowledge-Distilled Lightweight Acoustic Enhancer

简体中文 | [English](README_EN.md)

---

## 简介
- **KDLAE-T (Teacher)**: 基于 Restormer 修改的教师网络，支持可控去噪强度与超分，用于高质量去噪。
- **KDLAE-S (Student)**: 轻量多帧去噪网络，采用 3D 卷积进行时空特征融合，面向实时/低资源场景。

本目录包含模型实现、权重与两个示例笔记本：`KDLAE_T.ipynb` 与 `KDLAE-S.ipynb`，可直接运行进行推理与对比。

---

## 目录结构
- `KDLAE_model.py`: 包含 `KDLAE_teacher`, `KDLAE_student` 与基础 `Restormer` 实现
- `weights/`: 预训练权重，例如 `KDLAE-S-FLS.pth`, `KDLAE-S-US.pth`
- `KDLAE_T.ipynb`: 教师模型推理/增强示例
- `KDLAE-S.ipynb`: 学生模型多帧去噪示例

---

## 预训练权重
将权重放在 `weights/` 下：
- 超声 (US) 示例：`weights/KDLAE-S-US.pth`
- 声纳/腹腔镜等 (FLS) 示例：`weights/KDLAE-S-FLS.pth`

> 若使用自定义权重，请修改笔记本或脚本中的权重路径。

---

## 快速开始

### KDLAE-T 教师模型推理
在 `KDLAE_T.ipynb` 中：
1) 准备输入影像与可选的 `denoise_rate`（控制去噪强度）
2) 运行推理，导出 `hq`（去噪）与可选 `sr`（增强）

### KDLAE-S 学生模型推理
在 `KDLAE-S.ipynb` 中：
1) 选择输入序列（多帧 PNG/JPG）
2) 加载相应权重（US/FLS）
3) 运行推理单元保存去噪结果

---

## 训练与蒸馏
- 教师模型：基于 `KDLAE_teacher` 与 Restormer 结构，支持 `denoise_rate` 控制强度。
- 学生模型：`KDLAE_student` 使用 3D UNet 风格编码-融合-解码，适合蒸馏落地。

说明：本仓库当前提供推理/对比示例。若需完整训练脚本与蒸馏流程，请在此基础上扩展数据加载、损失函数与调度器。

---

## 引用
若本项目对你的研究或产品有帮助，请在文档或论文中引用致谢： 论文录用后更新


