## 中文

本部分是 KDLAE 及 ASDQE 的训练代码。

- 快速导航： [English](README_EN.md)

### 说明
本目录包含用于训练和测试 KDLAE 与 ASDQE 模型的脚本、示例与参考说明。若你只想运行或复现实验，请优先参阅本目录下的 `TRAINING_GUIDE.md`。

### 依赖
- Python 3.8+（建议使用虚拟环境或 conda 环境）
- 见仓库根目录的 `requirements.txt`：
  ```bash
  pip install -r ../requirements.txt
  ```

（注：实际训练可能需要 CUDA 与对应版本的 PyTorch，详情见 `requirements.txt` 与 `TRAINING_GUIDE.md`）

### 快速开始
1. 准备数据集。

<figure>
        <img src="https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/gif/SFLS.gif" alt ="Audio Art" >
        <figcaption>
            Datasets Download: <a href="https://www.kaggle.com/datasets/taihongyang59/quantitative-simulated-forward-looking-sonar/data">[SFLS-Q]</a>
        </figcaption>
</figure>

2. 查看并编辑训练配置 / 脚本
    （例如 `train.sh`、或 `ASDQE.py` 等）。
     修改路径Denoising/Options/paper202508/中配置文件的数据集路径

3. 运行训练脚本（示例，视具体配置而定）:
   ```bash
   # KDLAE
   bash train.sh Denoising/Options/paper202508/KDLAES.yml 

   # ASDQE
   python ASDQE.py
   ```

### 相关文件
- `TRAINING_GUIDE.md`：训练流程、配置与常见问题。
- `ASDQE_model.py`、`ASDQE_test.py`：ASDQE 的训练与测试入口（同名文件在仓库其他目录也存在，请注意路径）。
- 上层目录下的 `ASDQE/` 与 `KDLAE/`：各模型的详细说明与预训练权重（例如 `ASDQE/README_EN.md`、`KDLAE/README_EN.md`）。

### 致谢
本代码基于[Restomer](https://github.com/swz30/Restormer), [BasicSR](https://github.com/xinntao/BasicSR) 与 [HINet](https://github.com/megvii-model/HINet)。

