## English

This folder contains training code for KDLAE and ASDQE.

- Quick navigation: [中文](README_ZH.md)

### Overview
This directory contains scripts, examples, and notes for training and testing the KDLAE and ASDQE models. If you want to run or reproduce experiments, start with `TRAINING_GUIDE.md` in this folder.

### Dependencies
- Python 3.8+ (use a virtualenv or conda)
- Install base requirements from the repository root:
  ```bash
  pip install -r ../requirements.txt
  ```

Note: Training typically requires CUDA and a matching PyTorch build—see `requirements.txt` and `TRAINING_GUIDE.md` for details.

### Quick start
1. Prepare datasets.

<figure>
        <img src="https://github.com/yangtaihong59/Rethink_Acoustic_Image_Enhancement/releases/download/gif/SFLS.gif" alt ="Audio Art" >
        <figcaption>
            Datasets Download: <a href="https://www.kaggle.com/datasets/taihongyang59/quantitative-simulated-forward-looking-sonar/data">[SFLS-Q]</a>
        </figcaption>
</figure>

2. Review and edit training configs / scripts
    (for example `train.sh`, or `ASDQE.py`).
    Modify dataset paths inside the configuration files located at `Denoising/Options/paper202508/`.

3. Run the training script (example, adjust to your config):
   ```bash
   # KDLAE
   bash train.sh Denoising/Options/paper202508/KDLAES.yml 

   # ASDQE
   python ASDQE.py
   ```

### Related files
- `TRAINING_GUIDE.md`: training flow, configuration and FAQs.
- `ASDQE_model.py`, `ASDQE_test.py`: entry points for ASDQE training and testing (note similar files exist elsewhere—check paths).
- Top-level `ASDQE/` and `KDLAE/` folders contain model READMEs and pretrained weights (e.g. `ASDQE/README_EN.md`, `KDLAE/README_EN.md`).

### Acknowledgment
This code is based on [Restormer](https://github.com/swz30/Restormer), [BasicSR](https://github.com/xinntao/BasicSR) and [HINet](https://github.com/megvii-model/HINet).

---

If you want me to expand these READMEs with more detailed "training command examples / config templates / hyperparameter notes / FAQ" sections, I can add sample configs and a small verification script.
