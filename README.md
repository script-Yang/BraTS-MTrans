# 🧠 Medical Image Reconstruction on BraTS Dataset

This repository provides an implementation for **medical image reconstruction** based on the BraTS dataset. The framework is adapted from [MTrans](https://github.com/chunmeifeng/MTrans), with support for both **single-GPU** and **multi-GPU** training.

---

## Installation

```bash
# 1. Clone the repository
git https://github.com/script-Yang/BraTS-MTrans.git
cd BraTS-MTrans

# 2. Create and activate environment (recommended)
conda create -n recon python=3.8 -y
conda activate recon

# 3. Install required packages
pip install -r requirements.txt
```

---

## Training Instructions

### Single-GPU Training

```bash
python train.py --experiment reconstruction_single
```

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 train.py
```

Adjust `CUDA_VISIBLE_DEVICES` and `--nproc_per_node` based on your setup.

---

## 📌 Acknowledgment
This project is based on [MTrans](https://github.com/chunmeifeng/MTrans). Please cite their work if used.

