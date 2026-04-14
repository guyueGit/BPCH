# BPCH: Bidirectional Pyramid Multi-Scale Collaborative Hashing for Cross-Modal Retrieval

Official PyTorch implementation of our **IEEE Transactions on Multimedia (TMM)** paper.

[![Paper](https://img.shields.io/badge/Paper-IEEE--TMM-blue.svg)](Your_Link_Here)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📸 Framework Overview

![Methodology](./figures/method.png)

**BPCH** reframes multi-scale hashing as a collaborative learning process. It breaks the "scale-isolated" bottleneck through:
- **Bidirectional Pyramid Architecture (BPA)**: Enabling synergy between low-bit (semantic anchors) and high-bit (detail descriptors) codes.
- **Attention-Gated Cross-Fusion (AGCF)**: Acting as an adaptive quantization noise filter to suppress error propagation.
- **One-Pass Generation**: Synchronously optimizing 8, 16, 32, 64, and 128-bit hash codes in a single forward pass.

---

## ✨ Main Results (mAP)

Performance comparison on three major benchmarks:

| Dataset | 16-bit | 32-bit | 64-bit |
| :--- | :---: | :---: | :---: |
| **MIRFLICKR-25K** | 0.8901 | 0.9030 | 0.9146 |
| **NUS-WIDE** | 0.8062 | 0.8193 | 0.8352 |
| **MS COCO** | 0.7478 | 0.7961 | 0.8322 |

---

## 🛠️ Environment Setup

### 1. Requirements
- Python 3.8+
- PyTorch 1.12.1+ (CUDA 11.3+)
- `pip install transformers timm scikit-learn scipy tqdm pytorch-metric-learning`

### 2. Pre-trained Weights
- Download the pre-trained **ViT-B-32** weights (CLIP) and place it in the root directory:
  - `ViT-B-32.pt`

---

## 📂 Data Preparation

Please organize your datasets as follows in the `./dataset/` folder:

```text
dataset/
├── nuswide/
│   ├── index.mat
│   ├── caption.txt
│   └── label.mat
├── mirflickr/
└── mscoco/
