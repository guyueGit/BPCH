# BPCH: Bidirectional Pyramid Multi-Scale Collaborative Hashing for Cross-Modal Retrieval

This is the official PyTorch implementation of the paper: **"BPCH: Bidirectional Pyramid Multi-Scale Collaborative Hashing for Cross-Modal Retrieval"** (Submitted to IEEE Transactions on Multimedia).

BPCH addresses the "scale-isolated" bottleneck in cross-modal hashing by reframing multi-scale hash code generation as a collaborative learning process. It features a Bidirectional Pyramid Architecture (BPA), Weighted Level Module (WLM), and Attention-Gated Cross-Fusion (AGCF).

---

## 🛠️ Step-by-Step Installation Guide

Follow these steps to set up the environment and run the code from scratch.

### 1. Clone the Repository
```bash
git clone https://github.com/YourUsername/BPCH.git
cd BPCH
```

### 2. Environment Configuration
We recommend using **Conda** for environment management.
```bash
# Create a virtual environment
conda create -n bpch python=3.8 -y
conda activate bpch

# Install PyTorch and Torchvision (Tested on CUDA 11.3)
# Note: Adjust the cuXXX version according to your local machine
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
pip install transformers timm scikit-learn scipy tqdm h5py matplotlib pytorch-metric-learning
```

### 3. Download Pre-trained Backbones
The model uses CLIP (ViT-B/32) as the visual backbone.
1. Download the pre-trained weight `ViT-B-32.pt` from the [Official OpenAI CLIP Repository](https://github.com/openai/CLIP).
2. Place the `ViT-B-32.pt` file into the **root directory** of the project.

---

## 📂 Data Preparation

We support three major benchmarks: **MIRFLICKR-25K**, **NUS-WIDE**, and **MS COCO**.

### 1. Dataset Organization
Create a `dataset` directory and organize the files as follows (using `nuswide` as an example):

```text
BPCH/
├── dataset/
│   └── nuswide/
│       ├── index.mat     # Image path indices
│       ├── label.mat     # Binary semantic labels (N x num_classes)
│       └── caption.txt   # Raw text descriptions (one line per sample)
├── main.py
├── unified_config.py
└── ...
```

### 2. File Requirements
- `.mat` files should be compatible with `scipy.io.loadmat`.
- The number of samples across `index.mat`, `label.mat`, and `caption.txt` must be identical.

---

## 🏋️ Training and Optimization

BPCH utilizes a **Joint Training Paradigm**, optimizing 8, 16, 32, 64, and 128-bit hash codes simultaneously in a single forward pass.

To start training on the **NUS-WIDE** dataset:

```bash
python main.py --dataset nuswide \
               --numclass 21 \
               --is-train True \
               --output-dim 64 \
               --batch-size 128 \
               --lr 0.001 \
               --clip-lr 1e-5 \
               --epochs 50 \
               --save-dir ./result
```

**Key Arguments:**
- `--is-train True`: Set to training mode.
- `--output-dim`: Specifies the primary bit-length for logging (all scales are optimized jointly).
- `--clip-path`: Path to `ViT-B-32.pt` (default is root).
- `--save-dir`: Directory to save checkpoints and logs.

---

## 🔍 Evaluation and Testing

To evaluate a trained model and generate hash codes, set `--is-train` to `False` and specify the checkpoint path.

```bash
python main.py --dataset nuswide \
               --is-train False \
               --pretrained ./result/best_model_scales/64bits-best.mat \
               --output-dim 64 \
               --query-num 5000 \
               --train-num 10000
```

**Output Explanation:**
1. **Console Output:** The script will display the **mAP** scores for both Image-to-Text (I2T) and Text-to-Image (T2I) retrieval tasks.
2. **Hash Codes:** Generated binary codes for all scales will be saved as `.mat` files in `./result/test_results/`.
3. **Best Models:** The multi-scale hash codes for the best performing epoch are saved in `./result/best_model_scales/`.

---

## 📁 Project Structure

```text
BPCH/
├── dataset/             # Data storage directory
├── model/               # Model definitions (DSPH, BPA, WLM, AGCF)
├── utils/               # Loss functions (CPF, bit_var, etc.) and metrics
├── AdaTriplet/          # Triplet mining and loss modules
├── unified_config.py    # Global hyperparameter configuration
├── train_trainer.py     # Main training and validation logic
├── main.py              # Entry point for execution
└── ViT-B-32.pt          # Pre-trained visual backbone (to be downloaded)
```

---

## 📝 Citation

If you find this work or code useful for your research, please cite:

```bibtex
@article{hu2026bpch,
  title={BPCH: Bidirectional Pyramid Multi-Scale Collaborative Hashing for Cross-Modal Retrieval},
  author={Hu, Rui and Zhang, Li and Wu, Xiangqian},
  journal={IEEE Transactions on Multimedia},
  year={2026}
}
```

---

## 📧 Contact
For any questions regarding the code or paper, please contact:
**Rui Hu** (2022112776@stu.hit.edu.cn)
Harbin Institute of Technology, China.
```
