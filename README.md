# BPCH: Bidirectional Pyramid Multi-Scale Collaborative Hashing for Cross-Modal Retrieval

---

## 🛠️ 1. Environment Configuration

This project is tested on Ubuntu 20.04, PyTorch 1.12.1, and CUDA 11.3.

### Step-by-Step Setup
```bash
# Create a virtual environment
conda create -n bpch python=3.8 -y
conda activate bpch

# Install PyTorch and Torchvision
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install dependencies from the requirements file
pip install -r requirements.txt
```

---

## 📊 2. Data Preparation

### Download Original Data
You need to download the original data from the following sources:
- **MS COCO**: [COCO 2017](https://cocodataset.org/) (Include 2017 train, val, and annotations).
- **NUS-WIDE**: [Google Drive](https://drive.google.com/drive/folders/0B7IC9986m6R5flZ3SExueXpYLUU?resourcekey=0-V78W_9-P2P3p9p8Z4Z9Q).
- **MIRFLICKR-25K**: [Baidu Cloud](https://pan.baidu.com/s/1pL9u9e1) (Extraction Code: `u9e1`) or [Google Drive](https://drive.google.com/drive/folders/0B7IC9986m6R5fllGZ3SExueXpYLUU?resourcekey=0-V78W_9-P2P3p9p8Z4Z9Q).

### Generate Processed Files
Use the provided scripts in the `data/` directory (e.g., `make_coco.py`) to generate the required `.mat` files. After processing, organize the `dataset` directory as follows:

```text
dataset
├── base.py
├── __init__.py
├── dataloader.py
├── coco
│   ├── caption.mat 
│   ├── index.mat
│   └── label.mat 
├── flickr25k
│   ├── caption.mat
│   ├── index.mat
│   └── label.mat
└── nuswide
    ├── caption.txt  # Note: NUS-WIDE uses a .txt file
    ├── index.mat 
    └── label.mat
```

---

## 📥 3. Download CLIP Pre-trained Model

This code is based on the **ViT-B/32** backbone. 
1. Download `ViT-B-32.pt` from the official CLIP repository or [OpenAI's CDN](https://github.com/openai/CLIP).
2. Copy the `ViT-B-32.pt` file to the root directory of this project.

---

## 🏋️ 4. Training

BPCH supports **joint multi-scale optimization**. Run the following command to train on your target dataset (e.g., MS COCO):

```bash
python main.py --is-train True \
               --dataset coco \
               --caption-file caption.mat \
               --index-file index.mat \
               --label-file label.mat \
               --lr 0.001 \
               --output-dim 64 \
               --save-dir ./result/coco/64 \
               --clip-path ./ViT-B-32.pt \
               --batch-size 128 \
               --epochs 50
```

*Note: For NUS-WIDE, ensure you set `--caption-file caption.txt`.*

---

## 🔍 5. Testing & Evaluation

To evaluate a trained model and output mAP for all scales (16, 32, 64 bits):

```bash
python main.py --is-train False \
               --dataset coco \
               --pretrained ./result/coco/64/best_model_scales/64bits-best.mat \
               --output-dim 64 \
               --save-dir ./result/test_results
```

The script will:
1. Load the binary codes from the `.mat` checkpoint.
2. Calculate and display mAP for **Image-to-Text (I2T)** and **Text-to-Image (T2I)**.
3. Export hash codes into the `test_results` folder.

---

## 📁 Project Structure

```text
BPCH/
├── dataset/             # Data loading and processing
├── model/               # BPCH Architecture (BPA, WLM, AGCF)
├── utils/               # Loss functions (CPF, Triplet) and evaluation metrics
├── AdaTriplet/          # Metric learning supporting modules
├── unified_config.py    # Global hyper-parameters
├── train_trainer.py     # Training and validation logic
├── main.py              # Main entry point
└── requirements.txt     # Python dependencies
```


## 🙏 Acknowledgements

This project is built upon the following works:
- [CLIP](https://github.com/openai/CLIP)
- [DScPH (Deep Semantic-consistent Penalizing Hashing)](https://github.com/QinQibing/DScPH)

