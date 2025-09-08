# EnviroFusionNet

## 📖 Introduction

EnviroFusionNet is a **multimodal gas recognition** framework that integrates:
- **Image encoder (ResNet-like)** for spatial features from sensor-to-image transforms
- **Sequence encoder (Transformer)** for temporal features from raw sensor sequences
- **Environment encoder (MLP)** for temperature / humidity / wind speed
- **CAFF** (Cross-Attention Fusion Framework) to fuse image & sequence features
- **CMAC** (Cross-Modal Attention Compensation) to compensate with environment features

> Output: 10-class gas classification (extendable).

---

## 📊 Dataset

This project uses an **open gas sensor array dataset**.  
Steps:
1. Download the raw dataset from the official source.
2. Convert sensor signals to **images** using `preprocess/gas_data_to_images.py`.
3. Convert sensor signals to **sequences** using `preprocess/gas_data_to_sequence.py`.
4. Prepare a manifest CSV to drive the training pipeline:

```csv
id,image_path,sequence_path,env_path,label
000001,images/000001.png,seq/000001.npy,env/000001.csv,0
000002,images/000002.png,seq/000002.npy,,1
```

- `image_path`, `sequence_path`, `env_path` can be relative or absolute.
- If `env_path` is blank, environment features default to zeros.

---

## 📂 Project Structure

```
EnviroFusionNet/
├── models/
│   ├── image_encoder.py
│   ├── sequence_encoder.py
│   ├── environment_encoder.py
│   ├── caff_module.py
│   ├── cmac_module.py
│   └── fusion_model.py          # EnviroFusionNet
├── preprocess/
│   ├── gas_data_to_images.py    # convert raw sensor to images
│   └── gas_data_to_sequence.py  # convert raw sensor to (T×C) sequences
├── weights/
│   └── envirofusionnet_best.pth # (optional) trained model
├── main.py                      # train / eval / visualize
└── tsne_visualization.py        # t-SNE feature visualization
```

---

## ⚙️ Requirements

```txt
numpy==1.23.5
scipy==1.10.1
pandas==1.5.3
matplotlib==3.7.1
seaborn==0.12.2
torch==1.13.1
torchvision==0.14.1
tqdm==4.65.0
scikit-learn==1.2.2
Pillow==9.4.0
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 🚀 Train / Evaluate / Feature Visualization

### Train & Evaluate
```bash
python main.py   --train_manifest path/to/train.csv   --val_manifest   path/to/val.csv   --test_manifest  path/to/test.csv   --weights_dir    ./weights   --batch_size 32 --epochs 30 --lr 1e-3 --num_classes 10
```

Outputs:
- `envirofusionnet_best.pth` – best checkpoint
- `training_curves.png` – loss curves
- `confusion_matrix.png` – confusion matrix
- `history.json` – training logs

### Feature Visualization (t-SNE)
```bash
python tsne_visualization.py   --features feats/envirofusionnet.npy feats/resnet.npy   --labels   labels.npy   --names    class_names.txt   --out      figures/tsne_compare.png   --title    "EnviroFusionNet vs. ResNet"
```
---
