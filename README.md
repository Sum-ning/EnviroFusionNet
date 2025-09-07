# EnviroFusionNet

EnviroFusionNet is a **multimodal gas recognition** framework that integrates:
- **Image encoder (ResNet-like)** for spatial features from sensor-to-image transforms
- **Sequence encoder (Transformer)** for temporal features from raw sensor sequences
- **Environment encoder (MLP)** for temperature / humidity / wind speed, etc.
- **CAFF** (Cross-Attention Fusion Framework) to fuse image & sequence features
- **CMAC** (Cross-Modal Attention Compensation) to compensate with environment features

> Output: 10-class gas classification (extendable).

---

## 📦 Project Structure

```
EnviroFusionNet/
├── models/
│   ├── image_encoder.py
│   ├── sequence_encoder.py
│   ├── environment_encoder.py
│   ├── caff_module.py
│   ├── cmac_module.py
│   └── fusion_model.py           # EnviroFusionNet
├── preprocess/
│   ├── gas_data_to_images.py     # convert raw sensor to images
│   └── gas_data_to_sequence.py   # convert raw sensor to (T×C) sequences
├── weights/
│   └── envirofusionnet_best.pth  # (optional) trained model
├── main.py                       # train / eval / visualize (curves & confusion matrix)
└── tsne_visualization.py         # t-SNE feature visualization tool
```

---

## 🔧 Installation

```bash
# Python 3.9+ recommended
pip install -r requirements.txt
```

---

## 🧪 Dataset

This repo uses an **open dataset** of gas sensor arrays in open sampling settings (link to the official source).
Please download the original files from the dataset website and convert them into **images** and **sequences** by the scripts in `preprocess/`.

- Image conversion: `preprocess/gas_data_to_images.py`
- Sequence conversion: `preprocess/gas_data_to_sequence.py`

You will prepare a **manifest CSV** to drive the training pipeline:

```
id,image_path,sequence_path,env_path,label
000001,images/000001.png,seq/000001.npy,env/000001.csv,0
000002,images/000002.png,seq/000002.npy,,1
...
```

- `image_path` / `sequence_path` / `env_path` can be **absolute** or **relative** to the CSV directory.
- If `env_path` is blank, a zero vector will be used for environment features.

---

## 🚀 Train / Evaluate

```bash
python main.py   --train_manifest path/to/train.csv   --val_manifest   path/to/val.csv   --test_manifest  path/to/test.csv   --weights_dir    ./weights   --batch_size 32 --epochs 30 --lr 1e-3 --num_classes 10
```

Artifacts written to `weights/`:
- `envirofusionnet_best.pth` – best checkpoint (highest val acc)
- `training_curves.png` – loss curves
- `confusion_matrix.png` – test confusion matrix
- `history.json` – loss/acc history

---

## 🎨 Feature Visualization (t‑SNE)

```bash
python tsne_visualization.py   --features feats/envirofusionnet.npy feats/resnet.npy   --labels   labels.npy   --names    class_names.txt   --out      figures/tsne_compare.png   --title    "EnviroFusionNet vs. ResNet"   --perplexity 30 --n_iter 1500 --dpi 600
```

- Supports `.npy` and `.pt/.pth`. If dict, will try keys: `features / feats / embeddings`.

---

## ⚙️ Key Modules

- `ImageEncoder` – simplified ResNet backbone producing spatial features.
- `SequenceEncoder` – Transformer encoder producing temporal features.
- `EnvironmentEncoder` – MLP extracting environment-aware features.
- `CAFF` – cross-attention fusion of image & sequence features.
- `CMAC` – cross‑modal attention compensation with environment features.
- `EnviroFusionNet` – full model in `models/fusion_model.py`.

---

## 📑 Requirements

See [`requirements.txt`](requirements.txt). Main packages:
`torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `Pillow`, `tqdm`.

---

## 📄 License

Choose an OSI license you prefer (e.g., MIT or Apache-2.0), then replace this section.

---

## 🙌 Acknowledgements

- Gas sensor arrays dataset (add the official link here).
- This implementation follows your research design with CAFF & CMAC modules for multimodal fusion.
