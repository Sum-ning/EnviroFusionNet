# EnviroFusionNet

## 📖 Introduction

The electronic nose is an intelligent sensing system based on a gas sensor array and pattern recognition algorithms. In practical applications, external environmental factors such as temperature, humidity, and wind speed can significantly impact sensor responses, leading to a decrease in recognition performance. Most existing gas recognition methods focus on unimodal feature modeling and neglect the influence of environmental factors. To address these limitations, we propose EnviroFusionNet, a multimodal information fusion framework with environmental perception. This method introduces an attention mechanism into the gas recognition task by designing the Cross-Attention Feature Fusion (CAFF) module and the Cross-Modal Attention Compensation (CMAC) module. The fusion module efficiently integrates the image and sequence features of the gas to capture the spatio-temporal correlation of the data. The compensation module enables the primary modality gas data to adaptively aggregate key information from the auxiliary modality environment, thus realizing effective compensation of environmental factors.Our study demonstrates the role of multimodal fusion and environmental compensation in gas recognition.

EnviroFusionNet is a **multimodal gas recognition** framework that integrates:
- **Image encoder (ResNet-like)** for spatial features from sensor-to-image transforms
- **Sequence encoder (Transformer)** for temporal features from raw sensor sequences
- **Environment encoder (MLP)** for environmental features from temperature / humidity / wind speed /location infor
- **CAFF** (Cross-Attention Fusion Framework) to fuse image & sequence features
- **CMAC** (Cross-Modal Attention Compensation) to compensate with environment features

> Output: 10-class gas classification (extendable).

---

## 📊 Dataset

This project uses an **open gas sensor array dataset**.
This study uses the gas sensor arrays dataset in open sampling settings constructed by the BioCircuits Institute at the University of California, San Diego. To collect gas response data, the laboratory constructed a wind tunnel testbed, in which a gas source was placed at the far left end as the inlet, while a fan was installed at the opposite end as the outlet to drive the gas flow through the tunnel. Gas sensor arrays were positioned between the gas inlet and the fan outlet to capture the response characteristics of the gases. To minimize manual intervention, the testbed was controlled by a fully computerized device. Under the control of the software, the dataset collected 18,000 time series gas instances involving ten high-priority chemical gases (acetone, acetaldehyde, ammonia, butanol, ethylene, methane, methanol, carbon monoxide, benzene and toluene). The sensor array of the gas detection platform consists of nine identical modules, each with eight different metal oxide gas sensors, for a total of 72 gas sensors. The acquisition time for each gas sample is 260 seconds and the changes in the sensor resistance value are recorded at a frequency of 100 Hz during the acquisition period. These data contain the complete gas response process and provide the basis for converting the gas response waveforms into image data. Some environmental conditions were also set during the experiment, including three exhaust fan wind speeds (0.1 m/s, 0.21 m/s, 0.34 m/s), six acquisition positions (L1-L6), temperature and humidity information. This information is uniformly distributed in all gas samples, providing data support for the subsequent multimodal feature modeling and the implementation of environmental compensation strategies.

Steps:
1. Download the raw dataset from the official source(https://archive.ics.uci.edu/dataset/251/gas+sensor+arrays+in+open+sampling+settings).
2. Convert sensor signals to **images** using `preprocess/gas_data_to_images.py`.
3. Convert sensor signals to **sequences** using `preprocess/gas_data_to_sequence.py`.
4. Prepare a manifest CSV to drive the training pipeline:

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
