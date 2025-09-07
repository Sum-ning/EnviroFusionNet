"""
main.py — Training / Inference / Visualization for EnviroFusionNet

Expected data manifest CSV format (UTF-8):
------------------------------------------------
id,image_path,sequence_path,env_path,label
000001,images/000001.png,seq/000001.npy,env/000001.csv,0
000002,images/000002.png,seq/000002.npy,,1
...

Notes:
- Paths can be absolute or relative to the manifest's directory.
- If env_path is empty, a zero vector will be used.
- num_classes = 10 by default.
- Adjust image/sequence preprocessing to your own data.
"""

import os
import csv
import math
import time
import json
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --- Import your model modules ---
from models.fusion_model import EnviroFusionNet


# ===============
# Reproducibility
# ===============
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =======================
# Dataset / Data pipeline
# =======================
class GasMultiModalDataset(Dataset):
    """
    Dataset driven by a manifest CSV.
    Each row provides image_path, sequence_path (.npy), env_path (.csv optional), and integer label.
    """
    def __init__(self, manifest_path: str, transform_img=None, seq_pad_len: int = 260, seq_feat_dim: int = 72):
        self.manifest_path = Path(manifest_path)
        self.root_dir = self.manifest_path.parent
        self.rows = self._read_manifest(self.manifest_path)
        self.transform_img = transform_img
        self.seq_pad_len = seq_pad_len
        self.seq_feat_dim = seq_feat_dim

    @staticmethod
    def _read_manifest(csv_path: Path):
        rows = []
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required = {"id", "image_path", "sequence_path", "env_path", "label"}
            if not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"Manifest must contain columns: {required}, found: {reader.fieldnames}")
            for r in reader:
                rows.append(r)
        if len(rows) == 0:
            raise ValueError("Manifest is empty.")
        return rows

    def __len__(self):
        return len(self.rows)

    def _resolve(self, p: str) -> Path:
        if p is None or p == "":
            return Path("")
        pth = Path(p)
        if not pth.is_absolute():
            pth = (self.root_dir / pth).resolve()
        return pth

    def _load_image(self, path: Path) -> torch.Tensor:
        # Expect image with 9 channels stacked as 3x3 or single 9-channel PNG. If it's grayscale(=1ch) or RGB(=3ch),
        # we will replicate/adjust to 9 channels. You may customize here based on your pipeline.
        img = Image.open(path).convert("L")  # grayscale
        if self.transform_img is not None:
            img = self.transform_img(img)    # [1, H, W]
        else:
            img = transforms.ToTensor()(img) # [1, H, W]
        # Expand to 9 channels if needed
        if img.shape[0] == 1:
            img = img.repeat(9, 1, 1)  # [9, H, W]
        elif img.shape[0] == 3:
            # tile to 9 channels
            img = img.repeat(3, 1, 1)
        elif img.shape[0] != 9:
            # Fallback: pad or cut to 9 channels
            c = img.shape[0]
            if c > 9:
                img = img[:9]
            else:
                img = torch.cat([img, img.new_zeros(9 - c, *img.shape[1:])], dim=0)
        return img

    def _load_sequence(self, path: Path) -> torch.Tensor:
        arr = np.load(path)
        # Expect (T, C). Pad/trim to (seq_pad_len, seq_feat_dim)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, self.seq_feat_dim)
        T, C = arr.shape[0], arr.shape[1]
        # Trim/pad time
        if T >= self.seq_pad_len:
            arr = arr[: self.seq_pad_len, :]
        else:
            pad = np.zeros((self.seq_pad_len - T, C), dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)
        # Trim/pad features
        if C >= self.seq_feat_dim:
            arr = arr[:, : self.seq_feat_dim]
        else:
            padc = np.zeros((arr.shape[0], self.seq_feat_dim - C), dtype=arr.dtype)
            arr = np.concatenate([arr, padc], axis=1)
        return torch.from_numpy(arr).float()  # [T, C]

    def _load_env(self, path: Optional[Path]) -> torch.Tensor:
        # Expect a CSV with values in the first row. If not provided, returns zeros(3).
        if path is None or str(path) == "" or not path.exists():
            return torch.zeros(3, dtype=torch.float32)
        vals = []
        with path.open("r", encoding="utf-8") as f:
            rd = csv.reader(f)
            for row in rd:
                vals = [float(x) for x in row if x.strip() != ""]
                break
        if len(vals) == 0:
            return torch.zeros(3, dtype=torch.float32)
        return torch.tensor(vals, dtype=torch.float32)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img_path = self._resolve(r["image_path"])
        seq_path = self._resolve(r["sequence_path"])
        env_path = self._resolve(r["env_path"])
        label = int(r["label"])

        image = self._load_image(img_path)          # [9, H, W]
        sequence = self._load_sequence(seq_path)    # [T, C]
        env = self._load_env(env_path)              # [E]

        return {
            "id": r["id"],
            "image": image,
            "sequence": sequence,
            "env": env,
            "label": torch.tensor(label, dtype=torch.long)
        }


# ============
# Train / Eval
# ============
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        sequences = batch["sequence"].to(device)
        envs = batch["env"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images, sequences, envs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        running_acc += accuracy_from_logits(logits, labels) * labels.size(0)

    size = len(loader.dataset)
    return running_loss / size, running_acc / size


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    all_labels = []
    all_preds = []
    for batch in loader:
        images = batch["image"].to(device)
        sequences = batch["sequence"].to(device)
        envs = batch["env"].to(device)
        labels = batch["label"].to(device)

        logits = model(images, sequences, envs)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        running_acc += accuracy_from_logits(logits, labels) * labels.size(0)

        preds = torch.argmax(logits, dim=1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    size = len(loader.dataset)
    avg_loss = running_loss / size
    avg_acc = running_acc / size
    return avg_loss, avg_acc, np.array(all_labels), np.array(all_preds)


# ============
# Visualization
# ============
def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_curves(history, out_path):
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = plt.figure(figsize=(8, 4))
    # Loss
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# =====
# Main
# =====
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train EnviroFusionNet on gas recognition dataset")
    parser.add_argument("--train_manifest", type=str, required=True, help="Path to training manifest CSV")
    parser.add_argument("--val_manifest", type=str, required=True, help="Path to validation manifest CSV")
    parser.add_argument("--test_manifest", type=str, required=True, help="Path to test manifest CSV")
    parser.add_argument("--weights_dir", type=str, default="./weights", help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=48, help="H=W for image resize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)

    # Transforms for images
    img_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # Datasets and loaders
    train_ds = GasMultiModalDataset(args.train_manifest, transform_img=img_tf)
    val_ds = GasMultiModalDataset(args.val_manifest, transform_img=img_tf)
    test_ds = GasMultiModalDataset(args.test_manifest, transform_img=img_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    device = torch.device(args.device)
    model = EnviroFusionNet(num_classes=args.num_classes).to(device)

    # Optim / Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    os.makedirs(args.weights_dir, exist_ok=True)
    best_val_acc = -1.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[Epoch {epoch:03d}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f} train_acc={tr_acc:.4f} val_acc={val_acc:.4f}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = Path(args.weights_dir) / "envirofusionnet_best.pth"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_acc": val_acc}, best_path)
            print(f"  Saved new best checkpoint to: {best_path}")

    # Save curves
    plot_curves(history, Path(args.weights_dir) / "training_curves.png")
    with open(Path(args.weights_dir) / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    # Load best for testing
    best_path = Path(args.weights_dir) / "envirofusionnet_best.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint (epoch={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    # Test
    test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")
    print(classification_report(y_true, y_pred, digits=4))

    # Confusion matrix
    class_names = [str(i) for i in range(args.num_classes)]
    plot_confusion_matrix(y_true, y_pred, class_names, Path(args.weights_dir) / "confusion_matrix.png")


if __name__ == "__main__":
    main()

