"""
tsne_visualization.py

通用 t-SNE 可视化脚本：从特征文件与标签文件中生成 2D 可视化图片，支持多模型对比。

输入：
- 特征文件（.npy 或 .pt/.pth）形状为 (N, D)；
- 标签文件（.npy 或 .pt/.pth）形状为 (N,)；
- 可选类别名称文件（.txt，每行一个类名）；

用法：
python tsne_visualization.py \
  --features paths/modelA_feats.npy paths/modelB_feats.npy \
  --labels  path/to/labels.npy \
  --names   path/to/class_names.txt \
  --out     ./figures/tsne_compare.png \
  --title   "EnviroFusionNet vs Baselines" \
  --perplexity 30 --n_iter 1500 --dpi 600

注意：
- 多模型对比时，--features 可传多个路径；同一标签用于所有模型的样本对齐。
- 若传入 .pt/.pth，将尝试读取 Tensor 或字典中的 'features' / 'feats' 键。
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_array(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.suffix.lower() == ".npy":
        arr = np.load(p, allow_pickle=True)
    elif p.suffix.lower() in [".pt", ".pth"]:
        obj = torch.load(p, map_location="cpu")
        if isinstance(obj, dict):
            for k in ["features", "feats", "embeddings", "X"]:
                if k in obj:
                    obj = obj[k]
                    break
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
        else:
            arr = np.asarray(obj)
    else:
        raise ValueError(f"Unsupported file type: {p.suffix}")
    return np.asarray(arr)


def load_names(path: str, num_classes: int) -> List[str]:
    if path is None:
        return [str(i) for i in range(num_classes)]
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    if len(names) < num_classes:
        names += [f"class_{i}" for i in range(len(names), num_classes)]
    return names[:num_classes]


def tsne_fit(features: np.ndarray, perplexity: float, n_iter: int, random_state: int) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, learning_rate="auto", init="pca",
                random_state=random_state, verbose=1)
    return tsne.fit_transform(features)


def plot_tsne(ax, tsne_2d: np.ndarray, labels: np.ndarray, class_names: List[str], title: str):
    num_classes = len(class_names)
    # 自动生成颜色
    cmap = plt.get_cmap("tab10") if num_classes <= 10 else plt.get_cmap("tab20")
    for c in range(num_classes):
        idx = labels == c
        ax.scatter(tsne_2d[idx, 0], tsne_2d[idx, 1], s=6, alpha=0.8, label=class_names[c])
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=2, fontsize=7, frameon=False, ncol=2)


def main():
    parser = argparse.ArgumentParser(description="t-SNE visualization for gas recognition features")
    parser.add_argument("--features", nargs="+", required=True, help="One or more feature files (.npy/.pt/.pth)")
    parser.add_argument("--labels", required=True, help="Labels file (.npy/.pt/.pth), shape (N,)")
    parser.add_argument("--names", default=None, help="Optional class names .txt (one per line)")
    parser.add_argument("--out", required=True, help="Output image path (e.g., ./figures/tsne.png)")
    parser.add_argument("--title", default="t-SNE Visualization")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n_iter", type=int, default=1500)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    # 加载数据
    feats_list = [load_array(p) for p in args.features]
    labels = load_array(args.labels).astype(int).reshape(-1)
    N = labels.shape[0]
    # 校验每个特征与标签长度一致
    for i, X in enumerate(feats_list):
        if X.shape[0] != N:
            raise ValueError(f"features[{i}] has N={X.shape[0]} but labels has N={N}")

    num_classes = int(labels.max() + 1)
    class_names = load_names(args.names, num_classes)

    # 画布：多模型对比 -> 多子图
    cols = min(3, len(feats_list))
    rows = math.ceil(len(feats_list) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, X in enumerate(feats_list):
        print(f"[t-SNE] model-{i}: X shape = {X.shape}")
        X2d = tsne_fit(X, args.perplexity, args.n_iter, args.random_state)
        title = args.title if len(feats_list) == 1 else f"{args.title} (Model {i+1})"
        plot_tsne(axes[i], X2d, labels, class_names, title)

    # 布局与保存
    plt.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", format=out_path.suffix.lstrip(".") or "png")
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
