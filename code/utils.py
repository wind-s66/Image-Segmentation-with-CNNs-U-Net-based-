import csv
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(name):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            f'CUDA is unavailable in the current Python environment: {sys.executable}\n'
            f'PyTorch: {torch.__version__}; compiled CUDA: {torch.version.cuda}.\n'
            'Run this project with the Python interpreter where CUDA-enabled '
            'PyTorch is installed, and check the NVIDIA driver.'
        )
    return torch.device(name)


def dice_per_image(prediction, target, eps=1e-7):
    p, t = prediction.flatten(1), target.flatten(1)
    return (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)


def segmentation_loss(logits, target):
    return F.binary_cross_entropy_with_logits(logits, target) + 1 - dice_per_image(logits.sigmoid(), target).mean()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_history(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key in ('train_loss', 'val_loss'):
        axes[0].plot([r['epoch'] for r in rows], [r[key] for r in rows], label=key)
    axes[1].plot([r['epoch'] for r in rows], [r['train_dice'] for r in rows], label='train hard Dice')
    axes[1].plot([r['epoch'] for r in rows], [r['val_dice'] for r in rows], label='val hard Dice')
    axes[0].set_ylabel('BCE + Dice loss')
    axes[1].set_ylabel('Dice score')
    axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
