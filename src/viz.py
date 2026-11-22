"""Plotting helpers (sample grids, confusion matrices)."""
import matplotlib.pyplot as plt
import numpy as np

from .data import SHORT_NAMES


def plot_samples_per_class(images, labels, n_per_class=8, seed=0, title=None):
    """One row per class, n_per_class random images."""
    rng = np.random.default_rng(seed)
    classes = range(len(SHORT_NAMES))
    fig, axes = plt.subplots(len(SHORT_NAMES), n_per_class, figsize=(n_per_class * 1.2, len(SHORT_NAMES) * 1.25))
    for row, c in enumerate(classes):
        idx = np.flatnonzero(labels == c)
        pick = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        for col in range(n_per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(pick):
                ax.imshow(images[pick[col]])
        axes[row, 0].text(-8, 32, f"{c} {SHORT_NAMES[c]}", ha="right", va="center", fontsize=10)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(cm, ax=None, title="", normalize=True):
    """Rows = true class, columns = predicted class. Colour = row-normalised (recall); text = counts."""
    cm = np.asarray(cm)
    shown = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    n = len(SHORT_NAMES)
    ax.set_xticks(range(n), SHORT_NAMES, rotation=45)
    ax.set_yticks(range(n), SHORT_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if shown[i, j] > 0.5 else "black")
    return im


def tensor_grid(x, nrow: int, pad: int = 2) -> np.ndarray:
    """Tensor (N, 3, H, W) in [-1, 1] -> image grid (H', W', 3) in [0, 1] for imshow."""
    from torchvision.utils import make_grid

    g = make_grid((x.detach().cpu().float().clamp(-1, 1) + 1) / 2, nrow=nrow, padding=pad, pad_value=1.0)
    return g.permute(1, 2, 0).numpy()


def plot_grid(x, nrow: int, ax=None, title: str = "", row_labels=None, col_labels=None, figsize=None):
    """Show a grid of generated images with optional row / column labels."""
    size = x.shape[-1]
    step = size + 2
    if ax is None:
        n_rows = int(np.ceil(len(x) / nrow))
        _, ax = plt.subplots(figsize=figsize or (nrow * 0.95 + 1, n_rows * 0.95 + 0.7))
    ax.imshow(tensor_grid(x, nrow))
    ax.set_xticks([2 + step * j + size / 2 for j in range(len(col_labels))] if col_labels else [], col_labels or [])
    ax.set_yticks([2 + step * i + size / 2 for i in range(len(row_labels))] if row_labels else [], row_labels or [])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    return ax


def plot_class_grid(x, n_per_class: int, ax=None, title: str = ""):
    """Images ordered class by class (n_per_class each): one row per class."""
    return plot_grid(x, n_per_class, ax=ax, title=title, row_labels=SHORT_NAMES)
