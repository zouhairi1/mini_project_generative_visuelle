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
