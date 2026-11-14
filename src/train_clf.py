"""Classifier training and evaluation (precision, recall, F1, confusion matrix)."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .data import NUM_CLASSES, class_counts, make_loader
from .models import SmallCNN
from .utils import set_seed


def compute_metrics(y_true, y_pred, n_classes: int = NUM_CLASSES) -> dict:
    """Per-class and macro precision / recall / F1, plus the confusion matrix."""
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "precision": p.tolist(),
        "recall": r.tolist(),
        "f1": f.tolist(),
        "support": s.tolist(),
        "macro_precision": float(p.mean()),
        "macro_recall": float(r.mean()),
        "macro_f1": float(f.mean()),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, preds = [], []
    for x, y in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


def evaluate(model, images, labels, device, batch_size=256) -> dict:
    loader = make_loader(images, labels, batch_size=batch_size, train=False)
    y_true, y_pred = predict(model, loader, device)
    return compute_metrics(y_true, y_pred)


def train_classifier(train_x, train_y, val_x, val_y, cfg: dict, seed: int = 0,
                     strategy: str = "none", device=None, verbose: bool = True):
    """Train SmallCNN; keep the epoch with the best validation macro-F1.

    strategy: 'none' | 'weighted' (inverse-frequency loss weights) | 'oversample' (balanced sampler).
    Returns (model with best weights, history list of dicts).
    """
    assert strategy in ("none", "weighted", "oversample"), strategy
    set_seed(seed)
    device = device or torch.device("cpu")
    tc, mc = cfg["train"], cfg["model"]

    model = SmallCNN(NUM_CLASSES, mc["channels"], mc["dropout"]).to(device)
    loader = make_loader(train_x, train_y, cfg["data"]["batch_size"], train=True,
                         oversample=(strategy == "oversample"), seed=seed)

    weight = None
    if strategy == "weighted":
        counts = class_counts(train_y)
        weight = torch.tensor(len(train_y) / (NUM_CLASSES * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])

    rare = cfg.get("rare_class", 3)
    best_f1, best_state, history = -1.0, None, []
    for epoch in range(1, tc["epochs"] + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        scheduler.step()

        val = evaluate(model, val_x, val_y, device)
        row = {"epoch": epoch, "train_loss": total / n, "val_macro_f1": val["macro_f1"],
               "val_macro_recall": val["macro_recall"], "val_rare_recall": val["recall"][rare],
               "val_rare_f1": val["f1"][rare]}
        history.append(row)
        if val["macro_f1"] > best_f1:
            best_f1, best_state = val["macro_f1"], copy.deepcopy(model.state_dict())
            row["best"] = True
        if verbose:
            print(f"[{strategy} | seed {seed}] epoch {epoch:2d}/{tc['epochs']} "
                  f"loss {row['train_loss']:.3f} | val macro-F1 {row['val_macro_f1']:.3f} "
                  f"| val df recall {row['val_rare_recall']:.2f} | {time.time() - t0:.0f}s"
                  + ("  *" if row.get("best") else ""))

    model.load_state_dict(best_state)
    return model, history


def summarize(values) -> str:
    v = np.asarray(values, dtype=float)
    return f"{v.mean():.3f} ± {v.std():.3f}"
