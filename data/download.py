"""Download DermaMNIST (MedMNIST+) into this folder. The data itself is not committed.

Usage (from the repository root):
    python data/download.py            # 64 x 64 (default)
    python data/download.py --size 128
"""
import argparse
from pathlib import Path

import medmnist
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=64, choices=[28, 64, 128, 224])
    args = parser.parse_args()

    for split in ("train", "val", "test"):
        ds = medmnist.DermaMNIST(split=split, size=args.size, root=str(HERE), download=True)
        counts = np.bincount(ds.labels.squeeze(1), minlength=7)
        print(f"{split:5s} {ds.imgs.shape}  per class: {counts.tolist()}")


if __name__ == "__main__":
    main()
