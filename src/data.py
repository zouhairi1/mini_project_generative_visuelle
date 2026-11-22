"""DermaMNIST loading, normalisation and data loaders."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .utils import DATA_DIR

CLASS_NAMES = [
    "actinic keratoses",
    "basal cell carcinoma",
    "benign keratosis",
    "dermatofibroma",
    "melanoma",
    "melanocytic nevi",
    "vascular lesions",
]
SHORT_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
SPLITS = ("train", "val", "test")


def load_split(split: str, size: int = 64, root=DATA_DIR):
    """Return (images uint8 (N, H, W, 3), labels int64 (N,)) for one official split."""
    import medmnist

    root = str(root)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ds = medmnist.DermaMNIST(split=split, size=size, root=root, download=True)
    return ds.imgs, ds.labels.squeeze(1).astype(np.int64)


def class_counts(labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels, minlength=NUM_CLASSES)


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 (N, H, W, 3) -> float (N, 3, H, W) in [-1, 1]."""
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    return x * 2.0 - 1.0


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """float (N, 3, H, W) in [-1, 1] -> uint8 (N, H, W, 3)."""
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte()
    return x.permute(0, 2, 3, 1).cpu().numpy()


class ArrayDataset(Dataset):
    """In-memory dataset; optional random flips for training."""

    def __init__(self, images: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.x = to_tensor(images)
        self.y = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = x.flip(-1)
            if torch.rand(1).item() < 0.5:
                x = x.flip(-2)
        return x, self.y[i]


def make_loader(images, labels, batch_size=128, train=False, oversample=False, seed=0,
                power=None, drop_last=False):
    """DataLoader with num_workers=0.

    power: class re-weighting of the sampler, weight of an image = count(class) ** -power
           (0 or None = natural frequencies, 0.5 = sqrt-balanced, 1 = every class equally often).
    oversample=True is the same as power=1 (used in S2).
    """
    ds = ArrayDataset(images, labels, augment=train)
    gen = torch.Generator().manual_seed(seed)
    if oversample:
        power = 1.0
    if train and power:
        weights = class_counts(labels)[labels].astype(np.float64) ** (-power)
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double), num_samples=len(labels),
            replacement=True, generator=gen,
        )
        return DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=0, drop_last=drop_last)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, generator=gen, num_workers=0,
                      drop_last=drop_last)


def sample_per_class(images, labels, n, seed=0):
    """n random images of each class, ordered class by class -> (tensor in [-1, 1], labels)."""
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(NUM_CLASSES):
        candidates = np.flatnonzero(labels == c)
        idx.extend(rng.choice(candidates, size=n, replace=len(candidates) < n))
    idx = np.asarray(idx)
    return to_tensor(images[idx]), torch.from_numpy(labels[idx])
