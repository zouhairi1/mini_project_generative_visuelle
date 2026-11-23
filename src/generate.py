"""S4: generate fixed, reproducible synthetic sets, keep them separate from the real data, and mark them.

Rules (from the brief and impl.md):
- synthetic images are stored in their own files (data/synthetic/, not committed), flagged synthetic=True;
- they are only ever added to the TRAINING set; validation and test always come from load_split()
  and stay 100 % real (assert_real_only() checks it);
- every synthetic image shown outside the training arrays carries a "SYNTHETIC" banner (watermark()).
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .data import NUM_CLASSES, to_uint8
from .models import CVAE, CondGenerator
from .utils import CKPT_DIR, DATA_DIR

SYN_DIR = DATA_DIR / "synthetic"                     # inside data/, so ignored by Git
CHECKPOINTS = {"cvae": "cvae.pt", "cgan": "cgan_G.pt"}
S3_NOTEBOOKS = {"cvae": "03a_cvae.ipynb", "cgan": "03b_cgan.ipynb"}


def load_generator(name: str, z_dim: int = 64) -> CondGenerator:
    """Load the S3 generator: the CVAE decoder ('cvae') or the cGAN generator ('cgan'), on CPU, eval mode."""
    path = CKPT_DIR / CHECKPOINTS[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Extract the S3 results first: "
            f"python -m src.extract_results notebooks/{S3_NOTEBOOKS[name]}")
    state = torch.load(path, map_location="cpu")      # float16 weights; load_state_dict casts them to float32
    if name == "cvae":
        model = CVAE(z_dim, NUM_CLASSES)
        model.load_state_dict(state)
        generator = model.decoder
    else:
        generator = CondGenerator(z_dim, NUM_CLASSES)
        generator.load_state_dict(state)
    return generator.float().eval()


@torch.no_grad()
def generate_set(generator: CondGenerator, n_per_class: int, seed: int = 0, batch_size: int = 400):
    """n_per_class images of every class -> (uint8 images (N, 64, 64, 3), int64 labels (N,)).

    Deterministic: z is drawn from a seeded CPU generator and the network runs on CPU,
    so the same seed always gives the same images.
    """
    g = torch.Generator().manual_seed(seed)
    generator = generator.cpu().eval()
    images, labels = [], []
    for c in range(NUM_CLASSES):
        z = torch.randn(n_per_class, generator.z_dim, generator=g)
        y = torch.full((n_per_class,), c, dtype=torch.long)
        for i in range(0, n_per_class, batch_size):
            images.append(to_uint8(generator(z[i:i + batch_size], y[i:i + batch_size])))
        labels.append(y.numpy())
    return np.concatenate(images), np.concatenate(labels).astype(np.int64)


def synthetic_path(name: str, n_per_class: int, seed: int) -> Path:
    return SYN_DIR / f"{name}_n{n_per_class}_seed{seed}.npz"


def save_synthetic(images, labels, name: str, n_per_class: int, seed: int) -> Path:
    """Save one synthetic set in its own compressed file, flagged as synthetic."""
    SYN_DIR.mkdir(parents=True, exist_ok=True)
    path = synthetic_path(name, n_per_class, seed)
    np.savez_compressed(path, images=images, labels=labels,
                        synthetic=np.ones(len(labels), dtype=bool), model=name, seed=seed)
    return path


def load_synthetic(path):
    """Load a synthetic set -> (images, labels). Refuses files that are not flagged synthetic."""
    with np.load(path) as d:
        assert d["synthetic"].all(), f"{path} is not flagged as synthetic"
        return d["images"], d["labels"]


def image_hashes(images) -> set:
    return {hashlib.md5(np.ascontiguousarray(im).tobytes()).hexdigest() for im in images}


def assert_real_only(images, synthetic_images, name: str = "set") -> None:
    """Fail if any synthetic image appears (exactly) in `images` (used on val and test)."""
    overlap = image_hashes(images) & image_hashes(synthetic_images)
    assert not overlap, f"{len(overlap)} synthetic images found in {name}!"
    print(f"OK: {name} contains no synthetic image ({len(images)} images checked)")


def watermark(image: np.ndarray, scale: int = 4, text: str = "SYNTHETIC") -> Image.Image:
    """uint8 image (H, W, 3) -> PIL image enlarged `scale` times, with a red banner saying SYNTHETIC."""
    im = Image.fromarray(image).resize((image.shape[1] * scale, image.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(im, "RGBA")
    w, h = im.size
    band = max(14, h // 7)
    draw.rectangle([0, h - band, w, h], fill=(200, 0, 0, 200))
    try:
        font = ImageFont.load_default(size=int(band * 0.7))
    except TypeError:                                   # Pillow < 10.1
        font = ImageFont.load_default()
    draw.text(((w - draw.textlength(text, font=font)) / 2, h - band + band * 0.12), text,
              fill=(255, 255, 255, 255), font=font)
    return im
