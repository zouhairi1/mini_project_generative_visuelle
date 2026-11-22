"""Small helpers: paths, seeds, device, config, latent interpolation, exporting results from Colab."""
import base64
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml."""
    with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for d in (DATA_DIR, RESULTS_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- latent interpolation

def lerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Linear interpolation (AE/VAE course, slide 9). a, b: (1, d); t: (n, 1) in [0, 1]."""
    return (1 - t) * a + t * b


def slerp(a: torch.Tensor, b: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical interpolation (GAN course, slide 21): N(0, I) in high dimension lives near a sphere,
    which the straight line between two codes leaves."""
    cos = (a / a.norm(dim=-1, keepdim=True) * b / b.norm(dim=-1, keepdim=True)).sum(-1, keepdim=True)
    omega = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
    return (torch.sin((1 - t) * omega) * a + torch.sin(t * omega) * b) / torch.sin(omega)


# ----------------------------------------------------------------------------- checkpoints and export

def save_half(state_dict: dict, path) -> None:
    """Save weights in float16 (half the size). load_state_dict converts them back to float32."""
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in state_dict.items()}, path)


def in_colab() -> bool:
    return "google.colab" in sys.modules or Path("/content").exists()


def export_results(paths, zip_name: str) -> Path:
    """Zip result files (paths inside the repository) and bring them back from Colab.

    - Browser Colab: files.download() downloads the zip.
    - VS Code + Colab kernel: the zip is printed as base64 text; save the notebook (Ctrl+S) and run
      `python -m src.extract_results notebooks/<notebook>.ipynb` locally.
    """
    zip_path = ROOT / zip_name
    paths = [Path(p) for p in paths if Path(p).exists()]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.resolve().relative_to(ROOT))
    data = zip_path.read_bytes()
    print(f"Created {zip_path} ({len(data) / 1e6:.1f} MB, {len(paths)} files)")
    if in_colab():
        try:
            from google.colab import files
            files.download(str(zip_path))                 # works in the browser only
        except Exception:
            pass
        print(f"ZIP_BASE64_START {zip_name} {len(data)} bytes")
        print(base64.b64encode(data).decode())
        print("ZIP_BASE64_END")
    return zip_path
