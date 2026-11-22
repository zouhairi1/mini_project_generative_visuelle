"""S3 training loops: conditional VAE and conditional GAN (course versions)."""
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NUM_CLASSES, SHORT_NAMES, make_loader, to_tensor
from .models import CVAE, CondDiscriminator, CondGenerator, init_dcgan_weights
from .utils import set_seed


# ----------------------------------------------------------------------------- helpers

def fixed_class_inputs(n_per_class: int, z_dim: int, device, seed: int = 123):
    """Fixed (labels, z) for sample grids: n_per_class images of each class, ordered class by class."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(NUM_CLASSES).repeat_interleave(n_per_class)
    z = torch.randn(len(y), z_dim, generator=g)
    return y.to(device), z.to(device)


@torch.no_grad()
def generate(generator: CondGenerator, y: torch.Tensor, z: torch.Tensor = None) -> torch.Tensor:
    """Sample images of classes y (CVAE decoder or cGAN generator). Puts the network in eval mode."""
    was_training = generator.training
    generator.eval()
    if z is None:
        z = torch.randn(len(y), generator.z_dim, device=y.device)
    out = generator(z, y)
    generator.train(was_training)
    return out


def pairwise_diversity(x: torch.Tensor) -> float:
    """Mean Euclidean distance between all pairs of images (pixel space)."""
    return torch.pdist(x.flatten(1).float()).mean().item()


def class_diversity(generator, images, labels, device, n: int = 64, seed: int = 0) -> pd.DataFrame:
    """Per class: diversity of n generated images vs n real training images.

    ratio = synthetic / real. A ratio well below 1 means the model produces too little variety
    (mode collapse, GAN course slide 16).
    """
    g = torch.Generator().manual_seed(seed)
    rows = []
    for c in range(NUM_CLASSES):
        real = to_tensor(images[labels == c])
        real = real[torch.randperm(len(real), generator=g)[:n]]
        fake = generate(generator, torch.full((n,), c, dtype=torch.long, device=device)).cpu()
        d_real, d_fake = pairwise_diversity(real), pairwise_diversity(fake)
        rows.append({"class": SHORT_NAMES[c], "real diversity": round(d_real, 2),
                     "synthetic diversity": round(d_fake, 2), "ratio": round(d_fake / d_real, 2)})
    return pd.DataFrame(rows).set_index("class")


# ----------------------------------------------------------------------------- CVAE

def kl_divergence(mu, logvar):
    """KL(q(z|x,y) || N(0, I)) per image (closed form, AE/VAE course slide 12)."""
    return (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean()


def cvae_loss(recon, x, mu, logvar, beta: float):
    """-ELBO = reconstruction (sum of squared errors per image, Gaussian decoder) + beta * KL."""
    rec = F.mse_loss(recon, x, reduction="none").flatten(1).sum(1).mean()
    kl = kl_divergence(mu, logvar)
    return rec + beta * kl, rec, kl


def train_cvae(train_x, train_y, cfg: dict, beta: float = 1.0, seed: int = 0, device=None, verbose=True):
    """Train a CVAE. Returns (model, history per epoch, sample grid after each epoch)."""
    set_seed(seed)
    device = device or torch.device("cpu")
    tc, z_dim = cfg["train"], cfg["model"]["z_dim"]

    model = CVAE(z_dim, NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"])
    loader = make_loader(train_x, train_y, cfg["data"]["batch_size"], train=True,
                         power=tc["class_power"], seed=seed, drop_last=True)
    fixed_y, fixed_z = fixed_class_inputs(cfg["sampling"]["n_per_class"], z_dim, device)

    history, grids = [], []
    for epoch in range(1, tc["epochs"] + 1):
        beta_t = beta * min(1.0, epoch / tc["kl_warmup_epochs"])     # KL warm-up
        model.train()
        t0, sums, n = time.time(), np.zeros(3), 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            recon, mu, logvar = model(x, y)
            loss, rec, kl = cvae_loss(recon, x, mu, logvar, beta_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            sums += np.array([loss.item(), rec.item(), kl.item()]) * len(y)
            n += len(y)
        loss_m, rec_m, kl_m = sums / n
        history.append({"epoch": epoch, "beta": beta_t, "loss": loss_m, "recon": rec_m, "kl": kl_m})
        grids.append(generate(model.decoder, fixed_y, fixed_z).cpu())
        if verbose:
            print(f"[CVAE beta={beta}] epoch {epoch:2d}/{tc['epochs']} | beta_t {beta_t:.2f} "
                  f"| recon {rec_m:7.1f} | KL {kl_m:6.1f} | {time.time() - t0:.0f}s")
    return model, history, grids


# ----------------------------------------------------------------------------- conditional GAN

def train_cgan(train_x, train_y, cfg: dict, seed: int = 0, device=None, verbose=True):
    """Train the course cGAN: BCE, non-saturating generator loss, label smoothing, Adam(2e-4, 0.5).

    Returns (G, D, history {"epochs": [...], "iters": {...}}, sample grid after each epoch).
    """
    set_seed(seed)
    device = device or torch.device("cpu")
    tc, z_dim = cfg["train"], cfg["model"]["z_dim"]

    G = CondGenerator(z_dim, NUM_CLASSES).to(device)
    D = CondDiscriminator(NUM_CLASSES).to(device)
    G.apply(init_dcgan_weights)
    D.apply(init_dcgan_weights)
    opt_G = torch.optim.Adam(G.parameters(), lr=tc["lr"], betas=(tc["beta1"], 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=tc["lr"], betas=(tc["beta1"], 0.999))
    bce = nn.BCEWithLogitsLoss()
    loader = make_loader(train_x, train_y, cfg["data"]["batch_size"], train=True,
                         power=tc["class_power"], seed=seed, drop_last=True)
    fixed_y, fixed_z = fixed_class_inputs(cfg["sampling"]["n_per_class"], z_dim, device)

    iters = {"loss_D": [], "loss_G": [], "D_real": [], "D_fake": []}
    epochs, grids = [], []
    for epoch in range(1, tc["epochs"] + 1):
        G.train()
        D.train()
        t0, start = time.time(), len(iters["loss_D"])
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            b = len(y)

            # 1) Discriminator: real (x, y) -> 0.9 (label smoothing), fake (G(z, y), y) -> 0
            fake = G(torch.randn(b, z_dim, device=device), y)
            d_real = D(x, y)
            d_fake = D(fake.detach(), y)                     # detach: no gradient into G here
            loss_D = bce(d_real, torch.full_like(d_real, tc["real_label"])) + bce(d_fake, torch.zeros_like(d_fake))
            opt_D.zero_grad(set_to_none=True)
            loss_D.backward()
            opt_D.step()

            # 2) Generator: new noise, non-saturating loss = make D(G(z, y), y) -> 1 (course slides 10 and 12)
            d_gen = D(G(torch.randn(b, z_dim, device=device), y), y)
            loss_G = bce(d_gen, torch.ones_like(d_gen))
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()

            iters["loss_D"].append(loss_D.item())
            iters["loss_G"].append(loss_G.item())
            iters["D_real"].append(torch.sigmoid(d_real).mean().item())
            iters["D_fake"].append(torch.sigmoid(d_fake).mean().item())

        means = {k: float(np.mean(v[start:])) for k, v in iters.items()}
        epochs.append({"epoch": epoch, **means})
        grids.append(generate(G, fixed_y, fixed_z).cpu())
        if verbose:
            print(f"[cGAN] epoch {epoch:2d}/{tc['epochs']} | loss_D {means['loss_D']:.3f} | loss_G {means['loss_G']:.3f} "
                  f"| D(x) {means['D_real']:.2f} | D(G(z)) {means['D_fake']:.2f} | {time.time() - t0:.0f}s")
    return G, D, {"epochs": epochs, "iters": iters}, grids
