"""Models.

S2: baseline CNN classifier (also the feature extractor for later evaluation).
S3: conditional generator shared by the CVAE decoder and the cGAN generator,
    CVAE encoder, conditional discriminator.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """4 conv blocks (64 -> 32 -> 16 -> 8 -> 4), global average pooling, dropout, linear head."""

    def __init__(self, n_classes: int = 7, channels=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        blocks, c_in = [], 3
        for c in channels:
            blocks.append(conv_block(c_in, c))
            c_in = c
        self.backbone = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_in, n_classes))
        self.feature_dim = c_in

    def features(self, x):
        """Penultimate features (used later for FID and memorization)."""
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.features(x))


# ----------------------------------------------------------------------------- S3: generative models

def one_hot_maps(y: torch.Tensor, n_classes: int, size: int) -> torch.Tensor:
    """Class labels (B,) -> one-hot maps (B, n_classes, size, size), concatenated to the image channels."""
    return F.one_hot(y, n_classes).float()[:, :, None, None].expand(-1, -1, size, size)


class CondGenerator(nn.Module):
    """DCGAN generator conditioned on the class (GAN course, slides 19-20 and 23).

    [z, embedding(y)] -> FC -> 256 x 4 x 4 -> transposed convs 4 -> 8 -> 16 -> 32 -> 64 -> tanh.
    The SAME architecture is the CVAE decoder and the cGAN generator, so only the training objective differs.
    """

    def __init__(self, z_dim: int = 64, n_classes: int = 7, emb_dim: int = 32, base: int = 256):
        super().__init__()
        self.z_dim, self.base = z_dim, base
        self.embed = nn.Embedding(n_classes, emb_dim)
        self.fc = nn.Sequential(
            nn.Linear(z_dim + emb_dim, base * 4 * 4, bias=False), nn.BatchNorm1d(base * 4 * 4), nn.ReLU(True),
        )

        def up(c_in, c_out):
            return nn.Sequential(nn.ConvTranspose2d(c_in, c_out, 4, 2, 1, bias=False),
                                 nn.BatchNorm2d(c_out), nn.ReLU(True))

        self.net = nn.Sequential(
            up(base, base),                                   # 8 x 8
            up(base, base // 2),                              # 16 x 16
            up(base // 2, base // 4),                         # 32 x 32
            nn.ConvTranspose2d(base // 4, 3, 4, 2, 1),        # 64 x 64
            nn.Tanh(),                                        # pixels in [-1, 1]
        )

    def forward(self, z, y):
        h = self.fc(torch.cat([z, self.embed(y)], dim=1)).view(-1, self.base, 4, 4)
        return self.net(h)


class CondEncoder(nn.Module):
    """CVAE encoder q(z | x, y): image + one-hot class maps -> (mu, log sigma^2) (AE/VAE course, slide 16)."""

    def __init__(self, z_dim: int = 64, n_classes: int = 7):
        super().__init__()
        self.n_classes = n_classes

        def down(c_in, c_out):
            return nn.Sequential(nn.Conv2d(c_in, c_out, 4, 2, 1, bias=False),
                                 nn.BatchNorm2d(c_out), nn.LeakyReLU(0.2, True))

        self.net = nn.Sequential(
            down(3 + n_classes, 32), down(32, 64), down(64, 128), down(128, 256), nn.Flatten(),  # 256 x 4 x 4
        )
        self.mu = nn.Linear(256 * 4 * 4, z_dim)
        self.logvar = nn.Linear(256 * 4 * 4, z_dim)       # two heads: mu and log-variance

    def forward(self, x, y):
        h = self.net(torch.cat([x, one_hot_maps(y, self.n_classes, x.size(-1))], dim=1))
        # Clamp log sigma^2 to [-10, 10]: with random initial weights it can be huge, exp() explodes and the KL
        # reaches billions in the first steps (seen in the first run). Trained values stay far inside this range.
        return self.mu(h), self.logvar(h).clamp(-10.0, 10.0)


class CVAE(nn.Module):
    """Conditional VAE: encoder q(z | x, y) + decoder p(x | z, y) (AE/VAE course, slide 21)."""

    def __init__(self, z_dim: int = 64, n_classes: int = 7):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = CondEncoder(z_dim, n_classes)
        self.decoder = CondGenerator(z_dim, n_classes)

    def forward(self, x, y):
        mu, logvar = self.encoder(x, y)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterisation trick (slide 15)
        return self.decoder(z, y), mu, logvar


class CondDiscriminator(nn.Module):
    """DCGAN discriminator D(x, y): strided convs, LeakyReLU(0.2), BatchNorm, no pooling (GAN course, slides 19-20).

    The class is given as one-hot maps concatenated to the image (cGAN, slide 23).
    Returns a logit; sigmoid(logit) = probability that (x, y) is real. Training uses BCEWithLogitsLoss,
    which is sigmoid + BCE in one numerically stable function.
    """

    def __init__(self, n_classes: int = 7, base: int = 64):
        super().__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Conv2d(3 + n_classes, base, 4, 2, 1), nn.LeakyReLU(0.2, True),                                   # 32
            nn.Conv2d(base, base * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 2), nn.LeakyReLU(0.2, True),      # 16
            nn.Conv2d(base * 2, base * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 4), nn.LeakyReLU(0.2, True),  # 8
            nn.Conv2d(base * 4, base * 8, 4, 2, 1, bias=False), nn.BatchNorm2d(base * 8), nn.LeakyReLU(0.2, True),  # 4
            nn.Conv2d(base * 8, 1, 4, 1, 0),                                                                     # 1
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, one_hot_maps(y, self.n_classes, x.size(-1))], dim=1)).view(-1)


def init_dcgan_weights(m: nn.Module) -> None:
    """DCGAN initialisation: weights ~ N(0, 0.02), BatchNorm scale ~ N(1, 0.02)."""
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
