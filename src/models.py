"""Models. S2: baseline CNN classifier (also the feature extractor for later evaluation)."""
import torch.nn as nn


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
