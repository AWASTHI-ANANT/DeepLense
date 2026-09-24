import torch.nn as nn

from .. import NUM_CLASSES
from .common import ConvBlock, Standardize


class LensingCNN(nn.Module):
    """Residual CNN designed from scratch for single-channel 150x150 lensing maps.

    A stride-1 stem keeps full resolution for the first block, since substructure
    signatures are only a few pixels wide; five strided residual stages then take
    150 -> 5 before global average pooling.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, widths=(32, 64, 128, 256, 512), dropout: float = 0.3):
        super().__init__()
        self.norm = Standardize()
        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.GELU(),
        )
        stages, in_ch = [], widths[0]
        for w in widths:
            stages.append(ConvBlock(in_ch, w, stride=2))   # 150 -> 75 -> 38 -> 19 -> 10 -> 5
            in_ch = w
        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(self.stem(self.norm(x))))
