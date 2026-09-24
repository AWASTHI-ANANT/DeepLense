import torch
import torch.nn as nn
import torchvision.models as tvm

from .. import NUM_CLASSES
from .common import Standardize, rgb_to_single_channel


def efficientnet_b0_features(pretrained: bool = True) -> nn.Sequential:
    """ImageNet EfficientNet-B0 feature extractor with a 1-channel stem.

    Output for a 150x150 input: (B, 1280, 5, 5).
    """
    weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    net = tvm.efficientnet_b0(weights=weights)
    old = net.features[0][0]
    new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    with torch.no_grad():
        new.weight.copy_(rgb_to_single_channel(old.weight))
    net.features[0][0] = new
    return net.features


class EfficientNetClassifier(nn.Module):
    """EfficientNet-B0 fine-tuned via transfer learning for 3-class substructure."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.norm = Standardize()
        self.backbone = efficientnet_b0_features(pretrained)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(self.norm(x)))
