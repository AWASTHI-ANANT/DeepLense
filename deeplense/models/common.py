import torch
import torch.nn as nn

from ..data import PIXEL_MEAN, PIXEL_STD


class Standardize(nn.Module):
    """(x - mean) / std with the training-set pixel statistics.

    Kept inside the model so datasets always hand over physical [0, 1] intensities,
    which the PINN needs for ray tracing.
    """

    def __init__(self, mean: float = PIXEL_MEAN, std: float = PIXEL_STD):
        super().__init__()
        self.mean, self.std = mean, std

    def forward(self, x):
        return (x - self.mean) / self.std


def rgb_to_single_channel(weight: torch.Tensor) -> torch.Tensor:
    """Collapse a pretrained (out, 3, k, k) filter bank to (out, 1, k, k).

    Summing (rather than averaging) the RGB kernels makes the new filter respond to a
    grayscale image x exactly as the original responded to the RGB image (x, x, x).
    """
    return weight.sum(dim=1, keepdim=True)


class ConvBlock(nn.Module):
    """Two 3x3 conv-BN-GELU layers with a residual (projected when shapes change)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Identity() if in_ch == out_ch and stride == 1
            else nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.body(x) + self.skip(x))
