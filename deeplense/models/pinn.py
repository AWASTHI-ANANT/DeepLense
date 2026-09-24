"""Physics-Informed Neural Network for substructure classification.

Architecture
------------
    image I (B,1,150,150)
        |
    EfficientNet-B0 encoder (shared)  ->  features (B,1280,5,5)
        |                                   |
        |                          physics decoder -> psi_hat, kappa_hat on a 64x64 grid
        |                                   |
        |                     alpha = grad psi_hat          (deflection field)
        |                     S_hat = I(theta - alpha)      (lens equation, ray tracing)
        |                                   |
        |                          source encoder on [S_hat, kappa_hat]
        |                                   |
        +------------- concat -------------+
                          |
                     class logits

Loss
----
    L = CE(logits, y) + lambda_poisson * mean( (laplacian psi_hat - 2 kappa_hat)^2 )

The Poisson term ties the predicted potential and the predicted mass density together
(nabla^2 psi = 2 kappa). Because kappa_hat >= 0 (softplus), it also forces psi_hat to
be subharmonic, i.e. to correspond to a non-negative projected mass: a real physical
constraint, not just a smoothness prior. The classification loss supervises psi_hat
through the ray-traced source and kappa_hat directly, so neither map can collapse to a
trivial solution that ignores the image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import IMG_SIZE, NUM_CLASSES
from ..physics import gradient, poisson_residual, ray_trace
from .common import ConvBlock, Standardize
from .efficientnet import efficientnet_b0_features


class PhysicsDecoder(nn.Module):
    """Upsamples encoder features to potential psi_hat and convergence kappa_hat maps."""

    def __init__(self, in_ch: int = 1280, grid: int = 64):
        super().__init__()
        self.grid = grid
        self.reduce = nn.Sequential(nn.Conv2d(in_ch, 128, 1, bias=False), nn.BatchNorm2d(128), nn.GELU())
        self.up = nn.ModuleList([ConvBlock(128, 64), ConvBlock(64, 32), ConvBlock(32, 16)])
        self.out = nn.Conv2d(16, 2, 3, padding=1)
        # Start from psi = 0 (no deflection, S_hat = I) and small kappa.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, f):
        x = self.reduce(f)
        for block in self.up:                          # 5 -> 10 -> 20 -> 40
            x = block(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))
        x = F.interpolate(x, size=(self.grid, self.grid), mode="bilinear", align_corners=True)
        raw = self.out(x)
        psi = raw[:, :1]
        psi = psi - psi.mean(dim=(-2, -1), keepdim=True)   # psi is defined up to a constant
        kappa = F.softplus(raw[:, 1:2])                     # projected mass density >= 0
        return psi, kappa


class LensingPINN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 grid: int = 64, dropout: float = 0.3):
        super().__init__()
        self.norm = Standardize()
        self.backbone = efficientnet_b0_features(pretrained)
        self.physics = PhysicsDecoder(1280, grid)
        self.source_encoder = nn.Sequential(
            ConvBlock(2, 16, stride=2),     # 150 -> 75
            ConvBlock(16, 32, stride=2),    # -> 38
            ConvBlock(32, 64, stride=2),    # -> 19
            ConvBlock(64, 128, stride=2),   # -> 10
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280 + 128, num_classes))

    def forward(self, x):
        f = self.backbone(self.norm(x))
        psi, kappa = self.physics(f)

        alpha = F.interpolate(gradient(psi), size=x.shape[-2:], mode="bilinear", align_corners=True)
        source = ray_trace(x, alpha)
        kappa_full = F.interpolate(kappa, size=x.shape[-2:], mode="bilinear", align_corners=True)

        g = self.source_encoder(torch.cat([self.norm(source), kappa_full], dim=1))
        logits = self.head(torch.cat([f.mean(dim=(-2, -1)), g], dim=1))
        return {"logits": logits, "psi": psi, "kappa": kappa, "alpha": alpha, "source": source}


class PINNLoss(nn.Module):
    """CE + lambda * Poisson residual.  lambda = 0 gives the physics-free ablation."""

    def __init__(self, lambda_poisson: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.lambda_poisson = lambda_poisson
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, y):
        ce = self.ce(out["logits"], y)
        poisson = poisson_residual(out["psi"], out["kappa"]).pow(2).mean()
        loss = ce + self.lambda_poisson * poisson
        return loss, {"ce": ce.item(), "poisson": poisson.item()}


assert IMG_SIZE == 150
