"""PINN v2: physics-informed classifier built on an explicit, differentiable lens model.

Why v2
------
v1 let the network predict the lensing potential freely. With the Poisson loss on, it found
the trivial solution psi ~ 0 (no deflection), so the physics branch learned nothing. v2 makes
lensing *necessary*: the network must reproduce the observed image by forward-modelling it.

Model (per image, all differentiable)
-------------------------------------
    EfficientNet-B0 encoder -> features f
      |-- lens head   -> SIS Einstein radius theta_E, lens centre, external shear (g1, g2)
      |                  + elliptical Gaussian source (centre, size, axis ratio, angle, amplitude)
      |-- decoder     -> substructure potential d_psi and convergence kappa_sub >= 0 (64x64 grid)

    alpha    = alpha_SIS+shear (analytic)  +  grad d_psi (finite differences)
    beta     = theta - alpha                         (lens equation)
    I_hat    = S(beta)                               (surface brightness is conserved)
    residual = I - I_hat

    classifier( [ pooled f , CNN(residual, kappa_sub) , lens parameters ] ) -> logits

Loss
----
    L = CE
      + lambda_rec     * mean( ((I_hat - I) / sigma_pix)^2 )         reconstruct the image
      + lambda_poisson * mean( (laplacian d_psi - 2 kappa_sub)^2 )   Poisson equation
      + lambda_sparse  * mean( kappa_sub )                           substructure is small

A compact source can only become a ring through a correct Einstein radius, so zero deflection
is no longer a solution. The smooth macro-lens satisfies the Poisson equation analytically; the
finite-difference constraint acts on the learned perturbation, where substructure lives, and
with kappa_sub >= 0 it only admits positive-mass perturbations.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import NUM_CLASSES
from ..data import PIXEL_STD
from ..physics import gaussian_source, gradient, identity_grid, poisson_residual, sis_shear_deflection
from .common import ConvBlock, Standardize
from .efficientnet import efficientnet_b0_features
from .pinn import PhysicsDecoder

LENS_PARAMS = ("theta_e", "cx", "cy", "g1", "g2", "sx", "sy", "sigma", "q", "phi", "amp")


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


class LensParamHead(nn.Module):
    """Predicts the 11 macro-lens + source parameters, mapped to physical ranges.

    The last layer starts at zero weight with biases at a physical prior (theta_E ~ 25 px,
    centred lens and source, compact round-ish source), so training starts from a sensible
    lens rather than a random one.
    """

    def __init__(self, in_dim: int = 1280):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, len(LENS_PARAMS)))
        nn.init.zeros_(self.net[-1].weight)
        prior = [_inv_softplus(0.33), 0, 0, 0, 0, 0, 0, _inv_softplus(0.045), math.log(0.75 / 0.25), 0, _inv_softplus(1.0)]
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor(prior))

    def forward(self, g):
        r = self.net(g).float()
        return {
            "theta_e": F.softplus(r[:, 0]) + 0.02,
            "cx": 0.3 * torch.tanh(r[:, 1]),
            "cy": 0.3 * torch.tanh(r[:, 2]),
            "g1": 0.3 * torch.tanh(r[:, 3]),
            "g2": 0.3 * torch.tanh(r[:, 4]),
            "sx": 0.5 * torch.tanh(r[:, 5]),
            "sy": 0.5 * torch.tanh(r[:, 6]),
            "sigma": F.softplus(r[:, 7]) + 0.005,
            "q": 0.2 + 0.8 * torch.sigmoid(r[:, 8]),
            "phi": r[:, 9],
            "amp": F.softplus(r[:, 10]),
        }


class LensingPINNv2(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 grid: int = 64, dropout: float = 0.3):
        super().__init__()
        self.norm = Standardize()
        self.backbone = efficientnet_b0_features(pretrained)
        self.lens = LensParamHead(1280)
        self.physics = PhysicsDecoder(1280, grid)
        with torch.no_grad():                       # start with ~no substructure mass
            self.physics.out.bias[1] = -6.0
        self.residual_encoder = nn.Sequential(
            ConvBlock(2, 16, stride=2), ConvBlock(16, 32, stride=2),
            ConvBlock(32, 64, stride=2), ConvBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(1280 + 128 + len(LENS_PARAMS), num_classes))

    def forward(self, x):
        f = self.backbone(self.norm(x))
        g = f.mean(dim=(-2, -1))
        b, _, h, w = x.shape

        # Lens physics in float32 even under AMP: finite differences divide by h^2.
        with torch.autocast(device_type=x.device.type, enabled=False):
            f32, x32 = f.float(), x.float()
            p = self.lens(g.float())
            d_psi, kappa_sub = self.physics(f32)

            theta = identity_grid(b, h, w, x.device, torch.float32)
            alpha_macro = sis_shear_deflection(theta, p["theta_e"], p["cx"], p["cy"], p["g1"], p["g2"])
            alpha_sub = F.interpolate(gradient(d_psi), size=(h, w), mode="bilinear", align_corners=True)
            alpha = alpha_macro + alpha_sub
            beta = theta - alpha.permute(0, 2, 3, 1)
            recon = gaussian_source(beta, p["sx"], p["sy"], p["sigma"], p["q"], p["phi"], p["amp"])
            residual = x32 - recon
            kappa_full = F.interpolate(kappa_sub, size=(h, w), mode="bilinear", align_corners=True)

        z = self.residual_encoder(torch.cat([residual / PIXEL_STD, kappa_full], dim=1))
        params = torch.stack([p[k] for k in LENS_PARAMS], dim=1)
        logits = self.head(torch.cat([g, z.to(g.dtype), params.to(g.dtype)], dim=1))
        return {"logits": logits, "recon": recon, "residual": residual, "alpha": alpha,
                "alpha_macro": alpha_macro, "psi_sub": d_psi, "kappa_sub": kappa_sub, "lens": p}


class PINNv2Loss(nn.Module):
    """CE + image reconstruction + Poisson residual on the perturbation + kappa sparsity."""

    def __init__(self, lambda_rec: float = 1.0, lambda_poisson: float = 0.1,
                 lambda_sparse: float = 0.01, label_smoothing: float = 0.0):
        super().__init__()
        self.lambda_rec, self.lambda_poisson, self.lambda_sparse = lambda_rec, lambda_poisson, lambda_sparse
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, y):
        ce = self.ce(out["logits"].float(), y)
        rec = (out["residual"] / PIXEL_STD).pow(2).mean()
        poisson = poisson_residual(out["psi_sub"], out["kappa_sub"]).pow(2).mean()
        sparse = out["kappa_sub"].mean()
        loss = ce + self.lambda_rec * rec + self.lambda_poisson * poisson + self.lambda_sparse * sparse
        theta_e_px = out["lens"]["theta_e"].mean() * 74.5      # 150 px span 2 units
        return loss, {"ce": ce.item(), "rec": rec.item(), "poisson": poisson.item(),
                      "kappa_sub": sparse.item(), "theta_e_px": theta_e_px.item()}
