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
    """CE + image reconstruction + Poisson residual on the perturbation + kappa sparsity
    (+ optional minimum-deflection prior on the perturbation, used by v3).

    `lambda_defl` penalises mean |grad d_psi|^2. Without it the perturbation can use
    harmonic (laplacian = 0, i.e. massless) modes that satisfy the Poisson term with
    kappa_sub = 0; with it every extra deflection must pay for itself in the fit.
    """

    def __init__(self, lambda_rec: float = 1.0, lambda_poisson: float = 0.1,
                 lambda_sparse: float = 0.01, label_smoothing: float = 0.0, lambda_defl: float = 0.0):
        super().__init__()
        self.lambda_rec, self.lambda_poisson, self.lambda_sparse = lambda_rec, lambda_poisson, lambda_sparse
        self.lambda_defl = lambda_defl
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, out, y):
        ce = self.ce(out["logits"].float(), y)
        rec = (out["residual"] / PIXEL_STD).pow(2).mean()
        poisson = poisson_residual(out["psi_sub"], out["kappa_sub"]).pow(2).mean()
        sparse = out["kappa_sub"].mean()
        loss = ce + self.lambda_rec * rec + self.lambda_poisson * poisson + self.lambda_sparse * sparse
        terms = {}
        if self.lambda_defl:
            defl = gradient(out["psi_sub"]).pow(2).sum(1).mean()
            loss = loss + self.lambda_defl * defl
            terms["defl_sub"] = defl.item()
        theta_e_px = out["lens"]["theta_e"].mean() * 74.5      # 150 px span 2 units
        return loss, {"ce": ce.item(), "rec": rec.item(), "poisson": poisson.item(),
                      "kappa_sub": sparse.item(), "theta_e_px": theta_e_px.item(), **terms}


# ══════════════════════════════════════════════════════════════════════════════════════
# PINN v3: simulator-matched lens model + high-resolution substructure decoder
# ══════════════════════════════════════════════════════════════════════════════════════
#
# v2's kappa_sub turned out class-independent: an SIS cannot be elliptical, so kappa_sub
# absorbed the lens ellipticity instead of substructure, and it came from a 5x5 bottleneck
# so it could only be a smooth blob. v3 changes, motivated by the DeepLenseSim generator
# (lenstronomy: SIE + external shear main halo, elliptical Sersic source, Gaussian PSF,
# point-mass subhalos / a line of point masses for vortices):
#   * macro lens = SIE + shear and source = elliptical Sersic, both matching lenstronomy's
#     conventions (verified to ~1e-6), plus the instrument PSF and a background level.
#     Direct fits reach R^2 ~ 0.99, so the residual is (substructure + noise).
#   * substructure decoder = U-Net over EfficientNet's multi-scale features -> 75x75 grid.
#   * `physics_only=True`: the classifier sees only [residual, kappa_sub, lens params], and
#     these are detached from the classification loss, so the physics maps are shaped only
#     by reconstruction + Poisson + sparsity. This tests whether the physics branch on its
#     own carries the substructure signal.

from ..physics import gaussian_psf, sersic_source, shear_deflection, sie_deflection  # noqa: E402

LENS_PARAMS_V3 = ("theta_e", "e1", "e2", "cx", "cy", "g1", "g2",
                  "sx", "sy", "r_eff", "n", "se1", "se2", "amp", "bg")
SKIPS = (1, 2, 3, 5)                       # EfficientNet stages at 75, 38, 19, 10 px


class LensParamHeadV3(nn.Module):
    def __init__(self, in_dim: int = 1280):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 256), nn.GELU(), nn.Linear(256, len(LENS_PARAMS_V3)))
        nn.init.zeros_(self.net[-1].weight)
        # Prior: theta_E ~ 27 px, round centred lens, no shear, R_eff = 0.25" (5 px), n = 1.
        prior = [_inv_softplus(0.37), 0, 0, 0, 0, 0, 0, 0, 0, _inv_softplus(0.067), math.log(1 / 6),
                 0, 0, _inv_softplus(0.2), 0]
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor(prior))

    def forward(self, g):
        r = self.net(g).float()
        t = torch.tanh
        return {
            "theta_e": F.softplus(r[:, 0]) + 0.02,
            "e1": 0.4 * t(r[:, 1]), "e2": 0.4 * t(r[:, 2]),
            "cx": 0.3 * t(r[:, 3]), "cy": 0.3 * t(r[:, 4]),
            "g1": 0.3 * t(r[:, 5]), "g2": 0.3 * t(r[:, 6]),
            "sx": 0.5 * t(r[:, 7]), "sy": 0.5 * t(r[:, 8]),
            "r_eff": F.softplus(r[:, 9]) + 0.005,
            "n": 0.5 + 3.5 * torch.sigmoid(r[:, 10]),
            "se1": 0.4 * t(r[:, 11]), "se2": 0.4 * t(r[:, 12]),
            "amp": F.softplus(r[:, 13]),
            "bg": 0.1 * t(r[:, 14]),
        }


class SubstructureUNet(nn.Module):
    """Decodes (d_psi, kappa_sub >= 0) on a 75x75 grid from multi-scale encoder features."""

    def __init__(self, skip_ch=(16, 24, 40, 112), bottleneck_ch: int = 1280):
        super().__init__()
        self.reduce = nn.Sequential(nn.Conv2d(bottleneck_ch, 128, 1, bias=False), nn.BatchNorm2d(128), nn.GELU())
        widths = (96, 64, 32, 16)
        blocks, in_ch = [], 128
        for skip, w in zip(reversed(skip_ch), widths):        # 10, 19, 38, 75
            blocks.append(ConvBlock(in_ch + skip, w))
            in_ch = w
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Conv2d(in_ch, 2, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([0.0, -6.0]))      # d_psi = 0, kappa_sub ~ 0 at start

    def forward(self, bottleneck, skips):
        x = self.reduce(bottleneck)
        for block, s in zip(self.blocks, reversed(skips)):
            x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([x, s], dim=1))
        raw = self.out(x)
        d_psi = raw[:, :1] - raw[:, :1].mean(dim=(-2, -1), keepdim=True)
        return d_psi, F.softplus(raw[:, 1:2])


class LensingPINNv3(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True,
                 physics_only: bool = False, dropout: float = 0.3):
        super().__init__()
        self.physics_only = physics_only
        self.norm = Standardize()
        self.backbone = efficientnet_b0_features(pretrained)
        self.lens = LensParamHeadV3(1280)
        self.physics = SubstructureUNet()
        self.residual_encoder = nn.Sequential(
            ConvBlock(2, 32, stride=2), ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2), ConvBlock(128, 128, stride=2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        n_in = 128 + len(LENS_PARAMS_V3) + (0 if physics_only else 1280)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_in, num_classes))

    def encode(self, x):
        skips, h = [], self.norm(x)
        for i, stage in enumerate(self.backbone):
            h = stage(h)
            if i in SKIPS:
                skips.append(h)
        return h, skips

    def forward(self, x):
        f, skips = self.encode(x)
        g = f.mean(dim=(-2, -1))
        b, _, h, w = x.shape

        with torch.autocast(device_type=x.device.type, enabled=False):
            p = self.lens(g.float())
            d_psi, kappa_sub = self.physics(f.float(), [s.float() for s in skips])

            theta = identity_grid(b, h, w, x.device, torch.float32)
            alpha_macro = (sie_deflection(theta, p["theta_e"], p["e1"], p["e2"], p["cx"], p["cy"])
                           + shear_deflection(theta, p["g1"], p["g2"]))
            alpha_sub = F.interpolate(gradient(d_psi), size=(h, w), mode="bilinear", align_corners=True)
            alpha = alpha_macro + alpha_sub
            beta = theta - alpha.permute(0, 2, 3, 1)
            source = sersic_source(beta, p["sx"], p["sy"], p["r_eff"], p["n"], p["se1"], p["se2"], p["amp"])
            recon = gaussian_psf(source) + p["bg"][:, None, None, None]
            residual = x.float() - recon
            kappa_full = F.interpolate(kappa_sub, size=(h, w), mode="bilinear", align_corners=True)

        params = torch.stack([p[k] for k in LENS_PARAMS_V3], dim=1)
        phys_in = torch.cat([residual / PIXEL_STD, kappa_full], dim=1)
        if self.physics_only:                       # classification can't reshape the physics maps
            phys_in, params = phys_in.detach(), params.detach()
        z = self.residual_encoder(phys_in.to(g.dtype))
        feats = [z, params.to(g.dtype)] if self.physics_only else [g, z, params.to(g.dtype)]
        logits = self.head(torch.cat(feats, dim=1))
        return {"logits": logits, "recon": recon, "residual": residual, "alpha": alpha,
                "alpha_macro": alpha_macro, "psi_sub": d_psi, "kappa_sub": kappa_sub, "lens": p}
