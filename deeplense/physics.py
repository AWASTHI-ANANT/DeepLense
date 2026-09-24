"""Finite-difference operators and lensing physics used by the PINN.

Coordinates follow the `torch.nn.functional.grid_sample` convention: the image spans
[-1, 1] along each axis, so a grid of N pixels has spacing h = 2 / (N - 1). All
derivatives are divided by the proper powers of h, so the Poisson residual has
consistent units regardless of the grid resolution it is evaluated on.

Lensing relations (thin-lens, dimensionless units):
    deflection      alpha(theta) = grad psi(theta)
    lens equation   beta = theta - alpha(theta)
    Poisson eq.     laplacian psi = 2 kappa
"""

import torch
import torch.nn.functional as F

_LAPLACE = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view(1, 1, 3, 3)
_DX = torch.tensor([[0., 0., 0.], [-0.5, 0., 0.5], [0., 0., 0.]]).view(1, 1, 3, 3)
_DY = _DX.transpose(-1, -2).contiguous()


def grid_spacing(n: int) -> float:
    return 2.0 / (n - 1)


def laplacian(f: torch.Tensor) -> torch.Tensor:
    """5-point Laplacian of (B, 1, H, W), evaluated on the (H-2, W-2) interior."""
    h = grid_spacing(f.shape[-1])
    return F.conv2d(f, _LAPLACE.to(f), padding=0) / h ** 2


def gradient(f: torch.Tensor) -> torch.Tensor:
    """Central-difference gradient of (B, 1, H, W) -> (B, 2, H, W) as (d/dx, d/dy).

    Replicate padding keeps the output the same size as the input.
    """
    h = grid_spacing(f.shape[-1])
    fp = F.pad(f, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(fp, _DX.to(f)) / h
    gy = F.conv2d(fp, _DY.to(f)) / h
    return torch.cat([gx, gy], dim=1)


def poisson_residual(psi: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Pointwise residual  laplacian(psi) - 2 kappa  on the interior grid."""
    return laplacian(psi) - 2.0 * kappa[..., 1:-1, 1:-1]


def identity_grid(b: int, h: int, w: int, device=None, dtype=None) -> torch.Tensor:
    """Image-plane coordinates theta, shape (B, H, W, 2) in grid_sample's (x, y) order."""
    ys = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=-1).expand(b, h, w, 2)


def ray_trace(image: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Map the observed image back to the source plane with the lens equation.

    For every image-plane pixel theta we compute beta = theta - alpha(theta) and read the
    image at beta. For a correct deflection field this "de-lenses" arcs toward the
    compact source, which is the physically meaningful representation the classifier
    receives.

    image: (B, 1, H, W), alpha: (B, 2, H, W) in normalised units.
    """
    b, _, h, w = image.shape
    theta = identity_grid(b, h, w, image.device, image.dtype)
    beta = theta - alpha.permute(0, 2, 3, 1)
    return F.grid_sample(image, beta, mode="bilinear", padding_mode="zeros", align_corners=True)
