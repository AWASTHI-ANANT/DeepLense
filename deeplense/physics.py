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


def ray_trace(source: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Render a lensed image from a gridded source:  I(theta) = S(theta - alpha(theta)).

    This is the forward direction of the lens equation (surface brightness is conserved
    along rays). Note: v1 of the PINN applied this to the *observed image*, i.e.
    I(theta - alpha), which lenses the image again rather than reconstructing the source.
    That was a modelling error (masked by v1 learning alpha ~ 0); v2 renders an explicit
    source instead.

    source: (B, 1, H, W), alpha: (B, 2, H, W) in normalised units.
    """
    b, _, h, w = source.shape
    theta = identity_grid(b, h, w, source.device, source.dtype)
    beta = theta - alpha.permute(0, 2, 3, 1)
    return F.grid_sample(source, beta, mode="bilinear", padding_mode="zeros", align_corners=True)


# ── Analytic macro-lens (v2 PINN) ─────────────────────────────────────────────────────
# Parameters are per-image tensors of shape (B,). Coordinates are normalised [-1, 1].

def sis_shear_deflection(theta, theta_e, cx, cy, g1, g2, core: float = 0.005):
    """Deflection of a (softened) singular isothermal sphere plus external shear.

        psi_SIS   = theta_E * sqrt(|theta - c|^2 + s^2)      ->  kappa = theta_E (r^2 + 2 s^2) / (2 (r^2 + s^2)^1.5)
        psi_shear = 1/2 g1 (x^2 - y^2) + g2 x y               ->  kappa = 0 (no mass)

    Both terms satisfy the Poisson equation analytically, so only the learned
    perturbation needs the finite-difference constraint. theta: (B, H, W, 2) -> (B, 2, H, W).
    """
    v = lambda t: t[:, None, None]
    dx, dy = theta[..., 0] - v(cx), theta[..., 1] - v(cy)
    r = torch.sqrt(dx ** 2 + dy ** 2 + core ** 2)
    ax = v(theta_e) * dx / r + v(g1) * dx + v(g2) * dy
    ay = v(theta_e) * dy / r + v(g2) * dx - v(g1) * dy
    return torch.stack([ax, ay], dim=1)


def ellipticity_to_phi_q(e1, e2, max_c: float = 0.8, min_c: float = 1e-3):
    """lenstronomy's (e1, e2) -> (position angle, axis ratio q = minor/major).

    `min_c` keeps q < 1 so the SIE expressions (which divide by sqrt(1 - q^2)) stay finite.
    """
    phi = 0.5 * torch.atan2(e2, e1)
    c = torch.sqrt(e1 ** 2 + e2 ** 2 + min_c ** 2).clamp(max=max_c)
    return phi, (1 - c) / (1 + c)


def sie_deflection(theta, theta_e, e1, e2, cx, cy, s_scale: float = 1e-3):
    """Singular isothermal ellipsoid deflection, matching lenstronomy's 'SIE' convention.

    theta_e is lenstronomy's product-averaged Einstein radius. Keeton & Kochanek (1998):
        alpha_x' = b / sqrt(1-q^2) * arctan (sqrt(1-q^2) x' / (psi + s))
        alpha_y' = b / sqrt(1-q^2) * arctanh(sqrt(1-q^2) y' / (psi + q^2 s)),   psi = sqrt(q^2 (s^2 + x'^2) + y'^2)
    in the frame rotated by the position angle. theta: (B, H, W, 2) -> (B, 2, H, W).
    """
    v = lambda t: t[:, None, None]
    phi, q = ellipticity_to_phi_q(e1, e2)
    b = theta_e / torch.sqrt((1 + q ** 2) / (2 * q)) * torch.sqrt((1 + q ** 2) / 2)
    s = s_scale / torch.sqrt(q)
    cp, sp, q, b, s = v(torch.cos(phi)), v(torch.sin(phi)), v(q), v(b), v(s)
    dx, dy = theta[..., 0] - v(cx), theta[..., 1] - v(cy)
    x_, y_ = cp * dx + sp * dy, -sp * dx + cp * dy
    psi = torch.sqrt(q ** 2 * (s ** 2 + x_ ** 2) + y_ ** 2)
    k = torch.sqrt(1 - q ** 2)
    fx = b / k * torch.atan(k * x_ / (psi + s))
    fy = b / k * torch.atanh((k * y_ / (psi + q ** 2 * s)).clamp(-1 + 1e-6, 1 - 1e-6))
    return torch.stack([cp * fx - sp * fy, sp * fx + cp * fy], dim=1)


def shear_deflection(theta, g1, g2):
    """External shear, psi = 1/2 g1 (x^2 - y^2) + g2 x y (massless: laplacian psi = 0)."""
    v = lambda t: t[:, None, None]
    x, y = theta[..., 0], theta[..., 1]
    return torch.stack([v(g1) * x + v(g2) * y, v(g2) * x - v(g1) * y], dim=1)


def sersic_source(beta, sx, sy, r_eff, n, e1, e2, amp, r_floor: float = 1e-4):
    """Elliptical Sersic surface brightness at source-plane positions beta (lenstronomy
    'SERSIC_ELLIPSE', product-averaged radius):  I = amp * exp(-b_n ((R / R_eff)^(1/n) - 1)),
    b_n = 1.9992 n - 0.3271.  beta: (B, H, W, 2) -> (B, 1, H, W).
    """
    v = lambda t: t[:, None, None]
    phi, q = ellipticity_to_phi_q(e1, e2)
    cp, sp, q = v(torch.cos(phi)), v(torch.sin(phi)), v(q)
    bx, by = beta[..., 0] - v(sx), beta[..., 1] - v(sy)
    x_, y_ = (cp * bx + sp * by) * torch.sqrt(q), (-sp * bx + cp * by) / torch.sqrt(q)
    r = torch.sqrt(x_ ** 2 + y_ ** 2 + r_floor ** 2)
    bn = 1.9992 * v(n) - 0.3271
    return (v(amp) * torch.exp(-bn * ((r / v(r_eff)) ** (1 / v(n)) - 1)))[:, None]


def gaussian_psf(image, fwhm_px: float = 1.74):
    """Convolve (B, 1, H, W) with a normalised Gaussian PSF (DeepLenseSim: FWHM 0.087" = 1.74 px)."""
    sigma = fwhm_px / 2.3548
    r = int(3 * sigma + 1)
    ax = torch.arange(-r, r + 1, device=image.device, dtype=image.dtype)
    k1 = torch.exp(-0.5 * (ax / sigma) ** 2)
    k = (k1[:, None] * k1[None, :])
    k = (k / k.sum()).view(1, 1, 2 * r + 1, 2 * r + 1)
    return F.conv2d(image, k, padding=r)


def gaussian_source(beta, sx, sy, sigma, q, phi, amp):
    """Elliptical Gaussian surface brightness evaluated at source-plane positions beta.

    beta: (B, H, W, 2); q in (0, 1] is the axis ratio, phi the position angle -> (B, 1, H, W).
    """
    v = lambda t: t[:, None, None]
    bx, by = beta[..., 0] - v(sx), beta[..., 1] - v(sy)
    c, s = torch.cos(v(phi)), torch.sin(v(phi))
    u, w = c * bx + s * by, -s * bx + c * by
    su, sw = v(sigma) / torch.sqrt(v(q)), v(sigma) * torch.sqrt(v(q))
    return (v(amp) * torch.exp(-0.5 * ((u / su) ** 2 + (w / sw) ** 2)))[:, None]
