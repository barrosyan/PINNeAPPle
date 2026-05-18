"""Parametric airfoil shape generators.

Functions here produce surface-point tensors that can be used for mesh
generation, PINN collocation, CFD preprocessing, or design optimization.
"""
from __future__ import annotations

import math

import torch


def naca_parametric(
    m: float = 0.0,
    p: float = 0.0,
    t_c: float = 0.12,
    n_pts: int = 100,
) -> torch.Tensor:
    """Generate a NACA 4-digit airfoil surface as control-point coordinates.

    Uses the standard NACA thickness distribution and camber-line formulae
    with cosine spacing for denser clustering near leading/trailing edges.

    Parameters
    ----------
    m:
        Maximum camber as fraction of chord (0 for symmetric NACA 00xx).
    p:
        Position of maximum camber (fraction of chord).
    t_c:
        Thickness-to-chord ratio (e.g. 0.12 for NACA 0012).
    n_pts:
        Number of surface points (upper + lower surfaces combined).

    Returns
    -------
    torch.Tensor
        Shape ``(n_pts, 2)`` — ``(x, y)`` surface coordinates.
    """
    n_half = n_pts // 2
    beta = torch.linspace(0.0, math.pi, n_half)
    xc = 0.5 * (1.0 - torch.cos(beta))

    yt = (t_c / 0.2) * (
        0.2969 * xc.sqrt()
        - 0.1260 * xc
        - 0.3516 * xc ** 2
        + 0.2843 * xc ** 3
        - 0.1015 * xc ** 4
    )

    if m == 0 or p == 0:
        yc = torch.zeros_like(xc)
        dyc_dx = torch.zeros_like(xc)
    else:
        mask_fwd = xc < p
        yc = torch.where(
            mask_fwd,
            (m / p ** 2) * (2 * p * xc - xc ** 2),
            (m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * xc - xc ** 2),
        )
        dyc_dx = torch.where(
            mask_fwd,
            (2 * m / p ** 2) * (p - xc),
            (2 * m / (1 - p) ** 2) * (p - xc),
        )

    theta = torch.atan(dyc_dx)
    xu = xc - yt * torch.sin(theta)
    yu = yc + yt * torch.cos(theta)
    xl = xc + yt * torch.sin(theta)
    yl = yc - yt * torch.cos(theta)

    x_surf = torch.cat([xu, xl.flip(0)])
    y_surf = torch.cat([yu, yl.flip(0)])

    return torch.stack([x_surf, y_surf], dim=-1)


__all__ = ["naca_parametric"]
