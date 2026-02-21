from __future__ import annotations
"""Kernel autoencoder with MMD regularization."""

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .base import AEBase
from .dense_ae import _mlp


def _pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    (n,d), (m,d) -> (n,m) squared Euclidean distances
    Uses a stable algebraic form (no cdist dependency).
    """
    x = x.float()
    y = y.float()
    x2 = (x * x).sum(dim=1, keepdim=True)              # (n,1)
    y2 = (y * y).sum(dim=1, keepdim=True).transpose(0, 1)  # (1,m)
    dist2 = x2 - 2.0 * (x @ y.transpose(0, 1)) + y2
    return dist2.clamp_min(0.0)


def _rbf_kernel_from_dist2(dist2: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0:
        raise ValueError("sigma must be > 0 for RBF kernel.")
    return torch.exp(-dist2 / (2.0 * sigma * sigma))


def _median_heuristic_sigma(z: torch.Tensor) -> float:
    """
    Median heuristic for RBF bandwidth on z.
    Returns a python float > 0 (with a small floor for safety).
    """
    with torch.no_grad():
        z2 = z.detach()
        n = z2.shape[0]
        if n < 2:
            return 1.0
        dist2 = _pairwise_sq_dists(z2, z2)
        # remove diagonal for median
        mask = ~torch.eye(n, dtype=torch.bool, device=dist2.device)
        vals = dist2[mask]
        med = torch.median(vals)
        # sigma^2 ~ median(dist^2) / 2  (common convention), sigma = sqrt(...)
        sigma2 = (med / 2.0).clamp_min(1e-12)
        return float(torch.sqrt(sigma2).item())


def _mmd_rbf_unbiased(
    z: torch.Tensor,
    z_prior: torch.Tensor,
    sigmas: Sequence[float],
) -> torch.Tensor:
    """
    Unbiased MMD^2 estimate with RBF kernels:
      MMD^2 = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]
    where expectations are over distinct pairs (i != j).
    Supports multi-kernel by summing kernels over sigmas.
    """
    z = z.float()
    z_prior = z_prior.float()

    n = z.shape[0]
    m = z_prior.shape[0]
    if n < 2 or m < 2:
        # not enough pairs; fall back to biased but safe estimate
        # (keeps training from NaNing on tiny batches)
        dist_zz = _pairwise_sq_dists(z, z)
        dist_pp = _pairwise_sq_dists(z_prior, z_prior)
        dist_zp = _pairwise_sq_dists(z, z_prior)
        kzz = 0.0
        kpp = 0.0
        kzp = 0.0
        for s in sigmas:
            kzz = kzz + _rbf_kernel_from_dist2(dist_zz, s).mean()
            kpp = kpp + _rbf_kernel_from_dist2(dist_pp, s).mean()
            kzp = kzp + _rbf_kernel_from_dist2(dist_zp, s).mean()
        return kzz + kpp - 2.0 * kzp

    dist_zz = _pairwise_sq_dists(z, z)          # (n,n)
    dist_pp = _pairwise_sq_dists(z_prior, z_prior)  # (m,m)
    dist_zp = _pairwise_sq_dists(z, z_prior)    # (n,m)

    # Build multi-kernel K matrices
    Kzz = 0.0
    Kpp = 0.0
    Kzp = 0.0
    for s in sigmas:
        Kzz = Kzz + _rbf_kernel_from_dist2(dist_zz, s)
        Kpp = Kpp + _rbf_kernel_from_dist2(dist_pp, s)
        Kzp = Kzp + _rbf_kernel_from_dist2(dist_zp, s)

    # Unbiased: exclude diagonal in Kzz and Kpp
    Kzz_sum = Kzz.sum() - torch.diagonal(Kzz, 0).sum()
    Kpp_sum = Kpp.sum() - torch.diagonal(Kpp, 0).sum()

    mmd_xx = Kzz_sum / (n * (n - 1))
    mmd_yy = Kpp_sum / (m * (m - 1))
    mmd_xy = Kzp.mean()

    return mmd_xx + mmd_yy - 2.0 * mmd_xy


class KAEAutoencoder(AEBase):
    """
    Practical MMD-WAE style Autoencoder:
      - Dense AE
      - MMD penalty on latent space to match a simple prior.

    Backward-compatible behavior:
      - Keeps the same constructor args and loss dict keys.
      - Adds optional multi-kernel support via mmd_sigmas (optional).
      - If mmd_sigma <= 0, uses median heuristic per-batch.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden: List[int] = (512, 256),
        activation: str = "gelu",
        mmd_weight: float = 1.0,
        mmd_sigma: float = 1.0,
        prior: str = "normal",
        # Optional: multi-kernel MMD (keeps old API working if you ignore it)
        mmd_sigmas: Optional[List[float]] = None,
    ):
        super().__init__()
        act = {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU()}.get(
            activation.lower(), nn.GELU()
        )
        self.mmd_weight = float(mmd_weight)
        self.mmd_sigma = float(mmd_sigma)
        self.mmd_sigmas = mmd_sigmas  # if provided, overrides mmd_sigma
        self.prior = prior.lower().strip()

        enc_dims = [input_dim, *list(hidden), latent_dim]
        dec_dims = [latent_dim, *list(reversed(hidden)), input_dim]

        self.encoder = _mlp(enc_dims, act, last_act=False)
        self.decoder = _mlp(dec_dims, act, last_act=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], -1)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def _get_sigmas(self, z: torch.Tensor) -> List[float]:
        if self.mmd_sigmas is not None and len(self.mmd_sigmas) > 0:
            return [float(s) for s in self.mmd_sigmas if float(s) > 0]

        # Single sigma mode: if <= 0 use median heuristic
        if self.mmd_sigma <= 0:
            return [_median_heuristic_sigma(z)]

        return [float(self.mmd_sigma)]

    def loss_from_parts(self, *, x_hat: torch.Tensor, z: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.reshape(x.shape[0], -1)
        recon = torch.mean((x_hat - x) ** 2)

        if self.prior == "uniform":
            z_prior = (2.0 * torch.rand_like(z) - 1.0)
        else:
            z_prior = torch.randn_like(z)

        sigmas = self._get_sigmas(z)
        mmd = _mmd_rbf_unbiased(z, z_prior, sigmas=sigmas)

        total = recon + self.mmd_weight * mmd
        return {"recon": recon, "mmd": mmd, "total": total}
