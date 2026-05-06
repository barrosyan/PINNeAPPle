from __future__ import annotations
"""Radial basis function network for reservoir-style regression."""
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import RCBase, RCOutput


NormMode = Optional[Literal["standard"]]


class RBFNetwork(RCBase):
    """
    RBF Network (Gaussian features) + ridge-regression readout.

    Architecture:
      - Centers C (fixed or learned)
      - Features: phi_i(x) = exp(-||x-C_i||^2 / (2*sigma^2))
      - Output weights solved by ridge regression

    Notes (literature-aligned):
      - RBF models are very sensitive to input scaling. Normalization is strongly recommended.
      - If learn_centers=True, closed-form ridge alone does NOT "learn" centers; use fit_gd().
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_centers: int = 512,
        sigma: Optional[float] = None,
        l2: float = 1e-6,
        learn_centers: bool = False,
        *,
        normalize: NormMode = "standard",
        use_bias: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_centers = int(num_centers)
        self.l2 = float(l2)
        self.learn_centers = bool(learn_centers)
        self.normalize = normalize
        self.use_bias = bool(use_bias)

        self.centers = nn.Parameter(torch.randn(self.num_centers, self.in_dim))
        self.log_sigma = nn.Parameter(torch.tensor(0.0))

        if sigma is not None:
            self.log_sigma.data = torch.log(torch.tensor(float(sigma)).clamp_min(1e-12))

        if not self.learn_centers:
            self.centers.requires_grad_(False)

        # Normalization stats (buffers so they move with .to(device) and are saved)
        self.register_buffer("_x_mean", torch.zeros(self.in_dim))
        self.register_buffer("_x_std", torch.ones(self.in_dim))
        self._has_norm_stats = False

        # Readout has size = num_centers (+1 if bias)
        self.num_features = self.num_centers + (1 if self.use_bias else 0)
        self.W_out = nn.Parameter(torch.zeros(self.num_features, self.out_dim), requires_grad=False)

        self._fitted = False

    def _maybe_normalize_fit(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize is None:
            self._has_norm_stats = False
            return x

        if self.normalize == "standard":
            mean = x.mean(dim=0)
            std = x.std(dim=0, unbiased=False).clamp_min(1e-12)
            self._x_mean.copy_(mean.detach())
            self._x_std.copy_(std.detach())
            self._has_norm_stats = True
            return (x - self._x_mean) / self._x_std

        raise ValueError(f"Unknown normalize mode: {self.normalize}")

    def _maybe_normalize_apply(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize is None or not self._has_norm_stats:
            return x
        return (x - self._x_mean) / self._x_std

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,in_dim), centers:(M,in_dim) -> (N,M)
        """
        # x2: (N,1), c2: (1,M), xc: (N,M)
        x2 = (x ** 2).sum(dim=-1, keepdim=True)
        c2 = (self.centers ** 2).sum(dim=-1)[None, :]
        xc = x @ self.centers.t()

        dist2 = x2 + c2 - 2.0 * xc
        # numeric safety: dist2 can be slightly negative due to floating-point errors
        dist2 = dist2.clamp_min(0.0)

        sigma2 = torch.exp(2.0 * self.log_sigma).clamp_min(1e-12)
        return torch.exp(-0.5 * dist2 / sigma2)

    def _design_matrix(self, x: torch.Tensor) -> torch.Tensor:
        Phi = self._phi(x)
        if self.use_bias:
            ones = torch.ones(Phi.shape[0], 1, device=Phi.device, dtype=Phi.dtype)
            Phi = torch.cat([Phi, ones], dim=1)
        return Phi

    @staticmethod
    def _sample_indices(num_points: int, k: int, device: torch.device) -> torch.Tensor:
        # If k <= N: sample without replacement; else sample with replacement (fixes shape mismatch).
        if k <= num_points:
            return torch.randperm(num_points, device=device)[:k]
        return torch.randint(low=0, high=num_points, size=(k,), device=device)

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        init_centers_from_data: bool = True,
        set_sigma_with_median_heuristic: bool = True,
    ) -> "RBFNetwork":
        """
        Closed-form ridge fit for readout. Optionally initializes centers and sigma.

        x: (N,in_dim), y: (N,out_dim)
        """
        x = self._maybe_normalize_fit(x)
        N = x.shape[0]

        if init_centers_from_data:
            idx = self._sample_indices(N, self.num_centers, x.device)
            self.centers.copy_(x[idx].detach().clone())

        # median heuristic for sigma if user didn't set it explicitly
        if set_sigma_with_median_heuristic and sigma_is_default(self.log_sigma) and N >= 2:
            m = min(N, 256)
            ii = self._sample_indices(N, m, x.device)
            jj = self._sample_indices(N, m, x.device)
            d = (x[ii] - x[jj]).pow(2).sum(dim=-1).sqrt()
            med = torch.median(d).clamp_min(1e-6)
            self.log_sigma.copy_(torch.log(med))

        Phi = self._design_matrix(x)  # (N, M [+1])
        W = self.ridge_solve(Phi, y, l2=self.l2)  # (M [+1], out_dim)
        self.W_out.copy_(W)

        self._fitted = True
        return self

    def fit_gd(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        init_centers_from_data: bool = True,
        set_sigma_with_median_heuristic: bool = True,
        alt_ridge_every: int = 10,
    ) -> "RBFNetwork":
        """
        Gradient-based training for centers/sigma (and optionally alternating ridge update of readout).

        This is the literature-consistent way to make learn_centers=True meaningful.

        Strategy:
          - (optional) init centers from data
          - (optional) initialize sigma via median heuristic
          - optimize centers/log_sigma to reduce MSE
          - periodically re-solve W_out via ridge given current features (alternating minimization)
        """
        if not self.learn_centers:
            raise RuntimeError("fit_gd() requires learn_centers=True to update centers by gradient.")

        x_n = self._maybe_normalize_fit(x)
        N = x_n.shape[0]

        with torch.no_grad():
            if init_centers_from_data:
                idx = self._sample_indices(N, self.num_centers, x_n.device)
                self.centers.copy_(x_n[idx].detach().clone())

            if set_sigma_with_median_heuristic and sigma_is_default(self.log_sigma) and N >= 2:
                m = min(N, 256)
                ii = self._sample_indices(N, m, x_n.device)
                jj = self._sample_indices(N, m, x_n.device)
                d = (x_n[ii] - x_n[jj]).pow(2).sum(dim=-1).sqrt()
                med = torch.median(d).clamp_min(1e-6)
                self.log_sigma.copy_(torch.log(med))

            # initialize readout once
            Phi0 = self._design_matrix(x_n)
            W0 = self.ridge_solve(Phi0, y, l2=self.l2)
            self.W_out.copy_(W0)

        opt = torch.optim.Adam(
            [p for p in [self.centers, self.log_sigma] if p.requires_grad],
            lr=lr,
            weight_decay=weight_decay,
        )

        for ep in range(int(epochs)):
            opt.zero_grad(set_to_none=True)

            Phi = self._design_matrix(x_n)
            y_hat = Phi @ self.W_out
            loss = self.mse(y_hat, y)

            # optional: tiny regularization on sigma scale to avoid collapse/explosion
            # (kept minimal; you can tune/remove if desired)
            loss = loss + 0.0 * (self.log_sigma ** 2)

            loss.backward()
            opt.step()

            # alternating minimization: re-solve W_out periodically
            if alt_ridge_every > 0 and (ep + 1) % int(alt_ridge_every) == 0:
                with torch.no_grad():
                    Phi_r = self._design_matrix(x_n)
                    W = self.ridge_solve(Phi_r, y, l2=self.l2)
                    self.W_out.copy_(W)

        self._fitted = True
        return self

    def forward(
        self,
        x: torch.Tensor,
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> RCOutput:
        x_n = self._maybe_normalize_apply(x)
        Phi = self._design_matrix(x_n)
        y = Phi @ self.W_out

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return RCOutput(
            y=y,
            losses=losses,
            extras={
                "sigma": float(torch.exp(self.log_sigma).detach().cpu().item()),
                "fitted": self._fitted,
                "normalized": bool(self.normalize is not None and self._has_norm_stats),
                "use_bias": self.use_bias,
            },
        )


def sigma_is_default(log_sigma: torch.Tensor, eps: float = 1e-6) -> bool:
    # Interpret "default" as log_sigma ~ 0 (i.e., sigma ~ 1)
    return log_sigma.detach().abs().item() < eps