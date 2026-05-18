from __future__ import annotations
"""Proper Orthogonal Decomposition (POD) via SVD — robust ROM compression."""

from typing import Dict, Optional, Tuple

import torch

from .base import ROMBase, ROMOutput


class POD(ROMBase):
    """
    POD via full or randomised truncated SVD for ROM compression.

    Fit
    ---
    Snapshots X with shape (N_snap, D) or (B, T, D) — flattened to (N, D).
    Produces:
      mean_  : (1, D)   — global mean (if center=True)
      basis_ : (D, r_eff) — POD modes (columns of V from X = U S V^T)
      sv_    : (r_eff,)  — singular values (for energy analysis)

    Reconstruction: X_hat = (X - mean_) @ basis_ @ basis_.T + mean_

    Extra features vs. previous version
    ------------------------------------
    - randomised SVD option (random=True) for large D, using sketch of size 2*r
    - explained_variance_ratio_  property
    - reconstruction_error() convenience method
    - fit_incremental() for streaming / mini-batch updates (incremental SVD)
    """

    def __init__(
        self,
        r: int = 64,
        center: bool = True,
        energy: Optional[float] = None,
        randomised: bool = False,
        random_seed: int = 0,
    ):
        """
        Parameters
        ----------
        r           : maximum number of modes (if energy is None).
        center      : subtract global mean before decomposition.
        energy      : if in (0,1], select r automatically to capture this
                      fraction of total variance.
        randomised  : use randomised SVD (faster for very large D).
        random_seed : seed for the randomised projection (reproducibility).
        """
        super().__init__()
        self.r = int(r)
        self.center = bool(center)
        self.energy = None if energy is None else float(energy)
        self.randomised = bool(randomised)
        self.random_seed = int(random_seed)

        if self.r <= 0:
            raise ValueError("r must be positive.")
        if self.energy is not None and not (0.0 < self.energy <= 1.0):
            raise ValueError("energy must be in (0, 1].")

        self.register_buffer("mean_",  torch.empty(0))
        self.register_buffer("basis_", torch.empty(0))
        self.register_buffer("sv_",    torch.empty(0))
        self._fitted = False
        self._r_eff: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten(X: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple]]:
        if X.ndim == 3:
            B, T, D = X.shape
            return X.reshape(B * T, D), (B, T, D)
        if X.ndim == 2:
            return X, None
        raise ValueError(f"Expected X with ndim 2 or 3, got {X.ndim}.")

    def _svd(self, Xc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (S, Vh) respecting the randomised flag."""
        if self.randomised:
            gen = torch.Generator(device=Xc.device)
            gen.manual_seed(self.random_seed)
            k = min(2 * self.r + 10, min(Xc.shape))
            omega = torch.randn(Xc.shape[1], k, device=Xc.device, dtype=Xc.dtype, generator=gen)
            Y = Xc @ omega
            Q, _ = torch.linalg.qr(Y)
            B = Q.t() @ Xc
            _, S, Vh = torch.linalg.svd(B, full_matrices=False)
        else:
            _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        return S, Vh

    def _select_rank(self, S: torch.Tensor, max_r: int) -> int:
        if self.energy is not None:
            s2 = S ** 2
            cum = torch.cumsum(s2, 0) / s2.sum().clamp_min(torch.finfo(s2.dtype).eps)
            r_eff = int(
                torch.searchsorted(
                    cum, torch.tensor(self.energy, device=cum.device, dtype=cum.dtype)
                ).item()
            ) + 1
            return max(1, min(r_eff, max_r))
        return min(self.r, max_r)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "POD":
        """Fit POD basis from snapshots X of shape (N, D) or (B, T, D)."""
        X2, _ = self._flatten(X)

        if self.center:
            mu = X2.mean(dim=0, keepdim=True)
            Xc = X2 - mu
            self.mean_ = mu
        else:
            self.mean_ = torch.zeros((1, X2.shape[1]), device=X2.device, dtype=X2.dtype)
            Xc = X2

        S, Vh = self._svd(Xc)
        max_r = Vh.shape[0]
        r_eff = self._select_rank(S, max_r)

        Vr = Vh[:r_eff].t().contiguous()
        self.basis_ = Vr
        self.sv_    = S[:r_eff].clone()
        self._r_eff = r_eff
        self._fitted = True
        return self

    @torch.no_grad()
    def fit_incremental(self, X_new: torch.Tensor, *, weight: float = 0.5) -> "POD":
        """
        Incremental POD update: merge a new batch of snapshots with the
        existing basis via a rank-doubling + re-truncation approach.

        This is an approximation of exact incremental SVD, suitable for
        streaming scenarios where re-fitting from scratch is too expensive.

        weight : fraction of the new data in the merged matrix (0 < weight < 1).
        """
        if not self._fitted:
            return self.fit(X_new)

        X2, _ = self._flatten(X_new)
        if self.center:
            mu_new = X2.mean(dim=0, keepdim=True)
            mu = (1 - weight) * self.mean_ + weight * mu_new
            Xc = X2 - mu
            self.mean_ = mu
        else:
            Xc = X2

        # New partial basis
        _, S_new, Vh_new = torch.linalg.svd(Xc, full_matrices=False)
        r_new = min(self.r, Vh_new.shape[0])
        B_new = Vh_new[:r_new].t() * (weight ** 0.5)
        B_old = self.basis_ * ((1 - weight) ** 0.5)

        # Merge and re-truncate
        B_merged = torch.cat([B_old, B_new], dim=1)  # (D, 2r)
        _, S_m, Vh_m = torch.linalg.svd(B_merged.t(), full_matrices=False)
        max_r = Vh_m.shape[0]
        r_eff = self._select_rank(S_m, max_r)

        self.basis_ = Vh_m[:r_eff].t().contiguous()
        self.sv_    = S_m[:r_eff].clone()
        self._r_eff = r_eff
        return self

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("POD not fitted.")
        X2, _ = self._flatten(X)
        return (X2 - self.mean_) @ self.basis_

    def decode(self, a: torch.Tensor, *, shape: Optional[tuple] = None) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("POD not fitted.")
        X = a @ self.basis_.t() + self.mean_
        return X.reshape(*shape) if shape is not None else X

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @property
    def explained_variance_ratio_(self) -> Optional[torch.Tensor]:
        """Energy fraction for each mode: sv_i^2 / sum(sv^2)."""
        if not self._fitted or self.sv_.numel() == 0:
            return None
        s2 = self.sv_ ** 2
        return s2 / s2.sum().clamp_min(1e-12)

    def reconstruction_error(self, X: torch.Tensor) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics for snapshot matrix X.
        Returns dict with 'mse', 'relative_l2', 'max_abs'.
        """
        if not self._fitted:
            raise RuntimeError("POD not fitted.")
        with torch.no_grad():
            a    = self.encode(X)
            Xhat = self.decode(a)
            X2, _ = self._flatten(X)
            diff = Xhat - X2
            mse  = float((diff ** 2).mean())
            rel  = float(diff.norm() / (X2.norm().clamp_min(1e-12)))
            mx   = float(diff.abs().max())
        return {"mse": mse, "relative_l2": rel, "max_abs": mx}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        X2, orig = self._flatten(X)
        a    = self.encode(X2)
        Xhat = self.decode(a)
        y    = Xhat.reshape(*orig) if orig else Xhat

        losses: Dict[str, torch.Tensor] = {}
        if return_loss:
            mse = self.mse(y, X)
            losses = {"mse": mse, "total": mse}

        return ROMOutput(
            y=y,
            losses=losses,
            extras={
                "a":       a,
                "basis":   self.basis_,
                "mean":    self.mean_,
                "sv":      self.sv_,
                "fitted":  self._fitted,
                "r_eff":   self._r_eff,
            },
        )
