from __future__ import annotations
"""Proper orthogonal decomposition via SVD for ROM."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ROMBase, ROMOutput


class POD(ROMBase):
    """
    POD via SVD.

    Fit:
      snapshots X: (N_snap, D) or (B,T,D) -> flattened to (N, D)

    Produces:
      U_r: (D, r) basis
      a:   (N, r) coefficients

    Reconstruct:
      X_hat = a U_r^T
    """
    def __init__(self, r: int = 64, center: bool = True):
        super().__init__()
        self.r = int(r)
        self.center = bool(center)

        self.register_buffer("mean_", torch.zeros(1))
        self.register_buffer("basis_", torch.zeros(1))
        self._fitted = False

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "POD":
        X2 = X
        if X2.ndim == 3:
            B, T, D = X2.shape
            X2 = X2.reshape(B * T, D)

        if self.center:
            mu = X2.mean(dim=0, keepdim=True)
            Xc = X2 - mu
            self.mean_ = mu
        else:
            self.mean_ = torch.zeros((1, X2.shape[1]), device=X2.device, dtype=X2.dtype)
            Xc = X2

        # SVD of (N,D): Xc = U S V^T -> basis = V_r
        # torch.linalg.svd returns Vh
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        Vr = Vh[: self.r, :].t().contiguous()  # (D,r)
        self.basis_ = Vr
        self._fitted = True
        return self

    def encode(self, X: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("POD not fitted. Call fit(X) first.")
        X2 = X
        if X2.ndim == 3:
            B, T, D = X2.shape
            X2 = X2.reshape(B * T, D)
        Xc = X2 - self.mean_
        a = Xc @ self.basis_  # (N,r)
        return a

    def decode(self, a: torch.Tensor, *, shape: Optional[tuple] = None) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("POD not fitted. Call fit(X) first.")
        Xc = a @ self.basis_.t()
        X = Xc + self.mean_
        if shape is not None:
            X = X.reshape(*shape)
        return X

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        # autoencode
        orig_shape = X.shape
        a = self.encode(X)
        Xhat = self.decode(a, shape=orig_shape if X.ndim == 3 else None)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=X.device)}
        if return_loss:
            losses["mse"] = self.mse(Xhat, X)
            losses["total"] = losses["mse"]

        return ROMOutput(y=Xhat, losses=losses, extras={"a": a, "basis": self.basis_, "fitted": self._fitted})
