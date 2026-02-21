from __future__ import annotations
"""Dynamic mode decomposition for linear ROM dynamics."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ROMBase, ROMOutput


class DynamicModeDecomposition(ROMBase):
    """
    DMD MVP:
      Given snapshots X0 = [x0..x_{T-2}], X1 = [x1..x_{T-1}]
      Fit linear operator A in reduced space:
        A = X1 * pinv(X0)
    Supports:
      - optional truncation rank r (SVD)
    """
    def __init__(self, r: int = 64, center: bool = True, l2: float = 1e-8):
        super().__init__()
        self.r = int(r)
        self.center = bool(center)
        self.l2 = float(l2)

        self.register_buffer("mean_", torch.zeros(1))
        self.register_buffer("basis_", torch.zeros(1))   # (D,r)
        self.register_buffer("A_", torch.zeros(1))       # (r,r)
        self._fitted = False

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "DynamicModeDecomposition":
        # X: (T,D) or (B,T,D) -> fit on flattened batch as multiple sequences
        if X.ndim == 2:
            Xseq = X[None, :, :]
        else:
            Xseq = X
        B, T, D = Xseq.shape

        # center (global)
        Xflat = Xseq.reshape(B * T, D)
        if self.center:
            mu = Xflat.mean(dim=0, keepdim=True)
        else:
            mu = torch.zeros((1, D), device=X.device, dtype=X.dtype)
        self.mean_ = mu
        Xc = Xseq - mu

        # build X0, X1 by concatenating sequences in time
        X0 = Xc[:, :-1, :].reshape(-1, D).t()  # (D, N)
        X1 = Xc[:, 1:, :].reshape(-1, D).t()   # (D, N)

        # SVD of X0
        U, S, Vh = torch.linalg.svd(X0, full_matrices=False)
        r = min(self.r, U.shape[1])
        Ur = U[:, :r]           # (D,r)
        Sr = S[:r]              # (r,)
        Vr = Vh[:r, :].t()      # (N,r)

        # A_tilde = Ur^T X1 Vr Sr^{-1}
        Sr_inv = torch.diag(1.0 / Sr.clamp_min(1e-12))
        A_tilde = Ur.t() @ X1 @ Vr @ Sr_inv  # (r,r)

        self.basis_ = Ur
        self.A_ = A_tilde
        self._fitted = True
        return self

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        """
        x0: (B,D) in original space
        returns: (B,steps+1,D)
        """
        if not self._fitted:
            raise RuntimeError("DMD not fitted. Call fit(X) first.")
        B, D = x0.shape
        x = x0 - self.mean_
        a = x @ self.basis_  # (B,r)

        xs = [x0]
        for _ in range(int(steps)):
            a = a @ self.A_.t()
            xrec = a @ self.basis_.t() + self.mean_
            xs.append(xrec)
        return torch.stack(xs, dim=1)

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        # one-step predict over given sequence
        if X.ndim == 2:
            Xseq = X[None, :, :]
        else:
            Xseq = X
        B, T, D = Xseq.shape
        yhat = self.rollout(Xseq[:, 0, :], steps=T - 1)  # (B,T,D)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=X.device)}
        if return_loss:
            losses["mse"] = self.mse(yhat, Xseq)
            losses["total"] = losses["mse"]

        return ROMOutput(y=yhat if X.ndim == 3 else yhat[0], losses=losses, extras={"fitted": self._fitted})
