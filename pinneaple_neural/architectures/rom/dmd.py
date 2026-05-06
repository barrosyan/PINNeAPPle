from __future__ import annotations
"""Dynamic Mode Decomposition (DMD) for linear ROM dynamics (batch-capable)."""

from typing import Dict, Literal, Optional, Union

import torch

from .base import ROMBase, ROMOutput


CenterMode = Union[bool, Literal["global", "sequence"]]


class DynamicModeDecomposition(ROMBase):
    """
    Truncated (rank-r) DMD in reduced space.

    Given snapshots (optionally batched) X with shape:
      - (T, D) or (B, T, D)

    Build:
      X0 = [x0..x_{T-2}], X1 = [x1..x_{T-1}]   (concatenated over batch if B>1)

    Fit reduced operator:
      X0 = U S V^T
      A_tilde = U_r^T X1 V_r S_r^{-1}

    Options:
      - rank truncation r
      - centering: False / "global" / "sequence"
      - ridge regularization (l2): uses stabilized inverse of S:
            S^{-1}_ridge = diag( S / (S^2 + l2) )
        (When l2=0, recovers standard inverse 1/S)
    """

    def __init__(self, r: int = 64, center: CenterMode = "global", l2: float = 0.0):
        super().__init__()
        self.r = int(r)
        self.center: CenterMode = center
        self.l2 = float(l2)

        # Buffers (kept in state_dict; moved with .to(device))
        self.register_buffer("mean_", torch.zeros(1))   # (1,D) after fit
        self.register_buffer("basis_", torch.zeros(1))  # (D,r) after fit
        self.register_buffer("A_", torch.zeros(1))      # (r,r) after fit
        self._fitted = False

    def _as_btd(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 2:
            return X[None, :, :]
        if X.ndim == 3:
            return X
        raise ValueError(f"Expected X with shape (T,D) or (B,T,D), got {tuple(X.shape)}")

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "DynamicModeDecomposition":
        Xseq = self._as_btd(X)  # (B,T,D)
        B, T, D = Xseq.shape
        if T < 2:
            raise ValueError("Need at least T>=2 snapshots to fit DMD.")

        # --- Centering ---
        if self.center is False:
            mu = torch.zeros((1, D), device=Xseq.device, dtype=Xseq.dtype)
            Xc = Xseq
        elif self.center == "global" or self.center is True:
            # keep True as alias for "global" to be backwards friendly
            mu = Xseq.reshape(B * T, D).mean(dim=0, keepdim=True)  # (1,D)
            Xc = Xseq - mu
        elif self.center == "sequence":
            mu_seq = Xseq.mean(dim=1, keepdim=True)  # (B,1,D)
            Xc = Xseq - mu_seq
            # for rollout we can only store one mean; store global mean of sequence means
            mu = mu_seq.mean(dim=0).squeeze(1)  # (1,D)
        else:
            raise ValueError('center must be False, "global", "sequence", or True.')

        # store mean_ as buffer (in-place)
        self.mean_.resize_as_(mu).copy_(mu)

        # --- Build X0, X1 (concatenate transitions across batch) ---
        X0 = Xc[:, :-1, :].reshape(-1, D).t()  # (D, N)
        X1 = Xc[:, 1:, :].reshape(-1, D).t()   # (D, N)

        # --- Truncated SVD of X0 ---
        U, S, Vh = torch.linalg.svd(X0, full_matrices=False)
        r = min(self.r, U.shape[1])
        Ur = U[:, :r]                 # (D,r)
        Sr = S[:r]                    # (r,)
        Vr = Vh[:r, :].t()            # (N,r)

        # --- Stabilized inverse of Sr (ridge) ---
        # Standard: inv = 1/S
        # Ridge:    inv = S/(S^2 + l2)
        eps = 1e-12
        if self.l2 > 0.0:
            inv = Sr / (Sr * Sr + self.l2)
        else:
            inv = 1.0 / Sr.clamp_min(eps)
        Sr_inv = torch.diag(inv)      # (r,r)

        # --- Reduced operator ---
        A_tilde = Ur.t() @ X1 @ Vr @ Sr_inv  # (r,r)

        # store basis_ and A_ as buffers (in-place)
        self.basis_.resize_as_(Ur).copy_(Ur)
        self.A_.resize_as_(A_tilde).copy_(A_tilde)

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
        if x0.ndim != 2:
            raise ValueError(f"Expected x0 with shape (B,D), got {tuple(x0.shape)}")

        B, D = x0.shape
        if self.mean_.ndim != 2 or self.mean_.shape[-1] != D:
            raise RuntimeError(
                f"Fitted mean has shape {tuple(self.mean_.shape)} but x0 has D={D}. "
                "Did you fit on data with a different feature dimension?"
            )

        x = x0 - self.mean_                 # (B,D)
        a = x @ self.basis_                 # (B,r)

        xs = [x0]
        for _ in range(int(steps)):
            a = a @ self.A_.t()             # (B,r)
            xrec = a @ self.basis_.t() + self.mean_  # (B,D)
            xs.append(xrec)
        return torch.stack(xs, dim=1)       # (B,steps+1,D)

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        """
        One-step rollout over the full horizon.
        If X is (T,D), returns y as (T,D).
        If X is (B,T,D), returns y as (B,T,D).
        """
        Xseq = self._as_btd(X)
        B, T, D = Xseq.shape
        if T < 1:
            raise ValueError("Expected at least one timestep.")

        yhat = self.rollout(Xseq[:, 0, :], steps=T - 1)  # (B,T,D)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=Xseq.device)}
        if return_loss:
            losses["mse"] = self.mse(yhat, Xseq)
            losses["total"] = losses["mse"]

        y_out = yhat if X.ndim == 3 else yhat[0]
        return ROMOutput(y=y_out, losses=losses, extras={"fitted": self._fitted, "rank": int(self.basis_.shape[-1])})