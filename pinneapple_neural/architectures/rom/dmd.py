from __future__ import annotations
"""Dynamic Mode Decomposition (DMD) — batch-capable, stabilised, with spectral analysis."""

from typing import Dict, Literal, Optional, Tuple, Union

import torch

from .base import ROMBase, ROMOutput


CenterMode = Union[bool, Literal["global", "sequence"]]


class DynamicModeDecomposition(ROMBase):
    """
    Truncated (rank-r) DMD in reduced space.

    Accepts snapshots X with shape (T, D) or (B, T, D).

    Reduced operator
    ----------------
    X0 = U_r S_r V_r^T  (truncated SVD of left-snapshot matrix)
    A_tilde = U_r^T X1 V_r S_r^{-1}

    Ridge regularisation
    --------------------
    S_r^{-1}_ridge = diag(S / (S^2 + l2))   -- reduces to 1/S when l2=0

    Additional capabilities vs. the original
    -----------------------------------------
    - fit_pairs(X0, X1): accept explicit (N,D) snapshot pairs (used by HAVOK)
    - eig(): full spectral decomposition — eigenvalues, modes, frequencies
    - explained_variance_ratio_: energy fraction captured by each mode
    """

    def __init__(self, r: int = 64, center: CenterMode = "global", l2: float = 0.0):
        super().__init__()
        self.r = int(r)
        self.center: CenterMode = center
        self.l2 = float(l2)

        self.register_buffer("mean_",  torch.zeros(1))
        self.register_buffer("basis_", torch.zeros(1))
        self.register_buffer("A_",     torch.zeros(1))
        self._fitted = False
        self._singular_values: Optional[torch.Tensor] = None  # (r,) for explained variance

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _as_btd(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim == 2:
            return X[None]
        if X.ndim == 3:
            return X
        raise ValueError(f"Expected X with shape (T,D) or (B,T,D), got {tuple(X.shape)}")

    def _center_mean(self, Xseq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (Xc, mu) based on self.center mode."""
        B, T, D = Xseq.shape
        if self.center is False:
            mu = torch.zeros((1, D), device=Xseq.device, dtype=Xseq.dtype)
            return Xseq, mu
        if self.center in ("global", True):
            mu = Xseq.reshape(B * T, D).mean(dim=0, keepdim=True)
            return Xseq - mu, mu
        if self.center == "sequence":
            mu_seq = Xseq.mean(dim=1, keepdim=True)
            Xc = Xseq - mu_seq
            mu = mu_seq.mean(dim=0).view(1, D)
            return Xc, mu
        raise ValueError(f'center must be False, "global", "sequence", or True.')

    @staticmethod
    def _build_operator(
        X0: torch.Tensor,  # (D, N)
        X1: torch.Tensor,  # (D, N)
        r: int,
        l2: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Core DMD computation from paired snapshot matrices.
        Returns (Ur, A_tilde, S_r).
        """
        U, S, Vh = torch.linalg.svd(X0, full_matrices=False)
        r_eff = min(r, U.shape[1])
        Ur = U[:, :r_eff]
        Sr = S[:r_eff]
        Vr = Vh[:r_eff].t()

        eps = 1e-12
        inv = Sr / (Sr * Sr + l2) if l2 > 0.0 else 1.0 / Sr.clamp_min(eps)
        Sr_inv = torch.diag(inv)

        A_tilde = Ur.t() @ X1 @ Vr @ Sr_inv
        return Ur, A_tilde, Sr

    # ------------------------------------------------------------------
    # Fit API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "DynamicModeDecomposition":
        """Fit from sequential snapshots (T,D) or (B,T,D)."""
        Xseq = self._as_btd(X)
        B, T, D = Xseq.shape
        if T < 2:
            raise ValueError("Need at least T>=2 snapshots.")

        Xc, mu = self._center_mean(Xseq)
        self.mean_.resize_as_(mu).copy_(mu)

        X0 = Xc[:, :-1].reshape(-1, D).t()
        X1 = Xc[:, 1: ].reshape(-1, D).t()

        Ur, A_tilde, Sr = self._build_operator(X0, X1, self.r, self.l2)
        self.basis_.resize_as_(Ur).copy_(Ur)
        self.A_.resize_as_(A_tilde).copy_(A_tilde)
        self._singular_values = Sr.clone()
        self._fitted = True
        return self

    @torch.no_grad()
    def fit_pairs(
        self,
        X0: torch.Tensor,
        X1: torch.Tensor,
        *,
        center: bool = True,
    ) -> "DynamicModeDecomposition":
        """
        Fit from explicit (N, D) snapshot pairs.

        Accepts both 2-D (N,D) and 3-D (B,T,D) inputs; in the latter case
        the batch/time dimensions are flattened into N.

        Parameters
        ----------
        X0, X1 : (N, D) or (B, T, D) paired snapshot matrices.
        center  : if True, subtract the global mean of X0 before fitting.
        """
        # Flatten to (N, D)
        if X0.ndim == 3:
            X0 = X0.reshape(-1, X0.shape[-1])
        if X1.ndim == 3:
            X1 = X1.reshape(-1, X1.shape[-1])

        if X0.shape != X1.shape:
            raise ValueError(f"X0 and X1 must have the same shape; got {X0.shape} vs {X1.shape}")

        N, D = X0.shape
        if center:
            mu = X0.mean(dim=0, keepdim=True)
            X0c = X0 - mu
            X1c = X1 - mu
        else:
            mu = torch.zeros((1, D), device=X0.device, dtype=X0.dtype)
            X0c, X1c = X0, X1

        self.mean_.resize_as_(mu).copy_(mu)

        Ur, A_tilde, Sr = self._build_operator(
            X0c.t(), X1c.t(), self.r, self.l2
        )
        self.basis_.resize_as_(Ur).copy_(Ur)
        self.A_.resize_as_(A_tilde).copy_(A_tilde)
        self._singular_values = Sr.clone()
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def eig(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Eigendecomposition of the reduced operator A_tilde.

        Returns
        -------
        eigenvalues   : (r,) complex
        modes_reduced : (r, r) complex  — columns are reduced eigenvectors
        modes_full    : (D, r) complex  — full-space DMD modes (Phi = U W),
                        or None if basis_ is uninitialised.
        """
        if not self._fitted:
            raise RuntimeError("DMD not fitted.")
        lam, W = torch.linalg.eig(self.A_)
        Phi = self.basis_.to(torch.complex64) @ W  # (D, r)
        return lam, W, Phi

    @property
    def explained_variance_ratio_(self) -> Optional[torch.Tensor]:
        """Energy fraction captured by each singular value (after fit)."""
        if self._singular_values is None:
            return None
        s2 = self._singular_values ** 2
        return s2 / s2.sum().clamp_min(1e-12)

    def frequencies(self, dt: float = 1.0) -> Optional[torch.Tensor]:
        """
        Continuous-time frequencies from DMD eigenvalues: f = imag(log(lam)) / (2pi*dt).
        Returns real tensor of shape (r,).  dt is the physical time step.
        """
        lam, _, _ = self.eig()
        log_lam = torch.log(lam + 1e-30)
        return (log_lam.imag / (2.0 * torch.pi * dt))

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rollout(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        """
        x0 : (B, D) — initial state in original space.
        Returns (B, steps+1, D).
        """
        if not self._fitted:
            raise RuntimeError("DMD not fitted.")
        if x0.ndim != 2:
            raise ValueError(f"x0 must be (B, D), got {tuple(x0.shape)}")

        B, D = x0.shape
        if self.mean_.ndim != 2 or self.mean_.shape[-1] != D:
            raise RuntimeError(
                f"Fitted mean has shape {tuple(self.mean_.shape)} but x0 has D={D}."
            )

        a = (x0 - self.mean_) @ self.basis_
        xs = [x0]
        for _ in range(int(steps)):
            a = a @ self.A_.t()
            xs.append(a @ self.basis_.t() + self.mean_)
        return torch.stack(xs, dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        Xseq = self._as_btd(X)
        B, T, D = Xseq.shape
        yhat = self.rollout(Xseq[:, 0], steps=T - 1)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=Xseq.device)}
        if return_loss:
            losses["mse"]   = self.mse(yhat, Xseq)
            losses["total"] = losses["mse"]

        y_out = yhat if X.ndim == 3 else yhat[0]
        return ROMOutput(
            y=y_out,
            losses=losses,
            extras={
                "fitted": self._fitted,
                "rank":   int(self.basis_.shape[-1]) if self._fitted else 0,
            },
        )
