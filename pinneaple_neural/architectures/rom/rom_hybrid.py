from __future__ import annotations
"""Hybrid ROM: POD compression + operator inference + neural residual correction."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import ROMBase, ROMOutput
from .pod import POD
from .opinf import OperatorInference


class _ResidualCorrector(nn.Module):
    """Small residual MLP that learns to correct OpInf prediction errors."""
    def __init__(self, r: int, hidden: int = 64, n_layers: int = 2):
        super().__init__()
        layers: list = [nn.Linear(r, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, r))
        # initialise last layer near zero so it starts as identity correction
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return self.net(a)


class ROMHybrid(ROMBase):
    """
    Hybrid ROM combining three components:

    1. POD encoder/decoder  —  full-space <-> latent projection
    2. OperatorInference     —  analytical latent dynamics (1-step map)
    3. Neural corrector      —  residual MLP to compensate OpInf truncation error

    Workflow
    --------
    fit(X):
        a. fit POD on snapshots X
        b. encode X to latent trajectories a
        c. fit OpInf on a
        (corrector is separately trained by calling train_corrector)

    forward(X):
        a. encode X -> a
        b. OpInf rollout: a_opinf
        c. correct: a_hat = a_opinf + corrector(a_opinf)
        d. decode: X_hat

    Parameters
    ----------
    field_dim   : physical field dimension D (used for shape checks only)
    r           : latent dimension
    l2          : ridge regularisation for OpInf
    use_corrector : if True, enable neural corrector (requires further training)
    corrector_hidden : hidden width of corrector MLP
    """

    def __init__(
        self,
        field_dim: int,
        r: int = 64,
        l2: float = 1e-6,
        use_corrector: bool = True,
        corrector_hidden: int = 64,
    ):
        super().__init__()
        self.field_dim = int(field_dim)
        self.r_dim = int(r)
        self.pod = POD(r=r, center=True)
        self.dyn = OperatorInference(r=r, l2_linear=l2, l2_quad=l2, use_quadratic=True, use_bias=True)
        self.use_corrector = bool(use_corrector)
        self.corrector = _ResidualCorrector(r, hidden=corrector_hidden) if use_corrector else None
        self._fitted = False

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, X: torch.Tensor) -> "ROMHybrid":
        """
        Fit POD and OpInf from snapshot tensor X of shape (B, T, D).
        Does NOT train the corrector; call train_corrector() separately.
        """
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (B,T,D), got {tuple(X.shape)}")
        B, T, D = X.shape

        self.pod.fit(X)
        a = self.pod.encode(X).reshape(B, T, -1)
        self.dyn.fit(a)
        self._fitted = True
        return self

    def train_corrector(
        self,
        X: torch.Tensor,
        *,
        epochs: int = 500,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ) -> "ROMHybrid":
        """
        Train the neural corrector to minimise residual error of OpInf.

        X : (B, T, D) training snapshots — the same used for fit().
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before train_corrector().")
        if not self.use_corrector or self.corrector is None:
            raise RuntimeError("use_corrector=False; no corrector to train.")

        dev = torch.device(device) if device else X.device
        self.to(dev)
        X = X.to(dev)

        B, T, D = X.shape
        opt = torch.optim.Adam(self.corrector.parameters(), lr=lr)

        with torch.no_grad():
            a_true = self.pod.encode(X).reshape(B, T, -1)
            a_opinf_full = self.dyn(a_true).y  # (B,T,r)

        self.corrector.train()
        for _ in range(epochs):
            # teacher-forcing: correct one-step predictions
            inp = a_opinf_full[:, :-1].reshape(-1, self.r_dim)
            tgt = a_true[:, 1:].reshape(-1, self.r_dim)
            pred = a_opinf_full[:, :-1].reshape(-1, self.r_dim) + self.corrector(inp)
            loss = ((pred - tgt) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.corrector.eval()
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (B,T,D), got {tuple(X.shape)}")
        B, T, D = X.shape

        a = self.pod.encode(X).reshape(B, T, -1)
        a_hat = self.dyn(a).y  # (B,T,r)

        # Apply neural corrector autoregressively if available
        if self.use_corrector and self.corrector is not None:
            correction = self.corrector(a_hat.reshape(B * T, -1)).reshape(B, T, -1)
            a_hat = a_hat + correction

        X_hat = self.pod.decode(a_hat.reshape(B * T, -1), shape=(B, T, D))

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=X.device)}
        if return_loss:
            losses["mse"]         = self.mse(X_hat, X)
            losses["latent_mse"]  = self.mse(a_hat, a)
            losses["total"]       = losses["mse"]

        return ROMOutput(
            y=X_hat,
            losses=losses,
            extras={
                "a":     a,
                "a_hat": a_hat,
                "fitted": self._fitted,
            },
        )
