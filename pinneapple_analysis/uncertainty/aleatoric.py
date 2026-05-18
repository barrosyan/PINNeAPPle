"""Aleatoric (data) uncertainty via heteroscedastic NLL regression.

The model predicts both a mean μ(x) and a log-variance log σ²(x).
Loss = 0.5·log σ² + 0.5·(y − μ)²·exp(−log σ²)  — Gaussian NLL, numerically stable.

This captures *irreducible* uncertainty: noise in the data that no model can remove.
For reducible (epistemic) uncertainty use MCDropout, EnsembleUQ, or decompose_uncertainty.

Quick start::

    from pinneapple_analysis.uncertainty import AleatoricHead, aleatoric_nll_loss

    base = MyModel()
    model = AleatoricHead(base, out_dim=1)
    # model(x) → (mean, log_var), both shape (N, out_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for x_batch, y_batch in loader:
        mean, log_var = model(x_batch)
        loss = aleatoric_nll_loss(mean, log_var, y_batch)
        loss.backward(); optimizer.step(); optimizer.zero_grad()

    # Inference
    model.eval()
    with torch.no_grad():
        result = model.predict_with_uncertainty(x_test)
    # result.aleatoric_std  →  predicted noise std
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .core import UQResult


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def aleatoric_nll_loss(
    mean: Tensor,
    log_var: Tensor,
    target: Tensor,
    *,
    min_log_var: float = -10.0,
    max_log_var: float = 10.0,
) -> Tensor:
    """Heteroscedastic Gaussian NLL loss.

    Parameters
    ----------
    mean : Tensor — predicted mean, shape ``(N, D)`` or ``(N,)``.
    log_var : Tensor — predicted log-variance, same shape.
    target : Tensor — ground truth, same shape as *mean*.
    min_log_var, max_log_var : float — clamping bounds for numerical stability.

    Returns
    -------
    Tensor — scalar loss.
    """
    lv = log_var.clamp(min_log_var, max_log_var)
    sq_err = (target - mean) ** 2
    return (0.5 * lv + 0.5 * sq_err * torch.exp(-lv)).mean()


# ---------------------------------------------------------------------------
# Model head
# ---------------------------------------------------------------------------

class AleatoricHead(nn.Module):
    """Wrap a deterministic model with a learned aleatoric variance head.

    The base model is unchanged; a small MLP projects its output to a
    log-variance estimate. During training, optimize with
    :func:`aleatoric_nll_loss`.

    Parameters
    ----------
    base : nn.Module
        Pre-existing model returning ``(N, out_dim)`` tensors.
    out_dim : int
        Dimensionality of the model output (= number of predicted fields).
    hidden : int
        Width of the log-variance MLP (default 64).

    Outputs
    -------
    forward(x) → ``(mean, log_var)`` — both shape ``(N, out_dim)``.
    """

    def __init__(
        self,
        base: nn.Module,
        out_dim: int,
        *,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.base = base
        self.out_dim = out_dim
        self.log_var_head = nn.Sequential(
            nn.Linear(out_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return ``(mean, log_var)`` tensors, both shape ``(N, out_dim)``."""
        raw = self.base(x)
        if raw.ndim == 1:
            raw = raw.unsqueeze(-1)
        mean = raw
        log_var = self.log_var_head(raw)
        return mean, log_var

    def predict_with_uncertainty(
        self,
        x: Tensor,
        *,
        device: Optional[torch.device] = None,
    ) -> UQResult:
        """Run a deterministic forward pass and return a :class:`UQResult`.

        The returned ``aleatoric_std`` is the model-predicted noise standard
        deviation. ``epistemic_std`` is zero here — to estimate epistemic
        uncertainty, wrap this model with :class:`~pinneapple_uq.mc_dropout.MCDropoutWrapper`
        and call :func:`~pinneapple_uq.decomposition.decompose_uncertainty`.
        """
        if device is not None:
            self.to(device)
            x = x.to(device)

        self.eval()
        with torch.no_grad():
            mean, log_var = self(x)

        std = torch.exp(0.5 * log_var)
        return UQResult(
            mean=mean,
            std=std,
            aleatoric_std=std,
            epistemic_std=torch.zeros_like(std),
            metadata={"method": "aleatoric", "model": type(self.base).__name__},
        )
