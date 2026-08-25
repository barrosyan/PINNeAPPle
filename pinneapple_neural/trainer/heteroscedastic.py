"""Heteroscedastic (input-dependent variance) uncertainty quantification.

Complementary to a fixed-noise-scalar Bayesian PINN: instead of one global
noise level, the model predicts a per-point variance alongside its mean
prediction, learned jointly via a Gaussian negative-log-likelihood loss.
Combined with MC-dropout, this decomposes total predictive uncertainty into
aleatoric (irreducible, data-noise) and epistemic (reducible, model/data-
scarcity) components via the law of total variance.

Architecture-agnostic by design: wrap ANY base network whose output layer
has been sized to `2 * n_out_physical` columns (first half = mean, second
half = log-variance) — works with an MLP, a Fourier feature network, a graph
network, a neural operator, or anything else, rather than a fixed set of
hardcoded per-architecture subclasses.

Typical workflow
-----------------
>>> from pinneapple_neural.trainer.heteroscedastic import (
...     HeteroscedasticWrapper, init_logvar_bias, heteroscedastic_nll_loss,
...     compute_aleatoric_epistemic_uq,
... )
>>> base = MyNetwork(n_in=2, n_out=2 * n_physical_outputs, dropout=0.1)
>>> init_logvar_bias(base.output_layer, n_physical_outputs)  # optional but recommended
>>> model = HeteroscedasticWrapper(base, n_out_physical=n_physical_outputs)
>>> mean, logvar = model.forward_uq(x)
>>> loss = heteroscedastic_nll_loss(mean, logvar, target)
>>> # ... after training, with dropout > 0 in `base`:
>>> uq = compute_aleatoric_epistemic_uq(model, query_coords, n_mc=50)
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class HeteroscedasticWrapper(nn.Module):
    """Wraps a base network (output dim = 2 * n_out_physical) into a model
    exposing `.forward_uq(x) -> (mean, log_variance)`, each of shape
    (N, n_out_physical). `.forward(x)` returns the mean only, for drop-in
    compatibility with code expecting a single-prediction model."""

    def __init__(self, base_network: nn.Module, n_out_physical: int):
        super().__init__()
        self.base_network = base_network
        self.n_out_physical = n_out_physical

    def forward_uq(self, x: torch.Tensor, **forward_kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.base_network(x, **forward_kwargs)
        n = self.n_out_physical
        return out[:, :n], out[:, n:]

    def forward(self, x: torch.Tensor, **forward_kwargs: Any) -> torch.Tensor:
        return self.forward_uq(x, **forward_kwargs)[0]

    @property
    def dropout_p(self) -> float:
        """Best-effort lookup of the base network's dropout probability, used
        by `compute_aleatoric_epistemic_uq` to check MC-dropout is enabled.
        Override by setting `.dropout_p` directly on the wrapper if the base
        network doesn't expose it under this name."""
        return float(getattr(self.base_network, "dropout_p", 0.0))


def init_logvar_bias(output_layer: nn.Linear, n_out_physical: int, bias_value: float = -4.6) -> None:
    """Bias-initialize a network's final linear layer's log-variance output
    columns (indices [n_out_physical:]) toward a small initial predicted
    uncertainty (log(sigma^2) ~= bias_value, i.e. sigma ~= exp(bias_value/2)
    ~= 0.1 at the default). Call once, right after constructing the base
    network and before wrapping it — purely a training-stability nicety, not
    required for correctness.
    """
    with torch.no_grad():
        output_layer.bias[n_out_physical:].fill_(bias_value)


def heteroscedastic_nll_loss(
    pred_mean: torch.Tensor,
    pred_logvar: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood: L = 0.5*(log(sigma^2) + (y-mu)^2/sigma^2),
    with log-variance clamped to [-10, 10] for numerical stability."""
    logvar = pred_logvar.clamp(-10.0, 10.0)
    return 0.5 * (logvar + (target - pred_mean).pow(2) / logvar.exp()).mean()


def compute_aleatoric_epistemic_uq(
    model: HeteroscedasticWrapper,
    coords: torch.Tensor,
    n_mc: int = 50,
    forward_kwargs: Dict[str, Any] = None,
) -> Dict[str, torch.Tensor]:
    """Decompose predictive uncertainty via the law of total variance:

        E[Var[y|x]]  = aleatoric  (irreducible, from data noise)
        Var[E[y|x]]  = epistemic  (reducible, from model uncertainty)
        Total        = aleatoric + epistemic

    Requires `model` to expose `forward_uq()` (any `HeteroscedasticWrapper`)
    and its base network's dropout to be > 0 (MC-dropout is what makes the
    mean prediction vary across the `n_mc` stochastic passes; without it,
    epistemic_std would be identically zero).
    """
    if not hasattr(model, "forward_uq"):
        raise TypeError("compute_aleatoric_epistemic_uq requires a model exposing forward_uq()")
    if getattr(model, "dropout_p", 0.0) == 0.0:
        raise ValueError("the base network's dropout must be > 0 for epistemic uncertainty estimation")

    fwd_kwargs = forward_kwargs or {}
    model.train()  # keep dropout active during MC sampling
    means_list = []
    vars_list = []
    with torch.no_grad():
        for _ in range(n_mc):
            mu, logvar = model.forward_uq(coords, **fwd_kwargs)
            means_list.append(mu.unsqueeze(0))
            vars_list.append(logvar.clamp(-10, 10).exp().unsqueeze(0))
    model.eval()

    means_stack = torch.cat(means_list, dim=0)  # (n_mc, N, n_out)
    vars_stack = torch.cat(vars_list, dim=0)    # (n_mc, N, n_out)

    mean_pred = means_stack.mean(0)
    epistemic_var = means_stack.var(0, unbiased=False)
    aleatoric_var = vars_stack.mean(0)
    total_var = epistemic_var + aleatoric_var

    return {
        "mean": mean_pred,
        "aleatoric_std": aleatoric_var.clamp(min=0).sqrt(),
        "epistemic_std": epistemic_var.clamp(min=0).sqrt(),
        "total_std": total_var.clamp(min=0).sqrt(),
    }
