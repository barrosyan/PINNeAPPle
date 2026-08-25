"""pinneapple_systems.component_modeling.bayesian — SWAGApproximation: a
real, architecture-agnostic approximate Bayesian posterior over a model's
weights, via SWAG (Stochastic Weight Averaging Gaussian — Maddox et al.
2019, "A Simple Baseline for Bayesian Uncertainty in Deep Learning").

Why SWAG and not a "Bayesian last layer": a last-layer treatment needs to
identify each architecture's final linear layer + feature extractor, which
is fragile across heterogeneous backbones. SWAG treats the WHOLE parameter
vector uniformly — no architecture-specific hook needed, so it works for any
``nn.Module`` unmodified.

How it works: after reaching a normal point estimate (plain Adam training
against a supplied loss), it takes ``n_snapshots`` further short
optimization runs, recording the running mean and variance of every
parameter across those snapshots. That gives a diagonal Gaussian posterior
N(mean, diag(variance)) over the weights. Prediction resamples weights from
that posterior ``n_samples`` times and reports the resulting predictive
mean/std — real stochasticity from actual weight-space samples, not dropout
or ensemble disagreement (see ``ensemble.DeepEnsemble`` for that other,
structurally different UQ mechanism).

Self-contained: unlike the upstream implementation this was ported from,
this version does not assume the model exposes any particular ``.fit()``
convention — it runs its own minimal Adam training loop against a supplied
``loss_fn(model, coords) -> scalar tensor``.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


class SWAGApproximation:
    def __init__(self, model: Any):
        self.model = model
        self.mean: Optional[Dict[str, torch.Tensor]] = None
        self.variance: Optional[Dict[str, torch.Tensor]] = None

    def fit(
        self,
        coords: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        loss_fn: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        n_snapshots: int = 10,
        snapshot_epochs: int = 20,
        snapshot_lr: Optional[float] = None,
    ) -> "SWAGApproximation":
        if loss_fn is None:
            if targets is None:
                raise ValueError("SWAGApproximation.fit() needs `targets` or a custom `loss_fn`.")
            loss_fn = lambda m, c: F.mse_loss(_unwrap(m(c)), targets)

        # Phase 1: reach a normal point estimate (MAP-ish).
        self._train_loop(coords, loss_fn, epochs, lr)

        # Phase 2: collect n_snapshots weight samples, each after
        # snapshot_epochs more optimization steps at a smaller LR (standard
        # SWAG practice — small steps around the minimum, not a fresh
        # training run each time).
        snapshot_lr = snapshot_lr if snapshot_lr is not None else lr * 0.1
        params = dict(self.model.named_parameters())
        mean = {n: torch.zeros_like(p) for n, p in params.items()}
        sq_mean = {n: torch.zeros_like(p) for n, p in params.items()}

        for k in range(n_snapshots):
            self._train_loop(coords, loss_fn, snapshot_epochs, snapshot_lr)
            with torch.no_grad():
                for n, p in params.items():
                    mean[n] += (p.detach() - mean[n]) / (k + 1)
                    sq_mean[n] += (p.detach() ** 2 - sq_mean[n]) / (k + 1)

        self.mean = mean
        # Diagonal posterior variance per parameter — clamped so a parameter
        # that happened not to move at all across snapshots still gets a
        # tiny non-zero variance (avoids a literal zero-width Gaussian).
        self.variance = {n: (sq_mean[n] - mean[n] ** 2).clamp(min=1e-12) for n in mean}

        with torch.no_grad():
            for n, p in params.items():
                p.copy_(mean[n])  # leave the model at the SWA mean, a sensible point estimate
        return self

    def _train_loop(
        self,
        coords: torch.Tensor,
        loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
        epochs: int,
        lr: float,
    ) -> None:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(self.model, coords)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()

    def predict_with_uncertainty(self, coords: torch.Tensor, n_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mean is None or self.variance is None:
            raise RuntimeError("SWAGApproximation.fit() must run before predict_with_uncertainty().")

        params = dict(self.model.named_parameters())
        self.model.eval()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                for n, p in params.items():
                    noise = torch.randn_like(p) * self.variance[n].sqrt()
                    p.copy_(self.mean[n] + noise)
                preds.append(_unwrap(self.model(coords)).clone())
            for n, p in params.items():
                p.copy_(self.mean[n])  # restore the point estimate — predict() must be idempotent
        stack = torch.stack(preds, dim=0)
        return stack.mean(dim=0), stack.std(dim=0)


def _unwrap(out: Any) -> torch.Tensor:
    """Accept a plain Tensor or any wrapper exposing a `.y` Tensor
    (matches the convention used elsewhere in pinneapple_systems)."""
    return out.y if hasattr(out, "y") else out
