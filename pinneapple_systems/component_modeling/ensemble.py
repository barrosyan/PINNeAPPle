"""pinneapple_systems.component_modeling.ensemble — DeepEnsemble: N
independently-initialized copies of the same architecture, trained
separately, combined at inference time.

Ensemble disagreement (across-member std) is genuine epistemic uncertainty —
a different mechanism from MC-Dropout's within-model stochastic-forward-pass
uncertainty (see ``mc_dropout.mc_dropout_uncertainty``) and from
``bayesian.SWAGApproximation``'s weight-space Bayesian posterior. Per Wilson
& Izmailov (2020, "Bayesian Deep Learning and a Probabilistic Perspective of
Generalization"), a deep ensemble is a practical approximation to Bayesian
model averaging.

Decoupled from any specific registry: members are built from a supplied
``model_factory() -> nn.Module`` callable.
"""
from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F


class DeepEnsemble:
    def __init__(
        self,
        model_factory: Callable[[], Any],
        n_members: int = 5,
        base_seed: int = 0,
    ):
        self.members: List[Any] = []
        for i in range(n_members):
            # Seeded per-member (not left to whatever the global RNG stream
            # happens to be) so ensemble diversity — and any test asserting
            # on it — is reproducible across runs.
            torch.manual_seed(base_seed + i)
            self.members.append(model_factory())

    def fit(
        self,
        coords: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        loss_fn: Optional[Callable[[Any, torch.Tensor], torch.Tensor]] = None,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> List[List[float]]:
        """Trains every member independently (each already has different
        initial weights from __init__) on the same data — the only source
        of ensemble diversity here is initialization + optimization path,
        not bagging/bootstrapping (a legitimate, simpler deep-ensemble
        variant per Lakshminarayanan et al. 2017)."""
        if loss_fn is None:
            if targets is None:
                raise ValueError("DeepEnsemble.fit() needs `targets` or a custom `loss_fn`.")
            loss_fn = lambda m, c: F.mse_loss(_unwrap(m(c)), targets)

        histories: List[List[float]] = []
        for member in self.members:
            optimizer = torch.optim.Adam(member.parameters(), lr=lr)
            member.train()
            hist: List[float] = []
            for _ in range(epochs):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(member, coords)
                loss.backward()
                optimizer.step()
                hist.append(float(loss.item()))
            histories.append(hist)
        return histories

    def predict(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, std) across members — std is epistemic uncertainty
        from model disagreement, not from any single model's own noise."""
        for m in self.members:
            m.eval()
        with torch.no_grad():
            preds = torch.stack([_unwrap(m(coords)) for m in self.members], dim=0)
        return preds.mean(dim=0), preds.std(dim=0)


def _unwrap(out: Any) -> torch.Tensor:
    return out.y if hasattr(out, "y") else out
