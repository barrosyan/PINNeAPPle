from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from pinneapple_neural.architectures.registry import ModelSpec


class DynamicsAdapter:
    """Adapter for time-evolution / neural-ODE-style models.

    Batch convention:
      - 'x0': initial state, (B, state_dim)
      - 't' : time points to roll out over, (T,) or (B, T)
    Matches forward(x0_or_y0, t, ...) on neural_ode.py/latent_ode.py/
    ode_rnn.py/neural_cde.py/neural_sde.py/symplectic_rnn.py.
    """

    def can_handle(self, spec: ModelSpec) -> bool:
        return (spec.input_kind or "") == "dynamics"

    def forward_batch(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        if hasattr(model, "forward_batch"):
            return model.forward_batch(batch)
        x0 = batch.get("x0")
        t = batch.get("t")
        if x0 is None or t is None:
            raise KeyError("DynamicsAdapter requires 'x0' and 't' in batch.")
        y_true = batch.get("y_true")
        return model(x0, t, y_true=y_true, return_loss=(y_true is not None))
