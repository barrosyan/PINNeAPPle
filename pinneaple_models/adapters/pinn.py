from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from pinneaple_models.registry import ModelSpec


class PINNAdapter:
    """Adapter for pointwise coordinate PINNs.

    Convention:
      - prediction input is batch['x'] if present, else batch['x_col']
      - some models want (x, ctx) or (x, params). If the model implements
        forward_batch, we delegate to it.
    """

    def can_handle(self, spec: ModelSpec) -> bool:
        return (spec.input_kind or "pointwise_coords") == "pointwise_coords"

    def forward_batch(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        if hasattr(model, "forward_batch"):
            return model.forward_batch(batch)  # type: ignore[attr-defined]
        x = batch.get("x")
        if x is None:
            x = batch.get("x_col")
        if x is None:
            raise KeyError("PINNAdapter requires 'x' or 'x_col' in batch.")
        return model(x)
