from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import torch
import torch.nn as nn

from pinneaple_models.base import ModelOutput
from pinneaple_models.registry import ModelSpec


class ModelAdapter(Protocol):
    """Protocol for model adapters."""

    def can_handle(self, spec: ModelSpec) -> bool:
        ...

    def forward_batch(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        ...


@dataclass
class _DefaultAdapter:
    """Fallback adapter: try model.forward_batch else model(x)."""

    def can_handle(self, spec: ModelSpec) -> bool:
        return True

    def forward_batch(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        if hasattr(model, "forward_batch"):
            return model.forward_batch(batch)  # type: ignore[attr-defined]
        x = batch.get("x") or batch.get("x_col")
        if x is None:
            raise KeyError("Batch must include 'x' or 'x_col' for default adapter.")
        return model(x)


def select_adapter(spec: ModelSpec) -> ModelAdapter:
    """Pick an adapter based on model spec."""
    # Import here to avoid circular imports.
    from .pinn import PINNAdapter
    from .operators import OperatorAdapter
    from .gnn import GNNAdapter
    from .ts import TimeSeriesAdapter
    from .ae import AutoEncoderAdapter

    adapters = [
        PINNAdapter(),
        OperatorAdapter(),
        GNNAdapter(),
        TimeSeriesAdapter(),
        AutoEncoderAdapter(),
    ]
    for a in adapters:
        if a.can_handle(spec):
            return a
    return _DefaultAdapter()
