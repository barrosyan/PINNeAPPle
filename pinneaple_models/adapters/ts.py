from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from pinneaple_models.registry import ModelSpec


class TimeSeriesAdapter:
    """Adapter for time-series models.

    Batch convention (typical):
      - 'x': (B, T, D) history
      - optionally 'x_mark', 'y_mark', 'y0', etc
    """

    def can_handle(self, spec: ModelSpec) -> bool:
        return (spec.input_kind or "") == "sequence" or spec.family in (
            "transformers",
            "recurrent",
            "classical_ts",
            "reservoir_computing",
        )

    def forward_batch(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        if hasattr(model, "forward_batch"):
            return model.forward_batch(batch)  # type: ignore[attr-defined]
        x = batch.get("x")
        if x is None:
            raise KeyError("TimeSeriesAdapter expects batch['x'].")

        # Try richer signatures first
        for keys in (
            ("x", "x_mark", "y_mark"),
            ("x", "x_mark"),
        ):
            if all(k in batch for k in keys):
                try:
                    return model(*[batch[k] for k in keys])
                except TypeError:
                    pass
        return model(x)
