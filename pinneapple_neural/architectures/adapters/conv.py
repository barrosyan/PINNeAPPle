from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from pinneapple_neural.architectures.registry import ModelSpec


class ConvAdapter:
    """Adapter for grid/raster models (Conv1D/2D/3D, AFNO, UNO-grid-mode).

    Batch convention:
      - 'u_grid': rasterized field, channel-first (B, C, *spatial) for
        Conv1D/2D/3D, or whatever layout the model's own forward_batch expects
        (e.g. AFNO wants channel-last (B, H, W, C) via its own forward_batch).
    """

    def can_handle(self, spec: ModelSpec) -> bool:
        kind = spec.input_kind or ""
        return kind.startswith("grid") and spec.family != "neural_operators"

    def forward_batch(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        if hasattr(model, "forward_batch"):
            return model.forward_batch(batch)
        u = batch.get("u_grid")
        if u is None:
            raise KeyError("ConvAdapter requires 'u_grid' in batch.")
        y_true = batch.get("y_true")
        return model(u, y_true=y_true, return_loss=(y_true is not None))
