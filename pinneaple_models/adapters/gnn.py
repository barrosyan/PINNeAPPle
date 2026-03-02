from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from pinneaple_models.registry import ModelSpec


class GNNAdapter:
    """Adapter for graph/mesh-based models.

    Batch convention:
      - 'graph' : an object the model understands (PyG Data, custom mesh, etc.)
      - or ('x', 'edge_index', ...) depending on implementation

    For existing code, prefer implementing model.forward_batch.
    """

    def can_handle(self, spec: ModelSpec) -> bool:
        return (spec.input_kind or "") in ("graph", "mesh") or spec.family in ("graphnn",)

    def forward_batch(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        if hasattr(model, "forward_batch"):
            return model.forward_batch(batch)  # type: ignore[attr-defined]

        g = batch.get("graph")
        if g is not None:
            return model(g)

        # Fallback: try common PyG style args
        if "x" in batch and "edge_index" in batch:
            try:
                return model(batch["x"], batch["edge_index"])
            except TypeError:
                pass

        raise KeyError("GNNAdapter requires model.forward_batch or batch['graph'] or ('x','edge_index').")
