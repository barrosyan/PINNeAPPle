from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from pinneaple_models.registry import ModelSpec


class AutoEncoderAdapter:
    """Adapter for autoencoders.

    Batch convention:
      - 'x': input to reconstruct
      - prediction output can be 'recon'/'x_hat'/'y' depending on family
    """

    def can_handle(self, spec: ModelSpec) -> bool:
        return spec.family in ("autoencoders", "rom") or (spec.input_kind or "") == "autoencoder"

    def forward_batch(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        if hasattr(model, "forward_batch"):
            return model.forward_batch(batch)  # type: ignore[attr-defined]
        x = batch.get("x")
        if x is None:
            raise KeyError("AutoEncoderAdapter expects batch['x'].")
        return model(x)
