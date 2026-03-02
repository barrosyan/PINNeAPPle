from __future__ import annotations

from typing import Any, Dict
import torch.nn as nn

from pinneaple_models.registry import ModelSpec


class OperatorAdapter:
    """Forward adapter for Neural Operators.

    Supports:
      - DeepONet/MS-DeepONet: (u_branch, coords) -> (B,N,out)
      - FNO-1D: u_grid_1d (B,C,L) -> (B,Cout,L)
      - GNO: u_points (B,N,Cin) + coords_points (B,N,d) -> (B,N,out)
      - UNO: grid or points mode
      - PINO wrapper: physics_fn + physics_data optional
    """

    def can_handle(self, spec: ModelSpec) -> bool:
        return (spec.family or "") == "neural_operators"

    def forward_batch(self, model: nn.Module, batch: Dict[str, Any]) -> Any:
        y_true = batch.get("y_true", None)

        # DeepONet / MultiScaleDeepONet
        if ("u_branch" in batch) and ("coords" in batch):
            return model(batch["u_branch"], batch["coords"], y_true=y_true, return_loss=(y_true is not None))

        # GNO / mesh points
        if ("u_points" in batch) and ("coords_points" in batch):
            return model(batch["u_points"], coords=batch["coords_points"], w=batch.get("w", None),
                         y_true=y_true, return_loss=(y_true is not None))

        # UNO / grid
        if "u_grid" in batch:
            return model(batch["u_grid"], coords=batch.get("coords_points", None),
                         y_true=y_true, return_loss=(y_true is not None))

        # FNO 1D / grid_1d
        if "u_grid_1d" in batch:
            return model(batch["u_grid_1d"], y_true=y_true, return_loss=(y_true is not None))

        # PINO wrapper generic path
        if "u" in batch:
            return model(batch["u"], physics_fn=batch.get("physics_fn", None), physics_data=batch.get("physics_data", None))

        raise KeyError("OperatorAdapter: unsupported batch keys for this operator.")
