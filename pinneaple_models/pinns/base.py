from __future__ import annotations
"""Base classes and PINNOutput for physics-informed neural networks."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable

import torch
import torch.nn as nn


@dataclass
class PINNOutput:
    """
    Standard output for PINN-family models.
    """
    y: torch.Tensor
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class PINNBase(nn.Module):
    """
    Base class for PINN-family models in the catalog.

    Contract:
      - forward(*inputs, **kwargs) -> PINNOutput
      - physics_loss(...) optional hook:
          can be driven by pinneaple_pinn.factory loss_fn or other physics term.
    """
    def __init__(self):
        super().__init__()

    def predict(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        By default, forward returns PINNOutput; predict returns y only.
        """
        out = self.forward(*inputs, **kwargs)
        return out.y

    def physics_loss(
        self,
        *,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Default: no physics.
        If you provide physics_fn, we call it and expect (total_loss, loss_components) OR dict.
        """
        if physics_fn is None or physics_data is None:
            z = torch.tensor(0.0, device=next(self.parameters()).device)
            return {"physics": z}

        res = physics_fn(self, physics_data, **kwargs)
        if isinstance(res, tuple) and len(res) == 2:
            total, comps = res
            out = {"physics": total}
            if isinstance(comps, dict):
                for k, v in comps.items():
                    out[f"physics/{k}"] = v if torch.is_tensor(v) else torch.tensor(float(v), device=total.device)
            return out
        if isinstance(res, dict):
            # assume dict[str, tensor]
            if "physics" not in res:
                # try to pick a total
                total = None
                for k in ("total", "loss", "pde"):
                    if k in res and torch.is_tensor(res[k]):
                        total = res[k]
                        break
                if total is None:
                    total = torch.tensor(0.0, device=next(self.parameters()).device)
                res = dict(res)
                res["physics"] = total
            return res

        z = torch.tensor(0.0, device=next(self.parameters()).device)
        return {"physics": z}
