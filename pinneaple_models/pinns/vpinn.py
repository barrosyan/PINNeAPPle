from __future__ import annotations
"""Variational PINN with weak form and finite element discretization."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable, List, Tuple, Union

import torch
import torch.nn as nn


@dataclass
class PINNOutput:
    y: torch.Tensor
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class PINNBase(nn.Module):
    """
    Base class for PINN-family models in the catalog.

    Contract:
      - forward(...) -> PINNOutput
      - physics_loss(...) optional hook:
          can be driven by pinneaple_pinn.factory loss_fn or other physics term.
    """
    def __init__(self):
        super().__init__()

    def predict(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        out = self.forward(*inputs, **kwargs)
        return out.y

    def _ref_tensor(self, **kwargs) -> Optional[torch.Tensor]:
        for v in kwargs.values():
            if torch.is_tensor(v):
                return v
        for _, b in self.named_buffers(recurse=True):
            if torch.is_tensor(b):
                return b
        for p in self.parameters(recurse=True):
            return p
        return None

    def _zero_scalar(self, ref: Optional[torch.Tensor]) -> torch.Tensor:
        if ref is None:
            return torch.zeros((), device="cpu")
        return torch.zeros((), device=ref.device, dtype=ref.dtype)

    def _to_scalar_tensor(self, v: Any, ref: torch.Tensor) -> torch.Tensor:
        """
        Converte v para escalar tensor no device/dtype de ref.
        Se v for tensor não-escalar, reduz por sum().
        """
        if torch.is_tensor(v):
            t = v.to(device=ref.device, dtype=ref.dtype)
        else:
            t = torch.tensor(float(v), device=ref.device, dtype=ref.dtype)
        if t.ndim != 0:
            t = t.sum()
        return t

    def physics_loss(
        self,
        *,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Default: no physics.
        If you provide physics_fn, we call it and expect:
          - (total_loss, components_dict) OR
          - dict[str, tensor/number]

        """
        ref = self._ref_tensor(**kwargs)
        z = self._zero_scalar(ref)

        if physics_fn is None or physics_data is None:
            return {"physics": z}

        res = physics_fn(self, physics_data, **kwargs)

        # (total, comps)
        if isinstance(res, tuple) and len(res) == 2:
            total, comps = res
            total_t = self._to_scalar_tensor(total, z)
            out: Dict[str, torch.Tensor] = {"physics": total_t}
            if isinstance(comps, dict):
                for k, v in comps.items():
                    out[f"physics/{k}"] = self._to_scalar_tensor(v, total_t)
            return out

        # dict[str, tensor/number]
        if isinstance(res, dict):
            out = {k: self._to_scalar_tensor(v, z) for k, v in res.items()}
            if "physics" not in out:
                total = None
                for k in ("total", "loss", "pde", "weak"):
                    if k in out:
                        total = out[k]
                        break
                out["physics"] = total if total is not None else z
            return out

        return {"physics": z}


def _act(name: str) -> nn.Module:
    name = (name or "tanh").lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "elu":
        return nn.ELU()
    return nn.Tanh()


def pinn_factory_adapter(
    loss_fn: Callable[[nn.Module, Dict[str, Any]], Tuple[torch.Tensor, Dict[str, float]]]
) -> Callable[[nn.Module, Dict[str, Any]], Tuple[torch.Tensor, Dict[str, float]]]:
    def physics_fn(model: nn.Module, physics_data: Dict[str, Any], **kwargs):
        return loss_fn(model, physics_data)

    return physics_fn


class VPINN(PINNBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Tuple[int, ...] = (128, 128, 128),
        activation: str = "tanh",
        *,
        inverse_params_names: Optional[List[str]] = None,
        initial_guesses: Optional[Dict[str, float]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        act = _act(activation)
        dims = [self.in_dim, *list(hidden), self.out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
        self.net = nn.Sequential(*layers)

        self.inverse_params = nn.ParameterDict()
        if inverse_params_names:
            initial_guesses = initial_guesses or {}
            for name in inverse_params_names:
                init = float(initial_guesses.get(name, 0.1))
                self.inverse_params[name] = nn.Parameter(torch.tensor(init, dtype=dtype))

    def _pack_inputs(self, *inputs: torch.Tensor, x: Optional[torch.Tensor]) -> torch.Tensor:
        if x is not None:
            return x
        if len(inputs) == 0:
            raise ValueError("Provide either x=... or positional inputs (*inputs).")
        return torch.cat(inputs, dim=1)

    def forward(
        self,
        *inputs: torch.Tensor,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
        weak_fn: Optional[Callable[..., Any]] = None,
        weak_data: Optional[Dict[str, Any]] = None,
        x: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> PINNOutput:
        x_full = self._pack_inputs(*inputs, x=x)
        y = self.net(x_full)

        total0 = (y.sum() * 0.0)
        losses: Dict[str, torch.Tensor] = {"total": total0}
        extras: Dict[str, Any] = {}

        if physics_fn is None and weak_fn is not None:
            physics_fn = weak_fn
            physics_data = weak_data

        if physics_fn is not None:
            if physics_data is None:
                raise ValueError("physics_data must be provided when physics_fn is not None")

            phy = self.physics_loss(
                physics_fn=physics_fn,
                physics_data=physics_data,
                x=x_full,
                y=y,
                **kwargs,
            )

            phy_total = phy.get("physics", total0)
            for k, v in phy.items():
                if k == "physics":
                    continue
                losses[k] = v

            losses["physics"] = phy_total
            losses["total"] = losses["total"] + phy_total

        return PINNOutput(y=y, losses=losses, extras=extras)
