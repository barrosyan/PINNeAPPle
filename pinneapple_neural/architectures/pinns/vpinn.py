from __future__ import annotations
"""Variational PINN with weak form and finite element discretization."""

from typing import Any, Dict, Optional, Callable, List, Tuple, Union

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput  # single source of truth


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
