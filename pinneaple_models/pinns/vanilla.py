from __future__ import annotations
"""Vanilla PINN with strong-form PDE residuals."""

from typing import Dict, List, Optional, Callable, Any, Tuple

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput


def _act(name: str) -> nn.Module:
    name = (name or "tanh").lower()
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}.get(name, nn.Tanh())


class VanillaPINN(PINNBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: List[int] = (128, 128, 128, 128),
        activation: str = "tanh",
        *,
        inverse_params_names: Optional[List[str]] = None,
        initial_guesses: Optional[Dict[str, float]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        act = _act(activation)

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        dims = [self.in_dim, *list(hidden), self.out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
        self.net = nn.Sequential(*layers)

        # --- required by PINNFactory loss_fn ---
        self.inverse_params = nn.ParameterDict()
        if inverse_params_names:
            initial_guesses = initial_guesses or {}
            for name in inverse_params_names:
                init = float(initial_guesses.get(name, 0.1))
                self.inverse_params[name] = nn.Parameter(torch.tensor(init, dtype=dtype))

    def _concat_inputs(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(inputs) == 0:
            raise ValueError("VanillaPINN.forward expected at least 1 input tensor.")
        if len(inputs) == 1:
            x = inputs[0]
            if x.ndim == 1:
                x = x[:, None]
            return x
        cols = []
        for t in inputs:
            if t.ndim == 1:
                t = t[:, None]
            cols.append(t)
        return torch.cat(cols, dim=1)

    def forward(
        self,
        *inputs: torch.Tensor,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
    ) -> PINNOutput:
        x = self._concat_inputs(inputs)

        if (physics_fn is not None and physics_data is not None) and (not x.requires_grad):
            x = x.requires_grad_(True)

        y = self.net(x)

        z0 = torch.zeros((), device=y.device, dtype=y.dtype)
        losses: Dict[str, torch.Tensor] = {"total": z0}

        if physics_fn is not None and physics_data is not None:
            total_phys, comps = physics_fn(self, physics_data)

            losses["physics"] = total_phys
            for k, v in comps.items():
                if k == "total":
                    continue
                losses[k] = torch.as_tensor(v, device=y.device, dtype=y.dtype)

            losses["total"] = losses["total"] + losses["physics"]

        return PINNOutput(y=y, losses=losses, extras={})
