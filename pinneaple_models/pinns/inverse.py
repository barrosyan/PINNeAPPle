from __future__ import annotations

from typing import Dict, List, Optional, Callable, Any, Sequence

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput
from .vanilla import _act


class _FactoryPINNAdapter(nn.Module):
    """
    Adapter to make catalog-style models compatible with PINNFactory.

    Requirements from PINNFactory / DerivativeComputer:
      - model(*inputs) returns Tensor (B, out_dim)
      - model.inverse_params is a nn.ParameterDict with required names
    """
    def __init__(self, core: "InversePINN", num_inputs: int):
        super().__init__()
        self.core = core
        self.num_inputs = int(num_inputs)

    @property
    def inverse_params(self) -> nn.ParameterDict:
        return self.core.inverse_params

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) == 1:
            x = inputs[0]
        else:
            # PINNFactory passes inputs as a tuple of tensors (B,1) in the order of spec.independent_vars
            x = torch.cat(inputs, dim=1)

        if x.shape[1] != self.num_inputs:
            raise ValueError(
                f"Adapter expected x with {self.num_inputs} features, got shape {tuple(x.shape)}"
            )
        return self.core.net(x)


class InversePINN(PINNBase):
    """
    Catalog InversePINN (returns PINNOutput), with a compatibility hook for PINNFactory.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: List[int] = (128, 128, 128, 128),
        activation: str = "tanh",
        inverse_params: Optional[List[str]] = None,
        initial_guesses: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        act = _act(activation)

        dims = [int(in_dim), *list(hidden), int(out_dim)]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
        self.net = nn.Sequential(*layers)

        self.inverse_params = nn.ParameterDict()
        if inverse_params:
            initial_guesses = initial_guesses or {}
            for name in inverse_params:
                v0 = float(initial_guesses.get(name, 0.1))
                self.inverse_params[name] = nn.Parameter(torch.tensor(v0, dtype=torch.float32))

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

    def as_factory_model(self, *, independent_vars: Sequence[str]) -> nn.Module:
        """
        Returns an nn.Module compatible with PINNFactory:
          - forward(*inputs) -> Tensor
          - exposes .inverse_params

        independent_vars is only used for sanity (len check).
        """
        if len(independent_vars) != self.in_dim:
            raise ValueError(
                f"Expected len(independent_vars) == in_dim ({self.in_dim}), got {len(independent_vars)}"
            )
        return _FactoryPINNAdapter(self, num_inputs=self.in_dim)

    def forward(
        self,
        x: torch.Tensor,
        *,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
    ) -> PINNOutput:
        y = self.net(x)

        total = torch.zeros((), device=y.device, dtype=y.dtype)
        losses: Dict[str, torch.Tensor] = {"total": total}

        if physics_fn is not None and physics_data is not None:
            pd = dict(physics_data)
            pd["x"] = x
            pd["y"] = y
            pd["inverse_params"] = self.inverse_params

            pl = self.physics_loss(physics_fn=physics_fn, physics_data=pd)
            losses.update(pl)
            losses["total"] = losses["total"] + losses.get(
                "physics", torch.zeros((), device=y.device, dtype=y.dtype)
            )

        extras = {"inverse_params": {k: v.detach() for k, v in self.inverse_params.items()}}
        return PINNOutput(y=y, losses=losses, extras=extras)
