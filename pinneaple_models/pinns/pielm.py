from __future__ import annotations
"""Physics-informed extreme learning machine for fast PINN training."""

from typing import Dict, List, Optional, Callable, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PINNBase, PINNOutput


class PIELM(PINNBase):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        activation: str = "tanh",
        freeze_random: bool = True,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)

        act = (activation or "tanh").lower()
        self.phi = {
            "tanh": torch.tanh,
            "relu": torch.relu,
            "gelu": F.gelu,
            "silu": F.silu,
        }.get(act, torch.tanh)

        self.W = nn.Parameter(torch.randn(hidden_dim, in_dim) * (1.0 / (in_dim ** 0.5)))
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.Beta = nn.Parameter(torch.zeros(hidden_dim, out_dim))

        if freeze_random:
            self.W.requires_grad_(False)
            self.b.requires_grad_(False)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = x @ self.W.t() + self.b.unsqueeze(0)
        return self.phi(h)

    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return h @ self.Beta

    @torch.no_grad()
    def fit_ridge(self, x: torch.Tensor, y: torch.Tensor, l2: float = 1e-6) -> None:
        H = self.hidden(x)
        N, Hdim = H.shape
        Y = y

        if N >= Hdim:
            A = H.t() @ H
            I = torch.eye(Hdim, device=A.device, dtype=A.dtype)
            rhs = H.t() @ Y
            Beta = torch.linalg.solve(A + float(l2) * I, rhs)
        else:
            G = H @ H.t()
            I = torch.eye(N, device=G.device, dtype=G.dtype)
            tmp = torch.linalg.solve(G + float(l2) * I, Y)
            Beta = H.t() @ tmp

        self.Beta.copy_(Beta)

    def forward(
        self,
        x: torch.Tensor,
        *,
        y_true: Optional[torch.Tensor] = None,
        data_weight: float = 1.0,
        physics_weight: float = 1.0,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
        data_reduction: str = "mean",
    ) -> PINNOutput:
        h = self.hidden(x)
        y = h @ self.Beta

        device = y.device
        losses: Dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=device)

        if y_true is not None:
            if data_reduction == "sum":
                data_loss = F.mse_loss(y, y_true, reduction="sum")
            else:
                data_loss = F.mse_loss(y, y_true, reduction="mean")
            losses["data"] = data_loss
            total = total + float(data_weight) * data_loss

        if physics_fn is not None and physics_data is not None:
            pl = self.physics_loss(physics_fn=physics_fn, physics_data=physics_data, x=x, y=y)
            losses.update(pl)
            total = total + float(physics_weight) * losses.get("physics", torch.tensor(0.0, device=device))

        losses["total"] = total
        return PINNOutput(y=y, losses=losses, extras={"hidden": h})


class PIELMFactoryAdapter(nn.Module):

    def __init__(
        self,
        pielm: PIELM,
        *,
        inverse_params_names: Optional[List[str]] = None,
        initial_guesses: Optional[Dict[str, float]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.pielm = pielm

        self.inverse_params = nn.ParameterDict()
        if inverse_params_names:
            initial_guesses = initial_guesses or {}
            for name in inverse_params_names:
                init = float(initial_guesses.get(name, 0.1))
                self.inverse_params[name] = nn.Parameter(torch.tensor(init, dtype=dtype))

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x = torch.cat(inputs, dim=1)
        if hasattr(self.pielm, "forward_tensor"):
            return self.pielm.forward_tensor(x)
        h = self.pielm.hidden(x)
        return h @ self.pielm.Beta

    @torch.no_grad()
    def fit_ridge_from_inputs(
        self,
        inputs: Tuple[torch.Tensor, ...],
        y: torch.Tensor,
        l2: float = 1e-6,
    ) -> None:
        x = torch.cat(inputs, dim=1)
        self.pielm.fit_ridge(x, y, l2=l2)
