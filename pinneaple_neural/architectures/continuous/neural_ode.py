from __future__ import annotations
"""Neural ordinary differential equation models with RK4 integration."""
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


def _rk4_step(f, t, y, dt):
    # Expect t and dt already on y.device / y.dtype (handled in forward)
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class NeuralODE(ContinuousModelBase):
    """
    Neural ODE MVP:
      dy/dt = f(t, y) where f is a neural net.
      Integrate over time grid with Euler or RK4.
    """
    def __init__(
        self,
        state_dim: int,
        hidden: int = 128,
        num_layers: int = 3,
        method: Literal["euler", "rk4"] = "rk4",
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.method = method

        layers = [nn.Linear(state_dim + 1, hidden), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, state_dim)]
        self.f = nn.Sequential(*layers)

    def dynamics(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # t: scalar tensor or (B,1), y:(B,D)
        if t.ndim == 0:
            t = t.view(1, 1).expand(y.shape[0], 1)
        elif t.ndim == 1:
            t = t[:, None]
        inp = torch.cat([t.to(y.device, y.dtype), y], dim=-1)
        return self.f(inp)

    def forward(
        self,
        y0: torch.Tensor,                     # (B,D)
        t: torch.Tensor,                      # (T,) increasing
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,T,D)
        return_loss: bool = False,
    ) -> ContOutput:
        B, D = y0.shape
        T = t.numel()

        # Ensure time grid is on same device/dtype as state for safe arithmetic in solvers
        t_ = t.to(device=y0.device, dtype=y0.dtype)

        ys = [y0]
        y = y0

        for i in range(T - 1):
            ti = t_[i]                        # scalar on y.device/y.dtype
            dt = t_[i + 1] - t_[i]            # scalar on y.device/y.dtype

            if self.method == "euler":
                y = y + dt * self.dynamics(ti, y)
            else:
                y = _rk4_step(self.dynamics, ti, y, dt)

            ys.append(y)

        y_path = torch.stack(ys, dim=1)  # (B,T,D)

        losses: Dict[str, torch.Tensor] = {
            "total": torch.zeros((), device=y_path.device, dtype=y_path.dtype)
        }
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y_path, y_true)
            losses["total"] = losses["mse"]

        # Keep original t for extras (but ensure it's usable downstream if needed)
        return ContOutput(y=y_path, losses=losses, extras={"t": t_})
