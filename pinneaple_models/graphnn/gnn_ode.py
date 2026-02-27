from __future__ import annotations
"""GNN-ODE hybrid for continuous-time graph dynamics (fixed-step + optional torchdiffeq)."""
from typing import Dict, Optional, Literal

import torch
import torch.nn as nn

from .base import GraphModelBase, GraphBatch, GraphOutput
from .gnn import GraphNeuralNetwork


def _rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class FourierTimeFeatures(nn.Module):
    """
    Fourier features for scalar time t.
    Produces [sin(2π f_i t), cos(2π f_i t)] features.
    """
    def __init__(self, n_frequencies: int, *, max_frequency: float = 10.0):
        super().__init__()
        self.n_frequencies = int(n_frequencies)
        if self.n_frequencies <= 0:
            raise ValueError("n_frequencies must be > 0")

        # Log-spaced frequencies from 1 .. max_frequency (common practical choice)
        freqs = torch.logspace(0.0, torch.log10(torch.tensor(float(max_frequency))), steps=self.n_frequencies)
        self.register_buffer("freqs", freqs, persistent=False)  # (F,)

    @property
    def out_dim(self) -> int:
        return 2 * self.n_frequencies

    def forward(self, t: torch.Tensor, B: int, N: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """
        t: scalar tensor (or shape [])
        returns: (B,N,2F)
        """
        tt = t.to(device=device, dtype=dtype)
        # (F,) * scalar -> (F,)
        ang = 2.0 * torch.pi * self.freqs.to(device=device, dtype=dtype) * tt
        feat = torch.cat([torch.sin(ang), torch.cos(ang)], dim=0)  # (2F,)
        return feat.view(1, 1, -1).expand(B, N, -1)  # (B,N,2F)


class GraphNeuralODE(GraphModelBase):
    """
    Graph Neural ODE (MVP):

      dH/dt = f_theta(G, H, t)
    where f_theta is a GNN producing node-wise derivatives.

    Inputs:
      g: GraphBatch with x as initial node state H(t0)  -> shape (B,N,node_dim)
      t: (T,) time grid
    Output:
      y: (B,T,N,node_dim)
    """
    def __init__(
        self,
        node_dim: int,
        *,
        edge_dim: int = 0,
        hidden: int = 128,
        layers: int = 3,
        method: Literal["euler", "rk4", "dopri5", "adams"] = "rk4",
        time_features: int = 0,            # number of Fourier frequencies (0 => autonomous ODE)
        time_max_frequency: float = 10.0,  # max Fourier frequency
        ode_rtol: float = 1e-5,
        ode_atol: float = 1e-6,
        ode_adjoint: bool = False,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.method = method

        self.ode_rtol = float(ode_rtol)
        self.ode_atol = float(ode_atol)
        self.ode_adjoint = bool(ode_adjoint)

        self.time_embed: Optional[FourierTimeFeatures] = None
        if int(time_features) > 0:
            self.time_embed = FourierTimeFeatures(int(time_features), max_frequency=float(time_max_frequency))

        in_node_dim = self.node_dim + (self.time_embed.out_dim if self.time_embed is not None else 0)

        # derivative network outputs same dim as node state
        self.f = GraphNeuralNetwork(
            node_dim=in_node_dim,
            out_dim=self.node_dim,
            edge_dim=edge_dim,
            hidden=hidden,
            layers=layers,
        )

    def forward(
        self,
        g: GraphBatch,
        t: torch.Tensor,
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,T,N,node_dim)
        return_loss: bool = False,
    ) -> GraphOutput:
        x0 = g.x
        B, N, D = x0.shape
        if D != self.node_dim:
            raise ValueError(f"Expected g.x last dim == node_dim ({self.node_dim}), got {D}")

        # Move time grid once to match state
        t = t.to(device=x0.device, dtype=x0.dtype)
        T = int(t.numel())
        if T < 1:
            raise ValueError("t must have at least one time point")

        def f_eval(tt: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
            # Optionally augment node state with time features
            if self.time_embed is not None:
                te = self.time_embed(tt, B=B, N=N, dtype=hh.dtype, device=hh.device)  # (B,N,te_dim)
                x_in = torch.cat([hh, te], dim=-1)  # (B,N,D+te_dim)
            else:
                x_in = hh  # autonomous

            gg = GraphBatch(
                x=x_in,
                edge_index=g.edge_index,
                pos=g.pos,
                edge_attr=g.edge_attr,
                mask=g.mask,
            )
            return self.f(gg).y  # (B,N,node_dim)

        # --- ODE integration ---
        if self.method in ("dopri5", "adams"):
            try:
                if self.ode_adjoint:
                    from torchdiffeq import odeint_adjoint as odeint  # type: ignore
                else:
                    from torchdiffeq import odeint  # type: ignore
            except Exception as e:
                raise ImportError(
                    "method='dopri5'/'adams' requires torchdiffeq. "
                    "Install with: pip install torchdiffeq"
                ) from e

            # torchdiffeq expects f(t, y) -> dy/dt with same shape as y
            y_path = odeint(
                func=lambda tt, yy: f_eval(tt, yy),
                y0=x0,
                t=t,
                method=self.method,
                rtol=self.ode_rtol,
                atol=self.ode_atol,
            )  # shape (T,B,N,D)
            y_path = y_path.permute(1, 0, 2, 3).contiguous()  # (B,T,N,D)

        else:
            hs = [x0]
            h = x0
            for i in range(T - 1):
                dt = t[i + 1] - t[i]  # already correct dtype/device
                ti = t[i]
                if self.method == "euler":
                    h = h + dt * f_eval(ti, h)
                else:
                    h = _rk4_step(f_eval, ti, h, dt)
                hs.append(h)
            y_path = torch.stack(hs, dim=1)  # (B,T,N,D)

        # --- Losses ---
        losses: Dict[str, torch.Tensor] = {
            "total": torch.tensor(0.0, device=y_path.device, dtype=y_path.dtype)
        }
        if return_loss and y_true is not None:
            y_true = y_true.to(device=y_path.device, dtype=y_path.dtype)

            if g.mask is not None:
                mask = g.mask[:, None, :, None].to(device=y_path.device, dtype=y_path.dtype)  # (B,1,N,1)
                err = ((y_path - y_true) ** 2) * mask
                denom = mask.sum().clamp_min(torch.tensor(1.0, device=y_path.device, dtype=y_path.dtype))
                losses["mse"] = err.sum() / denom
            else:
                losses["mse"] = torch.mean((y_path - y_true) ** 2)

            losses["total"] = losses["mse"]

        return GraphOutput(y=y_path, losses=losses, extras={})