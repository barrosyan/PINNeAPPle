from __future__ import annotations
"""GNN-ODE hybrid for continuous-time graph dynamics."""
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
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


class GraphNeuralODE(GraphModelBase):
    """
    Graph Neural ODE (MVP):

      dH/dt = f_theta(G, H, t)
    where f_theta is a GNN producing node-wise derivatives.

    Inputs:
      g: GraphBatch with x as initial node state H(t0)
      t: (T,) time grid
    Output:
      y: (B,T,N,out_dim) (here out_dim == node_dim by default)
    """
    def __init__(
        self,
        node_dim: int,
        *,
        edge_dim: int = 0,
        hidden: int = 128,
        layers: int = 3,
        method: Literal["euler", "rk4"] = "rk4",
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.method = method

        # derivative network outputs same dim as node state
        self.f = GraphNeuralNetwork(
            node_dim=node_dim,
            out_dim=node_dim,
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
        T = t.numel()

        hs = [x0]
        h = x0

        def f_eval(tt, hh):
            gg = GraphBatch(
                x=hh,
                edge_index=g.edge_index,
                pos=g.pos,
                edge_attr=g.edge_attr,
                mask=g.mask,
            )
            return self.f(gg).y

        for i in range(T - 1):
            dt = (t[i + 1] - t[i]).to(dtype=h.dtype, device=h.device)
            ti = t[i]
            if self.method == "euler":
                h = h + dt * f_eval(ti, h)
            else:
                h = _rk4_step(f_eval, ti, h, dt)
            hs.append(h)

        y_path = torch.stack(hs, dim=1)  # (B,T,N,D)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y_path.device)}
        if return_loss and y_true is not None:
            if g.mask is not None:
                mask = g.mask[:, None, :, None].to(y_path.dtype)
                losses["mse"] = torch.mean(((y_path - y_true) ** 2) * mask)
            else:
                losses["mse"] = torch.mean((y_path - y_true) ** 2)
            losses["total"] = losses["mse"]

        return GraphOutput(y=y_path, losses=losses, extras={})
