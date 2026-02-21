from __future__ import annotations
"""Message passing graph neural network."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import GraphModelBase, GraphBatch, GraphOutput
from .utils import scatter_add


class GraphNeuralNetwork(GraphModelBase):
    """
    Message Passing Neural Network (MVP).

    For each layer:
      m_ij = phi_m([h_i, h_j, e_ij])
      agg_i = sum_{j->i} m_ij
      h_i = phi_u([h_i, agg_i])
    """
    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        *,
        edge_dim: int = 0,
        hidden: int = 128,
        layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden = int(hidden)

        self.in_proj = nn.Linear(node_dim, hidden)

        msg_in = 2 * hidden + (edge_dim if edge_dim > 0 else 0)
        self.phi_m = nn.ModuleList([
            nn.Sequential(
                nn.Linear(msg_in, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            for _ in range(int(layers))
        ])

        upd_in = hidden + hidden
        self.phi_u = nn.ModuleList([
            nn.Sequential(
                nn.Linear(upd_in, hidden),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            for _ in range(int(layers))
        ])

        self.out_proj = nn.Linear(hidden, out_dim)

    def forward(
        self,
        g: GraphBatch,
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,N,out_dim)
        return_loss: bool = False,
    ) -> GraphOutput:
        x, edge_index = g.x, g.edge_index
        B, N, _ = x.shape
        src, dst = edge_index[0], edge_index[1]  # (E,)

        h = self.in_proj(x)  # (B,N,H)

        eattr = g.edge_attr
        for phi_m, phi_u in zip(self.phi_m, self.phi_u):
            h_src = h[:, src, :]  # (B,E,H)
            h_dst = h[:, dst, :]  # (B,E,H)

            if self.edge_dim > 0:
                if eattr is None:
                    raise ValueError("edge_dim > 0 but g.edge_attr is None")
                m_in = torch.cat([h_dst, h_src, eattr], dim=-1)
            else:
                m_in = torch.cat([h_dst, h_src], dim=-1)

            m = phi_m(m_in)                       # (B,E,H)
            agg = scatter_add(m, dst, dim_size=N)  # (B,N,H)

            u_in = torch.cat([h, agg], dim=-1)
            dh = phi_u(u_in)
            h = h + dh

        y = self.out_proj(h)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            if g.mask is not None:
                mask = g.mask[..., None].to(y.dtype)
                losses["mse"] = torch.mean(((y - y_true) ** 2) * mask)
            else:
                losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return GraphOutput(y=y, losses=losses, extras={"h": h})
