from __future__ import annotations
"""Equivariant graph neural network for symmetry-preserving learning."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import GraphModelBase, GraphBatch, GraphOutput
from .utils import scatter_add


class EquivariantGNN(GraphModelBase):
    """
    EGNN-style MVP (E(n)-equivariant):

    - Maintains node embeddings h and positions p.
    - Updates p using learned, distance-based messages (translation equivariant).

    This is a practical scaffold for meshes/particle systems.
    """
    def __init__(
        self,
        node_dim: int,
        pos_dim: int,
        out_dim: int,
        *,
        hidden: int = 128,
        layers: int = 4,
        edge_dim: int = 0,
        dropout: float = 0.0,
        update_pos: bool = True,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.pos_dim = int(pos_dim)
        self.edge_dim = int(edge_dim)
        self.hidden = int(hidden)
        self.update_pos = bool(update_pos)

        self.h_in = nn.Linear(node_dim, hidden)

        m_in = 2 * hidden + 1 + (edge_dim if edge_dim > 0 else 0)  # +||p_i-p_j||^2
        self.phi_e = nn.ModuleList([
            nn.Sequential(
                nn.Linear(m_in, hidden),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            for _ in range(int(layers))
        ])

        self.phi_h = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden + hidden, hidden),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden, hidden),
                nn.SiLU(),
            )
            for _ in range(int(layers))
        ])

        # scalar for position update magnitude
        self.phi_x = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
            )
            for _ in range(int(layers))
        ])

        self.out = nn.Linear(hidden, out_dim)

    def forward(
        self,
        g: GraphBatch,
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> GraphOutput:
        if g.pos is None:
            raise ValueError("EquivariantGNN requires g.pos")
        x, p, edge_index = g.x, g.pos, g.edge_index
        B, N, _ = x.shape
        src, dst = edge_index[0], edge_index[1]

        h = self.h_in(x)
        eattr = g.edge_attr

        for phi_e, phi_h, phi_x in zip(self.phi_e, self.phi_h, self.phi_x):
            hi = h[:, dst, :]  # (B,E,H)
            hj = h[:, src, :]
            pi = p[:, dst, :]  # (B,E,P)
            pj = p[:, src, :]
            rij = pi - pj
            dij2 = (rij ** 2).sum(dim=-1, keepdim=True)  # (B,E,1)

            if self.edge_dim > 0:
                if eattr is None:
                    raise ValueError("edge_dim > 0 but g.edge_attr is None")
                e_in = torch.cat([hi, hj, dij2, eattr], dim=-1)
            else:
                e_in = torch.cat([hi, hj, dij2], dim=-1)

            e = phi_e(e_in)  # (B,E,H)

            # update h
            agg_e = scatter_add(e, dst, dim_size=N)  # (B,N,H)
            h = h + phi_h(torch.cat([h, agg_e], dim=-1))

            # update p (equivariant)
            if self.update_pos:
                s = phi_x(e)  # (B,E,1)
                dp_msg = s * rij  # (B,E,P)
                dp = scatter_add(dp_msg, dst, dim_size=N)  # (B,N,P)
                p = p + dp

        y = self.out(h)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            if g.mask is not None:
                mask = g.mask[..., None].to(y.dtype)
                losses["mse"] = torch.mean(((y - y_true) ** 2) * mask)
            else:
                losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return GraphOutput(y=y, losses=losses, extras={"h": h, "pos": p})
