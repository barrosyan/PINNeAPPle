from __future__ import annotations
"""GraphCast-style mesh-based forecasting architecture."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import GraphModelBase, GraphBatch, GraphOutput
from .utils import scatter_add


class _ProcessorBlock(nn.Module):
    def __init__(self, hidden: int, edge_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.edge_dim = int(edge_dim)

        msg_in = 2 * hidden + 1 + (edge_dim if edge_dim > 0 else 0)
        self.phi_e = nn.Sequential(
            nn.Linear(msg_in, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.phi_v = nn.Sequential(
            nn.Linear(hidden + hidden, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, h, pos, edge_index, edge_attr=None):
        B, N, H = h.shape
        src, dst = edge_index[0], edge_index[1]

        hi = h[:, dst, :]
        hj = h[:, src, :]

        if pos is None:
            dij2 = torch.zeros((B, src.numel(), 1), device=h.device, dtype=h.dtype)
        else:
            pi = pos[:, dst, :]
            pj = pos[:, src, :]
            dij2 = ((pi - pj) ** 2).sum(dim=-1, keepdim=True)

        if self.edge_dim > 0:
            if edge_attr is None:
                raise ValueError("edge_dim > 0 but edge_attr is None")
            e_in = torch.cat([hi, hj, dij2, edge_attr], dim=-1)
        else:
            e_in = torch.cat([hi, hj, dij2], dim=-1)

        e = self.phi_e(e_in)                    # (B,E,H)
        agg = scatter_add(e, dst, dim_size=N)   # (B,N,H)
        h = h + self.phi_v(torch.cat([h, agg], dim=-1))
        return h


class GraphCast(GraphModelBase):
    """
    GraphCast-inspired MVP:

      - Node encoder
      - Processor: K message-passing blocks
      - Node decoder

    Inputs:
      g.x: (B,N,node_in)
      g.pos: (B,N,pos_dim) optional
      g.edge_attr: (B,E,edge_dim) optional
    Output:
      y: (B,N,out_dim)
    """
    def __init__(
        self,
        node_in: int,
        out_dim: int,
        *,
        hidden: int = 256,
        processor_blocks: int = 8,
        edge_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.edge_dim = int(edge_dim)

        self.enc = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.proc = nn.ModuleList([
            _ProcessorBlock(hidden=hidden, edge_dim=edge_dim, dropout=dropout)
            for _ in range(int(processor_blocks))
        ])
        self.dec = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(
        self,
        g: GraphBatch,
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,N,out_dim)
        return_loss: bool = False,
    ) -> GraphOutput:
        x, edge_index = g.x, g.edge_index
        h = self.enc(x)

        for blk in self.proc:
            h = blk(h, g.pos, edge_index, g.edge_attr)

        y = self.dec(h)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            if g.mask is not None:
                mask = g.mask[..., None].to(y.dtype)
                losses["mse"] = torch.mean(((y - y_true) ** 2) * mask)
            else:
                losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return GraphOutput(y=y, losses=losses, extras={"h": h})
