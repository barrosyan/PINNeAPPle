from __future__ import annotations
"""GraphCast-style mesh-based forecasting architecture (fixed + variable graphs)."""
from typing import Dict, Optional, List

import torch
import torch.nn as nn

from .base import GraphModelBase, GraphBatch, GraphOutput
from .utils import scatter_add_batched, scatter_add_flat, flatten_graph_batch


class _ProcessorBlock(nn.Module):
    """
    Pre-LN MPNN block:
      - Edge message MLP phi_e
      - Aggregate to dst (sum)
      - Node update MLP phi_v (residual)

    Geometry:
      uses dp = (pos_dst - pos_src) and dij2 = ||dp||^2
    """
    def __init__(self, hidden: int, pos_dim: int = 0, edge_dim: int = 0, dropout: float = 0.0):
        super().__init__()
        self.hidden = int(hidden)
        self.pos_dim = int(pos_dim)
        self.edge_dim = int(edge_dim)

        geom_dim = (self.pos_dim if self.pos_dim > 0 else 0) + 1  # dp (pos_dim) + dij2 (1)
        msg_in = 2 * hidden + geom_dim + (edge_dim if edge_dim > 0 else 0)

        self.ln_e = nn.LayerNorm(msg_in)
        self.ln_v = nn.LayerNorm(hidden + hidden)

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

    # -------- fixed/shared graph path (B,N,H) ----------
    def forward_fixed(self, h, pos, edge_index, edge_attr=None):
        B, N, H = h.shape
        src, dst = edge_index[0], edge_index[1]  # (E,)

        hi = h[:, dst, :]  # (B,E,H)
        hj = h[:, src, :]  # (B,E,H)

        if (pos is None) or (self.pos_dim <= 0):
            dp = torch.zeros((B, src.numel(), 0), device=h.device, dtype=h.dtype)
            dij2 = torch.zeros((B, src.numel(), 1), device=h.device, dtype=h.dtype)
        else:
            pi = pos[:, dst, :]  # (B,E,P)
            pj = pos[:, src, :]  # (B,E,P)
            dp = (pi - pj)       # (B,E,P)
            dij2 = (dp ** 2).sum(dim=-1, keepdim=True)  # (B,E,1)

        parts = [hi, hj, dp, dij2]
        if self.edge_dim > 0:
            if edge_attr is None:
                raise ValueError("edge_dim > 0 but edge_attr is None")
            parts.append(edge_attr)  # (B,E,edge_dim)

        e_in = torch.cat(parts, dim=-1)          # (B,E,msg_in)
        e = self.phi_e(self.ln_e(e_in))          # (B,E,H)

        agg = scatter_add_batched(e, dst, dim_size=N)  # (B,N,H)
        hv_in = torch.cat([h, agg], dim=-1)      # (B,N,2H)
        h = h + self.phi_v(self.ln_v(hv_in))     # residual
        return h

    # -------- variable graph path (N_total,H) ----------
    def forward_flat(self, h_flat, pos_flat, edge_index_flat, edge_attr_flat=None, *, n_total: int):
        # h_flat: (N_total,H)
        src, dst = edge_index_flat[0], edge_index_flat[1]  # (E_total,)

        hi = h_flat[dst, :]  # (E_total,H)
        hj = h_flat[src, :]  # (E_total,H)

        if (pos_flat is None) or (self.pos_dim <= 0):
            dp = torch.zeros((src.numel(), 0), device=h_flat.device, dtype=h_flat.dtype)
            dij2 = torch.zeros((src.numel(), 1), device=h_flat.device, dtype=h_flat.dtype)
        else:
            pi = pos_flat[dst, :]   # (E_total,P)
            pj = pos_flat[src, :]   # (E_total,P)
            dp = (pi - pj)          # (E_total,P)
            dij2 = (dp ** 2).sum(dim=-1, keepdim=True)  # (E_total,1)

        parts = [hi, hj, dp, dij2]
        if self.edge_dim > 0:
            if edge_attr_flat is None:
                raise ValueError("edge_dim > 0 but edge_attr is None")
            parts.append(edge_attr_flat)  # (E_total,edge_dim)

        e_in = torch.cat(parts, dim=-1)          # (E_total,msg_in)
        e = self.phi_e(self.ln_e(e_in))          # (E_total,H)

        agg = scatter_add_flat(e, dst, dim_size=n_total)  # (N_total,H)
        hv_in = torch.cat([h_flat, agg], dim=-1)          # (N_total,2H)
        h_flat = h_flat + self.phi_v(self.ln_v(hv_in))    # residual
        return h_flat


class GraphCast(GraphModelBase):
    """
    GraphCast-inspired MVP (fixed mesh + variable graph support).

    Fixed/shared graph inputs:
      g.x: (B,N,node_in)
      g.pos: (B,N,pos_dim) optional
      g.edge_index: (2,E) shared across batch
      g.edge_attr: (B,E,edge_dim) optional

    Variable graph inputs:
      g.x_list: List[(N_b,node_in)]
      g.pos_list: List[(N_b,pos_dim)] optional
      g.edge_index_list: List[(2,E_b)]
      g.edge_attr_list: List[(E_b,edge_dim)] optional

    Output:
      y: (B,N,out_dim) for fixed
      y_list: List[(N_b,out_dim)] for variable (returned in extras), while y is a padded tensor if possible.
    """
    def __init__(
        self,
        node_in: int,
        out_dim: int,
        *,
        hidden: int = 256,
        processor_blocks: int = 8,
        pos_dim: int = 0,
        edge_dim: int = 0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.edge_dim = int(edge_dim)
        self.pos_dim = int(pos_dim)
        self.out_dim = int(out_dim)

        self.enc = nn.Sequential(
            nn.Linear(node_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

        self.proc = nn.ModuleList([
            _ProcessorBlock(hidden=hidden, pos_dim=self.pos_dim, edge_dim=edge_dim, dropout=dropout)
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
        y_true: Optional[torch.Tensor] = None,  # fixed: (B,N,out_dim) optional
        return_loss: bool = False,
    ) -> GraphOutput:

        # -------- detect mode --------
        is_variable = hasattr(g, "x_list") and getattr(g, "x_list") is not None

        losses: Dict[str, torch.Tensor] = {}

        if not is_variable:
            # ===== fixed/shared graph =====
            x, edge_index = g.x, g.edge_index
            h = self.enc(x)

            for blk in self.proc:
                h = blk.forward_fixed(h, g.pos, edge_index, g.edge_attr)

            y = self.dec(h)

            losses["total"] = torch.tensor(0.0, device=y.device, dtype=y.dtype)
            if return_loss and (y_true is not None):
                if getattr(g, "mask", None) is not None:
                    mask = g.mask[..., None].to(y.dtype)  # (B,N,1)
                    err2 = ((y - y_true) ** 2) * mask
                    losses["mse"] = err2.sum() / mask.sum().clamp_min(1.0)
                else:
                    losses["mse"] = self.mse(y, y_true)
                losses["total"] = losses["mse"]

            return GraphOutput(y=y, losses=losses, extras={"h": h})

        # ===== variable graph (per-sample) =====
        x_list: List[torch.Tensor] = g.x_list
        edge_index_list: List[torch.Tensor] = g.edge_index_list
        pos_list = getattr(g, "pos_list", None)
        edge_attr_list = getattr(g, "edge_attr_list", None)

        x_flat, edge_index_flat, pos_flat, edge_attr_flat, ptr = flatten_graph_batch(
            x_list=x_list,
            edge_index_list=edge_index_list,
            pos_list=pos_list,
            edge_attr_list=edge_attr_list,
        )

        h_flat = self.enc(x_flat)  # (N_total,H)

        for blk in self.proc:
            h_flat = blk.forward_flat(
                h_flat,
                pos_flat,
                edge_index_flat,
                edge_attr_flat,
                n_total=h_flat.shape[0],
            )

        y_flat = self.dec(h_flat)  # (N_total,out_dim)

        # unflatten to list
        y_list = []
        for b in range(len(x_list)):
            s = ptr[b].item()
            t = ptr[b + 1].item()
            y_list.append(y_flat[s:t, :])

        # Optionally pad to a dense tensor y (B, Nmax, out_dim) so GraphOutput stays consistent
        Nmax = max(x.shape[0] for x in x_list) if len(x_list) > 0 else 0
        B = len(x_list)
        y = torch.zeros((B, Nmax, self.out_dim), device=y_flat.device, dtype=y_flat.dtype)
        mask = torch.zeros((B, Nmax), device=y_flat.device, dtype=y_flat.dtype)

        for b, yb in enumerate(y_list):
            nb = yb.shape[0]
            y[b, :nb, :] = yb
            mask[b, :nb] = 1.0

        losses["total"] = torch.tensor(0.0, device=y.device, dtype=y.dtype)
        if return_loss and (y_true is not None):
            # Here y_true is expected padded (B,Nmax,out_dim) in variable mode if you want loss here.
            err2 = ((y - y_true) ** 2) * mask[..., None]
            losses["mse"] = err2.sum() / mask.sum().clamp_min(1.0)
            losses["total"] = losses["mse"]

        return GraphOutput(
            y=y,  # padded
            losses=losses,
            extras={"h_flat": h_flat, "y_list": y_list, "ptr": ptr, "mask": mask},
        )