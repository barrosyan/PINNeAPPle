from __future__ import annotations
"""Equivariant graph neural network for symmetry-preserving learning."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import GraphModelBase, GraphBatch, GraphOutput


def _scatter_add_batched(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """
    Batched scatter_add over node dimension.

    src:   (B, E, C)
    index: (E,) long, values in [0, dim_size)
    out:   (B, N, C)
    """
    if index.dtype != torch.long:
        index = index.long()
    B, E, C = src.shape
    out = src.new_zeros((B, dim_size, C))
    idx = index.view(1, E, 1).expand(B, E, C)
    return out.scatter_add(1, idx, src)


def _scatter_mean_batched(src: torch.Tensor, index: torch.Tensor, dim_size: int, eps: float = 1e-8) -> torch.Tensor:
    """
    Batched scatter_mean over node dimension.

    src:   (B, E, C)
    index: (E,) long
    out:   (B, N, C)
    """
    B, E, C = src.shape
    ones = src.new_ones((B, E, 1))
    num = _scatter_add_batched(src, index, dim_size)
    den = _scatter_add_batched(ones, index, dim_size).clamp_min(eps)  # (B,N,1)
    return num / den


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
        pos_step_init: float = 0.1,  # recommended for stability
        eps: float = 1e-8,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.pos_dim = int(pos_dim)
        self.edge_dim = int(edge_dim)
        self.hidden = int(hidden)
        self.update_pos = bool(update_pos)
        self.eps = float(eps)

        self.h_in = nn.Linear(node_dim, hidden)

        m_in = 2 * hidden + 1 + (edge_dim if edge_dim > 0 else 0)  # +||p_i-p_j||^2
        self.phi_e = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(m_in, hidden),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                )
                for _ in range(int(layers))
            ]
        )

        self.phi_h = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden + hidden, hidden),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                )
                for _ in range(int(layers))
            ]
        )

        # scalar for position update magnitude
        self.phi_x = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                for _ in range(int(layers))
            ]
        )

        # learnable global step for position updates (stability)
        self.pos_step = nn.Parameter(torch.tensor(float(pos_step_init)))

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
        src, dst = edge_index[0], edge_index[1]  # (E,), (E,)

        h = self.h_in(x)
        eattr = g.edge_attr

        # Node mask for padded graphs (recommended):
        # g.mask expected shape (B,N) with 1/0 or bool.
        node_mask: Optional[torch.Tensor] = None
        if g.mask is not None:
            node_mask = g.mask.to(dtype=h.dtype, device=h.device)  # (B,N)
            # ensure padded nodes stay neutral
            h = h * node_mask[..., None]
            p = p * node_mask[..., None]

        for phi_e, phi_h, phi_x in zip(self.phi_e, self.phi_h, self.phi_x):
            hi = h[:, dst, :]  # (B,E,H)
            hj = h[:, src, :]
            pi = p[:, dst, :]  # (B,E,P)
            pj = p[:, src, :]
            rij = pi - pj
            dij2 = (rij**2).sum(dim=-1, keepdim=True)  # (B,E,1)

            if self.edge_dim > 0:
                if eattr is None:
                    raise ValueError("edge_dim > 0 but g.edge_attr is None")
                e_in = torch.cat([hi, hj, dij2, eattr], dim=-1)
            else:
                e_in = torch.cat([hi, hj, dij2], dim=-1)

            e = phi_e(e_in)  # (B,E,H)

            # Edge mask derived from node mask (prevents padding from contributing)
            # edge_mask: (B,E,1)
            if node_mask is not None:
                em = (node_mask[:, src] * node_mask[:, dst]).unsqueeze(-1)
                e = e * em
            else:
                em = None

            # Degree-normalized aggregation (mean) for stability across varying degrees
            agg_e = _scatter_mean_batched(e, dst, dim_size=N, eps=self.eps)  # (B,N,H)

            # update h (residual)
            h = h + phi_h(torch.cat([h, agg_e], dim=-1))

            if node_mask is not None:
                h = h * node_mask[..., None]  # keep padded nodes neutral

            # update p (equivariant), with bounded scalar + step size
            if self.update_pos:
                # bounded scalar (common stability trick)
                s = torch.tanh(phi_x(e))  # (B,E,1)

                dp_msg = s * rij  # (B,E,P)

                if em is not None:
                    dp_msg = dp_msg * em  # zero out masked edges

                # degree-normalized position update
                dp = _scatter_mean_batched(dp_msg, dst, dim_size=N, eps=self.eps)  # (B,N,P)

                # global step (learnable)
                p = p + self.pos_step * dp

                if node_mask is not None:
                    p = p * node_mask[..., None]  # keep padded nodes neutral

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
