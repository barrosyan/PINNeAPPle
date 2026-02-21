from __future__ import annotations
"""Spatiotemporal graph neural network for space-time data."""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import GraphModelBase, GraphBatch, GraphOutput
from .gnn import GraphNeuralNetwork


class SpatiotemporalGNN(GraphModelBase):
    """
    Spatiotemporal GNN (MVP):

    - Spatial encoder: GNN at each time step
    - Temporal model: GRU over time per-node
    - Head: node-level output per time

    Inputs:
      x: (B,T,N,node_in)
      edge_index: shared across time
    Output:
      y: (B,T,N,out_dim)
    """
    def __init__(
        self,
        node_in: int,
        out_dim: int,
        *,
        edge_dim: int = 0,
        spatial_hidden: int = 128,
        spatial_layers: int = 3,
        temporal_hidden: int = 128,
        temporal_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spatial = GraphNeuralNetwork(
            node_dim=node_in,
            out_dim=spatial_hidden,
            edge_dim=edge_dim,
            hidden=spatial_hidden,
            layers=spatial_layers,
            dropout=dropout,
        )
        self.temporal = nn.GRU(
            input_size=spatial_hidden,
            hidden_size=temporal_hidden,
            num_layers=int(temporal_layers),
            batch_first=True,
            dropout=float(dropout) if temporal_layers > 1 else 0.0,
        )
        self.head = nn.Linear(temporal_hidden, out_dim)

    def forward(
        self,
        x: torch.Tensor,              # (B,T,N,node_in)
        edge_index: torch.Tensor,     # (2,E)
        *,
        pos: Optional[torch.Tensor] = None,       # (B,N,pos_dim) optional
        edge_attr: Optional[torch.Tensor] = None, # (B,E,edge_dim) optional
        mask: Optional[torch.Tensor] = None,      # (B,N)
        y_true: Optional[torch.Tensor] = None,    # (B,T,N,out_dim)
        return_loss: bool = False,
    ) -> GraphOutput:
        B, T, N, Din = x.shape

        spatial_feats = []
        for t in range(T):
            g = GraphBatch(
                x=x[:, t, :, :],
                edge_index=edge_index,
                pos=pos,
                edge_attr=edge_attr,
                mask=mask,
            )
            h = self.spatial(g).y  # (B,N,spatial_hidden)
            spatial_feats.append(h)

        H = torch.stack(spatial_feats, dim=1)  # (B,T,N,Hs)

        # GRU per node: reshape to (B*N, T, Hs)
        H2 = H.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        out, _ = self.temporal(H2)  # (B*N,T,Ht)
        y = self.head(out)          # (B*N,T,out_dim)
        y = y.reshape(B, N, T, -1).permute(0, 2, 1, 3)  # (B,T,N,out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y.device)}
        if return_loss and y_true is not None:
            if mask is not None:
                m = mask[:, None, :, None].to(y.dtype)
                losses["mse"] = torch.mean(((y - y_true) ** 2) * m)
            else:
                losses["mse"] = torch.mean((y - y_true) ** 2)
            losses["total"] = losses["mse"]

        return GraphOutput(y=y, losses=losses, extras={})
