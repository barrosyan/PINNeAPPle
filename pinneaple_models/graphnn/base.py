from __future__ import annotations
"""Base classes and GraphBatch for graph neural network models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class GraphBatch:
    """
    Minimal graph batch format (no external deps).

    x:          (B, N, node_dim) node features
    pos:        (B, N, pos_dim) optional node coordinates
    edge_index: (2, E) global edges, applied per-graph (shared topology) OR per-batch if you pre-offset
    edge_attr:  (B, E, edge_dim) optional edge features (can be None)
    mask:       (B, N) optional node mask (1 valid, 0 padding)
    """
    x: torch.Tensor
    edge_index: torch.Tensor
    pos: Optional[torch.Tensor] = None
    edge_attr: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None


@dataclass
class GraphOutput:
    """
    Standard output for GNN family.
    y:
      - node-level: (B, N, out_dim)
      - graph-level: (B, out_dim) (not used in this MVP)
    """
    y: torch.Tensor
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


class GraphModelBase(nn.Module):
    """
    Base for graph models.

    MVP uses a simple message passing core (no torch_geometric).
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def mse(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.mean((y_hat - y) ** 2)
