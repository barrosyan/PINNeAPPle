from __future__ import annotations
"""MeshGraphNet — GNN for unstructured FEM-style mesh inference.

Reference: Pfaff et al., ICLR 2021
  "Learning Mesh-Based Simulation with Graph Networks"
  https://arxiv.org/abs/2010.03409
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .base import BaseModel, ModelOutput


# ---------------------------------------------------------------------------
# Scatter helpers (no external deps)
# ---------------------------------------------------------------------------

def _scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim_size: int,
) -> torch.Tensor:
    """Scatter-sum along dim 0.

    Args:
        src: ``(E, F)`` source tensor.
        index: ``(E,)`` target node indices.
        dim_size: Number of target nodes ``N``.

    Returns:
        ``(N, F)`` aggregated tensor.
    """
    out = torch.zeros(dim_size, src.size(1), dtype=src.dtype, device=src.device)
    index_expanded = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, index_expanded, src)
    return out


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_hidden: int = 2) -> nn.Sequential:
    """Construct a small MLP with LayerNorm at the output."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    layers += [nn.Linear(hidden_dim, out_dim), nn.LayerNorm(out_dim)]
    return nn.Sequential(*layers)


class EdgeConv(nn.Module):
    """Message-passing edge convolution block.

    For each edge ``(i → j)`` computes a message from node features
    ``[h_i, h_j, e_ij]`` and aggregates at target nodes via sum-pooling.
    Node features are then updated from ``[h_j, agg_j]``.

    Args:
        node_dim: Input/output node feature dimension.
        edge_dim: Edge feature dimension (0 = no edge features).
        hidden_dim: Hidden width for message and update MLPs.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        msg_in = 2 * node_dim + edge_dim
        self.message_mlp = _mlp(msg_in, hidden_dim, node_dim)
        self.update_mlp = _mlp(2 * node_dim, hidden_dim, node_dim)

    def forward(
        self,
        h: torch.Tensor,          # (N, node_dim)
        edge_index: torch.Tensor,  # (2, E)  [src, dst]
        edge_attr: Optional[torch.Tensor],  # (E, edge_dim) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One round of message passing.

        Returns:
            Updated node features ``(N, node_dim)`` and updated edge features
            ``(E, node_dim)`` (edge embeddings produced by the message MLP).
        """
        N = h.size(0)
        src, dst = edge_index[0], edge_index[1]

        h_src = h[src]  # (E, node_dim)
        h_dst = h[dst]  # (E, node_dim)

        if edge_attr is not None:
            msg_in = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        else:
            msg_in = torch.cat([h_src, h_dst], dim=-1)

        msg = self.message_mlp(msg_in)           # (E, node_dim)
        agg = _scatter_sum(msg, dst, N)          # (N, node_dim)

        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))  # (N, node_dim)
        h_new = h_new + h  # residual

        return h_new, msg


class MeshGraphNet(BaseModel):
    """Graph neural network for unstructured mesh simulation/inference.

    Implements the MeshGraphNet architecture (Pfaff et al. 2021):

    1. **Node encoder**: linear projection of raw node features to ``hidden_dim``.
    2. **Edge encoder**: linear projection of raw edge features to ``hidden_dim``
       (optional; zeros used when no edge features provided).
    3. **Message-passing rounds**: ``n_message_passing`` rounds of
       :class:`EdgeConv`, each with residual connections and LayerNorm.
    4. **Node decoder**: linear projection to ``out_dim``.

    Input/output convention (no PyG dependency):

    * ``node_features``: ``(N, node_in_dim)`` — one row per mesh node.
    * ``edge_index``: ``(2, E)`` long tensor — source/target node pairs.
    * ``edge_features``: ``(E, edge_in_dim)`` optional edge attributes
      (e.g., relative displacement, edge length).

    Args:
        node_in_dim: Raw node feature dimension.
        edge_in_dim: Raw edge feature dimension (0 if not available).
        out_dim: Output field dimension per node.
        hidden_dim: Internal feature width.
        n_message_passing: Number of EdgeConv rounds.
    """

    family: str = "graphnn"
    name: str = "mesh_graph_net"

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        n_message_passing: int = 6,
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim

        # Encoders
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )
        if edge_in_dim > 0:
            self.edge_encoder: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(edge_in_dim, hidden_dim), nn.LayerNorm(hidden_dim)
            )
        else:
            self.edge_encoder = None

        # Message-passing blocks
        self.convs = nn.ModuleList(
            [EdgeConv(hidden_dim, hidden_dim if edge_in_dim > 0 else 0, hidden_dim)
             for _ in range(n_message_passing)]
        )

        # Decoder
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(  # type: ignore[override]
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        """Forward pass on a single graph.

        Args:
            node_features: ``(N, node_in_dim)`` node attribute tensor.
            edge_index: ``(2, E)`` long tensor of directed edges.
            edge_features: ``(E, edge_in_dim)`` optional edge attributes.

        Returns:
            :class:`~pinneaple_models.base.ModelOutput` with ``y`` of shape
            ``(N, out_dim)`` — per-node predictions.
        """
        h = self.node_encoder(node_features)  # (N, hidden_dim)

        e: Optional[torch.Tensor]
        if self.edge_encoder is not None and edge_features is not None:
            e = self.edge_encoder(edge_features)  # (E, hidden_dim)
        else:
            e = None

        for conv in self.convs:
            h, e = conv(h, edge_index, e)

        y = self.decoder(h)  # (N, out_dim)
        return ModelOutput(y=y)

    def forward_batch(self, batch: dict) -> ModelOutput:  # type: ignore[override]
        """Dict-based interface used by the Arena/Trainer.

        Expects keys:
            ``node_features``, ``edge_index``, optionally ``edge_features``.
        """
        return self.forward(
            node_features=batch["node_features"],
            edge_index=batch["edge_index"],
            edge_features=batch.get("edge_features"),
        )
