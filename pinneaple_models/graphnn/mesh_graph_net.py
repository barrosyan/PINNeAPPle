from __future__ import annotations
"""MeshGraphNet — Encoder-Process-Decoder GNN for mesh-based simulation.

Reference: Pfaff et al., ICLR 2021
  "Learning Mesh-Based Simulation with Graph Networks"
  https://arxiv.org/abs/2010.03409

Architecture:
  1. Node encoder  : MLP(node_in_dim [+ pos_dim]) → hidden_dim, LayerNorm
  2. Edge encoder  : MLP(edge_in_dim)             → hidden_dim, LayerNorm
  3. Processor     : K rounds of EdgeConvMGN (edge-then-node update, residual)
  4. Node decoder  : MLP(hidden_dim → out_dim), no activation / LN
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import GraphModelBase, GraphBatch, GraphOutput
from .utils import scatter_add


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int,
         dropout: float, layernorm: bool) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    if layernorm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)


class _ProcessorBlock(nn.Module):
    """One round of edge-then-node message passing with residual connections.

    Edge update  : e'_ij = MLP_e([e_ij, v_i, v_j]) + e_ij
    Node update  : v'_i  = MLP_v([v_i,  Σ_j e'_ij]) + v_i
    """

    def __init__(self, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.edge_mlp = _mlp(3 * hidden_dim, hidden_dim, hidden_dim,
                             n_layers, dropout, layernorm=True)
        self.node_mlp = _mlp(2 * hidden_dim, hidden_dim, hidden_dim,
                             n_layers, dropout, layernorm=True)

    def forward(
        self,
        h: torch.Tensor,          # (B, N, H)
        e: torch.Tensor,          # (B, E, H)
        src: torch.Tensor,        # (E,)
        dst: torch.Tensor,        # (E,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = h.size(1)

        h_src = h[:, src, :]                              # (B, E, H)
        h_dst = h[:, dst, :]                              # (B, E, H)

        e_in  = torch.cat([e, h_src, h_dst], dim=-1)     # (B, E, 3H)
        e_new = self.edge_mlp(e_in) + e                  # residual

        agg   = scatter_add(e_new, dst, dim_size=N)      # (B, N, H)
        v_in  = torch.cat([h, agg], dim=-1)              # (B, N, 2H)
        h_new = self.node_mlp(v_in) + h                  # residual

        return h_new, e_new


# ---------------------------------------------------------------------------
# MeshGraphNet
# ---------------------------------------------------------------------------

class MeshGraphNet(GraphModelBase):
    """Encoder-Process-Decoder GNN for unstructured mesh simulation.

    Args:
        node_in_dim: Raw node-feature dimension (e.g. velocity, pressure, type).
        out_dim: Output field dimension per node.
        edge_in_dim: Raw edge-feature dimension. When 0 and ``use_pos=True``,
            relative-position features are computed automatically from ``g.pos``.
        hidden_dim: Internal embedding width for all MLPs.
        n_layers: MLP depth (number of hidden layers) inside each block.
        n_message_passing: Number of Processor rounds.
        use_pos: When True, concatenate node positions to the node encoder input.
            Automatically detects spatial dimension from ``g.pos`` at runtime.
        dropout: Dropout rate applied inside MLPs.
    """

    def __init__(
        self,
        node_in_dim: int,
        out_dim: int,
        *,
        edge_in_dim: int = 0,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_message_passing: int = 6,
        use_pos: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_in_dim      = node_in_dim
        self.edge_in_dim      = edge_in_dim
        self.hidden_dim       = hidden_dim
        self.n_message_passing = n_message_passing
        self.use_pos          = use_pos

        # Node encoder — pos_dim resolved lazily on first forward
        self._node_encoder: Optional[nn.Module] = None
        self._node_in_total: Optional[int] = None
        self._n_layers_enc  = n_layers
        self._dropout_enc   = dropout

        # Edge encoder
        edge_enc_in = edge_in_dim if edge_in_dim > 0 else 0
        self._edge_enc_in   = edge_enc_in
        # Built lazily too, since auto-pos edge dim depends on pos at runtime
        self._edge_encoder: Optional[nn.Module] = None
        self._edge_in_total: Optional[int] = None

        # Processor
        self.processor = nn.ModuleList([
            _ProcessorBlock(hidden_dim, n_layers, dropout)
            for _ in range(n_message_passing)
        ])

        # Decoder (no LayerNorm, no activation — pure linear projection)
        self.decoder = nn.Linear(hidden_dim, out_dim)

    # ------------------------------------------------------------------
    # Lazy encoder initialisation (resolves pos_dim at runtime)
    # ------------------------------------------------------------------

    def _build_node_encoder(self, node_in_total: int) -> None:
        if self._node_in_total == node_in_total:
            return
        self._node_in_total = node_in_total
        self._node_encoder = _mlp(
            node_in_total, self.hidden_dim, self.hidden_dim,
            self._n_layers_enc, self._dropout_enc, layernorm=True,
        ).to(next(self.processor.parameters()).device)

    def _build_edge_encoder(self, edge_in_total: int) -> None:
        if self._edge_in_total == edge_in_total:
            return
        self._edge_in_total = edge_in_total
        self._edge_encoder = _mlp(
            edge_in_total, self.hidden_dim, self.hidden_dim,
            self._n_layers_enc, self._dropout_enc, layernorm=True,
        ).to(next(self.processor.parameters()).device)

    # ------------------------------------------------------------------
    # Auto edge features from node positions
    # ------------------------------------------------------------------

    @staticmethod
    def _pos_to_edge_attr(
        pos: torch.Tensor,   # (B, N, P)
        src: torch.Tensor,   # (E,)
        dst: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        """Relative displacement + distance → (B, E, P+1)."""
        rel = pos[:, dst, :] - pos[:, src, :]          # (B, E, P)
        dist = rel.norm(dim=-1, keepdim=True)           # (B, E, 1)
        return torch.cat([rel, dist], dim=-1)           # (B, E, P+1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        g: GraphBatch,
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> GraphOutput:
        """Forward pass over a batched graph.

        Args:
            g: :class:`~pinneaple_models.graphnn.base.GraphBatch` with:
                * ``x``         — ``(B, N, node_in_dim)`` node features
                * ``edge_index``— ``(2, E)`` directed edges (shared topology)
                * ``edge_attr`` — ``(B, E, edge_in_dim)`` or None
                * ``pos``       — ``(B, N, pos_dim)`` optional coordinates
                * ``mask``      — ``(B, N)`` optional validity mask
            y_true: ``(B, N, out_dim)`` ground truth for supervised loss.
            return_loss: Compute MSE loss against *y_true* when True.

        Returns:
            :class:`~pinneaple_models.graphnn.base.GraphOutput` with:
                * ``y``      — ``(B, N, out_dim)`` per-node predictions
                * ``losses`` — ``{"total": ..., "mse": ...}``
                * ``extras`` — ``{"h": final_node_embeddings, "e": final_edge_embeddings}``
        """
        src, dst = g.edge_index[0], g.edge_index[1]  # (E,)
        N = g.x.size(1)

        # ── Node features ──────────────────────────────────────────────
        x_input = g.x                                      # (B, N, node_in_dim)
        if self.use_pos and g.pos is not None:
            x_input = torch.cat([x_input, g.pos], dim=-1)  # (B, N, node_in_dim + pos_dim)

        self._build_node_encoder(x_input.size(-1))
        h = self._node_encoder(x_input)                    # (B, N, H)

        # ── Edge features ───────────────────────────────────────────────
        if g.edge_attr is not None:
            e_raw = g.edge_attr                            # (B, E, edge_in_dim)
            if self.use_pos and g.pos is not None:
                e_pos = self._pos_to_edge_attr(g.pos, src, dst)
                e_raw = torch.cat([e_raw, e_pos], dim=-1)
        elif self.use_pos and g.pos is not None:
            e_raw = self._pos_to_edge_attr(g.pos, src, dst)  # (B, E, pos_dim+1)
        else:
            # No edge features and no positions → zero edge init
            e_raw = torch.zeros(
                g.x.size(0), g.edge_index.size(1), self.hidden_dim,
                device=g.x.device, dtype=g.x.dtype,
            )

        if e_raw.size(-1) == self.hidden_dim and g.edge_attr is None and not self.use_pos:
            # Already at hidden_dim (zero init path) — skip encoder
            e = e_raw
        else:
            self._build_edge_encoder(e_raw.size(-1))
            e = self._edge_encoder(e_raw)                  # (B, E, H)

        # ── Processor ──────────────────────────────────────────────────
        for block in self.processor:
            h, e = block(h, e, src, dst)

        # ── Decoder ────────────────────────────────────────────────────
        y = self.decoder(h)                                # (B, N, out_dim)

        # ── Loss ───────────────────────────────────────────────────────
        losses: Dict[str, torch.Tensor] = {
            "total": torch.tensor(0.0, device=y.device)
        }
        if return_loss and y_true is not None:
            if g.mask is not None:
                mask = g.mask[..., None].to(y.dtype)
                losses["mse"] = torch.mean(((y - y_true) ** 2) * mask)
            else:
                losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return GraphOutput(y=y, losses=losses, extras={"h": h, "e": e})
