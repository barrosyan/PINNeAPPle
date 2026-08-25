from __future__ import annotations
"""MeshGraphNet — Encoder-Process-Decoder GNN for mesh-based simulation.

Reference: Pfaff et al., ICLR 2021
  "Learning Mesh-Based Simulation with Graph Networks"
  https://arxiv.org/abs/2010.03409

Scope
-----
This implements the single-edge-type variant of MeshGraphNet, appropriate
for domains without external contact (e.g. CFD / continuum meshes where a
single fixed mesh connectivity fully determines interactions). The full
paper additionally defines a *world-space* edge set — dynamically rebuilt
each step from spatial proximity, with its own edge encoder/embedding — to
let the network learn contact effects (e.g. cloth self-collision) that are
not captured by mesh-space (rest-pose) connectivity alone. That second edge
type is not implemented here; ``edge_index``/``edge_attr`` are treated as a
single, fixed mesh-space edge set for the whole rollout.

Architecture
------------
1. Node encoder  : MLP(node_in_dim [+ pos_dim]) → hidden_dim, LayerNorm
2. Edge encoder  : MLP(edge_in_dim [+ pos_dim+1]) → hidden_dim, LayerNorm
                   (omitted when no edge features and use_pos=False)
3. Processor     : K rounds of _ProcessorBlock (edge-then-node update, residual)
4. Node decoder  : Linear(hidden_dim → out_dim)  — no activation, no LayerNorm

Batch convention
----------------
All tensors carry an explicit batch dimension:
  node features : (B, N, node_in_dim)
  edge features : (B, E, edge_in_dim)   — optional
  positions     : (B, N, pos_dim)        — optional, used when use_pos=True
  edge_index    : (2, E)                 — shared topology across the batch
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .base import GraphModelBase, GraphBatch, GraphOutput
from .utils import scatter_add


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_layers: int,
    dropout: float,
    layernorm: bool,
) -> nn.Sequential:
    """Stack of Linear→GELU blocks with optional Dropout and final LayerNorm."""
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

    Edge update : e'_ij = MLP_e([e_ij, h_i, h_j]) + e_ij
    Node update : h'_i  = MLP_v([h_i, Σ_j e'_ij])  + h_i
    """

    def __init__(self, hidden_dim: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        self.edge_mlp = _mlp(3 * hidden_dim, hidden_dim, hidden_dim,
                             n_layers, dropout, layernorm=True)
        self.node_mlp = _mlp(2 * hidden_dim, hidden_dim, hidden_dim,
                             n_layers, dropout, layernorm=True)

    def forward(
        self,
        h: torch.Tensor,    # (B, N, H)
        e: torch.Tensor,    # (B, E, H)
        src: torch.Tensor,  # (E,)  source node indices
        dst: torch.Tensor,  # (E,)  destination node indices
    ) -> tuple[torch.Tensor, torch.Tensor]:
        N = h.size(1)

        h_src = h[:, src, :]                          # (B, E, H)
        h_dst = h[:, dst, :]                          # (B, E, H)

        e_in  = torch.cat([e, h_src, h_dst], dim=-1)  # (B, E, 3H)
        e_new = self.edge_mlp(e_in) + e               # residual

        agg   = scatter_add(e_new, dst, N)            # (B, N, H)
        h_new = self.node_mlp(torch.cat([h, agg], dim=-1)) + h  # residual

        return h_new, e_new


# ---------------------------------------------------------------------------
# MeshGraphNet
# ---------------------------------------------------------------------------

class MeshGraphNet(GraphModelBase):
    """Encoder-Process-Decoder GNN for unstructured mesh simulation.

    Single-edge-type variant: one fixed ``edge_index`` carries mesh-space
    connectivity for the whole rollout. Contact-rich domains (e.g. cloth)
    that need a dynamically-rebuilt world-space edge set with its own
    encoder, per the full paper, are not covered by this class.

    All encoders are built eagerly at construction time — they are proper
    ``nn.Module`` children so they appear in ``parameters()``, ``state_dict()``,
    and are moved correctly by ``.to(device)``.

    Parameters
    ----------
    node_in_dim:
        Dimension of raw node features.
    out_dim:
        Output field dimension per node.
    edge_in_dim:
        Dimension of raw edge features.  ``0`` means no explicit edge features.
    pos_dim:
        Spatial dimension of node coordinates (e.g. 2 for 2-D, 3 for 3-D).
        Required (> 0) when ``use_pos=True``.
    hidden_dim:
        Internal embedding width used throughout all MLPs.
    n_layers:
        Number of hidden layers inside each MLP block.
    n_message_passing:
        Number of Processor rounds (graph depth).
    use_pos:
        When ``True``, node positions ``g.pos`` are concatenated to the node
        encoder input, and relative-displacement features are added to edges.
        ``pos_dim`` must be > 0 in this case.
    dropout:
        Dropout probability inside MLP hidden layers (0 = disabled).

    Examples
    --------
    >>> model = MeshGraphNet(node_in_dim=5, out_dim=2, edge_in_dim=3)
    >>> g = GraphBatch(
    ...     x=torch.rand(2, 200, 5),
    ...     edge_index=torch.randint(0, 200, (2, 800)),
    ...     edge_attr=torch.rand(2, 800, 3),
    ... )
    >>> out = model(g)
    >>> out.y.shape
    torch.Size([2, 200, 2])
    """

    def __init__(
        self,
        node_in_dim: int,
        out_dim: int,
        *,
        edge_in_dim: int = 0,
        pos_dim: int = 0,
        hidden_dim: int = 128,
        n_layers: int = 2,
        n_message_passing: int = 6,
        use_pos: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if use_pos and pos_dim <= 0:
            raise ValueError(
                "MeshGraphNet: pos_dim must be > 0 when use_pos=True. "
                "Pass pos_dim=<spatial_dim> to the constructor."
            )

        self.node_in_dim       = node_in_dim
        self.edge_in_dim       = edge_in_dim
        self.pos_dim           = pos_dim
        self.hidden_dim        = hidden_dim
        self.n_message_passing = n_message_passing
        self.use_pos           = use_pos

        # ── Node encoder ─────────────────────────────────────────────────
        # Input: raw node features [+ positions when use_pos]
        node_enc_in = node_in_dim + (pos_dim if use_pos else 0)
        self.node_encoder = _mlp(
            node_enc_in, hidden_dim, hidden_dim, n_layers, dropout, layernorm=True
        )

        # ── Edge encoder ─────────────────────────────────────────────────
        # Input: raw edge features [+ relative displacement + distance when use_pos]
        # pos_edge_dim = pos_dim (rel displacement) + 1 (distance)
        pos_edge_dim = (pos_dim + 1) if use_pos else 0
        edge_enc_in  = edge_in_dim + pos_edge_dim

        if edge_enc_in > 0:
            self.edge_encoder: Optional[nn.Module] = _mlp(
                edge_enc_in, hidden_dim, hidden_dim, n_layers, dropout, layernorm=True
            )
        else:
            # No edge information at all: edges initialised to zero at hidden_dim.
            # No encoder needed — zeros are valid initial embeddings.
            self.edge_encoder = None

        # ── Processor ────────────────────────────────────────────────────
        self.processor = nn.ModuleList([
            _ProcessorBlock(hidden_dim, n_layers, dropout)
            for _ in range(n_message_passing)
        ])

        # ── Decoder ──────────────────────────────────────────────────────
        # No activation / LayerNorm — raw linear projection to output field
        self.decoder = nn.Linear(hidden_dim, out_dim)

    # ------------------------------------------------------------------
    # Positional edge features
    # ------------------------------------------------------------------

    @staticmethod
    def _pos_to_edge_attr(
        pos: torch.Tensor,   # (B, N, P)
        src: torch.Tensor,   # (E,)
        dst: torch.Tensor,   # (E,)
    ) -> torch.Tensor:
        """Compute relative displacement and distance for each edge → (B, E, P+1)."""
        rel  = pos[:, dst, :] - pos[:, src, :]   # (B, E, P)
        dist = rel.norm(dim=-1, keepdim=True)      # (B, E, 1)
        return torch.cat([rel, dist], dim=-1)      # (B, E, P+1)

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

        Parameters
        ----------
        g:
            :class:`~pinneapple_models.graphnn.base.GraphBatch` with:

            * ``x``          — ``(B, N, node_in_dim)`` node features
            * ``edge_index`` — ``(2, E)`` directed edges (shared topology)
            * ``edge_attr``  — ``(B, E, edge_in_dim)`` or ``None``
            * ``pos``        — ``(B, N, pos_dim)`` coordinates (when ``use_pos=True``)
            * ``mask``       — ``(B, N)`` validity mask (1 = valid, 0 = padding)

        y_true:
            ``(B, N, out_dim)`` ground truth.  Required when ``return_loss=True``.
        return_loss:
            When ``True``, compute MSE loss against ``y_true``.

        Returns
        -------
        GraphOutput
            * ``y``      — ``(B, N, out_dim)`` per-node predictions
            * ``losses`` — ``{"total": scalar [, "mse": scalar]}``
            * ``extras`` — ``{"h": node_embeddings, "e": edge_embeddings}``
        """
        src, dst = g.edge_index[0], g.edge_index[1]  # (E,)
        B, N = g.x.size(0), g.x.size(1)

        # ── Node encoding ─────────────────────────────────────────────
        x_input = g.x                                      # (B, N, node_in_dim)
        if self.use_pos:
            if g.pos is None:
                raise ValueError("MeshGraphNet: use_pos=True but g.pos is None.")
            x_input = torch.cat([x_input, g.pos], dim=-1)  # (B, N, node_in_dim+pos_dim)

        h = self.node_encoder(x_input)                     # (B, N, hidden_dim)

        # ── Edge encoding ─────────────────────────────────────────────
        if self.edge_encoder is not None:
            e_parts: list[torch.Tensor] = []

            if self.edge_in_dim > 0:
                if g.edge_attr is None:
                    raise ValueError(
                        f"MeshGraphNet: edge_in_dim={self.edge_in_dim} but g.edge_attr is None."
                    )
                e_parts.append(g.edge_attr)                # (B, E, edge_in_dim)

            if self.use_pos:
                e_parts.append(self._pos_to_edge_attr(g.pos, src, dst))  # (B, E, pos_dim+1)

            e_raw = torch.cat(e_parts, dim=-1) if len(e_parts) > 1 else e_parts[0]
            e = self.edge_encoder(e_raw)                   # (B, E, hidden_dim)
        else:
            # Zero-initialised edge embeddings
            e = torch.zeros(B, g.edge_index.size(1), self.hidden_dim,
                            device=g.x.device, dtype=g.x.dtype)

        # ── Processor ────────────────────────────────────────────────
        for block in self.processor:
            h, e = block(h, e, src, dst)

        # ── Decoder ──────────────────────────────────────────────────
        y = self.decoder(h)                                # (B, N, out_dim)

        # ── Loss ─────────────────────────────────────────────────────
        losses: Dict[str, torch.Tensor] = {
            "total": torch.zeros((), device=y.device, dtype=y.dtype)
        }
        if return_loss and y_true is not None:
            if g.mask is not None:
                mask = g.mask[..., None].to(y.dtype)       # (B, N, 1)
                losses["mse"] = torch.mean(((y - y_true) ** 2) * mask)
            else:
                losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]

        return GraphOutput(y=y, losses=losses, extras={"h": h, "e": e})

    def forward_batch(self, batch: Dict[str, Any]) -> GraphOutput:
        """Dict-based interface used by the Arena / GNNAdapter.

        Accepted keys
        -------------
        ``x`` or ``node_features``
            ``(B, N, node_in_dim)`` node features.
        ``edge_index``
            ``(2, E)`` long tensor of directed edges.
        ``edge_attr`` or ``edge_features``
            ``(B, E, edge_in_dim)`` edge features (optional).
        ``pos``
            ``(B, N, pos_dim)`` node coordinates (required when ``use_pos=True``).
        ``mask``
            ``(B, N)`` validity mask (optional).
        ``y_true`` or ``y``
            Ground-truth field for loss computation (optional).
        """
        x          = batch["x"] if "x" in batch else batch.get("node_features")
        edge_index = batch["edge_index"]
        edge_attr  = batch.get("edge_attr") if "edge_attr" in batch else batch.get("edge_features")
        pos        = batch.get("pos")
        mask       = batch.get("mask")
        y_true     = batch.get("y_true") if "y_true" in batch else batch.get("y")

        if x is None:
            raise KeyError("forward_batch: batch must contain 'x' or 'node_features'.")

        g = GraphBatch(x=x, edge_index=edge_index, pos=pos, edge_attr=edge_attr, mask=mask)
        return self.forward(g, y_true=y_true, return_loss=(y_true is not None))
