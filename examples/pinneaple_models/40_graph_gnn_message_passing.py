"""GraphNN showcase: message passing GNN with masking.

We build a simple ring graph and train the GNN to predict a *local* operator:
  y_i = x_{i-1} + x_{i+1}  (sum of neighbor features)

This demonstrates:
- `GraphBatch` structure
- masked MSE loss inside the model (`return_loss=True`)

Run:
  python examples/pinneaple_models_showcase/40_graph_gnn_message_passing.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script: add repo root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from pinneaple_models.graphnn.base import GraphBatch
from pinneaple_models.graphnn.gnn import GraphNeuralNetwork


def build_ring_edge_index(N: int, device: torch.device) -> torch.Tensor:
    src = torch.arange(N, device=device)
    dst = (src + 1) % N
    # bidirectional edges
    edge_index = torch.cat([
        torch.stack([src, dst], dim=0),
        torch.stack([dst, src], dim=0),
    ], dim=1)
    return edge_index  # (2, 2N)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    B, N, node_dim = 8, 32, 4
    out_dim = node_dim

    edge_index = build_ring_edge_index(N, device)
    E = edge_index.shape[1]

    # edge_attr: just distance (1.0) here
    edge_attr = torch.ones(B, E, 1, device=device)

    model = GraphNeuralNetwork(
        node_dim=node_dim,
        out_dim=out_dim,
        edge_dim=1,
        hidden=128,
        layers=4,
        dropout=0.0,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    def make_batch() -> tuple[GraphBatch, torch.Tensor]:
        x = torch.randn(B, N, node_dim, device=device)
        # neighbor sum target (ring)
        x_left = torch.roll(x, shifts=1, dims=1)
        x_right = torch.roll(x, shifts=-1, dims=1)
        y_true = x_left + x_right
        mask = (torch.rand(B, N, device=device) > 0.2)
        g = GraphBatch(x=x, edge_index=edge_index, edge_attr=edge_attr, mask=mask)
        return g, y_true

    print("Training a message passing GNN on a toy neighbor-sum task...")
    for step in range(1, 401):
        g, y_true = make_batch()
        out = model(g, y_true=y_true, return_loss=True)
        loss = out.losses["total"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step {step:4d} | masked mse {float(out.losses['mse']):.6f}")

    # eval
    model.eval()
    with torch.no_grad():
        g, y_true = make_batch()
        y_pred = model(g).y
        mse = torch.mean((y_pred - y_true) ** 2).item()

    print("Done. unmasked eval mse:", mse)


if __name__ == "__main__":
    main()
