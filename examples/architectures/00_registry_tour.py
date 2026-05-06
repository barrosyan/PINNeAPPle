"""Registry + Catalog tour (quick).

Goal:
- show *all* available families and models
- build a few representative models by name
- run a forward pass with dummy tensors

Run:
  python examples/pinneaple_models_showcase/00_registry_tour.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script: add repo root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import textwrap

import torch

from pinneaple_models.catalog import ModelCatalog
from pinneaple_models.register_all import register_all
from pinneaple_models.registry import ModelRegistry


def _short(s: str, n: int = 90) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> None:
    register_all()

    print("\n=== Families ===")
    print(ModelRegistry.families())

    print("\n=== Total registered models ===")
    print(len(ModelRegistry.list()))

    # show a few specs
    print("\n=== Sample specs ===")
    for name in [
        "fno",
        "vanilla_pinn",
        "pod",
        "gnn",
        "informer",
    ]:
        try:
            spec = ModelRegistry.spec(name)
            print(f"- {spec.name} | family={spec.family} | tags={spec.tags} | {_short(spec.description)}")
        except Exception as e:
            print(f"- {name}: (not found) {e}")

    cat = ModelCatalog()

    print("\n=== Catalog listing (by family) ===")
    print("autoencoders:", cat.autoencoders.list()[:10], "...")
    print("pinns:", cat.pinns.list()[:10], "...")
    print("neural_operators:", cat.neural_operators.list()[:10], "...")
    print("rom:", cat.rom.list()[:10], "...")
    print("graphnn:", cat.graphnn.list()[:10], "...")
    print("transformers:", cat.transformers.list()[:10], "...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== Build + forward (representative models) ===")

    # 1) Neural Operator (FNO-1D)
    fno = ModelRegistry.build("fno", in_channels=2, out_channels=1, width=32, modes=12, layers=3).to(device)
    u = torch.randn(4, 2, 128, device=device)
    y = fno(u).y
    print("FNO out:", tuple(y.shape))

    # 2) PINN (Vanilla)
    pinn = ModelRegistry.build("vanilla_pinn", in_dim=2, out_dim=1, hidden=[64, 64, 64], activation="tanh").to(device)
    x = torch.rand(256, 1, device=device)
    t = torch.rand(256, 1, device=device)
    y = pinn(x, t).y
    print("VanillaPINN out:", tuple(y.shape))

    # 3) ROM (POD)
    pod = ModelRegistry.build("pod", r=8, center=True).to(device)
    X = torch.randn(32, 64, device=device)
    pod.fit(X)
    out = pod(X, return_loss=True)
    print("POD recon:", tuple(out.y.shape), "| mse:", float(out.losses["mse"]))

    # 4) GraphNN (GNN)
    from pinneaple_models.graphnn.base import GraphBatch

    gnn = ModelRegistry.build("gnn", node_dim=3, out_dim=2, edge_dim=1, hidden=64, layers=3).to(device)

    # toy graph: ring
    B, N = 2, 16
    x = torch.randn(B, N, 3, device=device)
    src = torch.arange(N, device=device)
    dst = (src + 1) % N
    edge_index = torch.stack([src, dst], dim=0)  # (2,E)
    edge_attr = torch.randn(B, N, 1, device=device)  # E=N for ring
    mask = (torch.rand(B, N, device=device) > 0.3)

    gb = GraphBatch(x=x, edge_index=edge_index, edge_attr=edge_attr, mask=mask)
    y = gnn(gb).y
    print("GNN out:", tuple(y.shape))

    # 5) Time-series Transformer (VanillaTransformer) — fast forward demo
    ts = ModelRegistry.build(
        "transformer",
        in_dim=3,
        out_dim=2,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        pool="none",
    ).to(device)

    x_ts = torch.randn(4, 48, 3, device=device)
    y_ts = ts(x_ts)
    y_t = y_ts.y if hasattr(y_ts, "y") else y_ts
    print("Transformer out:", tuple(y_t.shape))

    print("\nDone.")


if __name__ == "__main__":
    main()
