"""Time-series transformer showcase: next-step forecasting.

We train a small `VanillaTransformer` to predict y_{t+1} from a window x_{t-T+1:t}.

This demonstrates:
- transformer family inside `pinneaple_models`
- a minimal training loop on CPU

Run:
  python examples/pinneaple_models_showcase/50_timeseries_transformer_forecast_toy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script: add repo root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import math

import torch

from pinneaple_models.transformers.transformer import VanillaTransformer


def make_batch(B: int, T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (x_seq, y_next)."""
    t = torch.linspace(0, 1, T + 1, device=device)[None, :, None]  # (1,T+1,1)
    freq = torch.rand(B, 1, 1, device=device) * 3.0 + 0.5
    phase = torch.rand(B, 1, 1, device=device) * 2 * math.pi
    amp = torch.rand(B, 1, 1, device=device) * 0.9 + 0.1

    y = amp * torch.sin(2 * math.pi * freq * t + phase)
    y = y + 0.05 * torch.randn_like(y)

    x_seq = y[:, :T, :]          # (B,T,1)
    y_next = y[:, T, :]          # (B,1)
    return x_seq, y_next


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    B, T = 64, 48

    model = VanillaTransformer(
        in_dim=1,
        out_dim=1,
        d_model=96,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        pool="last",
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    print("Training a tiny transformer forecaster...")
    for step in range(1, 401):
        x, y_true = make_batch(B, T, device)
        y_pred = model(x)  # (B,1)
        loss = torch.mean((y_pred - y_true) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            print(f"step {step:4d} | mse {float(loss.detach()):.6f}")

    # quick eval
    model.eval()
    with torch.no_grad():
        x, y_true = make_batch(256, T, device)
        y_pred = model(x)
        mse = torch.mean((y_pred - y_true) ** 2).item()

    print("Done. eval mse:", mse)


if __name__ == "__main__":
    main()
