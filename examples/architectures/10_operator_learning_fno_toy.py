"""Toy operator-learning with FNO-1D.

We learn an operator G:
  y(x) = u(x)^2 + 0.3 * d/dx u(x)

This is *not* a PDE dataset; it's a quick, self-contained demo that shows:
- how to instantiate `FourierNeuralOperator` (FNO)
- how to train using the model's built-in `return_loss=True`

Run:
  python examples/pinneaple_models_showcase/10_operator_learning_fno_toy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script: add repo root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import math

import torch

from pinneaple_models.neural_operators.fno import FourierNeuralOperator


def make_batch(B: int, L: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns u, y_true.

    u: (B,1,L)
    y: (B,1,L)
    """
    x = torch.linspace(0, 1, L, device=device)[None, None, :]  # (1,1,L)

    # random Fourier series
    K = 6
    amps = torch.randn(B, 1, K, device=device) * 0.6
    phases = torch.rand(B, 1, K, device=device) * 2 * math.pi
    freqs = torch.arange(1, K + 1, device=device)[None, None, :]

    # broadcast to (B,1,K,L)
    x4 = x[..., None, :]                 # (1,1,1,L)
    f4 = freqs[..., :, None]             # (1,1,K,1)
    s = torch.sin(2 * math.pi * f4 * x4 + phases[..., :, None])
    u = torch.sum(amps[..., :, None] * s, dim=2)  # (B,1,L)

    # finite-difference derivative
    du = torch.zeros_like(u)
    du[..., 1:-1] = 0.5 * (u[..., 2:] - u[..., :-2]) * (L - 1)
    du[..., 0] = (u[..., 1] - u[..., 0]) * (L - 1)
    du[..., -1] = (u[..., -1] - u[..., -2]) * (L - 1)

    y = u**2 + 0.3 * du
    return u, y


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    B, L = 32, 256

    model = FourierNeuralOperator(
        in_channels=1,
        out_channels=1,
        width=48,
        modes=24,
        layers=4,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    print("Training FNO on a toy operator...")
    for step in range(1, 301):
        u, y_true = make_batch(B, L, device)
        out = model(u, y_true=y_true, return_loss=True)
        loss = out.losses["total"]

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 50 == 0:
            print(f"step {step:4d} | mse {float(out.losses['mse']):.6f}")

    # quick evaluation
    model.eval()
    with torch.no_grad():
        u, y_true = make_batch(4, L, device)
        y_pred = model(u).y
        mse = torch.mean((y_pred - y_true) ** 2).item()

    print("Done. eval mse:", mse)


if __name__ == "__main__":
    main()
