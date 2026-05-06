"""ROM showcase: POD compression + DMD dynamics.

We create a synthetic linear dynamical system in a high-dimensional space, then:
1) Fit POD to learn a low-rank basis
2) Fit DMD to learn linear dynamics in the reduced space
3) Roll out future states and measure reconstruction error

Run:
  python examples/pinneaple_models_showcase/30_rom_pod_dmd.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script: add repo root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from pinneaple_models.rom.pod import POD
from pinneaple_models.rom.dmd import DynamicModeDecomposition


def make_linear_system(D: int, seed: int = 0) -> torch.Tensor:
    """Stable-ish linear operator A (D,D)."""
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(D, D, generator=g) / (D**0.5)
    # make it contractive
    u, s, vh = torch.linalg.svd(M)
    s = 0.95 * (s / s.max())
    A = (u * s) @ vh
    return A


def generate_sequences(B: int, T: int, D: int, A: torch.Tensor, noise: float = 0.01) -> torch.Tensor:
    g = torch.Generator().manual_seed(123)
    x = torch.randn(B, D, generator=g)
    xs = [x]
    for _ in range(T - 1):
        x = x @ A.t() + noise * torch.randn_like(x, generator=g)
        xs.append(x)
    return torch.stack(xs, dim=1)  # (B,T,D)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    B, T, D = 16, 60, 128
    r = 12

    A = make_linear_system(D).to(device)
    X = generate_sequences(B, T, D, A).to(device)

    # 1) POD
    pod = POD(r=r, center=True).to(device)
    pod.fit(X)
    out_pod = pod(X, return_loss=True)
    print("POD fitted. recon mse:", float(out_pod.losses["mse"]))

    # 2) DMD (fit on POD coefficients)
    with torch.no_grad():
        a = pod.encode(X)  # (B*T,r) flattened
        a_seq = a.reshape(B, T, r)

    dmd = DynamicModeDecomposition(r=r, center=True).to(device)
    dmd.fit(a_seq)

    # rollout in reduced space, then decode back
    with torch.no_grad():
        a_hat = dmd(a_seq, return_loss=True).y  # (B,T,r)
        X_hat = pod.decode(a_hat.reshape(B * T, r), shape=(B, T, D))

        mse = torch.mean((X_hat - X) ** 2).item()

    print("POD+DMD rollout mse (state space):", mse)


if __name__ == "__main__":
    main()
