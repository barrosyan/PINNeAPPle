"""
Demo: Spectral / pseudo-spectral (burgers1d, ks1d)
"""

import torch

from pinneaple_solvers.registry import SolverCatalog


def main():
    device = torch.device("cpu")
    dtype = torch.float32
    cat = SolverCatalog()

    N = 512
    L = 2.0 * torch.pi
    x = torch.linspace(0.0, L, N, device=device, dtype=dtype, endpoint=False)

    # burgers initial condition
    u0 = torch.sin(x)
    spec = cat.build("spectral", equation="burgers1d", L=float(L), nu=0.02, record_history=True, dealias=True)
    out = spec(u0, dt=5e-4, steps=2000)
    print("\nSpectral burgers1d")
    print("  uT:", out.result.shape, "hist:", out.extras["u_hist"].shape)

    # ks initial condition (small random perturbation)
    u0 = 0.1 * torch.randn(N, device=device, dtype=dtype)
    spec = cat.build("spectral", equation="ks1d", L=float(L), nu=0.0, record_history=False, dealias=True)
    out = spec(u0, dt=1e-3, steps=5000)
    print("\nSpectral ks1d")
    print("  uT:", out.result.shape)


if __name__ == "__main__":
    main()