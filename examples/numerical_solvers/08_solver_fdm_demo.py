"""
Demo: FDM (heat1d / wave1d / heat2d)
"""

import torch

from pinneaple_solvers.registry import SolverCatalog


def main():
    device = torch.device("cpu")
    dtype = torch.float32
    cat = SolverCatalog()

    # --- heat1d
    N = 256
    x = torch.linspace(0.0, 1.0, N, device=device, dtype=dtype)
    u0 = torch.exp(-200.0 * (x - 0.3) ** 2)
    fdm = cat.build("fdm", equation="heat1d", alpha=0.01, dx=float(x[1] - x[0]), bc="dirichlet", record_history=True)
    out = fdm(u0, dt=1e-3, steps=500)
    print("\nFDM heat1d")
    print("  uT:", out.result.shape, "hist:", out.extras["u_hist"].shape)

    # --- wave1d
    u0 = torch.exp(-200.0 * (x - 0.5) ** 2)
    fdm = cat.build("fdm", equation="wave1d", c=1.0, dx=float(x[1] - x[0]), bc="dirichlet", record_history=True)
    out = fdm(u0, dt=5e-4, steps=800)
    print("\nFDM wave1d")
    print("  uT:", out.result.shape, "hist:", out.extras["u_hist"].shape)

    # --- heat2d
    H, W = 128, 128
    Y, X = torch.meshgrid(
        torch.linspace(0.0, 1.0, H, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, W, device=device, dtype=dtype),
        indexing="ij",
    )
    u0 = torch.exp(-200.0 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2))
    fdm = cat.build("fdm", equation="heat2d", alpha=0.01, dx=1.0 / (W - 1), dy=1.0 / (H - 1), bc="dirichlet", record_history=False)
    out = fdm(u0, dt=1e-3, steps=400)
    print("\nFDM heat2d")
    print("  uT:", out.result.shape)


if __name__ == "__main__":
    main()