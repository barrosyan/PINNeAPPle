"""Example: Poisson equation with FDM vs Spectral FFT.

Run:
  python examples/solvers/03_poisson_solvers.py
"""

import torch

from pinneaple_solvers import SolverRegistry, register_all


def main():
    register_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H = W = 64

    # f = sin(2pi x) sin(2pi y)
    xs = torch.linspace(0, 1, W, device=device)
    ys = torch.linspace(0, 1, H, device=device)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")
    f = torch.sin(2 * torch.pi * X) * torch.sin(2 * torch.pi * Y)

    # Dirichlet BC = 0
    bc = torch.zeros_like(f)
    fdm = SolverRegistry.build("fdm", iters=2000, omega=0.9).to(device)
    u_fdm = fdm(f, bc, dx=1 / (W - 1), dy=1 / (H - 1)).result
    print("u_fdm", u_fdm.shape)

    # periodic spectral
    spec = SolverRegistry.build("spectral").to(device)
    u_spec = spec(f).result
    print("u_spec", u_spec.shape)

    # compare rough
    err = torch.mean((u_fdm - u_spec) ** 2).item()
    print("MSE(fdm vs spectral):", err)


if __name__ == "__main__":
    main()
