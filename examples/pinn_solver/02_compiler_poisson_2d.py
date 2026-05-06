"""Poisson equation (2D) with the *compiler* API.

This example showcases:
  - pinneaple_environment.ProblemSpec + ConditionSpec
  - pinneaple_pinn.compile_problem() (autograd-based PDE residuals)
  - a minimal training loop with a vanilla MLP

Problem (unit square):
  u_xx + u_yy = f(x,y),  (x,y) in (0,1)^2
  u = 0 on boundary

Analytic solution we target:
  u(x,y) = sin(pi x) sin(pi y)
  => Laplacian(u) = -2*pi^2 * u
  => f(x,y) = -2*pi^2 * sin(pi x) sin(pi y)

Run:
  python examples/pinneaple_pinn/02_compiler_poisson_2d.py
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pinneaple_environment.conditions import DirichletBC
from pinneaple_environment.spec import PDETermSpec, ProblemSpec
from pinneaple_pinn import LossWeights, compile_problem


class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 4):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, out_dim)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def u_true_np(X: np.ndarray) -> np.ndarray:
    x = X[:, 0]
    y = X[:, 1]
    return (np.sin(math.pi * x) * np.sin(math.pi * y))[:, None].astype(np.float32)


def f_source_np(X: np.ndarray, ctx) -> np.ndarray:
    # f = Laplacian(u_true) = -2*pi^2*u_true
    return (-2.0 * (math.pi**2) * u_true_np(X)).astype(np.float32)


def sample_interior(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    X = rng.random((n, 2), dtype=np.float32)
    return torch.from_numpy(X)


def sample_boundary(n: int, seed: int = 1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    # sample points on the 4 edges uniformly
    t = rng.random((n,), dtype=np.float32)
    edge = rng.integers(0, 4, size=(n,))
    X = np.zeros((n, 2), dtype=np.float32)
    # x=0, x=1, y=0, y=1
    X[edge == 0] = np.stack([np.zeros_like(t[edge == 0]), t[edge == 0]], axis=1)
    X[edge == 1] = np.stack([np.ones_like(t[edge == 1]), t[edge == 1]], axis=1)
    X[edge == 2] = np.stack([t[edge == 2], np.zeros_like(t[edge == 2])], axis=1)
    X[edge == 3] = np.stack([t[edge == 3], np.ones_like(t[edge == 3])], axis=1)
    return torch.from_numpy(X)


def main() -> None:
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # --- Problem spec (compiler API) ---
    spec = ProblemSpec(
        name="poisson_2d_unit_square",
        dim=2,
        coords=("x", "y"),
        fields=("u",),
        pde=PDETermSpec(kind="poisson", fields=("u",), coords=("x", "y"), params={}),
        conditions=(
            DirichletBC(
                name="u_zero",
                fields=("u",),
                # values default to zeros if value_fn is None
            ),
        ),
    )

    loss_fn = compile_problem(
        spec,
        weights=LossWeights(w_pde=1.0, w_bc=50.0, w_ic=1.0, w_data=1.0),
    )

    model = MLP(in_dim=2, out_dim=1, width=64, depth=4).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    # training points (fixed for a deterministic demo)
    x_col = sample_interior(4096, seed=0).to(device=device, dtype=dtype)
    x_bc = sample_boundary(1024, seed=1).to(device=device, dtype=dtype)
    y_bc = torch.zeros((x_bc.shape[0], 1), device=device, dtype=dtype)

    batch = {
        "x_col": x_col,
        "x_bc": x_bc,
        "y_bc": y_bc,
        "ctx": {"source_fn": f_source_np},
    }

    # --- Train ---
    steps = 1500
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        losses = loss_fn(model, None, batch)  # y_hat is ignored (kept for compatibility)
        total = losses["total"]
        total.backward()
        opt.step()

        if step % 200 == 0 or step == 1:
            print(
                f"step={step:04d} total={float(total):.4e} pde={float(losses['pde']):.4e} bc={float(losses['bc_u_zero']):.4e}"
            )

    # --- Evaluate on a grid ---
    g = 64
    xs = torch.linspace(0, 1, g, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    Xg = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
    with torch.no_grad():
        pred = model(Xg).cpu().numpy()
    true = u_true_np(Xg.cpu().numpy())
    rel_l2 = float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))
    print(f"rel_L2(grid) = {rel_l2:.3e}")


if __name__ == "__main__":
    main()
