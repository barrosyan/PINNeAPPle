"""1D Burgers equation with the *compiler* API.

Showcases:
  - time-dependent PDE residuals (coords include "t")
  - initial condition + boundary condition
  - minimal PINN training loop

We use the classic viscous Burgers equation on (t,x) in [0,1]x[-1,1]:
  u_t + u u_x - nu u_xx = 0

IC/BC used for a stable demo:
  u(0,x) = -sin(pi x)
  u(t,-1) = u(t,1) = 0

This is a standard PINN benchmark (exact solution exists but is a bit long);
here we focus on showing how to wire the losses and train.

Run:
  python examples/pinneaple_pinn/03_compiler_burgers_1d.py
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pinneaple_environment.conditions import DirichletBC, InitialCondition
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


def u0_np(X: np.ndarray, ctx) -> np.ndarray:
    # X: (N,2) -> [t,x]
    x = X[:, 1]
    return (-np.sin(math.pi * x))[:, None].astype(np.float32)


def bc_np(X: np.ndarray, ctx) -> np.ndarray:
    # homogeneous boundary
    return np.zeros((X.shape[0], 1), dtype=np.float32)


def sample_collocation(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = rng.random((n, 1), dtype=np.float32)  # [0,1]
    x = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    return torch.from_numpy(np.concatenate([t, x], axis=1))


def sample_initial(n: int, seed: int = 1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = np.zeros((n, 1), dtype=np.float32)
    x = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    return torch.from_numpy(np.concatenate([t, x], axis=1))


def sample_boundary(n: int, seed: int = 2) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = rng.random((n, 1), dtype=np.float32)
    side = rng.integers(0, 2, size=(n, 1))
    x = np.where(side == 0, -1.0, 1.0).astype(np.float32)
    return torch.from_numpy(np.concatenate([t, x], axis=1))


def main() -> None:
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    nu = 0.01 / math.pi

    spec = ProblemSpec(
        name="burgers_1d",
        dim=2,
        coords=("t", "x"),
        fields=("u",),
        pde=PDETermSpec(kind="burgers", fields=("u",), coords=("t", "x"), params={"nu": nu}),
        conditions=(
            InitialCondition(name="u0", fields=("u",), value_fn=u0_np, weight=1.0),
            DirichletBC(name="bc", fields=("u",), value_fn=bc_np, weight=1.0),
        ),
    )

    loss_fn = compile_problem(
        spec,
        weights=LossWeights(w_pde=1.0, w_bc=10.0, w_ic=10.0, w_data=1.0),
    )

    model = MLP(in_dim=2, out_dim=1, width=64, depth=4).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    x_col = sample_collocation(8192, seed=0).to(device=device, dtype=dtype)
    x_ic = sample_initial(1024, seed=1).to(device=device, dtype=dtype)
    x_bc = sample_boundary(1024, seed=2).to(device=device, dtype=dtype)

    batch = {
        "x_col": x_col,
        "x_ic": x_ic,
        "x_bc": x_bc,
        # leave y_ic/y_bc as None to trigger value_fn evaluation inside compile_problem
        "ctx": {},
    }

    steps = 2500
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        losses = loss_fn(model, None, batch)
        total = losses["total"]
        total.backward()
        opt.step()

        if step % 250 == 0 or step == 1:
            ic = float(losses.get("ic_u0", torch.tensor(0.0)))
            bc = float(losses.get("bc_bc", torch.tensor(0.0)))
            print(f"step={step:04d} total={float(total):.4e} pde={float(losses['pde']):.4e} ic={ic:.4e} bc={bc:.4e}")

    # quick sanity check: print IC fit MSE
    with torch.no_grad():
        pred0 = model(x_ic).cpu().numpy()
    true0 = u0_np(x_ic.cpu().numpy(), {})
    mse0 = float(np.mean((pred0 - true0) ** 2))
    print(f"IC MSE = {mse0:.3e}")


if __name__ == "__main__":
    main()
