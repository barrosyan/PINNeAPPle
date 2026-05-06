"""1D Wave equation with the *compiler* API.

Showcases:
  - time-dependent PDE with second-order time derivative (u_tt)
  - initial condition for displacement AND velocity
  - Dirichlet boundary conditions
  - the built-in "wave_equation" PDE kind

Problem (string on a line):
  u_tt - c^2 u_xx = 0,   (t, x) in [0, 1] x [0, 1]

  IC:   u(0, x) = sin(pi x)          (displacement)
        u_t(0, x) = 0                 (zero initial velocity)

  BC:   u(t, 0) = u(t, 1) = 0        (fixed ends)

Analytic solution:
  u(t, x) = sin(pi x) cos(pi c t)

With c = 1 this gives a standing wave with frequency pi.

Run:
  python examples/pinneaple_pinn/06_compiler_wave_1d.py
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pinneaple_environment.conditions import DirichletBC, InitialCondition
from pinneaple_environment.spec import PDETermSpec, ProblemSpec
from pinneaple_pinn import LossWeights, compile_problem


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class MLP(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 5):
        super().__init__()
        layers = [torch.nn.Linear(in_dim, width), torch.nn.Tanh()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
        layers += [torch.nn.Linear(width, out_dim)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Exact solution & initial / boundary conditions
# ---------------------------------------------------------------------------

C_WAVE = 1.0  # wave speed


def u_exact_np(X: np.ndarray) -> np.ndarray:
    """Analytic solution u(t,x) = sin(pi x) cos(pi c t)."""
    t, x = X[:, 0], X[:, 1]
    return (np.sin(math.pi * x) * np.cos(math.pi * C_WAVE * t))[:, None].astype(
        np.float32
    )


def u0_np(X: np.ndarray, ctx) -> np.ndarray:
    """Initial displacement: u(0, x) = sin(pi x)."""
    x = X[:, 1]
    return (np.sin(math.pi * x))[:, None].astype(np.float32)


def bc_np(X: np.ndarray, ctx) -> np.ndarray:
    """Homogeneous Dirichlet BC: u(t, 0) = u(t, 1) = 0."""
    return np.zeros((X.shape[0], 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_collocation(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = rng.random((n, 1), dtype=np.float32)
    x = rng.random((n, 1), dtype=np.float32)
    return torch.from_numpy(np.concatenate([t, x], axis=1))


def sample_initial(n: int, seed: int = 1) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = np.zeros((n, 1), dtype=np.float32)
    x = rng.random((n, 1), dtype=np.float32)
    return torch.from_numpy(np.concatenate([t, x], axis=1))


def sample_boundary(n: int, seed: int = 2) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    t = rng.random((n, 1), dtype=np.float32)
    side = rng.integers(0, 2, size=(n, 1))
    x = np.where(side == 0, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(np.concatenate([t, x], axis=1))


# ---------------------------------------------------------------------------
# Initial-velocity loss (u_t(0,x) = 0) — added as a custom penalty
# ---------------------------------------------------------------------------


def initial_velocity_loss(
    model: torch.nn.Module, x_ic: torch.Tensor, t_index: int = 0
) -> torch.Tensor:
    """MSE of u_t(0, x) which should be zero for a standing wave."""
    x_ic = x_ic.detach().requires_grad_(True)
    u = model(x_ic)
    (u_t,) = torch.autograd.grad(
        u, x_ic, torch.ones_like(u), create_graph=True
    )
    # u_t is the full Jacobian row; pick the t-component
    return torch.mean(u_t[:, t_index:t_index + 1] ** 2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # --- Problem spec ---
    spec = ProblemSpec(
        name="wave_1d",
        dim=2,
        coords=("t", "x"),
        fields=("u",),
        pde=PDETermSpec(
            kind="wave_equation",
            fields=("u",),
            coords=("t", "x"),
            params={"c": C_WAVE},
        ),
        conditions=(
            InitialCondition(name_or_values="u0", fields=("u",), value_fn=u0_np, weight=1.0),
            DirichletBC(name_or_values="bc", fields=("u",), value_fn=bc_np, weight=1.0),
        ),
    )

    loss_fn = compile_problem(
        spec,
        weights=LossWeights(w_pde=1.0, w_bc=10.0, w_ic=10.0, w_data=1.0),
    )

    model = MLP(in_dim=2, out_dim=1, width=64, depth=5).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10000)

    # --- Sampling ---
    x_col = sample_collocation(10000, seed=0).to(device=device, dtype=dtype)
    x_ic = sample_initial(1024, seed=1).to(device=device, dtype=dtype)
    x_bc = sample_boundary(1024, seed=2).to(device=device, dtype=dtype)

    batch = {
        "x_col": x_col,
        "x_ic": x_ic,
        "x_bc": x_bc,
        "ctx": {},
    }

    # --- Train ---
    w_vel = 10.0  # weight for u_t(0,x) = 0
    steps = 10000
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        losses = loss_fn(model, None, batch)

        # add initial-velocity penalty (standing wave: u_t(0,x) = 0)
        vel_loss = initial_velocity_loss(model, x_ic, t_index=0)
        total = losses["total"] + w_vel * vel_loss
        total.backward()
        opt.step()
        sched.step()

        if step % 1000 == 0 or step == 1:
            ic = float(losses.get("ic_u0", torch.tensor(0.0)).detach())
            bc = float(losses.get("bc_bc", torch.tensor(0.0)).detach())
            print(
                f"step={step:05d}  total={float(total.detach()):.4e}  "
                f"pde={float(losses['pde'].detach()):.4e}  "
                f"ic={ic:.4e}  bc={bc:.4e}  "
                f"vel={float(vel_loss.detach()):.4e}"
            )

    # --- Evaluate ---
    g = 64
    ts = torch.linspace(0, 1, g, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, g, device=device, dtype=dtype)
    tt, xx = torch.meshgrid(ts, xs, indexing="ij")
    Xg = torch.stack([tt.reshape(-1), xx.reshape(-1)], dim=1)

    with torch.no_grad():
        pred = model(Xg).cpu().numpy()
    true = u_exact_np(Xg.cpu().numpy())
    rel_l2 = float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))
    print(f"\nRelative L2 error on {g}x{g} grid: {rel_l2:.3e}")


if __name__ == "__main__":
    main()
