"""06_custom_problem.py — How to plug a custom physics problem into the Arena.

This example shows two custom problems:

  1. ReactionDiffusion1D  (1-D, input_dim=1)
     eps*u'' - k*u = f(x)  on [0,1],  u(0)=u(1)=0
     Manufactured solution: u = sin(πx)

  2. ModifiedPoisson2D    (2-D, input_dim=2)
     Δu + k*u = f(x,y)  on [0,1]²,  u=0 on boundary
     Manufactured solution: u = sin(πx)*sin(πy)

Steps to create a custom problem
---------------------------------
  1. Subclass ``ArenaProblem`` and set name/description/domain/input_dim/output_dim.
  2. Implement ``analytical``   — return a dict {field_name: ndarray} or None.
  3. Implement ``pinn_residuals`` — return (pde_loss, bc_loss) tensors.
  4. Implement ``supervised_data`` — return the 7-tuple expected by Arena.
  5. Call ``register_problem(instance)`` before building the Arena config.
  6. Reference the name in the config and run as usual.

Usage
-----
    python examples/arena_pipelines/06_custom_problem.py
    python examples/arena_pipelines/06_custom_problem.py --problem 1d
    python examples/arena_pipelines/06_custom_problem.py --problem 2d
    python examples/arena_pipelines/06_custom_problem.py --epochs 3000
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from pinneaple_arena import Arena, ArenaConfig, ArenaProblem, register_problem


# ── helpers (same as in the built-in problems) ────────────────────────────────

def _unwrap(out):
    if torch.is_tensor(out):
        return out
    if hasattr(out, "y") and torch.is_tensor(out.y):
        return out.y
    return out


def _grad(y, x):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0]


# ── Custom problem 1: 1-D Reaction-Diffusion ─────────────────────────────────

class ReactionDiffusion1D(ArenaProblem):
    """eps*u'' - k*u = f(x)  on [0,1],  u(0)=u(1)=0.

    Manufactured solution: u(x) = sin(πx)
    Source term:           f(x) = -(eps*π² + k)*sin(πx)
    """
    name = "reaction_diffusion_1d_custom"
    description = "1-D reaction-diffusion with manufactured solution u=sin(πx)"
    domain = "Custom"
    input_dim = 1
    output_dim = 1

    # ── analytical solution ───────────────────────────────────────────────────

    def analytical(self, x, eps=0.1, k=1.0, **kw):
        return {"u": np.sin(math.pi * x)}

    # ── PINN physics residual ─────────────────────────────────────────────────

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, eps=0.1, k=1.0, **kw):
        xi = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xi))
        x = xi[:, 0:1]
        u_x = _grad(u, xi)[:, 0:1]
        u_xx = _grad(u_x, xi)[:, 0:1]
        f = -(eps * math.pi**2 + k) * torch.sin(math.pi * x)
        pde_loss = ((eps * u_xx - k * u - f) ** 2).mean()
        bc_loss = ((_unwrap(net(xy_bc)) - uv_bc) ** 2).mean()
        return pde_loss, bc_loss

    # ── training / evaluation data ────────────────────────────────────────────

    def supervised_data(self, n_train=200, n_bc=20, grid_n=100,
                        eps=0.1, k=1.0, **kw):
        rng = np.random.default_rng(42)

        # interior points
        x_int = rng.uniform(0, 1, n_train).reshape(-1, 1)
        Y_int = self.analytical(x_int.ravel(), eps=eps, k=k)["u"].reshape(-1, 1)

        # boundary points: u(0)=0, u(1)=0
        x_bc = np.array([[0.0], [1.0]])
        Y_bc = np.zeros((2, 1))

        # evaluation grid
        x_e = np.linspace(0, 1, grid_n).reshape(-1, 1)
        Y_e = self.analytical(x_e.ravel(), eps=eps, k=k)["u"].reshape(-1, 1)

        return x_int, Y_int, x_bc, Y_bc, x_e, Y_e, ["u"]


# ── Custom problem 2: 2-D Modified Poisson ───────────────────────────────────

class ModifiedPoisson2D(ArenaProblem):
    """Δu + k*u = f(x,y)  on [0,1]²,  u=0 on boundary.

    Manufactured solution: u(x,y) = sin(πx)*sin(πy)
    Source term:           f(x,y) = (k - 2π²)*sin(πx)*sin(πy)
    """
    name = "modified_poisson_2d_custom"
    description = "2-D modified Poisson with manufactured solution u=sin(πx)sin(πy)"
    domain = "Custom"
    input_dim = 2
    output_dim = 1

    def analytical(self, x, y, k=1.0, **kw):
        return {"u": np.sin(math.pi * x) * np.sin(math.pi * y)}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, k=1.0, **kw):
        xi = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xi))
        u_x = _grad(u, xi)[:, 0:1]
        u_y = _grad(u, xi)[:, 1:2]
        u_xx = _grad(u_x, xi)[:, 0:1]
        u_yy = _grad(u_y, xi)[:, 1:2]
        x, y = xi[:, 0:1], xi[:, 1:2]
        f = (k - 2 * math.pi**2) * torch.sin(math.pi * x) * torch.sin(math.pi * y)
        pde_loss = ((u_xx + u_yy + k * u - f) ** 2).mean()
        bc_loss = ((_unwrap(net(xy_bc)) - uv_bc) ** 2).mean()
        return pde_loss, bc_loss

    def supervised_data(self, n_train=400, n_bc=200, grid_n=40, k=1.0, **kw):
        rng = np.random.default_rng(42)

        # interior points
        xy = rng.uniform(0, 1, (n_train, 2))
        Y_int = self.analytical(xy[:, 0], xy[:, 1], k=k)["u"].reshape(-1, 1)

        # boundary: all four edges
        n_e = max(n_bc // 4, 1)
        t = rng.uniform(0, 1, n_e)
        xb = np.concatenate([np.zeros(n_e), np.ones(n_e), t, t])
        yb = np.concatenate([t, t, np.zeros(n_e), np.ones(n_e)])
        x_bc = np.stack([xb, yb], axis=1)
        Y_bc = np.zeros((len(xb), 1))

        # evaluation grid
        gx = np.linspace(0, 1, grid_n)
        GX, GY = np.meshgrid(gx, gx)
        x_e = np.stack([GX.ravel(), GY.ravel()], axis=1)
        Y_e = self.analytical(x_e[:, 0], x_e[:, 1], k=k)["u"].reshape(-1, 1)

        return xy, Y_int, x_bc, Y_bc, x_e, Y_e, ["u"]


# ── Register both custom problems ─────────────────────────────────────────────

register_problem(ReactionDiffusion1D())
register_problem(ModifiedPoisson2D())


# ── Arena configs ─────────────────────────────────────────────────────────────

CONFIG_1D = {
    "problem": {
        "name": "reaction_diffusion_1d_custom",
        "params": {"eps": 0.1, "k": 1.0},
        "grid_n": 100, "n_col": 1000, "n_bc": 20,
        "n_train_supervised": 200, "n_mesh_nodes": 300,
    },
    "models": [
        {
            "name": "VanillaPINN",
            "type": "vanilla_pinn",
            "network": {"hidden": [64, 64, 64], "activation": "tanh"},
            "training": {"epochs": 3000, "lr": 1e-3},
        },
        {
            "name": "ModifiedMLP",
            "type": "modified_mlp",
            "network": {"hidden": [64, 64, 64], "activation": "tanh"},
            "training": {"epochs": 3000, "lr": 1e-3},
        },
    ],
    "output": {
        "dir": "outputs/custom_1d/",
        "prefix": "rd1d",
        "save_figures": True,
    },
}

CONFIG_2D = {
    "problem": {
        "name": "modified_poisson_2d_custom",
        "params": {"k": 1.0},
        "grid_n": 40, "n_col": 2000, "n_bc": 200,
        "n_train_supervised": 400, "n_mesh_nodes": 600,
    },
    "models": [
        {
            "name": "VanillaPINN",
            "type": "vanilla_pinn",
            "network": {"hidden": [64, 64, 64, 64], "activation": "tanh"},
            "training": {"epochs": 4000, "lr": 1e-3},
        },
        {
            "name": "SIREN",
            "type": "siren",
            "network": {"hidden": [64, 64, 64, 64], "omega_0": 30.0},
            "training": {"epochs": 4000, "lr": 5e-4},
        },
    ],
    "output": {
        "dir": "outputs/custom_2d/",
        "prefix": "mpoisson",
        "save_figures": True,
    },
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Custom problem Arena example")
    parser.add_argument(
        "--problem", choices=["1d", "2d", "both"], default="both",
        help="Which custom problem to run (default: both)"
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training epochs for all models")
    args = parser.parse_args()

    def _apply_epochs(cfg, epochs):
        if epochs is None:
            return cfg
        import copy
        cfg = copy.deepcopy(cfg)
        for m in cfg["models"]:
            m["training"]["epochs"] = epochs
        return cfg

    if args.problem in ("1d", "both"):
        print("\n" + "=" * 60)
        print("  Running custom 1-D Reaction-Diffusion problem")
        print("=" * 60)
        cfg = _apply_epochs(CONFIG_1D, args.epochs)
        Arena(ArenaConfig.from_dict(cfg)).run()

    if args.problem in ("2d", "both"):
        print("\n" + "=" * 60)
        print("  Running custom 2-D Modified Poisson problem")
        print("=" * 60)
        cfg = _apply_epochs(CONFIG_2D, args.epochs)
        Arena(ArenaConfig.from_dict(cfg)).run()


if __name__ == "__main__":
    main()
