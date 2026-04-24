"""
Bridge between the app and pinneaple_solvers.
Provides simplified run functions for the UI.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np


def run_lbm(problem: Dict, steps: int = 3000, save_every: int = 500) -> Dict:
    """Run an LBM simulation from a problem dict."""
    import torch
    from pinneaple_solvers.lbm import LBMSolver, cylinder_mask, rectangle_mask

    p = problem.get("params", {})
    nx = int(p.get("nx", 128)); ny = int(p.get("ny", 64))
    Re = float(p.get("Re", 100)); u_in = float(p.get("u_in", 0.05))

    # Build obstacle if requested
    obstacle = None
    obs_cfg = p.get("obstacle")
    if obs_cfg and obs_cfg.get("type") == "cylinder":
        obstacle = cylinder_mask(nx, ny,
            cx=obs_cfg.get("cx", nx // 4),
            cy=obs_cfg.get("cy", ny // 2),
            r=obs_cfg.get("r", ny // 8),
        )
    elif obs_cfg and obs_cfg.get("type") == "rectangle":
        obstacle = rectangle_mask(nx, ny,
            x0=obs_cfg.get("x0", 20), x1=obs_cfg.get("x1", 30),
            y0=obs_cfg.get("y0", 20), y1=obs_cfg.get("y1", 44),
        )

    solver = LBMSolver(nx=nx, ny=ny, Re=Re, u_in=u_in, obstacle_mask=obstacle)
    out = solver.forward(steps=steps, save_every=save_every)
    e = out.extras
    return {
        "type":       "lbm",
        "ux":         e["ux"].numpy(),
        "uy":         e["uy"].numpy(),
        "rho":        e["rho"].numpy(),
        "vel_mag":    e["vel_mag"].numpy(),
        "trajectory_ux": [t.numpy() for t in e["trajectory_ux"]],
        "trajectory_uy": [t.numpy() for t in e["trajectory_uy"]],
        "obstacle":   obstacle.numpy() if obstacle is not None else None,
        "nx": nx, "ny": ny, "Re": Re,
    }


def run_fdm(problem: Dict, nx: int = 64, ny: int = 64) -> Dict:
    """Run FDM solver (Poisson/Laplace/Heat)."""
    from pinneaple_solvers.fdm import FDMSolver
    import torch

    domain = problem.get("domain", {"x": (0, 1), "y": (0, 1)})
    keys = list(domain.keys())[:2]
    x0, x1 = domain[keys[0]]
    y0, y1 = domain[keys[1]]
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Source term for Poisson
    import math
    f = 2 * (math.pi**2) * np.sin(math.pi * X) * np.sin(math.pi * Y)

    solver = FDMSolver(nx=nx, ny=ny, dx=(x1-x0)/(nx-1), dy=(y1-y0)/(ny-1))
    try:
        out = solver.forward(
            source=torch.tensor(f, dtype=torch.float32),
            bc_val=0.0,
        )
        field = out.result.numpy() if hasattr(out.result, "numpy") else np.array(out.result)
    except Exception:
        field = np.zeros((nx, ny))

    return {
        "type":   "fdm",
        "x":      X, "y": Y,
        "field":  field,
        "label":  "u (solution)",
    }


def run_fem(problem: Dict, nx: int = 20, ny: int = 20) -> Dict:
    """Run FEM solver."""
    from pinneaple_solvers.fem import FEMSolver
    import torch

    domain = problem.get("domain", {"x": (0, 1), "y": (0, 1)})
    keys   = list(domain.keys())[:2]
    bounds = (domain[keys[0]][0], domain[keys[0]][1],
              domain[keys[1]][0], domain[keys[1]][1])
    solver = FEMSolver(nx=nx, ny=ny)
    try:
        out = solver.forward(mesh=(nx, ny, bounds), params={})
        nodes  = out.extras.get("nodes", np.zeros((nx*ny, 2)))
        field  = out.result.numpy() if hasattr(out.result, "numpy") else np.zeros(nx*ny)
    except Exception:
        nn = (nx+1)*(ny+1)
        nodes = np.zeros((nn, 2)); field = np.zeros(nn)

    return {
        "type":   "fem",
        "nodes":  nodes,
        "field":  field,
        "label":  "u (FEM solution)",
    }


def quick_fdm_poisson(nx: int = 64, ny: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return X, Y, u for the unit-square Poisson benchmark (analytical solution available)."""
    import math
    x = np.linspace(0, 1, nx); y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    # Analytical solution for ∇²u = -2π²sin(πx)sin(πy): u = sin(πx)sin(πy)
    u_exact = np.sin(math.pi * X) * np.sin(math.pi * Y)
    # Approximate with FDM (5-point stencil)
    dx = 1.0 / (nx - 1); dy = 1.0 / (ny - 1)
    f = 2 * math.pi**2 * np.sin(math.pi * X) * np.sin(math.pi * Y)
    u = np.zeros_like(f)
    for _ in range(5000):
        u[1:-1, 1:-1] = 0.25 * (
            u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:]
            + dx**2 * f[1:-1, 1:-1]
        )
    return X, Y, u
