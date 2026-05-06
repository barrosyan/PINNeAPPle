from __future__ import annotations

import numpy as np
import torch

from _utils import ensure_repo_on_path

ensure_repo_on_path()

from pinneaple_environment import ns_incompressible_2d_default
from pinneaple_pinn.compiler import LossWeights, compile_problem
from pinneaple_models.pinns.vanilla import VanillaPINN


def sample_uniform(n: int, low: np.ndarray, high: np.ndarray, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n, low.size), dtype=np.float32) * (high - low) + low).astype(np.float32)


def main():
    device = "cpu"

    # Navier–Stokes incompressible (2D + time)
    spec = ns_incompressible_2d_default(Re=100.0, Umax=1.0)

    loss_fn = compile_problem(
        spec,
        weights=LossWeights(w_pde=1.0, w_bc=1.0, w_ic=1.0, w_data=1.0),
    )

    # Domain: (x,y,t) in [0,1]^3
    bmin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    bmax = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    n_col = 80_000
    n_each = 6_000  # inlet/outlet/walls total ~ 18k

    x_col = sample_uniform(n_col, bmin, bmax)

    rng = np.random.default_rng(7)
    # Inlet: x=0
    inlet = sample_uniform(n_each, bmin, bmax)
    inlet[:, 0] = bmin[0]

    # Outlet: x=1
    outlet = sample_uniform(n_each, bmin, bmax)
    outlet[:, 0] = bmax[0]

    # Walls: y=0 or y=1
    walls = sample_uniform(n_each, bmin, bmax)
    top = rng.random((n_each,)) > 0.5
    walls[top, 1] = bmax[1]
    walls[~top, 1] = bmin[1]

    x_bc = np.concatenate([inlet, outlet, walls], axis=0).astype(np.float32)

    # Normals for Neumann outlet dp/dn = 0
    n_inlet = np.tile(np.array([-1.0, 0.0, 0.0], dtype=np.float32), (n_each, 1))
    n_outlet = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (n_each, 1))
    n_walls = np.zeros((n_each, 3), dtype=np.float32)
    n_walls[top, 1] = 1.0
    n_walls[~top, 1] = -1.0
    n_bc = np.concatenate([n_inlet, n_outlet, n_walls], axis=0).astype(np.float32)

    # Context required by the preset inlet parabolic profile (uses bounds)
    ctx = {
        "bounds": {"min": bmin, "max": bmax},
        "tag_masks": {
            "inlet": np.concatenate([np.ones(n_each, bool), np.zeros(n_each, bool), np.zeros(n_each, bool)]),
            "outlet": np.concatenate([np.zeros(n_each, bool), np.ones(n_each, bool), np.zeros(n_each, bool)]),
            "walls": np.concatenate([np.zeros(n_each, bool), np.zeros(n_each, bool), np.ones(n_each, bool)]),
        },
    }

    # Targets: keep y_bc as full out_dim; compile_problem will slice for each condition
    y_bc = np.zeros((x_bc.shape[0], len(spec.fields)), dtype=np.float32)

    masks = {}
    for cond in spec.conditions:
        if cond.kind in ("dirichlet", "neumann", "robin"):
            masks[f"mask_{cond.name}"] = cond.mask(x_bc, ctx)

    batch = {
        "x_col": torch.tensor(x_col, device=device),
        "x_bc": torch.tensor(x_bc, device=device),
        "y_bc": torch.tensor(y_bc, device=device),
        "n_bc": torch.tensor(n_bc, device=device),
        "x_ic": torch.zeros((0, len(spec.coords)), device=device),
        "y_ic": torch.zeros((0, len(spec.fields)), device=device),
        "x_data": torch.zeros((0, len(spec.coords)), device=device),
        "y_data": torch.zeros((0, len(spec.fields)), device=device),
        "ctx": ctx,
        **{k: torch.as_tensor(v, device=device, dtype=torch.bool) for k, v in masks.items()},
    }

    model = VanillaPINN(
        in_dim=len(spec.coords),
        out_dim=len(spec.fields),
        hidden=(128, 128, 128, 128),
        activation="tanh",
    ).to(device)

    out = loss_fn(model, None, batch)
    print("loss keys:", list(out.keys()))
    print("total:", float(out["total"].detach().cpu()))

    # Sanity: how many points each tag got
    for tag, m in ctx["tag_masks"].items():
        print(tag, int(np.sum(m)))


if __name__ == "__main__":
    main()
