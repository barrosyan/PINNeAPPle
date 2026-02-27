from __future__ import annotations

import numpy as np
import torch

from _utils import ensure_repo_on_path

ensure_repo_on_path()

from pinneaple_environment import laplace_2d_default
from pinneaple_pinn.compiler import LossWeights, compile_problem
from pinneaple_models.pinns.vanilla import VanillaPINN


def sample_square(n: int, *, low=-1.0, high=1.0, seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.uniform(low, high, size=(n, 2)).astype(np.float32)
    return X


def sample_square_boundary(n: int, *, low=-1.0, high=1.0, seed=7) -> np.ndarray:
    """Uniformly sample the boundary of a square."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(low, high, size=(n,)).astype(np.float32)
    side = rng.integers(0, 4, size=(n,))

    X = np.empty((n, 2), dtype=np.float32)
    # left/right: x fixed
    m = side == 0
    X[m, 0] = low
    X[m, 1] = t[m]

    m = side == 1
    X[m, 0] = high
    X[m, 1] = t[m]

    # bottom/top: y fixed
    m = side == 2
    X[m, 0] = t[m]
    X[m, 1] = low

    m = side == 3
    X[m, 0] = t[m]
    X[m, 1] = high

    return X


def main():
    device = "cpu"

    # 1) Problem spec (from environment preset)
    spec = laplace_2d_default()

    # 2) Compile the physics + BC loss
    loss_fn = compile_problem(
        spec,
        weights=LossWeights(w_pde=1.0, w_bc=1.0, w_ic=1.0, w_data=1.0),
    )

    # 3) Build a *manual* batch (2D):
    n_col, n_bc = 20_000, 4_000
    x_col = sample_square(n_col)
    x_bc = sample_square_boundary(n_bc)

    # Dirichlet u=0 on boundary
    y_bc = np.zeros((n_bc, len(spec.fields)), dtype=np.float32)

    # Context for tag-based selectors
    ctx = {
        "tag_masks": {
            # Laplace preset expects tag='boundary'
            "boundary": np.ones((n_bc,), dtype=bool),
        }
    }

    # Precompute masks per condition name (optional but efficient)
    masks = {}
    for cond in spec.conditions:
        if cond.kind in ("dirichlet", "neumann", "robin"):
            masks[f"mask_{cond.name}"] = cond.mask(x_bc, ctx)

    batch = {
        "x_col": torch.tensor(x_col, device=device),
        "x_bc": torch.tensor(x_bc, device=device),
        "y_bc": torch.tensor(y_bc, device=device),
        "n_bc": torch.zeros((n_bc, 2), device=device),  # not used for Dirichlet
        "x_ic": torch.zeros((0, 2), device=device),
        "y_ic": torch.zeros((0, len(spec.fields)), device=device),
        "x_data": torch.zeros((0, 2), device=device),
        "y_data": torch.zeros((0, len(spec.fields)), device=device),
        "ctx": ctx,
        **{k: torch.as_tensor(v, device=device, dtype=torch.bool) for k, v in masks.items()},
    }

    # 4) Minimal PINN
    model = VanillaPINN(
        in_dim=len(spec.coords),
        out_dim=len(spec.fields),
        hidden=(128, 128, 128),
        activation="tanh",
    ).to(device)

    # 5) One loss evaluation
    out = loss_fn(model, None, batch)
    print("keys:", list(out.keys()))
    print("pde:", float(out["pde"].detach().cpu()))
    print("total:", float(out["total"].detach().cpu()))


if __name__ == "__main__":
    main()
