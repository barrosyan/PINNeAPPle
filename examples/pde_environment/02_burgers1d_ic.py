from __future__ import annotations

import numpy as np
import torch

from _utils import ensure_repo_on_path

ensure_repo_on_path()

from pinneaple_environment import burgers_1d_default
from pinneaple_pinn.compiler import LossWeights, compile_problem
from pinneaple_models.pinns.vanilla import VanillaPINN


def sample_xt(n: int, *, xlim=(-1.0, 1.0), tlim=(0.0, 1.0), seed=7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(xlim[0], xlim[1], size=(n, 1)).astype(np.float32)
    t = rng.uniform(tlim[0], tlim[1], size=(n, 1)).astype(np.float32)
    return np.concatenate([x, t], axis=1)


def main():
    device = "cpu"
    spec = burgers_1d_default()

    loss_fn = compile_problem(
        spec,
        weights=LossWeights(w_pde=1.0, w_bc=1.0, w_ic=1.0, w_data=1.0),
    )

    n_col = 40_000
    n_ic = 8_000

    x_col = sample_xt(n_col)

    # Initial condition points: t=0 exactly
    rng = np.random.default_rng(7)
    x0 = rng.uniform(-1.0, 1.0, size=(n_ic, 1)).astype(np.float32)
    t0 = np.zeros((n_ic, 1), dtype=np.float32)
    x_ic = np.concatenate([x0, t0], axis=1)

    # Evaluate IC values directly from spec (so the example stays consistent with preset)
    ctx = {}
    ic_cond = next(c for c in spec.conditions if c.kind == "initial")
    y_ic = ic_cond.values(x_ic, ctx).astype(np.float32)

    batch = {
        "x_col": torch.tensor(x_col, device=device),
        "x_bc": torch.zeros((0, 2), device=device),
        "y_bc": torch.zeros((0, len(spec.fields)), device=device),
        "n_bc": torch.zeros((0, 2), device=device),
        "x_ic": torch.tensor(x_ic, device=device),
        "y_ic": torch.tensor(y_ic, device=device),
        "x_data": torch.zeros((0, 2), device=device),
        "y_data": torch.zeros((0, len(spec.fields)), device=device),
        "ctx": ctx,
    }

    model = VanillaPINN(
        in_dim=len(spec.coords),
        out_dim=len(spec.fields),
        hidden=(128, 128, 128),
        activation="tanh",
    ).to(device)

    out = loss_fn(model, None, batch)
    print("keys:", list(out.keys()))
    print("pde:", float(out["pde"].detach().cpu()))
    print("total:", float(out["total"].detach().cpu()))


if __name__ == "__main__":
    main()
