"""End-to-end Arena-style example (no external data).

This script demonstrates the flow you described:

A) PINN benchmark (with physics loss) on a *generic, parametric geometry* + PDE + BC/IC
   - Geometry: unit square [0,1]x[0,1] (treated as a "genetic" shape via parameters; extendable)
   - PDE: 2D heat equation with manufactured solution
     T(x,y,t) = sin(pi x) sin(pi y) exp(-2 pi^2 alpha t)
   - BC: Dirichlet T=0 on boundary
   - IC: T(x,y,0)=sin(pi x) sin(pi y)
   - Data: sparse supervised points inside the domain
   - Models compared:
       1) vanilla_pinn   (physics loss enabled)
       2) supervised_mlp (no physics; data-only baseline)

B) Digital-twin-like temporal dataset (no physics)
   - Synthetic multivariate sensor signal generated from coupled oscillators + drift
   - Task: predict next-step state from a lag window (flattened features)
   - Models compared:
       1) linear_forecaster (very simple baseline)
       2) mlp_forecaster    (nonlinear baseline)

Outputs
-------
Creates run folders under:
  runs/examples_generic/heat_pinn/*
  runs/examples_generic/digital_twin/*

And writes comparison tables/plots under:
  runs/examples_generic/compare/*

Run
---
python examples/arena/04_end_to_end_generic_pinn_and_digital_twin.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

# --- Pinneaple imports
from pinneaple_environment.spec import ProblemSpec, PDETermSpec
from pinneaple_environment.conditions import DirichletBC, InitialCondition, DataConstraint

from pinneaple_arena.pipeline.dataset_builder import build_from_solver, build_from_real_data
from pinneaple_arena.runner.compare import compare_runs

from pinneaple_models.registry import ModelRegistry
from pinneaple_models.register_all import register_all

from pinneaple_pinn.compiler.compile import compile_problem
from pinneaple_pinn.compiler.loss import LossWeights
from pinneaple_pinn.compiler.dataset import SingleBatchDataset, dict_collate

from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.losses import build_loss


# -----------------------------
# Utilities
# -----------------------------

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _seed_all(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class XYPreprocess:
    """Pick supervised (x,y) from batch.

    - If x_data/y_data exist -> supervised regression.
    - Else -> physics-only mode (no y).
    """

    prefer_data: bool = True

    def fit(self, batch_list):
        return self

    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(batch)
        if self.prefer_data and isinstance(out.get("x_data"), torch.Tensor) and isinstance(out.get("y_data"), torch.Tensor):
            if out["x_data"].numel() > 0 and out["y_data"].numel() > 0:
                out["x"] = out["x_data"]
                out["y"] = out["y_data"]
                return out
        out["x"] = out.get("x_col")
        return out


def _train_one(
    *,
    model_name: str,
    model_kwargs: Dict[str, Any],
    batch: Dict[str, Any],
    out_dir: Path,
    problem_spec: ProblemSpec | None,
    supports_physics_loss: bool,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 1024,
    device: str | None = None,
) -> Path:
    """Train a single model and write predictions.npz into out_dir/model_name."""

    run_dir = _ensure_dir(out_dir / model_name)

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_t = torch.device(dev)

    # DataLoader from unified PINN-style batch
    ds = SingleBatchDataset(batch)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=dict_collate)
    val_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=dict_collate)

    # Build model
    model = ModelRegistry.build(model_name, **model_kwargs)
    model.to(device_t)

    # Physics loss
    physics_loss_fn = None
    if problem_spec is not None and supports_physics_loss:
        physics_loss_fn = compile_problem(problem_spec, weights=LossWeights())

    loss = build_loss(
        problem_spec=problem_spec,
        model_capabilities={"supports_physics_loss": bool(supports_physics_loss)},
        weights={"supervised": 1.0, "physics": 1.0, "bc": 1.0, "ic": 1.0},
        supervised_kind="mse",
        physics_loss_fn=physics_loss_fn,
    )

    pre = XYPreprocess(prefer_data=True)

    def loss_fn(model_mod: torch.nn.Module, y_hat: torch.Tensor, b: Dict[str, Any]):
        return loss(model_mod, y_hat, b)

    trainer = Trainer(model, loss_fn=loss_fn, preprocess=pre)
    cfg = TrainConfig(
        epochs=int(epochs),
        lr=float(lr),
        device=dev,
        run_name=model_name,
        log_dir=str(out_dir),
        save_best=True,
        amp=False,
    )

    trainer.fit(loader, val_loader, cfg)

    # Predictions on y_data if available else on x_col
    model.eval()
    with torch.no_grad():
        if isinstance(batch.get("x_data"), torch.Tensor) and isinstance(batch.get("y_data"), torch.Tensor) and batch["x_data"].numel() > 0:
            x_eval = batch["x_data"].to(device_t)
            y_eval = batch["y_data"].detach().cpu().numpy()
        else:
            x_eval = batch["x_col"].to(device_t)
            y_eval = None

        y_hat = model(x_eval)
        if hasattr(y_hat, "y"):
            y_hat_t = y_hat.y  # type: ignore
        else:
            y_hat_t = y_hat
        y_hat_np = y_hat_t.detach().cpu().numpy()

    np.savez(run_dir / "predictions.npz", y_hat=y_hat_np, y=(y_eval if y_eval is not None else y_hat_np))
    return run_dir


# -----------------------------
# A) PINN benchmark (physics vs no-physics)
# -----------------------------

def manufactured_heat_solution(xyt: np.ndarray, alpha: float) -> np.ndarray:
    x = xyt[:, 0]
    y = xyt[:, 1]
    t = xyt[:, 2]
    T = np.sin(math.pi * x) * np.sin(math.pi * y) * np.exp(-2.0 * (math.pi**2) * alpha * t)
    return T.astype(np.float32)[:, None]


def make_heat_problem(alpha: float = 0.1) -> ProblemSpec:
    pde = PDETermSpec(kind="heat_equation", fields=("T",), coords=("x", "y", "t"), params={"alpha": float(alpha)})

    bc = DirichletBC(
        name_or_values="dirichlet_T0",
        fields=("T",),
        selector_type="all",
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
        weight=1.0,
    )

    ic = InitialCondition(
        name_or_values="ic_sine",
        fields=("T",),
        selector_type="all",
        value_fn=lambda X, ctx: manufactured_heat_solution(X, alpha=ctx["alpha"]),
        weight=1.0,
    )

    data = DataConstraint(
        name_or_values="data_sparse",
        fields=("T",),
        selector_type="all",
        value_fn=lambda X, ctx: manufactured_heat_solution(X, alpha=ctx["alpha"]),
        weight=1.0,
    )

    return ProblemSpec(
        name="heat2d_manufactured",
        dim=3,
        coords=("x", "y", "t"),
        fields=("T",),
        pde=pde,
        conditions=(bc, ic, data),
        sample_defaults={"n_collocation": 8192, "n_bc": 4096, "n_ic": 4096, "n_data": 2048},
    )


def solver_fn_heat(*, problem_spec: ProblemSpec, geometry: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Generate collocation + BC/IC + sparse data for heat equation."""

    alpha = float(cfg.get("alpha", 0.1))
    n_col = int(cfg.get("n_collocation", 8192))
    n_bc = int(cfg.get("n_boundary", 4096))
    n_ic = int(cfg.get("n_initial", 4096))
    n_data = int(cfg.get("n_data", 2048))

    # Domain
    x0, x1 = float(geometry.get("x0", 0.0)), float(geometry.get("x1", 1.0))
    y0, y1 = float(geometry.get("y0", 0.0)), float(geometry.get("y1", 1.0))
    t0, t1 = float(geometry.get("t0", 0.0)), float(geometry.get("t1", 1.0))

    # Collocation points (interior)
    x = np.random.uniform(x0, x1, size=(n_col, 1)).astype(np.float32)
    y = np.random.uniform(y0, y1, size=(n_col, 1)).astype(np.float32)
    t = np.random.uniform(t0, t1, size=(n_col, 1)).astype(np.float32)
    x_col = np.concatenate([x, y, t], axis=1)

    # Boundary points: sample on the 4 edges, with random t
    n_edge = max(1, n_bc // 4)
    tb = np.random.uniform(t0, t1, size=(n_edge, 1)).astype(np.float32)
    ys = np.random.uniform(y0, y1, size=(n_edge, 1)).astype(np.float32)
    xs = np.random.uniform(x0, x1, size=(n_edge, 1)).astype(np.float32)

    left = np.concatenate([np.full_like(ys, x0), ys, tb], axis=1)
    right = np.concatenate([np.full_like(ys, x1), ys, tb], axis=1)
    bottom = np.concatenate([xs, np.full_like(xs, y0), tb], axis=1)
    top = np.concatenate([xs, np.full_like(xs, y1), tb], axis=1)
    x_bc = np.concatenate([left, right, bottom, top], axis=0)[:n_bc]
    y_bc = np.zeros((x_bc.shape[0], 1), dtype=np.float32)

    # Initial condition points: t=0
    xi = np.random.uniform(x0, x1, size=(n_ic, 1)).astype(np.float32)
    yi = np.random.uniform(y0, y1, size=(n_ic, 1)).astype(np.float32)
    ti = np.full((n_ic, 1), t0, dtype=np.float32)
    x_ic = np.concatenate([xi, yi, ti], axis=1)
    y_ic = manufactured_heat_solution(x_ic, alpha=alpha)

    # Sparse data (like measurements)
    xd = np.random.uniform(x0, x1, size=(n_data, 1)).astype(np.float32)
    yd = np.random.uniform(y0, y1, size=(n_data, 1)).astype(np.float32)
    td = np.random.uniform(t0, t1, size=(n_data, 1)).astype(np.float32)
    x_data = np.concatenate([xd, yd, td], axis=1)
    y_data = manufactured_heat_solution(x_data, alpha=alpha)

    return {
        "x_col": x_col,
        "x_bc": x_bc,
        "y_bc": y_bc,
        "x_ic": x_ic,
        "y_ic": y_ic,
        "x_data": x_data,
        "y_data": y_data,
        "ctx": {"alpha": alpha},
    }


# -----------------------------
# B) Digital twin temporal dataset (no physics)
# -----------------------------

def generate_digital_twin_timeseries(
    *,
    T: int = 3000,
    dt: float = 0.01,
    noise: float = 0.01,
) -> np.ndarray:
    """Synthetic multivariate time series: coupled oscillators with mild nonlinearity."""
    x = np.zeros((T, 4), dtype=np.float32)
    # state: [p1, v1, p2, v2]
    p1, v1, p2, v2 = 0.2, 0.0, -0.1, 0.0
    for i in range(T):
        # dynamics
        a1 = -1.5 * p1 - 0.2 * v1 + 0.4 * (p2 - p1) - 0.1 * (p1**3)
        a2 = -1.0 * p2 - 0.15 * v2 + 0.4 * (p1 - p2) - 0.08 * (p2**3)

        v1 = v1 + dt * a1
        p1 = p1 + dt * v1
        v2 = v2 + dt * a2
        p2 = p2 + dt * v2

        x[i] = [p1, v1, p2, v2]

    x += noise * np.random.randn(*x.shape).astype(np.float32)
    return x


def make_lagged_dataset(series: np.ndarray, lag: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten lag window -> predict next step."""
    X, Y = [], []
    for i in range(lag, len(series) - 1):
        window = series[i - lag : i].reshape(-1)
        target = series[i + 1]
        X.append(window)
        Y.append(target)
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


# -----------------------------
# Register example-only baseline models
# -----------------------------


class SupervisedMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.net(x)


class LinearForecaster(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor):
        return self.lin(x)


def register_example_models() -> None:
    # Avoid re-register on repeated runs.
    if "supervised_mlp" not in ModelRegistry.list():
        ModelRegistry.register(
            name="supervised_mlp",
            family="examples",
            description="Simple supervised MLP baseline (no physics loss).",
            input_kind="pointwise_coords",
            supports_physics_loss=False,
            expects=["x_data", "y_data"],
            predicts=["T"],
        )(SupervisedMLP)  # type: ignore

    if "linear_forecaster" not in ModelRegistry.list():
        ModelRegistry.register(
            name="linear_forecaster",
            family="examples",
            description="Linear baseline for lagged digital twin forecasting.",
            input_kind="pointwise_coords",
            supports_physics_loss=False,
            expects=["x_data", "y_data"],
            predicts=["state"],
        )(LinearForecaster)  # type: ignore

    if "mlp_forecaster" not in ModelRegistry.list():
        ModelRegistry.register(
            name="mlp_forecaster",
            family="examples",
            description="MLP baseline for lagged digital twin forecasting.",
            input_kind="pointwise_coords",
            supports_physics_loss=False,
            expects=["x_data", "y_data"],
            predicts=["state"],
        )(SupervisedMLP)  # reuse


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    _seed_all(7)

    # Populate built-in models
    register_all()
    register_example_models()

    root = _ensure_dir("runs/examples_generic")

    # ------------------
    # A) Heat equation PINN benchmark
    # ------------------
    alpha = 0.1
    problem = make_heat_problem(alpha=alpha)

    geometry = {"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0, "t0": 0.0, "t1": 1.0}
    solver_cfg = {
        "solver_fn": solver_fn_heat,
        "alpha": alpha,
        "n_collocation": 8192,
        "n_boundary": 4096,
        "n_initial": 4096,
        "n_data": 4096,
    }
    ds_heat = build_from_solver(problem, geometry, solver_cfg)

    heat_out = _ensure_dir(root / "heat_pinn")

    heat_models = [
        {
            "name": "vanilla_pinn",
            "kwargs": {"in_dim": 3, "out_dim": 1, "hidden": 128, "depth": 5, "dropout": 0.0},
            "physics": True,
            "epochs": 400,
            "lr": 2e-3,
        },
        {
            "name": "supervised_mlp",
            "kwargs": {"in_dim": 3, "out_dim": 1, "hidden": 256, "depth": 4, "dropout": 0.0},
            "physics": False,
            "epochs": 400,
            "lr": 2e-3,
        },
    ]

    heat_run_dirs: List[Path] = []
    for m in heat_models:
        heat_run_dirs.append(
            _train_one(
                model_name=m["name"],
                model_kwargs=m["kwargs"],
                batch=ds_heat.batch,
                out_dir=heat_out,
                problem_spec=problem,
                supports_physics_loss=bool(m["physics"]),
                epochs=int(m["epochs"]),
                lr=float(m["lr"]),
                batch_size=1024,
            )
        )

    compare_root = _ensure_dir(root / "compare")
    compare_runs(heat_run_dirs, out_dir=compare_root / "heat", title="heat_pinn_compare", make_plots=True)

    # ------------------
    # B) Digital twin temporal dataset benchmark (no physics)
    # ------------------
    series = generate_digital_twin_timeseries(T=4000, dt=0.01, noise=0.01)
    lag = 20
    X, Y = make_lagged_dataset(series, lag=lag)

    # Train/test split
    n = X.shape[0]
    n_train = int(0.8 * n)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]

    # Build "real data" style dataset (supervised only)
    ds_dt_train = build_from_real_data({"x_data": X_train, "y_data": Y_train, "ctx": {"kind": "digital_twin"}})

    dt_out = _ensure_dir(root / "digital_twin")

    dt_models = [
        {
            "name": "linear_forecaster",
            "kwargs": {"in_dim": X_train.shape[1], "out_dim": Y_train.shape[1]},
            "epochs": 200,
            "lr": 2e-3,
        },
        {
            "name": "mlp_forecaster",
            "kwargs": {"in_dim": X_train.shape[1], "out_dim": Y_train.shape[1], "hidden": 256, "depth": 4},
            "epochs": 200,
            "lr": 2e-3,
        },
    ]

    dt_run_dirs: List[Path] = []
    for m in dt_models:
        # For compare.py we want y/y_hat on test set; so we train on train batch,
        # then overwrite predictions.npz using test arrays.
        run_dir = _train_one(
            model_name=m["name"],
            model_kwargs=m["kwargs"],
            batch=ds_dt_train.batch,
            out_dir=dt_out,
            problem_spec=None,
            supports_physics_loss=False,
            epochs=int(m["epochs"]),
            lr=float(m["lr"]),
            batch_size=2048,
        )

        # Load the trained model weights back? MVP: _train_one doesn't save ckpt path.
        # Instead, we re-instantiate and quickly retrain for a few epochs on train for simplicity.
        # If you want exact reuse, extend Trainer to save best.ckpt and load.
        model = ModelRegistry.build(m["name"], **m["kwargs"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)

        # quick-fit (small) to get usable model for test export
        batch = ds_dt_train.batch
        ds = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
        loader = torch.utils.data.DataLoader(ds, batch_size=2048, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=float(m["lr"]))
        for _ in range(3):
            for xb, yb in loader:
                xb = xb.to(dev)
                yb = yb.to(dev)
                opt.zero_grad()
                pred = model(xb)
                loss = torch.mean((pred - yb) ** 2)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            y_hat = model(torch.from_numpy(X_test).to(dev)).detach().cpu().numpy()
        np.savez(run_dir / "predictions.npz", y_hat=y_hat, y=Y_test)

        dt_run_dirs.append(run_dir)

    compare_runs(dt_run_dirs, out_dir=compare_root / "digital_twin", title="digital_twin_compare", make_plots=True)

    print("\nDone.")
    print(f"Heat runs: {[str(p) for p in heat_run_dirs]}")
    print(f"Digital-twin runs: {[str(p) for p in dt_run_dirs]}")
    print(f"Compare outputs: {compare_root}")


if __name__ == "__main__":
    main()
