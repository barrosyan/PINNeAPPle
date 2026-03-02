
from __future__ import annotations

"""Train a single model run from a JSON config (no on-disk bundle required).

This runner is designed for sweeps (between-model parallelism) where the dataset
source may be:
  - solver-generated (arena.data_source='solver')
  - real data adapters (arena.data_source='real')
  - legacy bundle on disk (arena.data_source='bundle')

It writes into a run directory:
  - run_cfg.json
  - metrics.json
  - predictions.npz (with y_hat and y)
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from pinneaple_models.registry import ModelRegistry
from pinneaple_models.adapters.base import select_adapter
from pinneaple_models.register_all import register_all
from pinneaple_train.trainer import Trainer
from pinneaple_train.losses import build_loss
from pinneaple_pinn.compiler import compile_problem
from pinneaple_pinn.compiler.dataset import SingleBatchDataset
from pinneaple_pinn.compiler.collate import dict_collate


from pinneaple_environment.spec import ProblemSpec, PDETermSpec
from pinneaple_environment.conditions import DirichletBC, InitialCondition
from pinneaple_geom.gen.genetic_sdf import make_union_circles_sdf


def _build_geometry_from_cfg(cfg: Dict[str, Any]):
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        return cfg
    kind = str(cfg.get("kind", "union_circles_sdf")).lower()
    if kind == "union_circles_sdf":
        return make_union_circles_sdf(
            n_circles=int(cfg.get("n_circles", 3)),
            seed=int(cfg.get("seed", 0)),
            bounds_min=tuple(cfg.get("bounds_min", (0.0, 0.0))),
            bounds_max=tuple(cfg.get("bounds_max", (1.0, 1.0))),
            r_min=float(cfg.get("r_min", 0.12)),
            r_max=float(cfg.get("r_max", 0.28)),
            intersect_box=bool(cfg.get("intersect_box", True)),
        )
    raise ValueError(f"Unknown geometry kind: {kind}")


def _build_problem_spec_from_cfg(cfg: Dict[str, Any] | None):
    if cfg is None:
        return None
    if isinstance(cfg, ProblemSpec):
        return cfg
    if not isinstance(cfg, dict):
        raise TypeError("problem_spec_cfg must be dict or ProblemSpec")

    name = str(cfg.get("name", "heat2d"))
    coords = tuple(cfg.get("coords", ("x", "y", "t")))
    fields = tuple(cfg.get("fields", ("u",)))
    alpha = float(cfg.get("alpha", 0.01))

    pde = PDETermSpec(kind="heat_equation", fields=fields, coords=coords, params={"alpha": alpha})

    # BC: Dirichlet u=0
    bc = DirichletBC(name="bc_dirichlet", fields=fields, value_fn=None, weight=float(cfg.get("w_bc", 1.0)))

    # IC: provided by dataset_builder via y_ic; in compiler, initial condition uses x_ic/y_ic
    ic = InitialCondition(name="ic", fields=fields, value_fn=None, weight=float(cfg.get("w_ic", 1.0)))

    return ProblemSpec(
        name=name,
        dim=2,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(bc, ic),
        sample_defaults=dict(cfg.get("sample_defaults", {})),
    )

from pinneaple_arena.pipeline.dataset_builder import build_from_bundle, build_from_solver, build_from_real_data
from pinneaple_arena.bundle.loader import load_bundle


def _device_from_cfg(train_cfg: Dict[str, Any]) -> torch.device:
    dev_str = str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    return torch.device(dev_str)


def run_one(
    *,
    run_cfg: Dict[str, Any],
    run_dir: str | Path,
    bundle_root: Optional[str] = None,
) -> Dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    register_all()  # populate model registry

    train_cfg = dict(run_cfg.get("train", {}))
    arena_cfg = dict(run_cfg.get("arena", {}))
    model_cfg = dict(run_cfg.get("model", {}))

    device = _device_from_cfg(train_cfg)

    data_source = str(arena_cfg.get("data_source", "solver")).strip().lower()

    # problem spec is optional (needed for physics)
    problem_spec = _build_problem_spec_from_cfg(arena_cfg.get("problem_spec_cfg", None))

    if data_source == "bundle":
        if not bundle_root:
            raise ValueError("bundle_root is required when arena.data_source='bundle'")
        bundle = load_bundle(bundle_root)
        batch_like = build_from_bundle(
            bundle,
            n_collocation=int(arena_cfg.get("n_collocation", 4096)),
            n_boundary=int(arena_cfg.get("n_boundary", 2048)),
            n_data=int(arena_cfg.get("n_data", 2048)),
            device=None,
        )
    elif data_source == "real":
        batch_like = build_from_real_data(arena_cfg.get("adapter_cfg", {}), device=None)
    else:
        # default: solver
        batch_like = build_from_solver(
            problem_spec,
            _build_geometry_from_cfg(arena_cfg.get("geometry_cfg", None)),
            dict(arena_cfg.get("solver_cfg", {})),
            device=torch.device("cpu"),
        )

    base_batch = batch_like.batch

    # Trainer expects a primary 'x' tensor to feed through the model each step.
    if "x" not in base_batch:
        base_batch["x"] = base_batch.get("x_data") if base_batch.get("x_data") is not None and base_batch.get("x_data").numel() > 0 else base_batch.get("x_col")


    # Build model
    model_name = str(model_cfg.get("name") or "").strip()
    if not model_name:
        raise ValueError("run_cfg.model.name is required.")

    model_kwargs = dict(model_cfg.get("kwargs", {}))
    model = ModelRegistry.build(model_name, **model_kwargs)

    spec = ModelRegistry.spec(model_name)
    adapter = select_adapter(spec)

    # Compile physics (optional)
    physics_loss_fn = None
    if problem_spec is not None and getattr(spec, "supports_physics_loss", False):
        physics_loss_fn = compile_problem(problem_spec, weights=arena_cfg.get("physics_weights", None))

    # Loss builder
    
    loss = build_loss(
        problem_spec=problem_spec,
        model_capabilities={
            "supports_physics_loss": bool(getattr(spec, "supports_physics_loss", False)),
        },
        weights=arena_cfg.get("loss_weights", None),
        supervised_kind=str(arena_cfg.get("supervised_kind", "mse")),
        physics_loss_fn=physics_loss_fn,
    )

    # Wrap model so Trainer can feed only x but loss/physics can still access full batch.
    class _BatchWrapped(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner
            self._last_batch: Optional[Dict[str, Any]] = None

        def set_batch(self, batch: Dict[str, Any]):
            self._last_batch = batch

        def forward(self, x: torch.Tensor):
            if self._last_batch is None:
                return self.inner(x)
            return adapter.forward_batch(self.inner, self._last_batch)

    wrapped = _BatchWrapped(model)

    def loss_fn(model_mod: torch.nn.Module, y_hat: Any, batch: Dict[str, Any]):
        # ensure model uses the same full batch
        wrapped.set_batch(batch)
        pred = y_hat
        # CombinedLoss returns dict[str, tensor]
        return loss(model_mod, pred, batch)


    # DataLoader
    ds = SingleBatchDataset(base_batch)
    bs = int(train_cfg.get("batch_size", 1024))
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=dict_collate)

    # Trainer
    trainer = Trainer(
        model=wrapped,
        loss_fn=loss_fn,
        optimizer_cfg=train_cfg.get("optimizer", {"name": "adam", "lr": 1e-3}),
        max_steps=int(train_cfg.get("max_steps", 2000)),
        log_every=int(train_cfg.get("log_every", 100)),
        device=device,
        ddp=bool(train_cfg.get("ddp", False)),
        amp=bool(train_cfg.get("amp", False)),
        out_dir=str(run_dir),
    )
    metrics = trainer.fit(loader)

    # Inference on data points (x_data -> y_hat) and compare to y_data
    model.eval()
    with torch.no_grad():
        x_data = base_batch.get("x_data")
        y = base_batch.get("y_data")
        if x_data is None or y is None:
            y_hat = torch.zeros((0, 1))
            y = torch.zeros((0, 1))
        else:
            # create a minimal batch for adapter
            b = dict(base_batch)
            b["x"] = x_data
            out = adapter.forward_batch(model, b)
            y_hat = out.y if hasattr(out, "y") else out
    y_hat_np = y_hat.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    np.savez(run_dir / "predictions.npz", y_hat=y_hat_np, y=y_np)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"train": metrics}, f, indent=2)

    with open(run_dir / "run_cfg.json", "w") as f:
        json.dump(run_cfg, f, indent=2)

    return {"run_dir": str(run_dir), "metrics": metrics}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_cfg", type=str, required=True, help="Path to run_cfg.json")
    ap.add_argument("--run_dir", type=str, required=True, help="Output run directory")
    ap.add_argument("--bundle_root", type=str, default="", help="Optional bundle root if using bundle source")
    args = ap.parse_args()

    with open(args.run_cfg, "r") as f:
        run_cfg = json.load(f)

    bundle_root = args.bundle_root or None
    run_one(run_cfg=run_cfg, run_dir=args.run_dir, bundle_root=bundle_root)


if __name__ == "__main__":
    main()
