"""Run a complete Arena experiment from a unified YAML config.

This is the main entry point for the new Arena workflow that supports:
- Any problem preset registered in ``pinneaple_environment``
- Multiple models per experiment (comparison benchmarks)
- Configurable metrics from ``pinneaple_train``
- Inference and visualization via ``pinneaple_inference``

YAML schema
-----------
See ``examples/pinneaple_arena/configs/experiment_burgers_1d.yaml`` for a
full annotated example.

Usage
-----
python -m pinneaple_arena.runner.run_arena_yaml \\
    --config examples/pinneaple_arena/configs/experiment_burgers_1d.yaml \\
    --out_dir data/artifacts/experiments/burgers_1d
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from pinneaple_arena.io.yamlx import load_yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _device(cfg: Dict[str, Any]) -> torch.device:
    d = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    return torch.device(d)


def _to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32)).to(device)


def _load_problem_spec(problem_cfg: Dict[str, Any]):
    """Load a ProblemSpec from a preset name + optional param overrides."""
    from pinneaple_environment.presets.registry import get_preset
    problem_id = str(problem_cfg.get("id", "burgers_1d"))
    params = dict(problem_cfg.get("params", {}))
    return get_preset(problem_id, **params)


def _build_dataset(
    problem_cfg: Dict[str, Any],
    problem_spec,
    arena_cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Build the training batch dict from the chosen data source."""
    data_source = str(problem_cfg.get("data_source", "solver")).lower()
    n_col = int(arena_cfg.get("n_collocation", 8192))
    n_bc = int(arena_cfg.get("n_boundary", 2048))
    n_ic = int(arena_cfg.get("n_ic", 2048))
    n_data = int(arena_cfg.get("n_data", 0))
    seed = int(arena_cfg.get("seed", 0))

    if data_source == "bundle":
        from pinneaple_arena.bundle.loader import load_bundle
        from pinneaple_arena.pipeline.dataset_builder import build_from_bundle
        bundle_root = problem_cfg.get("bundle_root")
        if not bundle_root:
            raise ValueError("problem.bundle_root is required when data_source='bundle'")
        bundle = load_bundle(bundle_root)
        bl = build_from_bundle(bundle, n_collocation=n_col, n_boundary=n_bc, n_data=n_data)
        batch = bl.batch
    elif data_source == "real":
        from pinneaple_arena.pipeline.dataset_builder import build_from_real_data
        adapter_cfg = dict(problem_cfg.get("adapter_cfg", {}))
        bl = build_from_real_data(adapter_cfg)
        batch = bl.batch
    else:
        # solver (default)
        from pinneaple_solvers.problem_runner import generate_pinn_dataset
        solver_cfg_override = dict(problem_cfg.get("solver_cfg", {}))
        batch_np = generate_pinn_dataset(
            problem_spec,
            n_col=n_col,
            n_bc=n_bc,
            n_ic=n_ic,
            n_data=n_data,
            seed=seed,
            solver_cfg_override=solver_cfg_override if solver_cfg_override else None,
        )
        # Convert numpy → torch
        batch = {}
        for k, v in batch_np.items():
            if isinstance(v, np.ndarray):
                batch[k] = _to_tensor(v, device)
            else:
                batch[k] = v

    # Ensure 'x' key for Trainer
    if "x" not in batch:
        batch["x"] = batch.get("x_data") if (batch.get("x_data") is not None and isinstance(batch.get("x_data"), torch.Tensor) and batch["x_data"].numel() > 0) else batch.get("x_col")

    # Move all tensors to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    return batch


class _SingleBatchDataset(torch.utils.data.Dataset):
    """Dataset that returns the same full batch every step."""
    def __init__(self, batch: Dict[str, Any]):
        self._batch = batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self._batch


def _dict_collate(samples):
    return samples[0]


def _build_model(model_cfg: Dict[str, Any]):
    """Build a model from the model registry or a built-in fallback."""
    model_name = str(model_cfg.get("name", "")).strip()
    model_kwargs = dict(model_cfg.get("kwargs", {}))

    if model_name:
        try:
            from pinneaple_models.registry import ModelRegistry
            from pinneaple_models.register_all import register_all
            register_all()
            model = ModelRegistry.build(model_name, **model_kwargs)
            spec = ModelRegistry.spec(model_name)
            return model, spec
        except Exception:
            pass

    # Fallback: simple MLP
    hidden = list(model_cfg.get("hidden", [64, 64, 64, 64]))
    activation = str(model_cfg.get("activation", "tanh"))
    in_dim = int(model_cfg.get("in_dim", 2))
    out_dim = int(model_cfg.get("out_dim", 1))

    act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}.get(activation.lower(), nn.Tanh)
    layers: List[nn.Module] = []
    dims = [in_dim] + list(hidden) + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_fn())
    model = nn.Sequential(*layers)
    return model, None


def _build_physics_loss(problem_spec, model_spec, weights_cfg: Dict[str, Any]):
    """Compile physics losses from problem spec if model supports it."""
    if problem_spec is None:
        return None
    supports_physics = getattr(model_spec, "supports_physics_loss", False) if model_spec else False
    if not supports_physics:
        return None
    try:
        from pinneaple_pinn.compiler import compile_problem
        return compile_problem(problem_spec, weights=weights_cfg if weights_cfg else None)
    except Exception:
        return None


def _build_loss_fn(problem_spec, model_spec, physics_weights_cfg: Dict[str, Any]):
    """Build a combined loss function for PINN training."""
    physics_loss_fn = _build_physics_loss(problem_spec, model_spec, physics_weights_cfg)

    try:
        from pinneaple_train.losses import build_loss
        loss_obj = build_loss(
            problem_spec=problem_spec,
            model_capabilities={"supports_physics_loss": physics_loss_fn is not None},
            weights=physics_weights_cfg if physics_weights_cfg else None,
            supervised_kind="mse",
            physics_loss_fn=physics_loss_fn,
        )

        def loss_fn(model: nn.Module, y_hat: Any, batch: Dict[str, Any]):
            return loss_obj(model, y_hat, batch)

        return loss_fn
    except Exception:
        pass

    # Fallback: basic PINN loss using PDE conditions from spec
    def _basic_pinn_loss(model: nn.Module, y_hat: Any, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        device = next(model.parameters()).device

        # Data/supervised loss
        x_data = batch.get("x_data")
        y_data = batch.get("y_data")
        if x_data is not None and y_data is not None and x_data.numel() > 0 and y_data.numel() > 0:
            pred_data = model(x_data)
            if not isinstance(pred_data, torch.Tensor):
                for attr in ("y", "pred", "out"):
                    if hasattr(pred_data, attr):
                        pred_data = getattr(pred_data, attr)
                        break
            losses["supervised"] = torch.mean((pred_data - y_data) ** 2)

        # BC/IC losses
        for key in ("x_bc", "x_ic"):
            y_key = key.replace("x_", "y_")
            x_cond = batch.get(key)
            y_cond = batch.get(y_key)
            if x_cond is not None and y_cond is not None and x_cond.numel() > 0 and y_cond.numel() > 0:
                pred = model(x_cond)
                if not isinstance(pred, torch.Tensor):
                    for attr in ("y", "pred", "out"):
                        if hasattr(pred, attr):
                            pred = getattr(pred, attr)
                            break
                # Match field dims if needed
                if pred.shape[-1] != y_cond.shape[-1]:
                    min_dim = min(pred.shape[-1], y_cond.shape[-1])
                    pred = pred[..., :min_dim]
                    y_cond = y_cond[..., :min_dim]
                losses[key.replace("x_", "") + "_loss"] = torch.mean((pred - y_cond) ** 2)

        if not losses:
            losses["supervised"] = torch.tensor(0.0, device=device, requires_grad=True)

        total = sum(losses.values())
        losses["total"] = total
        return losses

    return _basic_pinn_loss


def _train_model(
    model: nn.Module,
    loss_fn,
    batch: Dict[str, Any],
    train_cfg: Dict[str, Any],
    run_dir: Path,
    model_id: str,
    metrics_obj=None,
) -> Dict[str, Any]:
    """Train a single model with the Trainer, return metrics + history."""
    from pinneaple_train.trainer import Trainer, TrainConfig

    epochs = int(train_cfg.get("epochs", 1000))
    lr = float(train_cfg.get("lr", 1e-3))
    device_str = str(train_cfg.get("device", "cpu"))
    seed = train_cfg.get("seed", None)
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))

    cfg = TrainConfig(
        epochs=epochs,
        lr=lr,
        device=device_str,
        seed=int(seed) if seed is not None else None,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
        log_dir=str(run_dir),
        run_name=model_id,
        save_best=True,
    )

    trainer = Trainer(model=model, loss_fn=loss_fn, metrics=metrics_obj)

    ds = _SingleBatchDataset(batch)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=_dict_collate)

    result = trainer.fit(loader, loader, cfg)
    return result


def _run_inference(
    model: nn.Module,
    problem_spec,
    infer_cfg: Dict[str, Any],
    device: torch.device,
) -> Optional[Any]:
    """Run grid inference and return InferenceResult."""
    try:
        from pinneaple_inference import infer_on_grid_1d, infer_on_grid_2d
    except ImportError:
        return None

    coord_names = tuple(problem_spec.coords)
    fields = list(problem_spec.fields)
    domain_bounds = dict(getattr(problem_spec, "domain_bounds", {}))

    model.eval()
    model.to(device)

    is_2d_spatial = len([c for c in coord_names if c != "t"]) >= 2

    resolution = infer_cfg.get("resolution", [100, 100])
    if isinstance(resolution, int):
        resolution = [resolution, resolution]

    spatial_coords = [c for c in coord_names if c != "t"]

    if is_2d_spatial:
        c0, c1 = spatial_coords[0], spatial_coords[1]
        x_range = domain_bounds.get(c0, (0.0, 1.0))
        y_range = domain_bounds.get(c1, (0.0, 1.0))
        return infer_on_grid_2d(
            model,
            x_range,
            y_range,
            nx=int(resolution[0]),
            ny=int(resolution[1]),
            device=str(device),
            field_names=fields,
            coord_names=(c0, c1),
        )
    else:
        # 1D + time
        c0 = spatial_coords[0] if spatial_coords else coord_names[0]
        c1 = "t" if "t" in coord_names else coord_names[1]
        x_range = domain_bounds.get(c0, (-1.0, 1.0))
        t_range = domain_bounds.get(c1, (0.0, 1.0))
        return infer_on_grid_1d(
            model,
            x_range,
            t_range,
            nx=int(resolution[0]),
            nt=int(resolution[1]),
            device=str(device),
            field_names=fields,
            coord_names=(c0, c1),
        )


def _compute_metrics(
    metrics_obj,
    batch: Dict[str, Any],
    model: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Compute evaluation metrics on data points."""
    if metrics_obj is None:
        return {}
    x_data = batch.get("x_data")
    y_data = batch.get("y_data")
    if x_data is None or y_data is None or x_data.numel() == 0:
        return {}
    model.eval()
    with torch.no_grad():
        y_hat = model(x_data.to(device))
        if not isinstance(y_hat, torch.Tensor):
            for attr in ("y", "pred", "out"):
                if hasattr(y_hat, attr):
                    y_hat = getattr(y_hat, attr)
                    break
        if y_hat.shape[-1] != y_data.shape[-1]:
            min_dim = min(y_hat.shape[-1], y_data.shape[-1])
            y_hat = y_hat[..., :min_dim]
            y_data_use = y_data[..., :min_dim]
        else:
            y_data_use = y_data
        return metrics_obj.compute(y_hat.cpu(), y_data_use.cpu())


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_arena_experiment(
    yaml_path: Union[str, Path],
    *,
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run a complete Arena experiment from a YAML config file.

    Parameters
    ----------
    yaml_path : path to the experiment YAML config
    out_dir : output directory (overrides artifacts_dir in config)

    Returns
    -------
    dict with keys:
      - "experiment_name": str
      - "models": dict of model_id → {"metrics": ..., "train_result": ...}
      - "visualizations": dict of description → file path
      - "out_dir": str
    """
    cfg = load_yaml(str(yaml_path))

    exp_cfg = dict(cfg.get("experiment", {}))
    exp_name = str(exp_cfg.get("name", "arena_experiment"))

    problem_cfg = dict(cfg.get("problem", {}))
    models_cfg = list(cfg.get("models", []))
    metrics_names = list(cfg.get("metrics", ["mse", "rmse", "rel_l2"]))
    viz_cfgs = list(cfg.get("visualizations", []))
    global_arena_cfg = dict(cfg.get("arena", {}))
    infer_cfg = dict(cfg.get("inference", {}))

    if out_dir is None:
        out_dir = Path(str(cfg.get("artifacts_dir", "data/artifacts/experiments"))) / exp_name
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build problem spec
    problem_spec = _load_problem_spec(problem_cfg)

    # Build metrics
    from pinneaple_train.metrics_cfg import build_metrics_from_cfg
    metrics_obj = build_metrics_from_cfg(metrics_names)

    # Infer device from first model or default
    first_train_cfg = dict(models_cfg[0].get("train", {})) if models_cfg else {}
    device = _device(first_train_cfg)

    # Build dataset (shared across models)
    arena_cfg = {**global_arena_cfg, **dict(problem_cfg.get("arena", {}))}
    batch = _build_dataset(problem_cfg, problem_spec, arena_cfg, device)

    # Train each model
    all_model_results: Dict[str, Dict[str, Any]] = {}
    for model_entry in models_cfg:
        model_id = str(model_entry.get("id", f"model_{len(all_model_results)}"))
        model_cfg = dict(model_entry.get("model", {}))
        train_cfg = dict(model_entry.get("train", {}))
        physics_weights_cfg = dict(model_entry.get("physics_weights", {}))

        model_device = _device(train_cfg)
        run_dir = out_dir / "runs" / model_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[Arena] Training model: {model_id}")
        t0 = time.time()

        # Infer in_dim/out_dim from problem spec if not explicitly set
        if "in_dim" not in model_cfg:
            model_cfg["in_dim"] = len(problem_spec.coords)
        if "out_dim" not in model_cfg:
            model_cfg["out_dim"] = len(problem_spec.fields)

        model, model_spec = _build_model(model_cfg)
        model = model.to(model_device)
        loss_fn = _build_loss_fn(problem_spec, model_spec, physics_weights_cfg)

        train_result = _train_model(
            model, loss_fn, batch, train_cfg, run_dir, model_id, metrics_obj
        )
        elapsed = time.time() - t0
        print(f"[Arena] Done in {elapsed:.1f}s — best_val={train_result.get('best_val', float('nan')):.4g}")

        # Evaluate metrics
        eval_metrics = _compute_metrics(metrics_obj, batch, model, model_device)

        # Inference on grid
        infer_resolution = infer_cfg.get("resolution", arena_cfg.get("resolution", [100, 100]))
        infer_cfg_model = {"resolution": infer_resolution}
        inference_result = _run_inference(model, problem_spec, infer_cfg_model, model_device)
        if inference_result is not None:
            inference_result.model_id = model_id

        all_model_results[model_id] = {
            "train_result": train_result,
            "eval_metrics": eval_metrics,
            "inference_result": inference_result,
            "loss_history": [],  # Trainer doesn't expose epoch history yet
            "elapsed_sec": elapsed,
        }

        # Save per-model metrics
        with open(run_dir / "metrics.json", "w") as f:
            json.dump({"train": train_result, "eval": eval_metrics, "elapsed_sec": elapsed}, f, indent=2)

        # Save model weights
        torch.save(model.state_dict(), run_dir / "model.pt")

    # Visualizations
    viz_results: Dict[str, str] = {}
    if viz_cfgs:
        try:
            from pinneaple_inference.visualize import render_visualizations
            viz_out = out_dir / "plots"
            viz_results = render_visualizations(
                viz_cfgs,
                model_results=all_model_results,
                problem_spec=problem_spec,
                out_dir=viz_out,
            )
            print(f"\n[Arena] Saved {len(viz_results)} visualizations to {viz_out}")
        except Exception as e:
            print(f"[Arena] Visualization skipped: {e}")

    # Write summary
    summary = {
        "experiment_name": exp_name,
        "problem_id": problem_cfg.get("id"),
        "models": {
            mid: {
                "best_val": mdata["train_result"].get("best_val"),
                "eval_metrics": mdata["eval_metrics"],
                "elapsed_sec": mdata["elapsed_sec"],
            }
            for mid, mdata in all_model_results.items()
        },
        "visualizations": viz_results,
        "out_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Arena] Experiment '{exp_name}' complete. Results at: {out_dir}")
    return {
        "experiment_name": exp_name,
        "models": all_model_results,
        "visualizations": viz_results,
        "out_dir": str(out_dir),
        "summary": summary,
    }


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Run a pinneaple Arena experiment from YAML")
    ap.add_argument("--config", type=str, required=True, help="Path to experiment YAML")
    ap.add_argument("--out_dir", type=str, default="", help="Output directory (overrides artifacts_dir in config)")
    args = ap.parse_args()

    out_dir = args.out_dir or None
    run_arena_experiment(args.config, out_dir=out_dir)


if __name__ == "__main__":
    main()
