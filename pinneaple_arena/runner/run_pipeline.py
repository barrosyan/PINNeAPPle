"""End-to-end pipeline: geometry → solver → dataset → train → predict → report.

Orchestrates the complete workflow:
  1. Load or generate geometry (pinneaple_geom)
  2. Generate mesh (mesher_2d or gmsh)
  3. Run physical solver (OpenFOAM or FEniCS) to generate ground truth
  4. Build training dataset from solver output
  5. Train surrogate model (PINN, GNN, DeepONet, etc.)
  6. Run inference and compute error metrics
  7. Generate report: error plots, field comparisons, loss curves
  8. Save model + all logs + HTML/JSON report

YAML config example
-------------------
pipeline:
  name: my_experiment
  out_dir: results/my_experiment

geometry:
  type: rectangle         # rectangle | sdf | gmsh | mesh_file
  params:
    xlim: [0, 1]
    ylim: [0, 1]
    obstacle: {type: circle, center: [0.3, 0.5], radius: 0.05}

solver:
  backend: openfoam       # openfoam | fenics | builtin
  params:
    solver: simpleFoam
    n_iterations: 500
    n_cores: 4

problem:
  id: ns_incompressible_2d
  params:
    nu: 0.001
    Re: 100

dataset:
  n_collocation: 10000
  n_boundary: 4000
  strategy: lhs           # uniform | lhs | sobol | adaptive

models:
  - id: pinn_mlp
    type: VanillaPINN
    params:
      hidden: [128, 128, 128, 128]
      activation: tanh
    train:
      epochs: 2000
      lr: 1e-3
      device: cuda
      amp: true
      grad_clip: 1.0
      compile: true       # torch.compile()
      grad_accum_steps: 4
    physics_weights:
      pde: 1.0
      bc: 10.0
      data: 5.0

  - id: deeponet
    type: DeepONet
    ...

metrics: [mse, rmse, rel_l2, r2, max_error]

report:
  plots:
    - type: field_2d
      field: u
    - type: error_map_2d
      field: u
    - type: loss_curve
    - type: comparison_table
  format: html            # html | json | both
  save_model: true
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# ---------------------------------------------------------------------------
# Optional deps — imported lazily inside functions or wrapped in try/except
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.utils.data
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger(__name__)


def _configure_logging(level: int = logging.INFO) -> None:
    """Set up a simple console handler if the root logger has no handlers."""
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                              datefmt="%H:%M:%S")
        )
        root.addHandler(handler)
    root.setLevel(level)


# ---------------------------------------------------------------------------
# YAML loading helper (mirrors run_arena_yaml)
# ---------------------------------------------------------------------------

def _load_config(config_path_or_dict: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """Accept a file path (str/Path) or a pre-built dict."""
    if isinstance(config_path_or_dict, dict):
        return config_path_or_dict
    path = Path(config_path_or_dict)
    # Prefer the project-local YAML loader; fall back to PyYAML
    try:
        from pinneaple_arena.io.yamlx import load_yaml
        return load_yaml(str(path))
    except ImportError:
        pass
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load YAML configs. "
            "Install it with: pip install pyyaml"
        ) from exc


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _step_geometry(geo_cfg: Dict[str, Any]) -> Any:
    """Step 1 — load or generate a geometry/mesh object.

    Parameters
    ----------
    geo_cfg:
        Sub-dict from ``config["geometry"]``.

    Returns
    -------
    A geometry object understood by downstream steps, or ``None`` when the
    geometry section is absent / empty.
    """
    if not geo_cfg:
        log.debug("No geometry config — skipping geometry step.")
        return None

    geo_type = str(geo_cfg.get("type", "rectangle")).lower()
    params: Dict[str, Any] = dict(geo_cfg.get("params", {}))

    log.info("Building geometry: type=%s  params=%s", geo_type, params)
    t0 = time.perf_counter()

    geometry = None

    # Try pinneaple_geom first
    try:
        from pinneaple_geom.factory import build_geometry
        geometry = build_geometry(geo_type, **params)
        log.info("Geometry built via pinneaple_geom in %.2fs", time.perf_counter() - t0)
        return geometry
    except ImportError:
        log.debug("pinneaple_geom not available — trying fallback.")
    except Exception as exc:
        log.warning("pinneaple_geom build_geometry failed: %s", exc)

    # Minimal built-in fallback — return a plain dict describing the domain
    if geo_type == "rectangle":
        xlim = tuple(params.get("xlim", [0.0, 1.0]))
        ylim = tuple(params.get("ylim", [0.0, 1.0]))
        obstacle = params.get("obstacle")
        geometry = {
            "type": "rectangle",
            "xlim": xlim,
            "ylim": ylim,
            "obstacle": obstacle,
        }
    elif geo_type in ("sdf", "mesh_file", "gmsh"):
        geometry = {"type": geo_type, **params}
    else:
        geometry = {"type": geo_type, **params}

    log.info("Geometry (builtin dict) created in %.2fs", time.perf_counter() - t0)
    return geometry


def _step_solver(
    solver_cfg: Dict[str, Any],
    problem_spec: Any,
    geometry: Any,
    out_dir: Path,
) -> Dict[str, np.ndarray]:
    """Step 2 — run the physical solver to obtain ground-truth field data.

    Parameters
    ----------
    solver_cfg:
        Sub-dict from ``config["solver"]``.
    problem_spec:
        A ``ProblemSpec`` (or ``None``).
    geometry:
        Output of :func:`_step_geometry`.
    out_dir:
        Directory where solver output is written.

    Returns
    -------
    Dict mapping field names (e.g. ``"u"``, ``"v"``, ``"p"``) to numpy arrays.
    An empty dict is returned when no solver is configured.
    """
    if not solver_cfg:
        log.info("No solver config — skipping solver step.")
        return {}

    backend = str(solver_cfg.get("backend", "builtin")).lower()
    params: Dict[str, Any] = dict(solver_cfg.get("params", {}))

    log.info("Running solver: backend=%s", backend)
    t0 = time.perf_counter()

    solver_out_dir = out_dir / "solver_output"
    solver_out_dir.mkdir(parents=True, exist_ok=True)

    solver_data: Dict[str, np.ndarray] = {}

    if backend == "openfoam":
        try:
            from pinneaple_solvers.openfoam_runner import run_openfoam
            solver_data = run_openfoam(
                problem_spec=problem_spec,
                geometry=geometry,
                out_dir=str(solver_out_dir),
                **params,
            )
        except ImportError:
            log.warning("pinneaple_solvers.openfoam_runner not found — returning empty solver data.")
        except Exception as exc:
            log.error("OpenFOAM solver failed: %s", exc, exc_info=True)

    elif backend == "fenics":
        try:
            from pinneaple_solvers.fenics_runner import run_fenics
            solver_data = run_fenics(
                problem_spec=problem_spec,
                geometry=geometry,
                out_dir=str(solver_out_dir),
                **params,
            )
        except ImportError:
            log.warning("pinneaple_solvers.fenics_runner not found — returning empty solver data.")
        except Exception as exc:
            log.error("FEniCS solver failed: %s", exc, exc_info=True)

    else:
        # builtin — delegate to pinneaple_solvers.problem_runner if available
        try:
            from pinneaple_solvers.problem_runner import run_builtin_solver
            solver_data = run_builtin_solver(
                problem_spec=problem_spec,
                geometry=geometry,
                out_dir=str(solver_out_dir),
                **params,
            )
        except ImportError:
            log.debug("pinneaple_solvers not available — builtin solver skipped.")
        except Exception as exc:
            log.error("Builtin solver failed: %s", exc, exc_info=True)

    elapsed = time.perf_counter() - t0
    n_fields = len(solver_data)
    log.info(
        "Solver finished in %.2fs — %d field(s) returned: %s",
        elapsed,
        n_fields,
        list(solver_data.keys()),
    )
    return solver_data


def _step_dataset(
    dataset_cfg: Dict[str, Any],
    problem_spec: Any,
    solver_data: Dict[str, np.ndarray],
    geometry: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Step 3 — build train and validation batch dicts.

    Parameters
    ----------
    dataset_cfg:
        Sub-dict from ``config["dataset"]``.
    problem_spec:
        A ``ProblemSpec`` (or ``None``).
    solver_data:
        Ground-truth arrays returned by :func:`_step_solver`.
    geometry:
        Geometry object / dict.

    Returns
    -------
    (train_batch, val_batch) — dicts of str → numpy arrays (device transfer
    happens inside :func:`_step_train_model`).
    """
    n_col = int(dataset_cfg.get("n_collocation", 8192))
    n_bc = int(dataset_cfg.get("n_boundary", 2048))
    n_ic = int(dataset_cfg.get("n_ic", 2048))
    n_data = int(dataset_cfg.get("n_data", 0))
    strategy = str(dataset_cfg.get("strategy", "lhs")).lower()
    val_split = float(dataset_cfg.get("val_split", 0.1))
    seed = int(dataset_cfg.get("seed", 0))

    log.info(
        "Building dataset: n_col=%d  n_bc=%d  n_ic=%d  strategy=%s  seed=%d",
        n_col, n_bc, n_ic, strategy, seed,
    )
    t0 = time.perf_counter()

    batch_np: Dict[str, np.ndarray] = {}

    # Prefer the solver-based dataset generator
    try:
        from pinneaple_solvers.problem_runner import generate_pinn_dataset
        solver_cfg_override = {"sampling_strategy": strategy} if strategy != "lhs" else None
        batch_np = generate_pinn_dataset(
            problem_spec,
            n_col=n_col,
            n_bc=n_bc,
            n_ic=n_ic,
            n_data=n_data,
            seed=seed,
            solver_cfg_override=solver_cfg_override,
        )
    except ImportError:
        log.debug("pinneaple_solvers not available — building minimal dataset.")
    except Exception as exc:
        log.warning("generate_pinn_dataset failed: %s — building minimal dataset.", exc)

    # If solver-based generation produced nothing, stitch together from solver_data
    if not batch_np and solver_data:
        log.info("Stitching dataset from solver_data arrays.")
        batch_np = {k: v for k, v in solver_data.items() if isinstance(v, np.ndarray)}

    # If still empty and problem_spec is available, build a uniform grid fallback
    if not batch_np and problem_spec is not None:
        log.info("Generating uniform collocation points from domain_bounds.")
        rng = np.random.default_rng(seed)
        bounds = dict(getattr(problem_spec, "domain_bounds", {}))
        n_coords = len(problem_spec.coords)
        if bounds:
            lows = np.array([bounds.get(c, (0.0, 1.0))[0] for c in problem_spec.coords], dtype=np.float32)
            highs = np.array([bounds.get(c, (0.0, 1.0))[1] for c in problem_spec.coords], dtype=np.float32)
        else:
            lows = np.zeros(n_coords, dtype=np.float32)
            highs = np.ones(n_coords, dtype=np.float32)
        x_col = rng.uniform(lows, highs, size=(n_col, n_coords)).astype(np.float32)
        batch_np["x_col"] = x_col

    # Ensure "x" key exists (mirrors run_arena_yaml convention)
    if "x" not in batch_np:
        for candidate in ("x_data", "x_col"):
            v = batch_np.get(candidate)
            if v is not None and isinstance(v, np.ndarray) and v.size > 0:
                batch_np["x"] = v
                break

    # Train / val split on collocation points
    train_batch: Dict[str, Any] = {}
    val_batch: Dict[str, Any] = {}

    for key, arr in batch_np.items():
        if not isinstance(arr, np.ndarray) or arr.ndim < 1:
            train_batch[key] = arr
            val_batch[key] = arr
            continue
        n = arr.shape[0]
        n_val = max(1, int(n * val_split))
        n_train = n - n_val
        rng_split = np.random.default_rng(seed + 1)
        idx = rng_split.permutation(n)
        train_batch[key] = arr[idx[:n_train]]
        val_batch[key] = arr[idx[n_train:]]

    log.info(
        "Dataset ready in %.2fs — train keys=%s",
        time.perf_counter() - t0,
        list(train_batch.keys()),
    )
    return train_batch, val_batch


class _SingleBatchDataset(torch.utils.data.Dataset if _HAS_TORCH else object):
    """Wraps a single dict-batch so it can be used with a DataLoader."""

    def __init__(self, batch: Dict[str, Any]):
        self._batch = batch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._batch


def _dict_collate(samples: List[Any]) -> Any:
    return samples[0]


def _numpy_batch_to_torch(
    batch: Dict[str, Any],
    device: "torch.device",
) -> Dict[str, Any]:
    """Convert numpy arrays in a batch dict to torch tensors on *device*."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v.astype(np.float32)).to(device)
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _step_train_model(
    model_cfg: Dict[str, Any],
    problem_spec: Any,
    train_batch: Dict[str, Any],
    val_batch: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Any, Dict[str, Any]]:
    """Step 4 — build and train a single surrogate model.

    Parameters
    ----------
    model_cfg:
        One entry from ``config["models"]`` — contains ``id``, ``type``,
        ``params``, ``train``, and ``physics_weights`` sub-dicts.
    problem_spec:
        A ``ProblemSpec`` (or ``None``).
    train_batch, val_batch:
        Numpy-valued batch dicts from :func:`_step_dataset`.
    out_dir:
        Root output directory; a sub-directory ``runs/<model_id>`` is created.

    Returns
    -------
    (model, history) where *history* is the dict returned by ``Trainer.fit``.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for the training step.")

    model_id = str(model_cfg.get("id", "model_0"))
    model_type = str(model_cfg.get("type", "VanillaPINN"))
    model_params: Dict[str, Any] = dict(model_cfg.get("params", {}))
    train_cfg: Dict[str, Any] = dict(model_cfg.get("train", {}))
    physics_weights: Dict[str, Any] = dict(model_cfg.get("physics_weights", {}))

    run_dir = out_dir / "runs" / model_id
    run_dir.mkdir(parents=True, exist_ok=True)

    device_str = str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    device = torch.device(device_str)

    log.info("=== Training model: %s (type=%s, device=%s) ===", model_id, model_type, device)
    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Infer in_dim / out_dim from problem_spec when not explicitly given
    # ------------------------------------------------------------------
    if "in_dim" not in model_params and problem_spec is not None:
        model_params["in_dim"] = len(problem_spec.coords)
    if "out_dim" not in model_params and problem_spec is not None:
        model_params["out_dim"] = len(problem_spec.fields)
    in_dim = int(model_params.get("in_dim", 2))
    out_dim = int(model_params.get("out_dim", 1))

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model: nn.Module = _build_model_instance(model_type, model_params, in_dim, out_dim)

    # Optional: torch.compile
    do_compile = bool(train_cfg.get("compile", False))
    if do_compile:
        try:
            from pinneaple_train import maybe_compile
            model = maybe_compile(model)
            log.info("torch.compile applied to %s.", model_id)
        except ImportError:
            try:
                model = torch.compile(model)  # type: ignore[attr-defined]
                log.info("torch.compile applied to %s (direct).", model_id)
            except Exception as exc:
                log.warning("torch.compile skipped: %s", exc)

    model = model.to(device)

    # ------------------------------------------------------------------
    # Build loss function
    # ------------------------------------------------------------------
    loss_fn = _build_loss_fn(problem_spec, physics_weights)

    # ------------------------------------------------------------------
    # Build metrics
    # ------------------------------------------------------------------
    metrics_obj = None
    try:
        from pinneaple_train.metrics_cfg import build_metrics_from_cfg
        metric_names = list(train_cfg.get("metrics", ["mse", "rmse", "rel_l2"]))
        metrics_obj = build_metrics_from_cfg(metric_names)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Convert numpy batches → torch tensors
    # ------------------------------------------------------------------
    torch_train = _numpy_batch_to_torch(train_batch, device)
    torch_val = _numpy_batch_to_torch(val_batch, device)

    # ------------------------------------------------------------------
    # Build TrainConfig
    # ------------------------------------------------------------------
    from pinneaple_train.trainer import Trainer, TrainConfig

    epochs = int(train_cfg.get("epochs", 1000))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    seed = train_cfg.get("seed", None)
    use_amp = bool(train_cfg.get("amp", False))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))

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

    # Optionally attach AMP / GradAccum if the Trainer supports them
    trainer_kwargs: Dict[str, Any] = {"model": model, "loss_fn": loss_fn}
    if metrics_obj is not None:
        trainer_kwargs["metrics"] = metrics_obj

    # Try advanced Trainer features
    if use_amp or grad_accum_steps > 1:
        try:
            from pinneaple_train import AMPContext, GradAccumConfig, GradAccumTrainer
            if grad_accum_steps > 1:
                accum_cfg = GradAccumConfig(steps=grad_accum_steps)
                trainer_kwargs["grad_accum_cfg"] = accum_cfg
                trainer = GradAccumTrainer(**trainer_kwargs)
            else:
                amp_ctx = AMPContext(enabled=use_amp, device_type=device_str)
                trainer_kwargs["amp_ctx"] = amp_ctx
                trainer = Trainer(**trainer_kwargs)
        except ImportError:
            trainer = Trainer(**trainer_kwargs)
    else:
        trainer = Trainer(**trainer_kwargs)

    # ------------------------------------------------------------------
    # Throughput monitor (optional)
    # ------------------------------------------------------------------
    try:
        from pinneaple_train import ThroughputMonitor
        throughput_monitor = ThroughputMonitor(log_every=max(1, epochs // 10))
        trainer.throughput_monitor = throughput_monitor  # type: ignore[attr-defined]
    except ImportError:
        pass

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_ds = _SingleBatchDataset(torch_train)
    val_ds = _SingleBatchDataset(torch_val)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=False, collate_fn=_dict_collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=_dict_collate
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    log.info("Calling Trainer.fit — epochs=%d  lr=%g  grad_clip=%g", epochs, lr, grad_clip)
    history = trainer.fit(train_loader, val_loader, cfg)
    elapsed = time.perf_counter() - t0

    best_val = history.get("best_val", float("nan")) if isinstance(history, dict) else float("nan")
    log.info("Model %s done in %.2fs — best_val=%.4g", model_id, elapsed, best_val)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    ckpt_path = run_dir / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    log.info("Checkpoint saved: %s", ckpt_path)

    # Persist per-model metrics
    metrics_path = run_dir / "metrics.json"
    history_to_save = history if isinstance(history, dict) else {}
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"train": history_to_save, "elapsed_sec": elapsed},
            fh,
            indent=2,
            default=str,
        )

    return model, {"history": history, "elapsed_sec": elapsed, "model_id": model_id}


def _step_inference(
    model: Any,
    problem_spec: Any,
    infer_cfg: Dict[str, Any],
    device: "torch.device",
) -> Optional[Any]:
    """Step 5 — run grid inference and return an InferenceResult.

    Parameters
    ----------
    model:
        Trained ``nn.Module``.
    problem_spec:
        A ``ProblemSpec`` (or ``None``).
    infer_cfg:
        Sub-dict with optional keys ``resolution``, ``x_range``, ``y_range``.
    device:
        Target device for inference.

    Returns
    -------
    ``InferenceResult`` or ``None`` if inference is unavailable/unsupported.
    """
    try:
        from pinneaple_inference import infer_on_grid_2d
    except ImportError:
        log.debug("pinneaple_inference not available — skipping inference.")
        return None

    if problem_spec is None:
        log.debug("No problem_spec — skipping inference.")
        return None

    resolution = infer_cfg.get("resolution", [100, 100])
    if isinstance(resolution, int):
        resolution = [resolution, resolution]

    coord_names = tuple(problem_spec.coords)
    fields = list(problem_spec.fields)
    domain_bounds: Dict[str, Any] = dict(getattr(problem_spec, "domain_bounds", {}))

    spatial_coords = [c for c in coord_names if c != "t"]

    model.eval()  # type: ignore[attr-defined]
    model.to(device)

    log.info("Running grid inference: resolution=%s  fields=%s", resolution, fields)
    t0 = time.perf_counter()

    result = None

    if len(spatial_coords) >= 2:
        c0, c1 = spatial_coords[0], spatial_coords[1]
        x_range = tuple(domain_bounds.get(c0, (0.0, 1.0)))
        y_range = tuple(domain_bounds.get(c1, (0.0, 1.0)))
        try:
            result = infer_on_grid_2d(
                model,
                x_range,
                y_range,
                nx=int(resolution[0]),
                ny=int(resolution[1]),
                device=str(device),
                field_names=fields,
                coord_names=(c0, c1),
            )
        except Exception as exc:
            log.warning("infer_on_grid_2d failed: %s", exc)
    else:
        try:
            from pinneaple_inference import infer_on_grid_1d
            c0 = spatial_coords[0] if spatial_coords else coord_names[0]
            c1 = "t" if "t" in coord_names else (coord_names[1] if len(coord_names) > 1 else "t")
            x_range = tuple(domain_bounds.get(c0, (-1.0, 1.0)))
            t_range = tuple(domain_bounds.get(c1, (0.0, 1.0)))
            result = infer_on_grid_1d(
                model,
                x_range,
                t_range,
                nx=int(resolution[0]),
                nt=int(resolution[1]),
                device=str(device),
                field_names=fields,
                coord_names=(c0, c1),
            )
        except Exception as exc:
            log.warning("infer_on_grid_1d failed: %s", exc)

    log.info("Inference completed in %.2fs", time.perf_counter() - t0)
    return result


def _step_metrics(
    model: Any,
    val_batch: Dict[str, Any],
    device: "torch.device",
    metric_names: List[str],
) -> Dict[str, float]:
    """Step 6 — compute scalar evaluation metrics on the validation set.

    Parameters
    ----------
    model:
        Trained ``nn.Module``.
    val_batch:
        Numpy-valued validation batch dict.
    device:
        Target device.
    metric_names:
        List of metric identifiers (e.g. ``["mse", "rmse", "rel_l2"]``).

    Returns
    -------
    Dict of metric_name → float value.
    """
    result: Dict[str, float] = {}

    torch_val = _numpy_batch_to_torch(val_batch, device)
    x_data = torch_val.get("x_data") or torch_val.get("x")
    y_data = torch_val.get("y_data")

    if x_data is None or y_data is None or x_data.numel() == 0:
        log.debug("Skipping metrics — no labelled data in val_batch.")
        return result

    # Try the project metric framework
    metrics_obj = None
    try:
        from pinneaple_train.metrics_cfg import build_metrics_from_cfg
        metrics_obj = build_metrics_from_cfg(metric_names)
    except Exception:
        pass

    model.eval()
    with torch.no_grad():
        y_hat = model(x_data)
        if not isinstance(y_hat, torch.Tensor):
            for attr in ("y", "pred", "out"):
                if hasattr(y_hat, attr):
                    y_hat = getattr(y_hat, attr)
                    break

        if y_hat.shape[-1] != y_data.shape[-1]:
            min_dim = min(y_hat.shape[-1], y_data.shape[-1])
            y_hat = y_hat[..., :min_dim]
            y_data = y_data[..., :min_dim]

        if metrics_obj is not None:
            try:
                result = metrics_obj.compute(y_hat.cpu(), y_data.cpu())
                log.info("Metrics: %s", result)
                return result
            except Exception as exc:
                log.warning("metrics_obj.compute failed: %s — using manual fallback.", exc)

        # Manual fallback metrics
        diff = y_hat.cpu().float() - y_data.cpu().float()
        mse_val = float(torch.mean(diff ** 2))
        result["mse"] = mse_val

        if "rmse" in metric_names:
            result["rmse"] = float(mse_val ** 0.5)

        if "rel_l2" in metric_names:
            denom = float(torch.mean(y_data.cpu().float() ** 2)) ** 0.5
            result["rel_l2"] = float((mse_val ** 0.5) / (denom + 1e-12))

        if "r2" in metric_names:
            ss_res = float(torch.sum(diff ** 2))
            ss_tot = float(torch.sum((y_data.cpu().float() - y_data.cpu().float().mean()) ** 2))
            result["r2"] = 1.0 - ss_res / (ss_tot + 1e-12)

        if "max_error" in metric_names:
            result["max_error"] = float(torch.max(torch.abs(diff)))

    log.info("Metrics: %s", result)
    return result


def _step_report(
    all_results: Dict[str, Any],
    cfg: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, str]:
    """Step 7 — generate plots, field comparisons, and a summary report.

    Parameters
    ----------
    all_results:
        Aggregated results keyed by model id, produced by the training loop.
    cfg:
        Full pipeline config dict (for report sub-section settings).
    out_dir:
        Root output directory; plots go in ``<out_dir>/plots``.

    Returns
    -------
    Dict of description → file path for all generated artefacts.
    """
    report_cfg: Dict[str, Any] = dict(cfg.get("report", {}))
    plot_cfgs: List[Dict[str, Any]] = list(report_cfg.get("plots", []))
    fmt = str(report_cfg.get("format", "json")).lower()
    save_model_flag = bool(report_cfg.get("save_model", True))

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    artefacts: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Visualizations via pinneaple_inference
    # ------------------------------------------------------------------
    if plot_cfgs:
        try:
            from pinneaple_inference.visualize import render_visualizations
            viz = render_visualizations(
                plot_cfgs,
                model_results=all_results,
                out_dir=plots_dir,
            )
            artefacts.update(viz)
            log.info("Rendered %d visualizations to %s", len(viz), plots_dir)
        except ImportError:
            log.debug("pinneaple_inference.visualize not available — skipping plots.")
        except Exception as exc:
            log.warning("render_visualizations failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Fallback: matplotlib loss curves
    # ------------------------------------------------------------------
    if not artefacts:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for model_id, mdata in all_results.items():
                history = (mdata.get("history", {}) or {}) if isinstance(mdata, dict) else {}
                train_losses = history.get("train_losses") or history.get("loss_history") or []
                val_losses = history.get("val_losses") or []

                if train_losses:
                    fig, ax = plt.subplots(figsize=(7, 4))
                    ax.semilogy(train_losses, label="train")
                    if val_losses:
                        ax.semilogy(val_losses, label="val")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title(f"Loss curve — {model_id}")
                    ax.legend()
                    fig.tight_layout()
                    out_path = plots_dir / f"{model_id}_loss_curve.png"
                    fig.savefig(out_path, dpi=120)
                    plt.close(fig)
                    artefacts[f"{model_id}_loss_curve"] = str(out_path)
                    log.info("Saved loss curve: %s", out_path)

        except ImportError:
            log.debug("matplotlib not available — skipping loss curves.")
        except Exception as exc:
            log.warning("Loss curve generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Summary JSON (always written)
    # ------------------------------------------------------------------
    summary: Dict[str, Any] = {
        "pipeline_name": cfg.get("pipeline", {}).get("name", "pipeline"),
        "models": {},
        "artefacts": artefacts,
    }
    for model_id, mdata in all_results.items():
        if isinstance(mdata, dict):
            summary["models"][model_id] = {
                "metrics": mdata.get("metrics", {}),
                "elapsed_sec": mdata.get("elapsed_sec"),
                "best_val": (mdata.get("history") or {}).get("best_val"),
            }

    summary_path = out_dir / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    artefacts["pipeline_summary"] = str(summary_path)
    log.info("Pipeline summary written to %s", summary_path)

    # ------------------------------------------------------------------
    # HTML report
    # ------------------------------------------------------------------
    if fmt in ("html", "both"):
        html_path = _write_html_report(summary, artefacts, out_dir)
        if html_path:
            artefacts["html_report"] = str(html_path)

    return artefacts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_model_instance(
    model_type: str,
    model_params: Dict[str, Any],
    in_dim: int,
    out_dim: int,
) -> "nn.Module":
    """Build a model — tries PINNFactory, then ModelRegistry, then built-in MLP."""
    # 1. PINNFactory
    try:
        from pinneaple_pinn.factory import PINNFactory
        model = PINNFactory.build(model_type, in_dim=in_dim, out_dim=out_dim, **model_params)
        log.debug("Model built via PINNFactory: %s", model_type)
        return model
    except ImportError:
        log.debug("pinneaple_pinn.factory not available.")
    except Exception as exc:
        log.debug("PINNFactory.build failed for '%s': %s", model_type, exc)

    # 2. ModelRegistry
    try:
        from pinneaple_models.registry import ModelRegistry
        from pinneaple_models.register_all import register_all
        register_all()
        model = ModelRegistry.build(model_type, in_dim=in_dim, out_dim=out_dim, **model_params)
        log.debug("Model built via ModelRegistry: %s", model_type)
        return model
    except ImportError:
        log.debug("pinneaple_models.registry not available.")
    except Exception as exc:
        log.debug("ModelRegistry.build failed for '%s': %s", model_type, exc)

    # 3. Fallback MLP
    log.info(
        "Using built-in MLP fallback for model type '%s' (in=%d, out=%d).",
        model_type, in_dim, out_dim,
    )
    hidden: List[int] = list(model_params.get("hidden", [64, 64, 64, 64]))
    activation: str = str(model_params.get("activation", "tanh")).lower()
    act_map = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "sigmoid": nn.Sigmoid,
    }
    act_cls = act_map.get(activation, nn.Tanh)
    dims = [in_dim] + hidden + [out_dim]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_cls())
    return nn.Sequential(*layers)


def _build_loss_fn(
    problem_spec: Any,
    physics_weights: Dict[str, Any],
):
    """Build a physics-informed + supervised loss function."""
    # Try the project loss builder
    try:
        from pinneaple_train.losses import build_loss
        # Attempt to compile PDE residuals
        physics_loss_fn = None
        try:
            from pinneaple_pinn.compiler import compile_problem
            physics_loss_fn = compile_problem(
                problem_spec, weights=physics_weights if physics_weights else None
            )
        except Exception:
            pass

        supports_physics = physics_loss_fn is not None
        loss_obj = build_loss(
            problem_spec=problem_spec,
            model_capabilities={"supports_physics_loss": supports_physics},
            weights=physics_weights if physics_weights else None,
            supervised_kind="mse",
            physics_loss_fn=physics_loss_fn,
        )

        def _project_loss_fn(model: nn.Module, y_hat: Any, batch: Dict[str, Any]):
            return loss_obj(model, y_hat, batch)

        return _project_loss_fn
    except ImportError:
        pass
    except Exception as exc:
        log.debug("build_loss failed: %s — using basic PINN fallback.", exc)

    # Fallback: basic PINN loss
    pde_w = float(physics_weights.get("pde", 1.0))
    bc_w = float(physics_weights.get("bc", 10.0))
    data_w = float(physics_weights.get("data", 5.0))

    def _basic_pinn_loss(
        model: nn.Module,
        y_hat: Any,
        batch: Dict[str, Any],
    ) -> Dict[str, "torch.Tensor"]:
        losses: Dict[str, torch.Tensor] = {}
        dev = next(model.parameters()).device

        def _get_pred(x: torch.Tensor) -> torch.Tensor:
            pred = model(x)
            if not isinstance(pred, torch.Tensor):
                for attr in ("y", "pred", "out"):
                    if hasattr(pred, attr):
                        return getattr(pred, attr)
            return pred  # type: ignore[return-value]

        def _safe_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            if a.shape[-1] != b.shape[-1]:
                m = min(a.shape[-1], b.shape[-1])
                a, b = a[..., :m], b[..., :m]
            return torch.mean((a - b) ** 2)

        # Supervised / data loss
        x_data = batch.get("x_data")
        y_data = batch.get("y_data")
        if x_data is not None and y_data is not None and x_data.numel() > 0:
            losses["data"] = data_w * _safe_mse(_get_pred(x_data), y_data)

        # BC / IC losses
        for x_key, y_key, w in [
            ("x_bc", "y_bc", bc_w),
            ("x_ic", "y_ic", bc_w),
        ]:
            x_c = batch.get(x_key)
            y_c = batch.get(y_key)
            if x_c is not None and y_c is not None and x_c.numel() > 0:
                losses[x_key] = w * _safe_mse(_get_pred(x_c), y_c)

        if not losses:
            losses["supervised"] = torch.tensor(0.0, device=dev, requires_grad=True)

        total = sum(losses.values())
        losses["total"] = total
        return losses

    return _basic_pinn_loss


def _write_html_report(
    summary: Dict[str, Any],
    artefacts: Dict[str, str],
    out_dir: Path,
) -> Optional[Path]:
    """Write a minimal HTML summary report and return its path."""
    rows = ""
    for model_id, mdata in summary.get("models", {}).items():
        metrics = mdata.get("metrics") or {}
        metrics_str = ", ".join(f"{k}={v:.4g}" for k, v in metrics.items() if isinstance(v, (int, float)))
        elapsed = mdata.get("elapsed_sec", "")
        best_val = mdata.get("best_val", "")
        rows += (
            f"<tr><td>{model_id}</td><td>{metrics_str or 'n/a'}</td>"
            f"<td>{best_val}</td><td>{elapsed}</td></tr>\n"
        )

    img_tags = ""
    for desc, path in artefacts.items():
        if str(path).lower().endswith((".png", ".jpg", ".jpeg", ".svg")):
            rel = Path(path).relative_to(out_dir) if Path(path).is_relative_to(out_dir) else path
            img_tags += f'<figure><figcaption>{desc}</figcaption><img src="{rel}" style="max-width:800px"></figure>\n'

    pipeline_name = summary.get("pipeline_name", "Pipeline")
    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{pipeline_name} — Report</title>
<style>body{{font-family:sans-serif;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:.4rem .8rem}}th{{background:#f0f0f0}}</style>
</head>
<body>
<h1>{pipeline_name}</h1>
<h2>Model Results</h2>
<table>
<tr><th>Model</th><th>Metrics</th><th>Best Val Loss</th><th>Elapsed (s)</th></tr>
{rows}
</table>
<h2>Plots</h2>
{img_tags or '<p>No plots generated.</p>'}
</body></html>
"""
    html_path = out_dir / "report.html"
    try:
        with open(html_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        log.info("HTML report written to %s", html_path)
        return html_path
    except Exception as exc:
        log.warning("Could not write HTML report: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_full_pipeline(
    config_path_or_dict: Union[str, Path, Dict[str, Any]],
    *,
    out_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Run the complete end-to-end pipeline.

    Stages
    ------
    1. Geometry — build or load domain geometry / mesh.
    2. Solver — run a physical solver to generate ground-truth fields.
    3. Dataset — sample collocation/BC/data points for PINN training.
    4. Train — train each surrogate model defined in ``config["models"]``.
    5. Inference — evaluate each model on a regular grid.
    6. Metrics — compute scalar error metrics against labelled data.
    7. Report — render error plots and write HTML/JSON summary.

    Parameters
    ----------
    config_path_or_dict:
        Either a path to a YAML file (str or :class:`pathlib.Path`) or a
        pre-built config dict following the schema described in the module
        docstring.
    out_dir:
        Root output directory.  Overrides ``pipeline.out_dir`` in the config
        when supplied.

    Returns
    -------
    Summary dict with keys:

    - ``"pipeline_name"`` – str
    - ``"out_dir"`` – str
    - ``"geometry"`` – geometry object or None
    - ``"solver_data"`` – dict of field arrays (may be empty)
    - ``"models"`` – dict of model_id → result dict
    - ``"artefacts"`` – dict of description → file path
    - ``"timing"`` – dict of step → elapsed seconds
    - ``"success"`` – bool
    """
    _configure_logging()

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    cfg = _load_config(config_path_or_dict)

    pipeline_cfg: Dict[str, Any] = dict(cfg.get("pipeline", {}))
    pipeline_name = str(pipeline_cfg.get("name", "pipeline"))

    if out_dir is None:
        out_dir = Path(str(pipeline_cfg.get("out_dir", f"results/{pipeline_name}")))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("========== Pipeline: %s ==========", pipeline_name)
    log.info("Output directory: %s", out_dir)

    timing: Dict[str, float] = {}
    all_model_results: Dict[str, Dict[str, Any]] = {}
    success = True

    # ------------------------------------------------------------------
    # Load problem spec
    # ------------------------------------------------------------------
    problem_spec = None
    problem_cfg: Dict[str, Any] = dict(cfg.get("problem", {}))
    if problem_cfg:
        try:
            from pinneaple_environment.presets.registry import get_preset
            problem_id = str(problem_cfg.get("id", ""))
            params = dict(problem_cfg.get("params", {}))
            if problem_id:
                problem_spec = get_preset(problem_id, **params)
                log.info("Loaded problem spec: %s", problem_id)
        except ImportError:
            log.debug("pinneaple_environment not available — problem_spec will be None.")
        except Exception as exc:
            log.warning("Could not load problem spec: %s", exc)

    # ------------------------------------------------------------------
    # Step 1: Geometry
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    geo_cfg: Dict[str, Any] = dict(cfg.get("geometry", {}))
    geometry = None
    try:
        geometry = _step_geometry(geo_cfg)
    except Exception as exc:
        log.error("Geometry step failed: %s", exc, exc_info=True)
        success = False
    timing["geometry"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Step 2: Solver
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    solver_cfg: Dict[str, Any] = dict(cfg.get("solver", {}))
    solver_data: Dict[str, np.ndarray] = {}
    try:
        solver_data = _step_solver(solver_cfg, problem_spec, geometry, out_dir)
    except Exception as exc:
        log.error("Solver step failed: %s", exc, exc_info=True)
        success = False
    timing["solver"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Step 3: Dataset
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    dataset_cfg: Dict[str, Any] = dict(cfg.get("dataset", {}))
    train_batch: Dict[str, Any] = {}
    val_batch: Dict[str, Any] = {}
    try:
        train_batch, val_batch = _step_dataset(dataset_cfg, problem_spec, solver_data, geometry)
    except Exception as exc:
        log.error("Dataset step failed: %s", exc, exc_info=True)
        success = False
    timing["dataset"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Steps 4–6: Per-model train → infer → metrics
    # ------------------------------------------------------------------
    models_cfg: List[Dict[str, Any]] = list(cfg.get("models", []))
    metric_names: List[str] = list(cfg.get("metrics", ["mse", "rmse", "rel_l2", "r2", "max_error"]))
    infer_cfg: Dict[str, Any] = dict(cfg.get("inference", {}))

    model_iter = models_cfg
    if _HAS_TQDM:
        model_iter = _tqdm(models_cfg, desc="Models", unit="model")

    for model_entry in model_iter:
        model_id = str(model_entry.get("id", f"model_{len(all_model_results)}"))
        log.info("--- Model: %s ---", model_id)

        model_result: Dict[str, Any] = {
            "model_id": model_id,
            "history": {},
            "metrics": {},
            "inference_result": None,
            "elapsed_sec": 0.0,
            "error": None,
        }

        # Train
        t0 = time.perf_counter()
        trained_model = None
        try:
            if not _HAS_TORCH:
                raise RuntimeError("PyTorch is not installed.")
            trained_model, train_info = _step_train_model(
                model_entry,
                problem_spec,
                train_batch,
                val_batch,
                out_dir,
            )
            model_result["history"] = train_info.get("history", {})
            model_result["elapsed_sec"] = train_info.get("elapsed_sec", 0.0)
        except Exception as exc:
            log.error("Training failed for model '%s': %s", model_id, exc, exc_info=True)
            model_result["error"] = str(exc)
            success = False

        timing[f"train_{model_id}"] = time.perf_counter() - t0

        if trained_model is None:
            all_model_results[model_id] = model_result
            continue

        device_str = str(model_entry.get("train", {}).get("device", "cpu"))
        device = torch.device(device_str)

        # Inference
        t0 = time.perf_counter()
        try:
            resolution = infer_cfg.get(
                "resolution",
                dataset_cfg.get("resolution", [100, 100]),
            )
            infer_model_cfg = {**infer_cfg, "resolution": resolution}
            inference_result = _step_inference(trained_model, problem_spec, infer_model_cfg, device)
            model_result["inference_result"] = inference_result
            if inference_result is not None:
                try:
                    inference_result.model_id = model_id  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception as exc:
            log.warning("Inference failed for model '%s': %s", model_id, exc)
        timing[f"infer_{model_id}"] = time.perf_counter() - t0

        # Metrics
        t0 = time.perf_counter()
        try:
            metrics = _step_metrics(trained_model, val_batch, device, metric_names)
            model_result["metrics"] = metrics
        except Exception as exc:
            log.warning("Metrics computation failed for model '%s': %s", model_id, exc)
        timing[f"metrics_{model_id}"] = time.perf_counter() - t0

        all_model_results[model_id] = model_result

    # ------------------------------------------------------------------
    # Step 7: Report
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    artefacts: Dict[str, str] = {}
    try:
        artefacts = _step_report(all_model_results, cfg, out_dir)
    except Exception as exc:
        log.error("Report step failed: %s", exc, exc_info=True)
        success = False
    timing["report"] = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = sum(timing.values())
    log.info(
        "========== Pipeline '%s' finished in %.1fs (success=%s) ==========",
        pipeline_name,
        total_elapsed,
        success,
    )
    for step, elapsed in timing.items():
        log.info("  %-30s %.2fs", step, elapsed)

    return {
        "pipeline_name": pipeline_name,
        "out_dir": str(out_dir),
        "geometry": geometry,
        "solver_data": solver_data,
        "models": all_model_results,
        "artefacts": artefacts,
        "timing": timing,
        "success": success,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="pinneaple end-to-end pipeline: geometry → solver → dataset → train → predict → report"
    )
    ap.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the pipeline YAML config file.",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory (overrides pipeline.out_dir in the config).",
    )
    ap.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    args = ap.parse_args()

    _configure_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))

    result = run_full_pipeline(
        args.config,
        out_dir=args.out_dir or None,
    )

    print(json.dumps(
        {
            "pipeline_name": result["pipeline_name"],
            "out_dir": result["out_dir"],
            "success": result["success"],
            "timing": result["timing"],
            "models": {
                mid: {
                    "metrics": mdata.get("metrics"),
                    "elapsed_sec": mdata.get("elapsed_sec"),
                }
                for mid, mdata in result["models"].items()
            },
        },
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
