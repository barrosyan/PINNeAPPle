"""pinneapple_tools — Visualization, model export, HPO, benchmarking, and compute backends.

Sub-modules
-----------
visualization     (was pinneapple_viz)
    CFD-style visualization: scalar/vector fields, streamlines, PINN loss
    histories, vorticity, Q-criterion, mesh plots, animations.

model_export      (was pinneapple_export)
    Model deployment: TorchScript, ONNX, CSV, NPZ export.

hpo_experiments   (was pinneapple_researcher)
    Literature pipeline: discover papers, build knowledge base,
    extract problem-solution pairs, reproduce benchmark results.

benchmark_suite   (was pinneapple_arena)
    Benchmark suite: Arena, PINNArenaBenchmark, TransferBenchmarkPipeline,
    MetaBenchmarkPipeline; task/backend registries.

compute_backends  (was pinneapple_backend)
    Multi-backend abstraction: PyTorch (default) and JAX (jit_pinn,
    vmap_residual, torch↔jax conversion).

Integration helpers
-------------------
``plot(model, domain, ...)``
    Quick visualization shortcut that auto-selects 1D or 2D plot.
``export_model(model, path, fmt, ...)``
    Format-dispatching export shortcut.
``run_benchmark(models, tasks, ...)``
    PINNArenaBenchmark shortcut for quick model comparison.

Usage
-----
>>> from pinneapple_tools import plot, export_model, run_benchmark
>>> plot(model, x_test, field_name="u")
>>> export_model(model, "model.onnx", fmt="onnx", example_input=x)
>>> results = run_benchmark({"my_pinn": model}, tasks=["burgers_1d"])
"""
from __future__ import annotations

# ── sub-modules (new descriptive names) ───────────────────────────────────────
from . import visualization
from . import model_export
from . import hpo_experiments
from . import benchmark_suite
from . import compute_backends

# backward-compat aliases
viz        = visualization
export     = model_export
researcher = hpo_experiments
arena      = benchmark_suite
backend    = compute_backends

# ── visualization re-exports ──────────────────────────────────────────────────
from .visualization import (
    use_cfd_style, get_cmap, make_figure, CMAPS, DEFAULT_CMAP,
    plot_scalar, plot_scalar_3d, plot_vectors, plot_streamlines,
    compare_fields, plot_error,
    plot_loss_history, plot_multi_loss, plot_collocation,
    plot_pinn_prediction, plot_pde_residual, plot_gradient_magnitude,
    predict_and_plot,
    plot_solver_output, plot_fem_result, plot_fvm_result, plot_residuals,
    plot_mesh, plot_boundary, plot_point_cloud,
    plot_voxel_slice, plot_voxel_3d, plot_voxel_histogram,
    animate_scalar_field, animate_streamlines, make_gif,
    compute_vorticity_2d, compute_q_criterion_2d,
    compute_q_criterion_3d, compute_lambda2_3d,
    plot_vorticity, plot_q_criterion_2d, plot_q_criterion_3d,
    plot_vortex_identification, plot_lbm_flow, plot_flow_panel,
)

# ── model_export re-exports ───────────────────────────────────────────────────
from .model_export import (
    export_torchscript,
    export_onnx,
    export_csv,
    export_npz,
)

# ── hpo_experiments re-exports ────────────────────────────────────────────────
from .hpo_experiments import (
    ResearcherConfig,
    discover,
    build_kb,
    extract_problem_solutions,
    reproduce,
)

# ── benchmark_suite re-exports ────────────────────────────────────────────────
from .benchmark_suite import (
    Arena, ArenaResult, ArenaCompareResult,
    run_arena_experiment,
    BenchmarkReport, ModelRunResult,
    TASK_REGISTRY, BACKEND_REGISTRY,
    register_task, register_backend,
    get_task, get_backend, list_tasks, list_backends,
    PINNArenaBenchmark, BenchmarkConfig, BenchmarkResult,
    BenchmarkTaskBase, ModelSpec, DEFAULT_MODELS,
)

try:
    from .benchmark_suite import (
        TransferBenchmarkPipeline, TransferBenchmarkConfig,
        TransferBenchmarkResult, TransferScenario,
    )
except Exception:
    pass

try:
    from .benchmark_suite import (
        MetaBenchmarkPipeline, MetaBenchmarkConfig,
        MetaBenchmarkResult, MetaBenchmarkFamily,
    )
except Exception:
    pass

try:
    from .benchmark_suite import run_full_pipeline
except Exception:
    pass

# ── compute_backends re-exports ───────────────────────────────────────────────
from .compute_backends import (
    Backend, get_backend as get_compute_backend, set_backend,
    JAXBackend, jax_available, jax_pinn, jit_pinn, vmap_residual,
)


# ── Integration helpers ────────────────────────────────────────────────────────

def plot(model, x, *, field_name: str = "u", dim: int = None, **kwargs):
    """Quick visualization of a model prediction over points ``x``."""
    import torch
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    d = dim or x.shape[-1]
    with torch.no_grad():
        y = model(x).cpu().numpy()
    x_np = x.cpu().numpy()
    if d == 1:
        return plot_scalar(x_np[:, 0], y[:, 0], xlabel="x", ylabel=field_name, **kwargs)
    return plot_scalar(x_np, y[:, 0], field_name=field_name, **kwargs)


def export_model(model, path: str, fmt: str = "torchscript", **kwargs) -> str:
    """Export a trained model to the requested format."""
    _DISPATCH = {
        "torchscript": export_torchscript,
        "onnx":        export_onnx,
        "csv":         export_csv,
        "npz":         export_npz,
    }
    fn = _DISPATCH.get(fmt)
    if fn is None:
        raise ValueError(f"Unknown export format '{fmt}'. Choose from: {list(_DISPATCH)}")
    return fn(model, path, **kwargs)


def run_benchmark(models: dict, tasks: list | None = None, **bench_kwargs) -> "BenchmarkResult":
    """Run a quick benchmark comparison across models."""
    cfg = BenchmarkConfig(tasks=tasks, **bench_kwargs)
    bench = PINNArenaBenchmark(cfg)
    return bench.run(models)


__all__ = [
    # Sub-modules (new names)
    "visualization", "model_export", "hpo_experiments", "benchmark_suite", "compute_backends",
    # Sub-modules (old aliases — backward compat)
    "viz", "export", "researcher", "arena", "backend",
    # Integration
    "plot", "export_model", "run_benchmark",
    # visualization
    "use_cfd_style", "get_cmap", "make_figure", "CMAPS", "DEFAULT_CMAP",
    "plot_scalar", "plot_scalar_3d", "plot_vectors", "plot_streamlines",
    "compare_fields", "plot_error",
    "plot_loss_history", "plot_multi_loss", "plot_collocation",
    "plot_pinn_prediction", "plot_pde_residual", "plot_gradient_magnitude",
    "predict_and_plot",
    "plot_solver_output", "plot_fem_result", "plot_fvm_result", "plot_residuals",
    "plot_mesh", "plot_boundary", "plot_point_cloud",
    "plot_voxel_slice", "plot_voxel_3d", "plot_voxel_histogram",
    "animate_scalar_field", "animate_streamlines", "make_gif",
    "compute_vorticity_2d", "compute_q_criterion_2d",
    "compute_q_criterion_3d", "compute_lambda2_3d",
    "plot_vorticity", "plot_q_criterion_2d", "plot_q_criterion_3d",
    "plot_vortex_identification", "plot_lbm_flow", "plot_flow_panel",
    # model_export
    "export_torchscript", "export_onnx", "export_csv", "export_npz",
    # hpo_experiments
    "ResearcherConfig", "discover", "build_kb",
    "extract_problem_solutions", "reproduce",
    # benchmark_suite
    "Arena", "ArenaResult", "ArenaCompareResult",
    "run_arena_experiment",
    "BenchmarkReport", "ModelRunResult",
    "TASK_REGISTRY", "BACKEND_REGISTRY",
    "register_task", "register_backend",
    "get_task", "get_backend", "list_tasks", "list_backends",
    "PINNArenaBenchmark", "BenchmarkConfig", "BenchmarkResult",
    "BenchmarkTaskBase", "ModelSpec", "DEFAULT_MODELS",
    # compute_backends
    "Backend", "get_compute_backend", "set_backend",
    "JAXBackend", "jax_available", "jax_pinn", "jit_pinn", "vmap_residual",
]
