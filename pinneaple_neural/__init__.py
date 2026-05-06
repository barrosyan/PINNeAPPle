"""pinneaple_neural — Neural network architectures, training, and inference.

Sub-modules
-----------
architectures  (was pinneaple_models)
    Neural network architectures and registry: SIREN, ModifiedMLP, HashGridMLP,
    AFNO, MeshGraphNet, ModelRegistry, ModelCatalog.

trainer        (was pinneaple_train)
    Full training infrastructure: Trainer, TwoPhaseTrainer, TimeMarchingTrainer,
    DDPPINNTrainer, CausalPINNTrainer, loss combiners, weight schedulers,
    normalizers, data splitters, HPC utilities (FSDP, SLURM, profiler).

predictor      (was pinneaple_inference)
    Post-training inference: batched_inference, grid evaluation (1-D / 2-D),
    FlowVisualizer, isosurface, streamlines, design-opt plots.

Integration helpers
-------------------
``build_model(name, **kwargs)``      — ModelRegistry.build shortcut
``train_model(model, losses, ...)``  — Trainer shortcut
``predict(model, x, ...)``           — batched_inference shortcut
``train_and_predict(...)``           — full pipeline in one call

Usage
-----
>>> from pinneaple_neural import build_model, train_model, predict
>>> model = build_model("SIREN", in_dim=2, out_dim=1, hidden_dim=64, n_layers=4)
>>> result = train_model(model, losses, epochs=5000)
>>> pred = predict(model, x_test)
"""
from __future__ import annotations

# ── sub-modules (new descriptive names) ───────────────────────────────────────
from . import architectures
from . import trainer
from . import predictor

# backward-compat aliases (old names still work)
models    = architectures
train     = trainer
inference = predictor

# ── architectures re-exports ──────────────────────────────────────────────────
from .architectures import (
    BaseModel,
    ModelOutput,
    ModelRegistry,
    ModelSpec,
    InstantiateReport,
    ModelCatalog,
    select_adapter,
    SIREN,
    SineLayer,
    ModifiedMLP,
    FourierFeatureEmbedding,
    HashGridMLP,
    HashGridEncoding,
    MeshGraphNet,
    AFNO,
    AFNOLayer,
    GroupBCatalog,
)

# ── trainer re-exports ────────────────────────────────────────────────────────
from .trainer import (
    # Trainers
    TwoPhaseConfig, TwoPhaseHistory, TwoPhaseTrainer, UnnormModel,
    TimeMarchingTrainer,
    DDPTrainerConfig, DDPPINNTrainer, is_distributed, get_rank, get_world_size,
    CausalWeightScheduler, CausalPINNTrainer,
    Trainer, TrainConfig,
    # HPC
    FSDPConfig, wrap_fsdp,
    wrap_zero_optimizer,
    CUDAGraphModule,
    register_powersgd_hook, register_topk_hook,
    TorchRunConfig, SLURMConfig,
    build_torchrun_cmd, build_slurm_script,
    ProfilerConfig, PINNeAPPleProfiler,
    AutoBatchSizeFinder,
    # Data utilities
    SplitSpec, split_indices,
    Normalizer, StandardScaler, MinMaxScaler,
    PreprocessPipeline, SolverFeatureStep,
    DataModule, ItemAdapter, FnAdapter, AdaptedSequenceDataset,
    # Losses & weights
    CombinedLoss, SupervisedLoss, PhysicsLossHook,
    WeightScheduler, WeightSchedulerConfig,
    SelfAdaptiveWeights, GradNormBalancer, LossRatioBalancer, NTKWeightBalancer,
    # Metrics & callbacks
    Metrics, RegressionMetrics, default_metrics, build_metrics_from_cfg,
    EarlyStopping, ModelCheckpoint,
    # Audit & parallelism
    RunLogger, set_seed, set_deterministic,
    best_device, count_gpus, gpu_info, maybe_compile,
    AMPContext, wrap_data_parallel, unwrap_model,
    CUDAPrefetcher, GradAccumConfig, GradAccumTrainer,
    SweepConfig, run_parallel_sweep,
    batched_inference as train_batched_inference,
    enable_gradient_checkpointing, ThroughputMonitor,
)

# ── predictor re-exports ──────────────────────────────────────────────────────
from .predictor import (
    infer,
    infer_on_grid_1d,
    infer_on_grid_2d,
    InferenceResult,
    batched_inference,
    plot_field_1d,
    plot_field_2d,
    plot_error_map_1d,
    plot_error_map_2d,
    plot_loss_curve,
    plot_model_comparison_1d,
    plot_model_comparison_2d,
    render_visualizations,
    plot_velocity_slice,
    plot_velocity_magnitude_slice,
    plot_streamlines_2d,
    plot_centerline_velocity,
    plot_vorticity_slice,
    plot_internal_flow_summary,
    plot_design_opt_convergence,
    plot_pareto_front_2d,
    compute_streamlines,
    compute_isosurface,
    plot_streamlines_2d_from_model,
    plot_isosurface_3d,
    plot_volume_slice,
    FlowVisualizer,
)


# ── Integration helpers ────────────────────────────────────────────────────────

def build_model(name: str, **kwargs):
    """Build a model from the registry by name."""
    return ModelRegistry.build(name, **kwargs)


def train_model(model, losses, *, epochs: int = 5000, device: str = "cpu", **cfg_kwargs):
    """Train a model with the given losses."""
    cfg = TrainConfig(n_epochs=epochs, device=device, **cfg_kwargs)
    t = Trainer(model, losses, cfg)
    return t.train()


def predict(model, x, *, device: str = "cpu", batch_size: int = 4096):
    """Run batched inference on input ``x``."""
    import torch
    dev = torch.device(device)
    model = model.to(dev)
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x.to(dev)
    return batched_inference(model, x, batch_size=batch_size)


def train_and_predict(model, losses, x_test, *, epochs: int = 5000, device: str = "cpu"):
    """Convenience: train then evaluate on test points."""
    train_result = train_model(model, losses, epochs=epochs, device=device)
    preds = predict(model, x_test, device=device)
    return {"train_result": train_result, "predictions": preds}


__all__ = [
    # Sub-modules (new names)
    "architectures", "trainer", "predictor",
    # Sub-modules (old aliases — backward compat)
    "models", "train", "inference",
    # Integration
    "build_model", "train_model", "predict", "train_and_predict",
    # architectures
    "BaseModel", "ModelOutput", "ModelRegistry", "ModelSpec",
    "InstantiateReport", "ModelCatalog", "select_adapter",
    "SIREN", "SineLayer", "ModifiedMLP", "FourierFeatureEmbedding",
    "HashGridMLP", "HashGridEncoding", "MeshGraphNet",
    "AFNO", "AFNOLayer", "GroupBCatalog",
    # trainer
    "TwoPhaseConfig", "TwoPhaseHistory", "TwoPhaseTrainer", "UnnormModel",
    "TimeMarchingTrainer",
    "DDPTrainerConfig", "DDPPINNTrainer", "is_distributed", "get_rank", "get_world_size",
    "CausalWeightScheduler", "CausalPINNTrainer",
    "Trainer", "TrainConfig",
    "FSDPConfig", "wrap_fsdp", "wrap_zero_optimizer", "CUDAGraphModule",
    "register_powersgd_hook", "register_topk_hook",
    "TorchRunConfig", "SLURMConfig", "build_torchrun_cmd", "build_slurm_script",
    "ProfilerConfig", "PINNeAPPleProfiler", "AutoBatchSizeFinder",
    "SplitSpec", "split_indices",
    "Normalizer", "StandardScaler", "MinMaxScaler",
    "PreprocessPipeline", "SolverFeatureStep",
    "DataModule", "ItemAdapter", "FnAdapter", "AdaptedSequenceDataset",
    "CombinedLoss", "SupervisedLoss", "PhysicsLossHook",
    "WeightScheduler", "WeightSchedulerConfig",
    "SelfAdaptiveWeights", "GradNormBalancer", "LossRatioBalancer", "NTKWeightBalancer",
    "Metrics", "RegressionMetrics", "default_metrics", "build_metrics_from_cfg",
    "EarlyStopping", "ModelCheckpoint",
    "RunLogger", "set_seed", "set_deterministic",
    "best_device", "count_gpus", "gpu_info", "maybe_compile",
    "AMPContext", "wrap_data_parallel", "unwrap_model",
    "CUDAPrefetcher", "GradAccumConfig", "GradAccumTrainer",
    "SweepConfig", "run_parallel_sweep",
    "train_batched_inference", "enable_gradient_checkpointing", "ThroughputMonitor",
    # predictor
    "infer", "infer_on_grid_1d", "infer_on_grid_2d",
    "InferenceResult", "batched_inference",
    "plot_field_1d", "plot_field_2d", "plot_error_map_1d", "plot_error_map_2d",
    "plot_loss_curve", "plot_model_comparison_1d", "plot_model_comparison_2d",
    "render_visualizations",
    "plot_velocity_slice", "plot_velocity_magnitude_slice",
    "plot_streamlines_2d", "plot_centerline_velocity",
    "plot_vorticity_slice", "plot_internal_flow_summary",
    "plot_design_opt_convergence", "plot_pareto_front_2d",
    "compute_streamlines", "compute_isosurface",
    "plot_streamlines_2d_from_model", "plot_isosurface_3d", "plot_volume_slice",
    "FlowVisualizer",
]
