from .two_phase import TwoPhaseConfig, TwoPhaseHistory, TwoPhaseTrainer, UnnormModel
from .hpc import (
    FSDPConfig, wrap_fsdp,
    wrap_zero_optimizer,
    CUDAGraphModule,
    register_powersgd_hook, register_topk_hook,
    TorchRunConfig, SLURMConfig,
    build_torchrun_cmd, build_slurm_script,
    ProfilerConfig, PINNeAPPleProfiler,
    AutoBatchSizeFinder,
)
from .splits import SplitSpec, split_indices
from .normalizers import Normalizer, StandardScaler, MinMaxScaler
from .preprocess import PreprocessPipeline, SolverFeatureStep
from .metrics import Metrics, RegressionMetrics, default_metrics
from .losses import CombinedLoss, SupervisedLoss, PhysicsLossHook
from .callbacks import EarlyStopping, ModelCheckpoint
from .trainer import Trainer, TrainConfig
from .datamodule import DataModule, ItemAdapter, FnAdapter, AdaptedSequenceDataset
from .audit import RunLogger, set_seed, set_deterministic
from .metrics_cfg import build_metrics_from_cfg
from .weight_scheduler import (
    WeightScheduler,
    WeightSchedulerConfig,
    SelfAdaptiveWeights,
    GradNormBalancer,
    LossRatioBalancer,
    NTKWeightBalancer,
)
from .parallel import (
    best_device,
    count_gpus,
    gpu_info,
    maybe_compile,
    AMPContext,
    wrap_data_parallel,
    unwrap_model,
    CUDAPrefetcher,
    GradAccumConfig,
    GradAccumTrainer,
    SweepConfig,
    run_parallel_sweep,
    batched_inference,
    enable_gradient_checkpointing,
    ThroughputMonitor,
)

__all__ = [
    "TwoPhaseConfig", "TwoPhaseHistory", "TwoPhaseTrainer", "UnnormModel",
    # HPC
    "FSDPConfig", "wrap_fsdp",
    "wrap_zero_optimizer",
    "CUDAGraphModule",
    "register_powersgd_hook", "register_topk_hook",
    "TorchRunConfig", "SLURMConfig",
    "build_torchrun_cmd", "build_slurm_script",
    "ProfilerConfig", "PINNeAPPleProfiler",
    "AutoBatchSizeFinder",
    "SplitSpec",
    "split_indices",
    "Normalizer",
    "StandardScaler",
    "MinMaxScaler",
    "PreprocessPipeline",
    "SolverFeatureStep",
    "Metrics",
    "RegressionMetrics",
    "CombinedLoss",
    "SupervisedLoss",
    "PhysicsLossHook",
    "EarlyStopping",
    "ModelCheckpoint",
    "Trainer",
    "TrainConfig",
    "default_metrics",
    "DataModule",
    "ItemAdapter", 
    "FnAdapter", 
    "AdaptedSequenceDataset",
    "RunLogger",
    "set_seed",
    "set_deterministic",
    "build_metrics_from_cfg",
    # Weight scheduling
    "WeightScheduler",
    "WeightSchedulerConfig",
    "SelfAdaptiveWeights",
    "GradNormBalancer",
    "LossRatioBalancer",
    "NTKWeightBalancer",
    # Parallel / GPU
    "best_device",
    "count_gpus",
    "gpu_info",
    "maybe_compile",
    "AMPContext",
    "wrap_data_parallel",
    "unwrap_model",
    "CUDAPrefetcher",
    "GradAccumConfig",
    "GradAccumTrainer",
    "SweepConfig",
    "run_parallel_sweep",
    "batched_inference",
    "enable_gradient_checkpointing",
    "ThroughputMonitor",
]
