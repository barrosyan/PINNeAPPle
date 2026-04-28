"""Advanced HPC utilities for large-scale PINN training.

Extends pinneaple_train.parallel with production-grade features:

Memory-efficient parallelism:
  wrap_fsdp()               Fully Sharded Data Parallel (PyTorch FSDP)
  wrap_zero_optimizer()     ZeRO-1 optimizer state sharding

Accelerated compute:
  CUDAGraphModule           Captures CUDA graphs for static shapes
  CUDAGraphContext          Context manager for manual graph capture

Communication efficiency:
  PowerSGDCompressor        Low-rank gradient compression for DDP
  TopKGradCompressor        Top-K sparsification hook for DDP

Cluster / multi-node:
  TorchRunConfig            torchrun launch configuration
  SLURMConfig               SLURM job configuration
  build_torchrun_cmd()      Generate torchrun launch command string
  build_slurm_script()      Generate a ready-to-submit SLURM batch script

Profiling and diagnostics:
  ProfilerConfig            Configuration for torch.profiler
  PINNeAPPleProfiler        Context-manager profiler with Chrome-trace export
  AutoBatchSizeFinder       Binary-search for maximum fitting batch size

References
----------
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- ZeroRedundancyOptimizer: https://pytorch.org/docs/stable/distributed.optim.html
- PowerSGD: Vogels et al. (2019) "PowerSGD: Practical Low-Rank Gradient
  Compression for Distributed Optimization"
- torch.profiler: https://pytorch.org/docs/stable/profiler.html
"""
from __future__ import annotations

import logging
import os
import textwrap
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Fully Sharded Data Parallel (FSDP)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FSDPConfig:
    """Configuration for FSDP wrapping.

    Parameters
    ----------
    sharding_strategy : str
        "FULL_SHARD"    — ZeRO-3: shards params + gradients + optimizer
        "SHARD_GRAD_OP" — ZeRO-2: shards gradients + optimizer only
        "NO_SHARD"      — equivalent to DDP (for debugging)
    mixed_precision : str or None
        "fp16" / "bf16" — enable mixed precision in FSDP.
        None — use default (fp32 everywhere).
    min_num_params : int
        Minimum number of parameters for auto-wrapping a sub-module.
        Smaller modules are kept intact.
    activation_checkpointing : bool
        Apply activation checkpointing to wrapped sub-modules.
    cpu_offload : bool
        Offload parameters to CPU when not in use (saves GPU memory,
        adds latency).
    """
    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: Optional[str] = "bf16"
    min_num_params: int = 1_000_000
    activation_checkpointing: bool = False
    cpu_offload: bool = False


def wrap_fsdp(
    model: nn.Module,
    cfg: Optional[FSDPConfig] = None,
    *,
    process_group=None,
) -> nn.Module:
    """Wrap a model with Fully Sharded Data Parallel.

    Requires an initialised process group (e.g. via ``dist.init_process_group``).
    Falls back to the unwrapped model if FSDP is unavailable.

    Parameters
    ----------
    model : nn.Module
    cfg : FSDPConfig, optional
    process_group : optional
        Custom process group.  Defaults to the global group.

    Returns
    -------
    nn.Module
        FSDP-wrapped model (or original model on import failure).

    Example
    -------
    >>> import torch.distributed as dist
    >>> dist.init_process_group("nccl")
    >>> model = wrap_fsdp(model, FSDPConfig(sharding_strategy="FULL_SHARD"))
    """
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            ShardingStrategy,
            CPUOffload,
            MixedPrecision,
        )
        from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
        import functools
    except ImportError:
        warnings.warn("torch.distributed.fsdp not available. Returning unwrapped model.")
        return model

    cfg = cfg or FSDPConfig()

    # Sharding strategy
    _strategies = {
        "FULL_SHARD":    ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD":      ShardingStrategy.NO_SHARD,
    }
    strategy = _strategies.get(cfg.sharding_strategy.upper())
    if strategy is None:
        raise ValueError(f"Unknown sharding_strategy: {cfg.sharding_strategy!r}")

    # Mixed precision policy
    mp_policy = None
    if cfg.mixed_precision == "fp16":
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    elif cfg.mixed_precision == "bf16":
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

    # Auto-wrap policy
    auto_wrap = functools.partial(
        size_based_auto_wrap_policy,
        min_num_params=cfg.min_num_params,
    )

    # CPU offload
    cpu_offload = CPUOffload(offload_params=True) if cfg.cpu_offload else None

    fsdp_model = FSDP(
        model,
        sharding_strategy=strategy,
        mixed_precision=mp_policy,
        auto_wrap_policy=auto_wrap,
        cpu_offload=cpu_offload,
        process_group=process_group,
    )

    # Optional activation checkpointing
    if cfg.activation_checkpointing:
        try:
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                apply_activation_checkpointing,
            )
            apply_activation_checkpointing(fsdp_model)
            logger.info("FSDP: activation checkpointing applied.")
        except ImportError:
            logger.warning("Activation checkpointing via FSDP not available in this PyTorch version.")

    logger.info(
        f"FSDP wrapped: strategy={cfg.sharding_strategy}, "
        f"mp={cfg.mixed_precision}, cpu_offload={cfg.cpu_offload}"
    )
    return fsdp_model


# ──────────────────────────────────────────────────────────────────────────────
# 2. ZeRO-1: ZeroRedundancyOptimizer
# ──────────────────────────────────────────────────────────────────────────────

def wrap_zero_optimizer(
    optimizer_class: Type[torch.optim.Optimizer],
    params,
    **optimizer_kwargs,
) -> torch.optim.Optimizer:
    """Wrap an optimizer with ZeroRedundancyOptimizer (ZeRO-1 sharding).

    ZeRO-1 shards optimizer states across ranks without changing the model.
    Memory savings ~= 1/world_size for optimizer state.

    Parameters
    ----------
    optimizer_class : type
        e.g. ``torch.optim.Adam``
    params : iterable
        Model parameters.
    **optimizer_kwargs
        Passed to the wrapped optimizer (e.g. ``lr=1e-3``).

    Returns
    -------
    ZeroRedundancyOptimizer (or plain optimizer if not in distributed context)

    Example
    -------
    >>> opt = wrap_zero_optimizer(torch.optim.Adam, model.parameters(), lr=1e-3)
    """
    try:
        import torch.distributed as dist
        from torch.distributed.optim import ZeroRedundancyOptimizer

        if not dist.is_available() or not dist.is_initialized():
            logger.info("ZeRO: distributed not initialised; using plain optimizer.")
            return optimizer_class(params, **optimizer_kwargs)

        return ZeroRedundancyOptimizer(
            params,
            optimizer_class=optimizer_class,
            **optimizer_kwargs,
        )
    except ImportError:
        logger.warning("ZeroRedundancyOptimizer not available; using plain optimizer.")
        return optimizer_class(params, **optimizer_kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# 3. CUDA Graph capture
# ──────────────────────────────────────────────────────────────────────────────

class CUDAGraphModule(nn.Module):
    """Wrap a module to use CUDA graph replay for static-shape inference.

    After a warm-up phase, the forward pass is captured as a CUDA graph and
    replayed on subsequent calls.  Delivers 10-40% speedup for small batches
    where kernel launch overhead dominates.

    Constraints:
    - Input shape must be **fixed** across all calls after capture.
    - No dynamic control flow based on tensor values.
    - CPU-GPU synchronisation must not happen inside the forward pass.

    Parameters
    ----------
    module : nn.Module
    warmup_steps : int
        Number of forward passes before capturing the graph.
    device : str

    Example
    -------
    >>> model = CUDAGraphModule(pinn, warmup_steps=3, device="cuda")
    >>> y = model(x)   # captured on first call after warmup
    """

    def __init__(
        self,
        module: nn.Module,
        warmup_steps: int = 3,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDAGraphModule requires a CUDA device.")
        self.module = module.to(device)
        self.warmup_steps = int(warmup_steps)
        self.device = torch.device(device)
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_x: Optional[torch.Tensor] = None
        self._static_y: Optional[torch.Tensor] = None
        self._warmup_count = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._graph is None:
            if self._warmup_count < self.warmup_steps:
                # Warm-up: run normally to let caching allocator stabilise
                torch.cuda.synchronize()
                out = self.module(x.to(self.device))
                self._warmup_count += 1
                if hasattr(out, "y"):
                    out = out.y
                return out
            else:
                # Capture the graph
                self._static_x = x.to(self.device).clone()
                with torch.cuda.graph(torch.cuda.CUDAGraph()) as g:
                    self._static_y = self.module(self._static_x)
                    if hasattr(self._static_y, "y"):
                        self._static_y = self._static_y.y
                self._graph = g
                logger.info("CUDAGraphModule: CUDA graph captured.")

        # Replay
        self._static_x.copy_(x.to(self.device))
        self._graph.replay()
        return self._static_y.clone()

    def reset_graph(self) -> None:
        """Force re-capture on next call (e.g. after shape change)."""
        self._graph = None
        self._warmup_count = 0


# ──────────────────────────────────────────────────────────────────────────────
# 4. Gradient compression for DDP
# ──────────────────────────────────────────────────────────────────────────────

def register_powersgd_hook(
    model: nn.Module,
    *,
    matrix_approximation_rank: int = 1,
    start_powersgd_iter: int = 10,
    use_error_feedback: bool = True,
) -> None:
    """Register a PowerSGD gradient compression hook on a DDP model.

    PowerSGD (Vogels et al. 2019) factorises gradient matrices into low-rank
    factors, significantly reducing DDP all-reduce communication bandwidth.

    Parameters
    ----------
    model : DistributedDataParallel
        Must be wrapped in DDP.
    matrix_approximation_rank : int
        Rank r of the low-rank approximation.  Smaller → more compression,
        less accuracy.  Typical values: 1–4.
    start_powersgd_iter : int
        Iterations to warm up with no compression (needed for first few steps).
    use_error_feedback : bool
        Accumulate compression errors and add them to future gradients.

    Example
    -------
    >>> from torch.nn.parallel import DistributedDataParallel as DDP
    >>> model = DDP(model)
    >>> register_powersgd_hook(model, matrix_approximation_rank=2)
    """
    try:
        import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
    except ImportError:
        warnings.warn("PowerSGD hook not available in this PyTorch version.")
        return

    if not hasattr(model, "register_comm_hook"):
        warnings.warn(
            "register_powersgd_hook: model must be wrapped in DDP. "
            "Call torch.nn.parallel.DistributedDataParallel(model) first."
        )
        return

    state = powerSGD.PowerSGDState(
        process_group=None,
        matrix_approximation_rank=matrix_approximation_rank,
        start_powersgd_iter=start_powersgd_iter,
        use_error_feedback=use_error_feedback,
    )
    model.register_comm_hook(state, powerSGD.powerSGD_hook)
    logger.info(
        f"PowerSGD hook registered (rank={matrix_approximation_rank}, "
        f"error_feedback={use_error_feedback})."
    )


def register_topk_hook(model: nn.Module, *, compression_ratio: float = 0.1) -> None:
    """Register a Top-K gradient sparsification hook on a DDP model.

    Sends only the top ``compression_ratio`` fraction of gradients by
    absolute value; the rest are zeroed.  Simple and effective for
    bandwidth-limited settings.

    Parameters
    ----------
    model : DistributedDataParallel
    compression_ratio : float
        Fraction of gradients to communicate (0 < ratio <= 1).
    """
    if not hasattr(model, "register_comm_hook"):
        warnings.warn("register_topk_hook: model must be DDP-wrapped.")
        return

    def _topk_hook(state, bucket):
        tensor = bucket.get_gradients()[0].view(-1)
        k = max(1, int(len(tensor) * compression_ratio))
        _, top_idx = torch.topk(tensor.abs(), k)
        mask = torch.zeros_like(tensor)
        mask[top_idx] = 1.0
        tensor.mul_(mask)
        fut = torch.distributed.all_reduce(tensor, async_op=True).get_future()
        return fut.then(lambda f: [f.value()[0]])

    model.register_comm_hook(None, _topk_hook)
    logger.info(f"TopK hook registered (ratio={compression_ratio:.2%}).")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Cluster / multi-node launch helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TorchRunConfig:
    """Configuration for a ``torchrun`` launch.

    Parameters
    ----------
    n_nodes : int
        Number of compute nodes.
    n_procs_per_node : int
        GPUs (or processes) per node.
    master_addr : str
        IP or hostname of the master node.
    master_port : int
        TCP port for rendezvous.
    rdzv_backend : str
        "static" | "c10d" | "etcd".
    max_restarts : int
        Automatic fault-tolerance restarts (torchrun elastic).
    """
    n_nodes: int = 1
    n_procs_per_node: int = 1
    master_addr: str = "localhost"
    master_port: int = 29500
    rdzv_backend: str = "static"
    max_restarts: int = 0

    @property
    def world_size(self) -> int:
        return self.n_nodes * self.n_procs_per_node


@dataclass
class SLURMConfig:
    """SLURM batch script configuration.

    Parameters
    ----------
    job_name : str
    n_nodes : int
    n_gpus_per_node : int
    n_cpus_per_task : int
        Usually = n_gpus_per_node × threads_per_gpu.
    time_limit : str
        SLURM time limit, e.g. "08:00:00".
    partition : str
    account : str, optional
    memory_gb : int, optional
        Memory per node in GB.  0 = let SLURM decide.
    modules : list of str
        Environment modules to load before the Python command.
    conda_env : str, optional
        Conda environment to activate.
    """
    job_name: str = "pinneaple_job"
    n_nodes: int = 1
    n_gpus_per_node: int = 4
    n_cpus_per_task: int = 8
    time_limit: str = "04:00:00"
    partition: str = "gpu"
    account: str = ""
    memory_gb: int = 0
    modules: List[str] = field(default_factory=list)
    conda_env: str = ""


def build_torchrun_cmd(
    script: str,
    cfg: TorchRunConfig,
    *,
    script_args: str = "",
) -> str:
    """Generate a ``torchrun`` launch command string.

    Parameters
    ----------
    script : str
        Path to the Python training script.
    cfg : TorchRunConfig
    script_args : str
        Additional arguments forwarded to ``script``.

    Returns
    -------
    str
        Ready-to-execute command string.

    Example
    -------
    >>> cmd = build_torchrun_cmd("train.py", TorchRunConfig(n_nodes=2, n_procs_per_node=4))
    >>> print(cmd)
    """
    parts = [
        f"torchrun",
        f"--nnodes={cfg.n_nodes}",
        f"--nproc_per_node={cfg.n_procs_per_node}",
        f"--master_addr={cfg.master_addr}",
        f"--master_port={cfg.master_port}",
        f"--rdzv_backend={cfg.rdzv_backend}",
    ]
    if cfg.max_restarts > 0:
        parts.append(f"--max_restarts={cfg.max_restarts}")
    parts.append(script)
    if script_args:
        parts.append(script_args)
    return " ".join(parts)


def build_slurm_script(
    torchrun_cmd: str,
    slurm: SLURMConfig,
    *,
    output_dir: str = "slurm_logs",
) -> str:
    """Generate a complete SLURM batch script as a string.

    Parameters
    ----------
    torchrun_cmd : str
        The ``torchrun`` command (from :func:`build_torchrun_cmd`).
    slurm : SLURMConfig
    output_dir : str
        Directory for SLURM stdout/stderr.

    Returns
    -------
    str
        Complete SLURM script ready to submit with ``sbatch``.

    Example
    -------
    >>> script = build_slurm_script(cmd, SLURMConfig(n_nodes=2))
    >>> Path("job.sh").write_text(script)
    >>> # Then: sbatch job.sh
    """
    lines = ["#!/bin/bash"]
    lines += [
        f"#SBATCH --job-name={slurm.job_name}",
        f"#SBATCH --nodes={slurm.n_nodes}",
        f"#SBATCH --ntasks-per-node={slurm.n_gpus_per_node}",
        f"#SBATCH --gres=gpu:{slurm.n_gpus_per_node}",
        f"#SBATCH --cpus-per-task={slurm.n_cpus_per_task}",
        f"#SBATCH --time={slurm.time_limit}",
        f"#SBATCH --partition={slurm.partition}",
        f"#SBATCH --output={output_dir}/%j_out.txt",
        f"#SBATCH --error={output_dir}/%j_err.txt",
    ]
    if slurm.account:
        lines.append(f"#SBATCH --account={slurm.account}")
    if slurm.memory_gb > 0:
        lines.append(f"#SBATCH --mem={slurm.memory_gb}G")

    lines.append("")
    lines.append(f"mkdir -p {output_dir}")
    lines.append("")

    for mod in slurm.modules:
        lines.append(f"module load {mod}")
    if slurm.modules:
        lines.append("")

    if slurm.conda_env:
        lines.append(f"conda activate {slurm.conda_env}")
        lines.append("")

    lines += [
        "# Networking setup for multi-node",
        "export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)",
        "export MASTER_PORT=29500",
        "export WORLD_SIZE=$SLURM_NTASKS",
        "",
        f"srun {torchrun_cmd}",
    ]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Profiling
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ProfilerConfig:
    """Configuration for PINNeAPPleProfiler.

    Parameters
    ----------
    trace_dir : str
        Directory to save Chrome trace files.
    record_shapes : bool
    profile_memory : bool
    with_flops : bool
        Estimate FLOPs per operator (requires record_shapes=True).
    with_stack : bool
        Record source code stack traces (slower but more informative).
    schedule_wait : int
        Profiling schedule: wait N steps before starting.
    schedule_warmup : int
        Warmup N steps (recorded but not exported).
    schedule_active : int
        Record N active steps.
    schedule_repeat : int
        Repeat the wait-warmup-active cycle this many times (0 = run once).
    row_limit : int
        Top-N operators to print in the table.
    """
    trace_dir: str = "profiler_traces"
    record_shapes: bool = True
    profile_memory: bool = True
    with_flops: bool = True
    with_stack: bool = False
    schedule_wait: int = 2
    schedule_warmup: int = 2
    schedule_active: int = 5
    schedule_repeat: int = 1
    row_limit: int = 20


class PINNeAPPleProfiler:
    """Context-manager profiler that wraps ``torch.profiler.profile``.

    Records per-operator timing, memory, and FLOPs; exports Chrome traces
    for visualisation in ``chrome://tracing`` or Perfetto UI.

    Usage
    -----
    >>> cfg = ProfilerConfig(trace_dir="traces", with_flops=True)
    >>> with PINNeAPPleProfiler(cfg) as prof:
    ...     for batch in loader:
    ...         loss = compute_loss(batch)
    ...         loss.backward()
    ...         optimizer.step()
    ...         prof.step()
    >>> prof.print_summary()
    """

    def __init__(self, config: Optional[ProfilerConfig] = None) -> None:
        self.cfg = config or ProfilerConfig()
        self._prof = None

    def __enter__(self) -> "PINNeAPPleProfiler":
        Path(self.cfg.trace_dir).mkdir(parents=True, exist_ok=True)
        sched = torch.profiler.schedule(
            wait=self.cfg.schedule_wait,
            warmup=self.cfg.schedule_warmup,
            active=self.cfg.schedule_active,
            repeat=self.cfg.schedule_repeat,
        )

        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._prof = torch.profiler.profile(
            activities=activities,
            schedule=sched,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                self.cfg.trace_dir
            ),
            record_shapes=self.cfg.record_shapes,
            profile_memory=self.cfg.profile_memory,
            with_flops=self.cfg.with_flops,
            with_stack=self.cfg.with_stack,
        )
        self._prof.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._prof is not None:
            self._prof.__exit__(*args)

    def step(self) -> None:
        """Call once per training step to advance the profiling schedule."""
        if self._prof is not None:
            self._prof.step()

    def print_summary(self, sort_by: str = "cuda_time_total") -> None:
        """Print a table of the most time-consuming operators."""
        if self._prof is None:
            print("[PINNeAPPleProfiler] No profiling data (not inside context).")
            return
        try:
            table = self._prof.key_averages().table(
                sort_by=sort_by,
                row_limit=self.cfg.row_limit,
            )
            print(table)
        except Exception as e:
            print(f"[PINNeAPPleProfiler] Could not print summary: {e}")

    def export_chrome_trace(self, path: Optional[str] = None) -> str:
        """Export Chrome trace JSON to ``path`` and return the path."""
        if self._prof is None:
            raise RuntimeError("Profiler context has not been entered.")
        p = path or str(Path(self.cfg.trace_dir) / "trace.json")
        self._prof.export_chrome_trace(p)
        logger.info(f"Chrome trace exported to: {p}")
        return p


# ──────────────────────────────────────────────────────────────────────────────
# 7. Automatic batch size finder
# ──────────────────────────────────────────────────────────────────────────────

class AutoBatchSizeFinder:
    """Find the maximum batch size that fits in GPU memory via binary search.

    Parameters
    ----------
    trial_fn : callable(batch_size: int) -> bool
        Return True if training step succeeds with this batch size.
        Should catch ``torch.cuda.OutOfMemoryError`` internally and return False.
    lo : int
        Minimum batch size to try.
    hi : int
        Maximum batch size to try.
    verbose : bool

    Example
    -------
    >>> def trial(bs):
    ...     try:
    ...         x = torch.randn(bs, 2, device="cuda")
    ...         loss = model(x).sum()
    ...         loss.backward()
    ...         model.zero_grad()
    ...         return True
    ...     except torch.cuda.OutOfMemoryError:
    ...         torch.cuda.empty_cache()
    ...         return False
    >>>
    >>> finder = AutoBatchSizeFinder(trial, lo=8, hi=4096)
    >>> best = finder.find()
    >>> print(f"Max batch size: {best}")
    """

    def __init__(
        self,
        trial_fn: Callable[[int], bool],
        lo: int = 1,
        hi: int = 8192,
        verbose: bool = True,
    ) -> None:
        self.trial_fn = trial_fn
        self.lo = int(lo)
        self.hi = int(hi)
        self.verbose = verbose

    def find(self) -> int:
        """Run binary search.  Returns the largest successful batch size."""
        lo, hi = self.lo, self.hi
        best = lo

        # Quick sanity check: can we do the minimum?
        if not self.trial_fn(lo):
            warnings.warn(f"AutoBatchSizeFinder: even batch_size={lo} failed.")
            return 0

        while lo <= hi:
            mid = (lo + hi) // 2
            if self.verbose:
                print(f"  [AutoBatchSize] trying {mid}...", end=" ", flush=True)
            ok = self.trial_fn(mid)
            if ok:
                best = mid
                lo = mid + 1
                if self.verbose:
                    print("OK")
            else:
                hi = mid - 1
                if self.verbose:
                    print("OOM")
            # Clear any cached allocations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.verbose:
            print(f"  [AutoBatchSize] => best batch size: {best}")
        return best


__all__ = [
    # FSDP
    "FSDPConfig",
    "wrap_fsdp",
    # ZeRO
    "wrap_zero_optimizer",
    # CUDA Graphs
    "CUDAGraphModule",
    # Gradient compression
    "register_powersgd_hook",
    "register_topk_hook",
    # Launch helpers
    "TorchRunConfig",
    "SLURMConfig",
    "build_torchrun_cmd",
    "build_slurm_script",
    # Profiling
    "ProfilerConfig",
    "PINNeAPPleProfiler",
    # Batch size
    "AutoBatchSizeFinder",
]
