"""Parallelization and GPU optimization utilities for pinneaple training.

Provides:
- Multi-GPU DataParallel / DistributedDataParallel wrappers
- Automatic mixed-precision (AMP) context manager
- Gradient accumulation trainer mixin
- Asynchronous data prefetching
- Torch.compile() integration (PyTorch 2.x+)
- CPU thread pool for data-parallel inference
- Parallel hyperparameter sweep runner
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------

def best_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_gpus() -> int:
    """Return the number of available CUDA GPUs."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def gpu_info() -> List[Dict[str, Any]]:
    """Return info dict for each available GPU."""
    info = []
    for i in range(count_gpus()):
        props = torch.cuda.get_device_properties(i)
        info.append({
            "index": i,
            "name": props.name,
            "total_memory_GB": props.total_memory / 1024 ** 3,
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count,
        })
    return info


# ---------------------------------------------------------------------------
# Model compilation (PyTorch 2.x)
# ---------------------------------------------------------------------------

def maybe_compile(
    model: nn.Module,
    *,
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> nn.Module:
    """
    Wrap model with torch.compile() if supported (PyTorch >= 2.0).

    Parameters
    ----------
    model     : the model to compile
    mode      : "default" | "reduce-overhead" | "max-autotune"
    fullgraph : if True, require full-graph compilation (strict)
    dynamic   : if True, allow dynamic shapes (slower but more general)
    """
    if not hasattr(torch, "compile"):
        logger.info("torch.compile not available (PyTorch < 2.0); skipping compilation.")
        return model
    try:
        compiled = torch.compile(model, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
        logger.info(f"Model compiled with torch.compile (mode={mode})")
        return compiled
    except Exception as exc:
        logger.warning(f"torch.compile failed: {exc}. Returning uncompiled model.")
        return model


# ---------------------------------------------------------------------------
# Automatic Mixed Precision (AMP)
# ---------------------------------------------------------------------------

class AMPContext:
    """
    Context manager for automatic mixed precision training.

    Supports CUDA AMP (fp16/bf16) and falls back to no-op on CPU/MPS.
    Uses the non-deprecated ``torch.amp`` API (PyTorch 2.x).

    Usage
    -----
    >>> ctx = AMPContext(device="cuda", dtype=torch.bfloat16)
    >>> with ctx.autocast():
    ...     y = model(x)
    >>> ctx.scaler.scale(loss).backward()
    >>> ctx.scaler.step(optimizer)
    >>> ctx.scaler.update()
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enabled: bool = True,
    ) -> None:
        self.device_str = str(device)
        self.device_type = "cuda" if self.device_str.startswith("cuda") else "cpu"
        self.dtype = dtype
        self.enabled = bool(enabled) and self.device_type == "cuda"
        # Use updated API; fall back for older PyTorch
        try:
            self.scaler = torch.amp.GradScaler(self.device_type, enabled=self.enabled)
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)  # type: ignore[attr-defined]

    def autocast(self):
        """Return an autocast context manager."""
        return torch.amp.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.enabled,
        )


# ---------------------------------------------------------------------------
# Data parallel wrappers
# ---------------------------------------------------------------------------

def wrap_data_parallel(model: nn.Module, device_ids: Optional[List[int]] = None) -> nn.Module:
    """
    Wrap model in DataParallel if multiple GPUs are available.

    Uses DataParallel (single-machine, single-process multi-GPU).
    For multi-node, use DistributedDataParallel via Trainer.ddp=True.
    """
    n = count_gpus()
    if n <= 1:
        return model
    ids = device_ids or list(range(n))
    logger.info(f"Wrapping model in DataParallel on GPUs: {ids}")
    return nn.DataParallel(model, device_ids=ids)


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel / DistributedDataParallel to get the raw model."""
    if hasattr(model, "module"):
        return model.module
    return model


# ---------------------------------------------------------------------------
# Prefetch dataloader
# ---------------------------------------------------------------------------

class CUDAPrefetcher:
    """
    Asynchronous GPU prefetcher that overlaps CPU→GPU data transfer
    with GPU computation using CUDA streams.

    Usage
    -----
    >>> prefetcher = CUDAPrefetcher(loader, device="cuda")
    >>> batch = prefetcher.next()
    >>> while batch is not None:
    ...     # process batch (already on GPU)
    ...     batch = prefetcher.next()
    """

    def __init__(self, loader: DataLoader, device: str = "cuda") -> None:
        self.loader = loader
        self.device = torch.device(device)
        self._stream = torch.cuda.Stream() if self.device.type == "cuda" else None
        self._iter: Optional[Iterator] = None
        self._next_batch: Optional[Any] = None

    def __iter__(self) -> "CUDAPrefetcher":
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self) -> None:
        try:
            self._next_batch = next(self._iter)
        except StopIteration:
            self._next_batch = None
            return
        if self._stream is not None:
            with torch.cuda.stream(self._stream):
                self._next_batch = self._move_to_device(self._next_batch)

    def _move_to_device(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            moved = [self._move_to_device(v) for v in obj]
            return type(obj)(moved)
        return obj

    def next(self) -> Optional[Any]:
        if self._stream is not None:
            torch.cuda.current_stream().wait_stream(self._stream)
        batch = self._next_batch
        self._preload()
        return batch

    def __next__(self) -> Any:
        batch = self.next()
        if batch is None:
            raise StopIteration
        return batch


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------

@dataclass
class GradAccumConfig:
    """Configuration for gradient accumulation."""
    accumulate_steps: int = 4     # accumulate gradients over N micro-batches
    clip_grad_norm: float = 1.0   # max gradient norm (0 = disabled)


class GradAccumTrainer:
    """
    Mixin / wrapper that adds gradient accumulation to any training loop.

    Effective batch size = micro_batch_size * accumulate_steps.

    Usage
    -----
    >>> gat = GradAccumTrainer(model, optimizer, cfg=GradAccumConfig(4))
    >>> for micro_batch in loader:
    ...     loss = gat.step(micro_batch, loss_fn)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: Optional[GradAccumConfig] = None,
        amp_ctx: Optional[AMPContext] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg or GradAccumConfig()
        self.amp_ctx = amp_ctx
        self._step = 0

    def step(
        self,
        batch: Any,
        loss_fn: Callable[[nn.Module, Any], torch.Tensor],
    ) -> Optional[float]:
        """
        Process one micro-batch.  Returns the optimizer step loss (or None if
        still accumulating).
        """
        self._step += 1
        is_update_step = (self._step % self.cfg.accumulate_steps == 0)

        ctx = self.amp_ctx.autocast() if self.amp_ctx else _null_ctx()
        with ctx:
            loss = loss_fn(self.model, batch) / self.cfg.accumulate_steps

        if self.amp_ctx and self.amp_ctx.enabled:
            self.amp_ctx.scaler.scale(loss).backward()
        else:
            loss.backward()

        if is_update_step:
            if self.cfg.clip_grad_norm > 0:
                if self.amp_ctx and self.amp_ctx.enabled:
                    self.amp_ctx.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad_norm
                )
            if self.amp_ctx and self.amp_ctx.enabled:
                self.amp_ctx.scaler.step(self.optimizer)
                self.amp_ctx.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            return float(loss.item()) * self.cfg.accumulate_steps

        return None


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


# ---------------------------------------------------------------------------
# Parallel hyperparameter sweep
# ---------------------------------------------------------------------------

@dataclass
class SweepConfig:
    """Hyperparameter sweep configuration."""
    param_grid: Dict[str, List[Any]]     # {"lr": [1e-3, 1e-4], "hidden": [64, 128]}
    n_jobs: int = 4                       # parallel workers
    backend: str = "thread"              # "thread" | "process"
    timeout_per_trial: Optional[float] = None

    def grid_points(self) -> List[Dict[str, Any]]:
        """Return all combinations in the param grid."""
        import itertools
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def run_parallel_sweep(
    trial_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    cfg: SweepConfig,
) -> List[Dict[str, Any]]:
    """
    Run a hyperparameter sweep in parallel.

    Parameters
    ----------
    trial_fn : callable
        Function that takes a dict of hyperparameters and returns a dict
        of results (e.g. {"val_loss": 0.01, "epoch": 50}).
    cfg : SweepConfig
        Grid and parallelism configuration.

    Returns
    -------
    List of result dicts, each including the hyperparameters that produced it.
    """
    points = cfg.grid_points()
    results: List[Dict[str, Any]] = []

    Executor = ThreadPoolExecutor if cfg.backend == "thread" else ProcessPoolExecutor

    with Executor(max_workers=cfg.n_jobs) as pool:
        futures = {pool.submit(trial_fn, p): p for p in points}
        for fut in as_completed(futures, timeout=cfg.timeout_per_trial):
            params = futures[fut]
            try:
                res = fut.result()
                res.update({"params": params})
                results.append(res)
                logger.info(f"[sweep] params={params} -> {res}")
            except Exception as exc:
                logger.warning(f"[sweep] trial failed params={params}: {exc}")
                results.append({"params": params, "error": str(exc)})

    # Sort by first numeric result key
    for key in results[0] if results else {}:
        if key != "params" and isinstance(results[0].get(key), (int, float)):
            results.sort(key=lambda r: r.get(key, float("inf")))
            break

    return results


# ---------------------------------------------------------------------------
# Batched GPU inference helper
# ---------------------------------------------------------------------------

def batched_inference(
    model: nn.Module,
    x: torch.Tensor,
    *,
    batch_size: int = 4096,
    device: Optional[str] = None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Run model inference over a large tensor in batches to avoid OOM.

    Parameters
    ----------
    model      : nn.Module in eval mode
    x          : (N, D) input tensor
    batch_size : points per forward pass
    device     : device to move chunks to (defaults to model's device)
    use_amp    : use autocast for faster fp16 inference
    """
    model.eval()
    dev = next(model.parameters()).device if device is None else torch.device(device)

    outputs: List[torch.Tensor] = []
    N = x.shape[0]

    with torch.no_grad():
        for start in range(0, N, batch_size):
            chunk = x[start: start + batch_size].to(dev)
            if use_amp and dev.type == "cuda":
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    out = model(chunk)
            else:
                out = model(chunk)

            # Unwrap PINNOutput / OperatorOutput
            if hasattr(out, "y"):
                out = out.y

            outputs.append(out.cpu())

    return torch.cat(outputs, dim=0)


# ---------------------------------------------------------------------------
# Memory-efficient training with gradient checkpointing
# ---------------------------------------------------------------------------

def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Enable gradient checkpointing on transformer / large MLP models.

    Works with models that have a ``gradient_checkpointing_enable`` method
    (HuggingFace style) or by manually wrapping forward layers.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled (HuggingFace API).")
    else:
        logger.info(
            "Model does not support gradient_checkpointing_enable. "
            "Use torch.utils.checkpoint.checkpoint manually in forward()."
        )
    return model


# ---------------------------------------------------------------------------
# Training throughput profiler
# ---------------------------------------------------------------------------

class ThroughputMonitor:
    """
    Training throughput and GPU health monitor.

    Tracks samples/sec, GPU memory (allocated, reserved, peak), GPU
    utilisation (if ``pynvml`` is available), and per-epoch timings.

    Usage
    -----
    >>> mon = ThroughputMonitor()
    >>> mon.start_epoch()
    >>> # training loop...
    >>> mon.end_epoch(n_samples=4096)
    >>> print(mon.summary())
    >>> mon.print_epoch_table()
    """

    def __init__(self, device_index: int = 0) -> None:
        self._t0: float = 0.0
        self._epochs: List[Dict[str, Any]] = []
        self._device_index = device_index
        self._nvml_handle = None
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        except Exception:
            pass

    def _gpu_util(self) -> Optional[int]:
        """Return GPU utilisation % via pynvml, or None."""
        if self._nvml_handle is None:
            return None
        try:
            import pynvml  # type: ignore
            rates = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            return int(rates.gpu)
        except Exception:
            return None

    def start_epoch(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()

    def end_epoch(self, n_samples: int) -> Dict[str, Any]:
        dt = time.perf_counter() - self._t0
        samples_sec = n_samples / max(dt, 1e-9)
        info: Dict[str, Any] = {
            "epoch": len(self._epochs) + 1,
            "elapsed_s": round(dt, 3),
            "samples_sec": round(samples_sec, 1),
        }
        if torch.cuda.is_available():
            info["gpu_mem_alloc_GB"]    = round(torch.cuda.memory_allocated()  / 1024 ** 3, 3)
            info["gpu_mem_reserved_GB"] = round(torch.cuda.memory_reserved()   / 1024 ** 3, 3)
            info["gpu_mem_peak_GB"]     = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        util = self._gpu_util()
        if util is not None:
            info["gpu_util_pct"] = util
        self._epochs.append(info)
        return info

    def summary(self) -> Dict[str, Any]:
        if not self._epochs:
            return {}
        avg_sps = sum(e["samples_sec"] for e in self._epochs) / len(self._epochs)
        peak_mem = max((e.get("gpu_mem_peak_GB", 0) for e in self._epochs), default=0.0)
        return {
            "n_epochs": len(self._epochs),
            "avg_samples_sec": round(avg_sps, 1),
            "total_elapsed_s": round(sum(e["elapsed_s"] for e in self._epochs), 2),
            "peak_gpu_mem_GB": peak_mem,
        }

    def print_epoch_table(self) -> None:
        """Print a compact per-epoch table to stdout."""
        if not self._epochs:
            print("[ThroughputMonitor] No epochs recorded.")
            return
        header = f"{'Epoch':>6}  {'Elapsed(s)':>10}  {'Samp/s':>8}  {'MemAlloc(GB)':>13}  {'Peak(GB)':>9}"
        print(header)
        print("-" * len(header))
        for e in self._epochs:
            print(
                f"{e['epoch']:>6}  {e['elapsed_s']:>10.3f}  "
                f"{e['samples_sec']:>8.1f}  "
                f"{e.get('gpu_mem_alloc_GB', 0.0):>13.3f}  "
                f"{e.get('gpu_mem_peak_GB', 0.0):>9.3f}"
            )
