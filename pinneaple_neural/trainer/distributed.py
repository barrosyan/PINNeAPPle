"""Distributed Data Parallel (DDP) training for PINNeAPPle.

Wraps the standard Trainer with torch.distributed DDP for multi-GPU /
multi-node training.

Launch with torchrun::

    torchrun --nproc_per_node=4 my_script.py

In the script::

    trainer = DDPPINNTrainer(model, config=DDPTrainerConfig())
    history = trainer.train(loss_fn, n_epochs=10000)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.multiprocessing as mp
    _DIST_AVAILABLE = True
except Exception:  # pragma: no cover
    dist = None  # type: ignore[assignment]
    DDP = None   # type: ignore[assignment]
    mp = None    # type: ignore[assignment]
    _DIST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def is_distributed() -> bool:
    """Return True when running inside an active distributed process group."""
    return _DIST_AVAILABLE and dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Return the rank of the current process (0 when not distributed)."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Return the total number of processes (1 when not distributed)."""
    return dist.get_world_size() if is_distributed() else 1


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class DDPTrainerConfig:
    """Configuration for DDP training.

    Attributes
    ----------
    backend          : ``"nccl"`` for GPU–GPU communication (recommended),
                       ``"gloo"`` for CPU or cross-platform.
    world_size       : total number of processes; ``-1`` auto-detects from
                       ``CUDA_VISIBLE_DEVICES`` (falls back to 1 for CPU).
    master_addr      : hostname/IP of rank-0 process.
    master_port      : TCP port used for the rendezvous.
    gradient_sync_every : sync gradients every N steps (1 = always sync,
                          N>1 = gradient accumulation across ranks).
    find_unused_parameters : forward to DDP; set ``True`` when your model
                             has branches not always exercised.
    grad_clip        : maximum gradient norm (0 = disabled).
    amp              : enable automatic mixed precision (requires CUDA).
    print_every      : log interval (epochs).  Only rank-0 prints.
    """

    backend: str = "nccl"
    world_size: int = -1
    master_addr: str = "localhost"
    master_port: str = "12355"
    gradient_sync_every: int = 1
    find_unused_parameters: bool = False
    grad_clip: float = 0.0
    amp: bool = False
    print_every: int = 500


# ---------------------------------------------------------------------------
# DDPPINNTrainer
# ---------------------------------------------------------------------------

class DDPPINNTrainer:
    """Multi-GPU PINN trainer using PyTorch DDP.

    Designed to work both when launched via ``torchrun`` (process group
    already initialised by the launcher) and via the ``spawn_workers``
    helper that calls ``mp.spawn``.

    Parameters
    ----------
    model  : the PINN model (CPU or single-GPU — will be moved to the
             correct device inside ``setup``).
    config : ``DDPTrainerConfig`` instance.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[DDPTrainerConfig] = None,
    ):
        if not _DIST_AVAILABLE:
            raise RuntimeError(
                "torch.distributed is not available in this environment."
            )
        self.model = model
        self.config = config or DDPTrainerConfig()
        self._ddp_model: Optional[nn.Module] = None
        self._device: Optional[torch.device] = None
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None  # type: ignore[name-defined]

    # ------------------------------------------------------------------
    # Process-group lifecycle
    # ------------------------------------------------------------------

    def setup(self, rank: int, world_size: int) -> None:
        """Initialise the process group and move model to the correct device.

        Parameters
        ----------
        rank       : rank of the *calling* process.
        world_size : total number of processes.
        """
        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port

        dist.init_process_group(
            backend=self.config.backend,
            rank=rank,
            world_size=world_size,
        )

        if torch.cuda.is_available():
            self._device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            torch.cuda.set_device(self._device)
        else:
            self._device = torch.device("cpu")

        self.model = self.model.to(self._device)
        self._ddp_model = DDP(
            self.model,
            device_ids=[self._device.index] if self._device.type == "cuda" else None,
            find_unused_parameters=self.config.find_unused_parameters,
        )

        if self.config.amp and self._device.type == "cuda":
            self._scaler = torch.cuda.amp.GradScaler()  # type: ignore[attr-defined]

    def cleanup(self) -> None:
        """Destroy the process group cleanly."""
        if is_distributed():
            dist.destroy_process_group()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        loss_fn: Callable[[nn.Module, int], torch.Tensor],
        n_epochs: int,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler=None,
    ) -> Dict[str, List[float]]:
        """Training loop with gradient synchronisation across GPUs.

        Parameters
        ----------
        loss_fn  : callable(model, epoch) -> scalar Tensor *or* dict with
                   ``"total"`` key.  The model passed is the DDP-wrapped one
                   (or the plain model when not distributed).
        n_epochs : total number of gradient steps.
        optimizer: optional pre-built optimizer.  When ``None``, Adam is used
                   with ``lr=1e-3``.
        scheduler: optional LR scheduler; ``step()`` is called after every
                   gradient update.

        Returns
        -------
        History dict (only populated on rank 0).
        """
        active_model = self._ddp_model if self._ddp_model is not None else self.model
        device = self._device or next(self.model.parameters()).device

        if optimizer is None:
            optimizer = optim.Adam(active_model.parameters(), lr=1e-3)

        cfg = self.config
        rank = get_rank()
        history: Dict[str, List[float]] = {"total": [], "epoch": []}

        for epoch in range(1, n_epochs + 1):
            # -- gradient accumulation across ranks --
            sync_context = (
                active_model.no_sync()  # type: ignore[union-attr]
                if isinstance(active_model, DDP) and epoch % cfg.gradient_sync_every != 0
                else _null_context()
            )

            with sync_context:
                optimizer.zero_grad(set_to_none=True)

                if cfg.amp and self._scaler is not None:
                    with torch.cuda.amp.autocast():  # type: ignore[attr-defined]
                        loss_out = loss_fn(active_model, epoch)
                    total = _extract_total(loss_out)
                    self._scaler.scale(total).backward()
                    if cfg.grad_clip > 0.0:
                        self._scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(
                            active_model.parameters(), cfg.grad_clip
                        )
                    self._scaler.step(optimizer)
                    self._scaler.update()
                else:
                    loss_out = loss_fn(active_model, epoch)
                    total = _extract_total(loss_out)
                    total.backward()
                    if cfg.grad_clip > 0.0:
                        nn.utils.clip_grad_norm_(
                            active_model.parameters(), cfg.grad_clip
                        )
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            # -- logging (rank 0 only) --
            if rank == 0:
                history["total"].append(float(total.detach()))
                history["epoch"].append(epoch)

                if cfg.print_every > 0 and (epoch % cfg.print_every == 0 or epoch == 1):
                    extra = ""
                    if isinstance(loss_out, dict):
                        extra = "  ".join(
                            f"{k}={float(v.detach()):.4e}"
                            for k, v in loss_out.items()
                            if k != "total" and isinstance(v, torch.Tensor)
                        )
                    print(
                        f"[DDP rank=0] epoch={epoch:05d}  "
                        f"total={float(total.detach()):.4e}  {extra}"
                    )

        return history

    # ------------------------------------------------------------------
    # Spawn helper
    # ------------------------------------------------------------------

    @staticmethod
    def spawn_workers(
        train_fn: Callable[[int, int], None],
        n_gpus: int,
        *args,
    ) -> None:
        """Launch training on n_gpus using mp.spawn.

        Parameters
        ----------
        train_fn : callable(rank, world_size, *args) -> None.
                   Should call ``trainer.setup(rank, world_size)``,
                   ``trainer.train(...)``, and ``trainer.cleanup()``.
        n_gpus   : number of worker processes to spawn.
        *args    : additional positional arguments forwarded to train_fn.
        """
        if mp is None:
            raise RuntimeError("torch.multiprocessing is not available.")
        world_size = n_gpus if n_gpus > 0 else 1
        mp.spawn(
            fn=lambda rank, *a: train_fn(rank, world_size, *a),
            args=args,
            nprocs=world_size,
            join=True,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _extract_total(loss_out) -> torch.Tensor:
    """Extract the scalar loss from a Tensor or dict."""
    if isinstance(loss_out, torch.Tensor):
        return loss_out
    if isinstance(loss_out, dict):
        if "total" not in loss_out:
            raise KeyError("loss_fn dict must contain a 'total' key.")
        return loss_out["total"]
    raise TypeError(f"loss_fn must return Tensor or dict, got {type(loss_out)}")


class _null_context:
    """No-op context manager (replaces model.no_sync when not needed)."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
