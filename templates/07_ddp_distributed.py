"""07_ddp_distributed.py — Distributed Data Parallel PINN training.

Demonstrates:
- DDPPINNTrainer with DDPTrainerConfig
- Multi-GPU setup using mp.spawn (falls back gracefully to single-GPU / CPU)
- 2D Poisson PINN as the training task

Launch with torchrun for true multi-GPU:
    torchrun --nproc_per_node=4 07_ddp_distributed.py

Or run as a plain Python script for single-process (mp.spawn with n_procs=1):
    python 07_ddp_distributed.py
"""

import os
import math
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np

from pinneaple_train.distributed import DDPPINNTrainer, DDPTrainerConfig, get_rank


# ---------------------------------------------------------------------------
# Shared model definition
# ---------------------------------------------------------------------------

def make_model():
    return nn.Sequential(
        nn.Linear(2, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 1),
    )


# ---------------------------------------------------------------------------
# PDE loss function: 2D Poisson Δu + 2π² sin(πx)sin(πy) = 0
# ---------------------------------------------------------------------------

def build_loss_fn(n_col: int = 2000):
    """Return a loss callable (model, epoch) -> scalar tensor."""

    def loss_fn(model: nn.Module, epoch: int) -> torch.Tensor:
        device = next(model.parameters()).device

        # Fresh collocation batch each epoch
        xy = torch.rand(n_col, 2, device=device)
        xy.requires_grad_(True)

        out = model(xy)
        if hasattr(out, "y"):
            out = out.y

        grad1 = torch.autograd.grad(out.sum(), xy, create_graph=True)[0]
        u_x = grad1[:, 0:1]
        u_y = grad1[:, 1:2]
        u_xx = torch.autograd.grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y.sum(), xy, create_graph=True)[0][:, 1:2]

        f = -2.0 * math.pi**2 * torch.sin(math.pi * xy[:, 0:1]) * torch.sin(math.pi * xy[:, 1:2])
        residual = u_xx + u_yy - f
        return residual.pow(2).mean()

    return loss_fn


# ---------------------------------------------------------------------------
# Worker function — one process per GPU rank
# ---------------------------------------------------------------------------

def worker(rank: int, world_size: int, config: DDPTrainerConfig, n_epochs: int):
    """Per-process training function (spawned by mp.spawn or torchrun)."""
    model = make_model()
    trainer = DDPPINNTrainer(model=model, config=config)
    trainer.setup(rank=rank, world_size=world_size)

    loss_fn = build_loss_fn(n_col=2000)
    history = trainer.train(loss_fn=loss_fn, n_epochs=n_epochs)

    if get_rank() == 0:
        final_loss = history["total"][-1] if history["total"] else float("nan")
        print(f"[rank 0] final loss = {final_loss:.4e}")

        # Save rank-0 model
        torch.save(trainer.model.state_dict(), "07_ddp_model.pt")
        print("Saved 07_ddp_model.pt")

    trainer.cleanup()


# ---------------------------------------------------------------------------
# Main — detects available GPUs and spawns workers
# ---------------------------------------------------------------------------

def main():
    n_gpus = torch.cuda.device_count()
    world_size = max(1, n_gpus)          # fall back to 1 (CPU or single GPU)
    backend    = "nccl" if n_gpus > 1 else "gloo"
    n_epochs   = 3000

    print(f"DDP training | world_size={world_size} | backend={backend}")

    config = DDPTrainerConfig(
        backend=backend,
        world_size=world_size,
        master_addr="localhost",
        master_port="29500",
        grad_clip=1.0,
        amp=(n_gpus > 0),
        print_every=500,
    )

    if world_size == 1:
        # Single-process path (no spawn overhead)
        worker(rank=0, world_size=1, config=config, n_epochs=n_epochs)
    else:
        # True multi-process DDP via mp.spawn
        mp.spawn(
            fn=worker,
            args=(world_size, config, n_epochs),
            nprocs=world_size,
            join=True,
        )

    print("Done.")


if __name__ == "__main__":
    main()
