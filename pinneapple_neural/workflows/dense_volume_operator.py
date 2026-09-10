"""A reusable train/predict workflow for a grid-based neural operator
(:class:`~pinneapple_neural.architectures.neural_operators.fno.FNO3d`)
trained on a DENSE-in-time real volume time series -- e.g. the output of
:func:`pinneapple_data.adapters.load_dense_volumes` from a ``.splash``
OpenFOAM archive, but this module itself takes plain numpy arrays and
never imports anything splash-specific, so it works for any
``(T, C, D, H, W)`` real time series.

Promoted from the splash-pinneapple downstream project's own
``pipeline/train_fno3d.py`` (single-step training + a text-only
autoregressive rollout sanity check) with one real fix added here:
:func:`train_dense_volume_operator`'s default config now also trains
with the **pushforward trick** (Brandstetter, Worrall & Welling 2022,
"Message Passing Neural PDE Solvers" -- the same idea independently
used to stabilize FNO/neural-operator rollouts elsewhere in the
literature): occasionally roll the model forward K>1 steps using its
OWN prediction as the next input (no gradient through steps 1..K-1),
and backprop only through the LAST step's loss against the real frame
K steps ahead. This exposes the model to the input DISTRIBUTION it will
actually see at autoregressive inference time (its own prior
predictions, which carry error, not always the clean ground truth) --
without this, a model trained purely on ground-truth-in/ground-truth-out
single steps can achieve a low single-step loss while still diverging
catastrophically once fed its own output repeatedly (this is exactly
what was observed training a pure single-step FNO3d on real channel-flow
data: train_mse=3.06e-2 single-step, but a 20-step autoregressive
rollout's RMSE grew past 5000% of the field's std -- see this module's
own tests for a synthetic reproduction of that failure mode and the
pushforward fix's effect on it).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from pinneapple_neural.architectures.neural_operators.fno import FNO3d
from pinneapple_neural.trainer.checkpoint import Checkpoint, save_checkpoint, load_checkpoint

__all__ = [
    "DenseVolumeOperatorConfig",
    "train_dense_volume_operator",
    "rollout_dense_volume_operator",
    "predict_single_step",
    "load_dense_volume_operator",
]


@dataclass
class DenseVolumeOperatorConfig:
    in_channels: int
    out_channels: int
    width: int = 16
    modes: int = 8
    layers: int = 4
    epochs: int = 600
    lr: float = 2e-3
    val_frac: float = 0.15
    batch_size: int = 4
    rollout_steps: int = 20  # for the post-training sanity check only
    log_every: int = 25
    device: str = "auto"
    # -- pushforward / multi-step stabilization (see module docstring) --
    pushforward_max_steps: int = 4
    pushforward_prob: float = 0.5  # fraction of training BATCHES that use a multi-step rollout instead of plain single-step
    pushforward_warmup_epochs: int = 50  # pure single-step for this many epochs first, then start mixing in pushforward -- gives the model a stable single-step baseline before asking it to correct its own compounding errors


def _best_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _make_pairs(frames_n: np.ndarray, max_k: int) -> List[Tuple[int, int]]:
    """(start_index, k) pairs such that frames_n[start : start+k+1] is a
    valid, in-bounds (k+1)-length rollout window."""
    T = frames_n.shape[0]
    return [(i, T - 1 - i) for i in range(T - 1)]  # k capped per-sample at call time by min(max_k, available)


def train_dense_volume_operator(
    frames: np.ndarray,  # (T, C, D, H, W) real, NOT normalized
    cfg: DenseVolumeOperatorConfig,
    *, seed: int = 0,
) -> Tuple[Checkpoint, List[Dict[str, Any]]]:
    """Train an :class:`FNO3d` on a real dense volume time series.

    Returns ``(checkpoint, history)`` -- ``checkpoint`` is NOT saved to
    disk here (call :func:`pinneapple_neural.trainer.checkpoint
    .save_checkpoint` explicitly); this keeps the function usable both
    from a CLI script and from an API route that wants to decide the
    storage path/backend itself.
    """
    device = _best_device(cfg.device)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    mean = frames.mean(axis=(0, 2, 3, 4), keepdims=True)
    std = frames.std(axis=(0, 2, 3, 4), keepdims=True).clip(min=1e-8)
    frames_n = (frames - mean) / std

    T = frames_n.shape[0]
    n_pairs = T - 1
    n_val = max(1, int(cfg.val_frac * n_pairs))
    n_train = n_pairs - n_val

    X_all = torch.as_tensor(frames_n)  # (T, C, D, H, W), index 0..T-1
    X_train_start = 0
    train_end = n_train  # frames[0..n_train] usable as training rollout windows (start index < n_train)
    val_end = n_pairs  # frames[n_train..n_pairs] is the held-out temporal tail

    net = FNO3d(
        in_channels=cfg.in_channels, out_channels=cfg.out_channels, width=cfg.width,
        modes1=cfg.modes, modes2=cfg.modes, modes3=cfg.modes, layers=cfg.layers, use_grid=True,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)
    loss_fn = nn.MSELoss()

    X_val = X_all[n_train:n_train + n_val].to(device)
    Y_val = X_all[n_train + 1:n_train + 1 + n_val].to(device)

    history: List[Dict[str, Any]] = []
    t_start = time.time()
    bs = max(1, cfg.batch_size)

    for epoch in range(cfg.epochs):
        net.train()
        use_pushforward = epoch >= cfg.pushforward_warmup_epochs and cfg.pushforward_max_steps > 1
        start_indices = rng.permutation(train_end)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, train_end, bs):
            idx = start_indices[i : i + bs]
            opt.zero_grad(set_to_none=True)

            if use_pushforward and rng.random() < cfg.pushforward_prob:
                # Multi-step pushforward: roll K steps (K chosen per-batch, capped
                # by how much real data is available ahead of each start index),
                # no gradient through steps 1..K-1, backprop only the last step.
                k_max_per_sample = [min(cfg.pushforward_max_steps, train_end - int(s)) for s in idx]
                k = max(1, min(k_max_per_sample))  # a single shared k for the whole batch (simplicity)
                cur = X_all[idx].to(device)
                with torch.no_grad():
                    for _ in range(k - 1):
                        cur = net(cur).y
                target = X_all[[int(s) + k for s in idx]].to(device)
                pred = net(cur).y
                loss = loss_fn(pred, target)
            else:
                x_batch = X_all[idx].to(device)
                y_batch = X_all[[int(s) + 1 for s in idx]].to(device)
                pred = net(x_batch).y
                loss = loss_fn(pred, y_batch)

            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        sched.step()
        epoch_loss /= max(n_batches, 1)

        if epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1:
            net.eval()
            with torch.no_grad():
                val_loss = loss_fn(net(X_val).y, Y_val).item() if X_val.shape[0] > 0 else float("nan")
            elapsed = time.time() - t_start
            history.append({
                "epoch": epoch, "train_mse": epoch_loss, "val_mse": val_loss,
                "elapsed_s": elapsed, "used_pushforward": use_pushforward,
            })

    checkpoint = Checkpoint(
        model_state=net.state_dict(), optim_state=opt.state_dict(),
        cfg={"in_channels": cfg.in_channels, "out_channels": cfg.out_channels},
        meta={
            "width": cfg.width, "modes": cfg.modes, "layers": cfg.layers,
            "mean": mean.tolist(), "std": std.tolist(), "epochs": cfg.epochs, "device": device,
            "n_params": n_params, "pushforward_max_steps": cfg.pushforward_max_steps,
            "pushforward_prob": cfg.pushforward_prob, "pushforward_warmup_epochs": cfg.pushforward_warmup_epochs,
        },
    )
    return checkpoint, history


def load_dense_volume_operator(checkpoint: Checkpoint, *, device: str = "cpu") -> nn.Module:
    """Reconstruct the real ``FNO3d`` from a checkpoint produced by
    :func:`train_dense_volume_operator` (or the same-shaped legacy
    checkpoints ``train_fno3d.py`` already produced before this module
    existed -- both save the same ``meta`` keys)."""
    meta, cfg = checkpoint.meta, checkpoint.cfg
    net = FNO3d(
        in_channels=cfg["in_channels"] if "in_channels" in cfg else 4,
        out_channels=cfg["out_channels"] if "out_channels" in cfg else 4,
        width=meta["width"], modes1=meta["modes"], modes2=meta["modes"], modes3=meta["modes"],
        layers=meta["layers"], use_grid=True,
    ).to(device)
    net.load_state_dict(checkpoint.model_state)
    net.eval()
    return net


def predict_single_step(checkpoint: Checkpoint, input_frame: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    """One forward pass: ``input_frame`` (C, D, H, W), real (not
    normalized) units in, real units out. This is the regime the
    network is most accurate in -- see the module docstring."""
    net = load_dense_volume_operator(checkpoint, device=device)
    mean = np.array(checkpoint.meta["mean"], dtype=np.float32)
    std = np.array(checkpoint.meta["std"], dtype=np.float32)
    x_n = (input_frame[None] - mean) / std
    with torch.no_grad():
        pred_n = net(torch.as_tensor(x_n, device=device)).y[0].cpu().numpy()
    return pred_n * std[0] + mean[0]


def rollout_dense_volume_operator(
    checkpoint: Checkpoint, initial_frame: np.ndarray, n_steps: int, *, device: str = "cpu",
) -> np.ndarray:
    """Autoregressive rollout from a real starting frame. Returns
    ``(n_steps + 1, C, D, H, W)`` real-unit frames, index 0 = the given
    ``initial_frame`` unchanged, indices 1..n_steps = the model's own
    successive predictions fed back in as input."""
    net = load_dense_volume_operator(checkpoint, device=device)
    mean = np.array(checkpoint.meta["mean"], dtype=np.float32)
    std = np.array(checkpoint.meta["std"], dtype=np.float32)

    cur_n = torch.as_tensor((initial_frame[None] - mean) / std, device=device)
    out = [initial_frame.astype(np.float32)]
    with torch.no_grad():
        for _ in range(n_steps):
            cur_n = net(cur_n).y
            pred = cur_n[0].cpu().numpy() * std[0] + mean[0]
            out.append(pred)
    return np.stack(out, axis=0)
