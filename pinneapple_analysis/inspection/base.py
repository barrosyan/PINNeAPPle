"""pinneapple_analysis.inspection.base -- shared PINN backbone and generic
training wrapper for the inspection/NDE modalities in this package.

``InspectionPINN`` is intentionally a thin MLP, following the exact
constructor/forward pattern of
``pinneapple_neural.architectures.pinns.vanilla.VanillaPINN`` (this
package does not need inverse-parameter support or anything else
VanillaPINN doesn't already offer, so it is not subclassed directly --
duplicating its small, stable contract keeps this package's only
dependency on the neural stack at the ``PINNBase`` level, per
``pinneapple_neural/architectures/pinns/base.py``'s own "single source of
truth" contract: ``forward(*inputs, **kwargs) -> PINNOutput``,
``predict(*inputs) -> Tensor``, autograd helpers, checkpoint I/O, and
ONNX/TorchScript export are all inherited for free).

``train_generic_inspection_pinn`` is a thin wrapper around
``pinneapple_neural.trainer.trainer.Trainer`` (the same Trainer used
throughout the rest of the codebase, see
``examples/trainer/03_physics_aware_pinn_ode.py`` and
``examples/end_to_end/03_pinn_physics_loss_training.py`` for the pattern
being followed here): it turns a plain dict of numpy/torch arrays into
the ``{"x", "y", "x_col"}``-keyed ``DataLoader`` batches Trainer expects,
wires an optional physics residual through
``pinneapple_neural.trainer.losses.CombinedLoss`` +
``PhysicsLossHook``, and calls ``Trainer.fit``. None of the six modality
modules in this package hand-roll a training loop -- they all end up
here.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pinneapple_neural.architectures.pinns.base import PINNBase, PINNOutput
from pinneapple_neural.trainer.trainer import Trainer, TrainConfig
from pinneapple_neural.trainer.losses import CombinedLoss, SupervisedLoss, PhysicsLossHook
from pinneapple_neural.trainer.metrics import default_metrics

PhysicsLossFn = Callable[[nn.Module, Dict[str, Any]], Tuple[torch.Tensor, Dict[str, Any]]]


def _act(name: str) -> nn.Module:
    name = (name or "tanh").lower()
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}.get(name, nn.Tanh())


class InspectionPINN(PINNBase):
    """Generic coordinate -> inspection-signal MLP, physics-informed via an
    externally supplied residual (see each modality module's
    ``make_*_physics_loss`` factory).

    Constructor mirrors ``VanillaPINN`` exactly (``in_dim``, ``out_dim``,
    ``hidden``, ``activation``) plus a ``modality`` label carried purely
    for bookkeeping (checkpoint metadata / repr) -- it has no effect on
    the network itself. Every modality module in this package builds one
    of these with a modality-appropriate ``(in_dim, out_dim)`` (e.g.
    ``(2, 2)`` for a complex field over an (r, z) plane, ``(2, 1)`` for a
    real scalar field over (x, t), etc.) and trains it via
    ``train_generic_inspection_pinn`` with a modality-specific physics
    residual.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: List[int] = (64, 64, 64),
        activation: str = "tanh",
        *,
        modality: str = "generic",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        act = _act(activation)

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.modality = str(modality)

        dims = [self.in_dim, *list(hidden), self.out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act)
        self.net = nn.Sequential(*layers)

    def _concat_inputs(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(inputs) == 0:
            raise ValueError("InspectionPINN.forward expected at least 1 input tensor.")
        if len(inputs) == 1:
            x = inputs[0]
            if x.ndim == 1:
                x = x[:, None]
            return x
        cols = []
        for t in inputs:
            if t.ndim == 1:
                t = t[:, None]
            cols.append(t)
        return torch.cat(cols, dim=1)

    def forward(
        self,
        *inputs: torch.Tensor,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
    ) -> PINNOutput:
        x = self._concat_inputs(inputs)

        if (physics_fn is not None and physics_data is not None) and (not x.requires_grad):
            x = x.requires_grad_(True)

        y = self.net(x)

        z0 = torch.zeros((), device=y.device, dtype=y.dtype)
        losses: Dict[str, torch.Tensor] = {"total": z0}

        if physics_fn is not None and physics_data is not None:
            total_phys, comps = physics_fn(self, physics_data)
            losses["physics"] = total_phys
            for k, v in comps.items():
                if k == "total":
                    continue
                losses[k] = torch.as_tensor(v, device=y.device, dtype=y.dtype)
            losses["total"] = losses["total"] + losses["physics"]

        return PINNOutput(y=y, losses=losses, extras={})


def _as_tensor(a: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if torch.is_tensor(a):
        return a.to(dtype)
    return torch.as_tensor(np.asarray(a), dtype=dtype)


class _InspectionDataset(Dataset):
    """Zips supervised (x, y) pairs with a (possibly differently-sized)
    pool of physics collocation points `x_col`, cycling through the pool
    via modulo indexing so DataLoader batching just works regardless of
    how `x`/`x_col` sizes compare."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, x_col: Optional[torch.Tensor]):
        self.x = x
        self.y = y
        self.x_col = x_col

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {"x": self.x[idx], "y": self.y[idx]}
        if self.x_col is not None:
            j = idx % self.x_col.shape[0]
            item["x_col"] = self.x_col[j]
        return item


def train_generic_inspection_pinn(
    model: InspectionPINN,
    synthetic_data: Dict[str, Any],
    physics_loss_fn: Optional[PhysicsLossFn] = None,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    val_fraction: float = 0.2,
    w_supervised: float = 1.0,
    w_physics: float = 1.0,
    device: str = "cpu",
    log_dir: str = "runs/inspection",
    run_name: str = "inspection_pinn",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Train an ``InspectionPINN`` on a synthetic-data dict via
    ``pinneapple_neural.trainer.trainer.Trainer`` -- the same Trainer, the
    same ``CombinedLoss(SupervisedLoss + PhysicsLossHook)`` pattern, used
    by every other physics-aware training example in this repo (see
    ``examples/trainer/03_physics_aware_pinn_ode.py``).

    Parameters
    ----------
    model : an ``InspectionPINN`` (or any ``PINNBase`` with a matching
        ``forward(x) -> PINNOutput`` contract).
    synthetic_data : dict with keys
        ``"x"``    -- (N, in_dim) supervised input coordinates
        ``"y"``    -- (N, out_dim) supervised targets
        ``"x_col"`` -- (M, in_dim) physics collocation points (optional;
                        falls back to reusing ``"x"`` when omitted).
    physics_loss_fn : callable ``(model, batch) -> (loss, comps)`` as
        expected by ``PhysicsLossHook``. Pass ``None`` for pure
        supervised training (no modality module in this package does,
        but the option exists e.g. for ablations).

    Returns
    -------
    The dict returned by ``Trainer.fit`` (``best_val``, ``best_path``,
    ``history``, ...).
    """
    x = _as_tensor(synthetic_data["x"])
    y = _as_tensor(synthetic_data["y"])
    if y.ndim == 1:
        y = y[:, None]
    x_col_raw = synthetic_data.get("x_col")
    x_col = _as_tensor(x_col_raw) if x_col_raw is not None else x

    n = x.shape[0]
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
    else:
        perm = torch.randperm(n)

    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 1
    n_val = min(n_val, n - 1) if n > 1 else 0
    val_idx = perm[:n_val] if n_val > 0 else perm[:1]
    train_idx = perm[n_val:] if n_val > 0 else perm

    train_ds = _InspectionDataset(x[train_idx], y[train_idx], x_col)
    val_ds = _InspectionDataset(x[val_idx], y[val_idx], x_col)

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)

    physics_hook = PhysicsLossHook(physics_loss_fn) if physics_loss_fn is not None else None
    combined = CombinedLoss(
        supervised=SupervisedLoss("mse"),
        physics=physics_hook,
        w_supervised=w_supervised,
        w_physics=w_physics,
    )

    def loss_fn(m: nn.Module, y_hat: Any, batch: Dict[str, Any]):
        return combined(m, y_hat, batch)

    trainer = Trainer(model=model, loss_fn=loss_fn, metrics=default_metrics())
    cfg = TrainConfig(
        epochs=epochs,
        lr=lr,
        device=device,
        log_dir=log_dir,
        run_name=run_name,
        seed=seed,
        save_best=True,
    )
    return trainer.fit(train_loader, val_loader, cfg)
