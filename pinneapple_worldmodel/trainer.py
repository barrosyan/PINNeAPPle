"""World model training loop.

:class:`WorldModelTrainer` wraps :class:`~pinneapple_train.Trainer` and adds:

* **Rollout loss** — multi-step auto-regressive loss to improve long-horizon
  stability.  Controlled by ``WorldModelConfig.rollout_steps``.
* **Physics consistency loss** — optional penalty ensuring the predicted field
  satisfies conservation laws, plugged in via ``pinneapple_validate``.
* **Validation** — per-epoch evaluation on a held-out test set with field-wise
  RMSE and relative-L2 metrics.

Training data format expected by the Trainer::

    batch = {
        "state_t":   (B, C, *grid),
        "state_tp1": (B, C, *grid),
        "context":   (B, context_dim),
        "params":    (B, n_params),
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, random_split

from .model import PhysicsWorldModel, WorldModelConfig
from .dataset import WorldModelDataset


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class WorldModelLoss:
    """Multi-step rollout loss for world model training.

    Parameters
    ----------
    rollout_steps : int
        Number of steps to unroll during training.  1 = standard next-step.
        Higher values improve long-horizon stability but cost more memory.
    rollout_weight : float
        Weight applied to the rollout loss component (beyond step 1).
    physics_loss_fn : optional callable(state_pred, state_true) → Tensor
        Additional physics consistency penalty.
    physics_weight : float
        Weight of the physics loss.
    """

    def __init__(
        self,
        rollout_steps: int = 1,
        rollout_weight: float = 0.5,
        physics_loss_fn: Optional[Callable] = None,
        physics_weight: float = 0.01,
    ) -> None:
        self.rollout_steps = rollout_steps
        self.rollout_weight = rollout_weight
        self.physics_loss_fn = physics_loss_fn
        self.physics_weight = physics_weight

    def __call__(
        self,
        model: PhysicsWorldModel,
        pred: Tensor,
        batch: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Compute loss dict compatible with :class:`~pinneapple_train.Trainer`.

        Parameters
        ----------
        model : PhysicsWorldModel
        pred : Tensor ``(B, C, *grid)`` — one-step prediction from Trainer.
        batch : dict from WorldModelDataset.

        Returns
        -------
        dict with keys ``"total"``, ``"next_step"``, optionally ``"rollout"``,
        ``"physics"``.
        """
        target = batch["state_tp1"]
        context = batch.get("context")

        # Next-step MSE
        next_step_loss = torch.mean((pred - target) ** 2)

        losses: Dict[str, Tensor] = {"next_step": next_step_loss}
        total = next_step_loss

        # Multi-step rollout loss
        if self.rollout_steps > 1:
            rollout_loss = self._rollout_loss(model, batch, context)
            losses["rollout"] = rollout_loss
            total = total + self.rollout_weight * rollout_loss

        # Physics consistency
        if self.physics_loss_fn is not None:
            phys_loss = self.physics_loss_fn(pred, target)
            losses["physics"] = phys_loss
            total = total + self.physics_weight * phys_loss

        losses["total"] = total
        return losses

    def _rollout_loss(
        self,
        model: PhysicsWorldModel,
        batch: Dict[str, Tensor],
        context: Optional[Tensor],
    ) -> Tensor:
        """Unroll from state_t for (rollout_steps - 1) additional steps."""
        state = batch["state_t"]
        target = batch["state_tp1"]
        loss = torch.tensor(0.0, device=state.device)

        # First step already computed in next_step_loss; start from step 2
        with torch.no_grad():
            state_next = model(state, context)

        for step in range(2, self.rollout_steps + 1):
            state_next = model(state_next.detach(), context)
            # Geometric decay: later steps weighted less
            w = self.rollout_weight ** (step - 1)
            loss = loss + w * torch.mean((state_next - target) ** 2)

        return loss / max(self.rollout_steps - 1, 1)


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------

@dataclass
class WorldModelTrainConfig:
    """Training hyperparameters for :class:`WorldModelTrainer`.

    Parameters
    ----------
    epochs : int
    lr : float — initial learning rate (AdamW).
    weight_decay : float
    batch_size : int
    val_fraction : float — fraction of dataset held out for validation.
    device : str
    amp : bool — mixed precision (CUDA only).
    grad_clip : float — gradient clipping norm (0 = disabled).
    rollout_steps : int — passed to WorldModelLoss.
    rollout_weight : float
    physics_weight : float
    patience : int — early stopping patience (0 = disabled).
    save_best : str or None — path to save best checkpoint.
    log_every : int — log every N epochs.
    """
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    val_fraction: float = 0.1
    device: str = "cpu"
    amp: bool = False
    grad_clip: float = 1.0
    rollout_steps: int = 1
    rollout_weight: float = 0.5
    physics_weight: float = 0.0
    patience: int = 20
    save_best: Optional[str] = None
    log_every: int = 10


# ---------------------------------------------------------------------------
# WorldModelTrainer
# ---------------------------------------------------------------------------

class WorldModelTrainer:
    """Train a :class:`PhysicsWorldModel` on a :class:`WorldModelDataset`.

    Integrates with ``pinneapple_train`` for optimizer / AMP / checkpointing;
    wraps it with rollout-aware loss and physics validation hooks.

    Parameters
    ----------
    model : PhysicsWorldModel
    config : WorldModelTrainConfig
    physics_loss_fn : optional callable added to the loss (conservation laws, etc.)

    Example
    -------
    >>> trainer = WorldModelTrainer(model, WorldModelTrainConfig(epochs=200))
    >>> history = trainer.fit(dataset)
    >>> print(f"Best val loss: {min(h['val_total'] for h in history):.4f}")
    """

    def __init__(
        self,
        model: PhysicsWorldModel,
        config: WorldModelTrainConfig,
        *,
        physics_loss_fn: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.loss_fn = WorldModelLoss(
            rollout_steps=config.rollout_steps,
            rollout_weight=config.rollout_weight,
            physics_loss_fn=physics_loss_fn,
            physics_weight=config.physics_weight,
        )
        self.device = torch.device(config.device)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: WorldModelDataset,
        *,
        val_dataset: Optional[WorldModelDataset] = None,
    ) -> List[Dict[str, float]]:
        """Train the model and return the history list.

        Parameters
        ----------
        dataset : WorldModelDataset — training (+ optional validation split).
        val_dataset : optional separate validation dataset.

        Returns
        -------
        list of dicts, one per epoch, keys: epoch, train_total, val_total,
        train_next_step, (optional) train_rollout, (optional) train_physics.
        """
        cfg = self.config
        model = self.model.to(self.device)

        # Validation split
        if val_dataset is None and cfg.val_fraction > 0:
            n_val = max(1, int(len(dataset) * cfg.val_fraction))
            n_train = len(dataset) - n_val
            train_ds, val_ds = random_split(
                dataset, [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
        else:
            train_ds = dataset
            val_ds = val_dataset

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, pin_memory=(cfg.device != "cpu"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
            num_workers=0,
        ) if val_ds is not None else None

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01
        )
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and self.device.type == "cuda")

        history: List[Dict[str, float]] = []
        best_val = float("inf")
        patience_count = 0

        for epoch in range(1, cfg.epochs + 1):
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, scaler, cfg
            )
            scheduler.step()

            record: Dict[str, float] = {"epoch": float(epoch), **train_metrics}

            if val_loader is not None:
                val_metrics = self._eval_epoch(model, val_loader)
                record.update({f"val_{k}": v for k, v in val_metrics.items()})

                val_loss = record.get("val_total", float("inf"))
                if val_loss < best_val:
                    best_val = val_loss
                    patience_count = 0
                    if cfg.save_best:
                        self._save(model, optimizer, epoch, val_loss, cfg.save_best)
                else:
                    patience_count += 1

                if cfg.patience > 0 and patience_count >= cfg.patience:
                    print(f"[WorldModelTrainer] Early stopping at epoch {epoch} "
                          f"(best val={best_val:.4g})")
                    history.append(record)
                    break

            history.append(record)

            if epoch % cfg.log_every == 0 or epoch == cfg.epochs:
                self._log(epoch, cfg.epochs, record)

        return history

    # ------------------------------------------------------------------
    # Train / eval loops
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        model: PhysicsWorldModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler,
        cfg: WorldModelTrainConfig,
    ) -> Dict[str, float]:
        model.train()
        accum: Dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, Tensor) else v
                     for k, v in batch.items()}
            state_t  = batch["state_t"]
            context  = batch.get("context")

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=cfg.amp and self.device.type == "cuda"):
                pred = model(state_t, context)
                loss_dict = self.loss_fn(model, pred, batch)

            scaler.scale(loss_dict["total"]).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            for k, v in loss_dict.items():
                accum[f"train_{k}"] = accum.get(f"train_{k}", 0.0) + v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    @torch.no_grad()
    def _eval_epoch(
        self,
        model: PhysicsWorldModel,
        loader: DataLoader,
    ) -> Dict[str, float]:
        model.eval()
        accum: Dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, Tensor) else v
                     for k, v in batch.items()}
            state_t  = batch["state_t"]
            context  = batch.get("context")

            pred = model(state_t, context)
            loss_dict = self.loss_fn(model, pred, batch)

            # Extra metric: relative L2
            target = batch["state_tp1"]
            rel_l2 = (torch.norm(pred - target) / (torch.norm(target) + 1e-8)).item()
            loss_dict["rel_l2"] = torch.tensor(rel_l2)  # type: ignore[assignment]

            for k, v in loss_dict.items():
                accum[k] = accum.get(k, 0.0) + (v.item() if isinstance(v, Tensor) else v)
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in accum.items()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save(
        self,
        model: PhysicsWorldModel,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        val_loss: float,
        path: str,
    ) -> None:
        torch.save({
            "epoch": epoch,
            "val_loss": val_loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": self.config,
            "model_config": model.config,
            "n_fields": model.n_fields,
            "grid_shape": model.grid_shape,
        }, path)

    @staticmethod
    def load_checkpoint(path: str, map_location: str = "cpu") -> "PhysicsWorldModel":
        """Restore a model from a checkpoint saved by :meth:`fit`."""
        ckpt = torch.load(path, map_location=map_location)
        model = PhysicsWorldModel(
            ckpt["model_config"],
            n_fields=ckpt["n_fields"],
            grid_shape=ckpt["grid_shape"],
        )
        model.load_state_dict(ckpt["model"])
        return model

    def _log(self, epoch: int, total: int, record: Dict[str, float]) -> None:
        parts = [f"epoch={epoch}/{total}"]
        for k in ("train_total", "val_total", "val_rel_l2"):
            if k in record:
                parts.append(f"{k}={record[k]:.4g}")
        print("[WorldModelTrainer] " + "  ".join(parts))
