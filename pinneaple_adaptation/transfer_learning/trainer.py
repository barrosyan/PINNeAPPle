"""TransferTrainer — fine-tuning loop for physics-informed neural networks.

Key design decisions
--------------------
* The source model is **deep-copied** on construction so the caller's
  original weights are never modified.
* ``prepare()`` must be called before ``finetune()`` or
  ``progressive_unfreeze()``; it applies the freezing strategy from
  :class:`~pinneaple_transfer.config.TransferConfig`.
* ``target_physics_fn`` follows the same contract as ``loss_fn`` in the
  main :class:`~pinneaple_train.trainer.Trainer`:
  ``callable(model, batch) -> Tensor | dict``.
  If it returns a dict it **must** contain a ``"total"`` key.
* History entries are plain dicts so they are JSON-serialisable.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .config import TransferConfig
from .freeze import (
    freeze_all_except,
    freeze_layers,
    layer_lr_groups,
    unfreeze_layers,
    count_trainable,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_loss(loss_out: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Normalise loss_fn output to ``(total_tensor, component_dict)``."""
    if isinstance(loss_out, torch.Tensor):
        return loss_out, {"total": float(loss_out.item())}
    if isinstance(loss_out, dict):
        if "total" not in loss_out:
            raise ValueError(
                "target_physics_fn returned a dict without a 'total' key."
            )
        components = {
            k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in loss_out.items()
        }
        return loss_out["total"], components
    raise TypeError(
        "target_physics_fn must return torch.Tensor or dict[str, Tensor]."
    )


def _move_data(data: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Recursively move tensors in a batch dict to *device*."""
    out: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = _move_data(v, device)
        else:
            out[k] = v
    return out


def _set_seed(seed: int) -> None:
    if seed >= 0:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TransferTrainer:
    """Transfer-learning trainer for PINNs.

    The trainer takes a **pre-trained source model** and a
    :class:`~pinneaple_transfer.config.TransferConfig`, deep-copies the
    model, then exposes :meth:`prepare`, :meth:`finetune`, and
    :meth:`progressive_unfreeze` to carry out the adaptation.

    Parameters
    ----------
    source_model:
        A trained ``nn.Module``.  It is deep-copied immediately so that the
        caller retains the original weights unchanged.
    config:
        Transfer hyperparameters and strategy specification.

    Example
    -------
    >>> cfg = TransferConfig(strategy="finetune", epochs=200, finetune_lr=1e-5)
    >>> trainer = TransferTrainer(source_model, cfg)
    >>> model = trainer.prepare()
    >>> result = trainer.finetune(
    ...     target_physics_fn=my_burgers_loss,
    ...     target_data={"x": x_new, "y": u_new},
    ... )
    >>> fine_tuned = result["model"]
    """

    def __init__(self, source_model: nn.Module, config: TransferConfig) -> None:
        _set_seed(config.seed)
        self.config = config
        self.device = torch.device(config.device)
        # Deep-copy so the original model is never mutated.
        self.model: nn.Module = copy.deepcopy(source_model).to(self.device)
        self._prepared = False

    # ------------------------------------------------------------------
    # Prepare
    # ------------------------------------------------------------------

    def prepare(self) -> nn.Module:
        """Apply the freezing strategy defined in :attr:`config`.

        Must be called before :meth:`finetune` or
        :meth:`progressive_unfreeze`.

        Returns
        -------
        nn.Module
            The (in-place modified) model, ready for fine-tuning.
        """
        cfg = self.config
        strategy = cfg.strategy

        if strategy == "finetune":
            # All parameters trainable; use finetune_lr globally.
            for p in self.model.parameters():
                p.requires_grad_(True)

        elif strategy == "feature_extract":
            # Only the head / adapter layers are trained.
            freeze_all_except(self.model, cfg.unfreeze_prefix)

        elif strategy == "partial_freeze":
            # Explicit prefix-based freeze; everything else remains trainable.
            for p in self.model.parameters():
                p.requires_grad_(True)
            freeze_layers(self.model, cfg.freeze_prefix)

        elif strategy == "progressive":
            # Start with a fully-frozen backbone; warmup phase happens inside
            # finetune() or progressive_unfreeze().
            freeze_all_except(self.model, [])  # freeze everything

        self._prepared = True
        return self.model

    # ------------------------------------------------------------------
    # Fine-tuning loop
    # ------------------------------------------------------------------

    def finetune(
        self,
        target_physics_fn: Callable[[nn.Module, Dict[str, Any]], Any],
        target_data: Optional[Dict[str, torch.Tensor]] = None,
        n_epochs: Optional[int] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Run the fine-tuning loop against a target physics loss.

        Parameters
        ----------
        target_physics_fn:
            ``callable(model, batch) -> Tensor | dict``.  The batch dict
            follows the standard pinneaple convention (``x_col``, ``x_bc``,
            ``y_bc``, …).  For simple use-cases the function can ignore the
            batch and generate collocation points internally.
        target_data:
            Optional supervised data for the target domain.
            Expected keys: ``"x"`` and ``"y"`` (both ``torch.Tensor``).
            When provided, a data-fit MSE loss is added and weighted by
            ``config.data_weight``.
        n_epochs:
            Override ``config.epochs`` for this call.
        callbacks:
            List of callback objects.  Each callback may implement any
            combination of:
            - ``on_epoch_end(epoch, logs)``
            - ``on_train_end(logs)``

        Returns
        -------
        dict
            ``{"model": nn.Module, "history": List[dict], "metrics": dict}``

            * ``model`` — the fine-tuned model (same object as
              :attr:`model`).
            * ``history`` — one dict per epoch with keys ``epoch``,
              ``loss_total``, ``loss_physics``, ``loss_data``,
              ``epoch_sec``, plus any component keys returned by
              ``target_physics_fn``.
            * ``metrics`` — final-epoch summary.
        """
        if not self._prepared:
            raise RuntimeError(
                "Call TransferTrainer.prepare() before finetune()."
            )

        cfg = self.config
        epochs = n_epochs if n_epochs is not None else cfg.epochs
        callbacks = callbacks or []

        # Build optimizer with per-layer LR if scale_dict is provided.
        if cfg.layer_lr_scale:
            param_groups = layer_lr_groups(
                self.model, cfg.finetune_lr, cfg.layer_lr_scale
            )
        else:
            lr = cfg.finetune_lr if cfg.strategy == "finetune" else cfg.base_lr
            param_groups = [
                {"params": [p for p in self.model.parameters() if p.requires_grad],
                 "lr": lr}
            ]

        optimizer = torch.optim.Adam(param_groups)

        # Move target data to device once.
        td: Optional[Dict[str, torch.Tensor]] = None
        if target_data is not None:
            td = {k: v.to(self.device) for k, v in target_data.items()}

        history: List[Dict[str, Any]] = []
        progressive = cfg.strategy == "progressive"

        for epoch in range(epochs):
            t0 = time.time()

            # Progressive warmup: unfreeze backbone after warmup_epochs.
            if progressive and epoch == cfg.warmup_epochs and cfg.warmup_epochs > 0:
                if cfg.unfreeze_prefix:
                    unfreeze_layers(self.model, cfg.unfreeze_prefix)
                else:
                    # Unfreeze everything if no specific prefixes given.
                    for p in self.model.parameters():
                        p.requires_grad_(True)
                # Rebuild optimizer to include newly unfrozen params.
                if cfg.layer_lr_scale:
                    param_groups = layer_lr_groups(
                        self.model, cfg.base_lr, cfg.layer_lr_scale
                    )
                else:
                    param_groups = [
                        {"params": [p for p in self.model.parameters() if p.requires_grad],
                         "lr": cfg.base_lr}
                    ]
                optimizer = torch.optim.Adam(param_groups)

            self.model.train()
            optimizer.zero_grad(set_to_none=True)

            # --- Physics loss ---
            phys_out = target_physics_fn(self.model, {})
            phys_loss, phys_components = _parse_loss(phys_out)
            loss = cfg.physics_weight * phys_loss

            # --- Data loss (optional) ---
            data_loss_val = 0.0
            if td is not None:
                x_d = td["x"]
                y_d = td["y"]
                y_hat = self.model(x_d)
                # Unwrap named-tuple / dataclass outputs.
                if not isinstance(y_hat, torch.Tensor):
                    for attr in ("y", "pred", "logits", "out"):
                        if hasattr(y_hat, attr):
                            candidate = getattr(y_hat, attr)
                            if isinstance(candidate, torch.Tensor):
                                y_hat = candidate
                                break
                data_loss = nn.functional.mse_loss(y_hat, y_d)
                loss = loss + cfg.data_weight * data_loss
                data_loss_val = float(data_loss.item())

            loss.backward()
            optimizer.step()

            epoch_sec = time.time() - t0
            logs: Dict[str, Any] = {
                "epoch": epoch,
                "loss_total": float(loss.item()),
                "loss_physics": float(phys_loss.item()),
                "loss_data": data_loss_val,
                "epoch_sec": epoch_sec,
                **{f"phys_{k}": v for k, v in phys_components.items() if k != "total"},
            }
            history.append(logs)

            for cb in callbacks:
                if hasattr(cb, "on_epoch_end"):
                    cb.on_epoch_end(epoch, logs)

        final_logs = history[-1] if history else {}
        for cb in callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_train_end(final_logs)

        metrics = {
            "final_loss": final_logs.get("loss_total"),
            "final_physics_loss": final_logs.get("loss_physics"),
            "final_data_loss": final_logs.get("loss_data"),
            "trainable_params": count_trainable(self.model),
        }

        return {"model": self.model, "history": history, "metrics": metrics}

    # ------------------------------------------------------------------
    # Progressive unfreeze
    # ------------------------------------------------------------------

    def progressive_unfreeze(
        self,
        target_physics_fn: Callable[[nn.Module, Dict[str, Any]], Any],
        unfreeze_schedule: List[Tuple[int, List[str]]],
        target_data: Optional[Dict[str, torch.Tensor]] = None,
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Fine-tune with a custom epoch-level layer-unfreezing schedule.

        The total number of epochs is determined by the last entry in
        *unfreeze_schedule* (plus ``config.epochs`` after the final unfreeze
        event).

        Parameters
        ----------
        target_physics_fn:
            Same contract as in :meth:`finetune`.
        unfreeze_schedule:
            A list of ``(epoch, prefixes)`` pairs, **sorted by epoch**.
            At each specified epoch the listed prefixes are unfrozen and the
            optimizer is rebuilt.  Example::

                [(0, ["head"]), (50, ["trunk.2"]), (150, ["trunk.0", "trunk.1"])]

        target_data:
            Optional ``{"x": Tensor, "y": Tensor}`` supervised data.
        callbacks:
            Callback objects (``on_epoch_end``, ``on_train_end``).

        Returns
        -------
        dict
            ``{"model": nn.Module, "history": List[dict], "metrics": dict}``
        """
        if not self._prepared:
            raise RuntimeError(
                "Call TransferTrainer.prepare() before progressive_unfreeze()."
            )

        # Sort schedule defensively.
        schedule = sorted(unfreeze_schedule, key=lambda t: t[0])

        # Determine total epochs: last schedule epoch + config.epochs.
        last_epoch = schedule[-1][0] if schedule else 0
        total_epochs = last_epoch + self.config.epochs
        callbacks = callbacks or []

        td: Optional[Dict[str, torch.Tensor]] = None
        if target_data is not None:
            td = {k: v.to(self.device) for k, v in target_data.items()}

        cfg = self.config
        # Build initial optimizer (only currently trainable params).
        optimizer = self._build_optimizer()

        history: List[Dict[str, Any]] = []
        schedule_idx = 0

        for epoch in range(total_epochs):
            t0 = time.time()

            # Apply any scheduled unfreezes at this epoch.
            rebuilt = False
            while schedule_idx < len(schedule) and schedule[schedule_idx][0] <= epoch:
                _, prefixes = schedule[schedule_idx]
                unfreeze_layers(self.model, prefixes)
                schedule_idx += 1
                rebuilt = True

            if rebuilt:
                optimizer = self._build_optimizer()

            self.model.train()
            optimizer.zero_grad(set_to_none=True)

            phys_out = target_physics_fn(self.model, {})
            phys_loss, phys_components = _parse_loss(phys_out)
            loss = cfg.physics_weight * phys_loss

            data_loss_val = 0.0
            if td is not None:
                x_d = td["x"]
                y_d = td["y"]
                y_hat = self.model(x_d)
                if not isinstance(y_hat, torch.Tensor):
                    for attr in ("y", "pred", "logits", "out"):
                        if hasattr(y_hat, attr):
                            candidate = getattr(y_hat, attr)
                            if isinstance(candidate, torch.Tensor):
                                y_hat = candidate
                                break
                data_loss = nn.functional.mse_loss(y_hat, y_d)
                loss = loss + cfg.data_weight * data_loss
                data_loss_val = float(data_loss.item())

            loss.backward()
            optimizer.step()

            epoch_sec = time.time() - t0
            logs: Dict[str, Any] = {
                "epoch": epoch,
                "loss_total": float(loss.item()),
                "loss_physics": float(phys_loss.item()),
                "loss_data": data_loss_val,
                "epoch_sec": epoch_sec,
                **{f"phys_{k}": v for k, v in phys_components.items() if k != "total"},
            }
            history.append(logs)

            for cb in callbacks:
                if hasattr(cb, "on_epoch_end"):
                    cb.on_epoch_end(epoch, logs)

        final_logs = history[-1] if history else {}
        for cb in callbacks:
            if hasattr(cb, "on_train_end"):
                cb.on_train_end(final_logs)

        metrics = {
            "final_loss": final_logs.get("loss_total"),
            "final_physics_loss": final_logs.get("loss_physics"),
            "final_data_loss": final_logs.get("loss_data"),
            "trainable_params": count_trainable(self.model),
        }

        return {"model": self.model, "history": history, "metrics": metrics}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Construct an Adam optimiser from the current trainable params."""
        cfg = self.config
        if cfg.layer_lr_scale:
            param_groups = layer_lr_groups(
                self.model, cfg.base_lr, cfg.layer_lr_scale
            )
        else:
            param_groups = [
                {"params": [p for p in self.model.parameters() if p.requires_grad],
                 "lr": cfg.base_lr}
            ]
        return torch.optim.Adam(param_groups)
