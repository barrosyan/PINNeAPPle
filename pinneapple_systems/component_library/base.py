"""pinneapple_systems.component_library.base — ``ComponentModel``: a
``BaseModel`` that additionally knows its physical constraint (a
``Physics``) and offers single-call ``.fit()``/``.evaluate()``
convenience, closing the two real gaps found against a plain
``BaseModel``/``PINNBase``:

1. No unified train-this-model-now entry point. ``.fit()`` here is
   deliberately thin — it builds a data+physics ``loss_fn`` and delegates
   to the existing ``pinneapple_neural.trainer.Trainer``, which already
   has a stronger set of loss balancers (ReLoBRaLo/SoftAdapt/PCGrad/
   AugmentedLagrangian) than a bespoke EMA balance would be. No training
   loop is duplicated here.
2. No metric computation on the model itself. ``.evaluate()`` is
   genuinely new: RMSE/MAPE/R² against held-out targets.

Composition, not reimplementation: a ``ComponentModel`` wraps an existing
registered ``pinneapple_neural.architectures`` backbone (built via
``ModelRegistry.build``) rather than defining new network layers — a
"pipe" or "pump" is a *physical role* attached to an architecture, not a
new architecture family.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from pinneapple_neural.architectures.base import BaseModel
from pinneapple_neural.trainer.trainer import Trainer, TrainConfig

from .physics import Physics


def _unwrap(out: Any) -> torch.Tensor:
    return out.y if hasattr(out, "y") else out


class ComponentModel(BaseModel):
    """A named plant component: an existing architecture + an optional
    ``Physics`` constraint."""

    family = "component"

    def __init__(
        self,
        *,
        architecture: str,
        architecture_kwargs: Optional[Dict[str, Any]] = None,
        physics: Optional[Physics] = None,
    ):
        super().__init__()
        from pinneapple_neural.architectures.registry import ModelRegistry

        self.backbone = ModelRegistry.build(architecture, **(architecture_kwargs or {}))
        self.physics = physics
        self._architecture = architecture
        self._architecture_kwargs = dict(architecture_kwargs or {})

    def forward(self, *args: Any, **kwargs: Any):
        return self.backbone(*args, **kwargs)

    def forward_batch(self, batch: Dict[str, Any]):
        return self.backbone.forward_batch(batch)

    # ------------------------------------------------------------------
    # Training / evaluation
    # ------------------------------------------------------------------

    def _loss_fn(self, data_weight: float, physics_weight: float):
        physics = self.physics

        def loss_fn(model: Any, y_hat: torch.Tensor, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
            y_hat_t = _unwrap(y_hat)
            ref = y_hat_t if torch.is_tensor(y_hat_t) else next(model.parameters())
            total = ref.new_zeros(())
            parts: Dict[str, torch.Tensor] = {}

            y = batch.get("y")
            if y is not None:
                data_loss = F.mse_loss(y_hat_t, y)
                parts["data"] = data_loss
                total = total + data_weight * data_loss

            if physics is not None:
                phys_loss: Optional[torch.Tensor] = None
                try:
                    phys_out = physics(model, y_hat_t, batch)
                    phys_loss = phys_out["total"] if isinstance(phys_out, dict) else phys_out
                except NotImplementedError:
                    # Graceful physics-failure degradation: fall back to a
                    # data-only step rather than crashing training, but
                    # loudly record that it happened (never a silent zero
                    # that would look like "physics satisfied").
                    parts["physics_error"] = ref.new_tensor(float("nan"))
                if phys_loss is not None:
                    parts["physics"] = phys_loss
                    total = total + physics_weight * phys_loss

            parts["total"] = total
            return parts

        return loss_fn

    def fit(
        self,
        train_loader,
        val_loader=None,
        *,
        epochs: int = 100,
        lr: float = 1e-3,
        data_weight: float = 1.0,
        physics_weight: float = 1.0,
        device: Optional[str] = None,
        run_name: str = "component_fit",
        **trainer_kwargs: Any,
    ) -> Dict[str, Any]:
        """Single-call training, delegating to the shared ``Trainer``.
        Returns ``Trainer.fit``'s result dict (``best_val``, ``best_path``,
        ``history``, ...)."""
        cfg = TrainConfig(
            epochs=epochs,
            lr=lr,
            device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
            run_name=run_name,
            **trainer_kwargs,
        )
        trainer = Trainer(self, self._loss_fn(data_weight, physics_weight))
        return trainer.fit(train_loader, val_loader if val_loader is not None else train_loader, cfg)

    def evaluate(self, coords: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """RMSE / MAPE / R² against ``targets``."""
        self.eval()
        with torch.no_grad():
            pred = _unwrap(self(coords))
        err = pred - targets
        rmse = torch.sqrt((err ** 2).mean()).item()
        denom = targets.abs().clamp_min(1e-8)
        mape = (err.abs() / denom).mean().item() * 100.0
        ss_res = (err ** 2).sum()
        ss_tot = ((targets - targets.mean()) ** 2).sum().clamp_min(1e-12)
        r2 = (1 - ss_res / ss_tot).item()
        return {"rmse": rmse, "mape": mape, "r2": r2}

    # ------------------------------------------------------------------
    # Checkpointing: reconstructable from (component name, init kwargs)
    # alone, the way pinneapple_hub.hub.from_pretrained already
    # reconstructs a plain ModelRegistry architecture from
    # (architecture name, architecture_config) — BaseModel.load_checkpoint
    # records `class_name` but still makes the caller supply matching
    # **init_kwargs by hand; this closes that gap for components built via
    # ComponentRegistry.build().
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        meta = dict(metadata or {})
        meta.setdefault("component_name", getattr(self, "_component_name", None))
        meta.setdefault("init_kwargs", getattr(self, "_init_kwargs", {}))
        return super().save_checkpoint(path, metadata=meta)

    @classmethod
    def load_checkpoint(cls, path: str, **init_kwargs: Any) -> "ComponentModel":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = ckpt.get("metadata", {}) or {}
        component_name = meta.get("component_name")
        if component_name and not init_kwargs:
            from .registry import ComponentRegistry

            model = ComponentRegistry.build(component_name, **meta.get("init_kwargs", {}))
            model.load_state_dict(ckpt["state_dict"])
            return model
        return super().load_checkpoint(path, **init_kwargs)
