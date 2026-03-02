from __future__ import annotations

"""Arena backend that trains any model registered in ``pinneaple_models``.

This backend is meant to be the "multi-model" workhorse for Arena:
  - loads bundle
  - builds a unified dict-batch (x_col/x_bc/.../x_data/y_data)
  - compiles physics loss if problem spec is present AND model supports it
  - trains with pinneaple_train.Trainer (optionally DDP)
  - exports predictions for later comparison
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from pinneaple_arena.bundle.loader import BundleData
from pinneaple_arena.pipeline.dataset_builder import build_from_bundle
from pinneaple_arena.registry import register_backend

from pinneaple_models.register_all import register_all
from pinneaple_models.registry import ModelRegistry
from pinneaple_models.adapters import select_adapter

from pinneaple_pinn.compiler.compile import compile_problem
from pinneaple_pinn.compiler.loss import LossWeights
from pinneaple_pinn.compiler.dataset import SingleBatchDataset, dict_collate

from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.losses import build_loss


@dataclass
class _XYFromBundle:
    """Preprocess step: chooses supervised x/y from a PINN-style batch."""

    prefer_data: bool = True

    def fit(self, batch_list):
        return self

    def apply(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(batch)

        # Prefer sensor/data supervision if available.
        if self.prefer_data and ("x_data" in out) and isinstance(out.get("x_data"), torch.Tensor):
            if out["x_data"].numel() > 0 and ("y_data" in out) and isinstance(out.get("y_data"), torch.Tensor):
                out["x"] = out["x_data"]
                out["y"] = out["y_data"]
                return out

        # Otherwise: supervised training is disabled (y absent). Still keep physics via x_col.
        out["x"] = out.get("x_col")
        # do NOT set y
        return out


@register_backend
class PinneapleModelsBackend:
    """Train any model from ModelRegistry."""

    name = "pinneaple_models"

    def train(self, bundle: BundleData, run_cfg: Dict[str, Any]) -> Dict[str, Any]:
        # Ensure global registry is populated.
        register_all()

        train_cfg = dict(run_cfg.get("train", {}))
        arena_cfg = dict(run_cfg.get("arena", {}))
        model_cfg = dict(run_cfg.get("model", {}))

        model_name = str(model_cfg.get("name") or "").strip()
        if not model_name:
            raise ValueError("run_cfg.model.name is required for pinneaple_models backend.")

        # Build dataset batch from bundle
        dev_str = str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        device = torch.device(dev_str)
        batch_like = build_from_bundle(
            bundle,
            n_collocation=int(arena_cfg.get("n_collocation", 4096)),
            n_boundary=int(arena_cfg.get("n_boundary", 2048)),
            n_data=int(arena_cfg.get("n_data", 2048)),
            device=None,  # Trainer moves
        )
        base_batch = batch_like.batch

        # DataLoader
        ds = SingleBatchDataset(base_batch)
        bs = int(train_cfg.get("batch_size", 1024))
        loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=dict_collate)
        # MVP: use same loader for val
        val_loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=dict_collate)

        # Build model
        kwargs = dict(model_cfg.get("kwargs", {}))
        model = ModelRegistry.build(model_name, **kwargs)
        spec = ModelRegistry.spec(model_name)
        adapter = select_adapter(spec)

        # Wrap model so Trainer can call forward_batch-based models without being modified.
        # We override forward(x) to ignore x and use the full batch when available.
        class _BatchWrapped(torch.nn.Module):
            def __init__(self, inner: torch.nn.Module):
                super().__init__()
                self.inner = inner
                self._last_batch: Optional[Dict[str, Any]] = None

            def set_batch(self, batch: Dict[str, Any]):
                self._last_batch = batch

            def forward(self, x: torch.Tensor):
                if self._last_batch is None:
                    return self.inner(x)
                return adapter.forward_batch(self.inner, self._last_batch)

        wrapped = _BatchWrapped(model)

        # Physics loss (optional)
        problem_spec = run_cfg.get("problem_spec", None)
        physics_loss_fn = None
        if problem_spec is not None and bool(spec.supports_physics_loss):
            # Allow overriding weights from config
            w_cfg = dict(train_cfg.get("physics_weights", {}))
            lw = LossWeights(**w_cfg) if w_cfg else None
            physics_loss_fn = compile_problem(problem_spec, weights=lw)

        loss = build_loss(
            problem_spec=problem_spec,
            model_capabilities={"supports_physics_loss": bool(spec.supports_physics_loss)},
            weights=dict(train_cfg.get("weights", {})),
            supervised_kind=str(train_cfg.get("supervised_kind", "mse")),
            physics_loss_fn=physics_loss_fn,
        )

        # Trainer config
        cfg = TrainConfig(
            epochs=int(train_cfg.get("epochs", 50)),
            lr=float(train_cfg.get("lr", 1e-3)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
            grad_clip=float(train_cfg.get("grad_clip", 0.0)),
            amp=bool(train_cfg.get("amp", False)),
            device=dev_str,
            seed=train_cfg.get("seed", None),
            deterministic=bool(train_cfg.get("deterministic", False)),
            log_dir=str(train_cfg.get("log_dir", "runs")),
            run_name=str(train_cfg.get("run_name", model_name)),
            save_best=bool(train_cfg.get("save_best", True)),
            ddp=bool(train_cfg.get("ddp", False)),
            ddp_backend=str(train_cfg.get("ddp_backend", "nccl")),
            ddp_find_unused_parameters=bool(train_cfg.get("ddp_find_unused_parameters", False)),
        )

        # Use preprocess to select supervised x/y. Also inject batch into wrapper per step.
        pre = _XYFromBundle(prefer_data=True)

        def loss_fn(model_mod: torch.nn.Module, y_hat: torch.Tensor, batch: Dict[str, Any]):
            # ensure wrapper sees full batch
            if hasattr(model_mod, "set_batch"):
                model_mod.set_batch(batch)  # type: ignore[attr-defined]
            return loss(model_mod, y_hat, batch)

        trainer = Trainer(wrapped, loss_fn=loss_fn, preprocess=pre)
        history = trainer.fit(loader, val_loader, cfg)

        # Export predictions on "test" points (prefer sensors)
        wrapped.eval()
        with torch.no_grad():
            x_test = base_batch.get("x_data")
            y_test = base_batch.get("y_data")
            if isinstance(x_test, torch.Tensor) and x_test.numel() > 0 and isinstance(y_test, torch.Tensor) and y_test.numel() > 0:
                x_eval = x_test.to(device)
                y_eval = y_test.detach().cpu().numpy()
            else:
                x_eval = base_batch["x_col"].to(device)
                y_eval = None

            # Use wrapper without batch context (it's fine for pointwise models), but keep last batch to allow forward_batch.
            wrapped.set_batch({**base_batch, "x": x_eval})
            y_hat = wrapped(x_eval)
            if hasattr(y_hat, "y"):
                y_hat_t = y_hat.y  # type: ignore
            else:
                y_hat_t = y_hat
            y_hat_np = y_hat_t.detach().cpu().numpy()

        out_dir = Path(str(train_cfg.get("out_dir", cfg.log_dir))) / str(cfg.run_name)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / "predictions.npz", y_hat=y_hat_np, y=(y_eval if y_eval is not None else y_hat_np))

        return {
            "model": wrapped,
            "model_name": model_name,
            "history": history,
            "out_dir": str(out_dir),
        }
