"""Unified Trainer with physics-aware losses, callbacks, and audit logging."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Callable, Union

import time
import torch
import torch.nn as nn

from .preprocess import PreprocessPipeline
from .callbacks import EarlyStopping, ModelCheckpoint

from .audit import set_seed, set_deterministic, RunLogger, env_fingerprint
from .checkpoint import Checkpoint, save_checkpoint

from .metrics import Metrics, Metric, default_metrics, MetricBundle


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    amp: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # auditability
    seed: Optional[int] = None
    deterministic: bool = False
    log_dir: str = "runs"
    run_name: str = "run"
    save_best: bool = True


class Trainer:
    """
    Unified trainer:
      - supports physics-aware losses via loss_fn(model, y_hat, batch)->dict
      - supports preprocess pipeline
      - supports callbacks (early stopping / checkpoint)
      - supports auditable logging + deterministic runs + best checkpoint

    Batches:
      - dict with "x" and optional "y"
      - or tuple (x, y)
    """
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[nn.Module, torch.Tensor, Dict[str, Any]], Dict[str, torch.Tensor]],
        metrics: Optional[Union[Metrics, List[Metric]]] = None,
        preprocess: Optional[PreprocessPipeline] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint: Optional[ModelCheckpoint] = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.preprocess = preprocess
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint

        if metrics is None:
            self.metrics_obj: Optional[Metrics] = MetricBundle(default_metrics())
        elif isinstance(metrics, list):
            self.metrics_obj = MetricBundle(metrics)
        else:
            self.metrics_obj = metrics

    def _xy_batch(self, batch: Any) -> Dict[str, Any]:
        if isinstance(batch, dict):
            return batch
        # tuple/list (x,y)
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return {"x": batch[0], "y": batch[1]}
        raise TypeError("Batch must be dict or (x,y) tuple")

    def _move(self, obj: Any, device: torch.device) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        return obj

    def _move_batch(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                out[k] = {kk: self._move(vv, device) for kk, vv in v.items()}
            else:
                out[k] = self._move(v, device)
        return out

    def _unwrap_pred(self, y_hat: Any, batch: Optional[Dict[str, Any]] = None) -> Any:
        """
        Convert ModelOutput / AEOutput / etc -> torch.Tensor (prediction).

        Rule of thumb:
          - if model returns AEOutput and batch has y:
              - if y matches z shape -> use z
              - elif y matches x_hat shape -> use x_hat
              - else -> default x_hat
          - otherwise try common attributes: y, pred, x_hat, recon, logits
        """
        # AEOutput-like
        if hasattr(y_hat, "x_hat") and hasattr(y_hat, "z"):
            x_hat = y_hat.x_hat
            z = y_hat.z

            y = None
            if batch is not None and isinstance(batch, dict):
                y = batch.get("y")

            if isinstance(y, torch.Tensor):
                # Prefer the head that matches target shape (ignoring batch dim issues)
                if z.shape == y.shape:
                    return z
                if x_hat.shape == y.shape:
                    return x_hat

                # Sometimes y is (B, latent_dim) and z is (B, latent_dim) but x_hat is (B, input_dim)
                # If last dim matches, use that.
                if z.ndim == y.ndim and z.shape[-1] == y.shape[-1]:
                    return z
                if x_hat.ndim == y.ndim and x_hat.shape[-1] == y.shape[-1]:
                    return x_hat

            # No y to compare: default to reconstruction
            return x_hat

        # PINNOutput / generic outputs
        for attr in ("y", "pred", "logits", "x_hat", "recon"):
            if hasattr(y_hat, attr):
                return getattr(y_hat, attr)

        return y_hat
 

    def fit(self, train_loader, val_loader, cfg: TrainConfig) -> Dict[str, Any]:
        if cfg.seed is not None:
            set_seed(cfg.seed)
        set_deterministic(cfg.deterministic)

        device = torch.device(cfg.device)
        self.model.to(device)

        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp) and device.type == "cuda")

        logger = RunLogger(out_dir=cfg.log_dir, run_name=cfg.run_name)
        logger.save_config(
            {
                "train_config": asdict(cfg),
                "env": env_fingerprint(),
                "model": self.model.__class__.__name__,
            }
        )

        # Fit preprocess on train (peek few batches)
        if self.preprocess is not None:
            peek = []
            for i, b in enumerate(train_loader):
                peek.append(self._xy_batch(b))
                if i >= 8:
                    break
            self.preprocess.fit(peek)

        best_val = float("inf")
        history: List[Dict[str, float]] = []

        for epoch in range(cfg.epochs):
            t_epoch = time.time()

            # -------- train
            self.model.train()
            tr_total = 0.0
            n_tr = 0

            for raw in train_loader:
                batch = self._xy_batch(raw)
                if self.preprocess is not None:
                    batch = self.preprocess.apply(batch)
                batch = self._move_batch(batch, device)

                x = batch["x"]
                y = batch.get("y")

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=bool(cfg.amp) and device.type == "cuda"):
                    y_hat_raw = self.model(x)
                    y_hat = self._unwrap_pred(y_hat_raw)
                    losses = self.loss_fn(self.model, y_hat, batch)
                    loss = losses["total"]

                scaler.scale(loss).backward()

                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

                scaler.step(opt)
                scaler.update()

                bs = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim > 0 else 1
                tr_total += float(loss.item()) * bs
                n_tr += bs

            train_total = tr_total / max(1, n_tr)

            # -------- val
            self.model.eval()
            va_total = 0.0
            n_va = 0
            yhat_acc, y_acc = [], []

            with torch.no_grad():
                for raw in val_loader:
                    batch = self._xy_batch(raw)
                    if self.preprocess is not None:
                        batch = self.preprocess.apply(batch)
                    batch = self._move_batch(batch, device)

                    x = batch["x"]
                    y = batch.get("y")

                    y_hat_raw = self.model(x)
                    y_hat = self._unwrap_pred(y_hat_raw)
                    losses = self.loss_fn(self.model, y_hat, batch)
                    loss = losses["total"]

                    bs = x.shape[0] if isinstance(x, torch.Tensor) and x.ndim > 0 else 1
                    va_total += float(loss.item()) * bs
                    n_va += bs

                    if y is not None:
                        yhat_acc.append(y_hat.detach().cpu())
                        y_acc.append(y.detach().cpu())

            val_total = va_total / max(1, n_va)

            rec: Dict[str, float] = {
                "epoch": float(epoch),
                "train_total": float(train_total),
                "val_total": float(val_total),
                "epoch_time_s": float(time.time() - t_epoch),
            }

            if yhat_acc and self.metrics_obj is not None:
                yh = torch.cat(yhat_acc, dim=0)
                yt = torch.cat(y_acc, dim=0)
                m = self.metrics_obj.compute(yh, yt)
                for k, v in m.items():
                    rec[f"val_{k}"] = float(v)

            logger.log(rec)
            history.append(rec)

            # optional checkpoint callback (your old one)
            if self.checkpoint is not None:
                saved = self.checkpoint.maybe_save(self.model, extra={}, logs=rec)

            # best checkpoint (audit style)
            if cfg.save_best and val_total < best_val:
                best_val = val_total
                ckpt = Checkpoint(
                    model_state=self.model.state_dict(),
                    optim_state=opt.state_dict(),
                    cfg={"train": asdict(cfg)},
                    meta={"best_val": float(best_val), "epoch": epoch},
                    normalizers=None,  # you can store preprocess scalers here later
                )
                save_checkpoint(path=f"{cfg.log_dir}/{cfg.run_name}.best.pt", ckpt=ckpt)

            if self.early_stopping is not None:
                self.early_stopping.update(rec)
                if self.early_stopping.stop:
                    break

        return {"history": history, "best_val": float(best_val), "best_path": f"{cfg.log_dir}/{cfg.run_name}.best.pt"}
