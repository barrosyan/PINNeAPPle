"""Unified Trainer with physics-aware losses, callbacks, and audit logging.

Key capabilities:
- loss_fn may return torch.Tensor OR dict with a mandatory 'total' key.
- optional DistributedDataParallel (DDP) support via TrainConfig.ddp=True.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, List, Callable, Union

import os
import time

import torch
import torch.nn as nn

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
except Exception:  # pragma: no cover
    dist = None
    DDP = None
    DistributedSampler = None

from .preprocess import PreprocessPipeline
from .callbacks import EarlyStopping, ModelCheckpoint
from .audit import set_seed, set_deterministic, RunLogger, env_fingerprint
from .checkpoint import Checkpoint, save_checkpoint
from .metrics import Metrics, Metric, default_metrics, MetricBundle


LossOut = Union[torch.Tensor, Dict[str, torch.Tensor]]


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

    # DistributedDataParallel
    ddp: bool = False
    ddp_backend: str = "nccl"  # "nccl" (GPU) or "gloo" (CPU)
    ddp_find_unused_parameters: bool = False


class Trainer:
    """Trainer for PINNs and general supervised models."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable[[nn.Module, torch.Tensor, Dict[str, Any]], LossOut],
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

    def _unwrap_pred(self, y_hat: Any, batch: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if isinstance(y_hat, torch.Tensor):
            return y_hat
        for attr in ("y", "pred", "logits", "x_hat", "recon", "out"):
            if hasattr(y_hat, attr):
                v = getattr(y_hat, attr)
                if isinstance(v, torch.Tensor):
                    return v
        raise TypeError("Model output is not a torch.Tensor and has no known tensor attribute.")

    def _parse_loss(self, loss_out: LossOut) -> Dict[str, torch.Tensor]:
        if isinstance(loss_out, torch.Tensor):
            return {"total": loss_out}
        if isinstance(loss_out, dict):
            if "total" not in loss_out:
                raise ValueError("loss_fn returned dict but missing 'total'.")
            return loss_out
        raise TypeError("loss_fn must return torch.Tensor or dict[str, torch.Tensor].")

    def _ddp_info(self) -> Dict[str, int]:
        return {
            "rank": int(os.environ.get("RANK", "0")),
            "world": int(os.environ.get("WORLD_SIZE", "1")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        }

    def _ddp_init(self, cfg: TrainConfig) -> Dict[str, int]:
        if dist is None:
            raise ImportError("torch.distributed is not available.")
        info = self._ddp_info()
        if not dist.is_initialized():
            dist.init_process_group(backend=cfg.ddp_backend)
        return info

    def _ddp_wrap(self, cfg: TrainConfig, info: Dict[str, int]) -> nn.Module:
        if DDP is None:
            raise ImportError("DistributedDataParallel is not available.")
        if cfg.device.startswith("cuda"):
            torch.cuda.set_device(info["local_rank"])
            return DDP(
                self.model,
                device_ids=[info["local_rank"]],
                find_unused_parameters=cfg.ddp_find_unused_parameters,
            )
        return DDP(self.model, find_unused_parameters=cfg.ddp_find_unused_parameters)

    def _maybe_rewrap_loader_ddp(self, loader, *, shuffle: bool):
        if DistributedSampler is None:
            return loader
        ds = loader.dataset
        return torch.utils.data.DataLoader(
            ds,
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=getattr(loader, "num_workers", 0),
            pin_memory=getattr(loader, "pin_memory", False),
            collate_fn=getattr(loader, "collate_fn", None),
            sampler=DistributedSampler(ds, shuffle=shuffle),
            drop_last=getattr(loader, "drop_last", False),
        )

    def fit(self, train_loader, val_loader, cfg: TrainConfig) -> Dict[str, Any]:
        if cfg.seed is not None:
            set_seed(cfg.seed)
        if cfg.deterministic:
            set_deterministic(True)

        ddp_info = {"rank": 0, "world": 1, "local_rank": 0}
        if cfg.ddp:
            ddp_info = self._ddp_init(cfg)

        # device selection
        device = torch.device(cfg.device)
        if cfg.ddp and cfg.device.startswith("cuda"):
            device = torch.device(f"cuda:{ddp_info['local_rank']}")

        self.model.to(device)

        # DDP samplers
        if cfg.ddp:
            train_loader = self._maybe_rewrap_loader_ddp(train_loader, shuffle=True)
            val_loader = self._maybe_rewrap_loader_ddp(val_loader, shuffle=False)

        # DDP wrap
        if cfg.ddp:
            self.model = self._ddp_wrap(cfg, ddp_info)

        is_rank0 = ddp_info["rank"] == 0

        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp) and device.type == "cuda")

        logger = RunLogger(cfg.log_dir, cfg.run_name) if is_rank0 else None
        if logger is not None:
            logger.save_config(asdict(cfg))
            logger.save_artifact(f"{cfg.run_name}.env.json", env_fingerprint())

        best_val = float("inf")
        best_path: Optional[str] = None

        for epoch in range(cfg.epochs):
            t_epoch = time.time()

            if cfg.ddp and hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            # ----- train
            self.model.train()
            tr_total, n_tr = 0.0, 0
            tr_parts: Dict[str, float] = {}

            for raw in train_loader:
                batch = self._xy_batch(raw)
                if self.preprocess is not None:
                    batch = self.preprocess.apply(batch)
                batch = self._move_batch(batch, device)

                x = batch.get("x") or batch.get("x_col")
                if x is None:
                    raise ValueError("Batch must include 'x' or 'x_col'.")

                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=bool(cfg.amp) and device.type == "cuda"):
                    y_hat_raw = self.model(x)
                    y_hat = self._unwrap_pred(y_hat_raw, batch=batch)
                    loss_out = self.loss_fn(self.model, y_hat, batch)
                    losses = self._parse_loss(loss_out)
                    loss = losses["total"]

                scaler.scale(loss).backward()

                if cfg.grad_clip and cfg.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)

                scaler.step(opt)
                scaler.update()

                bs = int(x.shape[0]) if isinstance(x, torch.Tensor) and x.ndim > 0 else 1
                tr_total += float(loss.item()) * bs
                n_tr += bs

                for k, v in losses.items():
                    if not isinstance(v, torch.Tensor):
                        continue
                    tr_parts[k] = tr_parts.get(k, 0.0) + float(v.detach().item()) * bs

            train_total = tr_total / max(1, n_tr)
            train_parts = {k: v / max(1, n_tr) for k, v in tr_parts.items()}

            if cfg.ddp and dist is not None and dist.is_initialized():
                t = torch.tensor([train_total], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                train_total = float((t / ddp_info["world"]).item())

            # ----- val
            self.model.eval()
            va_total, n_va = 0.0, 0
            va_parts: Dict[str, float] = {}
            yhat_acc, y_acc = [], []

            with torch.no_grad():
                for raw in val_loader:
                    batch = self._xy_batch(raw)
                    if self.preprocess is not None:
                        batch = self.preprocess.apply(batch)
                    batch = self._move_batch(batch, device)

                    x = batch.get("x") or batch.get("x_col")
                    if x is None:
                        raise ValueError("Batch must include 'x' or 'x_col'.")
                    y = batch.get("y")

                    y_hat_raw = self.model(x)
                    y_hat = self._unwrap_pred(y_hat_raw, batch=batch)

                    loss_out = self.loss_fn(self.model, y_hat, batch)
                    losses = self._parse_loss(loss_out)
                    loss = losses["total"]

                    bs = int(x.shape[0]) if isinstance(x, torch.Tensor) and x.ndim > 0 else 1
                    va_total += float(loss.item()) * bs
                    n_va += bs

                    for k, v in losses.items():
                        if not isinstance(v, torch.Tensor):
                            continue
                        va_parts[k] = va_parts.get(k, 0.0) + float(v.detach().item()) * bs

                    if y is not None and isinstance(y_hat, torch.Tensor) and isinstance(y, torch.Tensor):
                        yhat_acc.append(y_hat.detach().cpu())
                        y_acc.append(y.detach().cpu())

            val_total = va_total / max(1, n_va)
            val_parts = {k: v / max(1, n_va) for k, v in va_parts.items()}

            if cfg.ddp and dist is not None and dist.is_initialized():
                t = torch.tensor([val_total], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                val_total = float((t / ddp_info["world"]).item())

            metrics_dict: Dict[str, float] = {}
            if self.metrics_obj is not None and yhat_acc and y_acc:
                yhat_all = torch.cat(yhat_acc, dim=0)
                y_all = torch.cat(y_acc, dim=0)
                metrics_dict = self.metrics_obj.compute(yhat_all, y_all)

            if is_rank0 and logger is not None:
                logger.log(
                    {
                        "epoch": epoch,
                        "train_total": train_total,
                        "val_total": val_total,
                        "train_parts": train_parts,
                        "val_parts": val_parts,
                        "metrics": metrics_dict,
                        "epoch_sec": time.time() - t_epoch,
                    }
                )

            logs = {"val_total": float(val_total), "train_total": float(train_total), **metrics_dict}

            if is_rank0 and self.checkpoint is not None:
                self.checkpoint.maybe_save(self.model, extra={"epoch": epoch, "cfg": asdict(cfg)}, logs=logs)

            if is_rank0 and self.early_stopping is not None:
                self.early_stopping.update(logs)
                if self.early_stopping.stop:
                    break

            if is_rank0 and cfg.save_best and val_total < best_val:
                best_val = float(val_total)
                best_path = os.path.join(cfg.log_dir, f"{cfg.run_name}.best.pt")
                ckpt = Checkpoint(
                    model_state=self.model.state_dict(),
                    optim_state=opt.state_dict(),
                    cfg=asdict(cfg),
                    meta={
                        "epoch": epoch,
                        "best_val": best_val,
                        "metrics": metrics_dict,
                        "train_parts": train_parts,
                        "val_parts": val_parts,
                    },
                )
                save_checkpoint(best_path, ckpt)

            if cfg.ddp and dist is not None and dist.is_initialized():
                t = torch.tensor([best_val], device=device)
                dist.broadcast(t, src=0)
                best_val = float(t.item())

        if cfg.ddp and dist is not None and dist.is_initialized():
            dist.barrier()

        return {"best_val": best_val, "best_path": best_path, "ddp": cfg.ddp, "rank0": is_rank0}