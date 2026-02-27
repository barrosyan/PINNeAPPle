from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class FitHistory:
    train: Dict[str, list]
    val: Dict[str, list]


class LSTMTrainer:
    """
    Trainer simples para modelos do tipo RecurrentModelBase que retornam RNNOutput(y, losses,...).

    Espera que cada batch do dataloader seja:
      - (x_past, y_future)  ou  {"x_past":..., "y_future":...}
    """
    def __init__(
        self,
        model: nn.Module,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: Optional[float] = 1.0,
        device: Optional[str] = None,
    ):
        self.model = model
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip

    def _unpack_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x_past, y_future = batch
        elif isinstance(batch, dict):
            x_past, y_future = batch["x_past"], batch["y_future"]
        else:
            raise ValueError("Batch deve ser (x_past, y_future) ou dict com chaves x_past/y_future.")
        return x_past.to(self.device), y_future.to(self.device)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        totals: Dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            x_past, y_future = self._unpack_batch(batch)
            out = self.model(x_past, y_future=y_future, return_loss=True)
            for k, v in out.losses.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach().item())
            n_batches += 1

        if n_batches == 0:
            return {"total": float("nan")}
        return {k: v / n_batches for k, v in totals.items()}

    def fit(
        self,
        train_loader: DataLoader,
        *,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 20,
        log_every: int = 1,
    ) -> FitHistory:
        history = FitHistory(train={"total": [], "mse": []}, val={"total": [], "mse": []})

        for epoch in range(1, int(epochs) + 1):
            self.model.train()
            train_totals: Dict[str, float] = {}
            n_batches = 0

            for batch in train_loader:
                x_past, y_future = self._unpack_batch(batch)

                self.optimizer.zero_grad(set_to_none=True)
                out = self.model(x_past, y_future=y_future, return_loss=True)

                loss = out.losses["total"]
                loss.backward()

                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.grad_clip))

                self.optimizer.step()

                for k, v in out.losses.items():
                    train_totals[k] = train_totals.get(k, 0.0) + float(v.detach().item())
                n_batches += 1

            train_metrics = {k: v / max(1, n_batches) for k, v in train_totals.items()}
            history.train["total"].append(train_metrics.get("total", float("nan")))
            history.train["mse"].append(train_metrics.get("mse", float("nan")))

            val_metrics = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history.val["total"].append(val_metrics.get("total", float("nan")))
                history.val["mse"].append(val_metrics.get("mse", float("nan")))

            if (epoch % log_every) == 0:
                if val_metrics is None:
                    print(f"Epoch {epoch:03d} | train: {train_metrics}")
                else:
                    print(f"Epoch {epoch:03d} | train: {train_metrics} | val: {val_metrics}")

        return history