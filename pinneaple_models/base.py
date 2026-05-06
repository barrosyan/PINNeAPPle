"""BaseModel and ModelOutput for Pinneaple model family.

This module defines a minimal *contract* that enables the Arena and Trainer to
run heterogeneous model families in a consistent way.

Key idea
--------
Every model can be executed in two ways:
  1) ``forward(x)``: classic PyTorch signature.
  2) ``forward_batch(batch)``: takes a dict batch (PINN / operator / graph / TS).

The default ``forward_batch`` implementation simply picks ``batch['x']`` (or
``batch['x_col']``) and calls ``forward(x)``.

Models that need different inputs (coords+params, graph objects, sequences,
fields, etc.) should override ``forward_batch``.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    y: torch.Tensor
    losses: Optional[Dict[str, torch.Tensor]] = None
    extras: Optional[Dict[str, Any]] = None


class BaseModel(nn.Module):
    """
    Unified base class for all Pinneaple models.
    """
    family: str = "generic"
    name: str = "base"

    def forward(self, *args, **kwargs) -> ModelOutput | torch.Tensor:
        raise NotImplementedError

    def forward_batch(self, batch: Dict[str, Any]) -> ModelOutput | torch.Tensor:
        """Default batch execution.

        The Trainer/Arena can provide a dict batch. This default implementation
        mirrors the Trainer's convention:
          - use ``batch['x']`` if present
          - otherwise use ``batch['x_col']`` (PINN collocation)
        and then call ``forward(x)``.
        """
        x = batch.get("x")
        if x is None:
            x = batch.get("x_col")
        if x is None:
            raise KeyError("forward_batch expects batch to include 'x' or 'x_col'.")
        return self.forward(x)

    # ------------------------------------------------------------------
    # Checkpoint I/O — available to all model families
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save model weights + metadata as a pinneaple checkpoint.

        The checkpoint dict contains:
          - ``"state_dict"``: model weights
          - ``"class_name"``: fully qualified class name
          - ``"metadata"``: user-provided dict

        Returns the path where the file was saved.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        ckpt = {
            "state_dict": self.state_dict(),
            "class_name": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "metadata": metadata or {},
        }
        torch.save(ckpt, path)
        return path

    @classmethod
    def load_checkpoint(cls, path: str, **init_kwargs) -> "BaseModel":
        """Load a model from a pinneaple checkpoint.

        Usage::
            model = VanillaPINN.load_checkpoint("model.pt", in_dim=2, out_dim=1, hidden=[64,64,64])
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(**init_kwargs)
        model.load_state_dict(ckpt["state_dict"])
        return model
