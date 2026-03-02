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
from dataclasses import dataclass
from typing import Dict, Any, Optional
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
