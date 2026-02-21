"""BaseModel and ModelOutput for Pinneaple model family."""
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
