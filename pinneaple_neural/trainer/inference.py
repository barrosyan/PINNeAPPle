"""Inference utilities: load_checkpoint and predict."""
from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn


def load_checkpoint(model: nn.Module, path: str, map_location: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    return ckpt


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
    if device is not None:
        model = model.to(device)
        x = x.to(device)
    model.eval()
    return model(x)
