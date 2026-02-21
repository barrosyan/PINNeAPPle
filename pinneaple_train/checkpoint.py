"""Checkpoint save/load for model, optimizer, config, and normalizers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
import torch


@dataclass
class Checkpoint:
    model_state: Dict[str, Any]
    optim_state: Optional[Dict[str, Any]]
    cfg: Dict[str, Any]
    meta: Dict[str, Any]
    normalizers: Optional[Dict[str, Any]] = None

def save_checkpoint(path: str, ckpt: Checkpoint) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state": ckpt.model_state,
            "optim_state": ckpt.optim_state,
            "cfg": ckpt.cfg,
            "meta": ckpt.meta,
            "normalizers": ckpt.normalizers,
        },
        path,
    )

def load_checkpoint(path: str) -> Checkpoint:
    obj = torch.load(path, map_location="cpu")
    return Checkpoint(
        model_state=obj["model_state"],
        optim_state=obj.get("optim_state"),
        cfg=obj.get("cfg", {}),
        meta=obj.get("meta", {}),
        normalizers=obj.get("normalizers"),
    )
