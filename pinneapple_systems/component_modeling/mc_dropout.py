"""pinneapple_systems.component_modeling.mc_dropout — MC-Dropout epistemic
uncertainty: n_samples stochastic forward passes with dropout kept active,
reduced to a (mean, std) estimate.

This is the one small capability missing from pinneapple's existing model
base classes (``pinneapple_neural.architectures.base.BaseModel`` /
``pinns.base.PINNBase`` cover checkpointing, ONNX/TorchScript export, and a
physics-loss adapter already; ``pinneapple_neural/trainer/heteroscedastic.py``
covers a *different* UQ mechanism — aleatoric noise via a learned variance
head). Implemented here as a standalone function rather than a new base
class so it works on ANY ``nn.Module`` that has at least one ``nn.Dropout``
submodule, with no inheritance requirement.
"""
from __future__ import annotations

from typing import Any, Tuple

import torch
import torch.nn as nn


def mc_dropout_uncertainty(model: nn.Module, x: torch.Tensor, n_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (mean, std) over `n_samples` stochastic forward passes with
    dropout active. Raises ValueError if `model` has no `nn.Dropout`
    submodule (nothing would actually be stochastic)."""
    if not any(isinstance(m, nn.Dropout) for m in model.modules()):
        raise ValueError(
            f"{type(model).__name__} has no nn.Dropout submodule — MC-Dropout "
            "uncertainty would be deterministic (zero variance) and is not meaningful here."
        )
    was_training = model.training
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x)
            preds.append(out.y if hasattr(out, "y") else out)
    model.train(was_training)
    stack = torch.stack(preds, dim=0)
    return stack.mean(dim=0), stack.std(dim=0)
