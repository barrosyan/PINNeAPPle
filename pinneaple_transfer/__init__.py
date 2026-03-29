from __future__ import annotations
"""pinneaple_transfer — Transfer Learning and fine-tuning for PINN families.

Enables reuse of a pre-trained PINN across related physics problems —
e.g., fine-tuning Burgers nu=0.01 → nu=0.05 without retraining from scratch.

Modules
-------
TransferConfig
    Dataclass holding all fine-tuning hyperparameters and freezing strategy.
TransferTrainer
    Orchestrates fine-tuning: freezing, progressive unfreezing, training loop.
freeze_layers / unfreeze_layers
    Utility functions to selectively freeze/unfreeze parameters by name prefix.
layer_lr_groups
    Create optimizer param groups with per-layer learning rates (discriminative LR).
ParametricFamilyTransfer
    Manages a collection of fine-tuned models for different parameter values,
    with linear weight interpolation between known variants.
PhysicsTransferAdapter
    Adapts a model between physics domains; includes MMD domain adaptation loss.

Quick start
-----------
>>> from pinneaple_transfer import TransferTrainer, TransferConfig
>>> cfg = TransferConfig(strategy="finetune", epochs=500, finetune_lr=1e-5)
>>> trainer = TransferTrainer(source_model=pretrained_pinn, config=cfg)
>>> result = trainer.finetune(target_physics_fn, target_data={"x": x_t, "y": y_t})
>>> fine_tuned_model = result["model"]
"""

from .config import TransferConfig
from .freeze import freeze_layers, unfreeze_layers, freeze_all_except, layer_lr_groups, count_trainable
from .trainer import TransferTrainer
from .parametric import ParametricFamilyTransfer
from .adapter import PhysicsTransferAdapter

__all__ = [
    "TransferConfig",
    "TransferTrainer",
    "freeze_layers",
    "unfreeze_layers",
    "freeze_all_except",
    "layer_lr_groups",
    "count_trainable",
    "ParametricFamilyTransfer",
    "PhysicsTransferAdapter",
]
