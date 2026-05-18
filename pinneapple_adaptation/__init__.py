"""pinneapple_adaptation — Transfer learning and meta-learning for PINNs.

Sub-modules
-----------
transfer_learning  (was pinneapple_transfer)
    Transfer learning: fine-tune a pre-trained PINN on a related physics
    problem. Supports layer freezing, progressive unfreezing, discriminative
    learning rates, parametric-family interpolation, and MMD domain adaptation.

meta_learning      (was pinneapple_meta)
    Meta-learning (MAML / Reptile): train a shared initialisation that can
    adapt to any member of a PDE family in a few gradient steps. Includes
    PDETaskSampler, MAMLTrainer, ReptileTrainer, and MetaModel.

Integration helpers
-------------------
``fine_tune(source_model, target_physics, ...)``
    Convenience wrapper for TransferTrainer.finetune().
``adapt_model(model, target, ...)``
    Auto-selects between transfer and meta-learning based on available context.

Usage
-----
>>> from pinneapple_adaptation import fine_tune, meta_train, fast_adapt
>>> result = fine_tune(pretrained, target_physics_fn, epochs=500)
>>> trainer = meta_train(model, sampler, algorithm="reptile")
>>> adapted = fast_adapt(trainer, new_task, n_steps=20)
"""
from __future__ import annotations

# ── sub-modules (new descriptive names) ───────────────────────────────────────
from . import transfer_learning
from . import meta_learning

# backward-compat aliases
transfer = transfer_learning
meta     = meta_learning

# ── transfer_learning re-exports ──────────────────────────────────────────────
from .transfer_learning import (
    TransferConfig,
    TransferTrainer,
    freeze_layers,
    unfreeze_layers,
    freeze_all_except,
    layer_lr_groups,
    count_trainable,
    ParametricFamilyTransfer,
    PhysicsTransferAdapter,
)

# ── meta_learning re-exports ──────────────────────────────────────────────────
from .meta_learning import (
    MAMLConfig,
    ReptileConfig,
    PDETaskSampler,
    MAMLTrainer,
    ReptileTrainer,
    MetaModel,
    meta_train,
    meta_adapt,
)


# ── Integration helpers ────────────────────────────────────────────────────────

def fine_tune(source_model, target_physics_fn, target_data=None, *,
              strategy: str = "finetune", epochs: int = 500, lr: float = 1e-5,
              **cfg_kwargs) -> dict:
    """Fine-tune a pre-trained PINN on a new physics problem."""
    cfg = TransferConfig(strategy=strategy, epochs=epochs, finetune_lr=lr, **cfg_kwargs)
    trainer = TransferTrainer(source_model=source_model, config=cfg)
    return trainer.finetune(target_physics_fn, target_data=target_data)


def adapt_model(model, target, *, mode: str = "auto", **kwargs):
    """Adapt a model to a new physics task (auto-selects transfer vs meta)."""
    if mode == "auto":
        mode = "meta" if isinstance(target, dict) and "support" in target else "transfer"
    if mode == "meta":
        from .meta_learning import meta_adapt as _adapt
        return _adapt(model, target, **kwargs)
    return fine_tune(model, target, **kwargs)


__all__ = [
    # Sub-modules (new names)
    "transfer_learning", "meta_learning",
    # Sub-modules (old aliases — backward compat)
    "transfer", "meta",
    # Integration
    "fine_tune", "adapt_model",
    # transfer_learning
    "TransferConfig", "TransferTrainer",
    "freeze_layers", "unfreeze_layers", "freeze_all_except",
    "layer_lr_groups", "count_trainable",
    "ParametricFamilyTransfer", "PhysicsTransferAdapter",
    # meta_learning
    "MAMLConfig", "ReptileConfig",
    "PDETaskSampler", "MAMLTrainer", "ReptileTrainer",
    "MetaModel", "meta_train", "meta_adapt",
]
