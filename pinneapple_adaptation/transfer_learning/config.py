"""TransferConfig — configuration dataclass for transfer learning runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class TransferConfig:
    """Configuration for a transfer-learning / fine-tuning run.

    Parameters
    ----------
    strategy:
        High-level transfer strategy to apply when :meth:`TransferTrainer.prepare`
        is called.

        * ``"finetune"`` — unfreeze all layers and train end-to-end at
          ``finetune_lr``.
        * ``"feature_extract"`` — freeze all layers except those matching
          ``unfreeze_prefix``; backbone acts as a fixed feature extractor.
        * ``"partial_freeze"`` — freeze layers matching ``freeze_prefix``,
          leave everything else trainable.
        * ``"progressive"`` — backbone is frozen for ``warmup_epochs``, then
          layers are unfrozen according to ``unfreeze_prefix`` (or fully).
          Use :meth:`TransferTrainer.progressive_unfreeze` for fine-grained
          epoch-level schedules.

    freeze_prefix:
        Layer name prefixes (matched against ``name`` in
        ``model.named_parameters()``) whose parameters will be frozen.

    unfreeze_prefix:
        Layer name prefixes that will remain trainable (used in
        ``"feature_extract"`` strategy, and as the final unfrozen set in
        ``"progressive"`` strategy).

    layer_lr_scale:
        Mapping of layer-name prefix → LR scale factor relative to
        ``base_lr``.  Used by :func:`layer_lr_groups` to build per-layer
        optimizer param groups for discriminative fine-tuning.
        E.g. ``{"net.0": 0.1, "net.2": 0.5}`` gives early layers a lower LR.

    base_lr:
        Learning rate applied to un-scaled / head layers.

    finetune_lr:
        Learning rate used when strategy is ``"finetune"`` (typically 5–10×
        lower than pre-training LR).

    epochs:
        Total number of fine-tuning epochs.

    physics_weight:
        Weight applied to the physics / PDE residual loss term.

    data_weight:
        Weight applied to the supervised data-fit loss term.

    warmup_epochs:
        Number of epochs with the backbone frozen before it is released
        (applies to ``"progressive"`` strategy).

    device:
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, ``"cuda:1"``.

    seed:
        Random seed for reproducibility.  ``0`` is a valid seed; pass
        ``-1`` to skip seeding.
    """

    strategy: str = "finetune"
    freeze_prefix: List[str] = field(default_factory=list)
    unfreeze_prefix: List[str] = field(default_factory=list)
    layer_lr_scale: Dict[str, float] = field(default_factory=dict)
    base_lr: float = 1e-4
    finetune_lr: float = 1e-5
    epochs: int = 500
    physics_weight: float = 1.0
    data_weight: float = 1.0
    warmup_epochs: int = 50
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        valid_strategies = {"finetune", "feature_extract", "partial_freeze", "progressive"}
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"TransferConfig.strategy must be one of {valid_strategies!r}, "
                f"got {self.strategy!r}."
            )
        if self.epochs < 1:
            raise ValueError("TransferConfig.epochs must be >= 1.")
        if self.warmup_epochs < 0:
            raise ValueError("TransferConfig.warmup_epochs must be >= 0.")
        if self.physics_weight < 0 or self.data_weight < 0:
            raise ValueError("Loss weights must be non-negative.")
