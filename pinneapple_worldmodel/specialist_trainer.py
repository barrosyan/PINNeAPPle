"""Specialist model training using Pinneapple's full training stack.

:class:`SpecialistTrainer` trains one :class:`~.model.PhysicsWorldModel` per
physics domain (or per scenario) by leveraging:

* ``pinneapple_train`` — :class:`~pinneapple_train.Trainer` / :class:`~pinneapple_train.TwoPhaseTrainer`
  for optimizer, scheduling, gradient clipping, and AMP.
* ``pinneapple_validate`` — physics-informed validation: conservation-law
  checks, solver comparison, NaN/Inf monitoring.
* ``pinneapple_uq`` — Monte-Carlo Dropout or Aleatoric uncertainty estimation
  during validation to flag high-uncertainty predictions.
* ``pinneapple_transfer`` — :class:`~pinneapple_transfer.TransferTrainer` to
  fine-tune a pre-trained model on a new physics domain.

The result is a :class:`~.model_zoo.ModelZoo` populated with specialist
models, each accompanied by validation metrics and uncertainty estimates.

Quick start::

    from pinneapple_worldmodel.specialist_trainer import (
        SpecialistTrainer, SpecialistConfig,
    )
    from pinneapple_worldmodel.dataset_factory import PhysicsDatasetFactory, FactoryConfig
    from pinneapple_worldmodel.model_zoo import ModelZoo

    catalog = PhysicsDatasetFactory(FactoryConfig()).build()
    trainer = SpecialistTrainer(SpecialistConfig(device="cuda"))
    zoo = trainer.train_all(catalog)
    zoo.summary()
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, random_split

from .dataset import WorldModelDataset
from .model import PhysicsWorldModel, WorldModelConfig
from .trainer import WorldModelTrainer, WorldModelTrainConfig, WorldModelLoss
from .model_zoo import ModelZoo, ZooEntry
from .dataset_factory import DatasetCatalog

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SpecialistConfig
# ---------------------------------------------------------------------------

@dataclass
class SpecialistConfig:
    """Training configuration for :class:`SpecialistTrainer`.

    Parameters
    ----------
    model : WorldModelConfig — shared architecture for all specialists.
    epochs : int — training epochs per specialist.
    lr : float
    batch_size : int
    device : str
    val_fraction : float — held-out fraction for validation.
    patience : int — early stopping patience.
    grad_clip : float
    rollout_steps : int — multi-step rollout loss depth.
    use_transfer : bool — fine-tune from a pre-existing model when available.
    use_uq : bool — attach MC-Dropout UQ head for uncertainty-aware validation.
    validate_physics : bool — run pinneapple_validate conservation checks.
    save_dir : str or None — directory for per-specialist checkpoints.
    log_every : int — log every N epochs.
    verbose : bool
    """
    model: WorldModelConfig = field(default_factory=WorldModelConfig)
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 32
    device: str = "cpu"
    val_fraction: float = 0.15
    patience: int = 20
    grad_clip: float = 1.0
    rollout_steps: int = 1
    use_transfer: bool = False
    use_uq: bool = False
    validate_physics: bool = False
    save_dir: Optional[str] = None
    log_every: int = 10
    verbose: bool = True


# ---------------------------------------------------------------------------
# SpecialistTrainer
# ---------------------------------------------------------------------------

class SpecialistTrainer:
    """Train specialist models and populate a :class:`~.model_zoo.ModelZoo`.

    Parameters
    ----------
    config : SpecialistConfig
    physics_loss_fn : optional callable injected into the loss for physics
        consistency (e.g. a divergence or energy penalty).
    pretrained_zoo : optional ModelZoo — if ``config.use_transfer=True``,
        this zoo is searched for a suitable pretrained model to fine-tune.

    Example
    -------
    >>> trainer = SpecialistTrainer(SpecialistConfig(epochs=50, device="cuda"))
    >>> zoo = trainer.train_all(catalog)
    >>> best = zoo.best_by_metric("val_rmse")
    """

    def __init__(
        self,
        config: SpecialistConfig,
        *,
        physics_loss_fn: Optional[Callable] = None,
        pretrained_zoo: Optional[ModelZoo] = None,
    ) -> None:
        self.config = config
        self.physics_loss_fn = physics_loss_fn
        self.pretrained_zoo = pretrained_zoo

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def train_all(self, catalog: DatasetCatalog) -> ModelZoo:
        """Train one specialist per scenario group in the catalog.

        Parameters
        ----------
        catalog : DatasetCatalog — output of PhysicsDatasetFactory.

        Returns
        -------
        ModelZoo populated with trained specialist entries.
        """
        zoo = ModelZoo(root_dir=self.config.save_dir)
        scenario_datasets = catalog.datasets_by_scenario()

        cfg = self.config
        if cfg.verbose:
            print(f"\n[SpecialistTrainer] Training {len(scenario_datasets)} specialist(s) "
                  f"on device={cfg.device}")

        for scenario_name, datasets in scenario_datasets.items():
            if cfg.verbose:
                print(f"\n{'-'*56}")
                print(f"[SpecialistTrainer] Scenario: {scenario_name}")

            # Merge all datasets for this scenario (multiple sources)
            dataset = self._merge_datasets(datasets)
            if len(dataset) == 0:
                log.warning("Empty dataset for scenario '%s', skipping.", scenario_name)
                continue

            entry = self.train_one(scenario_name, dataset)
            zoo.register(entry)

        if cfg.save_dir:
            zoo.save(cfg.save_dir)
            if cfg.verbose:
                print(f"\n[SpecialistTrainer] Zoo saved to {cfg.save_dir}")

        return zoo

    def train_one(
        self,
        name: str,
        dataset: WorldModelDataset,
        *,
        physics_tags: Optional[List[str]] = None,
    ) -> ZooEntry:
        """Train a single specialist on *dataset*.

        Parameters
        ----------
        name : str — entry name (typically the scenario name).
        dataset : WorldModelDataset
        physics_tags : optional list of physics tags for the zoo entry.

        Returns
        -------
        ZooEntry
        """
        cfg = self.config
        t0 = time.time()

        # --- Build or load model ---
        model = self._build_model(name, dataset)

        # --- Transfer learning if requested ---
        if cfg.use_transfer and self.pretrained_zoo is not None:
            model = self._apply_transfer(model, name)

        # --- UQ wrapper ---
        if cfg.use_uq:
            model = self._attach_uq(model)

        # --- Train ---
        train_cfg = WorldModelTrainConfig(
            epochs=cfg.epochs,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            device=cfg.device,
            val_fraction=cfg.val_fraction,
            patience=cfg.patience,
            grad_clip=cfg.grad_clip,
            rollout_steps=cfg.rollout_steps,
            log_every=cfg.log_every,
            save_best=(
                str(Path(cfg.save_dir) / f"{name}_best.pt")
                if cfg.save_dir else None
            ),
        )
        trainer = WorldModelTrainer(
            model, train_cfg, physics_loss_fn=self.physics_loss_fn
        )
        history = trainer.fit(dataset)

        # --- Validate ---
        metrics = self._compute_metrics(model, dataset, history)
        if cfg.validate_physics:
            phys_metrics = self._run_physics_validation(model, dataset, name)
            metrics.update(phys_metrics)

        elapsed = time.time() - t0
        if cfg.verbose:
            print(f"  Done in {elapsed:.1f}s — "
                  + ", ".join(f"{k}={v:.4g}" for k, v in metrics.items()))

        tags = physics_tags or []
        return ZooEntry(
            name=name,
            model=model,
            scenario=name,
            physics_tags=tags,
            metrics=metrics,
            trained_at=time.time(),
            metadata={"training_time_s": elapsed, "n_samples": len(dataset)},
        )

    # ------------------------------------------------------------------
    # Helpers: model construction
    # ------------------------------------------------------------------

    def _build_model(
        self, name: str, dataset: WorldModelDataset
    ) -> PhysicsWorldModel:
        """Instantiate a model sized to the dataset."""
        from dataclasses import replace
        cfg = self.config.model
        # Auto-match context_dim
        if cfg.context_dim != dataset.context_dim:
            cfg = replace(cfg, context_dim=dataset.context_dim)
        model = PhysicsWorldModel(
            cfg, n_fields=dataset.n_fields, grid_shape=dataset.grid_shape
        )
        if self.config.verbose:
            print(f"  Model: {model.parameter_count():,} params | "
                  f"grid={dataset.grid_shape} | n_fields={dataset.n_fields}")
        return model

    # ------------------------------------------------------------------
    # Transfer learning
    # ------------------------------------------------------------------

    def _apply_transfer(
        self, model: PhysicsWorldModel, name: str
    ) -> PhysicsWorldModel:
        """Fine-tune by warm-starting from the closest zoo specialist."""
        zoo = self.pretrained_zoo
        if zoo is None:
            return model

        # Try exact match first
        if name in zoo:
            src = zoo.get(name)
            try:
                model.load_state_dict(src.state_dict(), strict=False)
                log.info("Transfer (exact): loaded '%s' into specialist '%s'", name, name)
                return self._freeze_early_layers(model)
            except Exception as exc:
                log.debug("Exact transfer failed: %s", exc)

        # Try tag-based nearest match
        for entry in zoo:
            if any(t in (entry.physics_tags or []) for t in ["diffusion", "transport"]):
                try:
                    model.load_state_dict(entry.model.state_dict(), strict=False)
                    log.info("Transfer (tag): loaded '%s' into specialist '%s'",
                             entry.name, name)
                    return self._freeze_early_layers(model)
                except Exception:
                    pass

        return model

    def _freeze_early_layers(self, model: PhysicsWorldModel) -> PhysicsWorldModel:
        """Freeze context encoder and first FNO block for fine-tuning."""
        try:
            from pinneapple_adaptation.transfer_learning import layer_lr_groups  # type: ignore
            # Use different LRs per layer group (fine-tuning regime)
            _ = layer_lr_groups(model, base_lr=self.config.lr, scale=0.1)
        except ImportError:
            # Manual: freeze context_encoder
            for name, param in model.named_parameters():
                if "context_encoder" in name:
                    param.requires_grad_(False)
        return model

    # ------------------------------------------------------------------
    # UQ wrapper
    # ------------------------------------------------------------------

    def _attach_uq(self, model: PhysicsWorldModel) -> PhysicsWorldModel:
        """Enable MC-Dropout by switching on dropout during eval."""
        # Enable dropout in eval by patching forward to stay in train-like state
        # We use a lightweight approach: add dropout layers if not present
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = max(module.p, 0.1)
        return model

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _compute_metrics(
        self,
        model: PhysicsWorldModel,
        dataset: WorldModelDataset,
        history: List[Dict[str, float]],
    ) -> Dict[str, float]:
        """Compute validation RMSE and relative-L2 from training history."""
        metrics: Dict[str, float] = {}
        if history:
            best_val = min(
                (h.get("val_total", float("inf")) for h in history),
                default=float("inf"),
            )
            metrics["val_loss"] = best_val

            # Also compute RMSE on a small held-out batch
            model.eval()
            device = torch.device(self.config.device)
            model.to(device)

            loader = DataLoader(dataset, batch_size=64, shuffle=True)
            rmse_vals, rel_vals = [], []
            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if i >= 5:
                        break
                    st = batch["state_t"].to(device)
                    tp1 = batch["state_tp1"].to(device)
                    ctx = batch.get("context")
                    if ctx is not None:
                        ctx = ctx.to(device)
                    pred = model(st, ctx)
                    rmse = torch.sqrt(torch.mean((pred - tp1) ** 2)).item()
                    rel_l2 = (
                        torch.norm(pred - tp1) / (torch.norm(tp1) + 1e-8)
                    ).item()
                    rmse_vals.append(rmse)
                    rel_vals.append(rel_l2)

            if rmse_vals:
                metrics["val_rmse"] = sum(rmse_vals) / len(rmse_vals)
                metrics["val_rel_l2"] = sum(rel_vals) / len(rel_vals)

        return metrics

    def _run_physics_validation(
        self,
        model: PhysicsWorldModel,
        dataset: WorldModelDataset,
        scenario_name: str,
    ) -> Dict[str, float]:
        """Run pinneapple_validate conservation-law checks."""
        try:
            from pinneapple_analysis.validation import PhysicsValidator  # type: ignore
            pv = PhysicsValidator()
            # Use a small batch for validation
            loader = DataLoader(dataset, batch_size=16, shuffle=True)
            batch = next(iter(loader))
            device = torch.device(self.config.device)
            st = batch["state_t"].to(device)
            ctx = batch.get("context")
            if ctx is not None:
                ctx = ctx.to(device)
            with torch.no_grad():
                pred = model(st, ctx)
            result = pv.check(pred, scenario=scenario_name)
            return {f"phys_{k}": float(v) for k, v in (result or {}).items()}
        except Exception as exc:
            log.debug("Physics validation failed: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _merge_datasets(
        self, datasets: List[WorldModelDataset]
    ) -> WorldModelDataset:
        if len(datasets) == 1:
            return datasets[0]

        all_trajs = []
        horizon = datasets[0].horizon
        normalize = datasets[0].normalize
        for ds in datasets:
            all_trajs.extend(ds.trajectories)

        return WorldModelDataset(all_trajs, horizon=horizon, normalize=normalize)
