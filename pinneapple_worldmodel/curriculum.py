"""Curriculum learning for physics world models.

:class:`PhysicsCurriculum` manages a sequence of training stages ordered by
complexity.  Each stage uses a different :class:`~.scenario.PhysicsScenario`
(or a mix of scenarios) and the curriculum advances to the next stage once
the model achieves a target validation loss.

Typical progression::

    Stage 0: heat_2d        (linear, smooth, single param)
    Stage 1: burgers_1d     (nonlinear, shock formation)
    Stage 2: advection_2d   (multi-param transport)
    Stage 3: ns2d_cavity    (full Navier-Stokes)
    Stage 4: heat_multiscale (high-resolution fine-tuning)

The curriculum integrates with :class:`~.trainer.WorldModelTrainer` and
:class:`~.dataset.DatasetBuilder`; each stage can optionally preload
previously generated data.

Quick start::

    from pinneapple_worldmodel import PhysicsCurriculum, CurriculumConfig

    curriculum = PhysicsCurriculum(CurriculumConfig())
    model = curriculum.run()
    # model is a trained PhysicsWorldModel
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .scenario import PhysicsScenario, BUILTIN_SCENARIOS
from .dataset import DatasetBuilder, DatasetConfig, WorldModelDataset
from .model import PhysicsWorldModel, WorldModelConfig
from .trainer import WorldModelTrainer, WorldModelTrainConfig


# ---------------------------------------------------------------------------
# Stage definition
# ---------------------------------------------------------------------------

@dataclass
class CurriculumStage:
    """One stage in the curriculum.

    Parameters
    ----------
    scenarios : list of scenario names or PhysicsScenario objects.
    n_samples : int — trajectories per scenario for this stage.
    epochs : int — training epochs for this stage.
    target_val_loss : float — advance to next stage when val_loss drops below this.
    lr : float — learning rate (allows warm-up / cool-down across stages).
    rollout_steps : int — multi-step unrolling for this stage.
    description : str
    """
    scenarios: List[Any]
    n_samples: int = 500
    epochs: int = 50
    target_val_loss: float = 1e-3
    lr: float = 1e-3
    rollout_steps: int = 1
    description: str = ""


# ---------------------------------------------------------------------------
# CurriculumConfig
# ---------------------------------------------------------------------------

@dataclass
class CurriculumConfig:
    """Full curriculum configuration.

    Parameters
    ----------
    stages : list of CurriculumStage (default: the 5-stage progression above).
    model_config : WorldModelConfig — shared model hyperparameters.
    device : str
    batch_size : int
    patience : int — early-stopping patience per stage.
    save_dir : str or None — directory for stage checkpoints.
    verbose : bool
    """
    stages: List[CurriculumStage] = field(default_factory=lambda: _default_stages())
    model_config: WorldModelConfig = field(default_factory=WorldModelConfig)
    device: str = "cpu"
    batch_size: int = 32
    patience: int = 10
    save_dir: Optional[str] = None
    verbose: bool = True


def _default_stages() -> List[CurriculumStage]:
    return [
        CurriculumStage(
            scenarios=["heat_2d"],
            n_samples=500,
            epochs=50,
            target_val_loss=5e-4,
            rollout_steps=1,
            description="Stage 0: linear diffusion (warm-up)",
        ),
        CurriculumStage(
            scenarios=["burgers_1d", "wave_1d"],
            n_samples=800,
            epochs=80,
            target_val_loss=2e-3,
            lr=5e-4,
            rollout_steps=2,
            description="Stage 1: nonlinear 1-D dynamics",
        ),
        CurriculumStage(
            scenarios=["heat_2d", "advection_2d"],
            n_samples=600,
            epochs=60,
            target_val_loss=1e-3,
            lr=3e-4,
            rollout_steps=2,
            description="Stage 2: multi-physics 2-D",
        ),
        CurriculumStage(
            scenarios=["ns2d_cavity"],
            n_samples=300,
            epochs=100,
            target_val_loss=5e-3,
            lr=1e-4,
            rollout_steps=4,
            description="Stage 3: full Navier-Stokes (curriculum peak)",
        ),
        CurriculumStage(
            scenarios=["heat_multiscale", "advection_2d"],
            n_samples=200,
            epochs=40,
            target_val_loss=1e-3,
            lr=5e-5,
            rollout_steps=4,
            description="Stage 4: high-resolution fine-tuning",
        ),
    ]


# ---------------------------------------------------------------------------
# PhysicsCurriculum
# ---------------------------------------------------------------------------

class PhysicsCurriculum:
    """Train a world model through a staged physics curriculum.

    Parameters
    ----------
    config : CurriculumConfig

    Attributes
    ----------
    model : PhysicsWorldModel — the model being trained (built lazily).
    stage_histories : list of per-stage history lists.
    current_stage : int — index of the current stage.

    Example
    -------
    >>> from pinneapple_worldmodel import PhysicsCurriculum, CurriculumConfig
    >>> curriculum = PhysicsCurriculum(CurriculumConfig(device="cuda"))
    >>> model = curriculum.run()
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config
        self.model: Optional[PhysicsWorldModel] = None
        self.stage_histories: List[List[Dict]] = []
        self.current_stage: int = 0

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run(self) -> PhysicsWorldModel:
        """Execute all stages and return the trained world model."""
        cfg = self.config

        for stage_idx, stage in enumerate(cfg.stages):
            self.current_stage = stage_idx
            if cfg.verbose:
                print(f"\n{'='*60}")
                print(f"[Curriculum] {stage.description or f'Stage {stage_idx}'}")
                print(f"{'='*60}")

            # Build dataset for this stage
            dataset = self._build_stage_dataset(stage)

            # Build or reuse model
            if self.model is None:
                self.model = self._build_model(dataset)
                if cfg.verbose:
                    print(f"[Curriculum] Model: {self.model}")

            # Train this stage
            train_cfg = WorldModelTrainConfig(
                epochs=stage.epochs,
                lr=stage.lr,
                batch_size=cfg.batch_size,
                device=cfg.device,
                rollout_steps=stage.rollout_steps,
                patience=cfg.patience,
                save_best=(
                    f"{cfg.save_dir}/stage_{stage_idx}_best.pt"
                    if cfg.save_dir else None
                ),
                log_every=max(1, stage.epochs // 5),
            )
            trainer = WorldModelTrainer(self.model, train_cfg)
            history = trainer.fit(dataset)
            self.stage_histories.append(history)

            best_val = min(
                (h.get("val_total", float("inf")) for h in history),
                default=float("inf"),
            )
            if cfg.verbose:
                print(f"[Curriculum] Stage {stage_idx} done — "
                      f"best val={best_val:.4g}, target={stage.target_val_loss:.4g}")

            if best_val <= stage.target_val_loss:
                if cfg.verbose:
                    print(f"[Curriculum] Target met — advancing.")
            else:
                if cfg.verbose:
                    print(f"[Curriculum] Target not met but continuing.")

        assert self.model is not None
        return self.model

    def advance_stage(self) -> bool:
        """Manually advance to the next stage (returns False if at last stage)."""
        if self.current_stage < len(self.config.stages) - 1:
            self.current_stage += 1
            return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_stage_dataset(self, stage: CurriculumStage) -> WorldModelDataset:
        ds_cfg = DatasetConfig(
            scenarios=stage.scenarios,
            n_samples_per_scenario=stage.n_samples,
            horizon=max(1, stage.rollout_steps),
            device=self.config.device,
            verbose=self.config.verbose,
        )
        return DatasetBuilder(ds_cfg).build()

    def _build_model(self, dataset: WorldModelDataset) -> PhysicsWorldModel:
        """Construct the world model using dataset metadata."""
        cfg = self.config.model_config
        # Override context_dim from actual dataset
        actual_ctx_dim = dataset.context_dim
        if cfg.context_dim != actual_ctx_dim:
            from dataclasses import replace
            cfg = replace(cfg, context_dim=actual_ctx_dim)

        return PhysicsWorldModel(
            cfg,
            n_fields=dataset.n_fields,
            grid_shape=dataset.grid_shape,
        )

    def summary(self) -> Dict:
        """Return a summary of all completed stages."""
        out = {}
        for i, history in enumerate(self.stage_histories):
            if not history:
                continue
            best_val = min(h.get("val_total", float("inf")) for h in history)
            out[f"stage_{i}"] = {
                "epochs_run": len(history),
                "best_val": best_val,
                "description": self.config.stages[i].description,
            }
        return out
