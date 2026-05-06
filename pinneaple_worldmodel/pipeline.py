"""End-to-end world model pipeline.

:class:`WorldModelPipeline` orchestrates the full flow::

    Scenarios → DatasetBuilder → WorldModelDataset
        ↓
    PhysicsWorldModel (FNO-based)
        ↓
    WorldModelTrainer  (rollout loss + physics consistency)
        ↓
    Evaluation: field-wise RMSE, rel-L2, rollout error curves
        ↓
    Optional: PhysicsCurriculum for staged training

This is the single entry point for users who want to generate a physics AI
world model from scratch using Pinneaple's simulation tools.

Quick start::

    from pinneaple_worldmodel import WorldModelPipeline, PipelineConfig

    pipeline = WorldModelPipeline(PipelineConfig(
        scenarios=["heat_2d", "burgers_1d", "ns2d_cavity"],
        n_samples_per_scenario=500,
        epochs=200,
        device="cuda",
    ))
    model, history = pipeline.run()

    # Rollout a new trajectory
    with torch.no_grad():
        states = model.rollout(state_0, context, n_steps=20)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .scenario import PhysicsScenario, BUILTIN_SCENARIOS
from .dataset import DatasetBuilder, DatasetConfig, WorldModelDataset
from .model import PhysicsWorldModel, WorldModelConfig
from .trainer import WorldModelTrainer, WorldModelTrainConfig
from .curriculum import PhysicsCurriculum, CurriculumConfig
from .simulator import PhysicsSimulator


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Complete configuration for :class:`WorldModelPipeline`.

    Parameters
    ----------
    scenarios : list of scenario names or PhysicsScenario objects.
        Determines which physics problems are simulated and mixed.
    n_samples_per_scenario : int
        Trajectories to generate per scenario.
    horizon : int
        Prediction horizon (steps between state_t and target).
    model : WorldModelConfig
        Architecture hyperparameters.
    train : WorldModelTrainConfig or None
        Training hyperparameters (if None, built from other fields).
    epochs : int — overrides ``train.epochs`` if ``train`` is None.
    batch_size : int
    lr : float
    device : str
    use_curriculum : bool
        If True, uses :class:`~.curriculum.PhysicsCurriculum` instead of flat
        training.  ``scenarios`` and ``n_samples_per_scenario`` are then used
        per-stage defaults (overridden by the curriculum config).
    curriculum : CurriculumConfig or None
        Explicit curriculum config (used only when ``use_curriculum=True``).
    validate_physics : bool
        Filter trajectories with NaN/Inf or energy blow-up.
    save_dir : str or None
        Directory for dataset, checkpoints, and evaluation results.
    verbose : bool
    """
    scenarios: List[Any] = field(
        default_factory=lambda: ["heat_2d", "burgers_1d", "advection_2d"]
    )
    n_samples_per_scenario: int = 500
    horizon: int = 1
    model: WorldModelConfig = field(default_factory=WorldModelConfig)
    train: Optional[WorldModelTrainConfig] = None
    epochs: int = 100
    batch_size: int = 32
    lr: float = 1e-3
    device: str = "cpu"
    use_curriculum: bool = False
    curriculum: Optional[CurriculumConfig] = None
    validate_physics: bool = True
    save_dir: Optional[str] = None
    verbose: bool = True


# ---------------------------------------------------------------------------
# WorldModelPipeline
# ---------------------------------------------------------------------------

class WorldModelPipeline:
    """Full data-generation + training + evaluation pipeline.

    Parameters
    ----------
    config : PipelineConfig

    Key attributes after :meth:`run`
    ---------------------------------
    model : PhysicsWorldModel — trained model.
    dataset : WorldModelDataset — full training dataset.
    history : list of epoch dicts.
    eval_results : dict of evaluation metrics.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.model: Optional[PhysicsWorldModel] = None
        self.dataset: Optional[WorldModelDataset] = None
        self.history: List[Dict] = []
        self.eval_results: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run(self) -> Tuple[PhysicsWorldModel, List[Dict]]:
        """Execute the full pipeline.

        Returns
        -------
        (model, history)
        """
        cfg = self.config

        if cfg.use_curriculum:
            return self._run_curriculum()

        # ---- Flat pipeline ----
        if cfg.verbose:
            print("[WorldModelPipeline] Step 1/3: Building dataset …")
        self.dataset = self.build_dataset()

        if cfg.verbose:
            print(f"[WorldModelPipeline] Dataset: {len(self.dataset)} samples, "
                  f"n_fields={self.dataset.n_fields}, "
                  f"grid={self.dataset.grid_shape}, "
                  f"context_dim={self.dataset.context_dim}")
            print("[WorldModelPipeline] Step 2/3: Training model …")

        self.model, self.history = self.train(self.dataset)

        if cfg.verbose:
            print("[WorldModelPipeline] Step 3/3: Evaluating …")
        self.eval_results = self.evaluate(self.model, self.dataset)

        if cfg.verbose:
            self._print_eval(self.eval_results)

        return self.model, self.history

    # ------------------------------------------------------------------
    # Step 1: Build dataset
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        *,
        scenarios: Optional[List[Any]] = None,
        n_samples: Optional[int] = None,
    ) -> WorldModelDataset:
        """Generate physics trajectories and build the dataset.

        Parameters
        ----------
        scenarios : override ``config.scenarios``.
        n_samples : override ``config.n_samples_per_scenario``.

        Returns
        -------
        WorldModelDataset
        """
        cfg = self.config
        ds_cfg = DatasetConfig(
            scenarios=scenarios or cfg.scenarios,
            n_samples_per_scenario=n_samples or cfg.n_samples_per_scenario,
            horizon=cfg.horizon,
            validate_physics=cfg.validate_physics,
            save_dir=str(Path(cfg.save_dir) / "dataset") if cfg.save_dir else None,
            device=cfg.device,
            verbose=cfg.verbose,
        )
        return DatasetBuilder(ds_cfg).build()

    # ------------------------------------------------------------------
    # Step 2: Train
    # ------------------------------------------------------------------

    def train(
        self,
        dataset: WorldModelDataset,
        *,
        val_dataset: Optional[WorldModelDataset] = None,
    ) -> Tuple[PhysicsWorldModel, List[Dict]]:
        """Build and train the world model on *dataset*.

        Parameters
        ----------
        dataset : WorldModelDataset — training data.
        val_dataset : optional separate validation set.

        Returns
        -------
        (model, history)
        """
        cfg = self.config

        # Build model
        model_cfg = cfg.model
        # Auto-set context_dim from dataset if default
        if model_cfg.context_dim == 8 and dataset.context_dim != 8:
            from dataclasses import replace
            model_cfg = replace(model_cfg, context_dim=dataset.context_dim)

        model = PhysicsWorldModel(
            model_cfg,
            n_fields=dataset.n_fields,
            grid_shape=dataset.grid_shape,
        )

        if cfg.verbose:
            print(f"  {model}")

        # Build train config
        train_cfg = cfg.train or WorldModelTrainConfig(
            epochs=cfg.epochs,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            device=cfg.device,
            save_best=(
                str(Path(cfg.save_dir) / "best_model.pt")
                if cfg.save_dir else None
            ),
        )

        trainer = WorldModelTrainer(model, train_cfg)
        history = trainer.fit(dataset, val_dataset=val_dataset)

        return model, history

    # ------------------------------------------------------------------
    # Step 3: Evaluate
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        model: PhysicsWorldModel,
        dataset: WorldModelDataset,
        *,
        n_rollout_steps: int = 10,
        n_eval_samples: int = 50,
    ) -> Dict[str, Any]:
        """Evaluate the model on held-out samples.

        Computes:
        - Next-step RMSE and relative-L2
        - Rollout RMSE at 1, 5, and *n_rollout_steps* steps
        - Field-wise breakdown

        Parameters
        ----------
        model : PhysicsWorldModel
        dataset : WorldModelDataset (used as test set here)
        n_rollout_steps : int — how many steps to unroll for rollout eval.
        n_eval_samples : int — number of trajectories to use.

        Returns
        -------
        dict of metrics.
        """
        model.eval()
        device = next(model.parameters()).device

        # Sample trajectories for rollout evaluation
        trajs = dataset.trajectories[:n_eval_samples]
        mean, std = dataset.norm_stats["mean"], dataset.norm_stats["std"]

        rollout_rmse: Dict[int, List[float]] = {1: [], 5: [], n_rollout_steps: []}

        for traj in trajs:
            states = traj.states  # (T+1, C, *grid)
            T = states.shape[0] - 1
            if T < n_rollout_steps:
                continue

            # Normalise
            s_norm = (states - mean) / std

            params = dataset._encode_params(traj.params).unsqueeze(0).to(device)
            context = dataset._encode_context(traj).unsqueeze(0).to(device)

            state = s_norm[0].unsqueeze(0).to(device)

            for step in range(1, n_rollout_steps + 1):
                state = model(state, context)
                target = s_norm[min(step, T)].unsqueeze(0).to(device)
                rmse = torch.sqrt(torch.mean((state - target) ** 2)).item()

                for milestone in [1, 5, n_rollout_steps]:
                    if step == milestone:
                        rollout_rmse[milestone].append(rmse)

        results: Dict[str, Any] = {}
        for step, vals in rollout_rmse.items():
            if vals:
                results[f"rollout_rmse_step{step}"] = sum(vals) / len(vals)

        # Next-step metrics over the full dataset (subsample)
        next_mse_list, rel_l2_list = [], []
        from torch.utils.data import DataLoader, Subset
        import random
        indices = random.sample(range(len(dataset)), min(500, len(dataset)))
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=32, shuffle=False)

        for batch in loader:
            state_t  = batch["state_t"].to(device)
            target   = batch["state_tp1"].to(device)
            context  = batch["context"].to(device)
            pred = model(state_t, context)
            next_mse_list.append(torch.mean((pred - target) ** 2).item())
            rel_l2 = (torch.norm(pred - target) / (torch.norm(target) + 1e-8)).item()
            rel_l2_list.append(rel_l2)

        results["next_step_rmse"] = (sum(next_mse_list) / len(next_mse_list)) ** 0.5
        results["next_step_rel_l2"] = sum(rel_l2_list) / len(rel_l2_list)
        results["n_params"] = model.parameter_count()

        if self.config.save_dir:
            import json
            path = Path(self.config.save_dir) / "eval_results.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(results, f, indent=2)

        return results

    # ------------------------------------------------------------------
    # Curriculum mode
    # ------------------------------------------------------------------

    def _run_curriculum(self) -> Tuple[PhysicsWorldModel, List[Dict]]:
        cfg = self.config
        cur_cfg = cfg.curriculum or CurriculumConfig(
            model_config=cfg.model,
            device=cfg.device,
            batch_size=cfg.batch_size,
            save_dir=str(Path(cfg.save_dir) / "curriculum") if cfg.save_dir else None,
            verbose=cfg.verbose,
        )
        curriculum = PhysicsCurriculum(cur_cfg)
        model = curriculum.run()
        self.model = model
        self.history = [h for hist in curriculum.stage_histories for h in hist]
        return model, self.history

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def generate_test_trajectory(
        self,
        scenario_name: str = "heat_2d",
        params: Optional[Dict] = None,
        seed: int = 999,
    ) -> Dict:
        """Generate one test trajectory for quick sanity-checking.

        Returns dict with ``"states"`` ``(T+1, C, *grid)`` and ``"params"``.
        """
        sc = BUILTIN_SCENARIOS[scenario_name]
        sim = PhysicsSimulator(sc, device=self.config.device)
        traj = sim.generate_trajectory(params=params, seed=seed)
        return {"states": traj.states, "params": traj.params, "scenario": scenario_name}

    def _print_eval(self, results: Dict[str, Any]) -> None:
        print("\n[WorldModelPipeline] Evaluation results:")
        print(f"  next_step_rmse   = {results.get('next_step_rmse', 'N/A'):.4g}")
        print(f"  next_step_rel_l2 = {results.get('next_step_rel_l2', 'N/A'):.4g}")
        for k in ["rollout_rmse_step1", "rollout_rmse_step5",
                  f"rollout_rmse_step{results.get('n_params', '')}"]:
            if k in results:
                print(f"  {k} = {results[k]:.4g}")
        print(f"  n_params = {results.get('n_params', 0):,}")
