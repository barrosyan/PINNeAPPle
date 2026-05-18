"""Physics AI Pipeline — unified entry point.

:class:`PhysicsAIPipeline` is the single entry point that orchestrates the
complete flow for building a *generalist physics AI world model* using all
Pinneapple tools.

Pipeline stages
---------------
::

    Stage 1 — Multi-source data generation
        PhysicsDatasetFactory (solver + PINN + symbolic + collocation)
        → DatasetCatalog (organised by scenario and source)

    Stage 2 — Specialist training
        SpecialistTrainer (pinneapple_train + validate + uq + transfer)
        → ModelZoo (one specialist per scenario, with metrics and tags)

    Stage 3 — Meta-learning
        MetaLearner (MAML / Reptile, or pinneapple_meta)
        → meta-initialised PhysicsWorldModel (few-shot adaptable)

    Stage 4 — Foundation model assembly
        Weight-averaging soup of specialists → warm start
        PhysicsFoundationModel (FNO + cross-attention + LoRA)
        Fine-tuned on the full merged catalog
        → PhysicsFoundationModel (mega generalist)

    Stage 5 — Benchmark evaluation
        PhysicsBenchmark (6 standard tasks, conservation checks)
        → BenchmarkResult leaderboard

Two entry points
----------------
* :class:`PhysicsAIPipeline` — opinionated full pipeline (recommended).
* :class:`~.orchestrator.PhysicsOrchestrator` — flexible tool-based solver
  for any physics problem (forward / inverse / design / discovery / …).

Quick start::

    from pinneapple_worldmodel import PhysicsAIPipeline, PhysicsAIConfig

    pipeline = PhysicsAIPipeline(PhysicsAIConfig(
        scenarios=["heat_2d", "burgers_1d", "advection_2d", "ns2d_cavity"],
        n_samples=500,
        device="cuda",
        save_dir="./physics_ai_output",
    ))
    result = pipeline.run()
    mega_model = result.mega_model
    zoo        = result.zoo

    # Solve a new physics problem with the trained model
    from pinneapple_worldmodel import PhysicsOrchestrator, ProblemStatement
    orch = PhysicsOrchestrator()
    res  = orch.solve(ProblemStatement(kind="forward", pde_hint="heat_2d"))
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .scenario import BUILTIN_SCENARIOS, PhysicsScenario
from .dataset import DatasetConfig, DatasetBuilder, WorldModelDataset
from .model import PhysicsWorldModel, WorldModelConfig
from .trainer import WorldModelTrainer, WorldModelTrainConfig
from .curriculum import PhysicsCurriculum, CurriculumConfig
from .dataset_factory import PhysicsDatasetFactory, FactoryConfig, DatasetCatalog
from .model_zoo import ModelZoo, ZooEntry
from .specialist_trainer import SpecialistTrainer, SpecialistConfig
from .meta_learning import MetaLearner, MetaConfig
from .mega_model import PhysicsFoundationModel, FoundationConfig
from .benchmark import PhysicsBenchmark, BenchmarkResult
from .orchestrator import PhysicsOrchestrator, ProblemStatement


# ---------------------------------------------------------------------------
# PhysicsAIConfig
# ---------------------------------------------------------------------------

@dataclass
class PhysicsAIConfig:
    """Complete configuration for :class:`PhysicsAIPipeline`.

    Parameters
    ----------
    scenarios : list of str — physics scenarios to include.
        Subset of BUILTIN_SCENARIOS keys or custom PhysicsScenario objects.
    sources : list of str — data generation routes
        (``"solver"``, ``"pinn"``, ``"symbolic"``, ``"collocation"``).
    n_samples : int — trajectories per scenario per source.
    device : str
    save_dir : str or None — root output directory.

    Specialist training
    -------------------
    specialist_epochs : int
    specialist_lr : float
    use_transfer : bool — fine-tune from existing zoo when available.
    use_uq : bool — attach MC-Dropout during specialist validation.
    validate_physics : bool — conservation-law checks after each specialist.

    Meta-learning
    -------------
    meta_algorithm : ``"reptile"`` | ``"maml"`` | ``"auto"``.
    meta_epochs : int
    skip_meta : bool — skip meta-learning (faster, lower quality).

    Foundation model
    ----------------
    foundation : FoundationConfig — mega-model architecture.
    mega_epochs : int
    mega_lr : float
    skip_foundation : bool — skip mega-model training (use zoo only).

    Benchmark
    ---------
    run_benchmark : bool — run the standard physics AI benchmark at the end.

    Legacy flat mode
    ----------------
    use_curriculum : bool — run the old 5-stage curriculum instead.
    curriculum : CurriculumConfig or None

    Other
    -----
    verbose : bool
    """
    # Core
    scenarios: List[Any] = field(
        default_factory=lambda: ["heat_2d", "burgers_1d", "advection_2d"]
    )
    sources: List[str] = field(default_factory=lambda: ["solver"])
    n_samples: int = 500
    device: str = "cpu"
    save_dir: Optional[str] = None

    # Specialist
    specialist_epochs: int = 50
    specialist_lr: float = 1e-3
    specialist_batch_size: int = 32
    use_transfer: bool = False
    use_uq: bool = False
    validate_physics: bool = False

    # Meta
    meta_algorithm: str = "reptile"
    meta_epochs: int = 200
    skip_meta: bool = False

    # Foundation
    foundation: FoundationConfig = field(default_factory=FoundationConfig)
    mega_epochs: int = 100
    mega_lr: float = 5e-4
    skip_foundation: bool = False

    # Benchmark
    run_benchmark: bool = True

    # Legacy
    use_curriculum: bool = False
    curriculum: Optional[CurriculumConfig] = None

    verbose: bool = True


# ---------------------------------------------------------------------------
# PhysicsAIPipelineResult
# ---------------------------------------------------------------------------

@dataclass
class PhysicsAIPipelineResult:
    """All outputs of :class:`PhysicsAIPipeline.run`.

    Attributes
    ----------
    catalog : DatasetCatalog — multi-source physics datasets.
    zoo : ModelZoo — trained specialist models.
    meta_model : PhysicsWorldModel or None — meta-initialised model.
    mega_model : PhysicsFoundationModel or None — the generalist model.
    benchmark : dict or None — benchmark results.
    elapsed_s : float — total wall-clock time.
    """
    catalog: Optional[DatasetCatalog] = None
    zoo: Optional[ModelZoo] = None
    meta_model: Optional[PhysicsWorldModel] = None
    mega_model: Optional[PhysicsFoundationModel] = None
    benchmark: Optional[Dict[str, BenchmarkResult]] = None
    elapsed_s: float = 0.0

    def summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"{'PHYSICS AI PIPELINE COMPLETE':^60}")
        print(f"{'='*60}")
        print(f"  Wall time      : {self.elapsed_s:.1f}s")
        if self.catalog:
            print(f"  Catalog        : {len(self.catalog):,} total samples, "
                  f"{len(self.catalog.entries)} entries")
        if self.zoo:
            print(f"  Zoo            : {len(self.zoo)} specialists")
        if self.meta_model is not None:
            print(f"  Meta-model     : {self.meta_model.parameter_count():,} params")
        if self.mega_model is not None:
            print(f"  Mega-model     : {self.mega_model.parameter_count():,} params")
        if self.benchmark:
            scores = [r.overall_score() for r in self.benchmark.values()]
            print(f"  Benchmark      : {sum(scores)/len(scores):.3f} mean score "
                  f"({len(scores)} tasks)")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# PhysicsAIPipeline
# ---------------------------------------------------------------------------

class PhysicsAIPipeline:
    """Build a generalist physics AI world model using all Pinneapple tools.

    Parameters
    ----------
    config : PhysicsAIConfig

    Example
    -------
    >>> pipeline = PhysicsAIPipeline(PhysicsAIConfig(
    ...     scenarios=["heat_2d", "burgers_1d", "advection_2d", "ns2d_cavity"],
    ...     n_samples=500,
    ...     device="cuda",
    ...     save_dir="./out",
    ... ))
    >>> result = pipeline.run()
    >>> result.mega_model.rollout(state_0, descriptor, n_steps=20)
    """

    def __init__(self, config: PhysicsAIConfig) -> None:
        self.config = config
        self._result = PhysicsAIPipelineResult()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> PhysicsAIPipelineResult:
        """Execute the full pipeline.

        Returns
        -------
        PhysicsAIPipelineResult
        """
        cfg = self.config
        t0 = time.time()

        if cfg.use_curriculum:
            return self._run_curriculum(t0)

        self._print("Physics AI Pipeline starting …")
        self._print(f"  Scenarios : {cfg.scenarios}")
        self._print(f"  Sources   : {cfg.sources}")
        self._print(f"  n_samples : {cfg.n_samples}")
        self._print(f"  device    : {cfg.device}")

        # Stage 1: Multi-source dataset generation
        self._print("\n━━ Stage 1/5: Multi-source data generation ━━")
        catalog = self._build_catalog()
        self._result.catalog = catalog

        # Stage 2: Specialist training
        self._print("\n━━ Stage 2/5: Specialist model training ━━")
        zoo = self._train_specialists(catalog)
        self._result.zoo = zoo

        # Stage 3: Meta-learning
        if not cfg.skip_meta:
            self._print("\n━━ Stage 3/5: Meta-learning ━━")
            meta_model = self._meta_learn(catalog)
            self._result.meta_model = meta_model
        else:
            self._print("\n━━ Stage 3/5: Meta-learning [skipped] ━━")

        # Stage 4: Foundation model
        if not cfg.skip_foundation:
            self._print("\n━━ Stage 4/5: Physics Foundation Model ━━")
            mega_model = self._build_foundation(zoo, catalog)
            self._result.mega_model = mega_model
        else:
            self._print("\n━━ Stage 4/5: Foundation model [skipped] ━━")

        # Stage 5: Benchmark
        if cfg.run_benchmark:
            self._print("\n━━ Stage 5/5: Benchmark evaluation ━━")
            benchmark = self._run_benchmark()
            self._result.benchmark = benchmark
        else:
            self._print("\n━━ Stage 5/5: Benchmark [skipped] ━━")

        self._result.elapsed_s = time.time() - t0
        if cfg.verbose:
            self._result.summary()

        return self._result

    # ------------------------------------------------------------------
    # Stage 1: Multi-source data generation
    # ------------------------------------------------------------------

    def _build_catalog(self) -> DatasetCatalog:
        cfg = self.config
        factory_cfg = FactoryConfig(
            sources=cfg.sources,
            scenarios=cfg.scenarios,
            n_samples_per_scenario=cfg.n_samples,
            device=cfg.device,
            save_dir=str(Path(cfg.save_dir) / "datasets") if cfg.save_dir else None,
            verbose=cfg.verbose,
        )
        return PhysicsDatasetFactory(factory_cfg).build()

    # ------------------------------------------------------------------
    # Stage 2: Specialist training
    # ------------------------------------------------------------------

    def _train_specialists(self, catalog: DatasetCatalog) -> ModelZoo:
        cfg = self.config
        spec_cfg = SpecialistConfig(
            epochs=cfg.specialist_epochs,
            lr=cfg.specialist_lr,
            batch_size=cfg.specialist_batch_size,
            device=cfg.device,
            use_transfer=cfg.use_transfer,
            use_uq=cfg.use_uq,
            validate_physics=cfg.validate_physics,
            save_dir=str(Path(cfg.save_dir) / "specialists") if cfg.save_dir else None,
            verbose=cfg.verbose,
        )
        return SpecialistTrainer(spec_cfg).train_all(catalog)

    # ------------------------------------------------------------------
    # Stage 3: Meta-learning
    # ------------------------------------------------------------------

    def _meta_learn(self, catalog: DatasetCatalog) -> Optional[PhysicsWorldModel]:
        cfg = self.config
        try:
            meta_cfg = MetaConfig(
                algorithm=cfg.meta_algorithm,
                n_meta_epochs=cfg.meta_epochs,
                device=cfg.device,
                verbose=cfg.verbose,
            )
            learner = MetaLearner(meta_cfg)
            return learner.meta_train(catalog)
        except Exception as exc:
            self._print(f"  [WARN] Meta-learning failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Stage 4: Physics Foundation Model
    # ------------------------------------------------------------------

    def _build_foundation(
        self,
        zoo: ModelZoo,
        catalog: DatasetCatalog,
    ) -> PhysicsFoundationModel:
        cfg = self.config
        ref_ds = next(iter(catalog.entries)).dataset

        # Override context_dim from data
        from dataclasses import replace
        found_cfg = cfg.foundation
        if found_cfg.context_dim != ref_ds.context_dim:
            found_cfg = replace(found_cfg, context_dim=ref_ds.context_dim)

        mega = PhysicsFoundationModel(
            found_cfg,
            n_fields=ref_ds.n_fields,
            grid_shape=ref_ds.grid_shape,
        )
        self._print(f"  Foundation model: {mega}")

        # Warm-start from specialist soup
        if len(zoo) > 0:
            try:
                soup = zoo.soup()
                mega.load_state_dict(soup.state_dict(), strict=False)
                self._print(f"  ✓ Warm-started from weight-averaged soup of {len(zoo)} specialists.")
            except Exception as exc:
                self._print(f"  [WARN] Soup warm-start failed: {exc}")

        # Fine-tune on merged catalog
        merged_ds = catalog.merged()
        train_cfg = WorldModelTrainConfig(
            epochs=cfg.mega_epochs,
            lr=cfg.mega_lr,
            batch_size=cfg.specialist_batch_size,
            device=cfg.device,
            patience=30,
            log_every=max(1, cfg.mega_epochs // 5),
        )
        trainer = WorldModelTrainer(mega, train_cfg)
        history = trainer.fit(merged_ds)

        best_val = min(
            (h.get("val_total", float("inf")) for h in history),
            default=float("inf"),
        )
        self._print(f"  Foundation model trained — best val={best_val:.4g}")

        if cfg.save_dir:
            path = str(Path(cfg.save_dir) / "mega_model.pt")
            mega.save(path)
            self._print(f"  Saved → {path}")

        return mega

    # ------------------------------------------------------------------
    # Stage 5: Benchmark
    # ------------------------------------------------------------------

    def _run_benchmark(self) -> Optional[Dict[str, BenchmarkResult]]:
        cfg = self.config
        model = self._result.mega_model or (
            self._result.zoo.get(cfg.scenarios[0])
            if self._result.zoo and cfg.scenarios
            and isinstance(cfg.scenarios[0], str)
            and cfg.scenarios[0] in self._result.zoo
            else None
        )
        if model is None:
            self._print("  [WARN] No model available for benchmark.")
            return None

        bench = PhysicsBenchmark(device=cfg.device, verbose=cfg.verbose)
        results = bench.run(model)
        if cfg.verbose:
            bench.print_report(results)
        return results

    # ------------------------------------------------------------------
    # Legacy curriculum mode
    # ------------------------------------------------------------------

    def _run_curriculum(self, t0: float) -> PhysicsAIPipelineResult:
        cfg = self.config
        cur_cfg = cfg.curriculum or CurriculumConfig(
            device=cfg.device,
            verbose=cfg.verbose,
        )
        curriculum = PhysicsCurriculum(cur_cfg)
        model = curriculum.run()

        self._result.elapsed_s = time.time() - t0
        if cfg.verbose:
            self._result.summary()
        return self._result

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _print(self, msg: str) -> None:
        if self.config.verbose:
            print(f"[PhysicsAIPipeline] {msg}")

    @property
    def mega_model(self) -> Optional[PhysicsFoundationModel]:
        return self._result.mega_model

    @property
    def zoo(self) -> Optional[ModelZoo]:
        return self._result.zoo

    @property
    def catalog(self) -> Optional[DatasetCatalog]:
        return self._result.catalog

    # ------------------------------------------------------------------
    # Convenience: orchestrate a specific physics problem
    # ------------------------------------------------------------------

    @staticmethod
    def orchestrate(statement: ProblemStatement) -> Any:
        """Shortcut to :class:`~.orchestrator.PhysicsOrchestrator`.

        Parameters
        ----------
        statement : ProblemStatement

        Returns
        -------
        OrchestratorResult
        """
        return PhysicsOrchestrator().solve(statement)
