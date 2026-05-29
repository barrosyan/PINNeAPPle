"""Physics World Model training entry point.

Supports four training modes, building on top of the worldmodel's existing
Trainer / SpecialistTrainer / MetaLearner / PhysicsFoundationModel stack.

Modes
-----
specialist   Train one model per scenario from the catalog.
             Output: ModelZoo (one ZooEntry per scenario).

meta         Meta-train (MAML or Reptile) over the full catalog.
             Output: meta-initialised PhysicsWorldModel.

foundation   Assemble and fine-tune the PhysicsFoundationModel (mega-model)
             using LoRA adapters on top of a pre-trained base or from scratch.
             Output: PhysicsFoundationModel.

pipeline     Run the complete 5-stage PINNeAPPle pipeline end-to-end:
             generate → specialist → meta → foundation → benchmark.
             Output: PhysicsAIPipelineResult.

CLI usage
---------
    # Specialist training from a pre-generated catalog
    python -m pinneapple_worldmodel.train \\
        --mode specialist \\
        --catalog ./data/worldmodel/catalog.pkl \\
        --output  ./checkpoints/worldmodel

    # Generate data on-the-fly + train specialist
    python -m pinneapple_worldmodel.train \\
        --mode specialist \\
        --scenarios burgers_1d heat_2d ns2d_cavity \\
        --n-samples 200 --epochs 100

    # Meta-learning (Reptile, default)
    python -m pinneapple_worldmodel.train \\
        --mode meta \\
        --catalog ./data/worldmodel/catalog.pkl \\
        --meta-algorithm reptile --meta-epochs 200

    # Foundation model (LoRA fine-tune from existing base)
    python -m pinneapple_worldmodel.train \\
        --mode foundation \\
        --catalog  ./data/worldmodel/catalog.pkl \\
        --base-ckpt ./checkpoints/worldmodel/meta_model.pt

    # Full pipeline (generate + all training stages)
    python -m pinneapple_worldmodel.train --mode pipeline \\
        --scenarios all --n-samples 100

    # Smoke test (fast end-to-end check)
    python -m pinneapple_worldmodel.train --smoke-test
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .scenario import BUILTIN_SCENARIOS
from .dataset import WorldModelDataset
from .dataset_factory import DatasetCatalog
from .model import PhysicsWorldModel, WorldModelConfig
from .trainer import WorldModelTrainer, WorldModelTrainConfig
from .specialist_trainer import SpecialistTrainer, SpecialistConfig
from .meta_learning import MetaLearner, MetaConfig
from .mega_model import PhysicsFoundationModel, FoundationConfig
from .benchmark import PhysicsBenchmark, BUILTIN_TASKS
from .pipeline import PhysicsAIPipeline, PhysicsAIConfig

log = logging.getLogger(__name__)


# ── mode implementations ──────────────────────────────────────────────────────

def train_specialist(
    catalog: DatasetCatalog,
    output_dir: Path,
    *,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    patience: int = 10,
    n_modes: int = 16,
    width: int = 64,
    depth: int = 4,
    rollout_steps: int = 1,
    checkpoint_interval: int = 10,
    run_benchmark: bool = True,
) -> "ModelZoo":
    """Train one specialist model per scenario in the catalog.

    Parameters
    ----------
    catalog : DatasetCatalog
        Pre-built catalog (use ``generate_datasets.generate()`` to create one).
    output_dir : Path
        Directory to save checkpoints and the final zoo.
    """
    from .model_zoo import ModelZoo

    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Specialist training — %d scenarios", len(catalog.datasets_by_scenario()))

    model_cfg = WorldModelConfig(
        n_modes=n_modes,
        width=width,
        depth=depth,
        rollout_steps=rollout_steps,
    )
    sp_cfg = SpecialistConfig(
        model=model_cfg,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        patience=patience,
        rollout_steps=rollout_steps,
        save_dir=str(output_dir / "specialist_ckpts"),
    )

    trainer = SpecialistTrainer(sp_cfg)
    zoo = trainer.train_all(catalog)

    zoo_path = output_dir / "model_zoo.pkl"
    zoo.save(str(zoo_path))
    log.info("ModelZoo saved → %s  (%d models)", zoo_path, len(zoo.list_names()))

    if run_benchmark:
        _run_benchmark(zoo, catalog, output_dir, device)

    return zoo


def train_meta(
    catalog: DatasetCatalog,
    output_dir: Path,
    *,
    algorithm: str = "reptile",
    n_meta_epochs: int = 100,
    n_inner_steps: int = 5,
    inner_lr: float = 1e-2,
    outer_lr: float = 1e-3,
    n_tasks_per_batch: int = 4,
    device: str = "cpu",
    warm_start_path: Optional[str] = None,
    n_modes: int = 16,
    width: int = 64,
    depth: int = 4,
    run_benchmark: bool = True,
) -> PhysicsWorldModel:
    """Meta-train a shared initialisation over the full scenario catalog.

    Parameters
    ----------
    catalog : DatasetCatalog
    algorithm : str
        ``"maml"``, ``"reptile"``, or ``"auto"`` (choose by dataset size).
    warm_start_path : str | None
        Optional path to a checkpoint to initialise meta-training from.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_cfg = MetaConfig(
        algorithm=algorithm,
        n_meta_epochs=n_meta_epochs,
        n_inner_steps=n_inner_steps,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        n_tasks_per_batch=n_tasks_per_batch,
        device=device,
    )
    model_cfg = WorldModelConfig(n_modes=n_modes, width=width, depth=depth)

    learner = MetaLearner(meta_cfg, model_cfg)

    warm_start = None
    if warm_start_path:
        log.info("Loading warm-start checkpoint: %s", warm_start_path)
        warm_start = WorldModelTrainer.load_checkpoint(warm_start_path)

    log.info("Meta-training (%s) for %d epochs …", algorithm.upper(), n_meta_epochs)
    meta_model = learner.meta_train(catalog, warm_start=warm_start)

    ckpt_path = output_dir / "meta_model.pt"
    torch.save(meta_model.state_dict(), ckpt_path)
    log.info("Meta-model saved → %s", ckpt_path)

    if run_benchmark:
        _run_benchmark(meta_model, catalog, output_dir, device, prefix="meta")

    return meta_model


def train_foundation(
    catalog: DatasetCatalog,
    output_dir: Path,
    *,
    base_ckpt: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 5e-4,
    device: str = "cpu",
    lora_rank: int = 8,
    n_modes: int = 16,
    width: int = 64,
    depth: int = 4,
    n_heads: int = 4,
    n_context_layers: int = 2,
    run_benchmark: bool = True,
) -> PhysicsFoundationModel:
    """Build and fine-tune the PhysicsFoundationModel (mega-model with LoRA).

    If ``base_ckpt`` is provided, the FNO backbone is loaded from it and only
    the LoRA adapters + context encoder are trained.  Without it the model
    trains from scratch.

    Parameters
    ----------
    base_ckpt : str | None
        Path to a ``meta_model.pt`` or ``specialist`` checkpoint to transfer
        FNO weights from.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer field / grid shape from catalog
    first_ds = catalog.merged()
    sample = first_ds[0]
    state_shape = sample[0].shape  # (C, *grid)
    n_fields    = state_shape[0]

    fnd_cfg = FoundationConfig(
        n_modes=n_modes,
        width=width,
        depth=depth,
        context_dim=64,
        descriptor_vocab=128,
        n_heads=n_heads,
        n_context_layers=n_context_layers,
        lora_rank=lora_rank,
    )
    mega_model = PhysicsFoundationModel(fnd_cfg, n_fields=n_fields)

    if base_ckpt:
        log.info("Loading base FNO weights from: %s", base_ckpt)
        state = torch.load(base_ckpt, map_location="cpu")
        # Partial load — only backbone weights, LoRA stays random
        missing, unexpected = mega_model.load_state_dict(state, strict=False)
        if missing:
            log.info("  Missing keys (new LoRA layers): %d", len(missing))
        # Freeze base, train only LoRA + context encoder
        mega_model.freeze_backbone()

    mega_model = mega_model.to(device)

    # Train using WorldModelTrainer with the merged catalog dataset
    train_cfg = WorldModelTrainConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    merged_ds = catalog.merged()
    trainer   = WorldModelTrainer(mega_model, train_cfg)
    history   = trainer.fit(merged_ds)

    ckpt_path = output_dir / "foundation_model.pt"
    torch.save(mega_model.state_dict(), ckpt_path)
    log.info("Foundation model saved → %s", ckpt_path)

    if run_benchmark:
        _run_benchmark(mega_model, catalog, output_dir, device, prefix="foundation")

    return mega_model


def run_pipeline(
    scenarios: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    n_samples: int = 100,
    output_dir: str = "./checkpoints/worldmodel",
    device: str = "cpu",
    epochs: int = 50,
    meta_epochs: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> "PhysicsAIPipelineResult":
    """Run the complete 5-stage Physics AI pipeline.

    Stages: dataset generation → specialist training → meta-learning →
    foundation model → benchmark.

    Parameters
    ----------
    scenarios : list[str] | None
        Scenarios to include. Defaults to all built-ins.
    sources : list[str] | None
        Data sources. Defaults to ``["solver"]``.
    n_samples : int
        Samples per (scenario, source).
    epochs : int
        Specialist training epochs.
    meta_epochs : int
        Meta-learning epochs.
    """
    scenarios = scenarios or list(BUILTIN_SCENARIOS.keys())
    sources   = sources   or ["solver"]

    cfg = PhysicsAIConfig(
        scenarios=scenarios,
        sources=sources,
        n_samples_per_scenario=n_samples,
        device=device,
        output_dir=output_dir,
        seed=seed,
        # Specialist
        specialist_epochs=epochs,
        # Meta
        meta_epochs=meta_epochs,
        meta_algorithm="reptile",
        # Foundation
        foundation_epochs=max(epochs // 2, 10),
        lora_rank=8,
        # Benchmark
        run_benchmark=True,
    )

    pipeline = PhysicsAIPipeline(cfg)
    log.info("Running full 5-stage pipeline …")
    result = pipeline.run()

    if verbose:
        result.summary()

    return result


# ── benchmark helper ──────────────────────────────────────────────────────────

def _run_benchmark(model, catalog, output_dir: Path, device: str, prefix: str = "") -> None:
    try:
        bench = PhysicsBenchmark(device=device)
        tasks = {k: v for k, v in BUILTIN_TASKS.items()
                 if v.scenario_name in catalog.datasets_by_scenario()}

        if not tasks:
            log.info("No matching benchmark tasks — skipping")
            return

        # Build a simple context function (scenario name → context vector)
        def _ctx(scenario_name: str, n: int) -> torch.Tensor:
            return torch.zeros(n, 64, device=device)  # placeholder context

        log.info("Running benchmark on %d tasks …", len(tasks))
        results = bench.run(model, tasks, context_fn=_ctx)
        bench.print_report(results)

        import json
        scores = {k: v.overall_score() for k, v in results.items()}
        report_path = output_dir / f"{prefix}_benchmark.json".lstrip("_")
        with open(report_path, "w") as f:
            json.dump(scores, f, indent=2)
        log.info("Benchmark report → %s", report_path)
    except Exception as exc:
        log.warning("Benchmark failed (non-fatal): %s", exc)


# ── dataset loading helper ────────────────────────────────────────────────────

def _load_or_generate_catalog(
    catalog_path: Optional[str],
    scenarios: Optional[List[str]],
    sources: List[str],
    n_samples: int,
    output_dir: Path,
    device: str,
    seed: int,
) -> DatasetCatalog:
    """Load an existing catalog or generate a fresh one."""
    if catalog_path and Path(catalog_path).exists():
        log.info("Loading catalog from: %s", catalog_path)
        with open(catalog_path, "rb") as f:
            return pickle.load(f)

    log.info("No catalog found — generating data …")
    from .generate_datasets import generate
    return generate(
        scenarios=scenarios,
        sources=sources,
        n_samples=n_samples,
        output_dir=str(output_dir / "data"),
        device=device,
        seed=seed,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Physics World Models with PINNeAPPle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── mode ──
    p.add_argument(
        "--mode",
        choices=["specialist", "meta", "foundation", "pipeline"],
        default="specialist",
        help="Training mode.",
    )

    # ── data ──
    data = p.add_argument_group("Data")
    data.add_argument("--catalog", default=None,
                      help="Path to a pre-built DatasetCatalog (.pkl). "
                           "If omitted, data is generated on-the-fly.")
    data.add_argument("--scenarios", nargs="+", default=None,
                      metavar="NAME",
                      help="Scenario names (default: all built-ins).")
    data.add_argument("--sources", nargs="+", default=["solver"],
                      choices=["solver", "pinn", "symbolic", "collocation"])
    data.add_argument("--n-samples", type=int, default=100,
                      help="Samples per (scenario, source) when generating data.")

    # ── model ──
    arch = p.add_argument_group("Architecture")
    arch.add_argument("--n-modes", type=int, default=16)
    arch.add_argument("--width",   type=int, default=64)
    arch.add_argument("--depth",   type=int, default=4)
    arch.add_argument("--lora-rank", type=int, default=8,
                      help="LoRA rank for foundation mode.")

    # ── training ──
    trn = p.add_argument_group("Training")
    trn.add_argument("--epochs",         type=int,   default=50)
    trn.add_argument("--batch-size",     type=int,   default=32)
    trn.add_argument("--lr",             type=float, default=1e-3)
    trn.add_argument("--device",         default="cpu")
    trn.add_argument("--seed",           type=int,   default=42)
    trn.add_argument("--patience",       type=int,   default=10)
    trn.add_argument("--rollout-steps",  type=int,   default=1)
    trn.add_argument("--checkpoint-interval", type=int, default=10)

    # ── meta-learning ──
    meta = p.add_argument_group("Meta-learning (--mode meta)")
    meta.add_argument("--meta-algorithm", default="reptile",
                      choices=["maml", "reptile", "auto"])
    meta.add_argument("--meta-epochs",    type=int,   default=100)
    meta.add_argument("--inner-steps",    type=int,   default=5)
    meta.add_argument("--inner-lr",       type=float, default=1e-2)
    meta.add_argument("--outer-lr",       type=float, default=1e-3)
    meta.add_argument("--n-tasks",        type=int,   default=4)

    # ── foundation ──
    fnd = p.add_argument_group("Foundation model (--mode foundation)")
    fnd.add_argument("--base-ckpt", default=None,
                     help="Checkpoint to load FNO backbone from (e.g. meta_model.pt).")

    # ── output ──
    p.add_argument("--output", default="./checkpoints/worldmodel",
                   help="Output directory for checkpoints and reports.")
    p.add_argument("--no-benchmark", action="store_true",
                   help="Skip benchmark evaluation after training.")

    p.add_argument("--smoke-test", action="store_true",
                   help="Quick smoke test: 2 scenarios, 10 samples, 3 epochs.")

    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = _parse_args()

    # Smoke-test overrides
    if args.smoke_test:
        args.scenarios  = ["burgers_1d", "heat_2d"]
        args.n_samples  = 10
        args.epochs     = 3
        args.meta_epochs = 3
        args.mode       = args.mode  # keep chosen mode
        log.info("Smoke-test mode: 2 scenarios, 10 samples, 3 epochs")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_bench = not args.no_benchmark

    # ── pipeline mode: delegate entirely ──
    if args.mode == "pipeline":
        run_pipeline(
            scenarios=args.scenarios,
            sources=args.sources,
            n_samples=args.n_samples,
            output_dir=args.output,
            device=args.device,
            epochs=args.epochs,
            meta_epochs=args.meta_epochs,
            seed=args.seed,
        )
        return

    # ── load / generate catalog ──
    catalog = _load_or_generate_catalog(
        catalog_path=args.catalog,
        scenarios=args.scenarios,
        sources=args.sources,
        n_samples=args.n_samples,
        output_dir=output_dir,
        device=args.device,
        seed=args.seed,
    )

    t0 = time.perf_counter()

    if args.mode == "specialist":
        train_specialist(
            catalog=catalog,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            patience=args.patience,
            n_modes=args.n_modes,
            width=args.width,
            depth=args.depth,
            rollout_steps=args.rollout_steps,
            checkpoint_interval=args.checkpoint_interval,
            run_benchmark=run_bench,
        )

    elif args.mode == "meta":
        warm_ckpt = None
        # Check if a previous specialist zoo was saved in output_dir
        zoo_path = output_dir / "model_zoo.pkl"
        if zoo_path.exists() and not args.base_ckpt:
            log.info("Found specialist zoo at %s — using as warm start", zoo_path)
            warm_ckpt = str(zoo_path)

        train_meta(
            catalog=catalog,
            output_dir=output_dir,
            algorithm=args.meta_algorithm,
            n_meta_epochs=args.meta_epochs,
            n_inner_steps=args.inner_steps,
            inner_lr=args.inner_lr,
            outer_lr=args.outer_lr,
            n_tasks_per_batch=args.n_tasks,
            device=args.device,
            warm_start_path=warm_ckpt or args.base_ckpt,
            n_modes=args.n_modes,
            width=args.width,
            depth=args.depth,
            run_benchmark=run_bench,
        )

    elif args.mode == "foundation":
        train_foundation(
            catalog=catalog,
            output_dir=output_dir,
            base_ckpt=args.base_ckpt,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            lora_rank=args.lora_rank,
            n_modes=args.n_modes,
            width=args.width,
            depth=args.depth,
            run_benchmark=run_bench,
        )

    elapsed = time.perf_counter() - t0
    log.info("Training complete in %.1f s", elapsed)


if __name__ == "__main__":
    main()
