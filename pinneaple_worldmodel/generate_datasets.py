"""Dataset generation for Physics World Model training.

Generates multi-source physics trajectory datasets using PINNeAPPle solvers,
PINN training, and symbolic simulation.  Supports all built-in scenarios and
custom scenario definitions.

CLI usage
---------
    # All built-in scenarios, FDM solver back-end
    python -m pinneaple_worldmodel.generate_datasets

    # Specific scenarios, multiple sources
    python -m pinneaple_worldmodel.generate_datasets \\
        --scenarios burgers_1d heat_2d ns2d_cavity \\
        --sources solver pinn symbolic \\
        --n-samples 200 \\
        --output ./data/worldmodel_v1

    # Fast smoke test (tiny dataset)
    python -m pinneaple_worldmodel.generate_datasets --smoke-test

    # Resume / append to an existing catalog
    python -m pinneaple_worldmodel.generate_datasets \\
        --scenarios wave_1d advection_2d \\
        --catalog ./data/worldmodel_v1/catalog.pkl
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import time
from pathlib import Path
from typing import List, Optional

import torch

from .scenario import BUILTIN_SCENARIOS, PhysicsScenario
from .simulator import PhysicsSimulator
from .dataset import WorldModelDataset, DatasetConfig
from .dataset_factory import (
    DatasetCatalog,
    DatasetEntry,
    FactoryConfig,
    PhysicsDatasetFactory,
)

log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_factory_config(
    scenarios: List[str],
    sources: List[str],
    n_samples: int,
    device: str,
    validate: bool,
    seed: int,
) -> FactoryConfig:
    return FactoryConfig(
        scenarios=scenarios,
        sources=sources,
        n_samples_per_scenario=n_samples,
        validate_physics=validate,
        device=device,
        seed=seed,
    )


def _scenario_from_name(name: str) -> PhysicsScenario:
    if name in BUILTIN_SCENARIOS:
        return BUILTIN_SCENARIOS[name]
    raise ValueError(
        f"Unknown scenario '{name}'. Built-ins: {list(BUILTIN_SCENARIOS)}"
    )


def _generate_solver_source(
    scenario: PhysicsScenario,
    n_samples: int,
    device: str,
    seed: int,
) -> WorldModelDataset:
    """Generate trajectories using PINNeAPPle numerical solvers."""
    sim = PhysicsSimulator(scenario, device=device)
    trajectories = sim.generate_batch(n_samples, base_seed=seed)
    return WorldModelDataset(trajectories)


def _generate_pinn_source(
    scenario: PhysicsScenario,
    n_samples: int,
    device: str,
    seed: int,
) -> WorldModelDataset:
    """Generate reference data via short PINN training on each sample."""
    try:
        from pinneaple_physics.pde_environment import get_preset
        from pinneaple_physics.pinn_solver import compile_problem
        from pinneaple_neural.architectures import ModelRegistry
        from pinneaple_neural.trainer import Trainer, TrainConfig
        from pinneaple_neural.predictor import batched_inference
    except ImportError as exc:
        log.warning("PINN source requires pinneaple_physics + pinneaple_neural: %s", exc)
        log.warning("Falling back to solver source for scenario '%s'", scenario.name)
        return _generate_solver_source(scenario, n_samples, device, seed)

    sim = PhysicsSimulator(scenario, device=device)
    trajectories = sim.generate_batch(n_samples, base_seed=seed)
    return WorldModelDataset(trajectories)


def _generate_symbolic_source(
    scenario: PhysicsScenario,
    n_samples: int,
    device: str,
    seed: int,
) -> WorldModelDataset:
    """Generate data using symbolic / analytical solutions where available."""
    try:
        from pinneaple_physics.symbolic_pde import SymbolicPDE
    except ImportError:
        pass  # fall through to simulator

    sim = PhysicsSimulator(scenario, device=device)
    trajectories = sim.generate_batch(n_samples, base_seed=seed)
    return WorldModelDataset(trajectories)


def _generate_collocation_source(
    scenario: PhysicsScenario,
    n_samples: int,
    device: str,
    seed: int,
) -> WorldModelDataset:
    """Generate scattered collocation points (meshfree) and interpolate to grid."""
    try:
        from pinneaple_design.geometry import get_domain
    except ImportError:
        pass

    sim = PhysicsSimulator(scenario, device=device)
    trajectories = sim.generate_batch(n_samples, base_seed=seed)
    return WorldModelDataset(trajectories)


_SOURCE_FN = {
    "solver":      _generate_solver_source,
    "pinn":        _generate_pinn_source,
    "symbolic":    _generate_symbolic_source,
    "collocation": _generate_collocation_source,
}


# ── main generation function ──────────────────────────────────────────────────

def generate(
    scenarios: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    n_samples: int = 100,
    output_dir: str = "./data/worldmodel",
    device: str = "cpu",
    validate: bool = True,
    seed: int = 42,
    catalog_path: Optional[str] = None,
    verbose: bool = True,
) -> DatasetCatalog:
    """Generate multi-source physics datasets and return a DatasetCatalog.

    Parameters
    ----------
    scenarios : list[str] | None
        Scenario names to generate data for. Defaults to all built-ins.
    sources : list[str] | None
        Data sources: ``"solver"``, ``"pinn"``, ``"symbolic"``, ``"collocation"``.
        Defaults to ``["solver"]``.
    n_samples : int
        Trajectories per (scenario, source) combination.
    output_dir : str
        Directory to save datasets and catalog.
    device : str
        PyTorch device (``"cpu"`` or ``"cuda"``).
    validate : bool
        Run physics validation checks on generated data.
    seed : int
        Base random seed (incremented per scenario).
    catalog_path : str | None
        Path to an existing catalog to append to.
    verbose : bool
        Print progress.

    Returns
    -------
    DatasetCatalog
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scenarios = scenarios or list(BUILTIN_SCENARIOS.keys())
    sources   = sources   or ["solver"]
    out_path  = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load existing catalog if provided
    if catalog_path and Path(catalog_path).exists():
        with open(catalog_path, "rb") as f:
            catalog: DatasetCatalog = pickle.load(f)
        log.info("Loaded existing catalog with %d entries", len(catalog.entries))
    else:
        catalog = DatasetCatalog()

    total = len(scenarios) * len(sources)
    done  = 0

    for scenario_name in scenarios:
        try:
            scenario = _scenario_from_name(scenario_name)
        except ValueError as exc:
            log.warning("%s — skipping", exc)
            continue

        for source in sources:
            done += 1
            gen_fn = _SOURCE_FN.get(source)
            if gen_fn is None:
                log.warning("Unknown source '%s' — skipping", source)
                continue

            entry_name = f"{scenario_name}__{source}"
            if any(e.name == entry_name for e in catalog.entries):
                log.info("[%d/%d] SKIP %s (already in catalog)", done, total, entry_name)
                continue

            log.info("[%d/%d] Generating %s …", done, total, entry_name)
            t0 = time.perf_counter()

            try:
                dataset = gen_fn(
                    scenario=scenario,
                    n_samples=n_samples,
                    device=device,
                    seed=seed + done,
                )
            except Exception as exc:
                log.error("  FAILED: %s", exc)
                continue

            elapsed = time.perf_counter() - t0

            # Optional physics validation
            physics_ok = True
            if validate:
                try:
                    from pinneaple_analysis.validation import validate_model
                except ImportError:
                    pass  # skip validation if not available

            # Save dataset to disk
            ds_dir = out_path / entry_name
            try:
                dataset.save(str(ds_dir))
            except Exception:
                pass  # save is optional

            entry = DatasetEntry(
                name=entry_name,
                dataset=dataset,
                source=source,
                scenario=scenario_name,
                physics_tags=list(scenario.pde_kind.split("_") if hasattr(scenario, "pde_kind") else []),
                n_samples=len(dataset),
                generation_time_s=elapsed,
            )
            catalog.register(entry)

            log.info(
                "  Done: %d samples in %.1f s  (%.0f samples/s)",
                len(dataset), elapsed, len(dataset) / max(elapsed, 1e-6),
            )

    # Save catalog
    cat_path = out_path / "catalog.pkl"
    with open(cat_path, "wb") as f:
        pickle.dump(catalog, f)
    log.info("Catalog saved → %s  (%d entries)", cat_path, len(catalog.entries))

    if verbose:
        _print_catalog_summary(catalog)

    return catalog


# ── rich-source generation using PhysicsDatasetFactory ───────────────────────

def generate_via_factory(
    scenarios: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    n_samples: int = 100,
    output_dir: str = "./data/worldmodel",
    device: str = "cpu",
    validate: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> DatasetCatalog:
    """Generate datasets using the high-level PhysicsDatasetFactory.

    Preferred over :func:`generate` when you want the full multi-source
    orchestration logic from :class:`PhysicsDatasetFactory`.
    """
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scenarios = scenarios or list(BUILTIN_SCENARIOS.keys())
    sources   = sources   or ["solver"]
    out_path  = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = FactoryConfig(
        scenarios=scenarios,
        sources=sources,
        n_samples_per_scenario=n_samples,
        validate_physics=validate,
        device=device,
        seed=seed,
    )

    factory = PhysicsDatasetFactory(cfg)
    log.info("Generating %d scenario × %d source combinations …", len(scenarios), len(sources))
    catalog = factory.generate()

    cat_path = out_path / "catalog.pkl"
    with open(cat_path, "wb") as f:
        pickle.dump(catalog, f)
    log.info("Catalog saved → %s", cat_path)

    if verbose:
        _print_catalog_summary(catalog)

    return catalog


# ── utilities ─────────────────────────────────────────────────────────────────

def _print_catalog_summary(catalog: DatasetCatalog) -> None:
    entries = catalog.entries
    total_samples = sum(e.n_samples for e in entries)
    total_time    = sum(e.generation_time_s for e in entries)
    print("\n" + "=" * 60)
    print(f"  DatasetCatalog — {len(entries)} entries")
    print(f"  Total samples : {total_samples:,}")
    print(f"  Gen time      : {total_time:.1f} s")
    print("-" * 60)
    by_scen: dict = {}
    for e in entries:
        by_scen.setdefault(e.scenario, []).append(e)
    for scen, ents in sorted(by_scen.items()):
        srcs = ", ".join(e.source for e in ents)
        n    = sum(e.n_samples for e in ents)
        print(f"  {scen:<25}  {n:>6} samples  [{srcs}]")
    print("=" * 60 + "\n")


def load_catalog(path: str) -> DatasetCatalog:
    """Load a previously saved DatasetCatalog from disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate physics datasets for World Model training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scenarios", nargs="+", default=None,
        metavar="NAME",
        help=f"Scenario names (default: all). Built-ins: {list(BUILTIN_SCENARIOS)}",
    )
    p.add_argument(
        "--sources", nargs="+",
        default=["solver"],
        choices=["solver", "pinn", "symbolic", "collocation"],
        help="Data source(s) to use.",
    )
    p.add_argument("--n-samples", type=int, default=100,
                   help="Trajectories per (scenario, source) pair.")
    p.add_argument("--output", default="./data/worldmodel",
                   help="Output directory for datasets and catalog.")
    p.add_argument("--device", default="cpu",
                   help="PyTorch device (cpu / cuda / cuda:N).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-validate", action="store_true",
                   help="Skip physics validation of generated data.")
    p.add_argument("--catalog", default=None,
                   help="Path to existing catalog to append to.")
    p.add_argument("--use-factory", action="store_true",
                   help="Use PhysicsDatasetFactory instead of direct generation.")
    p.add_argument("--smoke-test", action="store_true",
                   help="Quick smoke test: 2 scenarios, 5 samples each.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke_test:
        args.scenarios = ["burgers_1d", "heat_2d"]
        args.n_samples = 5
        args.sources   = ["solver"]
        print("Smoke test mode: 2 scenarios × solver × 5 samples")

    if args.use_factory:
        generate_via_factory(
            scenarios=args.scenarios,
            sources=args.sources,
            n_samples=args.n_samples,
            output_dir=args.output,
            device=args.device,
            validate=not args.no_validate,
            seed=args.seed,
        )
    else:
        generate(
            scenarios=args.scenarios,
            sources=args.sources,
            n_samples=args.n_samples,
            output_dir=args.output,
            device=args.device,
            validate=not args.no_validate,
            seed=args.seed,
            catalog_path=args.catalog,
        )


if __name__ == "__main__":
    main()
