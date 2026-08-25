from __future__ import annotations

from pathlib import Path
import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT
DOCS_API_DIR = Path("api")

TARGET_PACKAGES = [
    "pinneapple",
    "pinneapple_physics",
    "pinneapple_neural",
    "pinneapple_analysis",
    "pinneapple_adaptation",
    "pinneapple_simulation",
    "pinneapple_systems",
    "pinneapple_design",
    "pinneapple_tools",
    "pinneapple_data",
    "pinneapple_pdb",
    "pinneapple_problemdesign",
    "pinneapple_arena",
    "pinneapple_quantum",
    "pinneapple_worldmodel",
    "pinneapple_models",
    "pinneapple_solvers",
    "pinneapple_train",
]

PACKAGE_OVERVIEWS = {
    "pinneapple": (
        "Top-level convenience API.\n\n"
        "Quick-start entry point re-exporting the pieces most users need first: "
        "listing problem presets, building models, and solving a PDE in a few lines."
    ),
    "pinneapple_physics": (
        "PDE definitions, PINN loss compiler, and symbolic differentiation.\n\n"
        "Covers pde_environment (problem specs, BCs/ICs, canonical presets), "
        "pinn_solver (PINN residual compiler, domain decomposition), and "
        "symbolic_pde (SymPy → autograd residual compiler)."
    ),
    "pinneapple_neural": (
        "Neural network architectures, training, and inference for physics.\n\n"
        "Covers architectures (PINNs, neural operators, GNNs, transformers, ROMs, "
        "reservoir computing, classical time-series models), trainer (distributed/"
        "causal/two-phase training), and predictor (batched inference, flow "
        "visualization)."
    ),
    "pinneapple_analysis": (
        "Uncertainty quantification, physics validation, and inverse problems.\n\n"
        "Covers uncertainty (MC-dropout, ensembles, conformal prediction), "
        "validation (conservation/BC/symmetry checks against reference solutions), "
        "and inverse_problems (noise models, regularizers, EKI, SINDy)."
    ),
    "pinneapple_adaptation": (
        "Transfer and meta-learning across PDE families.\n\n"
        "Covers transfer_learning (fine-tuning, layer freezing) and meta_learning "
        "(MAML, Reptile, few-shot task sampling)."
    ),
    "pinneapple_simulation": (
        "Classical numerical solvers and external-solver bridges.\n\n"
        "Covers numerical_solvers (FEM/FDM/FVM/spectral/SPH/LBM), particle_dynamics "
        "(MPM, SPH particles, rigid-body), and external_solvers (OpenFOAM, FEniCS, "
        "Modelica, MATLAB, MuJoCo, Genesis, TurboDesigner bridges)."
    ),
    "pinneapple_systems": (
        "Time series, co-simulation, and digital twins.\n\n"
        "Covers time_series (forecasting models and decomposition), cosimulation "
        "(graph-based multi-model coupling), and digital_twin (live state "
        "assimilation via EKF/EnKF, anomaly detection)."
    ),
    "pinneapple_design": (
        "Geometry representation and design optimization.\n\n"
        "Covers geometry (SDFs, CSG, mesh I/O, parametric shapes) and "
        "design_optimizer (adjoint, Pareto, Bayesian/evolutionary optimization)."
    ),
    "pinneapple_tools": (
        "Visualization, export, benchmarking, and compute backends.\n\n"
        "Covers visualization (CFD-style plots, streamlines, Q-criterion), "
        "model_export (TorchScript/ONNX), hpo_experiments (hyperparameter search, "
        "paper discovery), benchmark_suite (the Arena runner), and compute_backends "
        "(PyTorch/JAX abstraction)."
    ),
    "pinneapple_data": (
        "Unified Physical Data (UPD) format and dataset utilities.\n\n"
        "The PhysicalSample container (state/geometry/schema/domain/provenance), "
        "dataset adapters, synthetic data generators, and storage backends "
        "(Zarr, HDF5, PyTorch)."
    ),
    "pinneapple_pdb": (
        "Structured physics database.\n\n"
        "Builds and validates physical datasets from Earth-observation sources "
        "(NASA CMR / earthaccess) and derives physical quantities (vorticity, "
        "divergence) from gridded fields."
    ),
    "pinneapple_problemdesign": (
        "Natural-language-to-PDE problem design agent.\n\n"
        "Elicits a physics problem from a natural-language description and "
        "generates a runnable PINN pipeline (elicitation, knowledge base, "
        "code generation, report rendering)."
    ),
    "pinneapple_arena": (
        "Benchmark problems and leaderboard.\n\n"
        "Canonical PDEs with analytical or reference solutions, used to compare "
        "models and solvers on equal footing."
    ),
    "pinneapple_quantum": (
        "Quantum and quantum-inspired PINNs.\n\n"
        "Parameterized quantum circuits, quantum loss functions (e.g. Schrodinger "
        "residuals), and hybrid classical/quantum training pipelines."
    ),
    "pinneapple_worldmodel": (
        "Learned world models for physics.\n\n"
        "Built-in PDE simulators used as ground truth, foundation-model "
        "architectures, and meta-learning over simulated dynamics."
    ),
    "pinneapple_models": (
        "Compatibility shim.\n\n"
        "Re-exports from `pinneapple_neural.architectures` so legacy code "
        "importing `pinneapple_models.*` keeps working. New code should import "
        "from `pinneapple_neural.architectures` directly."
    ),
    "pinneapple_solvers": (
        "Compatibility shim.\n\n"
        "Re-exports from `pinneapple_simulation.numerical_solvers` so legacy code "
        "importing `pinneapple_solvers.*` keeps working. New code should import "
        "from `pinneapple_simulation.numerical_solvers` directly."
    ),
    "pinneapple_train": (
        "Compatibility shim.\n\n"
        "Re-exports from `pinneapple_neural.trainer` so legacy code importing "
        "`pinneapple_train.*` keeps working. New code should import from "
        "`pinneapple_neural.trainer` directly."
    ),
}

IGNORE_DIR_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "tests",
    "test",
    "testing",
    "examples",
    "example",
    "scripts",
    "benchmarks",
    "docs",
    "site",
    "build",
    "dist",
}

def is_python_module_file(p: Path) -> bool:
    if p.suffix != ".py":
        return False
    if p.name == "__init__.py":
        return False
    if p.name.startswith("_"):
        return False
    return True

def is_ignored_path(p: Path) -> bool:
    return any(part in IGNORE_DIR_PARTS for part in p.parts)

def iter_target_packages():
    for name in TARGET_PACKAGES:
        d = SRC / name
        if d.is_dir() and (d / "__init__.py").exists():
            yield d

def iter_submodules(pkg_dir: Path):
    for p in sorted(pkg_dir.rglob("*.py")):
        if is_ignored_path(p):
            continue
        if not is_python_module_file(p):
            continue
        yield p

def to_import_path(pkg_dir: Path, py_file: Path) -> str:
    rel = py_file.relative_to(pkg_dir).with_suffix("")
    return ".".join((pkg_dir.name, *rel.parts))

def main() -> None:
    # api/index.md
    with mkdocs_gen_files.open(DOCS_API_DIR / "index.md", "w") as f:
        f.write("# API Reference\n\n")
        f.write("Generated from docstrings.\n\n")
        f.write("## Packages\n\n")
        for name in TARGET_PACKAGES:
            # link para pasta do pacote (index.md)
            f.write(f"- [{name}]({name}/)\n")

    # SUMMARY (sidebar humana)
    nav_lines: list[str] = [
        "# SUMMARY\n",
        "* [Home](index.md)\n",
        "* [Philosophy](philosophy.md)\n",
        "* [Why PINNeAPPle?](why.md)\n",
        "* [Architecture](architecture/system_overview.md)\n",
        "  * [Execution Model](architecture/execution_model.md)\n",
        "  * [Package Layers](architecture/package_layers.md)\n",
        "* [Getting Started](getting_started/installation.md)\n",
        "  * [Quickstart](getting_started/quickstart.md)\n",
        "  * [First PINN](getting_started/first_pinn.md)\n",
        "  * [Project Layout](getting_started/project_layout.md)\n",
        "* [Core Concepts](core_concepts/overview.md)\n",
        "  * [PhysicalSample](core_concepts/physical_sample.md)\n",
        "  * [ProblemDefinition](core_concepts/problem_definition.md)\n",
        "  * [Geometry & Domain](core_concepts/geometry_domain.md)\n",
        "  * [Model](core_concepts/model.md)\n",
        "  * [PINN / Physics](core_concepts/pinn.md)\n",
        "  * [Solver](core_concepts/solver.md)\n",
        "  * [Backend](core_concepts/backend.md)\n",
        "  * [Training Pipeline](core_concepts/training_pipeline.md)\n",
        "  * [Researcher & Benchmarking](core_concepts/researcher_benchmarking.md)\n",
        "* [API Reference](api/index.md)\n",
    ]
    for name in TARGET_PACKAGES:
        nav_lines.append(f"  * [{name}](api/{name}/index.md)\n")

    # Gera: api/<pkg>/index.md + api/<pkg>/<submodule>.md
    for pkg_dir in iter_target_packages():
        pkg_name = pkg_dir.name

        submods: list[tuple[str, str, str]] = []  # (label, link, import_path)

        for py_file in iter_submodules(pkg_dir):
            import_path = to_import_path(pkg_dir, py_file)
            rel = py_file.relative_to(pkg_dir).with_suffix("")  # ex: adapters/upd_adapter
            out = DOCS_API_DIR / pkg_name / rel.with_suffix(".md")

            # link RELATIVO ao api/<pkg>/index.md (sem "api/<pkg>" prefix!)
            link_from_pkg_index = rel.with_suffix(".md").as_posix()
            label = ".".join(rel.parts)

            submods.append((label, link_from_pkg_index, import_path))

            with mkdocs_gen_files.open(out, "w") as f:
                f.write(f"# {import_path}\n\n")
                f.write(f"::: {import_path}\n")

        pkg_index = DOCS_API_DIR / pkg_name / "index.md"
        with mkdocs_gen_files.open(pkg_index, "w") as f:
            f.write(f"# {pkg_name}\n\n")
            f.write("## Overview\n\n")
            f.write(PACKAGE_OVERVIEWS.get(pkg_name, ""))
            f.write("\n\n## Modules\n\n")
            if submods:
                for label, link, _ in submods:
                    f.write(f"- [{label}]({link})\n")
            else:
                f.write("_No public submodules found._\n")
            f.write("\n\n## Package API\n\n")
            f.write(f"::: {pkg_name}\n")

    with mkdocs_gen_files.open("SUMMARY.md", "w") as f:
        f.writelines(nav_lines)

if __name__ == "__main__":
    main()