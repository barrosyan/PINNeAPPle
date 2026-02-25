from __future__ import annotations

from pathlib import Path
import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT
DOCS_API_DIR = Path("api")

TARGET_PACKAGES = [
    "pinneaple_arena",
    "pinneaple_data",
    "pinneaple_geom",
    "pinneaple_models",
    "pinneaple_pdb",
    "pinneaple_pinn",
    "pinneaple_problemdesign",
    "pinneaple_researcher",
    "pinneaple_solvers",
    "pinneaple_timeseries",
    "pinneaple_train",
]

PACKAGE_OVERVIEWS = {
    "pinneaple_arena": (
        "Execution layer and backend abstraction.\n\n"
        "Provides backends for running training and evaluation workflows. "
        "Decouples solver logic from execution engines."
    ),
    "pinneaple_data": (
        "Data representation and PhysicalSample abstraction.\n\n"
        "Unified containers for fields, coordinates and metadata, plus dataset utilities."
    ),
    "pinneaple_geom": (
        "Geometry and domain encoding.\n\n"
        "Domain definitions, boundary handling and sampling utilities."
    ),
    "pinneaple_models": (
        "Neural architecture abstractions.\n\n"
        "Physics-agnostic models (e.g., MLPs). Constraints live in the PINN layer."
    ),
    "pinneaple_pdb": (
        "Structured physics database.\n\n"
        "Stores/retrieves problems, benchmarks and experiment specs."
    ),
    "pinneaple_pinn": (
        "PINN core logic.\n\n"
        "Residual construction, autograd utilities and constraint composition."
    ),
    "pinneaple_problemdesign": (
        "Problem definition layer.\n\n"
        "Encodes equations, BC/IC and metadata into reusable definitions."
    ),
    "pinneaple_researcher": (
        "Research orchestration and benchmarking.\n\n"
        "Experiment management, metrics and reproducible comparisons."
    ),
    "pinneaple_solvers": (
        "Optimization strategies and solver logic.\n\n"
        "Training policy independent from execution runtime."
    ),
    "pinneaple_timeseries": (
        "Scientific time-series utilities.\n\n"
        "Forecasting and hybrid dynamical workflows integrated with Physics AI pipelines."
    ),
    "pinneaple_train": (
        "Training orchestration.\n\n"
        "High-level APIs wiring problem, data, model, physics, solver and backend."
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