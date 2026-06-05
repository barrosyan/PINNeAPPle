# -*- coding: utf-8 -*-
"""PINNeAPPle — Run All Examples & Benchmarks
===============================================

Discovers and runs every runnable example/benchmark in the repository.
Results (stdout, stderr, timing, exit code) are saved to ./run_results/.

Usage
-----
  # Fast mode — scripts < 3 min each (default)
  python run_all_examples.py

  # Run a specific category
  python run_all_examples.py --category getting_started
  python run_all_examples.py --category arena_pipelines
  python run_all_examples.py --category benchmark_suite
  python run_all_examples.py --category use_cases
  python run_all_examples.py --category visualizations
  python run_all_examples.py --category templates

  # Run everything (may take hours)
  python run_all_examples.py --mode all --timeout 600

  # Dry run (list scripts without executing)
  python run_all_examples.py --dry-run

  # Resume (skip already-passed scripts)
  python run_all_examples.py --resume

Options
-------
  --mode       fast | medium | all    (default: fast)
  --category   filter to one category
  --timeout    seconds per script     (default: 180)
  --workers    parallel workers       (default: 1)
  --output     output directory       (default: ./run_results)
  --dry-run    list scripts only
  --resume     skip already-completed
  --device     cpu | cuda | auto      (default: cpu)
  --seed       RNG seed               (default: 42)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).parent
PYTHON = sys.executable


# ============================================================================
# SCRIPT CATALOG
# ============================================================================

@dataclass
class Script:
    """One runnable example or benchmark."""
    path:       str                  # relative to REPO
    category:   str
    name:       str
    args:       List[str] = field(default_factory=list)
    timeout:    int        = 180     # seconds
    mode:       str        = "fast"  # fast | medium | slow
    requires:   List[str]  = field(default_factory=list)  # pip packages
    skip_if:    List[str]  = field(default_factory=list)  # executables


def catalog() -> List[Script]:
    """Full catalog of runnable scripts, ordered fast → slow."""
    return [

        # ── SANITY / INSTALL CHECK ──────────────────────────────────────
        Script("scripts/dev/sanity_check_install.py", "sanity", "install_check",
               timeout=30, mode="fast"),

        # ── GETTING STARTED (1-10) ──────────────────────────────────────
        *[Script(f"examples/getting_started/0{i}_{n}.py", "getting_started", f"gs_{i}_{n}",
                 args=["--epochs", "2000", "--device", "cpu", "--seed", "42"],
                 timeout=120, mode="fast")
          for i, n in [
              (1,"harmonic_oscillator"), (2,"damped_oscillator"),
              (3,"heat_diffusion_1d"),   (4,"wave_equation_1d"),
              (5,"logistic_growth"),     (6,"lotka_volterra"),
              (7,"nonlinear_pendulum"),  (8,"van_der_pol"),
              (9,"lorenz_system"),
          ]
        ],
        Script("examples/getting_started/10_coupled_oscillators.py", "getting_started",
               "gs_10_coupled", args=["--epochs", "2000", "--device", "cpu"],
               timeout=120, mode="fast"),

        # ── VISUALIZATIONS ──────────────────────────────────────────────
        Script("examples/visualizations/viz_03_vortex_dynamics.py",  "visualizations", "viz_03", timeout=30,  mode="fast"),
        Script("examples/visualizations/viz_04_phase_field.py",       "visualizations", "viz_04", timeout=30,  mode="fast"),
        Script("examples/visualizations/viz_01_flow_cylinder.py",     "visualizations", "viz_01",
               args=["--epochs","1000"], timeout=120, mode="fast"),
        Script("examples/visualizations/viz_02_heat_2d.py",           "visualizations", "viz_02",
               args=["--epochs","1000"], timeout=120, mode="fast"),
        Script("examples/visualizations/viz_05_wave_2d.py",           "visualizations", "viz_05",
               args=["--epochs","1000"], timeout=120, mode="fast"),
        Script("examples/visualizations/viz_06_structural.py",        "visualizations", "viz_06",
               args=["--epochs","1000"], timeout=120, mode="fast"),

        # ── BENCHMARKS (data I/O) ────────────────────────────────────────
        Script("benchmarks/data_io_bench.py",    "benchmarks", "data_io",
               args=["--steps","50","--n","200"], timeout=60,  mode="fast",
               requires=["zarr"]),
        Script("benchmarks/shard_balance_bench.py","benchmarks","shard_balance",
               args=["--n","500","--steps","100"], timeout=60, mode="fast"),

        # ── DATA PIPELINE ──────────────────────────────────────────────
        Script("examples/data_pipeline/01_physical_sample_basics.py", "data_pipeline", "dp_01", timeout=30, mode="fast"),
        Script("examples/data_pipeline/02_zarr_store_write_read.py",  "data_pipeline", "dp_02", timeout=30, mode="fast", requires=["zarr"]),
        Script("examples/data_pipeline/03_collocation_sampler.py",    "data_pipeline", "dp_03", timeout=30, mode="fast"),
        Script("examples/data_pipeline/04_zarr_prefetch_dataloader.py","data_pipeline","dp_04", timeout=30, mode="fast", requires=["zarr"]),
        Script("examples/data_pipeline/05_upd_dataset_validation.py", "data_pipeline", "dp_05", timeout=30, mode="fast"),
        Script("examples/data_pipeline/06_active_learning_sampler.py","data_pipeline","dp_06",  timeout=60, mode="fast"),
        Script("examples/data_pipeline/07_zarr_shard_prefetch.py",    "data_pipeline", "dp_07", timeout=30, mode="fast", requires=["zarr"]),
        Script("examples/data_pipeline/08_collate_pinn_factory_style_batches.py","data_pipeline","dp_08",timeout=30,mode="fast"),

        # ── GEOMETRY ───────────────────────────────────────────────────
        Script("examples/geometry/01_load_mesh_and_sample.py",     "geometry", "geo_01", timeout=30, mode="fast"),
        Script("examples/geometry/02_sdf_primitives_2d.py",         "geometry", "geo_02", timeout=30, mode="fast"),
        Script("examples/geometry/03_channel_with_obstacle.py",     "geometry", "geo_03", timeout=30, mode="fast"),
        Script("examples/geometry/04_airfoil_naca_profile.py",      "geometry", "geo_04", timeout=30, mode="fast"),
        Script("examples/geometry/05_3d_domain_box_sphere.py",      "geometry", "geo_05", timeout=30, mode="fast"),
        Script("examples/geometry/06_csg_union_difference.py",      "geometry", "geo_06", timeout=30, mode="fast"),
        Script("examples/geometry/07_csg_domain_demo.py",           "geometry", "geo_07", timeout=30, mode="fast"),

        # ── PHYSICS DB ──────────────────────────────────────────────────
        Script("examples/physics_db/00_quickstart_inspect.py",     "physics_db","pdb_00",timeout=15,mode="fast"),
        Script("examples/physics_db/01_browse_problems.py",        "physics_db","pdb_01",timeout=15,mode="fast"),
        Script("examples/physics_db/02_filter_and_search.py",      "physics_db","pdb_02",timeout=15,mode="fast"),
        Script("examples/physics_db/03_render_problem_spec.py",    "physics_db","pdb_03",timeout=15,mode="fast"),
        Script("examples/physics_db/04_parameterize_and_sweep.py", "physics_db","pdb_04",timeout=30,mode="fast"),
        Script("examples/physics_db/05_validate_solution.py",      "physics_db","pdb_05",timeout=30,mode="fast"),
        Script("examples/physics_db/06_supervised_tiny_train_from_upd.py","physics_db","pdb_06",
               args=["--epochs","100"], timeout=60, mode="fast"),

        # ── ARCHITECTURES ───────────────────────────────────────────────
        Script("examples/architectures/00_registry_tour.py",                       "architectures","arch_00",timeout=30, mode="fast"),
        Script("examples/architectures/10_operator_learning_fno_toy.py",           "architectures","arch_10",args=["--epochs","100"],timeout=60,mode="fast"),
        Script("examples/architectures/20_pinn_inverse_parameter_ode.py",          "architectures","arch_20",args=["--epochs","200"],timeout=60,mode="fast"),
        Script("examples/architectures/30_rom_pod_dmd.py",                         "architectures","arch_30",timeout=60, mode="fast"),
        Script("examples/architectures/40_graph_gnn_message_passing.py",           "architectures","arch_40",args=["--epochs","100"],timeout=60,mode="fast"),
        Script("examples/architectures/50_timeseries_transformer_forecast_toy.py", "architectures","arch_50",args=["--epochs","50"],timeout=60,mode="fast"),

        # ── ELECTRODYNAMICS ─────────────────────────────────────────────
        *[Script(f"examples/electrodynamics/0{i}_{n}.py","electrodynamics",f"em_0{i}",
                 args=["--epochs","500"],timeout=90,mode="fast")
          for i,n in [(1,"laplace_capacitor"),(2,"poisson_charge"),(3,"wave_1d"),
                      (4,"transient_rl"),(5,"dipole_antenna"),(6,"tm_waveguide")]
        ],

        # ── PINN SOLVER ─────────────────────────────────────────────────
        *[Script(f"examples/pinn_solver/0{i}_{n}.py","pinn_solver",f"ps_0{i}",
                 args=["--epochs","300"],timeout=90,mode="fast")
          for i,n in [(2,"compiler_poisson_2d"),(3,"compiler_heat_1d"),
                      (4,"compiler_ns_lid_cavity"),(5,"domain_decomp_1d"),
                      (6,"compiler_wave_1d")]
        ],

        # ── NUMERICAL SOLVERS ───────────────────────────────────────────
        Script("examples/numerical_solvers/01_solvers_fft_feature_train.py","numerical","ns_01",args=["--epochs","200"],timeout=90,mode="fast"),
        Script("examples/numerical_solvers/02_lbm_flow_and_pinn.py",        "numerical","ns_02",timeout=90, mode="fast"),
        Script("examples/numerical_solvers/03_sph_particles.py",            "numerical","ns_03",timeout=60, mode="fast"),
        Script("examples/numerical_solvers/04_fenics_bridge.py",            "numerical","ns_04",timeout=60, mode="fast", skip_if=["fenics","dolfinx"]),
        Script("examples/numerical_solvers/05_fdm_heat_3d.py",              "numerical","ns_05",timeout=90, mode="fast"),
        Script("examples/numerical_solvers/06_spectral_solver_burgers.py",  "numerical","ns_06",timeout=60, mode="fast"),
        Script("examples/numerical_solvers/07_fdm_wave_2d.py",              "numerical","ns_07",timeout=60, mode="fast"),
        Script("examples/numerical_solvers/08_lbm_channel_parallel.py",     "numerical","ns_08",timeout=60, mode="fast"),
        Script("examples/numerical_solvers/09_isph_dam_break.py",           "numerical","ns_09",timeout=90, mode="fast"),
        Script("examples/numerical_solvers/10_mpm_snow_compression.py",     "numerical","ns_10",timeout=90, mode="fast"),

        # ── TIME SERIES ─────────────────────────────────────────────────
        Script("examples/time_series/00_quickstart.py",         "time_series","ts_00",args=["--epochs","50"],timeout=60,mode="fast"),
        Script("examples/time_series/01_advanced_models.py",    "time_series","ts_01",args=["--epochs","50"],timeout=90,mode="fast"),
        Script("examples/time_series/02_multivariate.py",       "time_series","ts_02",args=["--epochs","50"],timeout=90,mode="fast"),
        Script("examples/time_series/03_physics_augmented.py",  "time_series","ts_03",args=["--epochs","50"],timeout=90,mode="fast"),
        Script("examples/time_series/04_transfer_learning.py",  "time_series","ts_04",args=["--epochs","50"],timeout=90,mode="fast"),
        Script("examples/time_series/05_uncertainty.py",        "time_series","ts_05",args=["--epochs","50"],timeout=90,mode="fast"),
        Script("examples/time_series/06_custom_model_registry.py","time_series","ts_06",timeout=30,mode="fast"),

        # ── ARENA PIPELINES ─────────────────────────────────────────────
        Script("examples/arena_pipelines/00_list_datasets.py",          "arena","arena_00",timeout=15,mode="fast"),
        Script("examples/arena_pipelines/01_physics_benchmark.py",      "arena","arena_01",args=["--fast"],timeout=180,mode="fast"),
        Script("examples/arena_pipelines/02_timeseries_benchmark.py",   "arena","arena_02",args=["--fast"],timeout=180,mode="fast"),
        Script("examples/arena_pipelines/03_kovasznay_ns_benchmark.py", "arena","arena_03",args=["--fast"],timeout=180,mode="fast"),
        Script("examples/arena_pipelines/05_all_problems_gallery.py",   "arena","arena_05",timeout=30, mode="fast"),
        Script("examples/arena_pipelines/06_custom_problem.py",         "arena","arena_06",args=["--epochs","200"],timeout=90,mode="fast"),
        Script("examples/arena_pipelines/07_easy_custom_problem.py",    "arena","arena_07",args=["--epochs","200"],timeout=90,mode="fast"),
        Script("examples/arena_pipelines/08_3d_geometry_pipeline.py",   "arena","arena_08",args=["--epochs","200"],timeout=120,mode="fast"),

        # ── BENCHMARK SUITE (fast mode) ─────────────────────────────────
        Script("examples/benchmark_suite/01_quickstart_native.py",         "benchmark_suite","bs_01",timeout=60, mode="medium"),
        Script("examples/benchmark_suite/02_arena_from_yaml.py",           "benchmark_suite","bs_02",timeout=60, mode="medium"),
        Script("examples/benchmark_suite/03_pinn_burgers_full_pipeline.py","benchmark_suite","bs_03",args=["--epochs","100"],timeout=120,mode="medium"),
        Script("examples/benchmark_suite/04_custom_task_poisson_bundle.py","benchmark_suite","bs_04",args=["--epochs","100"],timeout=120,mode="medium"),
        Script("examples/benchmark_suite/05_surrogate_deeponet_multifield.py","benchmark_suite","bs_05",args=["--epochs","100"],timeout=120,mode="medium"),
        Script("examples/benchmark_suite/10_datacenter_digital_twin.py",   "benchmark_suite","bs_10",args=["--epochs","100"],timeout=120,mode="medium"),
        Script("examples/benchmark_suite/12_physics_benchmark_suite.py",   "benchmark_suite","bs_12",args=["--fast"],timeout=300,mode="medium"),
        Script("examples/benchmark_suite/13_transfer_meta_benchmark.py",   "benchmark_suite","bs_13",args=["--fast"],timeout=300,mode="medium"),
        Script("examples/benchmark_suite/15_naca0012_aerodynamic_surrogate.py","benchmark_suite","bs_15",args=["--epochs","100"],timeout=180,mode="medium"),
        Script("examples/benchmark_suite/16_common_pinn_examples.py",      "benchmark_suite","bs_16",args=["--epochs","100"],timeout=120,mode="medium"),

        # ── TEMPLATES (reduced epochs) ───────────────────────────────────
        *[Script(f"templates/{n}.py","templates",f"tmpl_{n[:5]}",
                 args=["--epochs","50","--device","cpu"],timeout=90,mode="medium")
          for n in [
              "01_basic_pinn","02_symbolic_pde","03_navier_stokes_channel",
              "04_domain_decomp_xpinn","05_causal_pinn","06_time_marching",
              "08_csg_geometry","09_visualization_export","11_shape_optimization",
              "14_meta_learning_maml","16_inverse_problem","17_fno_operator",
              "18_deeponet_surrogate","19_active_learning","21_digital_twin",
              "24_koopman_operator","25_zarr_pipeline","28_physics_validation",
          ]
        ],

        # ── USE CASES (slow) ─────────────────────────────────────────────
        Script("examples/use_cases/terramechanics/terramechanics_rover_pinn.py",
               "use_cases","uc_terramechanics",args=["--epochs","1000"],timeout=300,mode="slow"),
        Script("examples/use_cases/terramechanics/benchmark_vs_omnilrs.py",
               "use_cases","uc_terramech_bench",timeout=300,mode="slow"),
        Script("examples/use_cases/crash_surrogate/crash_surrogate_pipeline.py",
               "use_cases","uc_crash",timeout=600,mode="slow"),
        Script("examples/use_cases/missile_aero/missile_aero_pipeline.py",
               "use_cases","uc_missile",timeout=600,mode="slow"),
        Script("examples/use_cases/solid_mechanics/solid_mechanics_pipeline.py",
               "use_cases","uc_solid",timeout=600,mode="slow"),
        Script("examples/use_cases/physics_data_factory/full_pipeline_example.py",
               "use_cases","uc_factory_pipeline",timeout=600,mode="slow"),
        Script("examples/use_cases/physics_data_factory/industrial_digital_twin.py",
               "use_cases","uc_digital_twin",timeout=300,mode="slow"),

        # ── PHYSICS DATA FACTORY — 3D / PyVista ──────────────────────────
        # Requires: pyvista (pip install pyvista vtk)
        Script("examples/use_cases/physics_data_factory/factory_3d_render.py",
               "factory_3d","fac_3d_pyvista",
               args=["--n-frames","48","--fps","24"],
               timeout=300, mode="slow",
               requires=["pyvista"]),
        Script("examples/use_cases/physics_data_factory/ultrarealistic_digital_twin.py",
               "factory_3d","fac_3d_ultra",
               args=["--frames","48","--samples","256"],
               timeout=600, mode="slow",
               requires=["pyvista"]),

        # ── PHYSICS DATA FACTORY — Blender Cycles ────────────────────────
        # Requires: blender executable (portable install at ~/blender-4.2)
        # Use blender_factory_launcher.py as the entry point — it handles
        # finding blender.exe, running the render script, and assembling MP4.
        Script("examples/use_cases/physics_data_factory/blender_factory_launcher.py",
               "factory_blender","fac_blender_preview",
               args=["--frames","12","--samples","64","--width","1280","--height","720"],
               timeout=600, mode="slow",
               skip_if=["blender", "blender.exe"]),   # needs blender in PATH or ~/blender-4.2
        Script("examples/use_cases/physics_data_factory/blender_factory_launcher.py",
               "factory_blender","fac_blender_hd",
               args=["--frames","192","--samples","256","--width","1920","--height","1080"],
               timeout=7200, mode="slow",
               skip_if=["blender", "blender.exe"]),

        # ── CLIENT EXAMPLES ──────────────────────────────────────────────
        Script("examples/clients/lane_emden_example.py","clients","cli_lane",    args=["--epochs","500"],timeout=90,mode="medium"),
        Script("examples/clients/alstom_example.py",    "clients","cli_alstom",  args=["--epochs","200"],timeout=90,mode="medium"),
        Script("examples/clients/splash_cfd_example.py","clients","cli_splash",  args=["--epochs","200"],timeout=90,mode="medium"),

        # ── END-TO-END ───────────────────────────────────────────────────
        Script("examples/end_to_end/01_upd_zarr_train_with_solvers.py",  "end_to_end","e2e_01",timeout=120,mode="medium",requires=["zarr"]),
        Script("examples/end_to_end/02_openfoam_to_pinn.py",             "end_to_end","e2e_02",timeout=120,mode="medium",skip_if=["openfoam","foamDictionary"]),
        Script("examples/end_to_end/03_digital_twin_kafka_stream.py",    "end_to_end","e2e_03",timeout=120,mode="medium"),
        Script("examples/end_to_end/04_multiphysics_cosim.py",           "end_to_end","e2e_04",timeout=120,mode="medium"),
        Script("examples/end_to_end/05_stl_cad_to_pinn.py",              "end_to_end","e2e_05",timeout=120,mode="medium"),
        Script("examples/end_to_end/06_stl_pipeline_heat.py",            "end_to_end","e2e_06",timeout=120,mode="medium"),
    ]


# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

def _is_available(name: str) -> bool:
    """Check if a package/executable is importable, in PATH, or at known locations."""
    import importlib, shutil
    # Python import check
    try:
        importlib.import_module(name.replace("-", "_"))
        return True
    except ImportError:
        pass
    # PATH check
    if shutil.which(name):
        return True
    # Extra: check common portable / out-of-PATH install locations
    _PORTABLE_LOCS = {
        "blender":     [Path.home() / "blender-4.2" / "blender.exe",
                        Path.home() / "blender-4.1" / "blender.exe",
                        Path.home() / "blender-4.3" / "blender.exe",
                        Path("C:/Program Files/Blender Foundation/Blender 4.2/blender.exe")],
        "blender.exe": [Path.home() / "blender-4.2" / "blender.exe",
                        Path.home() / "blender-4.1" / "blender.exe"],
        "openfoam":    [Path("/usr/bin/foamDictionary"), Path("/opt/openfoam10/bin/foamDictionary")],
        "fenics":      [Path("/usr/bin/fenics")],
        "dolfinx":     [Path("/usr/bin/dolfinx-version")],
    }
    for loc in _PORTABLE_LOCS.get(name, []):
        if loc.exists():
            return True
    return False


def can_run(script: Script) -> tuple[bool, str]:
    """Return (ok, reason_if_not_ok).

    script.requires  — pip packages or executables that MUST be available
    script.skip_if   — executables that must NOT be available (i.e. the script
                       requires an external tool that is unavailable, so we skip
                       when the tool is absent)
    """
    for pkg in script.requires:
        if not _is_available(pkg):
            return False, f"missing package: {pkg}"
    for exe in script.skip_if:
        # skip_if lists executables that the script NEEDS but that are typically
        # not installed (OpenFOAM, fenics, …). Skip when the tool is absent.
        if not _is_available(exe):
            return False, f"missing external tool: {exe}"
    if not (REPO / script.path).exists():
        return False, f"file not found: {script.path}"
    return True, ""


# ============================================================================
# RESULT
# ============================================================================

@dataclass
class RunResult:
    name:       str
    path:       str
    category:   str
    status:     str    # passed | failed | skipped | timeout | error
    elapsed_s:  float  = 0.0
    exit_code:  int    = 0
    skip_reason:str    = ""
    stdout_file:str    = ""
    stderr_file:str    = ""
    error_msg:  str    = ""


# ============================================================================
# RUNNER
# ============================================================================

def run_script(script: Script, out_dir: Path, device: str, seed: int) -> RunResult:
    """Execute one script and capture output."""
    result = RunResult(name=script.name, path=script.path,
                       category=script.category, status="error")

    ok, reason = can_run(script)
    if not ok:
        result.status = "skipped"
        result.skip_reason = reason
        return result

    script_path = REPO / script.path
    run_dir     = out_dir / script.category / script.name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build command, inject --device and --seed if the script likely accepts them
    cmd = [PYTHON, str(script_path)] + script.args

    stdout_f = run_dir / "stdout.txt"
    stderr_f = run_dir / "stderr.txt"
    result.stdout_file = str(stdout_f.relative_to(out_dir))
    result.stderr_file = str(stderr_f.relative_to(out_dir))

    # Force UTF-8 I/O in child process — fixes UnicodeEncodeError with Greek
    # letters, arrows etc. in plot labels on Windows cp1252 consoles.
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"]       = "1"

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=script.timeout,
            cwd=str(REPO),
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        elapsed = time.time() - t0
        stdout_f.write_text(proc.stdout or "(no output)", encoding="utf-8")
        stderr_f.write_text(proc.stderr or "(no output)", encoding="utf-8")
        result.elapsed_s = elapsed
        result.exit_code  = proc.returncode
        if proc.returncode == 0:
            result.status = "passed"
        else:
            combined = (proc.stderr or "") + (proc.stdout or "")
            lines     = combined.strip().splitlines()
            last_line = lines[-1] if lines else f"exit code {proc.returncode}"
            result.error_msg = last_line

            # Classify missing-module failures as skipped, not failed
            _SKIP_PATTERNS = (
                "ModuleNotFoundError",
                "No module named",
                "ImportError",
                "cannot import name",
            )
            if any(p in combined for p in _SKIP_PATTERNS):
                # Extract the missing module name
                m = re.search(r"No module named '([^']+)'", combined)
                mod = m.group(1) if m else "unknown"
                result.status     = "skipped"
                result.skip_reason = f"missing module: {mod}"
            else:
                result.status = "failed"

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        result.status    = "timeout"
        result.elapsed_s = elapsed
        result.error_msg = f"timed out after {script.timeout}s"
        stdout_f.write_text("(timed out)", encoding="utf-8")
        stderr_f.write_text("(timed out)", encoding="utf-8")

    except Exception as e:
        result.status    = "error"
        result.error_msg = str(e)
        result.elapsed_s = time.time() - t0

    return result


# ============================================================================
# REPORT
# ============================================================================

ICONS = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP",
         "timeout": "TIME", "error": "ERR "}

def _safe_print(msg: str) -> None:
    """Print safely on Windows cp1252 consoles — replace unencodable chars."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode(sys.stdout.encoding or "ascii", errors="replace").decode(sys.stdout.encoding or "ascii"))


def print_result(r: RunResult, verbose: bool = False):
    icon = ICONS.get(r.status, "???")
    t    = f"{r.elapsed_s:5.1f}s" if r.elapsed_s else "      "
    msg  = f"  [{icon}] {t}  {r.category}/{r.name}"
    if r.status in ("failed","timeout","error"):
        msg += f"  -- {r.error_msg[:80]}"
    elif r.status == "skipped":
        msg += f"  ({r.skip_reason})"
    _safe_print(msg)


def save_report(results: List[RunResult], out_dir: Path, args: argparse.Namespace):
    totals = {s: sum(1 for r in results if r.status == s)
              for s in ("passed","failed","skipped","timeout","error")}
    total_elapsed = sum(r.elapsed_s for r in results)

    # JSON
    report = {
        "mode":       args.mode,
        "category":   args.category,
        "device":     args.device,
        "timestamp":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "totals":     totals,
        "n_total":    len(results),
        "elapsed_s":  total_elapsed,
        "results":    [asdict(r) for r in results],
    }
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Markdown
    md_lines = [
        "# PINNeAPPle — Run All Examples Report",
        "",
        f"**Mode:** {args.mode}  |  **Category:** {args.category or 'all'}  "
        f"|  **Date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        "## Summary",
        "",
        f"| Status  | Count |",
        f"|---------|-------|",
        *[f"| {k:<7} | {v:>5} |" for k,v in totals.items()],
        f"| **Total** | **{len(results)}** |",
        "",
        f"Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)",
        "",
        "## Results by Category",
        "",
    ]
    by_cat: Dict[str, List[RunResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    for cat, cat_results in sorted(by_cat.items()):
        n_pass = sum(1 for r in cat_results if r.status == "passed")
        md_lines += [f"### {cat}  ({n_pass}/{len(cat_results)} passed)", ""]
        md_lines += ["| Script | Status | Time |", "|--------|--------|------|"]
        for r in cat_results:
            icon = {"passed":"✓","failed":"✗","skipped":"−","timeout":"⏱","error":"!"}.get(r.status,"?")
            t = f"{r.elapsed_s:.1f}s" if r.elapsed_s else "—"
            note = r.error_msg[:60] if r.status not in ("passed","skipped") else r.skip_reason[:40]
            md_lines.append(f"| `{r.name}` | {icon} {r.status} | {t} | {note} |")
        md_lines.append("")

    md_path = out_dir / "report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return json_path, md_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Run all PINNeAPPle examples & benchmarks")
    ap.add_argument("--mode",     choices=["fast","medium","all"], default="fast",
                    help="fast=<3min each, medium=<10min each, all=everything")
    ap.add_argument("--category", default="",
                    help="Run only scripts in this category (e.g. getting_started)")
    ap.add_argument("--timeout",  type=int, default=None,
                    help="Override per-script timeout (seconds)")
    ap.add_argument("--output",   default=str(REPO / "run_results"),
                    help="Output directory for results")
    ap.add_argument("--dry-run",  action="store_true",
                    help="List scripts without running")
    ap.add_argument("--resume",   action="store_true",
                    help="Skip scripts whose output already exists (passed)")
    ap.add_argument("--device",   default="cpu")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--verbose",  action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter catalog  ("all" includes every mode)
    mode_order = {"fast": 0, "medium": 1, "slow": 2, "all": 99}
    mode_limit = mode_order.get(args.mode, 0)
    all_scripts = [
        s for s in catalog()
        if mode_order.get(s.mode, 99) <= mode_limit
        and (not args.category or s.category == args.category)
    ]
    if args.timeout:
        for s in all_scripts:
            s.timeout = args.timeout

    print(f"\n{'='*65}")
    print(f"  PINNeAPPle — Run All Examples & Benchmarks")
    print(f"  Mode: {args.mode}  |  Category: {args.category or 'all'}")
    print(f"  Scripts: {len(all_scripts)}  |  Output: {out_dir}")
    print(f"{'='*65}\n")

    if args.dry_run:
        for s in all_scripts:
            ok, reason = can_run(s)
            marker = "[ ]" if not ok else "[x]"
            print(f"  {marker}  {s.category}/{s.name}  ({s.mode}, {s.timeout}s)  {reason}")
        print(f"\n  Total: {len(all_scripts)} scripts")
        return

    results: List[RunResult] = []
    t_start = time.time()

    for i, script in enumerate(all_scripts, 1):
        # Resume: skip if already passed
        if args.resume:
            run_dir = out_dir / script.category / script.name
            pass_marker = run_dir / "stdout.txt"
            if pass_marker.exists():
                r = RunResult(name=script.name, path=script.path,
                              category=script.category, status="skipped",
                              skip_reason="already completed (--resume)")
                results.append(r)
                print(f"  [SKIP] {script.category}/{script.name}  (already done)")
                continue

        pct = i / len(all_scripts) * 100
        elapsed_total = time.time() - t_start
        eta = elapsed_total / i * (len(all_scripts) - i) if i > 1 else 0
        print(f"  [{i:>3}/{len(all_scripts)}  {pct:4.0f}%  ETA {eta:.0f}s]  "
              f"{script.category}/{script.name}  (max {script.timeout}s)...",
              end="", flush=True)

        result = run_script(script, out_dir, args.device, args.seed)
        results.append(result)

        icon = ICONS.get(result.status, "???")
        t    = f"{result.elapsed_s:.1f}s"
        suffix = ""
        if result.status in ("failed","timeout","error"):
            suffix = f"  -- {result.error_msg[:70]}"
        elif result.status == "skipped":
            suffix = f"  ({result.skip_reason})"
        _safe_print(f"  {icon} {t}{suffix}")

    # Save report
    total_elapsed = time.time() - t_start
    json_path, md_path = save_report(results, out_dir, args)

    # Summary
    totals = {s: sum(1 for r in results if r.status == s)
              for s in ("passed","failed","skipped","timeout","error")}
    print(f"\n{'='*65}")
    print(f"  DONE in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  PASS={totals['passed']}  FAIL={totals['failed']}  "
          f"SKIP={totals['skipped']}  TIMEOUT={totals['timeout']}  ERR={totals['error']}")
    print(f"  Report: {md_path}")
    print(f"  JSON:   {json_path}")
    print(f"{'='*65}\n")

    # Print failures
    failures = [r for r in results if r.status in ("failed","error","timeout")]
    if failures:
        _safe_print("  Failed scripts:")
        for r in failures:
            _safe_print(f"    [{r.status.upper()}] {r.category}/{r.name}: {r.error_msg[:100]}")
        _safe_print("")

    sys.exit(0 if totals["failed"] == 0 and totals["error"] == 0 else 1)


if __name__ == "__main__":
    main()
