"""Tests for the automatic physical-code-generation tooling:

  * pinneapple_physics.codegen.fdm_script_generator.FDMScriptGenerator
  * pinneapple_physics.codegen.fenics_script_generator.FEniCSScriptGenerator
  * pinneapple_tools.sandbox.custom_solver_runner.run_custom_solver

FDM scripts are checked for real numerical accuracy against closed-form
solutions (not just "exit code 0"). FEniCS scripts are checked for valid
Python syntax and the expected UFL/dolfinx API surface, since dolfinx is
not installed in this test environment (see the module-level skip check
below) -- if it ever is, the diffusion case is additionally run end to end.
The sandbox runner is checked for its four required behaviors: a normal
script succeeds, an exception is caught and surfaced (not swallowed), an
infinite loop is killed within roughly its timeout, and an oversized
allocation is caught or raises.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import time

import numpy as np
import pytest

import pinneapple_physics as pp
from pinneapple_physics.codegen.fdm_script_generator import FDMScriptGenerator, PDEParser
from pinneapple_physics.codegen.fenics_script_generator import (
    FEniCSScriptGenerator,
    classify_physics_shape,
)
from pinneapple_tools.sandbox.custom_solver_runner import run_custom_solver

try:
    import dolfinx  # noqa: F401
    _DOLFINX_AVAILABLE = True
except ImportError:
    _DOLFINX_AVAILABLE = False


def _run_script(path, out_npz, extra_args=()):
    result = subprocess.run(
        [sys.executable, str(path), "--out", str(out_npz), *extra_args],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"generated script failed (rc={result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    return result


# ---------------------------------------------------------------------------
# FDMScriptGenerator: PDEParser scope
# ---------------------------------------------------------------------------

def test_pdeparser_rejects_uncovered_kind():
    spec = pp.ns_incompressible_2d_default()  # not one of the 3 curated FDM forms
    with pytest.raises(NotImplementedError, match="curated"):
        PDEParser().parse(spec)


# ---------------------------------------------------------------------------
# FDMScriptGenerator: 1D heat equation
# ---------------------------------------------------------------------------

def test_fdm_heat_1d_matches_closed_form(tmp_path):
    alpha = 0.05
    spec = pp.ProblemSpec(
        name="heat_1d_test",
        dim=1,
        coords=("x", "t"),
        fields=("T",),
        pde=pp.PDETermSpec(kind="heat_equation", fields=("T",), coords=("x", "t"), params={"alpha": alpha}),
        domain_bounds={"x": (0.0, 1.0), "t": (0.0, 0.1)},
    )
    script = tmp_path / "heat_1d.py"
    FDMScriptGenerator().write(spec, str(script), nx=200, nt=5000)

    source = script.read_text()
    ast.parse(source)  # syntactically valid standalone Python
    assert "import pinneapple" not in source

    out_npz = tmp_path / "heat_1d_out.npz"
    _run_script(script, out_npz)

    data = np.load(out_npz)
    x, T, t_final = data["x"], data["T"], float(data["t"][0])
    exact = np.exp(-alpha * np.pi ** 2 * t_final) * np.sin(np.pi * x)
    max_err = np.max(np.abs(T - exact))
    assert max_err < 1e-4, f"heat_1d FTCS vs. exact exp(-alpha*pi^2*t)*sin(pi*x): max_err={max_err}"


# ---------------------------------------------------------------------------
# FDMScriptGenerator: 1D viscous Burgers
# ---------------------------------------------------------------------------

def test_fdm_burgers_1d_matches_traveling_front(tmp_path):
    nu = 0.05
    c = 1.0
    spec = pp.burgers_1d_default(nu=nu)
    script = tmp_path / "burgers_1d.py"
    FDMScriptGenerator().write(
        spec, str(script),
        ic_type="tanh_front", front_speed=c, x_bounds=(-1.0, 1.0), t_bounds=(0.0, 0.4),
        nx=400, nt=20000,
    )

    source = script.read_text()
    ast.parse(source)
    assert "import pinneapple" not in source

    out_npz = tmp_path / "burgers_1d_out.npz"
    _run_script(script, out_npz)

    data = np.load(out_npz)
    x, u, t_final = data["x"], data["u"], float(data["t"][0])
    # Exact traveling viscous-shock solution of u_t + u*u_x = nu*u_xx
    # (verified symbolically with SymPy while writing the generator).
    exact = c - c * np.tanh(c * x / (2.0 * nu) - c * c * t_final / (2.0 * nu))
    max_err = np.max(np.abs(u - exact))
    assert max_err < 0.1, f"burgers_1d FTCS+upwind vs. exact tanh front: max_err={max_err}"


# ---------------------------------------------------------------------------
# FDMScriptGenerator: 2D Poisson (manufactured solution)
# ---------------------------------------------------------------------------

def test_fdm_poisson_2d_matches_manufactured_solution(tmp_path):
    spec = pp.poisson_2d_default()
    script = tmp_path / "poisson_2d.py"
    FDMScriptGenerator().write(spec, str(script), nx=64, ny=64)

    source = script.read_text()
    ast.parse(source)
    assert "import pinneapple" not in source

    out_npz = tmp_path / "poisson_2d_out.npz"
    _run_script(script, out_npz)

    data = np.load(out_npz)
    x, y, u = data["x"], data["y"], data["u"]
    XX, YY = np.meshgrid(x, y, indexing="ij")
    exact = np.sin(np.pi * XX) * np.sin(np.pi * YY)
    max_err = np.max(np.abs(u - exact))
    assert max_err < 1e-3, f"poisson_2d red-black SOR vs. exact sin(pi x)sin(pi y): max_err={max_err}"


# ---------------------------------------------------------------------------
# FEniCSScriptGenerator: physics-shape classifier
# ---------------------------------------------------------------------------

def test_classify_physics_shape_on_real_presets():
    assert classify_physics_shape(pp.poisson_2d_default()) == "diffusion"
    assert classify_physics_shape(pp.ns_incompressible_2d_default()) == "stokes"
    assert classify_physics_shape(pp.linear_elasticity_3d_default()) == "elasticity"


# ---------------------------------------------------------------------------
# FEniCSScriptGenerator: diffusion script
# ---------------------------------------------------------------------------

def test_fenics_diffusion_script_is_valid_and_has_expected_api(tmp_path):
    spec = pp.poisson_2d_default()
    script = tmp_path / "diffusion.py"
    FEniCSScriptGenerator().write(spec, str(script), nx=8, ny=8)
    source = script.read_text()

    ast.parse(source)
    assert "import pinneapple_physics" not in source

    for token in (
        "def solve(params: dict)",
        "ufl.TrialFunction",
        "ufl.TestFunction",
        "dolfinx.fem.petsc",
        "LinearProblem",
        "create_rectangle",
        "dirichletbc",
        "def build_fenics_config",
        "FEniCSConfig",
    ):
        assert token in source, f"expected {token!r} in generated diffusion script"

    if _DOLFINX_AVAILABLE:
        namespace = {}
        exec(compile(source, str(script), "exec"), namespace)
        result = namespace["solve"]({"nx": 8, "ny": 8})
        assert "coords" in result and "u" in result


# ---------------------------------------------------------------------------
# FEniCSScriptGenerator: Stokes script
# ---------------------------------------------------------------------------

def test_fenics_stokes_script_is_valid_and_has_expected_api(tmp_path):
    spec = pp.ns_incompressible_2d_default()
    script = tmp_path / "stokes.py"
    FEniCSScriptGenerator().write(spec, str(script), nx=6, ny=6)
    source = script.read_text()

    ast.parse(source)
    assert "import pinneapple_physics" not in source

    for token in (
        "def solve(params: dict)",
        "ufl.TrialFunctions",
        "ufl.TestFunctions",
        "mixed_element",
        "ufl.div",
        "LinearProblem",
    ):
        assert token in source, f"expected {token!r} in generated Stokes script"

    # Stokes has no counterpart in FEnicsBridge._SUPPORTED_KINDS -- no
    # build_fenics_config() should be emitted (see module docstring).
    assert "def build_fenics_config" not in source


# ---------------------------------------------------------------------------
# FEniCSScriptGenerator: linear elasticity + penalty contact script
# ---------------------------------------------------------------------------

def test_fenics_elasticity_contact_script_is_valid_and_has_expected_api(tmp_path):
    spec = pp.linear_elasticity_3d_default()
    script = tmp_path / "elasticity_contact.py"
    FEniCSScriptGenerator().write(spec, str(script), contact=True, nx=6, ny=4)
    source = script.read_text()

    ast.parse(source)
    assert "import pinneapple_physics" not in source

    for token in (
        "def solve(params: dict)",
        "ufl.conditional",
        "NonlinearProblem",
        "NewtonSolver",
        "penalty",
        "sigma(u)",
    ):
        assert token in source, f"expected {token!r} in generated elasticity-contact script"

    # Contact-enabled elasticity also has no FEnicsBridge counterpart today.
    assert "def build_fenics_config" not in source


def test_fenics_elasticity_without_contact_has_config_helper(tmp_path):
    spec = pp.linear_elasticity_3d_default()
    script = tmp_path / "elasticity.py"
    FEniCSScriptGenerator().write(spec, str(script), contact=False, nx=6, ny=4)
    source = script.read_text()

    ast.parse(source)
    assert "def build_fenics_config" in source
    assert "linear_elasticity_plane_stress" in source


# ---------------------------------------------------------------------------
# pinneapple_tools.sandbox.custom_solver_runner.run_custom_solver
# ---------------------------------------------------------------------------

_NORMAL_SCRIPT = """
import numpy as np

def solve(params: dict) -> dict:
    x = float(params.get("x", 0.0))
    return {"coords": np.array([0.0, 1.0, 2.0]), "doubled": x * 2.0}
"""

_RAISING_SCRIPT = """
def solve(params: dict) -> dict:
    raise ValueError("boom: deliberately broken solver")
"""

_INFINITE_LOOP_SCRIPT = """
def solve(params: dict) -> dict:
    while True:
        pass
"""

_HUGE_ALLOCATION_SCRIPT = """
import numpy as np

def solve(params: dict) -> dict:
    # Try to allocate ~800 GB -- far beyond any reasonable memory_limit_mb.
    huge = np.ones((10**11,), dtype=np.float64)
    return {"coords": huge}
"""


def test_sandbox_normal_script_returns_expected_dict(tmp_path):
    script = tmp_path / "normal.py"
    script.write_text(_NORMAL_SCRIPT)

    result = run_custom_solver(str(script), {"x": 21}, timeout=10, memory_limit_mb=1024)

    assert result["success"] is True
    assert np.allclose(result["coords"], [0.0, 1.0, 2.0])
    assert result["doubled"] == pytest.approx(42.0)


def test_sandbox_exception_is_caught_and_reported(tmp_path):
    script = tmp_path / "raising.py"
    script.write_text(_RAISING_SCRIPT)

    result = run_custom_solver(str(script), {}, timeout=10, memory_limit_mb=1024)

    assert result["success"] is False
    assert result["error"] == "ValueError"
    assert "boom" in result["message"]


def test_sandbox_infinite_loop_is_killed_within_timeout(tmp_path):
    script = tmp_path / "loop.py"
    script.write_text(_INFINITE_LOOP_SCRIPT)

    start = time.monotonic()
    result = run_custom_solver(str(script), {}, timeout=2, memory_limit_mb=512)
    elapsed = time.monotonic() - start

    assert result["success"] is False
    assert result["error"] == "TimeoutExpired"
    # Generous slack for process spawn/teardown overhead -- this must not
    # turn into a slow test, so keep the ceiling tight relative to timeout=2.
    assert elapsed < 15, f"expected the loop to be killed near timeout=2s, took {elapsed}s"


def test_sandbox_oversized_allocation_fails_gracefully(tmp_path):
    script = tmp_path / "huge_alloc.py"
    script.write_text(_HUGE_ALLOCATION_SCRIPT)

    result = run_custom_solver(str(script), {}, timeout=15, memory_limit_mb=128)

    # Depending on platform/allocator this surfaces as a Python MemoryError
    # (or even an ImportError, if the RLIMIT_AS is tight enough to break
    # numpy's own import) caught by the child driver and reported, or the
    # OS killing the process outright (ProcessTerminated) -- exactly which
    # one is platform/allocator-dependent, but it must not succeed, and it
    # must not be silently swallowed (an "error" key is always present).
    assert result["success"] is False
    assert result.get("error")
