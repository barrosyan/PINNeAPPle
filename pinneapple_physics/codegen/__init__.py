"""pinneapple_physics.codegen — automatic physical-code generation.

Different in kind from ``pinneapple_physics.symbolic_pde``: that module
compiles a SymPy PDE residual into an in-memory autograd *closure* used
during PINN training in the same process. Everything in this package
instead renders a **complete, standalone, saveable Python source file**
(text you write to disk with ``open(path, "w").write(...)``) that runs on
its own, with no import of this package, via a bare ``python script.py``.

Sub-modules
-----------
fdm_script_generator
    ``PDEParser`` + ``FDMScriptGenerator``: curated finite-difference
    scripts (1D heat, 1D Burgers, 2D Poisson/Laplace) written in NumPy
    only.

fenics_script_generator
    ``FEniCSScriptGenerator`` (+ ``classify_physics_shape``): curated
    UFL/dolfinx finite-element scripts (linear diffusion, Stokes flow,
    linear elasticity with an optional penalty-method contact term).

Both generators emit scripts that expose a ``solve(params: dict) -> dict``
function returning at least a ``"coords"`` key — the same convention used
by ``pinneapple_tools.sandbox.custom_solver_runner.run_custom_solver``,
which can execute any of these generated scripts (or a hand-written one
following the same convention) in a separate, resource-limited process.

Scope
-----
Both generators are curated over a small, explicitly documented set of
PDE/physics forms — not general symbolic-PDE-to-code compilers. See each
module's docstring for exactly what is (and isn't) covered.

Usage
-----
>>> import pinneapple_physics as pp
>>> from pinneapple_physics.codegen import FDMScriptGenerator
>>> spec = pp.burgers_1d_default(nu=0.01)
>>> FDMScriptGenerator().write(spec, "burgers_solver.py")
'burgers_solver.py'
"""
from .fdm_script_generator import FDMScriptGenerator, PDEParser
from .fenics_script_generator import FEniCSScriptGenerator, classify_physics_shape

__all__ = [
    "FDMScriptGenerator",
    "PDEParser",
    "FEniCSScriptGenerator",
    "classify_physics_shape",
]
