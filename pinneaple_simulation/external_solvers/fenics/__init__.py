"""FEniCS integration for PINNeAPPle.

This subpackage wraps pinneaple_solvers.fenics_bridge (low-level FEM) and adds
a workflow layer with structured configuration and UPD packaging.

Quick start
-----------
>>> from pinneaple_simulation.external_solvers.fenics import FEniCSConfig, solve_and_package
>>> cfg = FEniCSConfig(
...     pde="heat_equation_steady",
...     domain={"type": "rectangle", "x": [0, 1], "y": [0, 1], "nx": 32, "ny": 32},
...     bcs=[{"type": "dirichlet", "boundary": "left", "value": 0.0},
...          {"type": "dirichlet", "boundary": "right", "value": 1.0}],
...     params={"k": 1.0},
... )
>>> sample = solve_and_package(cfg)

For parametric sweeps:
>>> from pinneaple_simulation.external_solvers.fenics import FEniCSWorkflow
>>> samples = FEniCSWorkflow(cfg).sweep({"k": [0.5, 1.0, 2.0, 5.0]})
"""
from .solver import FEniCSConfig, solve_and_package, dof_to_upd, FEniCSWorkflow

__all__ = [
    "FEniCSConfig",
    "solve_and_package",
    "dof_to_upd",
    "FEniCSWorkflow",
]
