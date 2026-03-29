from __future__ import annotations
"""pinneaple_validate — Physical consistency validation for trained PINN models.

After training, validates that the model respects physical laws:
conservation, boundary conditions, symmetry, and comparison to analytical
or solver reference solutions.

Classes
-------
PhysicsValidator
    Orchestrator: add checks, then call ``.validate()`` → ValidationReport.
ValidationReport
    Summary of all checks with pass/fail status and a printable table.
ConservationCheck
    Checks integral conservation laws (mass, energy, custom integrals)
    using Monte Carlo integration and autograd.
BoundaryCheck
    Checks Dirichlet, Neumann, and periodicity boundary conditions.
SymmetryCheck
    Checks reflection and 2D rotational symmetry.

Functions
---------
validate_model(model, spec, ...) → ValidationReport
    Quick one-shot validation from a ProblemSpec.
compare_to_analytical(model, analytical_fn, ...) → dict
    Computes RMSE, rel-L2, and max error vs an analytical solution.
validate_against_solver(model, solver_data, ...) → dict
    Compares model predictions to solver reference data.

Quick start
-----------
>>> from pinneaple_validate import PhysicsValidator, validate_model
>>> validator = PhysicsValidator(model, coord_names=["x", "t"],
...                              domain_bounds={"x": (-1, 1), "t": (0, 1)})
>>> report = validator.validate()
>>> print(report.summary())
"""

from .core import CheckResult, ValidationReport
from .conservation import ConservationCheck
from .boundary import BoundaryCheck
from .symmetry import SymmetryCheck
from .validator import PhysicsValidator, validate_model
from .analytical import compare_to_analytical, validate_against_solver

__all__ = [
    "CheckResult",
    "ValidationReport",
    "ConservationCheck",
    "BoundaryCheck",
    "SymmetryCheck",
    "PhysicsValidator",
    "validate_model",
    "compare_to_analytical",
    "validate_against_solver",
]
