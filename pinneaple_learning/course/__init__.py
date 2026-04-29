"""pinneaple_learning.course — Step-by-step Physics AI course using PINNeAPPle.

Each lesson is a self-contained, runnable Python file that teaches one
concept by solving a real physics problem with PINNeAPPle APIs.

Quick start
-----------
>>> import pinneaple_learning.course as course
>>> course.list_lessons()
>>> course.run_lesson(1)       # run lesson 1 interactively

Or from the terminal:
    python -m pinneaple_learning.course.lesson_01_first_pinn
    python -m pinneaple_learning.course.lesson_03_forward_pde
    ...

Lesson map
----------
  01  Your first PINN              — harmonic oscillator, symbolic PDE, hard IC
  02  Understanding losses         — PDE / BC / IC / data terms, weight tuning
  03  Forward PDE (heat equation)  — spacetime PINN, SymbolicPDE, TwoPhaseTrainer
  04  Geometry generation          — CSG domains, SDF sampling, visualisation
  05  2D Poisson on L-shape        — custom domain, full 2D forward workflow
  06  Model architectures          — SIREN, ModifiedMLP, Fourier features
  07  Inverse problem              — recover diffusivity from noisy observations
  08  Uncertainty quantification   — MC Dropout, Deep Ensemble, Conformal
  09  Time-marching PINNs          — Van der Pol oscillator, stiff problems
  10  Physics validation           — PhysicsValidator, conservation checks
  11  Operator learning            — AFNO surrogate for parametric heat family
  12  Bridge to PhysicsNeMo        — ONNX export, validation, migration overview
"""

import importlib

_LESSONS = {
    1:  ("lesson_01_first_pinn",        "Harmonic oscillator — your first PINN with PINNeAPPle"),
    2:  ("lesson_02_loss_functions",    "Loss function anatomy — PDE, BC, IC, data terms"),
    3:  ("lesson_03_forward_pde",       "Heat equation forward solve — SymbolicPDE + TwoPhaseTrainer"),
    4:  ("lesson_04_geometry",          "Geometry generation — CSG domains and SDF sampling"),
    5:  ("lesson_05_2d_poisson",        "2D Poisson on L-shape — full 2D forward workflow"),
    6:  ("lesson_06_model_architectures","SIREN vs ModifiedMLP vs Tanh on Helmholtz"),
    7:  ("lesson_07_inverse_problem",   "Recover diffusivity from noisy observations"),
    8:  ("lesson_08_uq",                "Uncertainty quantification — MC Dropout, Ensemble, Conformal"),
    9:  ("lesson_09_time_marching",     "Time-marching — Van der Pol stiff oscillator"),
    10: ("lesson_10_validation",        "Physics validation — conservation and BC checks"),
    11: ("lesson_11_operator_learning", "Operator learning — AFNO for parametric heat family"),
    12: ("lesson_12_to_physicsnemo",    "Bridge to NVIDIA PhysicsNeMo — export and migrate"),
}


def list_lessons() -> None:
    """Print the course table of contents."""
    print("\n  PINNeAPPle Course — 12 Lessons\n")
    print(f"  {'#':>3}  {'Title':<55}  Module")
    print("  " + "─" * 75)
    for n, (mod, title) in _LESSONS.items():
        print(f"  {n:>3}  {title:<55}  {mod}")
    print()
    print("  Run a lesson:")
    print("    python -m pinneaple_learning.course.lesson_01_first_pinn")
    print("    # or:")
    print("    import pinneaple_learning.course as c; c.run_lesson(1)\n")


def run_lesson(n: int) -> None:
    """Run lesson n (calls its main() function)."""
    if n not in _LESSONS:
        raise ValueError(f"Lesson {n} not found. Valid: 1–12.")
    mod_name, title = _LESSONS[n]
    print(f"\n{'='*60}")
    print(f"  Lesson {n:02d}: {title}")
    print(f"{'='*60}\n")
    mod = importlib.import_module(f"pinneaple_learning.course.{mod_name}")
    mod.main()
