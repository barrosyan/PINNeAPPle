"""pinneaple_learning — Structured learning paths and hands-on course for Physics AI.

This module guides you from your very first physics-informed neural network
all the way to production-scale deployments with NVIDIA PhysicsNeMo, using
PINNeAPPle's real APIs at every step.

Quick start — the course
------------------------
>>> import pinneaple_learning.course as course
>>> course.list_lessons()        # see all 12 lessons
>>> course.run_lesson(1)         # run lesson 1 (harmonic oscillator)

Or run individual lessons from the terminal:
    python -m pinneaple_learning.course.lesson_01_first_pinn
    python -m pinneaple_learning.course.lesson_03_forward_pde
    python -m pinneaple_learning.course.lesson_07_inverse_problem
    ...

Quick start — learning paths and concepts
-----------------------------------------
>>> import pinneaple_learning as pl
>>> pl.where_am_i()              # self-assessment: which tier are you?
>>> pl.learning_path(tier=1)     # Explorer path
>>> pl.list_topics()             # all Physics AI concepts
>>> pl.explain("pinn")
>>> pl.explain("physicsnemo")

Course overview (12 lessons using PINNeAPPle APIs)
---------------------------------------------------
  01  First PINN                 — SymbolicPDE, HardBC
  02  Loss function anatomy      — DirichletBC, SelfAdaptiveWeights
  03  Forward PDE (heat)         — TwoPhaseTrainer, compare_to_analytical
  04  Geometry generation        — CSGRectangle, CSGCircle, CSGDifference
  05  2D Poisson on L-shape      — full 2D forward workflow
  06  Model architectures        — SIREN, ModifiedMLP, FourierFeatureEmbedding
  07  Inverse problem            — InverseProblemSolver, EnsembleKalmanInversion
  08  Uncertainty quantification — MCDropoutWrapper, EnsembleUQ, ConformalPredictor
  09  Time-marching              — TimeMarchingTrainer (stiff Van der Pol)
  10  Physics validation         — PhysicsValidator, compare_to_analytical
  11  Operator learning          — parametric heat family, FourierFeatureEmbedding
  12  Bridge to PhysicsNeMo      — export_onnx, production checklist, migration

Three tiers
-----------
  Tier 1 — Explorer     (start here if new to Physics AI)
  Tier 2 — Experimenter (compare architectures, inverse problems, UQ)
  Tier 3 — Builder      (scale, serve, migrate to PhysicsNeMo)

PhysicsNeMo roadmap
-------------------
  See: pinneaple_learning/physicsnemo_roadmap/migration_guide.md
"""

from .learning_path import LearningPath, where_am_i, learning_path
from .concepts import ConceptGuide, explain, list_topics

__all__ = [
    "LearningPath",
    "where_am_i",
    "learning_path",
    "ConceptGuide",
    "explain",
    "list_topics",
]

__version__ = "0.2.0"
