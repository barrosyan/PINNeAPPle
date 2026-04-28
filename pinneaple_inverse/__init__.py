"""pinneaple_inverse — Inverse Problems module for PINNeAPPle.

Provides a complete toolkit for physics-informed inverse problems:

Core components
---------------
Noise models (data-misfit terms):
  GaussianMisfit            standard L2 / Mahalanobis distance
  HuberMisfit               robust to moderate outliers
  CauchyMisfit              heavy-tailed; very robust
  StudentTMisfit            Student-t likelihood (controls tail weight)
  HeteroscedasticMisfit     per-observation noise variance

Regularization:
  TikhonovRegularizer       L2 ‖θ − θ₀‖², Mahalanobis variant
  SparsityRegularizer       L1 ‖θ‖₁ (smooth approximation)
  TotalVariationRegularizer TV(u) for spatial field parameters
  CompositeRegularizer      weighted sum of multiple regularizers
  LCurveSelector            automatic λ via L-curve + max curvature

Observation operators:
  PointObsOperator          evaluate u at sparse sensor locations
  LinearObsOperator         y = A u (projections, Radon, …)
  IntegralObsOperator       y = ∫ k(x) u(x) dx
  ComposedObsOperator       stack multiple operators

Sensitivity analysis:
  LocalSensitivity          Jacobian J = ∂G/∂θ; FIM = J^T Γ⁻¹ J
  IdentifiabilityAnalyzer   eigenanalysis of FIM; D/A/E-optimality
  GlobalSensitivity         Sobol variance-based indices (Saltelli method)

Derivative-free inversion:
  EnsembleKalmanInversion   standard EKI (Iglesias et al. 2013)
  IteratedEKI               Tikhonov-regularized EKI / TEKI (Chada et al. 2020)

High-level solver:
  InverseProblemSolver      wires all components; supports Adam / L-BFGS / EKI

Quick start
-----------
>>> import pinneaple_inverse as pinv
>>> import torch

>>> # Define forward model (PINN)
>>> from pinneaple_models import ModelRegistry
>>> model = ModelRegistry.build("VanillaPINN", in_dim=2, out_dim=1)

>>> # Observation operator: evaluate at 50 sensor points
>>> sensor_locs = torch.rand(50, 2)
>>> H = pinv.PointObsOperator(sensor_locs)

>>> # Noise model
>>> D = pinv.HuberMisfit(delta=0.3, noise_std=0.02)

>>> # Regularizer
>>> R = pinv.TikhonovRegularizer(lambda_reg=1e-3)

>>> # Solver
>>> cfg = pinv.InverseSolverConfig(method="adam", n_iters=2000, lr=5e-4)
>>> solver = pinv.InverseProblemSolver(model, H, D, R, cfg)

>>> # Synthetic observations
>>> y_obs = H(model, sensor_locs).detach() + 0.02 * torch.randn(50, 1)

>>> # Solve
>>> result = solver.solve(y_obs, sensor_locs)
>>> print(f"Final misfit: {result.final_misfit:.4e}")
"""
from __future__ import annotations

# Noise models
from .noise_models import (
    DataMisfitBase,
    GaussianMisfit,
    HuberMisfit,
    CauchyMisfit,
    StudentTMisfit,
    HeteroscedasticMisfit,
)

# Regularization
from .regularization import (
    RegularizerBase,
    TikhonovRegularizer,
    SparsityRegularizer,
    TotalVariationRegularizer,
    CompositeRegularizer,
    LCurveSelector,
    LCurveResult,
)

# Observation operators
from .obs_operator import (
    ObsOperatorBase,
    PointObsOperator,
    LinearObsOperator,
    IntegralObsOperator,
    ComposedObsOperator,
)

# Sensitivity analysis
from .sensitivity import (
    LocalSensitivity,
    LocalSensitivityResult,
    IdentifiabilityAnalyzer,
    IdentifiabilityResult,
    GlobalSensitivity,
    SobolResult,
)

# Ensemble Kalman Inversion
from .ensemble_kalman import (
    EKIConfig,
    EKIHistory,
    EnsembleKalmanInversion,
    IteratedEKI,
)

# High-level solver
from .solver import (
    InverseSolverConfig,
    InverseSolverResult,
    InverseProblemSolver,
)

__version__ = "0.1.0"

__all__ = [
    # Noise models
    "DataMisfitBase",
    "GaussianMisfit",
    "HuberMisfit",
    "CauchyMisfit",
    "StudentTMisfit",
    "HeteroscedasticMisfit",
    # Regularization
    "RegularizerBase",
    "TikhonovRegularizer",
    "SparsityRegularizer",
    "TotalVariationRegularizer",
    "CompositeRegularizer",
    "LCurveSelector",
    "LCurveResult",
    # Observation operators
    "ObsOperatorBase",
    "PointObsOperator",
    "LinearObsOperator",
    "IntegralObsOperator",
    "ComposedObsOperator",
    # Sensitivity
    "LocalSensitivity",
    "LocalSensitivityResult",
    "IdentifiabilityAnalyzer",
    "IdentifiabilityResult",
    "GlobalSensitivity",
    "SobolResult",
    # EKI
    "EKIConfig",
    "EKIHistory",
    "EnsembleKalmanInversion",
    "IteratedEKI",
    # Solver
    "InverseSolverConfig",
    "InverseSolverResult",
    "InverseProblemSolver",
]
