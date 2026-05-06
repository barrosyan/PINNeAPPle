"""pinneaple_analysis — Model analysis, validation, and inversion.

Sub-modules
-----------
uncertainty     (was pinneaple_uq)
    Uncertainty quantification: aleatoric (AleatoricHead), epistemic
    (MCDropout, EnsembleUQ), conformal prediction, calibration, quantile
    regression, and the unified ``uq_predict`` entry point.

validation      (was pinneaple_validate)
    Physical consistency checks: conservation laws, boundary conditions,
    symmetry, comparison to analytical / solver reference solutions.

inverse_problems  (was pinneaple_inverse)
    Inverse problems: noise models (Gaussian, Huber, …), regularizers
    (Tikhonov, TV, L-curve), observation operators, sensitivity analysis
    (local / global Sobol), EKI / TEKI, and the high-level
    ``InverseProblemSolver``. Also includes SINDy equation discovery.

Integration helpers
-------------------
``analyze_model(model, spec, x, ...)``
    Runs validation + UQ in one call; returns a combined result dict.
``invert(model, observations, ...)``
    High-level inverse-problem shortcut backed by InverseProblemSolver.

Usage
-----
>>> from pinneaple_analysis import validate_model, uq_predict, InverseProblemSolver
>>> report = validate_model(model, spec)
>>> uq = uq_predict(model, x_test, method="mc_dropout")
>>> solver = InverseProblemSolver(model, obs_op, misfit, regularizer, cfg)
"""
from __future__ import annotations

# ── sub-modules (new descriptive names) ───────────────────────────────────────
from . import uncertainty
from . import validation
from . import inverse_problems

# backward-compat aliases
uq       = uncertainty
validate = validation
inverse  = inverse_problems

# ── uncertainty re-exports ────────────────────────────────────────────────────
from .uncertainty import (
    UQResult, uq_predict,
    AleatoricHead, aleatoric_nll_loss,
    decompose_uncertainty,
    MCDropout, MCDropoutConfig, MCDropoutWrapper,
    EnsembleUQ, EnsembleConfig,
    ConformalPredictor,
    CalibrationMetrics,
    QuantileConfig, QuantileHead, QuantileLoss, pinball_loss_torch,
)

# ── validation re-exports ─────────────────────────────────────────────────────
from .validation import (
    CheckResult, ValidationReport,
    ConservationCheck, BoundaryCheck, SymmetryCheck,
    PhysicsValidator, validate_model,
    compare_to_analytical, validate_against_solver,
)

# ── inverse_problems re-exports ───────────────────────────────────────────────
from .inverse_problems import (
    # Noise models
    DataMisfitBase, GaussianMisfit, HuberMisfit,
    CauchyMisfit, StudentTMisfit, HeteroscedasticMisfit,
    # Regularization
    RegularizerBase, TikhonovRegularizer, SparsityRegularizer,
    TotalVariationRegularizer, CompositeRegularizer, LCurveSelector, LCurveResult,
    # Observation operators
    ObsOperatorBase, PointObsOperator, LinearObsOperator,
    IntegralObsOperator, ComposedObsOperator,
    # Sensitivity
    LocalSensitivity, LocalSensitivityResult,
    IdentifiabilityAnalyzer, IdentifiabilityResult,
    GlobalSensitivity, SobolResult,
    # EKI
    EKIConfig, EKIHistory, EnsembleKalmanInversion, IteratedEKI,
    # High-level solver
    InverseSolverConfig, InverseSolverResult, InverseProblemSolver,
    # Equation discovery
    CandidateLibrary, SINDyResult, SINDyIdentifier,
    ResidualAnalysisResult, ResidualAnalyzer,
    NeuralTermConfig, NeuralTermDiscovery,
)


# ── Integration helpers ────────────────────────────────────────────────────────

def analyze_model(model, spec, x_test, *, uq_method: str = "mc_dropout",
                  run_validate: bool = True, run_uq: bool = True) -> dict:
    """Run validation and uncertainty quantification in one call."""
    result: dict = {}
    if run_validate:
        from .validation import validate_model as _validate
        result["validation_report"] = _validate(model, spec)
    else:
        result["validation_report"] = None
    if run_uq:
        from .uncertainty import uq_predict as _uq
        result["uq_result"] = _uq(model, x_test, method=uq_method)
    else:
        result["uq_result"] = None
    return result


def invert(model, y_obs, sensor_locs, *,
           noise_std: float = 0.01,
           lambda_reg: float = 1e-3,
           method: str = "adam",
           n_iters: int = 2000) -> "InverseSolverResult":
    """High-level inverse-problem shortcut (Gaussian misfit + Tikhonov)."""
    H = PointObsOperator(sensor_locs)
    D = GaussianMisfit(noise_std=noise_std)
    R = TikhonovRegularizer(lambda_reg=lambda_reg)
    cfg = InverseSolverConfig(method=method, n_iters=n_iters)
    solver = InverseProblemSolver(model, H, D, R, cfg)
    return solver.solve(y_obs, sensor_locs)


__all__ = [
    # Sub-modules (new names)
    "uncertainty", "validation", "inverse_problems",
    # Sub-modules (old aliases — backward compat)
    "uq", "validate", "inverse",
    # Integration
    "analyze_model", "invert",
    # uncertainty
    "UQResult", "uq_predict",
    "AleatoricHead", "aleatoric_nll_loss", "decompose_uncertainty",
    "MCDropout", "MCDropoutConfig", "MCDropoutWrapper",
    "EnsembleUQ", "EnsembleConfig",
    "ConformalPredictor", "CalibrationMetrics",
    "QuantileConfig", "QuantileHead", "QuantileLoss", "pinball_loss_torch",
    # validation
    "CheckResult", "ValidationReport",
    "ConservationCheck", "BoundaryCheck", "SymmetryCheck",
    "PhysicsValidator", "validate_model",
    "compare_to_analytical", "validate_against_solver",
    # inverse_problems
    "DataMisfitBase", "GaussianMisfit", "HuberMisfit",
    "CauchyMisfit", "StudentTMisfit", "HeteroscedasticMisfit",
    "RegularizerBase", "TikhonovRegularizer", "SparsityRegularizer",
    "TotalVariationRegularizer", "CompositeRegularizer", "LCurveSelector", "LCurveResult",
    "ObsOperatorBase", "PointObsOperator", "LinearObsOperator",
    "IntegralObsOperator", "ComposedObsOperator",
    "LocalSensitivity", "LocalSensitivityResult",
    "IdentifiabilityAnalyzer", "IdentifiabilityResult",
    "GlobalSensitivity", "SobolResult",
    "EKIConfig", "EKIHistory", "EnsembleKalmanInversion", "IteratedEKI",
    "InverseSolverConfig", "InverseSolverResult", "InverseProblemSolver",
    "CandidateLibrary", "SINDyResult", "SINDyIdentifier",
    "ResidualAnalysisResult", "ResidualAnalyzer",
    "NeuralTermConfig", "NeuralTermDiscovery",
]
