"""Tests for pinneapple_problemdesign.method_selection: connecting a
computed flow regime (pinneapple_analysis.verification.dimensional_analysis
.classify_flow_regime) to a recommended (numerical_method, turbulence_model)
pair, validated against the real SolverRegistry and TurbulenceModel/
get_turbulence_closure.

Covers:
  (a) a clearly laminar (low-Re) case recommends a real, registered solver
      with no turbulence closure;
  (b) a clearly turbulent case recommends a real solver + a real,
      constructible TurbulenceModel;
  (c) insufficient info: recommend_method degrades to confidence="low"
      (never raises); recommend_method_from_spec returns None when
      ProblemSpec carries no structured numeric physics parameters (true of
      every spec today);
  (d) every solver name recommend_method can produce is cross-checked
      against the live SolverRegistry.list(), not a hardcoded expected list;
  (e) build_plan's existing behavior is byte-identical when
      recommend_method_from_spec returns None, mirroring the existing
      orchestrator-bridge non-breaking test pattern.
"""
from __future__ import annotations

import pytest

from pinneapple_analysis.verification.dimensional_analysis import classify_flow_regime
from pinneapple_physics.pde_environment.turbulence_selector import (
    TurbulenceModel,
    get_turbulence_closure,
)
from pinneapple_simulation.numerical_solvers.registry import SolverRegistry, register_all
from pinneapple_problemdesign.method_selection import (
    MethodRecommendation,
    recommend_method,
    recommend_method_from_spec,
)
from pinneapple_problemdesign.schema import GeometrySpec, PhysicsSpec, ProblemSpec
from pinneapple_problemdesign.knowledge.mapping import (
    build_plan,
    build_plan_cfd_first,
    build_plan_fno_first,
    build_plan_pinn_first,
)


def _live_solver_names():
    register_all()
    return set(SolverRegistry.list())


# ---------------------------------------------------------------------------
# (a) Clearly laminar case
# ---------------------------------------------------------------------------

def test_laminar_case_recommends_registered_solver_no_turbulence_closure():
    # Water-like flow in a small pipe: Re = U*L/nu = 0.05*0.01/1e-6 = 500 (laminar, internal pipe).
    rec = recommend_method(0.05, 0.01, 1.0e-6, geometry="internal_pipe")

    assert isinstance(rec, MethodRecommendation)
    assert rec.turbulence_model is None
    assert "laminar" in rec.flow_regime
    assert rec.numerical_method in _live_solver_names()
    assert rec.confidence == "high"
    # Rationale must cite the actual regime, not be a generic placeholder.
    assert rec.flow_regime in rec.rationale or "laminar" in rec.rationale


def test_stokes_creeping_flow_also_gets_no_turbulence_closure():
    # Re << 1 external bluff-body: Re = 0.001*0.01/1e-3 = 1e-5.
    rec = recommend_method(0.001, 0.01, 1.0e-3, geometry="external_bluff_body")
    assert rec.turbulence_model is None
    assert "Stokes" in rec.flow_regime or "creeping" in rec.flow_regime
    assert rec.numerical_method in _live_solver_names()


# ---------------------------------------------------------------------------
# (b) Clearly turbulent case
# ---------------------------------------------------------------------------

def test_turbulent_case_recommends_solver_and_real_turbulence_model():
    # Re = 10*1.0/1e-6 = 1e7 (well into turbulent, internal pipe).
    rec = recommend_method(10.0, 1.0, 1.0e-6, geometry="internal_pipe")

    assert "turbulent" in rec.flow_regime
    assert rec.numerical_method in _live_solver_names()
    assert isinstance(rec.turbulence_model, TurbulenceModel)

    # The recommended TurbulenceModel must actually be constructible for the
    # recommended solver's family -- re-derive the same solver_family
    # mapping method_selection uses (lbm -> "lbm", everything else -> "pinn")
    # and call the REAL get_turbulence_closure to prove it isn't a fabricated
    # enum value.
    solver_family = "lbm" if rec.numerical_method == "lbm" else "pinn"
    closure = get_turbulence_closure(rec.turbulence_model, dim=2, solver_family=solver_family)
    if rec.turbulence_model is TurbulenceModel.LAMINAR:
        assert closure is None or closure == 0.0
    else:
        assert closure is not None


def test_external_bluff_body_turbulent_wake_is_flagged_low_confidence():
    # Re >= 1e3 external bluff-body -> classify_flow_regime's own
    # "order-of-magnitude estimate" turbulent-wake label.
    rec = recommend_method(5.0, 1.0, 1.0e-5, geometry="external_bluff_body")
    assert "turbulent" in rec.flow_regime
    assert "order-of-magnitude" in rec.flow_regime
    assert rec.confidence == "low"


# ---------------------------------------------------------------------------
# (c) Insufficient information
# ---------------------------------------------------------------------------

def test_recommend_method_degrades_to_low_confidence_never_raises():
    # kinematic_viscosity=0 => Reynolds number cannot be computed (see
    # compute_dimensionless_numbers' `nu > 0` guard) => "unknown" regime.
    rec = recommend_method(1.0, 1.0, 0.0)
    assert "unknown" in rec.flow_regime
    assert rec.confidence == "low"
    assert rec.turbulence_model is None
    assert rec.numerical_method in _live_solver_names()


def test_recommend_method_from_spec_returns_none_for_ordinary_spec():
    """Today's ProblemSpec/PhysicsSpec schema has no structured numeric
    velocity/length_scale/kinematic_viscosity fields (parameters_known is
    just a list of bare names -- see codegen.py's handling of it), so this
    must gracefully return None rather than guessing or raising."""
    spec = ProblemSpec(
        title="Airfoil AoA sweep",
        goal="Predict lift and drag for an airfoil across an AoA sweep",
        task_type="other",
        physics=PhysicsSpec(
            numerical_method="LBM",
            parameters_known=["reynolds_number", "kinematic_viscosity"],
        ),
        geometry=GeometrySpec(domain="external airfoil", aoa_sweep_deg=[0.0, 10.0]),
    )
    assert recommend_method_from_spec(spec) is None


def test_recommend_method_from_spec_activates_if_numeric_fields_present():
    """Forward-compatibility check: if a PhysicsSpec instance happens to
    carry the (currently nonexistent) numeric attributes this wrapper looks
    for, it must actually use them rather than always returning None."""
    physics = PhysicsSpec()
    physics.velocity = 10.0
    physics.length_scale = 1.0
    physics.kinematic_viscosity = 1.0e-6
    spec = ProblemSpec(title="t", goal="g", task_type="other", physics=physics)

    rec = recommend_method_from_spec(spec)
    assert rec is not None
    assert isinstance(rec, MethodRecommendation)
    assert rec.numerical_method in _live_solver_names()


# ---------------------------------------------------------------------------
# (d) Every recommendable solver name is real (cross-checked against the
#     live registry across a spread of inputs, not a hardcoded list)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "velocity, length_scale, nu, geometry, compressible",
    [
        (0.05, 0.01, 1.0e-6, "internal_pipe", False),
        (10.0, 1.0, 1.0e-6, "internal_pipe", False),
        (2.5, 0.5, 1.0e-6, "internal_pipe", False),
        (0.001, 0.01, 1.0e-3, "external_bluff_body", False),
        (5.0, 1.0, 1.0e-5, "external_bluff_body", False),
        (300.0, 1.0, 1.5e-5, "internal_pipe", True),
        (1.0, 1.0, 0.0, "internal_pipe", False),
    ],
)
def test_recommended_solver_always_registered(velocity, length_scale, nu, geometry, compressible):
    rec = recommend_method(velocity, length_scale, nu, geometry=geometry, compressible=compressible)
    assert rec.numerical_method in _live_solver_names()


def test_invalid_recommended_name_would_raise_not_silently_pass():
    """Sanity-check the internal validator actually catches a bad name
    (proves _validate_registered is a real check, not a no-op)."""
    from pinneapple_problemdesign.method_selection import _validate_registered

    with pytest.raises(RuntimeError):
        _validate_registered("definitely_not_a_real_solver_xyz")


# ---------------------------------------------------------------------------
# (e) build_plan unchanged when recommend_method_from_spec returns None
# ---------------------------------------------------------------------------

def _cfd_spec() -> ProblemSpec:
    return ProblemSpec(
        title="Airfoil AoA sweep",
        goal="Predict lift and drag coefficients for an airfoil across an angle of attack sweep.",
        task_type="other",
        domain_context="external aerodynamics wind tunnel study",
        physics=PhysicsSpec(numerical_method="LBM"),
        geometry=GeometrySpec(aoa_sweep_deg=[0.0, 5.0, 10.0, 15.0]),
    )


def _pinn_spec() -> ProblemSpec:
    return ProblemSpec(title="Heat conduction", goal="Solve 2D heat conduction PDE", task_type="pde_solution")


def _fno_spec() -> ProblemSpec:
    return ProblemSpec(title="Sensor forecasting", goal="Forecast sensor readings", task_type="forecasting")


@pytest.mark.parametrize("spec_factory", [_cfd_spec, _pinn_spec, _fno_spec])
def test_build_plan_unchanged_when_method_selection_returns_none(spec_factory):
    """None of these specs carry structured numeric physics parameters, so
    recommend_method_from_spec(spec) is None and build_plan's output (with
    the orchestrator bridge disabled, isolating this feature) must be
    byte-identical to the pre-existing static sub-builder's output --
    mirroring test_build_plan_bridge_disabled_matches_sub_builder in
    tests/test_problemdesign_orchestrator_bridge.py."""
    spec = spec_factory()
    gaps = []

    assert recommend_method_from_spec(spec) is None

    plan_via_dispatch = build_plan(spec, gaps, use_orchestrator_bridge=False)

    if spec.task_type == "pde_solution":
        plan_direct = build_plan_pinn_first(spec, gaps)
    elif spec.task_type == "forecasting":
        plan_direct = build_plan_fno_first(spec, gaps)
    else:
        plan_direct = build_plan_cfd_first(spec, gaps)

    assert plan_via_dispatch == plan_direct


def test_build_plan_adds_method_selection_step_when_recommendation_exists():
    physics = PhysicsSpec()
    physics.velocity = 10.0
    physics.length_scale = 1.0
    physics.kinematic_viscosity = 1.0e-6
    spec = ProblemSpec(title="Heat conduction", goal="Solve 2D heat conduction PDE", task_type="pde_solution", physics=physics)

    static_plan = build_plan_pinn_first(spec, [])
    plan = build_plan(spec, [], use_orchestrator_bridge=False)

    assert len(plan.steps) == len(static_plan.steps) + 1
    extra = plan.steps[-1]
    assert extra.title == "Recommended numerical method"
    assert extra.actions
    # Everything else must be untouched.
    assert plan.recommended_approach == static_plan.recommended_approach
    assert plan.steps[: len(static_plan.steps)] == static_plan.steps
