"""Validates pinneapple_systems.process_components.explicit_equation_system
-- the generic safe-expression / equation-system engine -- using two
GENERIC example models (not the specific rheology or beam formulas it
generalizes from) plus direct safety/graph-resolution checks."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.process_components.explicit_equation_system import (
    AnalysisResult,
    ExplicitEquationError,
    ParameterSpec,
    analyze,
    build_definitions,
    calibrate,
    evaluate,
    evaluation_order,
    safe_eval,
)


# --- safe_eval / AST allowlist ----------------------------------------------
def test_safe_eval_basic_arithmetic():
    assert safe_eval("2 + 3 * 4", {}) == 14


def test_safe_eval_resolves_symbols():
    assert safe_eval("a * b + 1", {"a": 3.0, "b": 4.0}) == 13.0


def test_safe_eval_functions():
    assert safe_eval("sqrt(x)", {"x": 16.0}) == pytest.approx(4.0)
    assert safe_eval("max(a, b, c)", {"a": 1.0, "b": 5.0, "c": 3.0}) == 5.0


def test_safe_eval_heaviside_and_linspace():
    z = safe_eval("linspace(0, 10, 5)", {})
    assert list(z) == [0.0, 5.0, 10.0]
    h = safe_eval("heaviside(x, 1)", {"x": np.array([-1.0, 0.0, 1.0])})
    assert list(h) == [0.0, 1.0, 1.0]


def test_safe_eval_rejects_attribute_access():
    with pytest.raises(ExplicitEquationError):
        safe_eval("__import__('os').system('echo hi')", {})


def test_safe_eval_rejects_unknown_function():
    with pytest.raises(ExplicitEquationError):
        safe_eval("eval('1')", {})


def test_safe_eval_rejects_string_literal():
    with pytest.raises(ExplicitEquationError):
        safe_eval("'a' + 'b'", {})


def test_safe_eval_undefined_symbol_raises():
    with pytest.raises(ExplicitEquationError):
        safe_eval("undefined_symbol + 1", {})


def test_linspace_array_limit_enforced():
    with pytest.raises(ExplicitEquationError):
        safe_eval("linspace(0, 1, 1e-9)", {})


# --- build_definitions / evaluation_order -----------------------------------
def test_build_definitions_detects_cycle():
    equations = [{"name": "a_eq", "equation": "a = b + 1"}, {"name": "b_eq", "equation": "b = a + 1"}]
    definitions, errors = build_definitions(set(), set(), {"a", "b"}, set(), equations)
    assert not errors
    with pytest.raises(ExplicitEquationError):
        evaluation_order(definitions)


def test_build_definitions_flags_undeclared_symbol_reference():
    equations = [{"name": "y_eq", "equation": "y = x + z"}]
    definitions, errors = build_definitions({"x"}, set(), set(), {"y"}, equations)
    assert any("z" in e for e in errors)


def test_build_definitions_flags_missing_equation_for_declared_output():
    definitions, errors = build_definitions({"x"}, set(), set(), {"y"}, [])
    assert any("y" in e for e in errors)


def test_build_definitions_flags_duplicate_lhs():
    equations = [{"name": "e1", "equation": "y = 1"}, {"name": "e2", "equation": "y = 2"}]
    definitions, errors = build_definitions(set(), set(), set(), {"y"}, equations)
    assert any("more than one equation" in e for e in errors)


def test_evaluation_order_resolves_out_of_declaration_order():
    # C depends on B depends on A -- declared in reverse order.
    equations = [
        {"name": "c_eq", "equation": "C = B + 1"},
        {"name": "b_eq", "equation": "B = A + 1"},
        {"name": "a_eq", "equation": "A = 1"},
    ]
    definitions, errors = build_definitions(set(), set(), {"A", "B"}, {"C"}, equations)
    assert not errors
    order = evaluation_order(definitions)
    lhs_order = [d.lhs for d in order]
    assert lhs_order.index("A") < lhs_order.index("B") < lhs_order.index("C")


def test_analyze_reports_valid_true_for_a_correct_model():
    equations = [{"name": "y_eq", "equation": "y = 2*x + 1"}]
    result = analyze({"x"}, set(), set(), {"y"}, equations)
    assert result.valid
    assert result.evaluation_order == ["y"]


def test_analyze_reports_valid_false_with_errors_for_a_broken_model():
    result = analyze({"x"}, set(), set(), {"y"}, [])
    assert not result.valid
    assert result.errors


# --- a GENERIC "PV/YP-style" example: simple scalar chain ------------------
def test_scalar_chain_model_evaluates_correctly():
    # Generic analog of the rheology-style model: a temperature-corrected
    # base value feeding two derived readings and a difference.
    equations = [
        {"name": "base_eq", "equation": "base = base_ref * max(1.0 - 0.002*(T - T_ref), 0.30)"},
        {"name": "hi_eq", "equation": "hi = base + k * 2.0"},
        {"name": "lo_eq", "equation": "lo = base + k * 1.0"},
        {"name": "delta_eq", "equation": "delta = hi - lo"},
    ]
    definitions, errors = build_definitions(
        {"T"}, {"base_ref", "T_ref", "k"}, {"base", "hi", "lo"}, {"delta"}, equations,
    )
    assert not errors
    order = evaluation_order(definitions)
    env = evaluate({"T": 100.0}, {"base_ref": 10.0, "T_ref": 80.0, "k": 3.0}, order)
    expected_base = 10.0 * max(1.0 - 0.002 * (100.0 - 80.0), 0.30)
    assert env["base"] == pytest.approx(expected_base)
    assert env["delta"] == pytest.approx(3.0)  # hi - lo = k*2 - k*1 = k = 3.0


# --- a GENERIC "EBT-style" example: piecewise/array model -------------------
def test_piecewise_array_model_with_linspace_and_heaviside():
    # Generic analog of a piecewise closed-form beam formula: two
    # branches switched by heaviside at a cutoff position.
    equations = [
        {"name": "x_eq", "equation": "X = linspace(0, L, 1.0)"},
        {"name": "y_eq", "equation": "Y = (X*X)*heaviside(a - X, 1) + (a*a + 2*a*(X-a))*heaviside(X - a, 0)"},
    ]
    definitions, errors = build_definitions({"L", "a"}, set(), {"X"}, {"Y"}, equations)
    assert not errors
    order = evaluation_order(definitions)
    env = evaluate({"L": 10.0, "a": 5.0}, {}, order)
    X, Y = env["X"], env["Y"]
    assert X[0] == 0.0 and X[-1] == 10.0
    # Below the cutoff: Y = X^2
    idx_below = int(np.where(X == 3.0)[0][0])
    assert Y[idx_below] == pytest.approx(9.0)
    # Above the cutoff: Y = a^2 + 2a(X-a) (continuous at X=a)
    idx_above = int(np.where(X == 8.0)[0][0])
    assert Y[idx_above] == pytest.approx(5.0 ** 2 + 2 * 5.0 * (8.0 - 5.0))


# --- calibration -------------------------------------------------------------
def test_calibrate_recovers_a_known_linear_parameter():
    # y = m*x + b, fit m against noiseless synthetic data with b fixed.
    equations = [{"name": "y_eq", "equation": "y = m*x + b"}]
    definitions, errors = build_definitions({"x"}, {"m", "b"}, set(), {"y"}, equations)
    assert not errors
    order = evaluation_order(definitions)

    true_m = 3.7
    xs = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = true_m * xs + 1.0
    measured_rows = [{"x": float(x), "y": float(y)} for x, y in zip(xs, ys)]

    result = calibrate(
        input_columns={"x": xs},
        parameters=[ParameterSpec("m", value=1.0, fit=True, min=0.0, max=10.0), ParameterSpec("b", value=1.0, fit=False)],
        order=order, measured_rows=measured_rows, target_symbols=["y"],
    )
    assert result.fitted_parameters["m"] == pytest.approx(true_m, rel=1e-3)


def test_calibrate_requires_bounds_on_fit_parameters():
    equations = [{"name": "y_eq", "equation": "y = m*x"}]
    definitions, errors = build_definitions({"x"}, {"m"}, set(), {"y"}, equations)
    order = evaluation_order(definitions)
    with pytest.raises(ExplicitEquationError):
        calibrate(
            input_columns={"x": np.array([1.0, 2.0])},
            parameters=[ParameterSpec("m", value=1.0, fit=True, min=None, max=None)],
            order=order, measured_rows=[{"x": 1.0, "y": 2.0}], target_symbols=["y"],
        )
