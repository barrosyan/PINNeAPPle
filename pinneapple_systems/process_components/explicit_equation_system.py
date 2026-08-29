"""pinneapple_systems.process_components.explicit_equation_system -- a
generic, sandboxed "define your model as named equations" engine: given
a set of scalar/array symbols (inputs, parameters, computed
intermediates, and outputs) and a set of equations relating them, safely
parse and evaluate the system in dependency order, and optionally
calibrate bounded parameters against measured data via least squares.

This module has ZERO domain content -- it is the generalization of two
structurally near-identical, independently-written engines found doing
exactly this (one for an explicit rheology model, one for closed-form
beam-deflection formulas), unified here as a single reusable capability.
Any explicit-formula physical model (a rheology curve, a beam deflection
family, a valve-sizing correlation, ...) can be expressed as a
configuration of symbols + equation strings against this engine instead
of hand-written evaluation code.

SELECTED FORMULATION
---------------------
A model is: a set of INPUT symbols (leaf values the caller supplies), a
set of PARAMETER symbols (leaf values, optionally calibrated), a set of
INTERMEDIATE symbols (computed, available to other equations but not
part of the final reported result), and a set of OUTPUT symbols
(computed, the final reported result) -- each intermediate/output symbol
has EXACTLY one equation whose left-hand side is that symbol; inputs and
parameters have none (they are leaves). Equations may reference any
other declared symbol on their right-hand side; the DEPENDENCY GRAPH
those references form is topologically sorted (`evaluation_order`) so
equations do not need to be declared in evaluation order, and a cycle is
a validation error, not a runtime one.

SAFETY: the right-hand side of every equation is parsed with `ast.parse`
and walked through an explicit ALLOWLIST (`_validate_ast`) before ever
being evaluated -- only numeric literals, symbol names, the whitelisted
binary/unary/comparison operators, and calls to the whitelisted function
set below are permitted. Anything else (attribute access, subscripting,
comprehensions, string/bool literals, arbitrary calls) raises
`ExplicitEquationError` at parse time. This is what makes it safe to
accept equation strings from an untrusted caller (e.g. a UI where a user
types their own formula) rather than only ever running code the
developer wrote.

WHITELISTED FUNCTIONS: `abs, sqrt, exp, log, sin, cos, tan` (unary
NumPy ufuncs), `max, min` (variadic, via `functools.reduce`),
`heaviside(x, boundary_value)` (NumPy's two-argument step function --
the standard way to build a piecewise closed-form solution without an
`if`, which isn't in the safe-AST allowlist), `where(cond, a, b)`,
`maximum(a, b)`, `minimum(a, b)`, `clip(v, lo, hi)`, `array(*args)`, and
`linspace(start, stop, step)` -- NOTE this is NumPy's `arange`-like
start/stop/STEP convention (not `numpy.linspace`'s start/stop/COUNT),
matching the source convention this engine generalizes, capped at
`_ARRAY_LIMIT` points to bound memory/compute from a runaway step value.

NUMERICAL METHOD: `calibrate` uses `scipy.optimize.least_squares` with
explicit parameter bounds (Trust Region Reflective, the standard
algorithm for a bounded nonlinear least-squares problem), fitting the
full equation system (not a linearization of it) against measured rows,
with each target output's residual scaled by the mean absolute measured
value for that output so multi-output residuals are commensurable in one
combined objective vector.
"""
from __future__ import annotations

import ast
import operator
from dataclasses import dataclass, field
from functools import reduce
from typing import Any, Callable

import numpy as np
from scipy.optimize import least_squares

_SYMBOL_RE = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ARRAY_LIMIT = 20_001


class ExplicitEquationError(ValueError):
    pass


# --- safe expression evaluation --------------------------------------------
_CALLS: dict[str, Callable] = {
    "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp, "log": np.log,
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "float64": np.float64,
}
_REDUCE_CALLS: dict[str, Callable] = {"max": np.maximum, "min": np.minimum}
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: np.power, ast.Mod: operator.mod,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CMP_OPS = {
    ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
    ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
}
_SPECIAL_CALLS = {"linspace", "where", "maximum", "minimum", "clip", "array", "heaviside"}


def _validate_ast(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_ast(node.body)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ExplicitEquationError(f"unsupported literal: {node.value!r}")
    elif isinstance(node, ast.Name):
        pass
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BIN_OPS:
            raise ExplicitEquationError(f"unsupported operator: {type(node.op).__name__}")
        _validate_ast(node.left)
        _validate_ast(node.right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _UNARY_OPS:
            raise ExplicitEquationError(f"unsupported unary operator: {type(node.op).__name__}")
        _validate_ast(node.operand)
    elif isinstance(node, ast.Compare):
        if len(node.ops) != 1 or type(node.ops[0]) not in _CMP_OPS:
            raise ExplicitEquationError("unsupported comparison")
        _validate_ast(node.left)
        _validate_ast(node.comparators[0])
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExplicitEquationError("only direct function calls are supported")
        name = node.func.id
        if name not in _CALLS and name not in _REDUCE_CALLS and name not in _SPECIAL_CALLS:
            raise ExplicitEquationError(f"unknown function: {name!r}")
        if node.keywords:
            raise ExplicitEquationError("keyword arguments are not supported")
        for arg in node.args:
            _validate_ast(arg)
    else:
        raise ExplicitEquationError(f"unsupported syntax: {type(node).__name__}")


def _eval_ast(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ExplicitEquationError(f"undefined symbol: {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.BinOp):
        return _BIN_OPS[type(node.op)](_eval_ast(node.left, env), _eval_ast(node.right, env))
    if isinstance(node, ast.UnaryOp):
        return _UNARY_OPS[type(node.op)](_eval_ast(node.operand, env))
    if isinstance(node, ast.Compare):
        return _CMP_OPS[type(node.ops[0])](_eval_ast(node.left, env), _eval_ast(node.comparators[0], env))
    if isinstance(node, ast.Call):
        name = node.func.id
        args = [_eval_ast(a, env) for a in node.args]
        if name in _CALLS:
            if len(args) != 1:
                raise ExplicitEquationError(f"{name} takes exactly one argument")
            return _CALLS[name](args[0])
        if name in _REDUCE_CALLS:
            if len(args) < 2:
                raise ExplicitEquationError(f"{name} takes at least two arguments")
            return reduce(_REDUCE_CALLS[name], args)
        if name == "linspace":
            start, stop, step = args
            n = int(np.ceil(abs(stop - start) / max(abs(step), 1e-12))) + 1
            if n > _ARRAY_LIMIT:
                raise ExplicitEquationError(f"linspace would produce {n} points, exceeding the {_ARRAY_LIMIT} limit")
            return np.arange(start, stop + step * 0.5, step)
        if name == "where":
            cond, a, b = args
            return np.where(cond, a, b)
        if name == "maximum":
            return np.maximum(*args)
        if name == "minimum":
            return np.minimum(*args)
        if name == "clip":
            v, lo, hi = args
            return np.clip(v, lo, hi)
        if name == "array":
            return np.array(args)
        if name == "heaviside":
            x, boundary = args
            return np.heaviside(x, boundary)
        raise ExplicitEquationError(f"unknown function: {name!r}")
    raise ExplicitEquationError(f"unsupported syntax: {type(node).__name__}")


def safe_eval(expr: str, env: dict[str, Any]) -> Any:
    tree = ast.parse(expr, mode="eval")
    _validate_ast(tree)
    return _eval_ast(tree, env)


def _names_in_expr(expr: str) -> set[str]:
    tree = ast.parse(expr, mode="eval")
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - set(_CALLS) - set(_REDUCE_CALLS) - _SPECIAL_CALLS


# --- symbol/equation system --------------------------------------------------
@dataclass(frozen=True)
class Definition:
    name: str          # equation name/label (for error messages, not a symbol)
    lhs: str           # the symbol this equation computes
    rhs: str           # the expression string
    rhs_symbols: set[str]


def _split_equation(name: str, equation: str) -> tuple[str, str]:
    if "=" not in equation:
        raise ExplicitEquationError(f"equation {name!r} has no '=': {equation!r}")
    lhs, rhs = equation.split("=", 1)
    lhs = lhs.strip()
    if not _SYMBOL_RE.match(lhs):
        raise ExplicitEquationError(f"equation {name!r} left-hand side is not a bare symbol: {lhs!r}")
    return lhs, rhs.strip()


def build_definitions(
    inputs: set[str], parameters: set[str], intermediates: set[str], outputs: set[str],
    equations: list[dict[str, str]],
) -> tuple[list[Definition], list[str]]:
    """Validates and builds Definitions from raw {"name","equation"}
    dicts. Returns (definitions, errors) -- errors accumulate rather
    than failing on the first one, so a caller can report everything
    wrong with a submitted model at once."""
    errors: list[str] = []
    leaves = inputs | parameters
    computed = intermediates | outputs
    overlap = leaves & computed
    if overlap:
        errors.append(f"symbols declared as both leaf and computed: {sorted(overlap)}")

    definitions: list[Definition] = []
    seen_lhs: set[str] = set()
    for eq in equations:
        try:
            name = eq["name"]
            lhs, rhs = _split_equation(name, eq["equation"])
            if lhs in leaves:
                errors.append(f"equation {name!r} writes to leaf symbol {lhs!r} (inputs/parameters cannot have equations)")
                continue
            if lhs not in computed:
                errors.append(f"equation {name!r} writes to undeclared symbol {lhs!r}")
                continue
            if lhs in seen_lhs:
                errors.append(f"symbol {lhs!r} is written by more than one equation")
                continue
            rhs_symbols = _names_in_expr(rhs)
            unknown = rhs_symbols - leaves - computed
            if unknown:
                errors.append(f"equation {name!r} references undeclared symbols: {sorted(unknown)}")
                continue
            seen_lhs.add(lhs)
            definitions.append(Definition(name=name, lhs=lhs, rhs=rhs, rhs_symbols=rhs_symbols))
        except ExplicitEquationError as exc:
            errors.append(str(exc))

    missing = computed - seen_lhs
    if missing:
        errors.append(f"declared computed symbols with no equation: {sorted(missing)}")

    return definitions, errors


def evaluation_order(definitions: list[Definition]) -> list[Definition]:
    """Topological sort by LHS symbol, DFS with cycle detection."""
    by_lhs = {d.lhs: d for d in definitions}
    visiting: set[str] = set()
    visited: set[str] = set()
    order: list[Definition] = []

    def visit(symbol: str) -> None:
        if symbol in visited:
            return
        if symbol in visiting:
            raise ExplicitEquationError(f"cycle detected in evaluation order at {symbol!r}")
        visiting.add(symbol)
        d = by_lhs[symbol]
        for dep in sorted(d.rhs_symbols & set(by_lhs)):
            visit(dep)
        visiting.discard(symbol)
        visited.add(symbol)
        order.append(d)

    for symbol in by_lhs:
        visit(symbol)
    return order


def evaluate(
    input_values: dict[str, Any], parameter_values: dict[str, float],
    order: list[Definition],
) -> dict[str, Any]:
    """Evaluates every definition in the given (topological) order,
    returning the full symbol -> value environment (leaves + every
    computed symbol)."""
    env: dict[str, Any] = {**input_values, **parameter_values}
    for d in order:
        env[d.lhs] = safe_eval(d.rhs, env)
    return env


@dataclass(frozen=True)
class AnalysisResult:
    valid: bool
    errors: list[str]
    evaluation_order: list[str]


def analyze(
    inputs: set[str], parameters: set[str], intermediates: set[str], outputs: set[str],
    equations: list[dict[str, str]],
) -> AnalysisResult:
    """Pure validation/dry-run -- builds and orders the system, reports
    errors, does NOT evaluate any numbers."""
    definitions, errors = build_definitions(inputs, parameters, intermediates, outputs, equations)
    if errors:
        return AnalysisResult(False, errors, [])
    try:
        order = evaluation_order(definitions)
    except ExplicitEquationError as exc:
        return AnalysisResult(False, [str(exc)], [])
    return AnalysisResult(True, [], [d.lhs for d in order])


@dataclass(frozen=True)
class ParameterSpec:
    symbol: str
    value: float
    fit: bool = False
    min: float | None = None
    max: float | None = None


@dataclass(frozen=True)
class CalibrationResult:
    fitted_parameters: dict[str, float]
    start: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    n_evaluations: int
    final_scaled_residual_ssr: float
    termination_message: str


def calibrate(
    input_columns: dict[str, np.ndarray], parameters: list[ParameterSpec],
    order: list[Definition], measured_rows: list[dict[str, float]], target_symbols: list[str],
) -> CalibrationResult:
    """Bounded nonlinear least-squares calibration (Trust Region
    Reflective) of every parameter with `fit=True` against
    `measured_rows`'s target-symbol columns, re-evaluating the FULL
    equation system (not a linearization) at every trial parameter
    vector -- see module docstring's NUMERICAL METHOD."""
    fit_specs = [p for p in parameters if p.fit]
    if not fit_specs:
        raise ExplicitEquationError("no parameters marked fit=True")
    if not measured_rows:
        raise ExplicitEquationError("no measured_rows supplied for calibration")
    for p in fit_specs:
        if p.min is None or p.max is None or p.min >= p.max:
            raise ExplicitEquationError(f"parameter {p.symbol!r} needs min < max to be calibrated")

    fixed_params = {p.symbol: p.value for p in parameters if not p.fit}
    names = [p.symbol for p in fit_specs]
    start = np.array([p.value for p in fit_specs])
    lo = np.array([p.min for p in fit_specs])
    hi = np.array([p.max for p in fit_specs])

    targets: dict[str, np.ndarray] = {}
    scales: dict[str, float] = {}
    for sym in target_symbols:
        vals = np.array([row[sym] for row in measured_rows if sym in row], dtype=float)
        targets[sym] = vals
        scales[sym] = float(np.mean(np.abs(vals))) or 1.0

    def residual(theta: np.ndarray) -> np.ndarray:
        params = dict(fixed_params)
        params.update(zip(names, theta))
        env = evaluate(input_columns, params, order)
        pieces = []
        for sym in target_symbols:
            computed = np.asarray(env[sym], dtype=float).reshape(-1)
            pieces.append((computed[: len(targets[sym])] - targets[sym]) / scales[sym])
        return np.concatenate(pieces)

    result = least_squares(residual, start, bounds=(lo, hi), xtol=1e-12, ftol=1e-12)
    fitted = dict(zip(names, result.x))
    return CalibrationResult(
        fitted_parameters=fitted,
        start=dict(zip(names, start)),
        bounds={n: (float(l), float(h)) for n, l, h in zip(names, lo, hi)},
        n_evaluations=result.nfev,
        final_scaled_residual_ssr=float(np.sum(result.fun ** 2)),
        termination_message=result.message,
    )
