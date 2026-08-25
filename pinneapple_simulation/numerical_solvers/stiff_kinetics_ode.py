"""Stiff-ODE convenience wrapper for coupled reaction-network / chemical- or
biological-kinetics systems: dy_i/dt = rate_fn(t, y)[i], y_i(t0) = y0[i].

Reaction networks with rate constants spanning several orders of magnitude
(fast equilibria alongside slow accumulation) are numerically STIFF: an
explicit method (e.g. RK45) is forced to take vanishingly small time steps
for stability even where the solution itself is smooth, making it
impractically slow. This wraps `scipy.integrate.solve_ivp` with an implicit
method appropriate for stiff systems (Radau by default -- an implicit
Runge-Kutta method, A-stable and L-stable, generally the most robust choice
for genuinely stiff chemical kinetics; BDF is offered as an alternative,
typically faster for very large state vectors where each Radau stage's
implicit solve gets expensive), plus two conventions that matter in
practice for concentration-type state variables:

  1. Optional non-negative clipping of the state BEFORE each `rate_fn`
     evaluation (`clip_negative=True`, the default) -- floating-point
     under/overshoot near zero is common for a fast-depleting species close
     to depletion, and most rate laws (e.g. anything with a `sqrt` or a
     fractional-order term) are undefined or nonsensical for negative
     concentrations.
  2. Raising `RuntimeError` on integration failure rather than returning a
     silently-wrong/truncated result -- `solve_ivp` reports failure via a
     `.success` flag that is easy to forget to check.

No rate law is hardcoded here: `rate_fn(t, y) -> dy/dt` (and the optional
analytic Jacobian `jac_fn(t, y) -> d(rate)/dy`) are entirely caller-supplied,
so this applies to any reaction/kinetics system -- disinfection chemistry,
combustion, enzyme kinetics, epidemiological compartments, or any other
stiff first-order ODE system -- not a fixed catalog of reactions.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

RateFn = Callable[[float, np.ndarray], np.ndarray]
JacFn = Callable[[float, np.ndarray], np.ndarray]


def solve_stiff_reaction_network(
    rate_fn: RateFn,
    y0: Sequence[float],
    t_span: "tuple[float, float]",
    t_eval: Optional[np.ndarray] = None,
    jac_fn: Optional[JacFn] = None,
    method: str = "Radau",
    rtol: float = 1e-9,
    atol: float = 1e-12,
    clip_negative: bool = True,
    max_step: Optional[float] = None,
) -> dict:
    """Integrate dy/dt = rate_fn(t, y), y(t0) = y0, over t_span = (t0, tf)
    with a stiff-appropriate implicit method.

    rate_fn(t, y): y is (n,) -> returns (n,), the reaction rate law.
    jac_fn(t, y): optional (n,n) analytic Jacobian d(rate_i)/d(y_j) --
        strongly preferred when known in closed form: an implicit method's
        per-step cost is dominated by Jacobian evaluation/factorization, and
        an analytic Jacobian avoids both finite-difference approximation
        error and its extra rate_fn calls.
    method: "Radau" (default) or "BDF" (both implicit, suited to stiff
        systems); "LSODA" (automatic stiff/non-stiff switching) is also
        accepted, forwarded as-is to `scipy.integrate.solve_ivp`.
    clip_negative: clip y to >= 0 immediately before every rate_fn/jac_fn
        evaluation (default True) -- see module docstring point 1. Set False
        for state variables that are not physically non-negative
        concentrations (e.g. a signed potential or a displacement).

    Returns {"t": (m,), "y": (n, m), "success": bool}. Raises RuntimeError
    if the integration does not converge (never returns a silently-wrong
    partial result).
    """
    from scipy.integrate import solve_ivp

    y0_arr = np.asarray(y0, dtype=np.float64)

    def _rhs(t, y):
        y_eval = np.maximum(y, 0.0) if clip_negative else y
        return np.asarray(rate_fn(t, y_eval), dtype=np.float64)

    _jac = None
    if jac_fn is not None:
        def _jac(t, y):
            y_eval = np.maximum(y, 0.0) if clip_negative else y
            return np.asarray(jac_fn(t, y_eval), dtype=np.float64)

    kwargs = dict(method=method, rtol=rtol, atol=atol, jac=_jac)
    if t_eval is not None:
        kwargs["t_eval"] = np.asarray(t_eval, dtype=np.float64)
    if max_step is not None:
        kwargs["max_step"] = float(max_step)

    sol = solve_ivp(_rhs, t_span, y0_arr, **kwargs)
    if not sol.success:
        raise RuntimeError(f"solve_stiff_reaction_network: integration failed -- {sol.message}")

    return {"t": sol.t, "y": sol.y, "success": bool(sol.success)}
