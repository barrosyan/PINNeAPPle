"""Subdomain-chained X-TFC for extremely stiff ODE-IVP systems (target
stiffness ratio -- max|eigenvalue(Jacobian)| / min|eigenvalue(Jacobian)| --
roughly 1e10 to 1e14), plus a generic "extra scalar parameter" input.

Why subdomains at all
---------------------
`xtfc_ivp.py`'s single-domain X-TFC (see its module docstring) represents the
whole solution on [t0, tf] with ONE fixed-ELM basis over the WHOLE interval.
That works well when the solution's fastest and slowest timescales are within
a few orders of magnitude of each other. A genuinely stiff system -- e.g. a
fast-decaying reaction intermediate with rate constant k2 alongside a slow
accumulation with rate constant k1, k2/k1 ~ 1e10-1e14 -- has a transient that
is over EXTREMELY narrow near t=t0 (width ~ 1/k2) followed by SLOW dynamics
over the rest of [t0, tf] (width ~ 1/k1). No single fixed-width Chebyshev
collocation grid can resolve both scales at once without an impractically
large `n_collocation`: doubling the resolution near t0 (where you need it)
by using more collocation points spreads exactly as many points over the
SLOW part too, since Chebyshev-Gauss-Lobatto spacing is a smooth (not
scale-adaptive) function of index.

This module fixes that the same way classical stiff integrators effectively
do (variable step size), but for X-TFC's exact-least-squares construction
instead of a step-by-step method: chop [t0, tf] into LOGARITHMICALLY spaced
subdomains (tiny near t0, growing geometrically) and solve a fresh X-TFC
constrained expression on each subdomain independently, CHAINING them by
using subdomain k's own predicted final state as subdomain k+1's initial
condition (exact continuity by construction -- no separate continuity loss
term, the same way a single IVP's initial condition is satisfied exactly by
construction in `xtfc_ivp.py`, not via a penalty). Each subdomain then only
has to resolve whatever timescale is locally active in that decade, which a
modest fixed `n_basis`/`n_collocation` handles easily.

This module deliberately reuses `xtfc_ivp.py`'s ELM-basis machinery
(`_elm_basis`, `chebyshev_gauss_lobatto`, `_numerical_jacobian`) and its
Newton/Gauss-Newton per-domain solve loop almost verbatim -- the only new
numerical content here is (a) the log-spaced subdomain chaining and (b) the
extra-parameter input described below. See `xtfc_ivp.py`'s module docstring
for the underlying method and its references (Schiassi, Leake, De Florio,
Johnston, Furfaro & Mortari, arXiv:2005.10632; De Florio, Schiassi et al.,
arXiv:2008.05554).

Extra scalar parameter (`extra_param`)
---------------------------------------
A sibling project used this same X-TFC machinery to fit a chemistry system
parametrically in pH, by adding pH as a SECOND fixed random-feature input
dimension: phi_j(t, p) = activation(w_t_j*t + w_p_j*p + b_j). This module
generalizes that to any named `extra_param` (a single scalar, e.g. a rate
constant, an ambient temperature, or any other problem parameter that
`rhs_fn`/`jac_fn` close over). The key simplification that keeps this cheap:
within one `solve_xtfc_subdomains` call, `extra_param` is a FIXED number, so
w_p_j*p_norm + b_j collapses to a single effective per-feature bias
b_eff_j = b_j + w_p_j*p_norm -- i.e. the SAME 1D `_elm_basis` machinery from
`xtfc_ivp.py` applies unchanged, just with `b` replaced by `b_eff`. This is
an honest, useful simplification, not the fully general multivariate-TFC
collocation-over-a-(t,p)-grid that a true PARAMETRIC surrogate would need
(that would additionally require sampling collocation points across a RANGE
of `extra_param` values, as in the "Parametric Differential Equations" title
of arXiv:2005.10632's Section 4) -- but it captures the one part of that idea
that's cheap and broadly useful here: because `w_t`, `w_p`, `b` are sampled
ONCE (from `seed`) and never touch `extra_param`, the SAME random-feature
bank can be reused, unmodified, across a sweep of `extra_param` values (only
`beta` is refit per value, via the same fast linear/Gauss-Newton solve) --
so a parameter sweep does not require re-randomizing features each time.

Caveat: log-spaced subdomains help resolve a fast INITIAL transient (the
common stiff-kinetics shape: fast intermediate decay near t0, slow dynamics
after). A system whose fast dynamics occur somewhere in the MIDDLE of
[t0, tf] would need a different (not log-from-t0) subdomain placement; that
generalization is not attempted here.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry
from .xtfc_ivp import ACTIVATIONS, chebyshev_gauss_lobatto, _elm_basis, _numerical_jacobian, xtfc_predict

Array = np.ndarray


def log_subdomain_boundaries(t0: float, tf: float, n_subdomains: int,
                              first_width: "Optional[float]" = None) -> Array:
    """n_subdomains+1 boundaries [t0, ..., tf] with logarithmically GROWING
    widths -- tiny near t0 (to resolve a fast initial transient spanning many
    decades), geometrically larger towards tf.

    first_width: width of the first (smallest) subdomain. Defaults to
    (tf-t0)*1e-12, matching the docstring's target 1e10-1e14 stiffness ratio
    (the first subdomain should be a few multiples of the fastest timescale,
    1/k_fast ~ (tf-t0)/stiffness_ratio).
    """
    T = tf - t0
    if T <= 0:
        raise ValueError("tf must be > t0")
    if n_subdomains < 1:
        raise ValueError("n_subdomains must be >= 1")
    if first_width is None:
        first_width = T * 1e-12
    first_width = min(first_width, T * 0.5)
    if n_subdomains == 1:
        return np.array([t0, tf])
    interior = t0 + np.geomspace(first_width, T, n_subdomains)
    interior[-1] = tf  # exact endpoint (geomspace can be off by float error)
    return np.concatenate([[t0], interior])


def _solve_one_subdomain(
    rhs_fn: Callable[[Array, Array], Array],
    y0_arr: Array,
    t0: float,
    tf: float,
    w: Array,
    b: Array,
    activation: str,
    n_collocation: int,
    collocation: str,
    max_iter: int,
    tol: float,
    linear: bool,
    jac_fn: "Optional[Callable[[Array, Array], Array]]",
) -> dict:
    """One subdomain's X-TFC solve -- the same Newton/Gauss-Newton loop as
    `xtfc_ivp.solve_xtfc_ode`, generalized to accept a PRE-SAMPLED (w, b)
    pair (so the same random-feature bank persists across every subdomain in
    a chain, and across an `extra_param` sweep) instead of sampling its own
    from a seed.
    """
    T = tf - t0
    n_out = y0_arr.shape[0]
    n_basis = w.shape[0]

    if collocation == "chebyshev":
        x = chebyshev_gauss_lobatto(n_collocation, -1.0, 1.0)
    elif collocation == "uniform":
        x = np.linspace(-1.0, 1.0, n_collocation)
    else:
        raise ValueError(f"collocation={collocation!r} must be 'chebyshev' or 'uniform'")

    phi, dphi_dx = _elm_basis(x, w, b, activation)
    phi0, _ = _elm_basis(np.array([-1.0]), w, b, activation)
    psi = phi - phi0
    dpsi_dx = dphi_dx

    dxdt_factor = 2.0 / T
    t = t0 + (x + 1.0) * (T / 2.0)

    def y_of(beta: Array) -> Array:
        return psi @ beta + y0_arr[None, :]

    def dydt_of(beta: Array) -> Array:
        return dxdt_factor * (dpsi_dx @ beta)

    beta = np.zeros((n_basis, n_out))
    residual_history: "list[float]" = []
    n_iter_done = 0
    converged = False
    effective_max_iter = 1 if linear else max_iter

    for it in range(1, effective_max_iter + 1):
        y_k = y_of(beta)
        F_k = rhs_fn(t, y_k)
        dydt_k = dydt_of(beta)
        res_k = dydt_k - F_k
        res_norm = float(np.sqrt(np.mean(res_k ** 2)))
        residual_history.append(res_norm)
        n_iter_done = it
        if res_norm < tol:
            converged = True
            break

        J_k = jac_fn(t, y_k) if jac_fn is not None else _numerical_jacobian(rhs_fn, t, y_k)

        A = np.zeros((n_collocation * n_out, n_basis * n_out))
        rhs_vec = np.zeros(n_collocation * n_out)
        for i in range(n_out):
            row = slice(i * n_collocation, (i + 1) * n_collocation)
            col_i = slice(i * n_basis, (i + 1) * n_basis)
            A[row, col_i] += dxdt_factor * dpsi_dx
            for j in range(n_out):
                col_j = slice(j * n_basis, (j + 1) * n_basis)
                A[row, col_j] -= J_k[:, i, j][:, None] * psi
            rhs_vec[row] = F_k[:, i] - dydt_k[:, i]

        delta_vec, *_ = np.linalg.lstsq(A, rhs_vec, rcond=None)
        delta = delta_vec.reshape(n_out, n_basis).T
        beta_candidate = beta + delta

        if linear:
            beta = beta_candidate
            break

        res_new = dxdt_factor * (dpsi_dx @ beta_candidate) - rhs_fn(t, y_of(beta_candidate))
        res_new_norm = float(np.sqrt(np.mean(res_new ** 2)))
        if res_new_norm > res_norm and it > 1:
            break
        beta = beta_candidate
        if res_new_norm < tol:
            residual_history.append(res_new_norm)
            n_iter_done = it + 1
            converged = True
            break

    y_final = y_of(beta)[-1]  # x=+1 == t=tf (both Chebyshev-Gauss-Lobatto and
                              # uniform grids include the +1 endpoint exactly)

    def predict(t_query) -> Array:
        return xtfc_predict(np.asarray(t_query, dtype=np.float64), w, b, beta, y0_arr, activation, t0, tf)

    return {
        "beta": beta, "y0": y0_arr, "t0": t0, "tf": tf, "activation": activation,
        "residual_history": residual_history, "n_iter": n_iter_done, "converged": converged,
        "y_final": y_final, "predict": predict,
    }


def solve_xtfc_subdomains(
    rhs_fn: Callable[[Array, Array], Array],
    y0,
    t_span: "tuple[float, float]",
    n_subdomains: int = 12,
    first_width: "Optional[float]" = None,
    spacing: str = "log",
    n_basis: int = 30,
    n_collocation: int = 40,
    activation: str = "tanh",
    max_iter: int = 30,
    tol: float = 1e-10,
    seed: int = 0,
    w_range: "tuple[float, float]" = (-3.0, 3.0),
    b_range: "tuple[float, float]" = (-3.0, 3.0),
    w_param_range: "tuple[float, float]" = (-3.0, 3.0),
    collocation: str = "chebyshev",
    linear: bool = False,
    jac_fn: "Optional[Callable[[Array, Array], Array]]" = None,
    extra_param: float = 0.0,
    param_range: "tuple[float, float]" = (-1.0, 1.0),
) -> dict:
    """Solve a (possibly extremely stiff) dy_i/dt = rhs_fn(t, y)[:, i],
    y_i(t0) = y0[i] IVP over log-spaced subdomains of t_span, chaining
    continuity across subdomain boundaries (see module docstring).

    rhs_fn, jac_fn: same contract as `xtfc_ivp.solve_xtfc_ode` (jac_fn is
        strongly preferred for a genuinely stiff system -- the numerical
        fallback's finite-difference step can itself be numerically delicate
        across many stiffness decades).
    y0, t_span: as in `solve_xtfc_ode`.
    n_subdomains: number of subdomains to chain.
    first_width: width of the smallest (first) subdomain; see
        `log_subdomain_boundaries`. Defaults to (tf-t0)*1e-12.
    spacing: "log" (default, see module docstring) or "uniform" (plain
        `np.linspace` subdomain boundaries -- offered mainly as a baseline to
        demonstrate log-spacing's advantage for a stiff system, not because
        it is recommended for one).
    n_basis, n_collocation, activation, max_iter, tol, w_range, b_range,
        collocation, linear: per-subdomain X-TFC settings, same meaning as
        in `solve_xtfc_ode` (applied identically to every subdomain).
    seed: RNG seed for the ONE fixed (w, w_param, b) draw shared by every
        subdomain in the chain (and, if swept, by every `extra_param` value).
    w_param_range: sampling range for the fixed extra-parameter input
        weights (see module docstring's `extra_param` section).
    extra_param: a single scalar parameter, normalized via `param_range` into
        [-1, 1] the same way t is mapped into x in `xtfc_ivp.py`, then folded
        into a per-feature effective bias (b_eff = b + w_param*p_norm) shared
        by every subdomain.
    param_range: (lo, hi) expected range of `extra_param` across an eventual
        sweep, used only for the [-1,1] normalization above.

    Returns a dict: {boundaries (n_subdomains+1,), subdomains (list of each
    subdomain's solve dict, see `_solve_one_subdomain`), w, w_param, b,
    b_eff, extra_param, predict (callable t_query -> y_query, dispatching
    each query time to its owning subdomain)}.
    """
    t0, tf = t_span
    if tf <= t0:
        raise ValueError("t_span must have tf > t0")
    y0_arr = np.atleast_1d(np.asarray(y0, dtype=np.float64))
    n_out = y0_arr.shape[0]

    if spacing == "log":
        boundaries = log_subdomain_boundaries(t0, tf, n_subdomains, first_width)
    elif spacing == "uniform":
        boundaries = np.linspace(t0, tf, n_subdomains + 1)
    else:
        raise ValueError(f"spacing={spacing!r} must be 'log' or 'uniform'")

    rng = np.random.default_rng(seed)
    w = rng.uniform(w_range[0], w_range[1], size=n_basis)
    b = rng.uniform(b_range[0], b_range[1], size=n_basis)
    w_param = rng.uniform(w_param_range[0], w_param_range[1], size=n_basis)

    lo, hi = param_range
    p_norm = -1.0 + 2.0 * (extra_param - lo) / (hi - lo) if hi > lo else 0.0
    b_eff = b + w_param * p_norm

    y_current = y0_arr
    subdomains = []
    for k in range(n_subdomains):
        ta, tb = float(boundaries[k]), float(boundaries[k + 1])
        sub = _solve_one_subdomain(
            rhs_fn, y_current, ta, tb, w, b_eff, activation,
            n_collocation, collocation, max_iter, tol, linear, jac_fn,
        )
        subdomains.append(sub)
        y_current = sub["y_final"]

    def predict(t_query) -> Array:
        tq = np.atleast_1d(np.asarray(t_query, dtype=np.float64))
        out = np.empty((tq.shape[0], n_out))
        for i, tval in enumerate(tq):
            k = int(np.clip(np.searchsorted(boundaries, tval, side="right") - 1, 0, n_subdomains - 1))
            out[i] = subdomains[k]["predict"](np.array([tval]))[0]
        return out

    return {
        "boundaries": boundaries, "subdomains": subdomains,
        "w": w, "w_param": w_param, "b": b, "b_eff": b_eff,
        "extra_param": extra_param, "param_range": param_range,
        "n_subdomains": n_subdomains, "activation": activation,
        "converged": all(s["converged"] for s in subdomains),
        "predict": predict,
    }


@SolverRegistry.register(
    name="xtfc_subdomain",
    family="ode",
    description="Subdomain-chained X-TFC for extremely stiff ODE-IVP systems (log-spaced "
                "time subdomains, exact continuity chaining) with a generic extra_param input.",
    tags=["xtfc", "ivp", "elm", "zero-backprop", "stiff", "subdomain"],
)
class XTFCSubdomainSolver(SolverBase):
    """Thin `SolverBase`/registry wrapper around `solve_xtfc_subdomains`. The
    functional API (`solve_xtfc_subdomains`, `log_subdomain_boundaries`) is
    the primary entry point.
    """

    def __init__(self, n_subdomains: int = 12, first_width: "Optional[float]" = None,
                 spacing: str = "log", n_basis: int = 30, n_collocation: int = 40,
                 activation: str = "tanh", max_iter: int = 30, tol: float = 1e-10,
                 seed: int = 0, collocation: str = "chebyshev", linear: bool = False,
                 extra_param: float = 0.0, param_range: "tuple[float, float]" = (-1.0, 1.0)):
        super().__init__()
        self.n_subdomains = int(n_subdomains)
        self.first_width = first_width
        self.spacing = spacing
        self.n_basis = int(n_basis)
        self.n_collocation = int(n_collocation)
        self.activation = activation
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.collocation = collocation
        self.linear = bool(linear)
        self.extra_param = float(extra_param)
        self.param_range = param_range

    def forward(
        self,
        rhs_fn: Callable[[Array, Array], Array],
        y0,
        t_span: "tuple[float, float]",
        jac_fn: "Optional[Callable[[Array, Array], Array]]" = None,
        n_query: int = 200,
    ) -> SolverOutput:
        sol = solve_xtfc_subdomains(
            rhs_fn, y0, t_span,
            n_subdomains=self.n_subdomains, first_width=self.first_width, spacing=self.spacing,
            n_basis=self.n_basis, n_collocation=self.n_collocation, activation=self.activation,
            max_iter=self.max_iter, tol=self.tol, seed=self.seed, collocation=self.collocation,
            linear=self.linear, jac_fn=jac_fn, extra_param=self.extra_param, param_range=self.param_range,
        )
        # A plain linear query grid is fine here -- `predict` itself already
        # dispatches each query time to the correct (log-spaced) subdomain,
        # so this grid's own spacing doesn't need to match the subdomain
        # boundaries for correctness, only for how evenly the fast initial
        # transient gets sampled in the returned array.
        t_query = np.linspace(t_span[0], t_span[1], n_query)
        y_query = sol["predict"](t_query)
        return SolverOutput(
            result=torch.from_numpy(y_query.astype(np.float32)),
            losses={"residual": torch.tensor(
                sol["subdomains"][-1]["residual_history"][-1] if sol["subdomains"][-1]["residual_history"] else 0.0
            )},
            extras={
                "t_query": t_query.astype(np.float32),
                "boundaries": sol["boundaries"],
                "converged": sol["converged"],
                "method": "xtfc_subdomain",
            },
        )
