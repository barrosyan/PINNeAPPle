"""X-TFC (eXtreme Theory of Functional Connections) solver for 1D-space(+time)
PDEs posed on a rectangle x in [x0, x1], t in [t0, tf], with one initial
condition (at t=t0) and two Dirichlet boundary conditions (at x=x0, x=x1):

    u_t(x,t) = alpha * u_xx(x,t)                         ("heat_1d", linear)
    u_t(x,t) + u(x,t)*u_x(x,t) = nu * u_xx(x,t)           ("burgers_1d", nonlinear)

    u(x, t0) = u0(x),   u(x0, t) = uL(t),   u(x1, t) = uR(t)

Deliberately scoped to exactly these two named problems (a linear diffusion
case and one genuinely nonlinear case) rather than a fully generic PDE
front-end -- see module docstring of `xtfc_ivp.py` for the general X-TFC
method (fixed-ELM basis + Newton/least-squares, no gradient descent); this
module extends that SAME idea from a single time axis to a 2D (x,t) rectangle
via a multivariate ("Coons patch") constrained expression.

References
----------
- Schiassi, Leake, De Florio, Johnston, Furfaro & Mortari, "Extreme Theory of
  Functional Connections: A Physics-Informed Neural Network Method for
  Solving Parametric Differential Equations" (arXiv:2005.10632; Neurocomputing
  457 (2021) 334-356) -- Sections 2 and 4 extend the single-axis constrained
  expression used in `xtfc_ivp.py` to PDEs with a bivariate (x,t) free
  function; this module follows that extension.
- Leake & Mortari, "Deep Theory of Functional Connections: A New Method for
  Estimating the Solutions of Partial Differential Equations" (arXiv:
  2005.01219) -- the general multivariate/"Coons patch" transfinite-
  interpolation constrained-expression construction for a rectangle with
  conditions on 3 (not necessarily 4) of its edges, which is exactly the
  IC + 2 BC configuration here (t=tf is left free -- there is no terminal
  condition).

Method summary
--------------
1. Domain: the raw (x,t) rectangle is used directly (unlike `xtfc_ivp.py`,
   which remaps t into [-1,1] -- here the free function's random ELM
   features already absorb the domain scale via their random weights, so no
   remapping is needed; `w_range`/`b_range` should be set with the actual
   physical extent of x_span/t_span in mind, exactly as `xtfc_ivp.py`'s
   `w_range`/`b_range` are tuned for the already-remapped [-1,1] axis).
2. Free function: a bivariate fixed-ELM basis phi_j(x,t) = activation(wx_j*x
   + wt_j*t + b_j), j=1..M, with wx, wt, b sampled once and frozen (same
   "extreme learning machine" idea as `xtfc_ivp.py`, now with 2 fixed random
   input weights per feature instead of 1).
3. Constrained expression (Coons-patch / transfinite interpolation over a
   rectangle with 3 constrained edges -- Leake & Mortari arXiv:2005.01219):

       u(x,t) = g(x,t) + [u0(x) - g(x,t0)]
                       + s1(x)*[uL(t) - g(x0,t)] + s2(x)*[uR(t) - g(x1,t)]
                       - s1(x)*[uL(t0) - g(x0,t0)] - s2(x)*[uR(t0) - g(x1,t0)]

   where g(x,t) = phi(x,t)^T beta is the free function, and s1(x) = (x1-x)/
   (x1-x0), s2(x) = (x-x0)/(x1-x0) are the linear "switching functions" that
   blend the two boundary corrections (s1(x0)=1,s1(x1)=0; s2(x0)=0,s2(x1)=1).
   The last two ("corner correction") terms exist so the IC correction and
   the two BC corrections don't double-count each other's contribution at
   the two corners (x0,t0) and (x1,t0) -- the defining feature of a Coons
   patch. Because g is affine in beta, so is u:

       u(x,t) = psi(x,t)^T beta + f(x,t)

   with psi(x,t) = phi(x,t) - phi(x,t0) - s1(x)*Delta_L(t) - s2(x)*Delta_R(t),
   Delta_L(t) = phi(x0,t) - phi(x0,t0), Delta_R(t) = phi(x1,t) - phi(x1,t0),
   and f(x,t) = u0(x) + s1(x)*(uL(t)-uL(t0)) + s2(x)*(uR(t)-uR(t0)) collecting
   every term independent of beta. One can check directly that psi(x,t0) = 0
   and psi(x0,t) = psi(x1,t) = 0 identically (any beta), so:
       u(x,t0) = f(x,t0) = u0(x)                                    exactly
       u(x0,t) = f(x0,t) = u0(x0) + uL(t) - uL(t0)  =  uL(t)   iff u0(x0)=uL(t0)
       u(x1,t) = f(x1,t) = u0(x1) + uR(t) - uR(t0)  =  uR(t)   iff u0(x1)=uR(t0)
   i.e. the IC is satisfied identically for ANY beta, and the two BCs are
   satisfied identically for any beta PROVIDED the problem's own IC/BC data
   is corner-consistent (u0(x0)=uL(t0), u0(x1)=uR(t0)) -- a property of the
   physical problem, not of the fit. Both `heat_1d` and `burgers_1d` presets
   below satisfy this by construction; `solve_xtfc_pde` asserts it defensively
   on every call so a future preset that violates it fails loudly.
   Because s1, s2 are LINEAR in x, s1''=s2''=0, so the constrained expression
   contributes no boundary-correction term to u_xx beyond phi's own second
   x-derivative -- psi_xx(x,t) = phi_xx(x,t) - phi_xx(x,t0) and f_xx(x,t) =
   u0''(x), both independent of the switching functions.
4. Collocation: an (n_collocation_x x n_collocation_t) tensor grid of
   Chebyshev-Gauss-Lobatto points (default) or a uniform grid, reusing
   `chebyshev_gauss_lobatto` from `xtfc_ivp.py`. No separate IC/BC
   collocation points are needed -- both are satisfied by construction
   everywhere, including at t=t0, x=x0, x=x1 collocation nodes.
5. `heat_1d` (linear in u): the PDE residual u_t - alpha*u_xx is affine in
   beta at every collocation point -> ONE `numpy.linalg.lstsq` solve.
6. `burgers_1d` (nonlinear via the u*u_x advection term): Gauss-Newton
   iteration, following `xtfc_ivp.py`'s nonlinear branch (module docstring
   point 6 there) almost exactly -- but since Burgers' only nonlinearity is
   the single product term u*u_x, its Frechet derivative is known in closed
   form (product rule: d(u*u_x) ~= u_x_k*du + u_k*d(u_x) at the current
   iterate), so this module uses that EXACT analytic linearization rather
   than `xtfc_ivp.py`'s general finite-difference Jacobian fallback -- an
   exact-Jacobian specialization of the same scheme, avoiding unnecessary
   finite-difference truncation error for a nonlinearity that is known in
   closed form. Stopping rule (RMS residual < tol / max_iter / divergence
   guard) mirrors `xtfc_ivp.py`'s exactly.

Caveat: like `xtfc_ivp.py`, this is a forward-solve tool for a small, FIXED
catalog of two named PDEs (`heat_1d`, `burgers_1d`) -- it does not attempt to
parse or accept an arbitrary user PDE, unlike `xtfc_ivp.py`'s fully generic
`rhs_fn`. Extending to a third preset means adding IC/BC closed forms (with
their exact first/second derivatives) and, for a nonlinear PDE, its exact
Frechet linearization -- both problem-specific by design.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry
from .xtfc_ivp import chebyshev_gauss_lobatto

Array = np.ndarray


def _tanh_act(z: Array) -> Array:
    return np.tanh(z)


def _tanh_dact(z: Array) -> Array:
    t = np.tanh(z)
    return 1.0 - t * t


def _tanh_d2act(z: Array) -> Array:
    t = np.tanh(z)
    return -2.0 * t * (1.0 - t * t)


def _sigmoid_act(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_dact(z: Array) -> Array:
    s = _sigmoid_act(z)
    return s * (1.0 - s)


def _sigmoid_d2act(z: Array) -> Array:
    s = _sigmoid_act(z)
    d = s * (1.0 - s)
    return d * (1.0 - 2.0 * s)


def _sin_act(z: Array) -> Array:
    return np.sin(z)


def _sin_dact(z: Array) -> Array:
    return np.cos(z)


def _sin_d2act(z: Array) -> Array:
    return -np.sin(z)


# activation name -> (f, f', f''), all elementwise. Distinct from xtfc_ivp's
# ACTIVATIONS dict because this module additionally needs the second
# derivative (for u_xx) that the plain IVP solver never requires.
ACTIVATIONS2: Dict[str, "tuple"] = {
    "tanh": (_tanh_act, _tanh_dact, _tanh_d2act),
    "sigmoid": (_sigmoid_act, _sigmoid_dact, _sigmoid_d2act),
    "sin": (_sin_act, _sin_dact, _sin_d2act),
}


def _elm_basis2d(x: Array, t: Array, wx: Array, wt: Array, b: Array,
                  activation: str) -> "tuple[Array, Array, Array, Array]":
    """phi, dphi/dx, dphi/dt, d2phi/dx2 for M fixed bivariate random features,
    evaluated at N paired points (x[p], t[p]).

    x, t: (N,), same length (paired collocation/query points -- NOT a
    meshgrid; callers wanting a grid must flatten it themselves first, as
    `solve_xtfc_pde` does).
    wx, wt, b: (M,) fixed random weights/bias, sampled once.
    Returns 4 arrays, each (N, M).
    """
    act, dact, d2act = ACTIVATIONS2[activation]
    z = np.outer(x, wx) + np.outer(t, wt) + b[None, :]     # (N, M)
    phi = act(z)
    dphi_dx = dact(z) * wx[None, :]
    dphi_dt = dact(z) * wt[None, :]
    d2phi_dx2 = d2act(z) * (wx ** 2)[None, :]
    return phi, dphi_dx, dphi_dt, d2phi_dx2


def _zero_fn(x) -> Array:
    return np.zeros_like(np.asarray(x, dtype=np.float64))


def _heat_ic(x: Array) -> Array:
    return np.sin(np.pi * x)


def _heat_ic_prime(x: Array) -> Array:
    return np.pi * np.cos(np.pi * x)


def _heat_ic_dprime(x: Array) -> Array:
    return -(np.pi ** 2) * np.sin(np.pi * x)


def _burgers_ic(x: Array) -> Array:
    return -np.sin(np.pi * x)


def _burgers_ic_prime(x: Array) -> Array:
    return -np.pi * np.cos(np.pi * x)


def _burgers_ic_dprime(x: Array) -> Array:
    return (np.pi ** 2) * np.sin(np.pi * x)


# Named PDE presets: IC/BC closed forms (with exact 1st/2nd derivatives) plus
# the domain/parameter defaults matching each problem's standard benchmark
# form. `heat_1d` has a known closed-form solution (exp(-alpha*pi^2*t)*
# sin(pi*x)) used as an independent test reference; `burgers_1d` is the
# standard Raissi et al. benchmark (also used, with the same nu, IC and BC,
# by `pinneapple_tools.benchmark_suite.tasks.burgers_1d.Burgers1DTask`, whose
# method-of-lines/RK4 reference this module's tests validate against).
_PDE_PRESETS: Dict[str, Dict[str, Any]] = {
    "heat_1d": dict(
        linear=True,
        ic=_heat_ic, ic_prime=_heat_ic_prime, ic_dprime=_heat_ic_dprime,
        bc_left=_zero_fn, bc_left_prime=_zero_fn,
        bc_right=_zero_fn, bc_right_prime=_zero_fn,
        default_x_span=(0.0, 1.0), default_t_span=(0.0, 1.0),
        default_param=1.0,  # alpha (diffusivity)
    ),
    "burgers_1d": dict(
        linear=False,
        ic=_burgers_ic, ic_prime=_burgers_ic_prime, ic_dprime=_burgers_ic_dprime,
        bc_left=_zero_fn, bc_left_prime=_zero_fn,
        bc_right=_zero_fn, bc_right_prime=_zero_fn,
        default_x_span=(-1.0, 1.0), default_t_span=(0.0, 1.0),
        default_param=0.01 / np.pi,  # nu (viscosity)
    ),
}


def _assert_corner_consistent(preset: Dict[str, Any], x0: float, x1: float, t0: float) -> None:
    u0_x0 = float(preset["ic"](np.array([x0]))[0])
    u0_x1 = float(preset["ic"](np.array([x1]))[0])
    uL_t0 = float(preset["bc_left"](np.array([t0]))[0])
    uR_t0 = float(preset["bc_right"](np.array([t0]))[0])
    if not np.isclose(u0_x0, uL_t0, atol=1e-8):
        raise AssertionError(
            f"IC/left-BC corner inconsistency: u0(x0)={u0_x0} != uL(t0)={uL_t0} "
            "-- the Coons-patch constrained expression only satisfies the "
            "left BC identically when this holds."
        )
    if not np.isclose(u0_x1, uR_t0, atol=1e-8):
        raise AssertionError(
            f"IC/right-BC corner inconsistency: u0(x1)={u0_x1} != uR(t0)={uR_t0} "
            "-- the Coons-patch constrained expression only satisfies the "
            "right BC identically when this holds."
        )


def xtfc_pde_predict(x_query: Array, t_query: Array, wx: Array, wt: Array, b: Array,
                      beta: Array, pde: str, activation: str,
                      x_span: "tuple[float, float]", t_span: "tuple[float, float]") -> Array:
    """Evaluate the trained 2D TFC constrained expression at arbitrary
    (paired) query points -- a pure function of the persisted (wx, wt, b,
    beta, pde, activation, x_span, t_span), mirroring `xtfc_predict` in
    `xtfc_ivp.py`.

    x_query, t_query: (N,), paired (NOT a meshgrid).
    Returns u_query, shape (N,).
    """
    preset = _PDE_PRESETS[pde]
    x0, x1 = x_span
    t0, _tf = t_span
    xp = np.asarray(x_query, dtype=np.float64)
    tp = np.asarray(t_query, dtype=np.float64)

    sp1 = -1.0 / (x1 - x0)
    sp2 = 1.0 / (x1 - x0)
    s1 = (x1 - xp) / (x1 - x0)
    s2 = (xp - x0) / (x1 - x0)

    phi_pt, _, _, _ = _elm_basis2d(xp, tp, wx, wt, b, activation)
    t0_arr = np.full_like(xp, t0)
    phi_pt0, _, _, _ = _elm_basis2d(xp, t0_arr, wx, wt, b, activation)
    x0_arr = np.full_like(tp, x0)
    x1_arr = np.full_like(tp, x1)
    phi_x0t, _, _, _ = _elm_basis2d(x0_arr, tp, wx, wt, b, activation)
    phi_x1t, _, _, _ = _elm_basis2d(x1_arr, tp, wx, wt, b, activation)
    phi_x0t0, _, _, _ = _elm_basis2d(np.array([x0]), np.array([t0]), wx, wt, b, activation)
    phi_x1t0, _, _, _ = _elm_basis2d(np.array([x1]), np.array([t0]), wx, wt, b, activation)

    Delta_L = phi_x0t - phi_x0t0
    Delta_R = phi_x1t - phi_x1t0
    psi = phi_pt - phi_pt0 - s1[:, None] * Delta_L - s2[:, None] * Delta_R

    uL = preset["bc_left"](tp)
    uR = preset["bc_right"](tp)
    uL_t0 = float(preset["bc_left"](np.array([t0]))[0])
    uR_t0 = float(preset["bc_right"](np.array([t0]))[0])
    f = preset["ic"](xp) + s1 * (uL - uL_t0) + s2 * (uR - uR_t0)

    return psi @ beta + f


def solve_xtfc_pde(
    pde: str,
    x_span: "Optional[tuple[float, float]]" = None,
    t_span: "Optional[tuple[float, float]]" = None,
    n_basis: int = 60,
    n_collocation_x: int = 25,
    n_collocation_t: int = 25,
    activation: str = "tanh",
    max_iter: int = 60,
    tol: float = 1e-9,
    seed: int = 0,
    w_range: "tuple[float, float]" = (-3.0, 3.0),
    b_range: "tuple[float, float]" = (-3.0, 3.0),
    collocation: str = "chebyshev",
    param: "Optional[float]" = None,
) -> dict:
    """Solve `pde` in {"heat_1d", "burgers_1d"} via X-TFC (see module
    docstring for the full method).

    pde: "heat_1d" (u_t = param*u_xx, linear) or "burgers_1d" (u_t + u*u_x =
        param*u_xx, nonlinear). Each preset supplies its own standard IC/BC
        closed forms (see `_PDE_PRESETS`).
    x_span, t_span: domain rectangle; default to each preset's standard
        benchmark domain when omitted (heat_1d: x in [0,1], t in [0,1];
        burgers_1d: x in [-1,1], t in [0,1], the Raissi et al. benchmark).
    n_basis: M, number of fixed bivariate random ELM features.
    n_collocation_x, n_collocation_t: collocation grid resolution along each
        axis (total collocation points = the product).
    activation: one of ACTIVATIONS2's keys ("tanh", "sigmoid", "sin").
    max_iter, tol: outer Gauss-Newton iteration cap / RMS-residual stopping
        tolerance (burgers_1d only -- heat_1d is a single exact linear solve).
    seed: RNG seed for the fixed (wx, wt, b) draw.
    w_range, b_range: (lo, hi) uniform sampling range for the fixed ELM input
        weights/bias, applied to both wx and wt. Because the domain here is
        NOT remapped to [-1,1] (see module docstring point 1), the effective
        argument to each activation spans roughly w_range * (x1-x0 or tf-t0)
        -- widen/narrow relative to `xtfc_ivp.py`'s defaults if the domain is
        much larger/smaller than O(1).
    collocation: "chebyshev" (default, Gauss-Lobatto tensor grid) or
        "uniform".
    param: alpha (heat_1d) or nu (burgers_1d); defaults to the preset's
        standard benchmark value when omitted.

    Returns a dict: {wx, wt, b, beta, pde, activation, x_span, t_span, param,
    n_basis, n_collocation_x, n_collocation_t, collocation, x_colloc, t_colloc,
    u_colloc, residual_history, n_iter, converged, predict (callable
    (x_query, t_query) -> u_query)}.
    """
    if pde not in _PDE_PRESETS:
        raise ValueError(f"pde={pde!r} not in {list(_PDE_PRESETS)}")
    if activation not in ACTIVATIONS2:
        raise ValueError(f"activation={activation!r} not in {list(ACTIVATIONS2)}")
    preset = _PDE_PRESETS[pde]

    x_span = x_span or preset["default_x_span"]
    t_span = t_span or preset["default_t_span"]
    x0, x1 = x_span
    t0, tf = t_span
    if x1 <= x0 or tf <= t0:
        raise ValueError("x_span and t_span must each have hi > lo")
    param = preset["default_param"] if param is None else float(param)

    _assert_corner_consistent(preset, x0, x1, t0)

    rng = np.random.default_rng(seed)
    wx = rng.uniform(w_range[0], w_range[1], size=n_basis)
    wt = rng.uniform(w_range[0], w_range[1], size=n_basis)
    b = rng.uniform(b_range[0], b_range[1], size=n_basis)

    if collocation == "chebyshev":
        xs = chebyshev_gauss_lobatto(n_collocation_x, x0, x1)
        ts = chebyshev_gauss_lobatto(n_collocation_t, t0, tf)
    elif collocation == "uniform":
        xs = np.linspace(x0, x1, n_collocation_x)
        ts = np.linspace(t0, tf, n_collocation_t)
    else:
        raise ValueError(f"collocation={collocation!r} must be 'chebyshev' or 'uniform'")
    Xg, Tg = np.meshgrid(xs, ts, indexing="ij")
    xp = Xg.ravel()
    tp = Tg.ravel()
    n_pts = xp.shape[0]

    sp1 = -1.0 / (x1 - x0)
    sp2 = 1.0 / (x1 - x0)
    s1 = (x1 - xp) / (x1 - x0)
    s2 = (xp - x0) / (x1 - x0)

    phi_pt, dphi_dx_pt, dphi_dt_pt, d2phi_dx2_pt = _elm_basis2d(xp, tp, wx, wt, b, activation)
    t0_arr = np.full_like(xp, t0)
    phi_pt0, dphi_dx_pt0, _dphi_dt_pt0, d2phi_dx2_pt0 = _elm_basis2d(xp, t0_arr, wx, wt, b, activation)
    x0_arr = np.full_like(tp, x0)
    x1_arr = np.full_like(tp, x1)
    phi_x0t, _dx0, dphi_dt_x0t, _dxx0 = _elm_basis2d(x0_arr, tp, wx, wt, b, activation)
    phi_x1t, _dx1, dphi_dt_x1t, _dxx1 = _elm_basis2d(x1_arr, tp, wx, wt, b, activation)
    phi_x0t0, _, _, _ = _elm_basis2d(np.array([x0]), np.array([t0]), wx, wt, b, activation)
    phi_x1t0, _, _, _ = _elm_basis2d(np.array([x1]), np.array([t0]), wx, wt, b, activation)

    Delta_L = phi_x0t - phi_x0t0            # (N, M)
    Delta_R = phi_x1t - phi_x1t0
    dDelta_L_dt = dphi_dt_x0t               # phi_x0t0 is constant -> its t-derivative is 0
    dDelta_R_dt = dphi_dt_x1t

    psi = phi_pt - phi_pt0 - s1[:, None] * Delta_L - s2[:, None] * Delta_R
    psi_t = dphi_dt_pt - s1[:, None] * dDelta_L_dt - s2[:, None] * dDelta_R_dt
    psi_x = dphi_dx_pt - dphi_dx_pt0 - sp1 * Delta_L - sp2 * Delta_R
    psi_xx = d2phi_dx2_pt - d2phi_dx2_pt0

    u0 = preset["ic"](xp)
    u0p = preset["ic_prime"](xp)
    u0pp = preset["ic_dprime"](xp)
    uL = preset["bc_left"](tp)
    uLp = preset["bc_left_prime"](tp)
    uL_t0 = float(preset["bc_left"](np.array([t0]))[0])
    uR = preset["bc_right"](tp)
    uRp = preset["bc_right_prime"](tp)
    uR_t0 = float(preset["bc_right"](np.array([t0]))[0])

    f = u0 + s1 * (uL - uL_t0) + s2 * (uR - uR_t0)
    f_t = s1 * uLp + s2 * uRp
    f_x = u0p + sp1 * (uL - uL_t0) + sp2 * (uR - uR_t0)
    f_xx = u0pp

    # Sanity-check the module's own IC/BC identity promises (see docstring
    # point 3): psi(x,t0) == 0 and psi(x0,t) == psi(x1,t) == 0 IDENTICALLY,
    # for ANY beta (psi itself does not depend on beta) -- so u(x,t0) == f(x,t0)
    # == u0(x) and u(x0,t)/u(x1,t) reduce to the BC data exactly. Verified here
    # at the collocation grid's own t=t0 / x=x0 / x=x1 rows (both the
    # Chebyshev-Gauss-Lobatto and uniform grids include both endpoints) so a
    # future refactor that breaks the constrained-expression algebra fails
    # loudly instead of silently producing a slightly-wrong-at-the-boundary fit.
    _t0_rows = np.isclose(tp, t0)
    _x0_rows = np.isclose(xp, x0)
    _x1_rows = np.isclose(xp, x1)
    assert np.allclose(psi[_t0_rows], 0.0, atol=1e-8), "psi(x,t0) != 0: IC identity broken"
    assert np.allclose(psi[_x0_rows], 0.0, atol=1e-8), "psi(x0,t) != 0: left-BC identity broken"
    assert np.allclose(psi[_x1_rows], 0.0, atol=1e-8), "psi(x1,t) != 0: right-BC identity broken"

    def u_of(beta: Array) -> Array:
        return psi @ beta + f

    def ux_of(beta: Array) -> Array:
        return psi_x @ beta + f_x

    def ut_of(beta: Array) -> Array:
        return psi_t @ beta + f_t

    def uxx_of(beta: Array) -> Array:
        return psi_xx @ beta + f_xx

    beta = np.zeros(n_basis)
    residual_history: "list[float]" = []
    n_iter_done = 0
    converged = False

    if preset["linear"]:
        # heat_1d: u_t - param*u_xx affine in beta -> one exact lstsq solve.
        A = psi_t - param * psi_xx
        rhs_vec = -(f_t - param * f_xx)
        beta, *_ = np.linalg.lstsq(A, rhs_vec, rcond=None)
        res = ut_of(beta) - param * uxx_of(beta)
        residual_history = [float(np.sqrt(np.mean(res ** 2)))]
        n_iter_done = 1
        converged = True
    else:
        # burgers_1d: Gauss-Newton on the exact analytic linearization of
        # u*u_x (see module docstring point 6).
        for it in range(1, max_iter + 1):
            u_k = u_of(beta)
            ux_k = ux_of(beta)
            ut_k = ut_of(beta)
            uxx_k = uxx_of(beta)
            res_k = ut_k + u_k * ux_k - param * uxx_k
            res_norm = float(np.sqrt(np.mean(res_k ** 2)))
            residual_history.append(res_norm)
            n_iter_done = it
            if res_norm < tol:
                converged = True
                break

            A = psi_t + ux_k[:, None] * psi + u_k[:, None] * psi_x - param * psi_xx
            delta, *_ = np.linalg.lstsq(A, -res_k, rcond=None)
            beta_candidate = beta + delta

            u_c = u_of(beta_candidate)
            res_new = (ut_of(beta_candidate) + u_c * ux_of(beta_candidate)
                       - param * uxx_of(beta_candidate))
            res_new_norm = float(np.sqrt(np.mean(res_new ** 2)))
            if res_new_norm > res_norm and it > 1:
                break
            beta = beta_candidate
            if res_new_norm < tol:
                residual_history.append(res_new_norm)
                n_iter_done = it + 1
                converged = True
                break

    u_colloc = u_of(beta)

    def predict(x_query, t_query) -> Array:
        return xtfc_pde_predict(np.asarray(x_query, dtype=np.float64),
                                 np.asarray(t_query, dtype=np.float64),
                                 wx, wt, b, beta, pde, activation, x_span, t_span)

    return {
        "wx": wx, "wt": wt, "b": b, "beta": beta,
        "pde": pde, "activation": activation, "x_span": x_span, "t_span": t_span, "param": param,
        "n_basis": n_basis, "n_collocation_x": n_collocation_x, "n_collocation_t": n_collocation_t,
        "collocation": collocation,
        "x_colloc": xp, "t_colloc": tp, "u_colloc": u_colloc,
        "residual_history": residual_history, "n_iter": n_iter_done, "converged": converged,
        "predict": predict,
    }


@SolverRegistry.register(
    name="xtfc_pde",
    family="pde",
    description="X-TFC (Extreme Theory of Functional Connections) zero-backprop solver for "
                "heat_1d/burgers_1d on a rectangle -- bivariate fixed ELM basis + Coons-patch "
                "constrained expression + Newton/least-squares.",
    tags=["xtfc", "pde", "elm", "zero-backprop", "coons-patch"],
)
class XTFCPDESolver(SolverBase):
    """Thin `SolverBase`/registry wrapper around `solve_xtfc_pde` for callers
    that want the uniform `SolverOutput` interface. The functional API
    (`solve_xtfc_pde`, `xtfc_pde_predict`) is the primary entry point.
    """

    def __init__(self, pde: str = "burgers_1d", n_basis: int = 60,
                 n_collocation_x: int = 25, n_collocation_t: int = 25,
                 activation: str = "tanh", max_iter: int = 60, tol: float = 1e-9,
                 seed: int = 0, collocation: str = "chebyshev"):
        super().__init__()
        self.pde = pde
        self.n_basis = int(n_basis)
        self.n_collocation_x = int(n_collocation_x)
        self.n_collocation_t = int(n_collocation_t)
        self.activation = activation
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.collocation = collocation

    def forward(
        self,
        x_span: "Optional[tuple[float, float]]" = None,
        t_span: "Optional[tuple[float, float]]" = None,
        param: "Optional[float]" = None,
        n_query: int = 40,
    ) -> SolverOutput:
        sol = solve_xtfc_pde(
            self.pde, x_span=x_span, t_span=t_span,
            n_basis=self.n_basis, n_collocation_x=self.n_collocation_x,
            n_collocation_t=self.n_collocation_t, activation=self.activation,
            max_iter=self.max_iter, tol=self.tol, seed=self.seed,
            collocation=self.collocation, param=param,
        )
        xs = np.linspace(*sol["x_span"], n_query)
        ts = np.linspace(*sol["t_span"], n_query)
        Xg, Tg = np.meshgrid(xs, ts, indexing="ij")
        u_query = sol["predict"](Xg.ravel(), Tg.ravel()).reshape(n_query, n_query)
        return SolverOutput(
            result=torch.from_numpy(u_query.astype(np.float32)),
            losses={"residual": torch.tensor(sol["residual_history"][-1] if sol["residual_history"] else 0.0)},
            extras={
                "x_query": xs.astype(np.float32),
                "t_query": ts.astype(np.float32),
                "converged": sol["converged"],
                "n_iter": sol["n_iter"],
                "residual_history": sol["residual_history"],
                "method": "xtfc_pde",
                "pde": self.pde,
            },
        )
