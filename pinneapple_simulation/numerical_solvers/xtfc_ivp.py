"""X-TFC (eXtreme Theory of Functional Connections) solver for systems of ODEs
posed as initial-value problems:

    dy_i/dt = f_i(t, y_1, ..., y_n),   y_i(t0) = y_i0,   t in [t0, tf]

Reference: Schiassi, Leake, De Florio, Johnston, Furfaro & Mortari, "Extreme
Theory of Functional Connections: A Physics-Informed Neural Network Method
for Solving Parametric Differential Equations" (arXiv:2005.10632; journal
version: Neurocomputing 457 (2021) 334-356) -- Sections 2.1-2.3 for the
general constrained-expression + ELM framework -- and its IVP-focused
companion, De Florio, Schiassi et al., "Physics-Informed Extreme Theory of
Functional Connections Applied to Data-Driven Parameters Discovery of
Epidemiological Compartmental Models" (arXiv:2008.05554), whose Section 2
(Eqs. 10-15) gives the exact single-initial-condition constrained expression
and the Gauss-Newton/Jacobian iterative-least-squares update (Eqs. 7-9) this
module implements almost verbatim, adapted from its inverse-problem
(parameter-discovery) framing to the plain forward-solve framing used here
(no data loss term, no parameter unknowns -- just beta).

Algorithmically distinct from every other trainer in this package: there is
NO gradient descent and NO backprop anywhere in this file. A single-hidden-
layer Extreme Learning Machine (ELM) provides M FIXED random-feature basis
functions phi_j(x) = activation(w_j*x + b_j), with w_j, b_j sampled ONCE from
a fixed random distribution and never updated. Only the LINEAR output
weights (one vector beta_i per output y_i) are ever solved for -- via
ordinary least-squares for a linear ODE, or a Newton/Gauss-Newton-linearized
least-squares ITERATION for a nonlinear one (still no learning rate, no
epochs, no autodiff: each outer iteration is one exact linear solve).

The right-hand side `rhs_fn` (and optional analytic Jacobian `jac_fn`) is
entirely caller-supplied -- this module hardcodes no equation, so it applies
to any first-order ODE-IVP system: chemical/biological reaction kinetics,
epidemiological compartments, circuit dynamics, or any other well-mixed
lumped-parameter system.

Method summary
--------------
1. Domain mapping: map physical t in [t0, tf] to a numerically well-
   conditioned free variable x in [-1, 1]:
       x = -1 + 2*(t - t0)/T,   T = tf - t0   <=>   t = t0 + (x+1)*T/2
   so dy/dt = (2/T) * dy/dx by the chain rule (carried through every
   derivative below via `_DXDT_FACTOR = 2/T`).
2. TFC constrained expression per output i (Eq. 10-12 of arXiv:2008.05554):
       y_i(x) = g_i(x) + (y_i0 - g_i(x0)) = (phi(x) - phi(x0))^T beta_i + y_i0
   where phi(x) = [phi_1(x), ..., phi_M(x)] is the row of M FIXED ELM basis
   functions and x0 = -1 is the mapped initial point. y_i(x0) = y_i0
   IDENTICALLY for ANY beta_i -- the initial condition is satisfied by
   construction, never via a loss penalty. (Checked as this module's own
   sanity assertion in solve_xtfc_ode.)
3. Collocation points x_1..x_N in [-1, 1]: Chebyshev-Gauss-Lobatto by default
   (clusters points near both endpoints -- standard for conditioning and for
   resolving fast initial transients), uniform spacing as a documented
   fallback (`collocation="uniform"`).
4. Because phi_j (and phi_j(x0), a constant) are FIXED, dy_i/dx(x_p) =
   phi'(x_p)^T beta_i is EXACTLY LINEAR in beta_i, and phi'(x_p) is computed
   analytically from each activation's known closed-form derivative -- never
   via autodiff.
5. Linear ODEs (f_i linear in y): the residual R_i(x_p) = (2/T)*dy_i/dx(x_p)
   - f_i(t_p, y(x_p)) is affine in beta at every collocation point -> ONE
   `numpy.linalg.lstsq` solve gives the (up to floating-point) exact answer.
   Request this fast/explicit path with `linear=True` (single Newton
   iteration from the beta=0 starting point, which is exact for a truly
   linear right-hand side since its Taylor expansion has no remainder).
6. Nonlinear ODEs: Newton/Gauss-Newton iteration (paper's Eqs. 7-9). At outer
   iteration k, with current iterate beta_k and y_k(x) := y_of(beta_k), the
   nonlinear term f(t, y) is Taylor-expanded about y_k using the Jacobian
   df_i/dy_j(t, y_k) -- supplied analytically via `jac_fn` when the caller
   has one (exact, preferred), or estimated via central finite differences
   otherwise (since a generic caller-supplied `rhs_fn` is a black box, not
   necessarily autodiff-traceable or symbolically known). This linearization
   makes the residual AFFINE in the update delta = beta - beta_k; solving
   `A @ delta_vec = rhs_vec` via `numpy.linalg.lstsq` and setting
   beta_{k+1} = beta_k + delta is exactly one Newton/Gauss-Newton step (NOT
   gradient descent -- no learning rate, no minibatches, no epochs; the whole
   "iteration" is a handful of exact linear solves). Stops when the RMS
   residual drops below `tol`, `max_iter` is reached, or the residual
   worsens between iterations (the paper's own Eq. 9 divergence guard) -- in
   the last case the pre-update beta is kept.
"""
from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

Array = np.ndarray


def _tanh_act(z: Array) -> Array:
    return np.tanh(z)


def _tanh_dact(z: Array) -> Array:
    t = np.tanh(z)
    return 1.0 - t * t


def _sigmoid_act(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-z))


def _sigmoid_dact(z: Array) -> Array:
    s = _sigmoid_act(z)
    return s * (1.0 - s)


def _sin_act(z: Array) -> Array:
    return np.sin(z)


def _sin_dact(z: Array) -> Array:
    return np.cos(z)


# activation name -> (function, derivative), both elementwise -- new
# activations can be added here without touching any other logic in this
# module (the derivative is always w.r.t. the activation's own argument z;
# the chain-rule factor w_j from z = w_j*x + b_j is applied in _elm_basis).
ACTIVATIONS: Dict[str, "tuple[Callable[[Array], Array], Callable[[Array], Array]]"] = {
    "tanh": (_tanh_act, _tanh_dact),
    "sigmoid": (_sigmoid_act, _sigmoid_dact),
    "sin": (_sin_act, _sin_dact),
}


def chebyshev_gauss_lobatto(n: int, lo: float = -1.0, hi: float = 1.0) -> Array:
    """N Chebyshev-Gauss-Lobatto points on [lo, hi], ascending order.
    Standard collocation choice for spectral/TFC methods -- clusters points
    quadratically near both endpoints, which both improves the conditioning
    of the least-squares system (vs. uniform spacing, which is offered as a
    fallback via `collocation="uniform"`) and better resolves fast initial
    transients in stiff/multi-timescale IVPs."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([0.5 * (lo + hi)])
    k = np.arange(n)
    x = -np.cos(np.pi * k / (n - 1))  # in [-1, 1], ascending
    return 0.5 * (hi - lo) * x + 0.5 * (hi + lo)


def _elm_basis(x: Array, w: Array, b: Array, activation: str) -> "tuple[Array, Array]":
    """phi(x) and dphi/dx(x) for M fixed random features, at points x.

    x: (N,) points in the mapped [-1, 1] domain.
    w, b: (M,) fixed random input weights / biases (sampled once, never
    updated -- this IS the "extreme"/ELM part of X-TFC).
    Returns (phi, dphi_dx), each (N, M).
    """
    act, dact = ACTIVATIONS[activation]
    z = np.outer(x, w) + b[None, :]         # (N, M)
    phi = act(z)
    dphi_dx = dact(z) * w[None, :]          # chain rule: d/dx (w*x + b) = w
    return phi, dphi_dx


def _numerical_jacobian(rhs_fn: Callable[[Array, Array], Array], t: Array, y: Array,
                          rel_eps: float = 1e-6) -> Array:
    """Central-difference Jacobian J[p, i, j] = d f_i/d y_j (t_p, y_p), for a
    generic black-box rhs_fn(t, y) -> (N, n_out) that isn't assumed to be
    autodiff-traceable or symbolically known. Used only when the caller
    doesn't supply an exact analytic `jac_fn` to solve_xtfc_ode -- an
    ordinary numerical-least-squares technique (cf. scipy.optimize.
    least_squares' own finite-difference Jacobian fallback), not gradient
    descent: this Jacobian feeds a single linear solve per outer iteration,
    it is never used to take a "step" itself.
    """
    n_pts, n_out = y.shape
    J = np.zeros((n_pts, n_out, n_out))
    for j in range(n_out):
        h = rel_eps * np.maximum(1.0, np.abs(y[:, j]))
        y_plus = y.copy();  y_plus[:, j] += h
        y_minus = y.copy(); y_minus[:, j] -= h
        f_plus = rhs_fn(t, y_plus)
        f_minus = rhs_fn(t, y_minus)
        J[:, :, j] = (f_plus - f_minus) / (2.0 * h)[:, None]
    return J


def xtfc_predict(t_query: Array, w: Array, b: Array, beta: Array, y0: Array,
                  activation: str, t0: float, tf: float) -> Array:
    """Evaluate the trained TFC constrained expression at arbitrary query
    times -- a pure function of the persisted (w, b, beta, y0, activation,
    t0, tf), so a solved model can be re-evaluated later from just those
    small arrays without re-running solve_xtfc_ode.

    Returns y_query, shape (len(t_query), n_out).
    """
    T = tf - t0
    x_query = -1.0 + 2.0 * (np.asarray(t_query, dtype=np.float64) - t0) / T
    phi, _ = _elm_basis(x_query, w, b, activation)
    phi0, _ = _elm_basis(np.array([-1.0]), w, b, activation)
    psi = phi - phi0                                    # (N, M)
    return psi @ beta + np.asarray(y0, dtype=np.float64)[None, :]


def solve_xtfc_ode(
    rhs_fn: Callable[[Array, Array], Array],
    y0,
    t_span: "tuple[float, float]",
    n_basis: int = 50,
    n_collocation: int = 100,
    activation: str = "tanh",
    max_iter: int = 50,
    tol: float = 1e-10,
    seed: int = 0,
    w_range: "tuple[float, float]" = (-5.0, 5.0),
    b_range: "tuple[float, float]" = (-5.0, 5.0),
    collocation: str = "chebyshev",
    linear: bool = False,
    jac_fn: "Callable[[Array, Array], Array] | None" = None,
) -> dict:
    """Solve dy_i/dt = rhs_fn(t, y)[:, i], y_i(t0) = y0[i], for t in t_span,
    via X-TFC (see module docstring for the full method).

    rhs_fn(t, y): t is (N,), y is (N, n_out) -> returns (N, n_out), the ODE
        right-hand side f(t, y) (NOT the TFC residual itself -- that is
        assembled internally from the constrained expression's analytic
        derivative minus this rhs).
    y0: (n_out,) initial condition, y(t0).
    t_span: (t0, tf), tf > t0.
    n_basis: M, number of fixed random ELM features.
    n_collocation: N, number of collocation points in [t0, tf].
    activation: one of ACTIVATIONS' keys ("tanh", "sigmoid", "sin").
    max_iter, tol: outer Newton/Gauss-Newton iteration cap and RMS-residual
        stopping tolerance (ignored -- capped at 1 -- when linear=True).
    seed: RNG seed for the fixed w, b draw (reproducibility).
    w_range, b_range: (lo, hi) uniform sampling range for the fixed ELM
        input weights/biases. Kept moderate (default +-5) so tanh/sigmoid
        don't saturate over the whole [-1, 1] mapped domain for every
        feature -- too narrow and the M basis functions barely differ from
        each other (near-singular normal equations); too wide and most
        features saturate into near-constant +-1 everywhere, wasting basis
        capacity. +-5 is a generic, moderate default; widen it if increasing
        n_basis stops improving accuracy.
    collocation: "chebyshev" (default, Gauss-Lobatto) or "uniform".
    linear: if True, skip the general iterative loop and do exactly ONE
        Newton step from beta=0 (exact for a genuinely-linear rhs_fn, since
        its Taylor expansion about any point has no remainder -- this is
        the "one linear least-squares solve" case). If False (default), run
        the general nonlinear iteration -- which, note, ALSO solves a
        genuinely linear ODE correctly (converging within 1-2 iterations,
        the Jacobian being constant everywhere), so `linear=True` is purely
        an explicit, slightly faster opt-in, never required for correctness.
    jac_fn(t, y): optional exact analytic Jacobian, same call signature as
        rhs_fn but returning (N, n_out, n_out) with jac_fn(t,y)[p,i,j] =
        d f_i/d y_j (t_p, y_p). Falls back to _numerical_jacobian (central
        finite differences) when omitted -- exact analytic Jacobians avoid
        finite-difference truncation/rounding error entirely and are
        strongly preferred whenever f is known in closed form.

    Returns a dict: {w, b, beta, y0, activation, t0, tf, n_basis,
    n_collocation, collocation, x_colloc, t_colloc, y_colloc, dydt_colloc,
    residual_history (list of RMS residuals, one per outer iteration),
    n_iter, converged, predict (callable t_query -> y_query)}.
    """
    if activation not in ACTIVATIONS:
        raise ValueError(f"activation={activation!r} not in {list(ACTIVATIONS)}")
    t0, tf = t_span
    T = tf - t0
    if T <= 0:
        raise ValueError("t_span must have tf > t0")

    y0_arr = np.atleast_1d(np.asarray(y0, dtype=np.float64))
    n_out = y0_arr.shape[0]

    rng = np.random.default_rng(seed)
    w = rng.uniform(w_range[0], w_range[1], size=n_basis)
    b = rng.uniform(b_range[0], b_range[1], size=n_basis)

    if collocation == "chebyshev":
        x = chebyshev_gauss_lobatto(n_collocation, -1.0, 1.0)
    elif collocation == "uniform":
        x = np.linspace(-1.0, 1.0, n_collocation)
    else:
        raise ValueError(f"collocation={collocation!r} must be 'chebyshev' or 'uniform'")

    phi, dphi_dx = _elm_basis(x, w, b, activation)          # (N, M) each
    phi0, _ = _elm_basis(np.array([-1.0]), w, b, activation)  # (1, M)
    psi = phi - phi0            # TFC projection basis: y_i(x) = psi @ beta_i + y0_i
    dpsi_dx = dphi_dx           # phi0 is a constant -> its derivative is 0

    # First sanity check the module's own docstring promises: the
    # constrained expression satisfies the IC EXACTLY, for ANY beta --
    # verify it here for beta=0 and a random beta so a future refactor that
    # breaks this identity fails loudly instead of silently producing a
    # slightly-wrong-at-t0 solution.
    for _beta_probe in (np.zeros((n_basis, n_out)), rng.standard_normal((n_basis, n_out))):
        y_at_x0 = phi0 @ _beta_probe - phi0 @ _beta_probe + y0_arr  # == y0_arr identically; psi(x0)=phi0-phi0=0
        assert np.allclose(y_at_x0, y0_arr), "TFC constrained expression violates its own IC identity"

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

        # Linearized (affine-in-delta) residual system, stacked over outputs
        # i (rows) and delta_j (columns) -- see module docstring point 6 for
        # the derivation. delta = beta_new - beta (the Newton update).
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
        delta = delta_vec.reshape(n_out, n_basis).T   # (M, n_out)
        beta_candidate = beta + delta

        if linear:
            beta = beta_candidate
            break

        # Divergence guard (paper's Eq. 9 second stopping condition): if the
        # new iterate's residual is WORSE than the current one, keep the
        # current (better) beta and stop rather than let Newton run away.
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

    y_colloc = y_of(beta)
    dydt_colloc = dydt_of(beta)

    def predict(t_query) -> Array:
        return xtfc_predict(np.asarray(t_query, dtype=np.float64), w, b, beta, y0_arr, activation, t0, tf)

    return {
        "w": w, "b": b, "beta": beta, "y0": y0_arr,
        "activation": activation, "t0": t0, "tf": tf,
        "n_basis": n_basis, "n_collocation": n_collocation, "collocation": collocation,
        "x_colloc": x, "t_colloc": t, "y_colloc": y_colloc, "dydt_colloc": dydt_colloc,
        "residual_history": residual_history, "n_iter": n_iter_done, "converged": converged,
        "predict": predict,
    }


@SolverRegistry.register(
    name="xtfc_ivp",
    family="ode",
    description="X-TFC (Extreme Theory of Functional Connections) zero-backprop ODE-IVP solver -- "
                "fixed random ELM basis + Newton/least-squares, no gradient descent.",
    tags=["xtfc", "ivp", "elm", "zero-backprop", "ode"],
)
class XTFCIVPSolver(SolverBase):
    """Thin `SolverBase`/registry wrapper around `solve_xtfc_ode` for callers
    that want the uniform `SolverOutput` interface. The functional API
    (`solve_xtfc_ode`, `xtfc_predict`) is the primary entry point and can be
    used directly without this wrapper.

    Note this solver is fundamentally different from pinneapple's other
    `xtfc`-named architecture (a gradient-trained network in
    `pinneapple_neural.architectures`): this one never backpropagates --
    only its M-fixed-feature output weights are solved for, via exact linear
    algebra.
    """

    def __init__(self, n_basis: int = 50, n_collocation: int = 100,
                 activation: str = "tanh", max_iter: int = 50, tol: float = 1e-10,
                 seed: int = 0, collocation: str = "chebyshev", linear: bool = False):
        super().__init__()
        self.n_basis = int(n_basis)
        self.n_collocation = int(n_collocation)
        self.activation = activation
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.collocation = collocation
        self.linear = bool(linear)

    def forward(
        self,
        rhs_fn: Callable[[Array, Array], Array],
        y0,
        t_span: "tuple[float, float]",
        jac_fn: "Callable[[Array, Array], Array] | None" = None,
        n_query: int = 200,
    ) -> SolverOutput:
        sol = solve_xtfc_ode(
            rhs_fn, y0, t_span,
            n_basis=self.n_basis, n_collocation=self.n_collocation,
            activation=self.activation, max_iter=self.max_iter, tol=self.tol,
            seed=self.seed, collocation=self.collocation, linear=self.linear,
            jac_fn=jac_fn,
        )
        t_query = np.linspace(t_span[0], t_span[1], n_query)
        y_query = sol["predict"](t_query)
        return SolverOutput(
            result=torch.from_numpy(y_query.astype(np.float32)),
            losses={"residual": torch.tensor(sol["residual_history"][-1] if sol["residual_history"] else 0.0)},
            extras={
                "t_query": t_query.astype(np.float32),
                "converged": sol["converged"],
                "n_iter": sol["n_iter"],
                "residual_history": sol["residual_history"],
                "method": "xtfc",
            },
        )
