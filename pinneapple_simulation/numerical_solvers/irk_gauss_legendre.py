"""Butcher tableaus for the q-stage Gauss-Legendre implicit Runge-Kutta (IRK)
family -- a general-purpose, arbitrarily-high-order (order 2q), A-stable and
B-stable time integrator for any ODE-IVP system dy/dt = f(t,y), stiff or
non-stiff. Pure NumPy/SciPy, no model/framework dependency, so it can be
built, cached, and validated in complete isolation from whatever consumes
the tableau (e.g. a discrete-time collocation scheme, or a hand-rolled
implicit RK time-stepper).

Derivation
----------
The q collocation nodes c_1..c_q are the Gauss-Legendre quadrature nodes on
[0,1] (rescaled from the standard [-1,1] nodes). For a collocation-based IRK
method, the tableau is exactly:

    A[i,j] = \\int_0^{c_i} L_j(tau) dtau
    b[j]   = \\int_0^1   L_j(tau) dtau

where L_j is the degree-(q-1) Lagrange basis polynomial with L_j(c_k) =
delta_jk. Building A via a monomial Vandermonde solve is catastrophically
ill-conditioned once q gets into the dozens (Vandermonde condition number
grows exponentially in q). Instead we expand in the shifted-Legendre basis
P_k(2*tau-1), k=0..q-1, which is orthogonal and stays well-conditioned at
any q:

    P_k(2*c_j-1) = sum_j' M[k,j'] ... (evaluated at each node)

Since {L_j}_j and {P_k(2*tau-1)}_k are both bases of the same degree-(q-1)
polynomial space, and P_k(2*tau-1) has degree k <= q-1, Lagrange
interpolation is exact:

    P_k(2*tau-1) = sum_j P_k(2*c_j-1) * L_j(tau)   for all tau.

Integrating both sides from 0 to c_i and writing M[k,j] = P_k(2*c_j-1):

    rhs_i[k] := \\int_0^{c_i} P_k(2*tau-1) dtau = sum_j M[k,j] * A[i,j]

so each row of A solves the linear system  M @ A[i,:] = rhs_i, with M
q-by-q and well-conditioned (shifted-Legendre polynomials are orthogonal on
[0,1]). rhs_i[k] has a closed form via the standard Legendre antiderivative
identity  \\int P_k(s) ds = (P_{k+1}(s) - P_{k-1}(s)) / (2k+1)  (k>=1),
\\int P_0(s) ds = s, giving an EXACT (not quadrature-approximated) rhs -- see
_rhs_vector below. b is obtained the same way (rhs for the upper limit
tau=1 collapses to [1, 0, 0, ..., 0] since P_k(1)=1 for every k), and cross-
checked against the direct Gauss-Legendre quadrature weights.
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
from scipy.special import eval_legendre


def _shifted_legendre_matrix(c: np.ndarray, q: int) -> np.ndarray:
    """M[k, j] = P_k(2*c_j - 1), k=0..q-1, j=0..q-1."""
    s = 2.0 * c - 1.0
    return np.stack([eval_legendre(k, s) for k in range(q)], axis=0)


def _rhs_vector(c_i: float, q: int) -> np.ndarray:
    """rhs[k] = integral_0^{c_i} P_k(2*tau-1) dtau, exact closed form."""
    b_arg = 2.0 * c_i - 1.0
    rhs = np.empty(q)
    rhs[0] = c_i  # = (b_arg - (-1)) / 2
    for k in range(1, q):
        # (P_{k+1}(b) - P_{k-1}(b)) / (2k+1), halved for the tau->s Jacobian,
        # and the lower-limit (s=-1) contribution is exactly 0 for k>=1.
        rhs[k] = (eval_legendre(k + 1, b_arg) - eval_legendre(k - 1, b_arg)) / (2 * (2 * k + 1))
    return rhs


def build_gauss_legendre_tableau(q: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the q-stage Gauss-Legendre IRK tableau (A, b, c) on [0,1].

    Returns A (q,q), b (q,), c (q,) as float64 arrays.
    """
    if q < 1:
        raise ValueError(f"q must be >= 1, got {q}")

    nodes_pm1, weights_pm1 = np.polynomial.legendre.leggauss(q)  # on [-1,1]
    c = (nodes_pm1 + 1.0) / 2.0
    b_direct = weights_pm1 / 2.0

    M = _shifted_legendre_matrix(c, q)  # (q,q), well-conditioned (orthogonal basis)

    A = np.empty((q, q))
    for i in range(q):
        rhs_i = _rhs_vector(c[i], q)
        A[i, :] = np.linalg.solve(M, rhs_i)

    # b via the same linear system at the upper limit tau=1: rhs collapses
    # to [1,0,...,0] since P_k(1)=1 for every k. Cross-check against the
    # direct Gauss-Legendre quadrature weights (see validate_tableau).
    rhs_b = np.zeros(q)
    rhs_b[0] = 1.0
    b = np.linalg.solve(M, rhs_b)

    if not np.allclose(b, b_direct, atol=1e-8, rtol=1e-6):
        raise RuntimeError(
            f"IRK tableau build failed self-check at q={q}: b from the "
            f"shifted-Legendre solve does not match the direct Gauss-Legendre "
            f"quadrature weights (max abs diff = {np.max(np.abs(b - b_direct)):.3e})."
        )

    return A, b_direct, c


def _order_condition_C(A: np.ndarray, c: np.ndarray, q: int) -> float:
    """Residual of Butcher's simplifying assumption C(q):
    sum_j A[i,j] * c_j^{k-1} = c_i^k / k,  k=1..q, for every stage i.
    """
    max_res = 0.0
    for k in range(1, q + 1):
        lhs = A @ (c ** (k - 1))
        rhs = (c ** k) / k
        max_res = max(max_res, float(np.max(np.abs(lhs - rhs))))
    return max_res


def _order_condition_B(b: np.ndarray, c: np.ndarray, order: int) -> float:
    """Residual of Butcher's simplifying assumption B(order):
    sum_i b_i * c_i^{k-1} = 1/k,  k=1..order.
    """
    max_res = 0.0
    for k in range(1, order + 1):
        lhs = float(np.sum(b * (c ** (k - 1))))
        rhs = 1.0 / k
        max_res = max(max_res, abs(lhs - rhs))
    return max_res


def validate_tableau(A: np.ndarray, b: np.ndarray, c: np.ndarray, q: int, tol: float = 1e-9) -> dict:
    """Hard-assert the order conditions that certify a genuine 2q-order
    collocation method (C(q) and B(2q) jointly). Raises AssertionError with
    diagnostic residuals on failure; returns a dict of residuals on success.
    """
    res_C = _order_condition_C(A, c, q)
    res_B = _order_condition_B(b, c, 2 * q)
    row_sums = A.sum(axis=1)
    res_row_sum = float(np.max(np.abs(row_sums - c)))  # consistency: sum_j A[i,j] = c_i

    diagnostics = {"C(q)_residual": res_C, "B(2q)_residual": res_B, "row_sum_residual": res_row_sum}
    assert res_C < tol, f"q={q}: C(q) order-condition residual {res_C:.3e} exceeds tol {tol:.1e}: {diagnostics}"
    assert res_B < tol, f"q={q}: B(2q) order-condition residual {res_B:.3e} exceeds tol {tol:.1e}: {diagnostics}"
    assert res_row_sum < tol, f"q={q}: row-sum (consistency) residual {res_row_sum:.3e} exceeds tol {tol:.1e}: {diagnostics}"
    return diagnostics


def get_irk_tableau(q: int, cache_dir: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build-or-load-cached, then validate, the q-stage Gauss-Legendre IRK
    tableau. Raises loudly (AssertionError/RuntimeError) rather than
    returning a silently-wrong tableau.
    """
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"irk_gauss_q{q}.npz"
        if cache_path.exists():
            data = np.load(cache_path)
            A, b, c = data["A"], data["b"], data["c"]
            validate_tableau(A, b, c, q)
            return A, b, c

    A, b, c = build_gauss_legendre_tableau(q)
    validate_tableau(A, b, c, q)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, A=A, b=b, c=c)

    return A, b, c


def step_irk(
    rhs_fn,
    t: float,
    y: np.ndarray,
    h: float,
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    jac_fn=None,
    newton_tol: float = 1e-10,
    newton_max_iter: int = 25,
) -> np.ndarray:
    """Advance y(t) -> y(t+h) by one fully-implicit Runge-Kutta step using
    tableau (A, b, c) (e.g. from `get_irk_tableau`). Solves the q coupled
    stage equations

        K_i = rhs_fn(t + c_i*h, y + h * sum_j A[i,j]*K_j),  i=1..q

    via a fixed-point-seeded Newton iteration on the stacked (q*n,) stage
    vector, using a numerical Jacobian (central differences) when `jac_fn`
    is not supplied. y is (n,); returns y_new, shape (n,).

    This is a direct, general-purpose consumer of the tableau -- useful on
    its own for a plain implicit-RK time-stepper, independent of any
    particular collocation/PINN scheme that might also use the same
    tableau.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.shape[0]
    q = A.shape[0]

    def stage_residual(K_flat: np.ndarray) -> np.ndarray:
        K = K_flat.reshape(q, n)
        res = np.empty((q, n))
        for i in range(q):
            y_stage = y + h * (A[i, :] @ K)
            res[i] = K[i] - rhs_fn(t + c[i] * h, y_stage)
        return res.ravel()

    # Fixed-point seed: K_i^(0) = rhs_fn(t + c_i*h, y) (exact for h=0, a
    # reasonable starting guess for small/moderate h).
    K = np.tile(rhs_fn(t, y), (q, 1)).ravel()

    for _ in range(newton_max_iter):
        F = stage_residual(K)
        if np.sqrt(np.mean(F ** 2)) < newton_tol:
            break
        if jac_fn is not None:
            J_local = jac_fn(t, y)  # (n,n), assumed ~constant across the step (good for mildly nonlinear rhs)
            # Full stage Jacobian dF/dK = I - h*(A kron J_local); Newton step via that block structure.
            Jfull = np.eye(q * n) - h * np.kron(A, J_local)
        else:
            eps = 1e-6
            Jfull = np.eye(q * n)
            for k in range(q * n):
                Kp = K.copy(); Kp[k] += eps
                Km = K.copy(); Km[k] -= eps
                Jfull[:, k] = (stage_residual(Kp) - stage_residual(Km)) / (2 * eps)
        delta = np.linalg.solve(Jfull, -F)
        K = K + delta

    K = K.reshape(q, n)
    return y + h * (b @ K)
