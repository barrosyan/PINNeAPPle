"""The four canonical operator-learning ground-truth solvers from Lu, Jin &
Karniadakis 2019 (DeepONet), arXiv:1910.03193, Section 4: given a GRF-sampled
input function u(x) (`pinneapple_data.grf_sampler.GRFDraw`), each solver
computes the TRUE solution of a different dynamical system driven by u.
Standard, widely-used ground truth for benchmarking any operator-learning
architecture (DeepONet, FNO, or otherwise) — not tied to a specific dataset
or application. Each solver takes a GRFDraw plus query points and operates
on ONE case at a time; a dataset-generation loop over many draws assembles
whatever operator-learning training format the caller needs (e.g. sensor
values + query point + solution at that query point).
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from pinneapple_data.grf_sampler import GRFDraw


# ── 4.1a: linear ODE  ds/dx = u(x),  s(0) = 0 ───────────────────────────────

def solve_ode_linear(draw: GRFDraw, x_query: np.ndarray) -> np.ndarray:
    """Exact via the GRF spline's own antiderivative (piecewise-cubic, so
    integration is exact for the spline itself, not an ODE-solver
    approximation of it)."""
    antideriv = draw.spline.antiderivative()
    x0 = draw.x_grid[0]
    return antideriv(x_query) - antideriv(x0)


# ── 4.1b: nonlinear ODE  ds/dx = -s^2 + u(x),  s(0) = 0 ─────────────────────

def solve_ode_nonlinear(draw: GRFDraw, x_query: np.ndarray) -> np.ndarray:
    x0 = draw.x_grid[0]
    x_max_query = float(np.max(x_query))

    def rhs(x, s):
        return -s ** 2 + draw(x)

    sol = solve_ivp(rhs, (x0, x_max_query), [0.0], dense_output=True,
                     rtol=1e-9, atol=1e-11, method="RK45")
    if not sol.success:
        raise RuntimeError(f"solve_ode_nonlinear: integration failed -- {sol.message}")
    return sol.sol(x_query)[0]


# ── 4.2: driven pendulum (2-state ODE system) ───────────────────────────────
#   ds1/dx = s2
#   ds2/dx = -k*sin(s1) + u(x)
#   s1(0) = s2(0) = 0

def solve_pendulum(draw: GRFDraw, x_query: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Returns s1 (angle) at each query point."""
    x0 = draw.x_grid[0]
    x_max_query = float(np.max(x_query))

    def rhs(x, s):
        s1, s2 = s
        return [s2, -k * np.sin(s1) + draw(x)]

    sol = solve_ivp(rhs, (x0, x_max_query), [0.0, 0.0], dense_output=True,
                     rtol=1e-9, atol=1e-11, method="RK45")
    if not sol.success:
        raise RuntimeError(f"solve_pendulum: integration failed -- {sol.message}")
    return sol.sol(x_query)[0]


def pendulum_rk4_reference(draw: GRFDraw, x_max: float, k: float, n_steps: int = 20000) -> float:
    """Independent hand-written fixed-step RK4 integrator, useful ONLY to
    cross-check solve_ivp's adaptive integration in a validation script --
    not meant for production use (solve_pendulum's adaptive RK45 is both
    faster and, by default tolerance, at least as accurate)."""
    x0 = draw.x_grid[0]
    h = (x_max - x0) / n_steps
    s = np.array([0.0, 0.0])

    def f(x, s):
        return np.array([s[1], -k * np.sin(s[0]) + draw(x)])

    x = x0
    for _ in range(n_steps):
        k1 = f(x, s)
        k2 = f(x + h / 2, s + h / 2 * k1)
        k3 = f(x + h / 2, s + h / 2 * k2)
        k4 = f(x + h, s + h * k3)
        s = s + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x += h
    return float(s[0])


# ── 4.3: diffusion-reaction PDE ─────────────────────────────────────────────
#   ds/dt = D*d2s/dx2 + k*s^2 + u(x),  x in [0,1], t in [0,1]
#   s(x,0) = 0,  s(0,t) = s(1,t) = 0

def solve_diffusion_reaction(
    draw: GRFDraw,
    nx: int = 100,
    nt: int = 100,
    D: float = 0.01,
    k: float = 0.01,
    x_max: float = 1.0,
    t_max: float = 1.0,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Crank-Nicolson for the diffusion term + one Picard correction per step
    for the k*s^2 reaction term, tridiagonal (Thomas algorithm) solves.
    Returns (x_grid, t_grid, s) with s.shape == (nt, nx).
    """
    x = np.linspace(0.0, x_max, nx)
    dt = t_max / (nt - 1)
    dx = x[1] - x[0]
    r = D * dt / dx ** 2

    u_vals = draw(x)  # forcing term, evaluated once (time-independent)

    s = np.zeros((nt, nx))
    n_int = nx - 2  # interior points (Dirichlet BC fixes the two edges at 0)

    # Constant tridiagonal system matrix for the CN diffusion operator
    # (I - r/2 * Laplacian) s^{n+1} = (I + r/2 * Laplacian) s^n + dt*(reaction + forcing)
    a_sub = np.full(n_int, -r / 2)
    a_diag = np.full(n_int, 1.0 + r)
    a_sup = np.full(n_int, -r / 2)

    for n in range(nt - 1):
        s_n = s[n, 1:-1]
        lap_n = np.zeros(n_int)
        lap_n[1:-1] = s_n[2:] - 2 * s_n[1:-1] + s_n[:-2]
        lap_n[0] = s_n[1] - 2 * s_n[0] + s[n, 0]
        lap_n[-1] = s[n, -1] - 2 * s_n[-1] + s_n[-2]

        # Picard iteration on the nonlinear reaction term k*s^2 (2 sweeps is
        # plenty for weak nonlinearity at this dt; validate against the
        # exact k=0 linear case first if changing D/k substantially).
        s_guess = s_n.copy()
        for _ in range(2):
            rhs = s_n + (r / 2) * lap_n + dt * (k * s_guess ** 2 + u_vals[1:-1])
            diag = a_diag.copy()
            d = rhs.copy()
            for i in range(1, n_int):
                m = a_sub[i] / diag[i - 1]
                diag[i] -= m * a_sup[i - 1]
                d[i] -= m * d[i - 1]
            s_new = np.zeros(n_int)
            s_new[-1] = d[-1] / diag[-1]
            for i in range(n_int - 2, -1, -1):
                s_new[i] = (d[i] - a_sup[i] * s_new[i + 1]) / diag[i]
            s_guess = s_new

        s[n + 1, 1:-1] = s_guess
        s[n + 1, 0] = 0.0
        s[n + 1, -1] = 0.0

    t = np.linspace(0.0, t_max, nt)
    return x, t, s
