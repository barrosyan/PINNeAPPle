"""Pre-built time-series datasets for PINNeAPPle benchmarks.

All datasets are generated from ODEs or synthetic functions — reproducible,
no external files needed.  Every loader returns a dict with at minimum:
  t  — time vector (n_steps,)
  <state fields>  — arrays of shape (n_steps,) or (n_steps, n_dim)
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .registry import DatasetInfo, DatasetRegistry


def _ode_solve(rhs, y0, t_span, t_eval, method="RK45"):
    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval,
                        method=method, rtol=1e-8, atol=1e-10)
        return sol.t, sol.y.T     # (n_t,), (n_t, n_dim)
    except ImportError:
        # Simple Euler fallback
        dt = t_eval[1] - t_eval[0]
        y = np.array(y0, dtype=float)
        ys = [y.copy()]
        for _ in t_eval[1:]:
            y = y + dt * np.array(rhs(0, y))
            ys.append(y.copy())
        return t_eval, np.array(ys)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Lorenz 63  (chaotic attractor)
# ─────────────────────────────────────────────────────────────────────────────

def _load_lorenz63(dt: float = 0.01, T: float = 50.0,
                   sigma: float = 10.0, rho: float = 28.0,
                   beta: float = 8.0/3.0,
                   seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y0 = rng.uniform(-5, 5, 3).tolist()
    t_eval = np.arange(0.0, T, dt)

    def _rhs(t, y):
        x, y_, z = y
        return [sigma*(y_ - x), x*(rho - z) - y_, x*y_ - beta*z]

    t, Y = _ode_solve(_rhs, y0, [0.0, T], t_eval)
    return {
        "t": t,
        "x": Y[:, 0], "y": Y[:, 1], "z": Y[:, 2],
        "X": Y,       # (n_t, 3) — convenient for windowing
        "sigma": np.float64(sigma),
        "rho": np.float64(rho),
        "beta": np.float64(beta),
        "description": "Lorenz 63 attractor — canonical chaotic time series",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Spring-Mass 1-DOF
#    m·x'' + c·x' + k·x = A·sin(ω·t)
# ─────────────────────────────────────────────────────────────────────────────

def _load_spring_mass_1dof(dt: float = 0.02, T: float = 20.0,
                            m: float = 1.0, k: float = 4.0,
                            c: float = 0.4, A: float = 0.5,
                            omega: float = 1.0) -> Dict[str, np.ndarray]:
    t_eval = np.arange(0.0, T, dt)

    def _rhs(t, y):
        x, v = y
        F = A * math.sin(omega * t)
        return [v, (F - c*v - k*x) / m]

    t, Y = _ode_solve(_rhs, [0.0, 0.0], [0.0, T], t_eval)
    return {
        "t": t, "position": Y[:, 0], "velocity": Y[:, 1],
        "X": Y,
        "m": np.float64(m), "k": np.float64(k),
        "c": np.float64(c), "A": np.float64(A), "omega": np.float64(omega),
        "description": "Forced damped 1-DOF spring-mass oscillator",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Spring-Mass 2-DOF  (coupled system, same as cosim example)
# ─────────────────────────────────────────────────────────────────────────────

def _load_spring_mass_2dof(dt: float = 0.02, T: float = 20.0) -> Dict[str, np.ndarray]:
    M1, K1, C1 = 1.0, 4.0, 0.4
    M2, K2, C2 = 0.5, 2.0, 0.2
    KC = 1.0
    A_F, W_F = 0.5, 1.0
    t_eval = np.arange(0.0, T, dt)

    def _rhs(t, y):
        x1, v1, x2, v2 = y
        F = A_F * math.sin(W_F * t)
        a1 = (F  - C1*v1 - K1*x1 - KC*(x1 - x2)) / M1
        a2 = (KC*(x1 - x2) - C2*v2 - K2*x2) / M2
        return [v1, a1, v2, a2]

    t, Y = _ode_solve(_rhs, [0.0, 0.0, 0.0, 0.0], [0.0, T], t_eval)
    return {
        "t": t,
        "x1": Y[:, 0], "v1": Y[:, 1],
        "x2": Y[:, 2], "v2": Y[:, 3],
        "X": Y,
        "description": "2-DOF coupled spring-mass oscillator",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Van der Pol oscillator
#    x'' - μ(1-x²)x' + x = 0
# ─────────────────────────────────────────────────────────────────────────────

def _load_van_der_pol(dt: float = 0.02, T: float = 40.0,
                      mu: float = 2.0) -> Dict[str, np.ndarray]:
    t_eval = np.arange(0.0, T, dt)

    def _rhs(t, y):
        x, v = y
        return [v, mu*(1 - x**2)*v - x]

    t, Y = _ode_solve(_rhs, [2.0, 0.0], [0.0, T], t_eval)
    return {
        "t": t, "x": Y[:, 0], "v": Y[:, 1],
        "X": Y,
        "mu": np.float64(mu),
        "description": "Van der Pol oscillator — limit cycle dynamics",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Rössler attractor
#    dx/dt = -y - z,  dy/dt = x + a*y,  dz/dt = b + z*(x-c)
# ─────────────────────────────────────────────────────────────────────────────

def _load_rossler(dt: float = 0.05, T: float = 200.0,
                  a: float = 0.2, b: float = 0.2,
                  c: float = 5.7) -> Dict[str, np.ndarray]:
    t_eval = np.arange(0.0, T, dt)

    def _rhs(t, y):
        x, y_, z = y
        return [-y_ - z, x + a*y_, b + z*(x - c)]

    t, Y = _ode_solve(_rhs, [1.0, 0.0, 0.0], [0.0, T], t_eval)
    return {
        "t": t,
        "x": Y[:, 0], "y": Y[:, 1], "z": Y[:, 2],
        "X": Y,
        "a": np.float64(a), "b": np.float64(b), "c": np.float64(c),
        "description": "Rössler attractor — chaotic time series",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Double pendulum
# ─────────────────────────────────────────────────────────────────────────────

def _load_double_pendulum(dt: float = 0.02, T: float = 30.0,
                          m1: float = 1.0, m2: float = 1.0,
                          L1: float = 1.0, L2: float = 1.0,
                          g: float = 9.81) -> Dict[str, np.ndarray]:
    t_eval = np.arange(0.0, T, dt)

    def _rhs(t, y):
        th1, om1, th2, om2 = y
        d = th1 - th2
        den1 = (m1 + m2) * L1 - m2 * L1 * math.cos(d)**2
        den2 = (L2 / L1) * den1
        dom1 = (-m2 * L1 * om1**2 * math.sin(d) * math.cos(d)
                + m2 * g * math.sin(th2) * math.cos(d)
                - m2 * L2 * om2**2 * math.sin(d)
                - (m1 + m2) * g * math.sin(th1)) / den1
        dom2 = (m2 * L2 * om2**2 * math.sin(d) * math.cos(d)
                + (m1 + m2) * g * math.sin(th1) * math.cos(d)
                + (m1 + m2) * L1 * om1**2 * math.sin(d)
                - (m1 + m2) * g * math.sin(th2)) / den2
        return [om1, dom1, om2, dom2]

    t, Y = _ode_solve(_rhs, [math.pi/2, 0.0, math.pi/3, 0.0], [0.0, T], t_eval)
    return {
        "t": t,
        "theta1": Y[:, 0], "omega1": Y[:, 1],
        "theta2": Y[:, 2], "omega2": Y[:, 3],
        "X": Y,
        "description": "Double pendulum — chaotic mechanical system",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Lorenz 96  (high-dimensional chaos)
#    dX_i/dt = (X_{i+1} - X_{i-2})·X_{i-1} - X_i + F
# ─────────────────────────────────────────────────────────────────────────────

def _load_lorenz96(dt: float = 0.01, T: float = 20.0,
                   N: int = 20, F: float = 8.0,
                   seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    y0 = rng.uniform(-1, 1, N)
    y0[0] += 0.01   # small perturbation
    t_eval = np.arange(0.0, T, dt)

    def _rhs(t, y):
        d = np.zeros(N)
        for i in range(N):
            d[i] = (y[(i+1) % N] - y[(i-2) % N]) * y[(i-1) % N] - y[i] + F
        return d

    t, Y = _ode_solve(_rhs, y0.tolist(), [0.0, T], t_eval)
    return {
        "t": t,
        "X": Y,         # (n_t, N)
        "N": N,
        "F": np.float64(F),
        "description": f"Lorenz 96 model (N={N}, F={F}) — high-dimensional chaos",
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Synthetic multi-frequency sine + noise
# ─────────────────────────────────────────────────────────────────────────────

def _load_sine_noise(n: int = 2000, dt: float = 0.05,
                     frequencies: tuple = (0.5, 1.2, 3.0),
                     amplitudes: tuple = (1.0, 0.5, 0.25),
                     noise_std: float = 0.05,
                     seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    signal = np.zeros(n)
    for freq, amp in zip(frequencies, amplitudes):
        signal += amp * np.sin(2 * math.pi * freq * t)
    signal += rng.normal(0, noise_std, n)
    return {
        "t": t, "signal": signal,
        "X": signal.reshape(-1, 1),
        "frequencies": np.array(frequencies),
        "amplitudes": np.array(amplitudes),
        "noise_std": np.float64(noise_std),
        "description": "Synthetic multi-frequency sine wave with Gaussian noise",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

DatasetRegistry.register(
    DatasetInfo(
        id="lorenz63",
        name="Lorenz 63",
        category="timeseries",
        description="Lorenz 63 chaotic attractor. 3-state ODE, σ=10, ρ=28, β=8/3.",
        fields=["t", "x", "y", "z", "X"],
        tags=["chaos", "ode", "3d", "attractor"],
        reference="Lorenz 1963",
    ),
    _load_lorenz63,
)

DatasetRegistry.register(
    DatasetInfo(
        id="spring_mass_1dof",
        name="Spring-Mass 1-DOF",
        category="timeseries",
        description="Forced damped 1-DOF spring-mass oscillator.",
        fields=["t", "position", "velocity", "X"],
        tags=["mechanical", "ode", "oscillator", "vibration"],
    ),
    _load_spring_mass_1dof,
)

DatasetRegistry.register(
    DatasetInfo(
        id="spring_mass_2dof",
        name="Spring-Mass 2-DOF",
        category="timeseries",
        description="Coupled 2-DOF spring-mass system with external forcing.",
        fields=["t", "x1", "v1", "x2", "v2", "X"],
        tags=["mechanical", "ode", "coupled", "vibration", "cosim"],
    ),
    _load_spring_mass_2dof,
)

DatasetRegistry.register(
    DatasetInfo(
        id="van_der_pol",
        name="Van der Pol Oscillator",
        category="timeseries",
        description="Van der Pol oscillator — nonlinear limit-cycle dynamics.",
        fields=["t", "x", "v", "X"],
        tags=["nonlinear", "limit-cycle", "ode"],
        reference="Van der Pol 1927",
    ),
    _load_van_der_pol,
)

DatasetRegistry.register(
    DatasetInfo(
        id="rossler",
        name="Rössler Attractor",
        category="timeseries",
        description="Rössler chaotic attractor — 3-state ODE.",
        fields=["t", "x", "y", "z", "X"],
        tags=["chaos", "attractor", "ode"],
        reference="Rössler 1976",
    ),
    _load_rossler,
)

DatasetRegistry.register(
    DatasetInfo(
        id="double_pendulum",
        name="Double Pendulum",
        category="timeseries",
        description="Double pendulum — chaotic mechanical system with 4 state variables.",
        fields=["t", "theta1", "omega1", "theta2", "omega2", "X"],
        tags=["chaos", "mechanical", "pendulum", "ode"],
    ),
    _load_double_pendulum,
)

DatasetRegistry.register(
    DatasetInfo(
        id="lorenz96",
        name="Lorenz 96",
        category="timeseries",
        description="High-dimensional Lorenz 96 model — weather-like spatiotemporal chaos.",
        fields=["t", "X"],
        tags=["chaos", "high-dimensional", "weather", "ode"],
        reference="Lorenz 1996",
    ),
    _load_lorenz96,
)

DatasetRegistry.register(
    DatasetInfo(
        id="sine_noise",
        name="Synthetic Sine + Noise",
        category="timeseries",
        description="Multi-frequency sine waves with Gaussian noise — simple forecasting baseline.",
        fields=["t", "signal", "X"],
        tags=["synthetic", "sine", "noise", "forecasting"],
    ),
    _load_sine_noise,
)
