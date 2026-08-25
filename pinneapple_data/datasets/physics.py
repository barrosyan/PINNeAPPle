"""Pre-built physics datasets for PINNeAPPle benchmarks.

All datasets are generated analytically or via scipy method-of-lines.
No external data files required — everything is reproducible from code.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np

from .registry import DatasetInfo, DatasetRegistry


# ─────────────────────────────────────────────────────────────────────────────
# 1. Burgers 1D
#    u_t + u·u_x = ν·u_xx,  u(-1,t)=u(1,t)=0,  u(x,0)=-sin(πx)
# ─────────────────────────────────────────────────────────────────────────────

def _load_burgers_1d(Nx: int = 128, Nt: int = 101,
                     nu: float = 0.01 / math.pi) -> Dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, Nx)
    dx = x[1] - x[0]

    def _rhs(t, u):
        u = u.copy(); u[0] = u[-1] = 0.0
        u_fwd = (np.roll(u, -1) - u) / dx
        u_bwd = (u - np.roll(u, 1)) / dx
        ux = np.where(u >= 0, u_bwd, u_fwd)
        ux[0] = ux[-1] = 0.0
        uxx = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
        uxx[0] = uxx[-1] = 0.0
        return -u * ux + nu * uxx

    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            _rhs, [0.0, 1.0], -np.sin(np.pi * x),
            t_eval=np.linspace(0, 1, Nt),
            method="Radau", rtol=1e-6, atol=1e-8,
        )
        t_out = sol.t
        u_out = sol.y.T          # (Nt_actual, Nx)
    except ImportError:
        # No scipy: explicit-Euler FD fallback that actually solves the
        # (nonlinear) viscous Burgers equation, reusing the same upwind
        # advection + central-diff diffusion RHS used in the scipy path.
        # Substepped to an explicit-Euler stability bound so it stays stable:
        #   diffusion CFL: dt <= 0.5*dx^2/nu
        #   advection CFL: dt <= dx/|u|_max
        t_out = np.linspace(0.0, 1.0, Nt)
        u = -np.sin(np.pi * x)
        u[0] = u[-1] = 0.0
        u_out = np.zeros((Nt, Nx))
        u_out[0] = u
        u_bound = max(float(np.max(np.abs(u))), 1e-8)
        dt_stable = 0.8 * min(0.5 * dx**2 / nu, dx / u_bound)
        for i in range(1, Nt):
            t_cur, t_target = t_out[i - 1], t_out[i]
            n_sub = max(1, int(math.ceil((t_target - t_cur) / dt_stable)))
            dt_sub = (t_target - t_cur) / n_sub
            for _ in range(n_sub):
                u = u + dt_sub * _rhs(0.0, u)
                u[0] = u[-1] = 0.0
            u_out[i] = u

    return {
        "x": x, "t": t_out, "u": u_out,
        "nu": np.float64(nu),
        "pde": "u_t + u*u_x = nu*u_xx",
        "domain_x": np.array([-1.0, 1.0]),
        "domain_t": np.array([0.0, 1.0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Heat 1D  (analytical)
#    u_t = k·u_xx,  u(0,t)=u(1,t)=0,  u(x,0)=sin(πx)
#    Exact: u = sin(πx)·exp(-k·π²·t)
# ─────────────────────────────────────────────────────────────────────────────

def _load_heat_1d(Nx: int = 64, Nt: int = 64,
                  k: float = 0.4) -> Dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, Nx)
    t = np.linspace(0.0, 0.5, Nt)
    X, T = np.meshgrid(x, t)
    u = np.sin(np.pi * X) * np.exp(-k * np.pi**2 * T)
    return {
        "x": x, "t": t, "u": u,
        "k": np.float64(k),
        "pde": "u_t = k*u_xx",
        "domain_x": np.array([0.0, 1.0]),
        "domain_t": np.array([0.0, 0.5]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Heat 2D  (analytical)
#    u_t = k·(u_xx + u_yy),  zero BC on unit square
#    Exact: u = sin(πx)·sin(πy)·exp(-2k·π²·t)
# ─────────────────────────────────────────────────────────────────────────────

def _load_heat_2d(N: int = 32, Nt: int = 32,
                  k: float = 0.1) -> Dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    t = np.linspace(0.0, 0.5, Nt)
    X, Y, T = np.meshgrid(x, y, t, indexing="ij")
    u = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-2*k*np.pi**2 * T)
    return {
        "x": x, "y": y, "t": t,
        "u": u,       # shape (N, N, Nt)
        "k": np.float64(k),
        "pde": "u_t = k*(u_xx + u_yy)",
        "domain_x": np.array([0.0, 1.0]),
        "domain_y": np.array([0.0, 1.0]),
        "domain_t": np.array([0.0, 0.5]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Poisson 2D  (analytical)
#    -Δu = f,  f = 2π²sin(πx)sin(πy),  u=0 on ∂Ω
#    Exact: u = sin(πx)·sin(πy)
# ─────────────────────────────────────────────────────────────────────────────

def _load_poisson_2d(N: int = 64) -> Dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.sin(np.pi * X) * np.sin(np.pi * Y)
    f = 2 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    return {
        "x": x, "y": y,
        "u": u,       # (N, N)
        "f": f,       # forcing
        "pde": "-Delta(u) = f",
        "domain_x": np.array([0.0, 1.0]),
        "domain_y": np.array([0.0, 1.0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Wave 1D  (analytical)
#    u_tt = c²·u_xx,  u(0,t)=u(1,t)=0,  u(x,0)=sin(πx), u_t(x,0)=0
#    Exact: u = sin(πx)·cos(c·π·t)
# ─────────────────────────────────────────────────────────────────────────────

def _load_wave_1d(Nx: int = 64, Nt: int = 64,
                  c: float = 1.0) -> Dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, Nx)
    t = np.linspace(0.0, 1.0, Nt)
    X, T = np.meshgrid(x, t)
    u = np.sin(np.pi * X) * np.cos(c * np.pi * T)
    return {
        "x": x, "t": t, "u": u,
        "c": np.float64(c),
        "pde": "u_tt = c^2 * u_xx",
        "domain_x": np.array([0.0, 1.0]),
        "domain_t": np.array([0.0, 1.0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Kovasznay NS  (analytical, 2D steady)
#    Re=40, lambda = Re/2 - sqrt(Re²/4 + 4π²)
#    u = 1 - exp(λx)·cos(2πy)
#    v = λ/(2π)·exp(λx)·sin(2πy)
#    p = (1 - exp(2λx))/2
# ─────────────────────────────────────────────────────────────────────────────

def _load_kovasznay_ns(N: int = 64, Re: float = 40.0) -> Dict[str, np.ndarray]:
    lam = Re / 2.0 - math.sqrt(Re**2 / 4.0 + 4.0 * math.pi**2)
    x = np.linspace(-0.5, 1.0, N)
    y = np.linspace(-0.5, 1.5, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = 1.0 - np.exp(lam * X) * np.cos(2.0 * math.pi * Y)
    v = lam / (2.0 * math.pi) * np.exp(lam * X) * np.sin(2.0 * math.pi * Y)
    p = 0.5 * (1.0 - np.exp(2.0 * lam * X))
    return {
        "x": x, "y": y,
        "u": u, "v": v, "p": p,
        "Re": np.float64(Re),
        "lambda": np.float64(lam),
        "pde": "Navier-Stokes incompressible (Kovasznay steady)",
        "domain_x": np.array([-0.5, 1.0]),
        "domain_y": np.array([-0.5, 1.5]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Allen-Cahn 1D  (scipy)
#    u_t - ε²·u_xx + 5u³ - 5u = 0
#    u(x,0) = x²·cos(πx),  periodic BC
# ─────────────────────────────────────────────────────────────────────────────

def _load_allen_cahn_1d(Nx: int = 128, Nt: int = 101,
                        eps: float = 0.01) -> Dict[str, np.ndarray]:
    x = np.linspace(-1.0, 1.0, Nx, endpoint=False)
    dx = x[1] - x[0]

    def _rhs(t, u):
        # periodic diffusion
        uxx = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
        return eps**2 * uxx - 5.0*(u**3 - u)

    u0 = x**2 * np.cos(np.pi * x)

    try:
        from scipy.integrate import solve_ivp
        sol = solve_ivp(
            _rhs, [0.0, 1.0], u0,
            t_eval=np.linspace(0, 1, Nt),
            method="Radau", rtol=1e-5, atol=1e-7,
        )
        t_out = sol.t
        u_out = sol.y.T
    except ImportError:
        # No scipy: explicit-Euler FD fallback that actually solves the
        # (nonlinear) Allen-Cahn equation, reusing the same periodic-BC
        # diffusion + cubic-reaction RHS used in the scipy path. Substepped
        # to an explicit-Euler stability bound combining the diffusion CFL
        # (dt <= 0.5*dx^2/eps^2) with a reaction-stiffness bound derived from
        # d/du[-5*(u^3-u)] = -15*u^2+5, using |u|<=u_bound (u=+-1 are stable
        # equilibria bounding the trajectory given |u0|<=1).
        t_out = np.linspace(0.0, 1.0, Nt)
        u = u0.copy()
        u_out = np.zeros((Nt, Nx))
        u_out[0] = u
        u_bound = max(float(np.max(np.abs(u))), 1.0)
        dt_diffusion = 0.5 * dx**2 / eps**2
        dt_reaction = 2.0 / (15.0 * u_bound**2 + 5.0)
        dt_stable = 0.8 * min(dt_diffusion, dt_reaction)
        for i in range(1, Nt):
            t_cur, t_target = t_out[i - 1], t_out[i]
            n_sub = max(1, int(math.ceil((t_target - t_cur) / dt_stable)))
            dt_sub = (t_target - t_cur) / n_sub
            for _ in range(n_sub):
                u = u + dt_sub * _rhs(0.0, u)
            u_out[i] = u

    return {
        "x": x, "t": t_out, "u": u_out,
        "eps": np.float64(eps),
        "pde": "u_t - eps^2*u_xx + 5*u^3 - 5*u = 0",
        "domain_x": np.array([-1.0, 1.0]),
        "domain_t": np.array([0.0, 1.0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8. Helmholtz 2D  (analytical)
#    Δu + k²u = q,  Dirichlet BC
#    Exact: u = sin(a1*π*x)*sin(a2*π*y)
# ─────────────────────────────────────────────────────────────────────────────

def _load_helmholtz_2d(N: int = 64, k: float = 1.0,
                       a1: float = 1.0, a2: float = 1.0) -> Dict[str, np.ndarray]:
    x = np.linspace(0.0, 1.0, N)
    y = np.linspace(0.0, 1.0, N)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.sin(a1 * np.pi * X) * np.sin(a2 * np.pi * Y)
    lam = -(a1**2 + a2**2) * np.pi**2 + k**2
    q = lam * u
    return {
        "x": x, "y": y,
        "u": u, "q": q,
        "k": np.float64(k),
        "pde": "Delta(u) + k^2*u = q",
        "domain_x": np.array([0.0, 1.0]),
        "domain_y": np.array([0.0, 1.0]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Registration
# ─────────────────────────────────────────────────────────────────────────────

DatasetRegistry.register(
    DatasetInfo(
        id="burgers_1d",
        name="Burgers 1D",
        category="physics",
        description="Viscous Burgers equation with shock formation. IC: u=-sin(πx), BC: u=0 at x=±1.",
        fields=["x", "t", "u"],
        tags=["pde", "nonlinear", "shock", "1d", "transient"],
        reference="Raissi et al. 2019 (PINN benchmark)",
    ),
    _load_burgers_1d,
)

DatasetRegistry.register(
    DatasetInfo(
        id="heat_1d",
        name="Heat 1D",
        category="physics",
        description="1D heat equation with analytical solution. Exact: u=sin(πx)·exp(-kπ²t).",
        fields=["x", "t", "u"],
        tags=["pde", "linear", "diffusion", "1d", "transient", "analytical"],
    ),
    _load_heat_1d,
)

DatasetRegistry.register(
    DatasetInfo(
        id="heat_2d",
        name="Heat 2D",
        category="physics",
        description="2D heat equation on unit square with analytical solution.",
        fields=["x", "y", "t", "u"],
        tags=["pde", "linear", "diffusion", "2d", "transient", "analytical"],
    ),
    _load_heat_2d,
)

DatasetRegistry.register(
    DatasetInfo(
        id="poisson_2d",
        name="Poisson 2D",
        category="physics",
        description="2D Poisson equation with exact solution u=sin(πx)sin(πy).",
        fields=["x", "y", "u", "f"],
        tags=["pde", "linear", "elliptic", "2d", "steady", "analytical"],
    ),
    _load_poisson_2d,
)

DatasetRegistry.register(
    DatasetInfo(
        id="wave_1d",
        name="Wave 1D",
        category="physics",
        description="1D wave equation. Exact: u=sin(πx)cos(cπt).",
        fields=["x", "t", "u"],
        tags=["pde", "linear", "wave", "1d", "transient", "analytical"],
    ),
    _load_wave_1d,
)

DatasetRegistry.register(
    DatasetInfo(
        id="kovasznay_ns",
        name="Kovasznay NS (2D steady)",
        category="physics",
        description="2D steady incompressible Navier-Stokes (Kovasznay flow). Analytical benchmark for CFD PINNs.",
        fields=["x", "y", "u", "v", "p"],
        tags=["pde", "navier-stokes", "cfd", "2d", "steady", "analytical"],
        reference="Kovasznay 1948",
    ),
    _load_kovasznay_ns,
)

DatasetRegistry.register(
    DatasetInfo(
        id="allen_cahn_1d",
        name="Allen-Cahn 1D",
        category="physics",
        description="1D Allen-Cahn phase-field equation (stiff, sharp interface).",
        fields=["x", "t", "u"],
        tags=["pde", "phase-field", "stiff", "1d", "transient"],
        reference="Wight & Zhao 2020",
    ),
    _load_allen_cahn_1d,
)

DatasetRegistry.register(
    DatasetInfo(
        id="helmholtz_2d",
        name="Helmholtz 2D",
        category="physics",
        description="2D Helmholtz equation Δu + k²u = q with analytical solution.",
        fields=["x", "y", "u", "q"],
        tags=["pde", "helmholtz", "acoustics", "2d", "steady", "analytical"],
    ),
    _load_helmholtz_2d,
)
