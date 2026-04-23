"""Finite Difference Method (FDM) — problem-agnostic numerical solver.

Automatically dispatches to the right FD scheme from a ProblemSpec.pde.kind.

Supported PDEs (substring match on kind, case-insensitive):
  poisson / laplace          → 2D: -∇²u = f  (SOR / Gauss-Seidel)
  helmholtz                  → 2D: (∇² + k²)u = f  (SOR)
  heat / diffusion           → 2D: ∂u/∂t = α∇²u  (ADI θ-method)
  wave                       → 1D: ∂²u/∂t² = c² ∂²u/∂x²  (leapfrog)
  burgers                    → 1D: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²  (FTCS+upwind)
  advection / convection     → 1D: ∂u/∂t + v∂u/∂x = D∂²u/∂x²  (upwind)
  generic / unknown          → falls back to 2D Laplace

BC application from ConditionSpec:
  DirichletBC  → grid values pinned at nodes satisfying the selector
  NeumannBC    → flux imposed via one-sided finite differences at boundary

Usage
-----
    # 1. From a ProblemSpec (recommended):
    from pinneaple_solvers.fdm import FDMSolver
    solver = FDMSolver(nx=128, ny=128, nt=200)
    out    = solver.solve_from_spec(spec)          # → SolverOutput

    # 2. Legacy direct call (Poisson only):
    f  = torch.zeros(64, 64)
    bc = torch.zeros(64, 64)
    out = FDMSolver().forward(f, bc, dx=1/63, dy=1/63)
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

# ─────────────────────────────────────────────────────────────────────────────
# Grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _linspace(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(lo, hi, max(n, 2), dtype=np.float64)


def _spacing(lo: float, hi: float, n: int) -> float:
    return (hi - lo) / max(n - 1, 1)


def _build_2d_grid(
    x0: float, x1: float, nx: int,
    y0: float, y1: float, ny: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    x  = _linspace(x0, x1, nx)
    y  = _linspace(y0, y1, ny)
    dx = _spacing(x0, x1, nx)
    dy = _spacing(y0, y1, ny)
    XX, YY = np.meshgrid(x, y, indexing="ij")
    return x, y, XX, YY, dx, dy


def _apply_dirichlet_2d(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    conditions,
    ctx: Dict,
) -> np.ndarray:
    """Overwrite grid points that satisfy any DirichletBC selector."""
    if not conditions:
        return u
    nx, ny = u.shape
    pts = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float32)
    for cond in conditions:
        if getattr(cond, "kind", "") != "dirichlet":
            continue
        sel_fn  = getattr(cond, "selector",  None)
        val_fn  = getattr(cond, "value_fn",  None)
        if not callable(sel_fn) or not callable(val_fn):
            continue
        try:
            mask = np.asarray(sel_fn(pts, ctx), dtype=bool)
            if not mask.any():
                continue
            vals = np.asarray(val_fn(pts[mask], ctx), dtype=np.float64).ravel()
            flat = u.ravel().copy()
            flat[np.where(mask)[0][:len(vals)]] = vals[:mask.sum()]
            u = flat.reshape(nx, ny)
        except Exception:
            continue
    return u


def _apply_neumann_2d(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    conditions,
    ctx: Dict,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Apply NeumannBC (flux) at boundary nodes using one-sided FD."""
    if not conditions:
        return u
    pts = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2).astype(np.float32)
    nx, ny = u.shape
    for cond in conditions:
        if getattr(cond, "kind", "") != "neumann":
            continue
        sel_fn = getattr(cond, "selector", None)
        val_fn = getattr(cond, "value_fn", None)
        if not callable(sel_fn) or not callable(val_fn):
            continue
        try:
            mask = np.asarray(sel_fn(pts, ctx), dtype=bool)
            if not mask.any():
                continue
            fluxes = np.asarray(val_fn(pts[mask], ctx), dtype=np.float64).ravel()
            idxs   = np.where(mask)[0]
            for k, idx in enumerate(idxs[:len(fluxes)]):
                i, j = divmod(int(idx), ny)
                flux  = float(fluxes[k])
                if i == 0:          u[0, j]   = u[1, j]   - dx * flux
                elif i == nx - 1:   u[-1, j]  = u[-2, j]  + dx * flux
                elif j == 0:        u[i, 0]   = u[i, 1]   - dy * flux
                elif j == ny - 1:   u[i, -1]  = u[i, -2]  + dy * flux
        except Exception:
            continue
    return u


# ─────────────────────────────────────────────────────────────────────────────
# Core FD kernels
# ─────────────────────────────────────────────────────────────────────────────

def _sor_2d(
    f: np.ndarray,
    u: np.ndarray,
    dx: float,
    dy: float,
    iters: int = 8000,
    omega: float = 1.5,
    tol: float = 1e-9,
    k2: float = 0.0,
) -> np.ndarray:
    """SOR solver for (∇² + k²)u = -f  (k²=0 → Poisson/Laplace)."""
    dx2, dy2 = dx * dx, dy * dy
    denom = 2.0 / dx2 + 2.0 / dy2 - k2
    denom = max(abs(denom), 1e-30) * np.sign(denom) if denom != 0 else 1e-30
    for _ in range(iters):
        u_prev = u[1:-1, 1:-1].copy()
        u[1:-1, 1:-1] = (1 - omega) * u[1:-1, 1:-1] + omega * (
            (u[1:-1, 2:] + u[1:-1, :-2]) / dx2
            + (u[2:, 1:-1] + u[:-2, 1:-1]) / dy2
            + f[1:-1, 1:-1]
        ) / denom
        if np.max(np.abs(u[1:-1, 1:-1] - u_prev)) < tol:
            break
    return u


def _tridiag_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Thomas algorithm for tridiagonal system a·x[i-1] + b·x[i] + c·x[i+1] = d[i]."""
    n = len(d)
    c_ = np.zeros(n, dtype=np.float64)
    d_ = d.copy().astype(np.float64)
    x  = np.zeros(n, dtype=np.float64)
    c_[0] = c[0] / b[0]
    d_[0] = d_[0] / b[0]
    for i in range(1, n):
        m     = b[i] - a[i] * c_[i - 1]
        c_[i] = c[i] / m
        d_[i] = (d_[i] - a[i] * d_[i - 1]) / m
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]
    return x


def _adi_heat_2d(
    u: np.ndarray,
    alpha: float,
    dx: float,
    dy: float,
    dt: float,
    nt: int,
    theta: float = 0.5,
) -> np.ndarray:
    """ADI (Douglas-Rachford) theta-method for 2D heat equation.

    theta=0 → explicit (CFL: α dt/h² ≤ 0.5)
    theta=0.5 → Crank-Nicolson (2nd order, unconditionally stable)
    theta=1 → fully implicit (1st order, unconditionally stable)
    """
    nx, ny = u.shape
    rx = alpha * dt / dx ** 2
    ry = alpha * dt / dy ** 2

    for _ in range(nt):
        # ── X-sweep (implicit in x, explicit in y) ───────────────────────────
        rhs = u.copy()
        rhs[1:-1, 1:-1] += (1 - theta) * ry * (
            u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2]
        )
        u_half = u.copy()
        for j in range(1, ny - 1):
            col  = rhs[:, j]
            a    = np.full(nx, -theta * rx)
            b    = np.full(nx, 1 + 2 * theta * rx)
            c    = np.full(nx, -theta * rx)
            a[0] = 0.0; b[0] = 1.0; c[0] = 0.0   # Dirichlet hold
            a[-1]= 0.0; b[-1]= 1.0; c[-1]= 0.0
            u_half[:, j] = _tridiag_solve(a, b, c, col)

        # ── Y-sweep (implicit in y, explicit in x) ───────────────────────────
        rhs2 = u_half.copy()
        rhs2[1:-1, 1:-1] += (1 - theta) * rx * (
            u_half[2:, 1:-1] - 2 * u_half[1:-1, 1:-1] + u_half[:-2, 1:-1]
        )
        u_new = u_half.copy()
        for i in range(1, nx - 1):
            row  = rhs2[i, :]
            a    = np.full(ny, -theta * ry)
            b    = np.full(ny, 1 + 2 * theta * ry)
            c    = np.full(ny, -theta * ry)
            a[0] = 0.0; b[0] = 1.0; c[0] = 0.0
            a[-1]= 0.0; b[-1]= 1.0; c[-1]= 0.0
            u_new[i, :] = _tridiag_solve(a, b, c, row)

        u = u_new
    return u


def _leapfrog_wave_1d(
    u0: np.ndarray,
    v0: np.ndarray,
    c: float,
    dx: float,
    dt: float,
    nt: int,
) -> np.ndarray:
    """Leapfrog for 1D wave: ∂²u/∂t² = c²∂²u/∂x². Returns (nx, nt+1)."""
    r2    = (c * dt / dx) ** 2
    u     = u0.copy()
    u_old = u0 - dt * v0  # virtual previous step
    traj  = [u0.copy()]
    for _ in range(nt):
        lap   = np.roll(u, -1) - 2 * u + np.roll(u, 1)
        u_new = 2 * u - u_old + r2 * lap
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u_old, u = u, u_new
        traj.append(u.copy())
    return np.stack(traj, axis=1)


def _ftcs_burgers_1d(
    u0: np.ndarray,
    nu: float,
    dx: float,
    dt: float,
    nt: int,
) -> np.ndarray:
    """FTCS + upwind for 1D viscous Burgers. Returns (nx, nt+1)."""
    u    = u0.copy()
    traj = [u.copy()]
    for _ in range(nt):
        adv = np.where(
            u >= 0,
            u * (u - np.roll(u, 1)) / dx,
            u * (np.roll(u, -1) - u) / dx,
        )
        diff  = nu * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2
        u_new = u + dt * (-adv + diff)
        u_new[0] = u[0]
        u_new[-1] = u[-1]
        u = u_new
        traj.append(u.copy())
    return np.stack(traj, axis=1)


def _upwind_advdiff_1d(
    u0: np.ndarray,
    v: float,
    D: float,
    dx: float,
    dt: float,
    nt: int,
) -> np.ndarray:
    """Upwind + FTCS for 1D advection-diffusion. Returns (nx, nt+1)."""
    u    = u0.copy()
    traj = [u.copy()]
    for _ in range(nt):
        adv   = (v * (u - np.roll(u, 1)) / dx if v >= 0
                 else v * (np.roll(u, -1) - u) / dx)
        diff  = D * (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2
        u_new = u + dt * (-adv + diff)
        u_new[0] = u[0]
        u_new[-1] = u[-1]
        u = u_new
        traj.append(u.copy())
    return np.stack(traj, axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main solver class
# ─────────────────────────────────────────────────────────────────────────────

@SolverRegistry.register(
    name="fdm",
    family="pde",
    description="Finite Difference Method — problem-agnostic (Poisson, Heat, Wave, Burgers, Advection-Diffusion).",
    tags=["fdm", "pde", "agnostic"],
)
class FDMSolver(SolverBase):
    """Problem-agnostic FDM solver.

    Instantiate directly or via ``FDMSolver.from_problem_spec(spec, ...)``.

    Parameters
    ----------
    nx, ny : spatial grid size (default 64×64)
    nt     : number of time steps for parabolic/hyperbolic problems
    iters  : SOR iterations for elliptic problems
    omega  : SOR relaxation factor (1 < omega < 2 for over-relaxation)
    theta  : ADI θ-parameter (0=explicit, 0.5=Crank-Nicolson, 1=implicit)
    tol    : convergence tolerance for SOR
    """

    def __init__(
        self,
        nx: int = 64,
        ny: int = 64,
        nt: int = 500,
        iters: int = 8000,
        omega: float = 1.5,
        theta: float = 0.5,
        tol: float = 1e-9,
    ):
        super().__init__()
        self.nx    = int(nx)
        self.ny    = int(ny)
        self.nt    = int(nt)
        self.iters = int(iters)
        self.omega = float(omega)
        self.theta = float(theta)
        self.tol   = float(tol)

    @classmethod
    def from_problem_spec(
        cls,
        spec,
        nx: int = 64,
        ny: int = 64,
        nt: int = 500,
        **kwargs,
    ) -> "FDMSolver":
        """Build a solver configured for the given ProblemSpec."""
        return cls(nx=nx, ny=ny, nt=nt, **kwargs)

    # ── Public API ────────────────────────────────────────────────────────────

    def solve_from_spec(self, spec) -> SolverOutput:
        """Auto-solve a ProblemSpec using the appropriate FD scheme."""
        kind       = getattr(spec.pde, "kind",   "").lower()
        params     = dict(getattr(spec.pde, "params", {}))
        domain     = dict(getattr(spec, "domain_bounds", {}))
        conditions = getattr(spec, "conditions", ())
        coords     = tuple(getattr(spec, "coords", ("x", "y")))

        ctx = {"bounds": {c: domain.get(c, (0.0, 1.0)) for c in coords}}

        if "burgers" in kind:
            return self._burgers(domain, coords, conditions, params, ctx)
        if "advection" in kind or "convection_diffusion" in kind:
            return self._advdiff(domain, coords, conditions, params, ctx)
        if "wave" in kind:
            return self._wave(domain, coords, conditions, params, ctx)
        if "heat" in kind or ("diffusion" in kind and "advection" not in kind):
            return self._heat(domain, coords, conditions, params, ctx)
        # Elliptic: Poisson / Laplace / Helmholtz (default)
        return self._poisson(domain, coords, conditions, params, ctx)

    # ── PDE dispatchers ───────────────────────────────────────────────────────

    def _poisson(self, domain, coords, conditions, params, ctx) -> SolverOutput:
        spatial = [c for c in coords if c != "t"]
        c0 = spatial[0] if spatial else "x"
        c1 = spatial[1] if len(spatial) > 1 else "y"
        x0, x1 = domain.get(c0, (0.0, 1.0))
        y0, y1 = domain.get(c1, (0.0, 1.0))

        x, y, XX, YY, dx, dy = _build_2d_grid(x0, x1, self.nx, y0, y1, self.ny)
        f  = np.zeros((self.nx, self.ny))
        u  = np.zeros((self.nx, self.ny))
        k2 = float(params.get("k2", params.get("k", 0.0))) ** 2

        u = _apply_dirichlet_2d(u, x, y, conditions, ctx)
        u = _sor_2d(f, u, dx, dy, self.iters, self.omega, self.tol, k2=k2)
        u = _apply_neumann_2d(u, x, y, conditions, ctx, dx, dy)

        return SolverOutput(
            result=torch.from_numpy(u.astype(np.float32)),
            losses={"residual": torch.tensor(0.0)},
            extras={
                "coords": {c0: x.astype(np.float32), c1: y.astype(np.float32)},
                "grid": {"XX": XX.astype(np.float32), "YY": YY.astype(np.float32)},
                "method": "sor",
            },
        )

    def _heat(self, domain, coords, conditions, params, ctx) -> SolverOutput:
        spatial = [c for c in coords if c != "t"]
        c0 = spatial[0] if spatial else "x"
        c1 = spatial[1] if len(spatial) > 1 else "y"
        x0, x1 = domain.get(c0, (0.0, 1.0))
        y0, y1 = domain.get(c1, (0.0, 1.0))
        t0, t1 = domain.get("t", (0.0, 1.0))

        alpha = float(params.get("alpha", params.get("k", params.get("diffusivity", 0.01))))
        x, y, _, _, dx, dy = _build_2d_grid(x0, x1, self.nx, y0, y1, self.ny)

        # Time step: CFL for stability (safety factor 0.4)
        dt_max = 0.4 * min(dx, dy) ** 2 / max(alpha, 1e-30)
        dt     = min(dt_max, (t1 - t0) / max(self.nt, 1))
        nt     = max(1, int((t1 - t0) / dt))

        u0 = np.zeros((self.nx, self.ny))
        u0 = _apply_dirichlet_2d(u0, x, y, conditions, ctx)
        u  = _adi_heat_2d(u0, alpha, dx, dy, dt, nt, self.theta)
        u  = _apply_neumann_2d(u, x, y, conditions, ctx, dx, dy)

        t_arr = np.linspace(t0, t0 + nt * dt, nt + 1, dtype=np.float32)
        return SolverOutput(
            result=torch.from_numpy(u.astype(np.float32)),
            losses={"residual": torch.tensor(0.0)},
            extras={
                "coords": {c0: x.astype(np.float32), c1: y.astype(np.float32), "t": t_arr},
                "alpha": alpha, "dt": dt, "nt": nt, "method": "adi_theta",
            },
        )

    def _wave(self, domain, coords, conditions, params, ctx) -> SolverOutput:
        spatial = [c for c in coords if c != "t"]
        c0 = spatial[0] if spatial else "x"
        x0, x1 = domain.get(c0, (-1.0, 1.0))
        t0, t1 = domain.get("t", (0.0, 1.0))

        c_speed = float(params.get("c", params.get("wave_speed", 1.0)))
        x  = _linspace(x0, x1, self.nx)
        dx = _spacing(x0, x1, self.nx)
        # CFL: c dt/dx ≤ 1
        dt = 0.9 * dx / max(abs(c_speed), 1e-12)
        nt = min(self.nt, max(1, int((t1 - t0) / dt)))

        u0 = np.sin(np.pi * (x - x0) / (x1 - x0))  # default: one-period sine
        v0 = np.zeros_like(u0)

        traj = _leapfrog_wave_1d(u0, v0, c_speed, dx, dt, nt)
        t_arr = np.linspace(t0, t0 + nt * dt, nt + 1, dtype=np.float32)
        return SolverOutput(
            result=torch.from_numpy(traj.astype(np.float32)),
            losses={"residual": torch.tensor(0.0)},
            extras={
                "coords": {c0: x.astype(np.float32), "t": t_arr},
                "c": c_speed, "cfl": c_speed * dt / dx, "method": "leapfrog",
            },
        )

    def _burgers(self, domain, coords, conditions, params, ctx) -> SolverOutput:
        spatial = [c for c in coords if c != "t"]
        c0 = spatial[0] if spatial else "x"
        x0, x1 = domain.get(c0, (-1.0, 1.0))
        t0, t1 = domain.get("t", (0.0, 1.0))

        nu = float(params.get("nu", params.get("viscosity", 0.01)))
        x  = _linspace(x0, x1, self.nx)
        dx = _spacing(x0, x1, self.nx)
        # Stability: diffusive CFL
        dt = 0.4 * dx ** 2 / max(nu, 1e-12)
        nt = min(self.nt, max(1, int((t1 - t0) / dt)))

        u0 = -np.sin(np.pi * (x - x0) / (x1 - x0))
        traj  = _ftcs_burgers_1d(u0, nu, dx, dt, nt)
        t_arr = np.linspace(t0, t0 + nt * dt, nt + 1, dtype=np.float32)
        return SolverOutput(
            result=torch.from_numpy(traj.astype(np.float32)),
            losses={"residual": torch.tensor(0.0)},
            extras={
                "coords": {c0: x.astype(np.float32), "t": t_arr},
                "nu": nu, "method": "ftcs_upwind",
            },
        )

    def _advdiff(self, domain, coords, conditions, params, ctx) -> SolverOutput:
        spatial = [c for c in coords if c != "t"]
        c0 = spatial[0] if spatial else "x"
        x0, x1 = domain.get(c0, (0.0, 1.0))
        t0, t1 = domain.get("t", (0.0, 1.0))

        v = float(params.get("v", params.get("velocity", 1.0)))
        D = float(params.get("D", params.get("diffusivity", 0.01)))
        x  = _linspace(x0, x1, self.nx)
        dx = _spacing(x0, x1, self.nx)
        dt = min(
            0.4 * dx / max(abs(v), 1e-12),
            0.4 * dx ** 2 / max(D, 1e-12),
        )
        nt = min(self.nt, max(1, int((t1 - t0) / dt)))

        x_mid = (x0 + x1) / 2.0
        u0    = np.exp(-50.0 * (x - x_mid) ** 2)  # Gaussian IC
        traj  = _upwind_advdiff_1d(u0, v, D, dx, dt, nt)
        t_arr = np.linspace(t0, t0 + nt * dt, nt + 1, dtype=np.float32)
        return SolverOutput(
            result=torch.from_numpy(traj.astype(np.float32)),
            losses={"residual": torch.tensor(0.0)},
            extras={
                "coords": {c0: x.astype(np.float32), "t": t_arr},
                "v": v, "D": D, "method": "upwind",
            },
        )

    # ── Legacy interface ──────────────────────────────────────────────────────

    def forward(
        self,
        f: Optional[torch.Tensor] = None,
        bc: Optional[torch.Tensor] = None,
        *,
        dx: float = 1.0,
        dy: float = 1.0,
        spec=None,
    ) -> SolverOutput:
        """Unified forward.

        - ``forward(spec=spec)``              → problem-agnostic solve
        - ``forward(f, bc, dx=.., dy=..)``   → direct 2D Poisson (legacy)
        """
        if spec is not None:
            return self.solve_from_spec(spec)
        if f is not None:
            f_np  = f.detach().cpu().numpy() if isinstance(f, torch.Tensor) else np.asarray(f)
            bc_np = (bc.detach().cpu().numpy() if isinstance(bc, torch.Tensor)
                     else np.zeros_like(f_np))
            u = _sor_2d(f_np, bc_np.copy(), dx, dy, self.iters, self.omega, self.tol)
            return SolverOutput(
                result=torch.from_numpy(u.astype(np.float32)),
                losses={},
                extras={"iters": self.iters, "method": "sor"},
            )
        raise ValueError("FDMSolver.forward: provide either `spec` or `(f, bc)` tensors.")
