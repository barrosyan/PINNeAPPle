"""3D Finite Difference Method (FDM) solvers.

Provides native 3D solvers for:

- ``HeatConduction3D``  — 3D heat equation (explicit Euler + central differences)
- ``NavierStokes3D``    — 3D incompressible NS via fractional-step projection
- ``ElasticWave3D``     — 3D elastic wave equation via Verlet integration

All solvers operate on numpy arrays and return a ``SolverOutput3D`` with the
full field history plus coordinate grids.

Quick start
-----------
>>> from pinneaple_solvers.fdm3d import HeatConduction3D, HeatConfig3D
>>> cfg = HeatConfig3D(nx=32, ny=32, nz=32, nt=50, alpha=1e-4)
>>> solver = HeatConduction3D(cfg)
>>> out = solver.solve()
>>> out.u.shape   # (nt+1, nx, ny, nz)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Common output type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SolverOutput3D:
    """Output from a 3D FDM solver.

    Attributes
    ----------
    u : np.ndarray
        Primary field(s).  Shape and meaning depend on the solver:
        - HeatConduction3D  → (nt+1, nx, ny, nz)  temperature
        - NavierStokes3D    → (nt+1, 3, nx, ny, nz)  velocity [vx, vy, vz]
        - ElasticWave3D     → (nt+1, nx, ny, nz)  displacement
    coords : dict
        ``"x"``, ``"y"``, ``"z"`` 1-D arrays; ``"t"`` time array.
    meta : dict
        Solver parameters and diagnostics (CFL, max_div, etc.).
    """
    u: np.ndarray
    coords: Dict[str, np.ndarray]
    meta: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Helper: finite-difference Laplacian (periodic or Dirichlet zero at boundaries)
# ──────────────────────────────────────────────────────────────────────────────

def _laplacian3d(u: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Second-order central-difference Laplacian, Dirichlet-zero BCs."""
    lap = (
        (np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dx**2
        + (np.roll(u, -1, axis=1) - 2 * u + np.roll(u, 1, axis=1)) / dy**2
        + (np.roll(u, -1, axis=2) - 2 * u + np.roll(u, 1, axis=2)) / dz**2
    )
    # Enforce zero-Dirichlet on all 6 faces
    lap[0, :, :] = 0.0;  lap[-1, :, :] = 0.0
    lap[:, 0, :] = 0.0;  lap[:, -1, :] = 0.0
    lap[:, :, 0] = 0.0;  lap[:, :, -1] = 0.0
    return lap


def _divergence3d(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
    dx: float, dy: float, dz: float,
) -> np.ndarray:
    dvx = (np.roll(vx, -1, axis=0) - np.roll(vx, 1, axis=0)) / (2 * dx)
    dvy = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / (2 * dy)
    dvz = (np.roll(vz, -1, axis=2) - np.roll(vz, 1, axis=2)) / (2 * dz)
    return dvx + dvy + dvz


def _pressure_poisson(
    rhs: np.ndarray,
    dx: float, dy: float, dz: float,
    iterations: int = 50,
) -> np.ndarray:
    """Gauss-Seidel pressure Poisson solver (periodic BCs)."""
    p = np.zeros_like(rhs)
    dx2, dy2, dz2 = dx**2, dy**2, dz**2
    denom = 2.0 * (1 / dx2 + 1 / dy2 + 1 / dz2)
    for _ in range(iterations):
        px = np.roll(p, -1, axis=0) + np.roll(p, 1, axis=0)
        py = np.roll(p, -1, axis=1) + np.roll(p, 1, axis=1)
        pz = np.roll(p, -1, axis=2) + np.roll(p, 1, axis=2)
        p = (px / dx2 + py / dy2 + pz / dz2 - rhs) / denom
    return p


# ──────────────────────────────────────────────────────────────────────────────
# 1. Heat Conduction (3D)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class HeatConfig3D:
    """Configuration for 3D heat conduction solver.

    Parameters
    ----------
    nx, ny, nz : int
        Grid points along each axis.
    nt : int
        Number of time steps.
    lx, ly, lz : float
        Domain size along each axis.
    dt : float
        Time step.  Should satisfy ``dt < dx²/(6α)`` for stability.
    alpha : float
        Thermal diffusivity [m²/s].
    ic_fn : callable, optional
        Initial condition ``u0(X, Y, Z) → ndarray``.  If None, a Gaussian
        blob centred in the domain is used.
    source_fn : callable, optional
        Volumetric heat source ``q(X, Y, Z, t) → ndarray``.
    """
    nx: int = 32
    ny: int = 32
    nz: int = 32
    nt: int = 100
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0
    dt: float = 1e-4
    alpha: float = 1e-4
    ic_fn: Optional[Callable] = None
    source_fn: Optional[Callable] = None


class HeatConduction3D:
    """3D heat equation solver using explicit Euler + central differences.

    ∂T/∂t = α ∇²T + q(x,y,z,t)

    Boundary conditions: Dirichlet zero on all 6 faces (T=0 at walls).
    """

    def __init__(self, cfg: Optional[HeatConfig3D] = None):
        self.cfg = cfg or HeatConfig3D()

    def solve(self) -> SolverOutput3D:
        cfg = self.cfg
        x = np.linspace(0.0, cfg.lx, cfg.nx)
        y = np.linspace(0.0, cfg.ly, cfg.ny)
        z = np.linspace(0.0, cfg.lz, cfg.nz)
        t = np.linspace(0.0, cfg.nt * cfg.dt, cfg.nt + 1)

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Stability check
        r = cfg.alpha * cfg.dt * (1 / dx**2 + 1 / dy**2 + 1 / dz**2)
        if r > 0.5:
            import warnings
            warnings.warn(
                f"HeatConduction3D: CFL-like ratio r={r:.3f} > 0.5, solution may be unstable. "
                f"Reduce dt or increase grid spacing.",
                stacklevel=2,
            )

        # Initial condition
        if cfg.ic_fn is not None:
            u = np.asarray(cfg.ic_fn(X, Y, Z), dtype=np.float64)
        else:
            cx, cy, cz = cfg.lx / 2, cfg.ly / 2, cfg.lz / 2
            sigma = min(cfg.lx, cfg.ly, cfg.lz) / 5.0
            u = np.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * sigma**2))

        history = np.empty((cfg.nt + 1, cfg.nx, cfg.ny, cfg.nz), dtype=np.float64)
        history[0] = u

        for n in range(cfg.nt):
            lap = _laplacian3d(u, dx, dy, dz)
            source = 0.0
            if cfg.source_fn is not None:
                source = cfg.source_fn(X, Y, Z, t[n])
            u = u + cfg.dt * (cfg.alpha * lap + source)
            # Enforce Dirichlet-zero BCs
            u[0, :, :] = 0.0;  u[-1, :, :] = 0.0
            u[:, 0, :] = 0.0;  u[:, -1, :] = 0.0
            u[:, :, 0] = 0.0;  u[:, :, -1] = 0.0
            history[n + 1] = u

        return SolverOutput3D(
            u=history,
            coords={"x": x, "y": y, "z": z, "t": t},
            meta={"alpha": cfg.alpha, "cfl_r": float(r), "solver": "HeatConduction3D"},
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Navier-Stokes (3D, incompressible, projection method)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NavierStokesConfig3D:
    """Configuration for 3D incompressible Navier-Stokes solver.

    Uses a fractional-step (projection) method:
      1. Intermediate velocity: u* = u + dt[−(u·∇)u + ν∇²u + f]
      2. Pressure Poisson:      ∇²p = ρ/dt ∇·u*
      3. Velocity correction:   u = u* − dt/ρ ∇p

    Parameters
    ----------
    nx, ny, nz : int
        Grid points.
    nt : int
        Number of time steps.
    lx, ly, lz : float
        Domain size.
    dt : float
        Time step.
    nu : float
        Kinematic viscosity.
    rho : float
        Fluid density.
    pressure_iters : int
        Gauss-Seidel iterations for the Poisson solve.
    ic_fn : callable, optional
        Initial velocity ``(vx0, vy0, vz0) = ic_fn(X, Y, Z)``.
    force_fn : callable, optional
        Body force ``(fx, fy, fz) = force_fn(X, Y, Z, t)``.
    """
    nx: int = 24
    ny: int = 24
    nz: int = 24
    nt: int = 50
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0
    dt: float = 1e-3
    nu: float = 1e-3
    rho: float = 1.0
    pressure_iters: int = 50
    ic_fn: Optional[Callable] = None
    force_fn: Optional[Callable] = None


class NavierStokes3D:
    """3D incompressible Navier-Stokes via fractional-step (projection) method.

    Output ``u`` has shape ``(nt+1, 3, nx, ny, nz)`` where axis 1 is
    [vx, vy, vz].  Pressure is available in ``meta["p_final"]``.
    """

    def __init__(self, cfg: Optional[NavierStokesConfig3D] = None):
        self.cfg = cfg or NavierStokesConfig3D()

    def _advect(self, v: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                dx: float, dy: float, dz: float) -> np.ndarray:
        """First-order upwind advection of field v."""
        dvdx = np.where(
            vx > 0,
            (v - np.roll(v, 1, axis=0)) / dx,
            (np.roll(v, -1, axis=0) - v) / dx,
        )
        dvdy = np.where(
            vy > 0,
            (v - np.roll(v, 1, axis=1)) / dy,
            (np.roll(v, -1, axis=1) - v) / dy,
        )
        dvdz = np.where(
            vz > 0,
            (v - np.roll(v, 1, axis=2)) / dz,
            (np.roll(v, -1, axis=2) - v) / dz,
        )
        return vx * dvdx + vy * dvdy + vz * dvdz

    def solve(self) -> SolverOutput3D:
        cfg = self.cfg
        x = np.linspace(0.0, cfg.lx, cfg.nx)
        y = np.linspace(0.0, cfg.ly, cfg.ny)
        z = np.linspace(0.0, cfg.lz, cfg.nz)
        t = np.linspace(0.0, cfg.nt * cfg.dt, cfg.nt + 1)
        dx = x[1] - x[0] if cfg.nx > 1 else cfg.lx
        dy = y[1] - y[0] if cfg.ny > 1 else cfg.ly
        dz = z[1] - z[0] if cfg.nz > 1 else cfg.lz

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        if cfg.ic_fn is not None:
            vx, vy, vz = cfg.ic_fn(X, Y, Z)
            vx = np.asarray(vx, dtype=np.float64)
            vy = np.asarray(vy, dtype=np.float64)
            vz = np.asarray(vz, dtype=np.float64)
        else:
            vx = np.zeros((cfg.nx, cfg.ny, cfg.nz))
            vy = np.zeros((cfg.nx, cfg.ny, cfg.nz))
            vz = np.zeros((cfg.nx, cfg.ny, cfg.nz))

        p = np.zeros((cfg.nx, cfg.ny, cfg.nz))

        history = np.empty((cfg.nt + 1, 3, cfg.nx, cfg.ny, cfg.nz))
        history[0, 0] = vx;  history[0, 1] = vy;  history[0, 2] = vz

        max_div = []

        for n in range(cfg.nt):
            fx = fy = fz = 0.0
            if cfg.force_fn is not None:
                fx, fy, fz = cfg.force_fn(X, Y, Z, t[n])

            # Step 1: intermediate velocity (advection + diffusion + forcing)
            lap_vx = _laplacian3d(vx, dx, dy, dz)
            lap_vy = _laplacian3d(vy, dx, dy, dz)
            lap_vz = _laplacian3d(vz, dx, dy, dz)

            vx_s = vx + cfg.dt * (-self._advect(vx, vx, vy, vz, dx, dy, dz) + cfg.nu * lap_vx + fx)
            vy_s = vy + cfg.dt * (-self._advect(vy, vx, vy, vz, dx, dy, dz) + cfg.nu * lap_vy + fy)
            vz_s = vz + cfg.dt * (-self._advect(vz, vx, vy, vz, dx, dy, dz) + cfg.nu * lap_vz + fz)

            # Step 2: pressure Poisson
            div_s = _divergence3d(vx_s, vy_s, vz_s, dx, dy, dz)
            rhs = (cfg.rho / cfg.dt) * div_s
            p = _pressure_poisson(rhs, dx, dy, dz, iterations=cfg.pressure_iters)

            # Step 3: velocity correction
            dpdx = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dx)
            dpdy = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dy)
            dpdz = (np.roll(p, -1, axis=2) - np.roll(p, 1, axis=2)) / (2 * dz)
            vx = vx_s - (cfg.dt / cfg.rho) * dpdx
            vy = vy_s - (cfg.dt / cfg.rho) * dpdy
            vz = vz_s - (cfg.dt / cfg.rho) * dpdz

            div_final = _divergence3d(vx, vy, vz, dx, dy, dz)
            max_div.append(float(np.max(np.abs(div_final))))

            history[n + 1, 0] = vx
            history[n + 1, 1] = vy
            history[n + 1, 2] = vz

        return SolverOutput3D(
            u=history,
            coords={"x": x, "y": y, "z": z, "t": t},
            meta={
                "nu": cfg.nu,
                "rho": cfg.rho,
                "max_divergence": max_div,
                "p_final": p,
                "solver": "NavierStokes3D",
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. Elastic Wave (3D)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ElasticWaveConfig3D:
    """Configuration for 3D elastic wave solver.

    Solves the scalar wave equation (acoustic approximation):
      ρ ∂²u/∂t² = (λ+2μ)∇²u

    where c = sqrt((λ+2μ)/ρ) is the P-wave speed.

    Parameters
    ----------
    nx, ny, nz : int
        Grid points.
    nt : int
        Number of time steps.
    lx, ly, lz : float
        Domain size.
    dt : float
        Time step.  Courant stability: dt < dx / (c√3).
    c : float
        Wave speed [m/s].
    ic_u_fn : callable, optional
        Initial displacement ``u0(X, Y, Z)``.
    ic_du_fn : callable, optional
        Initial velocity ``du0(X, Y, Z)``.  Defaults to zero.
    source_fn : callable, optional
        Source term ``f(X, Y, Z, t)``.
    """
    nx: int = 32
    ny: int = 32
    nz: int = 32
    nt: int = 100
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0
    dt: float = 5e-4
    c: float = 1.0
    ic_u_fn: Optional[Callable] = None
    ic_du_fn: Optional[Callable] = None
    source_fn: Optional[Callable] = None


class ElasticWave3D:
    """3D elastic (acoustic) wave equation solver via Störmer-Verlet integration.

    ρ ∂²u/∂t² = c² ∇²u + f(x,y,z,t)

    Uses the leapfrog / Verlet scheme which is 2nd-order in time and energy-
    conserving for the homogeneous equation.

    Output ``u`` has shape ``(nt+1, nx, ny, nz)``.
    """

    def __init__(self, cfg: Optional[ElasticWaveConfig3D] = None):
        self.cfg = cfg or ElasticWaveConfig3D()

    def solve(self) -> SolverOutput3D:
        cfg = self.cfg
        x = np.linspace(0.0, cfg.lx, cfg.nx)
        y = np.linspace(0.0, cfg.ly, cfg.ny)
        z = np.linspace(0.0, cfg.lz, cfg.nz)
        t = np.linspace(0.0, cfg.nt * cfg.dt, cfg.nt + 1)
        dx = x[1] - x[0] if cfg.nx > 1 else cfg.lx
        dy = y[1] - y[0] if cfg.ny > 1 else cfg.ly
        dz = z[1] - z[0] if cfg.nz > 1 else cfg.lz

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # CFL stability check
        cfl = cfg.c * cfg.dt * np.sqrt(1 / dx**2 + 1 / dy**2 + 1 / dz**2)
        if cfl > 1.0:
            import warnings
            warnings.warn(
                f"ElasticWave3D: CFL={cfl:.3f} > 1.0, solution may be unstable. "
                f"Reduce dt or increase grid spacing.",
                stacklevel=2,
            )

        if cfg.ic_u_fn is not None:
            u = np.asarray(cfg.ic_u_fn(X, Y, Z), dtype=np.float64)
        else:
            cx, cy, cz = cfg.lx / 2, cfg.ly / 2, cfg.lz / 2
            sigma = min(cfg.lx, cfg.ly, cfg.lz) / 8.0
            u = np.exp(-((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) / (2 * sigma**2))

        if cfg.ic_du_fn is not None:
            du = np.asarray(cfg.ic_du_fn(X, Y, Z), dtype=np.float64)
        else:
            du = np.zeros_like(u)

        c2 = cfg.c**2

        # Bootstrap: u_prev = u - dt*du + dt²/2 * c²∇²u
        source0 = 0.0
        if cfg.source_fn is not None:
            source0 = cfg.source_fn(X, Y, Z, t[0])
        u_prev = u - cfg.dt * du + 0.5 * cfg.dt**2 * (c2 * _laplacian3d(u, dx, dy, dz) + source0)

        history = np.empty((cfg.nt + 1, cfg.nx, cfg.ny, cfg.nz))
        history[0] = u

        for n in range(cfg.nt):
            source = 0.0
            if cfg.source_fn is not None:
                source = cfg.source_fn(X, Y, Z, t[n])
            lap = _laplacian3d(u, dx, dy, dz)
            u_next = 2 * u - u_prev + cfg.dt**2 * (c2 * lap + source)
            # Dirichlet-zero BCs
            u_next[0, :, :] = 0.0;  u_next[-1, :, :] = 0.0
            u_next[:, 0, :] = 0.0;  u_next[:, -1, :] = 0.0
            u_next[:, :, 0] = 0.0;  u_next[:, :, -1] = 0.0
            u_prev = u
            u = u_next
            history[n + 1] = u

        return SolverOutput3D(
            u=history,
            coords={"x": x, "y": y, "z": z, "t": t},
            meta={"c": cfg.c, "cfl": float(cfl), "solver": "ElasticWave3D"},
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Lid-Driven Cavity (3D)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LidDrivenCavityConfig3D:
    """Configuration for the 3D lid-driven cavity solver.

    Parameters
    ----------
    nx, ny, nz : int
        Grid points in x, y, z.
    nt : int
        Number of time steps.
    size : float
        Cube side length L.  Domain is [0, L]^3.
    dt : float
        Time step.
    nu : float
        Kinematic viscosity (nu = 1/Re when U_lid = L = 1).
    rho : float
        Fluid density.
    lid_velocity : float
        Lid velocity U_lid applied at the z = L face.
    lid_axis : int
        Velocity component driven by the lid: 0 → x-velocity, 1 → y-velocity.
    pressure_iters : int
        Gauss-Seidel iterations for the pressure Poisson solve.
    """
    nx: int = 32
    ny: int = 32
    nz: int = 32
    nt: int = 200
    size: float = 1.0
    dt: float = 5e-4
    nu: float = 1e-2
    rho: float = 1.0
    lid_velocity: float = 1.0
    lid_axis: int = 0
    pressure_iters: int = 50


class LidDrivenCavitySolver3D:
    """3D Lid-Driven Cavity flow solver.

    Geometry: unit cube [0, L]^3.  The lid face at z = L (index nz-1) moves
    with velocity U_lid in the ``lid_axis`` direction (0 = x, 1 = y).  All
    other faces are no-slip walls.

    Algorithm: fractional-step (projection) method identical in structure to
    ``NavierStokes3D``, with explicit BC enforcement before and after the
    pressure-correction step.

    Output ``u`` has shape ``(nt+1, 4, nx, ny, nz)`` where axis 1 is
    [vx, vy, vz, p].
    """

    def __init__(self, cfg: Optional[LidDrivenCavityConfig3D] = None):
        self.cfg = cfg or LidDrivenCavityConfig3D()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _advect(v: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                dx: float, dy: float, dz: float) -> np.ndarray:
        """First-order upwind advection of scalar field v."""
        dvdx = np.where(
            vx > 0,
            (v - np.roll(v, 1, axis=0)) / dx,
            (np.roll(v, -1, axis=0) - v) / dx,
        )
        dvdy = np.where(
            vy > 0,
            (v - np.roll(v, 1, axis=1)) / dy,
            (np.roll(v, -1, axis=1) - v) / dy,
        )
        dvdz = np.where(
            vz > 0,
            (v - np.roll(v, 1, axis=2)) / dz,
            (np.roll(v, -1, axis=2) - v) / dz,
        )
        return vx * dvdx + vy * dvdy + vz * dvdz

    def _apply_bcs(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enforce lid and no-slip boundary conditions in-place."""
        cfg = self.cfg
        U = cfg.lid_velocity

        # Lid face: z = nz-1
        vx[:, :, -1] = U if cfg.lid_axis == 0 else 0.0
        vy[:, :, -1] = U if cfg.lid_axis == 1 else 0.0
        vz[:, :, -1] = 0.0

        # Bottom face: z = 0
        vx[:, :, 0] = 0.0
        vy[:, :, 0] = 0.0
        vz[:, :, 0] = 0.0

        # Left/right faces: x = 0, x = nx-1
        vx[0, :, :] = 0.0;  vy[0, :, :] = 0.0;  vz[0, :, :] = 0.0
        vx[-1, :, :] = 0.0; vy[-1, :, :] = 0.0; vz[-1, :, :] = 0.0

        # Front/back faces: y = 0, y = ny-1
        vx[:, 0, :] = 0.0;  vy[:, 0, :] = 0.0;  vz[:, 0, :] = 0.0
        vx[:, -1, :] = 0.0; vy[:, -1, :] = 0.0; vz[:, -1, :] = 0.0

        return vx, vy, vz

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> SolverOutput3D:
        cfg = self.cfg
        L = cfg.size

        x = np.linspace(0.0, L, cfg.nx)
        y = np.linspace(0.0, L, cfg.ny)
        z = np.linspace(0.0, L, cfg.nz)
        t = np.linspace(0.0, cfg.nt * cfg.dt, cfg.nt + 1)

        dx = x[1] - x[0] if cfg.nx > 1 else L
        dy = y[1] - y[0] if cfg.ny > 1 else L
        dz = z[1] - z[0] if cfg.nz > 1 else L

        # Initialise: zero velocity everywhere
        vx = np.zeros((cfg.nx, cfg.ny, cfg.nz))
        vy = np.zeros((cfg.nx, cfg.ny, cfg.nz))
        vz = np.zeros((cfg.nx, cfg.ny, cfg.nz))
        p  = np.zeros((cfg.nx, cfg.ny, cfg.nz))

        # Enforce BCs on the initial state
        vx, vy, vz = self._apply_bcs(vx, vy, vz)

        # History: axis 1 = [vx, vy, vz, p]
        history = np.empty((cfg.nt + 1, 4, cfg.nx, cfg.ny, cfg.nz), dtype=np.float64)
        history[0, 0] = vx
        history[0, 1] = vy
        history[0, 2] = vz
        history[0, 3] = p

        max_div: list = []

        for n in range(cfg.nt):
            # Step 1: intermediate velocity (advection + diffusion)
            lap_vx = _laplacian3d(vx, dx, dy, dz)
            lap_vy = _laplacian3d(vy, dx, dy, dz)
            lap_vz = _laplacian3d(vz, dx, dy, dz)

            vx_s = vx + cfg.dt * (
                -self._advect(vx, vx, vy, vz, dx, dy, dz) + cfg.nu * lap_vx
            )
            vy_s = vy + cfg.dt * (
                -self._advect(vy, vx, vy, vz, dx, dy, dz) + cfg.nu * lap_vy
            )
            vz_s = vz + cfg.dt * (
                -self._advect(vz, vx, vy, vz, dx, dy, dz) + cfg.nu * lap_vz
            )

            # Apply BCs to intermediate velocity before pressure solve
            vx_s, vy_s, vz_s = self._apply_bcs(vx_s, vy_s, vz_s)

            # Step 2: pressure Poisson (∇²p = ρ/dt ∇·u*)
            div_s = _divergence3d(vx_s, vy_s, vz_s, dx, dy, dz)
            rhs = (cfg.rho / cfg.dt) * div_s
            p = _pressure_poisson(rhs, dx, dy, dz, iterations=cfg.pressure_iters)

            # Dirichlet zero for p at all corners (pin pressure gauge)
            p[0, 0, 0] = 0.0

            # Step 3: velocity correction
            dpdx = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dx)
            dpdy = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dy)
            dpdz = (np.roll(p, -1, axis=2) - np.roll(p, 1, axis=2)) / (2 * dz)

            vx = vx_s - (cfg.dt / cfg.rho) * dpdx
            vy = vy_s - (cfg.dt / cfg.rho) * dpdy
            vz = vz_s - (cfg.dt / cfg.rho) * dpdz

            # Re-enforce BCs after projection (critical for LDC correctness)
            vx, vy, vz = self._apply_bcs(vx, vy, vz)

            div_final = _divergence3d(vx, vy, vz, dx, dy, dz)
            max_div.append(float(np.max(np.abs(div_final))))

            history[n + 1, 0] = vx
            history[n + 1, 1] = vy
            history[n + 1, 2] = vz
            history[n + 1, 3] = p

        Re = L * cfg.lid_velocity / cfg.nu

        return SolverOutput3D(
            u=history,
            coords={"x": x, "y": y, "z": z, "t": t},
            meta={
                "Re": Re,
                "nu": cfg.nu,
                "lid_velocity": cfg.lid_velocity,
                "lid_axis": cfg.lid_axis,
                "p_final": p,
                "solver": "LidDrivenCavitySolver3D",
                "max_divergence": max_div,
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Channel Flow / Pressure-Driven Poiseuille (3D)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChannelFlowConfig3D:
    """Configuration for the 3D pressure-driven channel flow solver.

    Parameters
    ----------
    nx, ny, nz : int
        Grid points in x (streamwise), y (wall-normal), z (spanwise).
    nt : int
        Number of time steps.
    lx : float
        Channel length (treated as periodic / roll in x).
    ly : float
        Channel height (wall-to-wall in y).
    lz : float
        Channel width (wall-to-wall in z).
    dt : float
        Time step.
    nu : float
        Kinematic viscosity.
    rho : float
        Fluid density.
    dp_dx : float
        Constant pressure-gradient body force applied to vx equation.
        Negative value drives flow in the +x direction.
    pressure_iters : int
        Gauss-Seidel iterations for the pressure Poisson solve.
    """
    nx: int = 32
    ny: int = 32
    nz: int = 32
    nt: int = 200
    lx: float = 2.0
    ly: float = 1.0
    lz: float = 1.0
    dt: float = 5e-4
    nu: float = 1e-2
    rho: float = 1.0
    dp_dx: float = -0.1
    pressure_iters: int = 50


class ChannelFlowSolver3D:
    """3D pressure-driven channel (Poiseuille-like) flow solver.

    Geometry: rectangular duct [0, Lx] × [0, Ly] × [0, Lz].
    - No-slip on four walls: y=0, y=Ly, z=0, z=Lz.
    - Streamwise direction (x) is handled with a periodic roll so that the
      domain is effectively infinite in x for this time-stepping scheme.
    - Driving force: constant body force  f_x = -dp_dx / rho  added to the
      x-momentum equation.

    Algorithm: same fractional-step projection method as NavierStokes3D.

    Output ``u`` has shape ``(nt+1, 3, nx, ny, nz)`` where axis 1 is
    [vx, vy, vz].
    """

    def __init__(self, cfg: Optional[ChannelFlowConfig3D] = None):
        self.cfg = cfg or ChannelFlowConfig3D()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _advect(v: np.ndarray, vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                dx: float, dy: float, dz: float) -> np.ndarray:
        """First-order upwind advection of scalar field v."""
        dvdx = np.where(
            vx > 0,
            (v - np.roll(v, 1, axis=0)) / dx,
            (np.roll(v, -1, axis=0) - v) / dx,
        )
        dvdy = np.where(
            vy > 0,
            (v - np.roll(v, 1, axis=1)) / dy,
            (np.roll(v, -1, axis=1) - v) / dy,
        )
        dvdz = np.where(
            vz > 0,
            (v - np.roll(v, 1, axis=2)) / dz,
            (np.roll(v, -1, axis=2) - v) / dz,
        )
        return vx * dvdx + vy * dvdy + vz * dvdz

    def _apply_wall_bcs(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Enforce no-slip on the four y- and z-walls."""
        # y-walls: y=0 (index 0) and y=ny-1
        vx[:, 0, :] = 0.0;  vy[:, 0, :] = 0.0;  vz[:, 0, :] = 0.0
        vx[:, -1, :] = 0.0; vy[:, -1, :] = 0.0; vz[:, -1, :] = 0.0
        # z-walls: z=0 (index 0) and z=nz-1
        vx[:, :, 0] = 0.0;  vy[:, :, 0] = 0.0;  vz[:, :, 0] = 0.0
        vx[:, :, -1] = 0.0; vy[:, :, -1] = 0.0; vz[:, :, -1] = 0.0
        return vx, vy, vz

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self) -> SolverOutput3D:
        cfg = self.cfg

        x = np.linspace(0.0, cfg.lx, cfg.nx)
        y = np.linspace(0.0, cfg.ly, cfg.ny)
        z = np.linspace(0.0, cfg.lz, cfg.nz)
        t = np.linspace(0.0, cfg.nt * cfg.dt, cfg.nt + 1)

        dx = x[1] - x[0] if cfg.nx > 1 else cfg.lx
        dy = y[1] - y[0] if cfg.ny > 1 else cfg.ly
        dz = z[1] - z[0] if cfg.nz > 1 else cfg.lz

        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        # Initialise with a Poiseuille-like approximate profile.
        # vx = dp_dx/(2*nu) * y*(Ly-y)*z*(Lz-z) / (Ly*Lz/4)
        # (will relax to the true Poiseuille profile over time)
        scale = cfg.dp_dx / (2.0 * cfg.nu * (cfg.ly * cfg.lz / 4.0))
        vx = scale * Y * (cfg.ly - Y) * Z * (cfg.lz - Z)
        vy = np.zeros_like(vx)
        vz = np.zeros_like(vx)

        # Enforce wall BCs on the initial state
        vx, vy, vz = self._apply_wall_bcs(vx, vy, vz)

        p = np.zeros((cfg.nx, cfg.ny, cfg.nz))

        history = np.empty((cfg.nt + 1, 3, cfg.nx, cfg.ny, cfg.nz), dtype=np.float64)
        history[0, 0] = vx
        history[0, 1] = vy
        history[0, 2] = vz

        # Constant body force from imposed pressure gradient
        fx_body = -cfg.dp_dx / cfg.rho  # body force per unit mass in +x direction

        for n in range(cfg.nt):
            # Step 1: intermediate velocity (advection + diffusion + body force)
            lap_vx = _laplacian3d(vx, dx, dy, dz)
            lap_vy = _laplacian3d(vy, dx, dy, dz)
            lap_vz = _laplacian3d(vz, dx, dy, dz)

            vx_s = vx + cfg.dt * (
                -self._advect(vx, vx, vy, vz, dx, dy, dz) + cfg.nu * lap_vx + fx_body
            )
            vy_s = vy + cfg.dt * (
                -self._advect(vy, vx, vy, vz, dx, dy, dz) + cfg.nu * lap_vy
            )
            vz_s = vz + cfg.dt * (
                -self._advect(vz, vx, vy, vz, dx, dy, dz) + cfg.nu * lap_vz
            )

            # Periodic roll in x to avoid inlet/outlet boundary complexity
            vx_s = np.roll(vx_s, 1, axis=0)
            vy_s = np.roll(vy_s, 1, axis=0)
            vz_s = np.roll(vz_s, 1, axis=0)

            # Enforce wall BCs on intermediate velocity
            vx_s, vy_s, vz_s = self._apply_wall_bcs(vx_s, vy_s, vz_s)

            # Step 2: pressure Poisson
            div_s = _divergence3d(vx_s, vy_s, vz_s, dx, dy, dz)
            rhs = (cfg.rho / cfg.dt) * div_s
            p = _pressure_poisson(rhs, dx, dy, dz, iterations=cfg.pressure_iters)

            # Step 3: velocity correction
            dpdx = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dx)
            dpdy = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2 * dy)
            dpdz = (np.roll(p, -1, axis=2) - np.roll(p, 1, axis=2)) / (2 * dz)

            vx = vx_s - (cfg.dt / cfg.rho) * dpdx
            vy = vy_s - (cfg.dt / cfg.rho) * dpdy
            vz = vz_s - (cfg.dt / cfg.rho) * dpdz

            # Re-enforce wall BCs after projection
            vx, vy, vz = self._apply_wall_bcs(vx, vy, vz)

            history[n + 1, 0] = vx
            history[n + 1, 1] = vy
            history[n + 1, 2] = vz

        return SolverOutput3D(
            u=history,
            coords={"x": x, "y": y, "z": z, "t": t},
            meta={
                "nu": cfg.nu,
                "rho": cfg.rho,
                "dp_dx": cfg.dp_dx,
                "solver": "ChannelFlowSolver3D",
                "p_final": p,
            },
        )


__all__ = [
    "SolverOutput3D",
    "HeatConfig3D",
    "HeatConduction3D",
    "NavierStokesConfig3D",
    "NavierStokes3D",
    "ElasticWaveConfig3D",
    "ElasticWave3D",
    "LidDrivenCavityConfig3D",
    "LidDrivenCavitySolver3D",
    "ChannelFlowConfig3D",
    "ChannelFlowSolver3D",
]
