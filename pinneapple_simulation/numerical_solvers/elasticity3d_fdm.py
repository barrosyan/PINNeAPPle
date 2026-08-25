"""Static 3D linear (isotropic, small-strain) elasticity via finite
differences: solves the Navier-Cauchy equilibrium equations

    (lambda + mu) * grad(div(u)) + mu * laplacian(u) + f = 0

on a structured rectangular grid, given a body force field f(x,y,z) and
Dirichlet (prescribed displacement) boundary conditions on all six faces,
via damped-Jacobi relaxation to steady state. lambda, mu are the Lame
parameters (computable from Young's modulus E and Poisson's ratio nu via
`lame_parameters`).

This complements this package's dynamic `ElasticWave3D` FDM solver (see
`fdm3d.py`) with the STATIC/quasi-static case (structural equilibrium under
a fixed load, no time dependence) -- generic to any body-force-driven
static elasticity problem: self-weight, thermal-expansion-equivalent body
loads, or any other volumetric loading.

Scope note: this solver currently supports Dirichlet displacement BCs on
the full domain boundary only (no free/traction boundary condition) --
useful for problems where the displacement state is known or approximated
on all six faces (e.g. a subdomain of a larger structure, or a
manufactured/prescribed-boundary test case). A free-surface (zero-traction)
boundary condition is a natural extension not yet implemented here.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry


def lame_parameters(E: float, nu: float) -> Tuple[float, float]:
    """Lame's first parameter (lambda) and shear modulus (mu) from Young's
    modulus E and Poisson's ratio nu."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def solve_static_elasticity_3d(
    lam: float,
    mu: float,
    body_force: np.ndarray,
    u_bc: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    n_iter: int = 20000,
    omega: float = 1.0,
    tol: float = 1e-10,
) -> dict:
    """Solve the static Navier-Cauchy equations on a structured grid via
    damped-Jacobi relaxation.

    body_force: (nx, ny, nz, 3) array, f(x,y,z) at every grid point (only
        interior values are used -- boundary body-force values are ignored
        since displacement there is prescribed by u_bc).
    u_bc: (nx, ny, nz, 3) array giving the prescribed displacement at every
        BOUNDARY grid point (i=0/nx-1, j=0/ny-1, or k=0/nz-1); interior
        values are used only as the initial guess (a physically-reasonable
        initial guess, e.g. linear interpolation of the boundary values,
        speeds convergence but is not required for correctness).
    omega: Jacobi relaxation damping, in (0, 1] (1.0 = plain, undamped
        Jacobi; the default and the fastest stable choice for this
        averaging-based update). Unlike true SOR on a Gauss-Seidel sweep,
        this is a convex blend of the old and newly-computed interior
        values -- omega > 1 (extrapolation) is NOT stable here and will
        diverge; use omega < 1.0 only if 1.0 itself fails to converge for
        an unusually stiff material or fine grid.

    Returns {"u": (nx,ny,nz,3), "n_iter": int, "converged": bool,
    "residual_history": list[float]}.
    """
    u = np.asarray(u_bc, dtype=np.float64).copy()
    f = np.asarray(body_force, dtype=np.float64)
    nx, ny, nz, _ = u.shape

    dx2, dy2, dz2 = dx * dx, dy * dy, dz * dz
    llmu = lam + mu       # coupling coefficient
    l2mu = lam + 2.0 * mu  # normal-strain coefficient

    # Diagonal coefficients for each displacement component's own
    # second-derivative terms (from the componentwise expansion of
    # (lambda+mu)*grad(div u) + mu*laplacian(u)):
    #   u: (lambda+2mu)*u_xx + mu*u_yy + mu*u_zz + (lambda+mu)*(v_xy+w_xz)
    #   v: (lambda+2mu)*v_yy + mu*v_xx + mu*v_zz + (lambda+mu)*(u_xy+w_yz)
    #   w: (lambda+2mu)*w_zz + mu*w_xx + mu*w_yy + (lambda+mu)*(u_xz+v_yz)
    diag_u = 2.0 * (l2mu / dx2 + mu / dy2 + mu / dz2)
    diag_v = 2.0 * (l2mu / dy2 + mu / dx2 + mu / dz2)
    diag_w = 2.0 * (l2mu / dz2 + mu / dx2 + mu / dy2)

    boundary_mask = np.ones((nx, ny, nz), dtype=bool)
    boundary_mask[1:-1, 1:-1, 1:-1] = False  # True on the 6 faces, False in the interior

    residual_history = []
    converged = False
    n_done = 0

    for it in range(1, n_iter + 1):
        U, V, W = u[..., 0], u[..., 1], u[..., 2]

        u_xx = U[2:, 1:-1, 1:-1] + U[:-2, 1:-1, 1:-1]
        u_yy = U[1:-1, 2:, 1:-1] + U[1:-1, :-2, 1:-1]
        u_zz = U[1:-1, 1:-1, 2:] + U[1:-1, 1:-1, :-2]
        v_xy = (V[2:, 2:, 1:-1] - V[2:, :-2, 1:-1] - V[:-2, 2:, 1:-1] + V[:-2, :-2, 1:-1]) / (4.0 * dx * dy)
        w_xz = (W[2:, 1:-1, 2:] - W[2:, 1:-1, :-2] - W[:-2, 1:-1, 2:] + W[:-2, 1:-1, :-2]) / (4.0 * dx * dz)
        u_new_int = (l2mu * u_xx / dx2 + mu * u_yy / dy2 + mu * u_zz / dz2
                     + llmu * (v_xy + w_xz) + f[1:-1, 1:-1, 1:-1, 0]) / diag_u

        v_xx = V[2:, 1:-1, 1:-1] + V[:-2, 1:-1, 1:-1]
        v_yy = V[1:-1, 2:, 1:-1] + V[1:-1, :-2, 1:-1]
        v_zz = V[1:-1, 1:-1, 2:] + V[1:-1, 1:-1, :-2]
        u_xy = (U[2:, 2:, 1:-1] - U[2:, :-2, 1:-1] - U[:-2, 2:, 1:-1] + U[:-2, :-2, 1:-1]) / (4.0 * dx * dy)
        w_yz = (W[1:-1, 2:, 2:] - W[1:-1, 2:, :-2] - W[1:-1, :-2, 2:] + W[1:-1, :-2, :-2]) / (4.0 * dy * dz)
        v_new_int = (l2mu * v_yy / dy2 + mu * v_xx / dx2 + mu * v_zz / dz2
                     + llmu * (u_xy + w_yz) + f[1:-1, 1:-1, 1:-1, 1]) / diag_v

        w_xx = W[2:, 1:-1, 1:-1] + W[:-2, 1:-1, 1:-1]
        w_yy = W[1:-1, 2:, 1:-1] + W[1:-1, :-2, 1:-1]
        w_zz = W[1:-1, 1:-1, 2:] + W[1:-1, 1:-1, :-2]
        u_xz = (U[2:, 1:-1, 2:] - U[2:, 1:-1, :-2] - U[:-2, 1:-1, 2:] + U[:-2, 1:-1, :-2]) / (4.0 * dx * dz)
        v_yz = (V[1:-1, 2:, 2:] - V[1:-1, 2:, :-2] - V[1:-1, :-2, 2:] + V[1:-1, :-2, :-2]) / (4.0 * dy * dz)
        w_new_int = (l2mu * w_zz / dz2 + mu * w_xx / dx2 + mu * w_yy / dy2
                     + llmu * (u_xz + v_yz) + f[1:-1, 1:-1, 1:-1, 2]) / diag_w

        u_old_int = u[1:-1, 1:-1, 1:-1, :].copy()
        u[1:-1, 1:-1, 1:-1, 0] = (1 - omega) * u_old_int[..., 0] + omega * u_new_int
        u[1:-1, 1:-1, 1:-1, 1] = (1 - omega) * u_old_int[..., 1] + omega * v_new_int
        u[1:-1, 1:-1, 1:-1, 2] = (1 - omega) * u_old_int[..., 2] + omega * w_new_int

        res = float(np.sqrt(np.mean((u[1:-1, 1:-1, 1:-1, :] - u_old_int) ** 2)))
        residual_history.append(res)
        n_done = it
        if res < tol:
            converged = True
            break

    return {"u": u, "n_iter": n_done, "converged": converged, "residual_history": residual_history}


@SolverRegistry.register(
    name="elasticity3d_fdm",
    family="pde",
    description="Static 3D linear elasticity (Navier-Cauchy equilibrium) via damped-Jacobi FDM -- "
                "generic body force, Dirichlet displacement BCs on the domain boundary.",
    tags=["fdm", "elasticity", "structural", "3d", "static"],
)
class Elasticity3DFDMSolver(SolverBase):
    """Thin `SolverBase`/registry wrapper. The functional API
    (`solve_static_elasticity_3d`, `lame_parameters`) is the primary entry
    point and can be used directly without this wrapper."""

    def __init__(self, n_iter: int = 20000, omega: float = 1.0, tol: float = 1e-10):
        super().__init__()
        self.n_iter = int(n_iter)
        self.omega = float(omega)
        self.tol = float(tol)

    def forward(
        self,
        E: float,
        nu: float,
        body_force: np.ndarray,
        u_bc: np.ndarray,
        dx: float,
        dy: float,
        dz: float,
    ) -> SolverOutput:
        lam, mu = lame_parameters(E, nu)
        out = solve_static_elasticity_3d(lam, mu, body_force, u_bc, dx, dy, dz,
                                          n_iter=self.n_iter, omega=self.omega, tol=self.tol)
        return SolverOutput(
            result=torch.from_numpy(out["u"].astype(np.float32)),
            losses={"residual": torch.tensor(out["residual_history"][-1] if out["residual_history"] else 0.0)},
            extras={"n_iter": out["n_iter"], "converged": out["converged"],
                    "residual_history": out["residual_history"], "method": "jacobi_navier_cauchy"},
        )
