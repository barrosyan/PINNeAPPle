"""2D axisymmetric magnetostatic / eddy-current FDM solver.

Governing equation (azimuthal magnetic vector potential A_theta, single
angular frequency omega, quasi-static / low-frequency eddy-current
approximation):

    d2A/dr2 + (1/r) dA/dr - A/r2 + d2A/dz2 - i*omega*mu*sigma*A = -mu*J_source

Solved via a sparse COMPLEX linear system (scipy.sparse + spsolve) on a
rectangular (r, z) grid, r in [r_min, r_max], z in [z_min, z_max], with
Dirichlet A=0 on the outer domain boundary by default. This is the standard
axisymmetric formulation used throughout applied electromagnetics for any
problem with rotational symmetry about the z-axis: induction heating,
transformers/inductors, magnetic bearings, eddy-current braking, non-
destructive eddy-current inspection, or any other axisymmetric conductor in
a time-harmonic magnetic field -- the equation and discretization are
generic; the material maps (mu, sigma) and source current density are
caller-supplied, not hardcoded to any one application's geometry.

At omega=0 (mu*sigma term vanishes) this reduces to the axisymmetric
magnetostatic (Poisson-like) equation for a DC current distribution.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

MU0 = 4.0 * np.pi * 1e-7  # H/m, vacuum permeability


def build_axisymmetric_helmholtz_system(
    r: np.ndarray,
    z: np.ndarray,
    k2: np.ndarray,
    rhs_2d: np.ndarray,
):
    """Assemble the sparse complex system M @ a = b for the axisymmetric
    operator

        L[A] - k2(r,z)*A = rhs(r,z)

    where L[A] = d2A/dr2 + (1/r)dA/dr - A/r2 + d2A/dz2 (the axisymmetric
    vector-Laplacian-like operator acting on a single scalar component
    A_theta), via a standard 5-point finite-difference stencil at interior
    points and Dirichlet A=0 at all boundary points.

    r, z: 1D coordinate arrays (uniform spacing assumed within each axis).
    k2: (len(r), len(z)) complex array -- e.g. k2 = 1j*omega*mu*sigma for the
        eddy-current equation, or k2 = 0 for pure magnetostatics.
    rhs_2d: (len(r), len(z)) complex array -- e.g. rhs = -mu*J_source.

    Returns (M_csr, b_dense) with flat node ordering n = i*nz + j.
    """
    from scipy.sparse import csr_matrix

    nr, nz = len(r), len(z)
    dr, dz = float(r[1] - r[0]), float(z[1] - z[0])
    N = nr * nz

    i_int = np.arange(1, nr - 1)
    j_int = np.arange(1, nz - 1)
    II, JJ = np.meshgrid(i_int, j_int, indexing="ij")
    II = II.ravel()
    JJ = JJ.ravel()
    ri = r[II]
    N_int = len(II)

    n_base = II * nz + JJ

    c_rp = 1.0 / dr ** 2 + 1.0 / (2.0 * ri * dr)
    c_rm = 1.0 / dr ** 2 - 1.0 / (2.0 * ri * dr)
    c_zp = 1.0 / dz ** 2 * np.ones(N_int)
    c_zm = 1.0 / dz ** 2 * np.ones(N_int)
    c_c = (-2.0 / dr ** 2 - 2.0 / dz ** 2 - 1.0 / ri ** 2 - k2[II, JJ])

    rows = np.concatenate([n_base, n_base, n_base, n_base, n_base])
    cols = np.concatenate([
        n_base,
        (II + 1) * nz + JJ,
        (II - 1) * nz + JJ,
        II * nz + (JJ + 1),
        II * nz + (JJ - 1),
    ])
    data = np.concatenate([c_c, c_rp, c_rm, c_zp, c_zm])

    all_n = np.arange(N, dtype=np.intp)
    in_mask = np.zeros(N, bool)
    in_mask[n_base] = True
    bnd_n = all_n[~in_mask]
    rows = np.concatenate([rows, bnd_n])
    cols = np.concatenate([cols, bnd_n])
    data = np.concatenate([data, np.ones(len(bnd_n), dtype=complex)])

    M = csr_matrix((data, (rows, cols)), shape=(N, N), dtype=complex)

    b = np.zeros(N, dtype=complex)
    b[n_base] = rhs_2d[II, JJ]

    return M, b


def annular_current_source(
    r: np.ndarray,
    z: np.ndarray,
    r_center: float,
    z_center: float,
    width_z: float,
    width_r: float,
    n_turns: float,
    current_amp: float,
) -> np.ndarray:
    """Uniform azimuthal current density J_theta (A/m^2) in a rectangular
    annular band centered at (r_center, z_center) -- the standard
    lumped-coil approximation for an axisymmetric current loop / solenoid
    turn, used to drive the eddy-current or magnetostatic equation above.
    Returns a (len(r), len(z)) real array.
    """
    RR, ZZ = np.meshgrid(r, z, indexing="ij")
    mask = (np.abs(ZZ - z_center) <= width_z / 2) & (np.abs(RR - r_center) <= width_r)
    J = np.zeros_like(RR)
    area = width_z * width_r * 2
    if area > 0:
        J[mask] = current_amp * n_turns / area
    return J


def solve_axisymmetric_eddy_current(
    r: np.ndarray,
    z: np.ndarray,
    omega: float,
    mu: np.ndarray,
    sigma: np.ndarray,
    source_current_density: np.ndarray,
) -> np.ndarray:
    """Solve for the azimuthal vector potential A_theta(r,z) at angular
    frequency `omega` (rad/s), given material maps mu(r,z) [H/m] and
    sigma(r,z) [S/m] and a source current density J_source(r,z) [A/m^2],
    all as (len(r), len(z)) arrays matching the r,z grids. `omega=0` solves
    the magnetostatic limit (no eddy currents).

    Returns A, shape (len(r), len(z)), complex (purely real when omega=0).
    """
    from scipy.sparse.linalg import spsolve

    k2 = (1j * omega * mu * sigma).astype(complex)
    rhs_2d = (-mu * source_current_density).astype(complex)
    M, b = build_axisymmetric_helmholtz_system(r, z, k2, rhs_2d)
    a_vec = spsolve(M, b)
    return a_vec.reshape(len(r), len(z))


def eddy_current_density(A: np.ndarray, omega: float, sigma: np.ndarray) -> np.ndarray:
    """Induced eddy-current density J_theta = i*omega*sigma*A, given the
    solved vector potential A and the same sigma(r,z) map used to solve it."""
    return 1j * omega * sigma * A


def axial_flux_density(A: np.ndarray, r: np.ndarray, z: np.ndarray) -> np.ndarray:
    """B_z(r,z) = (1/r) d(r*A)/dr, computed via central differences (the
    axisymmetric curl relating A_theta to the axial field component)."""
    RR = r[:, None] * np.ones((1, len(z)))
    dAdr = np.gradient(A.real, r, axis=0)
    return -(A.real / (RR + 1e-30) + dAdr)


@SolverRegistry.register(
    name="eddy_current_fdm",
    family="pde",
    description="2D axisymmetric magnetostatic / eddy-current FDM (complex Helmholtz-type system for the "
                "azimuthal vector potential) -- generic material maps and source, no fixed geometry.",
    tags=["fdm", "electromagnetics", "axisymmetric", "eddy-current", "complex", "magnetostatic"],
)
class EddyCurrentFDMSolver(SolverBase):
    """Thin `SolverBase`/registry wrapper. The functional API
    (`solve_axisymmetric_eddy_current`, `build_axisymmetric_helmholtz_system`,
    `annular_current_source`) is the primary entry point and can be used
    directly without this wrapper."""

    def __init__(self, nr: int = 60, nz: int = 120):
        super().__init__()
        self.nr = int(nr)
        self.nz = int(nz)

    def forward(
        self,
        r_bounds: Tuple[float, float],
        z_bounds: Tuple[float, float],
        omega: float,
        mu: np.ndarray,
        sigma: np.ndarray,
        source_current_density: np.ndarray,
    ) -> SolverOutput:
        r = np.linspace(r_bounds[0], r_bounds[1], self.nr)
        z = np.linspace(z_bounds[0], z_bounds[1], self.nz)
        A = solve_axisymmetric_eddy_current(r, z, omega, mu, sigma, source_current_density)
        J = eddy_current_density(A, omega, sigma)
        return SolverOutput(
            result=torch.from_numpy(np.stack([A.real, A.imag], axis=-1).astype(np.float32)),
            losses={"residual": torch.tensor(0.0)},
            extras={
                "r": r.astype(np.float32), "z": z.astype(np.float32),
                "J_eddy_real": J.real.astype(np.float32), "J_eddy_imag": J.imag.astype(np.float32),
                "omega": omega, "method": "axisymmetric_helmholtz_fdm",
            },
        )
