"""Euler-Bernoulli beam deflection: 4th-order finite-difference solve of the
beam BVP

    EI * d^4w/dz^4 = q(z),   z in [0, L]

for a uniform distributed load q, under one of three classical boundary
condition sets (simply-supported, cantilever, fixed-fixed), plus the
standard post-processing (moment, shear, bending/shear/von-Mises stress) for
a few common cross-sections. This is the classical numerical counterpart to
a PINN-based Euler-Bernoulli residual (e.g. this package's
`euler_bernoulli_beam` PDE compiler kind) -- same governing equation, solved
by direct sparse linear algebra on a fixed grid instead of by training a
network, useful as a fast reference/ground-truth or wherever no training is
wanted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .base import SolverBase, SolverOutput
from .registry import SolverRegistry

BC_SIMPLY_SUPPORTED = 0
BC_CANTILEVER = 1
BC_FIXED_FIXED = 2

# Minimum grid resolution: the interior 4th-order stencil needs at least one
# real interior row (range(2, n-2) non-empty) and the cantilever shear BC
# reaches back to column n-4 -- both need n >= 6 with margin to spare.
MIN_NX = 6

# Maximum grid resolution. This 4th-order FD stencil's coefficients scale as
# 1/dz^4 while the boundary rows scale as 1/dz..1/dz^3 -- as dz shrinks that
# spread in magnitude grows the matrix's condition number as O(dz^-4) (a
# textbook property of 4th-order FD systems, not specific to this beam
# equation), and floating-point solves start losing digits well before that
# condition number reaches 1/machine-epsilon. Verified against the
# closed-form solution: error stays within ~0.2% out to nx=3000 (already
# ~15x more resolution than needed -- accuracy plateaus by nx~1000) and
# becomes noisy/non-monotonic beyond nx~4000, so nx=3000 is a generous
# ceiling with real margin, not a tight one.
MAX_NX = 3000


@dataclass
class SectionProps:
    """Cross-section properties required for stress calculations.

    I_m4 : second moment of area [m^4]
    c_m  : extreme-fibre distance from neutral axis [m]
    Q_m3 : first moment of the half cross-section about the neutral axis [m^3]
    t_m  : section width at the neutral axis [m]
    A_m2 : total cross-sectional area [m^2]
    """
    I_m4: float
    c_m: float
    Q_m3: float
    t_m: float
    A_m2: float

    @classmethod
    def rectangular(cls, b: float, h: float) -> "SectionProps":
        I = b * h ** 3 / 12
        c = h / 2
        Q = b * c ** 2 / 2
        return cls(I, c, Q, b, b * h)

    @classmethod
    def hollow_circle(cls, OD: float, ID: float) -> "SectionProps":
        ro, ri = OD / 2.0, ID / 2.0
        I = math.pi * (OD ** 4 - ID ** 4) / 64.0
        Q = (2.0 / 3.0) * (ro ** 3 - ri ** 3)
        t = 2.0 * (ro - ri)
        A = math.pi * (ro ** 2 - ri ** 2)
        return cls(I, ro, Q, t, A)

    @classmethod
    def solid_circle(cls, D: float) -> "SectionProps":
        r = D / 2.0
        I = math.pi * D ** 4 / 64.0
        Q = (2.0 / 3.0) * r ** 3
        return cls(I, r, Q, D, math.pi * r ** 2)


def compute_stresses(moment: np.ndarray, shear: np.ndarray, section: SectionProps):
    """Bending stress (extreme fibre), shear stress (neutral axis), von Mises."""
    sigma_b = np.abs(moment) * section.c_m / section.I_m4
    tau = np.abs(shear) * section.Q_m3 / (section.I_m4 * section.t_m)
    sigma_vm = np.sqrt(sigma_b ** 2 + 3.0 * tau ** 2)
    return sigma_b, tau, sigma_vm


def second_derivative(f: np.ndarray, dz: float) -> np.ndarray:
    """Second derivative of `f`, second-order accurate everywhere including
    the two boundary points (3-point central stencil in the interior, a
    3-point one-sided stencil at each edge).

    Used to get the bending moment (EI * d2w/dz2) directly from the
    deflection, instead of differentiating twice with `np.gradient` (i.e.
    slope = gradient(w), then moment = gradient(slope)): that two-step path
    compounds np.gradient's first-order-accurate edge formula twice at the
    boundary points, which -- for the fixed/simply-supported ends that are
    exactly where bending stress peaks -- silently halves the reported
    moment there (verified against the closed-form Euler-Bernoulli
    solution: the error sits at -50% and does not shrink as the grid is
    refined, i.e. it isn't ordinary truncation error). A single proper
    second-derivative stencil doesn't have that failure mode.
    """
    n = len(f)
    d2f = np.empty_like(f, dtype=np.float64)
    d2f[1:-1] = (f[:-2] - 2.0 * f[1:-1] + f[2:]) / dz ** 2
    d2f[0] = (2.0 * f[0] - 5.0 * f[1] + 4.0 * f[2] - f[3]) / dz ** 2
    d2f[-1] = (2.0 * f[-1] - 5.0 * f[-2] + 4.0 * f[-3] - f[-4]) / dz ** 2
    return d2f


def _build_simply_supported(n: int, dz: float, q_over_EI: float):
    A = sparse.lil_matrix((n, n)); rhs = np.full(n, q_over_EI)
    c = 1.0 / dz ** 4
    for i in range(2, n - 2):
        A[i, i - 2] = c; A[i, i - 1] = -4 * c; A[i, i] = 6 * c
        A[i, i + 1] = -4 * c; A[i, i + 2] = c
    c2 = 1.0 / dz ** 2
    A[0, 0] = 1.0; rhs[0] = 0.0
    A[1, 0] = c2; A[1, 1] = -2 * c2; A[1, 2] = c2; rhs[1] = 0.0
    A[n - 1, n - 1] = 1.0; rhs[n - 1] = 0.0
    A[n - 2, n - 3] = c2; A[n - 2, n - 2] = -2 * c2; A[n - 2, n - 1] = c2; rhs[n - 2] = 0.0
    return A, rhs


def _build_cantilever(n: int, dz: float, q_over_EI: float):
    A = sparse.lil_matrix((n, n)); rhs = np.full(n, q_over_EI)
    c = 1.0 / dz ** 4
    for i in range(2, n - 2):
        A[i, i - 2] = c; A[i, i - 1] = -4 * c; A[i, i] = 6 * c
        A[i, i + 1] = -4 * c; A[i, i + 2] = c
    c1 = 1.0 / dz; c2 = 1.0 / dz ** 2; c3 = 1.0 / dz ** 3
    A[0, 0] = 1.0; rhs[0] = 0.0
    A[1, 0] = -c1; A[1, 1] = c1; rhs[1] = 0.0
    A[n - 2, n - 3] = c2; A[n - 2, n - 2] = -2 * c2; A[n - 2, n - 1] = c2; rhs[n - 2] = 0.0
    A[n - 1, n - 4] = -c3; A[n - 1, n - 3] = 3 * c3; A[n - 1, n - 2] = -3 * c3; A[n - 1, n - 1] = c3
    rhs[n - 1] = 0.0
    return A, rhs


def _build_fixed_fixed(n: int, dz: float, q_over_EI: float):
    A = sparse.lil_matrix((n, n)); rhs = np.full(n, q_over_EI)
    c = 1.0 / dz ** 4
    for i in range(2, n - 2):
        A[i, i - 2] = c; A[i, i - 1] = -4 * c; A[i, i] = 6 * c
        A[i, i + 1] = -4 * c; A[i, i + 2] = c
    c1 = 1.0 / dz
    A[0, 0] = 1.0; rhs[0] = 0.0
    A[1, 0] = -c1; A[1, 1] = c1; rhs[1] = 0.0
    A[n - 1, n - 1] = 1.0; rhs[n - 1] = 0.0
    A[n - 2, n - 2] = -c1; A[n - 2, n - 1] = c1; rhs[n - 2] = 0.0
    return A, rhs


_BUILDERS = {
    BC_SIMPLY_SUPPORTED: _build_simply_supported,
    BC_CANTILEVER: _build_cantilever,
    BC_FIXED_FIXED: _build_fixed_fixed,
}


def solve_euler_bernoulli_beam_1d(
    L_m: float = 10.0,
    E_Pa: float = 200e9,
    I_m4: float = 8.33e-6,
    q_N_per_m: float = 1000.0,
    nx: int = 200,
    bc_type: int = BC_SIMPLY_SUPPORTED,
    section: Optional[SectionProps] = None,
    yield_Pa: float = 0.0,
) -> dict:
    """Solve the 1D Euler-Bernoulli beam BVP EI*w''''=q under a uniform
    distributed load q_N_per_m, for one of the three BC_* boundary
    condition sets. `section` (default: an 0.10x0.10 m rectangular section)
    supplies the cross-section geometry used for the returned stresses --
    pass a `SectionProps` built to match I_m4 for a physically consistent
    stress calculation (a mismatched section only affects the stress
    outputs, not the deflection/moment/shear, which depend on I_m4 alone).

    Returns a dict: z, deflection_m, moment_Nm, shear_N, slope_rad,
    sigma_b_Pa, tau_Pa, sigma_vm_Pa, strain_b, failed, utilization, L,
    yield_Pa. `failed`/`utilization` are all-zero when yield_Pa<=0 (check
    disabled).
    """
    if bc_type not in _BUILDERS:
        raise ValueError(f"bc_type must be one of {sorted(_BUILDERS)} (0=simply supported, 1=cantilever, 2=fixed-fixed), got {bc_type}")
    if nx < MIN_NX:
        raise ValueError(f"nx must be >= {MIN_NX} (need enough grid points for the 4th-order stencil), got {nx}")
    if nx > MAX_NX:
        raise ValueError(f"nx must be <= {MAX_NX} -- the 4th-order stencil's matrix becomes ill-conditioned past this (accuracy already plateaus around nx=1000), got {nx}")
    if not (L_m > 0):
        raise ValueError(f"L_m must be > 0, got {L_m}")
    if not (E_Pa > 0):
        raise ValueError(f"E_Pa must be > 0 (Young's modulus can't be zero or negative), got {E_Pa}")
    if not (I_m4 > 0):
        raise ValueError(f"I_m4 must be > 0 (second moment of area can't be zero or negative), got {I_m4}")
    if yield_Pa < 0:
        raise ValueError(f"yield_Pa must be >= 0 (0 disables the failure check), got {yield_Pa}")

    if section is None:
        section = SectionProps.rectangular(0.10, 0.10)

    EI = E_Pa * I_m4
    z = np.linspace(0.0, L_m, nx)
    dz = z[1] - z[0]
    A_mat, rhs = _BUILDERS[bc_type](nx, dz, q_N_per_m / EI)

    w = spsolve(A_mat.tocsr(), rhs)
    slope = np.gradient(w, dz, edge_order=2)
    moment = EI * second_derivative(w, dz)
    shear = np.gradient(moment, dz, edge_order=2)
    sigma_b, tau, sigma_vm = compute_stresses(moment, shear, section)
    strain_b = sigma_b / E_Pa

    failed = (sigma_vm >= yield_Pa) if yield_Pa > 0 else np.zeros(nx, dtype=bool)
    utilization = sigma_vm / yield_Pa if yield_Pa > 0 else np.zeros(nx)

    return {
        "z": z, "deflection_m": w, "moment_Nm": moment, "shear_N": shear,
        "slope_rad": slope, "sigma_b_Pa": sigma_b, "tau_Pa": tau,
        "sigma_vm_Pa": sigma_vm, "strain_b": strain_b,
        "failed": failed.astype(np.float64),
        "utilization": utilization, "L": L_m, "yield_Pa": yield_Pa,
    }


@SolverRegistry.register(
    name="beam_bvp_fdm",
    family="pde",
    description="4th-order FDM solve of the Euler-Bernoulli beam BVP (simply-supported, cantilever, "
                "or fixed-fixed) under a uniform distributed load.",
    tags=["fdm", "beam", "structural", "bvp", "1d"],
)
class BeamBVPFDMSolver(SolverBase):
    """Thin `SolverBase`/registry wrapper. `solve_euler_bernoulli_beam_1d`
    is the primary entry point and can be used directly without this
    wrapper."""

    def __init__(self, nx: int = 200, bc_type: int = BC_SIMPLY_SUPPORTED):
        super().__init__()
        self.nx = int(nx)
        self.bc_type = int(bc_type)

    def forward(
        self,
        L_m: float,
        E_Pa: float,
        I_m4: float,
        q_N_per_m: float,
        section: Optional[SectionProps] = None,
        yield_Pa: float = 0.0,
    ) -> SolverOutput:
        res = solve_euler_bernoulli_beam_1d(
            L_m=L_m, E_Pa=E_Pa, I_m4=I_m4, q_N_per_m=q_N_per_m,
            nx=self.nx, bc_type=self.bc_type, section=section, yield_Pa=yield_Pa,
        )
        return SolverOutput(
            result=torch.from_numpy(res["deflection_m"].astype(np.float32)),
            losses={"residual": torch.tensor(0.0)},
            extras={k: v for k, v in res.items() if k != "deflection_m"},
        )
