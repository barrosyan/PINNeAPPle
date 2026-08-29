"""pinneapple_systems.process_components.beam_statics -- closed-form
static, linear-elastic Euler-Bernoulli beam solutions for the standard
cantilever and simply-supported load cases.

SELECTED FORMULATION: direct evaluation of the known closed-form
solutions of the static Euler-Bernoulli beam equation

    E*I * d4w/dx4 = q(x)                                              (1)

(small-deflection, linear-elastic, no geometric or material
nonlinearity -- see `beam_nonlinear_fem.solve_nonlinear_beam_static` in
this same package for a large-deflection/Von Karman treatment) under
each standard boundary-condition + load-distribution
combination, pre-derived via double/triple integration of (1) with the
case's own boundary conditions (cantilever: w(0)=0, w'(0)=0; simply
supported: w(0)=0, w(L)=0, M(0)=0, M(L)=0). These are textbook results
(see e.g. Roark's Formulas for Stress and Strain, or any standard AISC/
mechanics-of-materials beam table) -- not proprietary to any project.

Every case returns deflection w(x), slope w'(x), bending moment
M(x) = -E*I*w''(x), and shear force V(x) = dM/dx as arrays over a
caller-supplied position array x in [0, L]. Two independent closed-form
identities were used to cross-check the transcribed formulas against
this module's own test suite before trusting them: cantilever with a
full uniform load has tip deflection q*L^4/(8*E*I); simply-supported
with a full uniform load has midspan deflection 5*q*L^4/(384*E*I) --
both textbook-standard results, both reproduced exactly by evaluating
the general partial/full-load formulas at their limiting case.

Piecewise (partial-load / non-uniform) cases are evaluated with
`numpy.where` branch selection at the load-extent/midspan cutoff, so
each case is a single vectorized function over the whole position
array rather than a scalar function called per point.

STRESS: `von_mises_stress_rectangular_section` combines bending stress
(sigma = M*c/I, c = h/2) and shear stress (tau = V*Q/(I*b), Q = b*h^2/8,
the first moment of area at the neutral axis for a RECTANGULAR
cross-section only) into a Von Mises equivalent stress -- generalize
this helper's Q/I formulas for a non-rectangular section if needed; it
is not baked into the deflection/moment/shear solutions themselves,
which are section-shape-independent (I is a caller-supplied parameter).

VALIDITY ENVELOPE: linear-elastic, small-deflection (w << L) static
loading only. No dynamic/time-dependent response, no geometric
nonlinearity (large deflection), no material nonlinearity (plasticity).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

LoadCase = Literal[
    "cantilever_uniform",
    "cantilever_partial_uniform",
    "cantilever_triangular_tip_max",
    "cantilever_triangular_wall_max",
    "simply_supported_uniform",
    "simply_supported_partial_uniform",
    "simply_supported_triangular_far_support_max",
    "simply_supported_triangular_midspan_max",
]


@dataclass(frozen=True)
class BeamResult:
    x: np.ndarray
    deflection: np.ndarray
    slope: np.ndarray
    bending_moment: np.ndarray
    shear_force: np.ndarray


def rectangular_section_properties(width_m: float, height_m: float) -> tuple[float, float]:
    """Returns (I, Q) for a rectangular cross-section: second moment of
    area I = b*h^3/12, first moment of area at the neutral axis
    Q = b*h^2/8 (used for shear stress)."""
    I = width_m * height_m ** 3 / 12.0
    Q = width_m * height_m ** 2 / 8.0
    return I, Q


def von_mises_stress_rectangular_section(
    bending_moment: np.ndarray, shear_force: np.ndarray, I: float, Q: float, width_m: float, height_m: float,
) -> np.ndarray:
    sigma = bending_moment * (height_m / 2.0) / I
    tau = shear_force * Q / (I * width_m)
    return np.sqrt(sigma ** 2 + 3.0 * tau ** 2)


def solve_beam(
    case: LoadCase, x: np.ndarray, L: float, E: float, I: float, q0: float, load_extent: float | None = None,
) -> BeamResult:
    """Dispatches to the closed-form solution for `case`. `load_extent`
    (the partial-load length, measured from the load's own reference
    end per case) is required for the two partial-uniform-load cases
    and ignored otherwise."""
    EI = E * I
    if case == "cantilever_uniform":
        w = (q0 / 24 / EI) * x ** 2 * (6 * L ** 2 - 4 * L * x + x ** 2)
        theta = 4 * (q0 / 24 / EI) * x * (3 * L ** 2 - 3 * L * x + x ** 2)
        M = q0 * (L - x) ** 2 / 2
        V = q0 * (x - L)
    elif case == "cantilever_partial_uniform":
        a = _require(load_extent, "load_extent")
        below = x <= a
        w = np.where(
            below,
            (q0 / 24 / EI) * x ** 2 * (6 * a ** 2 - 4 * a * x + x ** 2),
            (q0 / 24 / EI) * a ** 3 * (4 * x - a),
        )
        theta = 4 * np.where(
            below,
            (q0 / 24 / EI) * x * (3 * a ** 2 - 3 * a * x + x ** 2),
            (q0 / 24 / EI) * a ** 3,
        )
        M = np.where(below, q0 * (a - x) ** 2 / 2, 0.0)
        V = np.where(below, q0 * (x - a), 0.0)
    elif case == "cantilever_triangular_tip_max":
        w = (q0 / 120 / L / EI) * x ** 2 * (20 * L ** 3 - 10 * L ** 2 * x + x ** 3)
        theta = 5 * (q0 / 120 / L / EI) * x * (8 * L ** 3 - 6 * L ** 2 * x + x ** 3)
        M = q0 * (2 * L ** 3 - 3 * L ** 2 * x + x ** 3) / 6 / L
        V = q0 * (x ** 2 - L ** 2) / 2 / L
    elif case == "cantilever_triangular_wall_max":
        w = (q0 / 120 / L / EI) * x ** 2 * (10 * L ** 3 - 10 * L ** 2 * x + 5 * L * x ** 2 - x ** 3)
        theta = 5 * (q0 / 120 / L / EI) * x * (4 * L ** 3 - 6 * L ** 2 * x + 4 * L * x ** 2 - x ** 3)
        M = q0 * (L ** 3 - 3 * L ** 2 * x + 3 * L * x ** 2 - x ** 3) / 6 / L
        V = -q0 * (L - x) ** 2 / 2 / L
    elif case == "simply_supported_uniform":
        w = (q0 / EI / 24) * x * (L ** 3 - 2 * L * x ** 2 + x ** 3)
        theta = (q0 / EI / 24) * (L ** 3 - 6 * L * x ** 2 + 4 * x ** 3)
        M = q0 * (x ** 2 - L * x) / 2
        V = q0 * (2 * x - L) / 2
    elif case == "simply_supported_partial_uniform":
        a = _require(load_extent, "load_extent")
        below = x <= a
        w = np.where(
            below,
            (q0 / EI / 24 / L) * x * (a ** 4 - 4 * a ** 3 * L + 4 * a ** 2 * L ** 2 + 2 * a ** 2 * x ** 2 - 4 * a * L * x ** 2 + L * x ** 3),
            (q0 / EI / 24 / L) * a ** 2 * (-a ** 2 * L + 4 * L ** 2 * x + a ** 2 * x - 6 * L * x ** 2 + 2 * x ** 3),
        )
        theta = np.where(
            below,
            (q0 / EI / 24 / L) * (a ** 4 - 4 * a ** 3 * L + 4 * a ** 2 * L ** 2 + 6 * a ** 2 * x ** 2 - 12 * a * L * x ** 2 + 4 * L * x ** 3),
            (q0 / EI / 24 / L) * a ** 2 * (4 * L ** 2 + a ** 2 - 12 * L * x + 6 * x ** 2),
        )
        M = np.where(
            below,
            q0 * (a ** 2 * x - 2 * a * L * x + L * x ** 2) / (2 * L),
            q0 * a ** 2 * (x - L) / (2 * L),
        )
        V = np.where(
            below,
            q0 * (a ** 2 - 2 * a * L + 2 * L * x) / (2 * L),
            q0 * a ** 2 / (2 * L),
        )
    elif case == "simply_supported_triangular_far_support_max":
        w = (q0 / 360 / L / EI) * x * (7 * L ** 4 - 10 * L ** 2 * x ** 2 + 3 * x ** 4)
        theta = (q0 / 360 / L / EI) * (7 * L ** 4 - 30 * L ** 2 * x ** 2 + 15 * x ** 4)
        M = q0 * (x ** 3 - L ** 2 * x) / 6 / L
        V = q0 * (3 * x ** 2 - L ** 2) / 6 / L
    elif case == "simply_supported_triangular_midspan_max":
        below = x <= L / 2
        xr = L - x
        w = np.where(
            below,
            (q0 / 960 / L / EI) * x * (5 * L ** 2 - 4 * x ** 2) ** 2,
            (q0 / 960 / L / EI) * xr * (5 * L ** 2 - 4 * xr ** 2) ** 2,
        )
        theta = np.where(
            below,
            5 * (q0 / 960 / L / EI) * (5 * L ** 2 - 4 * x ** 2) * (L ** 2 - 4 * x ** 2),
            -5 * (q0 / 960 / L / EI) * (5 * L ** 2 - 4 * xr ** 2) * (L ** 2 - 4 * xr ** 2),
        )
        M = np.where(
            below,
            q0 * (-3 * L ** 2 * x + 4 * x ** 3) / (12 * L),
            q0 * (-3 * L ** 2 * xr + 4 * xr ** 3) / (12 * L),
        )
        V = np.where(
            below,
            q0 * (4 * x ** 2 - L ** 2) / (4 * L),
            -q0 * (4 * xr ** 2 - L ** 2) / (4 * L),
        )
    else:
        raise ValueError(f"unknown load case: {case!r}")

    return BeamResult(x=x, deflection=w, slope=theta, bending_moment=M, shear_force=V)


def _require(value, name: str):
    if value is None:
        raise ValueError(f"{name} is required for this load case")
    return value
