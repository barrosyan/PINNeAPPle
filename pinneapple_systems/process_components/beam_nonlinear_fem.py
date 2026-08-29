"""pinneapple_systems.process_components.beam_nonlinear_fem -- static,
geometrically nonlinear (Von Karman) Euler-Bernoulli beam FEM via
load-stepped Newton-Raphson.

SELECTED FORMULATION: this reuses the transient spectral-element Von
Karman beam engine in `pinneapple_simulation.numerical_solvers.
nonlinear_beam_fem` (`NonlinearBeamNewmarkFEM`) at its exact static limit,
rather than re-deriving the nonlinear stiffness/tangent formulation a
second time. That engine's per-timestep residual is

    residual = (F_ext - F_therm) + M @ As + C @ Bs - (K_elastic + t3*M + t6*C) @ u

where `M`/`C` are the consistent mass/damping matrices scaled by
`density`/`damping_coeff`. Setting `density = 0` and `damping_coeff = 0`
makes `M` and `C` identically zero, which collapses this to the pure
static residual `F_ext - F_therm - K_elastic(u) @ u` -- i.e. the same
Newton-Raphson-per-load-step iteration the transient engine already
performs, with no inertial or viscous terms left to integrate. Running it
with `timesteps=1` per load step and reading the converged state after
the last load step is therefore an exact static nonlinear beam solve
using the transient engine's own (independently mass/stiffness-eigenvalue
-verified) element formulation -- not an approximation of it.

This avoids transcribing the Von Karman element stiffness/tangent a
second time (the error-prone part of any FEM port) and inherits whatever
correctness the transient engine already has. See that module's
docstring for the full DOF/boundary-condition-dict convention (shared
unchanged here): DOF 0=axial u, 1=transverse w, 2=slope dw/dz per node,
Dirichlet/Neumann dicts of the same shape.

STRESS RECOVERY: bending stress `sigma = -E*c*kappa` with curvature
`kappa = d(theta)/dz` obtained from the FEM's own slope DOF (not a
second finite-difference derivative of displacement -- `theta` is an
independent, exactly-computed nodal unknown in this Hermite formulation,
so only one numerical differentiation, not two, stands between it and
curvature) and `c` the caller-supplied fiber distance from the neutral
axis (height/2 for a symmetric rectangular section). This is the same
sigma = M*c/I identity used in `beam_statics.von_mises_stress_
rectangular_section` (M = E*I*kappa), just evaluated from the nonlinear
solve's own curvature field instead of a linear moment diagram.

VALIDITY ENVELOPE: geometric nonlinearity only (Von Karman moderate-
rotation strain-displacement), no material nonlinearity (stays linear
elastic), no dynamic response (this is the static limit -- see the
transient module directly for time-dependent problems), same fixed
two-node cubic-Hermite transverse element as the underlying engine
(`NODES_PER_ELEMENT == 2`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from pinneapple_simulation.numerical_solvers.nonlinear_beam_fem import (
    DOF_AXIAL,
    DOF_SLOPE,
    DOF_TRANSVERSE,
    DOFPN,
    NODES_PER_ELEMENT,
    NonlinearBeamNewmarkFEM,
)


@dataclass(frozen=True)
class StaticNonlinearBeamResult:
    z: np.ndarray
    u_m: np.ndarray
    w_m: np.ndarray
    theta_rad: np.ndarray
    bending_moment: np.ndarray
    bending_stress: np.ndarray
    load_step_deflections: np.ndarray
    iterations_last_step: int
    nonconverged: bool


def solve_nonlinear_beam_static(
    boundary: Dict[str, Any],
    L_m: float,
    A_m2: float,
    I_m4: float,
    E_Pa: float,
    fiber_distance_m: float,
    axial_dist_load_N_per_m: float = 0.0,
    transverse_dist_load_N_per_m: float = 0.0,
    num_elements: int = 50,
    load_steps: int = 10,
    newton_iterations: int = 30,
    newton_tolerance: float = 1e-8,
    reference_temperature: float = 0.0,
    temperature_field: Optional[np.ndarray] = None,
    thermal_expansion_coeff: float = 0.0,
) -> StaticNonlinearBeamResult:
    """Load-stepped Newton-Raphson static solve of the geometrically
    nonlinear (Von Karman) Euler-Bernoulli beam, at the static limit of
    `NonlinearBeamNewmarkFEM` (see module docstring). `fiber_distance_m`
    is the distance from the neutral axis to the outer fiber used for
    bending-stress recovery (height/2 for a symmetric rectangular
    section -- use `beam_statics.rectangular_section_properties` for I
    and pass `height_m/2` here).

    `boundary["N"]["Values"]` entries use the same `[dof, [const, sin_amp,
    cos_amp]]` shape as the transient engine; pass `sin_amp=cos_amp=0` for
    a plain static point/nodal load (they have no effect here regardless,
    since a static solve has no time axis to drive them with).
    """
    if fiber_distance_m <= 0:
        raise ValueError(f"fiber_distance_m must be > 0, got {fiber_distance_m}")
    if load_steps < 1:
        raise ValueError(f"load_steps must be >= 1, got {load_steps}")

    # The underlying engine's constructor requires density > 0 (it is a
    # generic transient solver). A vanishingly small placeholder density
    # makes the mass matrix -- and therefore its contribution to the
    # residual -- negligible to machine precision relative to the elastic
    # stiffness terms (which scale with E ~ 1e9-1e11 Pa) for any physical
    # geometry, without violating that precondition; it has no bearing on
    # the static solve, which performs no time-marching.
    _EPS_DENSITY = 1e-12

    engine = NonlinearBeamNewmarkFEM(
        num_elements=num_elements,
        nodes_per_element=NODES_PER_ELEMENT,
        length=L_m,
        area=A_m2,
        moment_of_inertia=I_m4,
        E=E_Pa,
        density=_EPS_DENSITY,
        damping_coeff=0.0,
        axial_dist_load=axial_dist_load_N_per_m,
        transverse_dist_load=transverse_dist_load_N_per_m,
        boundary=boundary,
        total_time=1.0,
        timesteps=1,
        newmark_beta=0.25,
        newmark_gamma=0.5,
        newton_relaxation=0.0,
        newton_iterations=newton_iterations,
        newton_tolerance=newton_tolerance,
        load_steps=load_steps,
        reference_temperature=reference_temperature,
        temperature_field=temperature_field,
        thermal_expansion_coeff=thermal_expansion_coeff,
        driving_freq_hz=0.0,
    )
    engine.run()

    u = engine.convergedSoln[DOF_AXIAL, -1, 1, :]
    w = engine.convergedSoln[DOF_TRANSVERSE, -1, 1, :]
    theta = engine.convergedSoln[DOF_SLOPE, -1, 1, :]
    z = engine.globCoord

    kappa = np.gradient(theta, z)
    M = E_Pa * I_m4 * kappa
    sigma = -E_Pa * fiber_distance_m * kappa

    tip_or_last_node_w_per_load_step = engine.convergedSoln[DOF_TRANSVERSE, :, 1, -1]

    return StaticNonlinearBeamResult(
        z=z,
        u_m=u,
        w_m=w,
        theta_rad=theta,
        bending_moment=M,
        bending_stress=sigma,
        load_step_deflections=tip_or_last_node_w_per_load_step,
        iterations_last_step=int(engine.countIter[-1, 1]),
        nonconverged=bool(engine.nonConvergence),
    )
