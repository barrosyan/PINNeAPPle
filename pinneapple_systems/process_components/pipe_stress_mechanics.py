"""pinneapple_systems.process_components.pipe_stress_mechanics --
combined-load stress analysis for a slender tube (axial + torsional +
bending + internal/external pressure), Euler column buckling, and
constrained-rod (Paslay-Dawson) buckling of a slender member confined
within a surrounding cylindrical clearance.

SELECTED FORMULATION
---------------------
Thick-wall (Lame) cylinder hoop stress under internal pressure at the
inner and outer fiber -- `sigma_theta(r) = p_i*ri**2/(ro**2-ri**2) *
(1 + ro**2/r**2)`, evaluated at r=ri (inner, maximum, where the `(1 +
ro**2/r**2)` factor becomes `1 + ro**2/ri**2`) and r=ro (outer, where it
becomes `1 + ro**2/ro**2 = 2` -- do not drop this factor of 2: doing so
under-predicts the outer-fiber hoop stress by half, and the thin-wall
limit `sigma_theta -> p_i*r_mean/t` this module's own test suite checks
against only comes out right with it kept);
radial stress `sigma_r(ri) = -p_i` at the inner wall (free-surface
condition), `sigma_r(ro) = 0` at the outer wall (also free-surface,
absent external pressure) -- both are standard results (Timoshenko &
Goodier, *Theory of Elasticity*).

Combined equivalent (Von Mises) stress uses the full triaxial
formulation `sigma_VM = sqrt(0.5*[(s1-s2)^2+(s2-s3)^2+(s3-s1)^2] +
3*tau^2)` for the three orthogonal normal stresses (axial/longitudinal,
hoop, radial -- these are principal directions for an axisymmetric thin-
or thick-walled tube) plus a shear stress `tau` acting in the hoop-axial
plane (as produced by torsion) -- this correctly reduces to the simpler
biaxial `sqrt(sigma_L**2+sigma_H**2-sigma_L*sigma_H+3*tau**2)` form when
the radial stress is zero (a common simplification at the outer, stress-
free fiber), and satisfies the textbook invariant that an equal triaxial
(hydrostatic) stress state gives zero equivalent stress -- both checked
in this module's test suite.

Buckling has two independent formulations, both included since they
answer different questions for the same slender member:
- Euler column buckling: `P_cr = pi**2*E*I / (K*L)**2`, the classical
  unconstrained-column critical load (Timoshenko, *Theory of Elastic
  Stability*), `K` the standard end-condition factor (0.5 fixed-fixed,
  0.7 fixed-pinned, 1.0 pinned-pinned, 2.0 fixed-free).
- Constrained-rod (Paslay-Dawson) buckling: for a slender rod confined
  within a surrounding cylindrical clearance under axial compression and
  its own distributed lateral load (e.g. buoyed self-weight component
  normal to the rod axis), the rod buckles sinusoidally at
  `F_sin = 2*sqrt(E*I*w_lateral/r_clearance)` and helically at
  `F_hel = 2*F_sin` (Paslay & Dawson 1964; Mitchell 1988) -- this is the
  governing mode whenever the surrounding clearance is much smaller than
  the unsupported Euler buckling length would otherwise imply (a rod
  inside a bore behaves very differently from one buckling in free
  space), and is the standard treatment for any slender member confined
  within a cylindrical bore (drill pipe in a wellbore, tubing in casing,
  a rod in a sleeve/guide), not specific to any one such application.

Beam-column second-order (P-Delta) moment amplification: a lateral-
bending stress under simultaneous axial compression is amplified beyond
its first-order (linear) value because the axial load acts through the
member's own bent shape. The standard structural-design approximation
(AISC/ACI "B1 factor" method, uniform-moment case) is
`AF = 1 / (1 - P/P_cr)` with `P_cr` the Euler critical load -- a
first-order-accurate approximation to the exact beam-column secant
solution, adequate up to P/P_cr ~ 0.9-ish and standard practice for
design-level (not research-level exact-elastica) beam-column analysis.

Rotating-bending stress cycle: a shaft/rod under a constant transverse
bending moment that itself rotates about the shaft axis (e.g. any
rotating shaft with a fixed bend, or a non-rotating bend seen by
material that rotates through it) experiences
`sigma(theta) = sigma_axial + sigma_bending*cos(theta)` as `theta` (the
material's rotational position) sweeps 0..2*pi -- the classic "rotating
bending" fatigue loading case (R.R. Moore-type). The stress envelope
over one full rotation gives the alternating and mean stress a fatigue
analysis needs: `sigma_alt = (sigma_max-sigma_min)/2`,
`sigma_mean = (sigma_max+sigma_min)/2`, evaluated after combining with
any other steady stress (hoop, torsion) via `von_mises_triaxial` at each
rotational position -- see `fatigue_analysis` for what consumes these.

VALIDITY ENVELOPE: linear-elastic, small-strain. The combined-stress
functions assume principal directions coincide with the
axial/hoop/radial frame (true for axisymmetric loading -- not valid
under a general 3D bending+torsion state with off-axis principal
stresses). Constrained-rod buckling assumes a straight, initially
concentric clearance and doesn't resolve the actual post-buckling shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def lame_hoop_stress_outer(p_internal: ArrayLike, r_outer: ArrayLike, r_inner: ArrayLike) -> ArrayLike:
    """sigma_theta at r=r_outer: 2*p_i*ri**2 / (ro**2-ri**2) (the general
    Lame hoop-stress formula's `(1 + ro**2/r**2)` factor evaluates to
    exactly 2 at r=r_outer -- see module docstring)."""
    denom = r_outer ** 2 - r_inner ** 2
    return np.where(denom > 1e-12, 2.0 * p_internal * r_inner ** 2 / np.where(denom > 1e-12, denom, 1.0), 0.0)


def lame_hoop_stress_inner(p_internal: ArrayLike, r_outer: ArrayLike, r_inner: ArrayLike) -> ArrayLike:
    """sigma_theta at r=r_inner (the maximum): p_i*(ri**2+ro**2) / (ro**2-ri**2)."""
    denom = r_outer ** 2 - r_inner ** 2
    return np.where(
        denom > 1e-12,
        p_internal * (r_inner ** 2 + r_outer ** 2) / np.where(denom > 1e-12, denom, 1.0),
        0.0,
    )


def torsional_shear_stress(torque: ArrayLike, r_outer: ArrayLike, J: ArrayLike) -> ArrayLike:
    """tau = T*c/J, c = r_outer (maximum shear, at the outer fiber)."""
    return torque * r_outer / J


def bending_stress_from_curvature(curvature: ArrayLike, E: ArrayLike, r_outer: ArrayLike) -> ArrayLike:
    """sigma_bend = E*kappa*r_outer (outer-fiber bending stress for a
    circular cross-section bent to curvature `kappa` [1/length])."""
    return E * curvature * r_outer


def von_mises_triaxial(sigma_axial: ArrayLike, sigma_hoop: ArrayLike,
                        sigma_radial: ArrayLike = 0.0, tau: ArrayLike = 0.0) -> ArrayLike:
    """Full triaxial Von Mises equivalent stress for three orthogonal
    principal-direction normal stresses plus a shear stress acting in
    the (axial, hoop) plane. See module docstring for the derivation and
    its biaxial (`sigma_radial=0`) reduction."""
    s1, s2, s3 = sigma_axial, sigma_hoop, sigma_radial
    return np.sqrt(0.5 * ((s1 - s2) ** 2 + (s2 - s3) ** 2 + (s3 - s1) ** 2) + 3.0 * tau ** 2)


def euler_critical_buckling_load(E: float, I: float, length: float, end_condition_factor: float = 1.0) -> float:
    """P_cr = pi**2*E*I / (K*L)**2. `end_condition_factor` K: 0.5 fixed-
    fixed, 0.7 fixed-pinned, 1.0 pinned-pinned (default), 2.0 fixed-free
    (cantilever)."""
    return np.pi ** 2 * E * I / (end_condition_factor * length) ** 2


@dataclass(frozen=True)
class ConstrainedRodBucklingResult:
    F_critical_sinusoidal: float
    F_critical_helical: float
    mode: str


def constrained_rod_buckling_load(E: float, I: float, lateral_load_per_length: float,
                                   radial_clearance: float) -> ConstrainedRodBucklingResult:
    """Paslay-Dawson sinusoidal/helical critical compressive loads for a
    slender rod confined within a surrounding cylindrical clearance (see
    module docstring). `lateral_load_per_length` is the distributed load
    normal to the rod axis (e.g. buoyed self-weight component); returns
    "not_applicable" mode with F -> inf if there is no lateral load or no
    clearance to buckle into."""
    if lateral_load_per_length <= 1e-9 or radial_clearance <= 1e-9:
        return ConstrainedRodBucklingResult(float("inf"), float("inf"), "not_applicable")
    F_sin = 2.0 * np.sqrt(E * I * lateral_load_per_length / radial_clearance)
    return ConstrainedRodBucklingResult(F_sin, 2.0 * F_sin, "applicable")


def classify_buckling_mode(compressive_force: ArrayLike, result: ConstrainedRodBucklingResult) -> ArrayLike:
    """0 = straight, 1 = sinusoidal, 2 = helical, per Paslay-Dawson
    critical loads."""
    F = np.asarray(compressive_force, dtype=float)
    return np.where(F < result.F_critical_sinusoidal, 0,
                     np.where(F < result.F_critical_helical, 1, 2))


def beam_column_moment_amplification_factor(
    compressive_force: ArrayLike, E: float, I: float, length: float, end_condition_factor: float = 1.0,
    max_ratio: float = 0.98,
) -> ArrayLike:
    """AF = 1 / (1 - P/P_cr), the standard beam-column second-order
    (P-Delta) moment-amplification approximation (see module docstring).
    `P/P_cr` is clipped to `max_ratio` before dividing, since the
    approximation (and physically, the member) diverges as the axial
    load approaches the critical load -- the clip reports a large but
    finite amplification instead of blowing up, flagging "at/beyond
    buckling" rather than returning inf or a negative factor."""
    P_cr = euler_critical_buckling_load(E, I, length, end_condition_factor)
    ratio = np.clip(np.asarray(compressive_force, dtype=float) / P_cr, 0.0, max_ratio)
    return 1.0 / (1.0 - ratio)


@dataclass(frozen=True)
class RotatingBendingCycleResult:
    theta_rad: np.ndarray
    sigma_von_mises: np.ndarray
    sigma_alternating: float
    sigma_mean: float


def rotating_bending_stress_cycle(
    sigma_axial: float, sigma_bending_amplitude: float, sigma_hoop: float = 0.0, tau: float = 0.0,
    n_angles: int = 36,
) -> RotatingBendingCycleResult:
    """Von Mises stress envelope over one full rotation of a shaft/rod
    under constant axial + hoop + torsional stress plus a bending stress
    that varies as `sigma_bending_amplitude*cos(theta)` (see module
    docstring), and the alternating/mean stress a fatigue analysis needs
    from that envelope."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    sigma_total = sigma_axial + sigma_bending_amplitude * np.cos(theta)
    sigma_vm = von_mises_triaxial(sigma_total, sigma_hoop, 0.0, tau)
    return RotatingBendingCycleResult(
        theta_rad=theta, sigma_von_mises=sigma_vm,
        sigma_alternating=float((sigma_vm.max() - sigma_vm.min()) / 2.0),
        sigma_mean=float((sigma_vm.max() + sigma_vm.min()) / 2.0),
    )
