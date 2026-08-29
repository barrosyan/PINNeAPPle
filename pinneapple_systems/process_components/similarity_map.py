"""pinneapple_systems.process_components.similarity_map — nondimensional
turbomachinery performance-map utilities (flow/head coefficients, tip
Mach and machine Reynolds corrections, surge/choke margins).

SELECTED FORMULATION: standard nondimensional similarity groups (Boyce,
"Centrifugal Compressors: A Basic Guide", PennWell 2003; consistent with
ASME PTC 10's corrected-flow/corrected-speed convention):

    phi (flow coefficient)      = Q1 / (N * D2^3)
    psi (head coefficient, per
         impeller stage)         = H_p_per_stage / U2^2
    U2  (impeller tip speed)     = pi * D2 * N            (N in rev/s)
    Mu2 (tip Mach number)         = U2 / a1                (a1 = suction speed of sound)
    Re  (machine Reynolds)         = U2 * D2 / nu1           (nu1 = suction kinematic viscosity)

where Q1 is the ACTUAL (not standard) inlet volumetric flow and D2 is the
impeller/casing reference diameter.

MULTISTAGE MACHINES: psi above is inherently a PER-STAGE head
coefficient (a single impeller's Euler head over U2^2). A machine with
`n_stages` impellers in series has total head n_stages * psi * U2^2 --
using the single-stage value directly for a multistage machine
understates the achievable head by a factor of n_stages, an error this
module's own development caught (an earlier single-stage treatment of a
multistage test case could not reach its target discharge pressure at
any allowed speed). `MapCoefficients.n_stages` defaults to 1 (a
single-stage machine); set it explicitly for a multistage casing.

THIS IS A SYNTHETIC-MAP TOOLKIT: `MapCoefficients` fields are meant to be
FIT to a real supplier map or test-stand data — this module supplies the
functional FORM (parabolic head-flow curve, a peaked efficiency curve,
bounded Mach/Reynolds corrections) and the nondimensional bookkeeping,
not real machine data. `make_map` derives the head-curve curvature from
a stated (BEP flow, BEP head, shutoff head) triple so those three numbers
can never silently drift out of mutual consistency the way a fourth,
independently hand-picked curvature parameter could.

Tip-Mach and machine-Reynolds corrections are applied as bounded,
monotonic multiplicative corrections (negligible near the reference
condition, growing toward choke/off-reference Reynolds) per the general
similarity-correction treatment in the turbomachinery literature; their
exact MAGNITUDE (not just existence) is a configurable, illustrative
parameter until fit to real off-design test data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class MapCoefficients:
    name: str
    D2_m: float
    n_stages: int = 1
    phi_ref_bep: float = 0.07
    psi_ref_bep: float = 0.65
    psi_shape_a: float = 1.0    # derived by make_map -- do not hand-pick, see module docstring
    psi0: float = 0.85
    eta_p_max: float = 0.80
    eta_p_shape_b: float = 55.0
    phi_surge: float = 0.04
    phi_choke: float = 0.10
    Mu2_ref: float = 0.80
    mach_correction_strength: float = 0.6
    reynolds_correction_exponent: float = 0.20


def derive_psi_shape_a(psi0: float, phi_ref_bep: float, psi_ref_bep: float) -> float:
    """Curvature 'a' in psi(phi) = psi0 - a*phi^2 that makes the curve
    pass exactly through (phi_ref_bep, psi_ref_bep)."""
    if phi_ref_bep <= 0:
        raise ValueError("phi_ref_bep must be positive")
    a = (psi0 - psi_ref_bep) / phi_ref_bep ** 2
    if a <= 0:
        raise ValueError("psi0 must exceed psi_ref_bep for a physically decreasing head-flow curve")
    return a


def make_map(
    name: str, D2_m: float, phi_ref_bep: float, psi_ref_bep: float, psi0: float,
    eta_p_max: float, eta_p_shape_b: float, phi_surge: float, phi_choke: float, Mu2_ref: float,
    n_stages: int = 1, **kwargs,
) -> MapCoefficients:
    psi_shape_a = derive_psi_shape_a(psi0, phi_ref_bep, psi_ref_bep)
    return MapCoefficients(
        name=name, D2_m=D2_m, n_stages=n_stages, phi_ref_bep=phi_ref_bep, psi_ref_bep=psi_ref_bep,
        psi_shape_a=psi_shape_a, psi0=psi0, eta_p_max=eta_p_max, eta_p_shape_b=eta_p_shape_b,
        phi_surge=phi_surge, phi_choke=phi_choke, Mu2_ref=Mu2_ref, **kwargs,
    )


@dataclass(frozen=True)
class MapEvaluation:
    phi: float
    psi: float
    eta_p: float
    Mu2: float
    Re: float
    surge_margin_frac: float
    choke_margin_frac: float


def flow_coefficient(Q1_m3_s: float, N_rpm: float, coeffs: MapCoefficients) -> float:
    return Q1_m3_s / (N_rpm / 60.0 * coeffs.D2_m ** 3)


def tip_speed_m_s(N_rpm: float, coeffs: MapCoefficients) -> float:
    return np.pi * coeffs.D2_m * (N_rpm / 60.0)


def tip_mach_number(N_rpm: float, coeffs: MapCoefficients, speed_of_sound_m_s: float) -> float:
    return tip_speed_m_s(N_rpm, coeffs) / speed_of_sound_m_s


def machine_reynolds_number(N_rpm: float, coeffs: MapCoefficients, kinematic_viscosity_m2_s: float) -> float:
    return tip_speed_m_s(N_rpm, coeffs) * coeffs.D2_m / kinematic_viscosity_m2_s


def _base_head_coefficient(phi: float, coeffs: MapCoefficients) -> float:
    return coeffs.psi0 - coeffs.psi_shape_a * phi ** 2


def _base_efficiency(phi: float, coeffs: MapCoefficients) -> float:
    eta = coeffs.eta_p_max - coeffs.eta_p_shape_b * (phi - coeffs.phi_ref_bep) ** 2
    return float(np.clip(eta, 0.05, coeffs.eta_p_max))


def _mach_correction(Mu2: float, coeffs: MapCoefficients) -> tuple[float, float]:
    d = Mu2 - coeffs.Mu2_ref
    psi_mult = 1.0 - coeffs.mach_correction_strength * max(d, 0.0) ** 2
    eta_mult = 1.0 - 0.5 * coeffs.mach_correction_strength * max(d, 0.0) ** 2
    return float(np.clip(psi_mult, 0.5, 1.05)), float(np.clip(eta_mult, 0.5, 1.05))


def _reynolds_correction(Re: float, Re_ref: float, coeffs: MapCoefficients) -> float:
    if Re <= 0 or Re_ref <= 0:
        return 1.0
    loss_ratio = (Re_ref / Re) ** coeffs.reynolds_correction_exponent
    return float(1.0 - (loss_ratio - 1.0) * 0.02)


def evaluate_map(
    coeffs: MapCoefficients, Q1_m3_s: float, N_rpm: float,
    speed_of_sound_m_s: float, kinematic_viscosity_m2_s: float, Re_ref: float | None = None,
) -> MapEvaluation:
    phi = flow_coefficient(Q1_m3_s, N_rpm, coeffs)
    Mu2 = tip_mach_number(N_rpm, coeffs, speed_of_sound_m_s)
    Re = machine_reynolds_number(N_rpm, coeffs, kinematic_viscosity_m2_s)

    psi_mult, eta_mult_mach = _mach_correction(Mu2, coeffs)
    eta_mult_re = _reynolds_correction(Re, Re_ref if Re_ref is not None else Re, coeffs)

    # Floored at a small positive value rather than 0: physically, well
    # beyond choke the map is out of its valid range anyway (flagged by
    # choke_margin_frac, computed independently below); flooring keeps
    # root-finders that transiently probe out-of-range trial points from
    # hitting a degenerate zero/negative head.
    psi = max(_base_head_coefficient(phi, coeffs) * psi_mult, 1.0e-3)
    eta_p = float(np.clip(_base_efficiency(phi, coeffs) * eta_mult_mach * eta_mult_re, 0.05, 0.92))

    return MapEvaluation(
        phi=phi, psi=psi, eta_p=eta_p, Mu2=Mu2, Re=Re,
        surge_margin_frac=(phi - coeffs.phi_surge) / coeffs.phi_surge,
        choke_margin_frac=(coeffs.phi_choke - phi) / coeffs.phi_choke,
    )


def polytropic_head_from_psi(psi: float, N_rpm: float, coeffs: MapCoefficients) -> float:
    """Total machine polytropic head (J/kg) = n_stages * psi * U2^2."""
    return coeffs.n_stages * psi * tip_speed_m_s(N_rpm, coeffs) ** 2


def required_speed_for_head(
    target_head_J_kg: float, Q1_m3_s: float, coeffs: MapCoefficients,
    speed_of_sound_m_s: float, kinematic_viscosity_m2_s: float,
    *, N_min_rpm: float, N_max_rpm: float,
) -> float:
    """Solves for the shaft speed producing target_head_J_kg at a fixed
    actual inlet volumetric flow, via bisection on the monotonic
    head(N) relation at fixed Q1."""
    def f(N_rpm):
        ev = evaluate_map(coeffs, Q1_m3_s, N_rpm, speed_of_sound_m_s, kinematic_viscosity_m2_s)
        return polytropic_head_from_psi(ev.psi, N_rpm, coeffs) - target_head_J_kg

    f_min, f_max = f(N_min_rpm), f(N_max_rpm)
    if f_min > 0:
        raise ValueError(f"target head already exceeded at minimum speed {N_min_rpm} rpm for this flow")
    if f_max < 0:
        raise ValueError(f"target head not reachable even at maximum speed {N_max_rpm} rpm for this flow")
    return brentq(f, N_min_rpm, N_max_rpm, xtol=1e-3)
