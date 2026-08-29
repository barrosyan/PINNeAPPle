"""pinneapple_systems.process_components.non_newtonian_pipe_flow --
Herschel-Bulkley rheology, the generalized (Metzner-Reed) Reynolds number
and friction factor for power-law/yield-stress fluids, and steady-state
1D pressure-drop integration along an arbitrarily inclined conduit
(straight pipe, annulus, or curved path -- pass a hydraulic diameter and
an inclination profile).

SELECTED FORMULATION
---------------------
Constitutive law: Herschel-Bulkley, `tau = tau_y + K*gamma_dot**n`
(Bird, Stewart & Lightfoot; the standard 3-parameter yield-pseudoplastic
model -- reduces to power-law at tau_y=0, to Bingham plastic at n=1).

Generalized Reynolds number (Metzner-Reed):
`Re_gen = rho*v**(2-n)*D**n / (K*8**(n-1))` -- the power-law
generalization of `Re = rho*v*D/mu` that collapses to the Newtonian
definition at n=1, K=mu.

Friction factor: `f = 16/Re_gen` in the laminar regime (Re_gen < 2100,
the same laminar/turbulent transition Reynolds number used for
Newtonian pipe flow -- consistent with the generalized-Re definition
being constructed so the laminar friction-factor relation keeps its
Newtonian form), `f = 0.0791*Re_gen**-0.25` in the turbulent regime
(Blasius correlation, extended to generalized Re per Metzner & Reed
(1955) -- Dodge & Metzner's more elaborate correlation is available
where a tighter transition-region fit is needed, not implemented here).

Pressure gradient: `dP/ds = rho*g*sin(inclination) + f*rho*v**2/(2*D_h)`
along a path coordinate `s` (hydrostatic term + Darcy-Weisbach-form
friction term using the generalized friction factor above), integrated
by explicit forward Euler -- consistent with the rest of this package's
"quasi-steady" 1D treatments (see `pipe_network_1d`): valid where flow
develops fast relative to the pressure-profile update rate, not for
acoustic/waterhammer transients.

Equivalent static density `rho_eq(s) = P(s) / (g * TVD(s))` (not
`P(s)/(g*s)`: for an inclined path, the true vertical depth `TVD`, not
the along-path length `s`, is what a hydrostatic-equivalent density
means physically -- passing `TVD = s` recovers the vertical-only
special case). Dimensionally this ratio is already kg/m^3 on its own
(Pa = kg*m^-1*s^-2, divided by g*TVD in m^2*s^-2 leaves kg/m^3): no
extra unit-conversion factor belongs in it, and none is applied here.

VALIDITY ENVELOPE: single-phase (or single effective mixture) Herschel-
Bulkley fluid, steady or slowly-varying flow, straight or gently curved
conduit describable by a hydraulic diameter + inclination profile. No
two-phase slip, no annular eccentricity effects, no acoustic transients.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def herschel_bulkley_stress(tau_y: ArrayLike, K: ArrayLike, n: ArrayLike, gamma_dot: ArrayLike) -> ArrayLike:
    """tau = tau_y + K*gamma_dot**n. `gamma_dot` is floored at 1e-9 s^-1."""
    gd = np.maximum(np.asarray(gamma_dot, dtype=float), 1e-9)
    return tau_y + K * gd ** n


def effective_viscosity(tau_y: ArrayLike, K: ArrayLike, n: ArrayLike, gamma_dot: ArrayLike) -> ArrayLike:
    """mu_eff = tau/gamma_dot = tau_y/gamma_dot + K*gamma_dot**(n-1)."""
    gd = np.maximum(np.asarray(gamma_dot, dtype=float), 1e-9)
    return tau_y / gd + K * gd ** (n - 1.0)


def generalized_reynolds_number(rho: ArrayLike, v: ArrayLike, D: ArrayLike, K: ArrayLike, n: ArrayLike) -> ArrayLike:
    """Re_gen = rho*|v|**(2-n)*D**n / (K*8**(n-1)); laminar if Re_gen < 2100."""
    v_abs = np.abs(np.asarray(v, dtype=float)).clip(min=1e-9)
    return rho * v_abs ** (2 - n) * D ** n / (K * 8 ** (n - 1) + 1e-12)


def metzner_reed_friction_factor(Re_gen: ArrayLike) -> ArrayLike:
    """f = 16/Re_gen (laminar, Re_gen < 2100) or 0.0791*Re_gen**-0.25
    (turbulent, Blasius/Metzner-Reed)."""
    Re = np.asarray(Re_gen, dtype=float)
    return np.where(Re < 2100.0, 16.0 / (Re + 1e-8), 0.0791 * (Re + 1e-8) ** (-0.25))


def pressure_gradient(rho: ArrayLike, v: ArrayLike, D_h: ArrayLike, f: ArrayLike,
                       inclination_rad: ArrayLike = np.pi / 2, g: float = 9.81) -> ArrayLike:
    """dP/ds = rho*g*sin(inclination) + f*rho*v**2/(2*D_h). `inclination_rad`
    is measured from horizontal (pi/2 = vertical, the default)."""
    return rho * g * np.sin(inclination_rad) + f * rho * v ** 2 / (2.0 * D_h + 1e-12)


@dataclass(frozen=True)
class PressureProfile:
    s_m: np.ndarray
    tvd_m: np.ndarray
    P_Pa: np.ndarray
    rho_eq_kg_m3: np.ndarray
    rho_kg_m3: np.ndarray
    friction_factor: np.ndarray
    reynolds_number: np.ndarray


def integrate_pressure_profile(
    length_m: float,
    v_m_s: float,
    D_h_m: float,
    K_Pa_sn: float,
    n_flow_index: float,
    P_inlet_Pa: float,
    rho_profile: Union[float, Callable[[float], float]] = 1000.0,
    inclination_deg_profile: Union[float, Callable[[float], float]] = 90.0,
    tvd_profile: Optional[Callable[[float], float]] = None,
    g: float = 9.81,
    n_steps: int = 300,
) -> PressureProfile:
    """Steady-state 1D pressure integration along path length `s` in
    [0, length_m], by explicit forward Euler on `dP/ds` from
    `pressure_gradient`. `rho_profile`/`inclination_deg_profile` may be a
    constant or a callable `f(s_m) -> value`. `tvd_profile(s_m)` supplies
    true vertical depth for the equivalent-density calculation; if not
    given, TVD is obtained by integrating `sin(inclination)` alongside
    pressure (exact for a vertical path, where TVD == s)."""
    s_arr = np.linspace(0.0, length_m, n_steps + 1)
    ds = length_m / n_steps
    P = np.zeros(n_steps + 1)
    tvd = np.zeros(n_steps + 1)
    rho_arr = np.zeros(n_steps + 1)
    f_arr = np.zeros(n_steps + 1)
    Re_arr = np.zeros(n_steps + 1)
    P[0] = P_inlet_Pa

    def _at(profile, s):
        return profile(s) if callable(profile) else float(profile)

    for i in range(n_steps):
        s_i = s_arr[i]
        rho_i = _at(rho_profile, s_i)
        inc_rad_i = np.radians(_at(inclination_deg_profile, s_i))
        Re_i = generalized_reynolds_number(rho_i, v_m_s, D_h_m, K_Pa_sn, n_flow_index)
        f_i = metzner_reed_friction_factor(Re_i)
        dPds = pressure_gradient(rho_i, v_m_s, D_h_m, f_i, inc_rad_i, g)
        P[i + 1] = P[i] + dPds * ds
        tvd[i + 1] = tvd[i] + np.sin(inc_rad_i) * ds if tvd_profile is None else _at(tvd_profile, s_arr[i + 1])
        rho_arr[i] = rho_i
        f_arr[i] = f_i
        Re_arr[i] = Re_i

    rho_arr[-1], f_arr[-1], Re_arr[-1] = rho_arr[-2], f_arr[-2], Re_arr[-2]
    tvd_safe = np.maximum(tvd, 0.1)
    rho_eq = P / (g * tvd_safe) * 1.0

    return PressureProfile(
        s_m=s_arr, tvd_m=tvd, P_Pa=P, rho_eq_kg_m3=rho_eq,
        rho_kg_m3=rho_arr, friction_factor=f_arr, reynolds_number=Re_arr,
    )
