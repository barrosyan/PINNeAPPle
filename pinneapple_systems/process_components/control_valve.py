"""pinneapple_systems.process_components.control_valve — compressible
and incompressible control-valve flow sizing per IEC 60534-2-1 (harmonized
with ANSI/ISA-75.01.01), plus actuator response and closed-seat leakage.

SELECTED FORMULATION: IEC 60534-2-1:2011 sizing equations for
compressible fluids:

    x   = dP / P1                              (pressure-drop ratio)
    Fk  = k / 1.40                              (specific-heat-ratio factor)
    Y   = 1 - x / (3 Fk xT)   for x <= Fk*xT      (expansion factor, unchoked)
    Y   = 2/3                  for x >  Fk*xT      (choked)
    w   = Cv * N6 * Y * sqrt(x * P1 * rho1)        (mass flow)

N6 = 2.73 (w in kg/h, P in kPa, rho in kg/m3, Cv in US gpm/psi^0.5 units)
per the IEC/ISA numerical-constant tables, converted here to coherent SI
so `compressible_mass_flow` takes/returns SI units directly. Cross-checked
independently from the liquid-service Cv DEFINITION (Q[gpm] =
Cv*sqrt(dP[psi]/SG)), converted through 1 US gpm = 6.30902e-5 m3/s, 1 psi
= 6894.757 Pa and a 1000 kg/m3 reference density: agreement to ~0.2%
between the two independent derivations (the standard's own table value
vs. a from-first-principles unit conversion), which is the confidence
check for this constant.

The installed Cv itself follows the valve's inherent characteristic --
`linear` or `equal_percentage` (equal-percentage's better rangeability
near small openings is standard practice for throttling/trim service).

ASSUMPTIONS: adiabatic single-phase flow through the trim; no piping
geometry factor Fp (adequate straight-run assumed); actuator response is
a first-order lag on commanded travel (captures the dominant dynamic --
stroke time -- without positioner-loop detail); closed-seat leakage is
represented as an effective leakage Cv that persists even at zero
commanded travel, exposed as an explicit, separately settable parameter
so a "passing while commanded closed" condition can be modeled and
(elsewhere, e.g. via counterfactual_attribution) inferred from
measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# US gpm/psi^0.5 -> SI mass-flow sizing coefficient: w[kg/s] = Cv_SI * Y *
# sqrt(x * P1[Pa] * rho1[kg/m3]). See module docstring for the
# cross-derivation this constant was checked against.
CV_US_GPM_TO_SI = 2.73 / (3600.0 * np.sqrt(1000.0))


@dataclass(frozen=True)
class ValveSpec:
    name: str
    Cv_max_us: float
    xT: float = 0.60
    characteristic: Literal["equal_percentage", "linear"] = "equal_percentage"
    equal_percentage_R: float = 50.0
    actuator_time_constant_s: float = 4.0
    leakage_cv_us: float = 0.0


@dataclass(frozen=True)
class ValveFlowResult:
    mass_flow_kg_s: float
    choked: bool
    x: float
    Y: float
    Cv_effective_us: float


def installed_cv(travel_fraction: float, spec: ValveSpec) -> float:
    """Cv at a given travel fraction (0=closed, 1=full open), per the
    inherent characteristic -- excludes leakage, see `effective_cv`."""
    t = float(np.clip(travel_fraction, 0.0, 1.0))
    if spec.characteristic == "linear":
        return spec.Cv_max_us * t
    if spec.characteristic == "equal_percentage":
        return float(spec.Cv_max_us * spec.equal_percentage_R ** (t - 1.0))
    raise ValueError(f"unknown characteristic {spec.characteristic!r}")


def effective_cv(travel_fraction: float, spec: ValveSpec, leakage_cv_us_override: float | None = None) -> float:
    """Cv actually passing flow, including the closed-seat leakage floor
    -- pass `leakage_cv_us_override` to represent a TRUE (e.g. degraded)
    leakage different from the spec's nameplate value."""
    leak = spec.leakage_cv_us if leakage_cv_us_override is None else leakage_cv_us_override
    return max(installed_cv(travel_fraction, spec), leak)


def compressible_mass_flow(Cv_us: float, P1_Pa: float, P2_Pa: float, rho1_kg_m3: float, k_isentropic: float, xT: float) -> ValveFlowResult:
    if P2_Pa > P1_Pa:
        raise ValueError("P2 (downstream) must not exceed P1 (upstream) for forward valve flow")
    x = (P1_Pa - P2_Pa) / P1_Pa
    Fk = k_isentropic / 1.40
    x_choke = Fk * xT
    choked = x >= x_choke
    Y = 2.0 / 3.0 if choked else (1.0 - x / (3.0 * x_choke) if x_choke > 0 else 0.0)
    x_eff = x_choke if choked else x
    w = Cv_us * CV_US_GPM_TO_SI * Y * np.sqrt(max(x_eff, 0.0) * P1_Pa * rho1_kg_m3)
    return ValveFlowResult(mass_flow_kg_s=float(w), choked=choked, x=float(x), Y=float(Y), Cv_effective_us=Cv_us)


def incompressible_mass_flow(Cv_us: float, P1_Pa: float, P2_Pa: float, rho_kg_m3: float) -> float:
    """Liquid-service sizing (no expansion factor / choking model beyond
    simple flashing checks are out of scope here) -- w = Cv_SI * sqrt(dP * rho)."""
    if P2_Pa > P1_Pa:
        raise ValueError("P2 (downstream) must not exceed P1 (upstream)")
    dP = P1_Pa - P2_Pa
    return float(Cv_us * CV_US_GPM_TO_SI * np.sqrt(max(dP, 0.0) * rho_kg_m3))


def actuator_response_rhs(travel_actual: float, travel_command: float, spec: ValveSpec) -> float:
    """d(travel_actual)/dt for the first-order actuator lag."""
    return (travel_command - travel_actual) / spec.actuator_time_constant_s
