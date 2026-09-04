"""pinneapple_systems.process_components.real_gas_eos — multi-component
real-gas mixture properties via a Helmholtz-energy equation of state.

SELECTED FORMULATION: the mixture Helmholtz-energy EOS exposed by
CoolProp's HEOS backend. For hydrocarbons, N2, CO2, H2S, H2O and the
other GERG-2008 components, CoolProp's HEOS mixture model uses the Kunz &
Wagner GERG-2008 pure-fluid and departure (binary interaction)
correlations directly (Kunz & Wagner, "The GERG-2008 Wide-Range Equation
of State for Natural Gases and Other Mixtures", GERG TM15 / Ind. Eng.
Chem. Res. 2012, 51, 11890-11901) — this is the same physics ISO 20765-2
and AGA Report No. 8 Part 2 standardize for custody transfer, not a
weaker substitute. CoolProp's mixture results have been independently
validated against the NIST/REFPROP reference implementation in the
literature (Bell, Wronski, Quoilin & Lemort, Ind. Eng. Chem. Res. 2014,
53, 2498-2508). If a licensed NIST REFPROP install is available, set
``backend="REFPROP"`` — CoolProp's AbstractState interface is identical,
so nothing downstream of this module needs to change.

ASSUMPTIONS
-----------
- Single-phase region only. This module does not track a phase envelope
  or dew point; ``specify_phase="gas"`` (the default) both matches that
  scope and is a substantial performance optimization for compiled
  multi-component mixture flashes (see PERFORMANCE below) — pass
  ``specify_phase=None`` to fall back to CoolProp's own phase-stability
  logic if two-phase behavior is possible in your application.
- A single, explicitly declared reference condition governs every
  standard-volume conversion (`standard_conditions`); callers must state
  it rather than relying on an implicit default, since "standard
  conditions" varies by industry/region (metric ISO 13443 vs. US 60degF).

STATE VARIABLES: any two of (P, T, h, s) fully resolve a fixed-composition
single-phase state (Gibbs phase rule).

NUMERICAL METHOD: CoolProp's compiled Helmholtz-mixture Newton solve;
this wrapper owns only composition handling, unit bookkeeping, envelope
validation and (optionally) central-difference derivatives for use
outside an autograd graph (CoolProp's own solve is not
autodiff-transparent — see `central_difference` below).

PERFORMANCE: an unconstrained (phase-stability-checking) AbstractState
was measured, for a 6-component mixture, at roughly two to three orders
of magnitude slower per enthalpy/entropy-basis flash than the same call
with ``specify_phase`` set — several seconds vs. tens of milliseconds.
Any application doing many flashes per second (an ODE integrator's RHS,
a training-data generator) should confirm single-phase-gas is a valid
assumption for its envelope and set ``specify_phase`` accordingly; this
is not a micro-optimization, it is the difference between a solver that
finishes in seconds and one that does not finish in a session.

EXPECTED UNCERTAINTY (GERG-2008-covered components, per Kunz & Wagner
2012 Table A3 and ISO 20765-2's stated accuracy class, for a
well-characterized mixture within the correlation's validated range):
density/compressibility +/-0.1-0.3%, speed of sound +/-0.1-0.5%, enthalpy/
entropy differences +/-0.5-1%, viscosity (a fitted correlation layered on
the EOS density, not itself part of GERG-2008 proper) +/-2-5%.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

try:
    import CoolProp.CoolProp as CP
    from CoolProp.CoolProp import AbstractState
    _COOLPROP_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only if CoolProp isn't installed
    _COOLPROP_AVAILABLE = False

_PHASE_CONST = {"gas": "iphase_gas", "liquid": "iphase_liquid", "supercritical": "iphase_supercritical_gas"}


class OutOfEnvelopeError(ValueError):
    """Raised when a requested state falls outside a caller-declared validity envelope."""


@dataclass(frozen=True)
class GasComposition:
    """A fixed-composition mixture, mole fractions summing to 1.

    `components` must be valid CoolProp fluid names (e.g. "Methane",
    "Ethane", "Nitrogen", "CarbonDioxide", "Water", "HydrogenSulfide",
    ...) -- any subset CoolProp/GERG-2008 supports, not a fixed list.
    """

    components: tuple[str, ...]
    mole_fractions: tuple[float, ...]

    def __post_init__(self):
        if len(self.components) != len(self.mole_fractions):
            raise ValueError("components and mole_fractions must be the same length")
        total = sum(self.mole_fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"mole fractions must sum to 1.0, got {total}")

    def as_fluid_string(self) -> str:
        return "&".join(self.components)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.components, self.mole_fractions))


@dataclass(frozen=True)
class ValidityEnvelope:
    """Optional, caller-declared (P, T) bounds -- states outside it raise
    OutOfEnvelopeError instead of silently extrapolating. Pass None to
    `state_from_*`'s `envelope` argument to skip this check entirely."""

    P_min_Pa: float
    P_max_Pa: float
    T_min_K: float
    T_max_K: float

    def check(self, P_Pa: float, T_K: float) -> None:
        if not (self.P_min_Pa <= P_Pa <= self.P_max_Pa):
            raise OutOfEnvelopeError(f"P={P_Pa/1e5:.2f} bara outside [{self.P_min_Pa/1e5:.1f}, {self.P_max_Pa/1e5:.1f}] bara")
        if not (self.T_min_K <= T_K <= self.T_max_K):
            raise OutOfEnvelopeError(f"T={T_K:.1f} K outside [{self.T_min_K:.1f}, {self.T_max_K:.1f}] K")


@dataclass(frozen=True)
class GasState:
    """A fully-resolved thermodynamic state point, SI units throughout."""

    P_Pa: float
    T_K: float
    rho_kg_m3: float
    Z: float
    h_J_kg: float
    s_J_kgK: float
    cp_J_kgK: float
    cv_J_kgK: float
    k_isentropic: float
    speed_of_sound_m_s: float
    viscosity_Pa_s: float
    molar_mass_kg_mol: float

    @property
    def v_m3_kg(self) -> float:
        return 1.0 / self.rho_kg_m3


@lru_cache(maxsize=128)
def _get_abstract_state(backend: str, fluid_string: str, mole_fractions: tuple[float, ...], specify_phase: str | None):
    if not _COOLPROP_AVAILABLE:
        raise ImportError("CoolProp is required for pinneapple_systems.process_components.real_gas_eos (pip install CoolProp)")
    AS = AbstractState(backend, fluid_string)
    AS.set_mole_fractions(list(mole_fractions))
    if specify_phase is not None:
        AS.specify_phase(getattr(CP, _PHASE_CONST[specify_phase]))
    return AS


def _resolve(
    gas: GasComposition, backend: str, specify_phase: str | None,
    input_pair, v1: float, v2: float, envelope: ValidityEnvelope | None,
) -> GasState:
    AS = _get_abstract_state(backend, gas.as_fluid_string(), gas.mole_fractions, specify_phase)
    AS.update(input_pair, v1, v2)
    P_Pa, T_K = AS.p(), AS.T()
    if envelope is not None:
        envelope.check(P_Pa, T_K)
    rho = AS.rhomass()
    cp, cv = AS.cpmass(), AS.cvmass()
    return GasState(
        P_Pa=P_Pa, T_K=T_K, rho_kg_m3=rho, Z=AS.compressibility_factor(),
        h_J_kg=AS.hmass(), s_J_kgK=AS.smass(), cp_J_kgK=cp, cv_J_kgK=cv,
        k_isentropic=cp / cv, speed_of_sound_m_s=AS.speed_sound(),
        viscosity_Pa_s=AS.viscosity(), molar_mass_kg_mol=AS.molar_mass(),
    )


def _require_coolprop() -> None:
    """``CP.<X>_INPUTS`` is evaluated as a plain function argument at each
    ``state_from_*`` call site below, i.e. *before* ``_resolve``/
    ``_get_abstract_state`` ever runs -- so without this guard, a missing
    CoolProp install raised a bare ``NameError: name 'CP' is not defined``
    at the call site instead of the clear, actionable ``ImportError``
    ``_get_abstract_state`` already raises internally (found via the
    pre-existing test suite: every real-gas test failed with exactly that
    confusing NameError instead of a "pip install CoolProp" message)."""
    if not _COOLPROP_AVAILABLE:
        raise ImportError("CoolProp is required for pinneapple_systems.process_components.real_gas_eos (pip install CoolProp)")


def state_from_PT(
    gas: GasComposition, P_Pa: float, T_K: float, *,
    backend: str = "HEOS", specify_phase: Literal["gas", "liquid", "supercritical"] | None = "gas",
    envelope: ValidityEnvelope | None = None,
) -> GasState:
    _require_coolprop()
    return _resolve(gas, backend, specify_phase, CP.PT_INPUTS, P_Pa, T_K, envelope)


def state_from_Ph(
    gas: GasComposition, P_Pa: float, h_J_kg: float, *,
    backend: str = "HEOS", specify_phase: Literal["gas", "liquid", "supercritical"] | None = "gas",
    envelope: ValidityEnvelope | None = None,
) -> GasState:
    """Pressure + specific enthalpy -> full state (temperature recovered
    by EOS flash). Used wherever an energy balance produces h directly."""
    _require_coolprop()
    return _resolve(gas, backend, specify_phase, CP.HmassP_INPUTS, h_J_kg, P_Pa, envelope)


def state_from_Ps(
    gas: GasComposition, P_Pa: float, s_J_kgK: float, *,
    backend: str = "HEOS", specify_phase: Literal["gas", "liquid", "supercritical"] | None = "gas",
    envelope: ValidityEnvelope | None = None,
) -> GasState:
    """Pressure + specific entropy -> full state (an isentropic endpoint)."""
    _require_coolprop()
    return _resolve(gas, backend, specify_phase, CP.PSmass_INPUTS, P_Pa, s_J_kgK, envelope)


@dataclass(frozen=True)
class StandardConditions:
    P_Pa: float
    T_K: float

    @staticmethod
    def iso_13443() -> "StandardConditions":
        """101.325 kPa / 15degC -- the metric 'standard cubic metre' convention."""
        return StandardConditions(101_325.0, 288.15)

    @staticmethod
    def us_60f() -> "StandardConditions":
        """14.696 psia / 60degF -- the US oil & gas convention."""
        return StandardConditions(101_325.0, 288.706)


def standard_volumetric_flow_to_mass_flow(q_std_m3_s: float, gas: GasComposition, std: StandardConditions) -> float:
    ref = state_from_PT(gas, std.P_Pa, std.T_K)
    return q_std_m3_s * ref.rho_kg_m3


def mass_flow_to_standard_volumetric_flow(m_dot_kg_s: float, gas: GasComposition, std: StandardConditions) -> float:
    ref = state_from_PT(gas, std.P_Pa, std.T_K)
    return m_dot_kg_s / ref.rho_kg_m3


def central_difference(
    gas: GasComposition, P_Pa: float, T_K: float, attr: str, wrt: Literal["P", "T"], *, rel_step: float = 1e-4,
) -> float:
    """d(attr)/d(wrt) via central differences -- for use OUTSIDE an
    autograd graph (CoolProp's compiled solve is not autodiff-transparent;
    any training loop that needs a gradient through this EOS should call
    this instead of tracing through `state_from_*` directly)."""
    if wrt == "P":
        step = max(P_Pa * rel_step, 1.0)
        plus = state_from_PT(gas, P_Pa + step, T_K, envelope=None)
        minus = state_from_PT(gas, P_Pa - step, T_K, envelope=None)
    else:
        step = max(T_K * rel_step, 1e-3)
        plus = state_from_PT(gas, P_Pa, T_K + step, envelope=None)
        minus = state_from_PT(gas, P_Pa, T_K - step, envelope=None)
    return (getattr(plus, attr) - getattr(minus, attr)) / (2 * step)
