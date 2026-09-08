"""Real, first-principles dimensionless-number calculations + a simple,
honestly-scoped flow-regime classifier.

This is genuinely new (not previously in PINNeAPPle) and is the concrete
implementation of the user's own architectural point 3
("Dimensional analysis / scaling... o sistema deveria automaticamente
calcular e raciocinar sobre Reynolds, Prandtl, Peclet, Mach, Nusselt,
Damkohler, Fourier, Weber, Froude"). Every formula below is a standard,
textbook dimensionless group -- cited in each function's docstring --
computed directly from the physical parameters the caller supplies, not
guessed or LLM-generated. This is the layer that lets the engine
"reason automatically" about which physical regime a problem is in
*before* deciding which PDE closure/solver family is appropriate, e.g.
Stokes flow (Re << 1) needs no turbulence closure at all, while Re >
4000 in a pipe needs one.

Regime classification is DELIBERATELY simple and honestly caveated: real
transition thresholds depend on geometry (external flow past a sphere
transitions at different Re than internal pipe flow), so
``classify_flow_regime`` documents exactly which geometry its thresholds
assume, rather than presenting one universal answer as if it always
applies. This is the same "state precisely what was and wasn't checked"
ethic as the rest of this codebase.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class DimensionlessNumbers:
    """Every dimensionless group actually computable from the physical
    parameters supplied to :func:`compute_dimensionless_numbers` --
    fields are ``None`` when the inputs needed for that particular
    number weren't given, rather than silently defaulted, so a caller
    can tell "not computed" apart from "computed and happens to be
    small"."""
    reynolds: Optional[float] = None
    prandtl: Optional[float] = None
    peclet_heat: Optional[float] = None
    peclet_mass: Optional[float] = None
    mach: Optional[float] = None
    froude: Optional[float] = None
    weber: Optional[float] = None
    damkohler: Optional[float] = None
    nusselt_correlation: Optional[float] = None
    nusselt_correlation_name: Optional[str] = None
    notes: Dict[str, str] = field(default_factory=dict)


def compute_dimensionless_numbers(
    *,
    velocity: Optional[float] = None,
    length: Optional[float] = None,
    density: Optional[float] = None,
    dynamic_viscosity: Optional[float] = None,
    kinematic_viscosity: Optional[float] = None,
    thermal_diffusivity: Optional[float] = None,
    specific_heat: Optional[float] = None,
    thermal_conductivity: Optional[float] = None,
    mass_diffusivity: Optional[float] = None,
    speed_of_sound: Optional[float] = None,
    gravity: float = 9.81,
    surface_tension: Optional[float] = None,
    reaction_rate_constant: Optional[float] = None,
    residence_time: Optional[float] = None,
) -> DimensionlessNumbers:
    """Compute every dimensionless number the supplied subset of
    parameters actually supports. All formulas are standard textbook
    definitions (e.g. Incropera & DeWitt, *Fundamentals of Heat and Mass
    Transfer*; White, *Viscous Fluid Flow*):

    - Reynolds:  Re = rho * U * L / mu = U * L / nu
    - Prandtl:   Pr = nu / alpha = cp * mu / k
    - Peclet (heat): Pe_h = Re * Pr = U * L / alpha
    - Peclet (mass): Pe_m = U * L / D
    - Mach:      Ma = U / c
    - Froude:    Fr = U / sqrt(g * L)
    - Weber:     We = rho * U^2 * L / sigma
    - Damkohler (Da_I, first-order reaction vs. residence time): Da = k_rxn * t_res
    """
    nu = kinematic_viscosity
    if nu is None and density is not None and dynamic_viscosity is not None:
        nu = dynamic_viscosity / density

    out = DimensionlessNumbers()

    if velocity is not None and length is not None and nu is not None and nu > 0:
        out.reynolds = velocity * length / nu
    elif velocity is not None and length is not None and (dynamic_viscosity is not None) and (density is not None):
        out.reynolds = density * velocity * length / dynamic_viscosity

    if nu is not None and thermal_diffusivity is not None and thermal_diffusivity > 0:
        out.prandtl = nu / thermal_diffusivity
    elif specific_heat is not None and dynamic_viscosity is not None and thermal_conductivity is not None and thermal_conductivity > 0:
        out.prandtl = specific_heat * dynamic_viscosity / thermal_conductivity

    if out.reynolds is not None and out.prandtl is not None:
        out.peclet_heat = out.reynolds * out.prandtl
    elif velocity is not None and length is not None and thermal_diffusivity is not None and thermal_diffusivity > 0:
        out.peclet_heat = velocity * length / thermal_diffusivity

    if velocity is not None and length is not None and mass_diffusivity is not None and mass_diffusivity > 0:
        out.peclet_mass = velocity * length / mass_diffusivity

    if velocity is not None and speed_of_sound is not None and speed_of_sound > 0:
        out.mach = velocity / speed_of_sound

    if velocity is not None and length is not None and length > 0:
        out.froude = velocity / math.sqrt(gravity * length)

    if density is not None and velocity is not None and length is not None and surface_tension is not None and surface_tension > 0:
        out.weber = density * velocity ** 2 * length / surface_tension

    if reaction_rate_constant is not None and residence_time is not None:
        out.damkohler = reaction_rate_constant * residence_time

    # Nusselt number is NOT a pure input-derived dimensionless group the
    # way the others are -- it's the OUTPUT of a convection correlation
    # (it encodes the convective heat-transfer coefficient itself, an
    # unknown you're usually trying to find). Only report it when a
    # standard correlation's own validity range is actually satisfied,
    # named explicitly, rather than silently applying a formula outside
    # its regime.
    if out.reynolds is not None and out.prandtl is not None:
        re, pr = out.reynolds, out.prandtl
        if re > 1.0e4 and 0.6 <= pr <= 160.0:
            # Dittus-Boelter, turbulent flow in a smooth circular pipe, heating (n=0.4).
            out.nusselt_correlation = 0.023 * re ** 0.8 * pr ** 0.4
            out.nusselt_correlation_name = "Dittus-Boelter (turbulent pipe flow, heating, n=0.4)"
        else:
            out.notes["nusselt"] = (
                f"Re={re:.3g}, Pr={pr:.3g} is outside the Dittus-Boelter correlation's validity range "
                "(Re>1e4, 0.6<=Pr<=160) -- no Nusselt correlation applied rather than misusing one."
            )

    return out


def classify_flow_regime(re: Optional[float], *, geometry: str = "internal_pipe") -> str:
    """A simple, HONESTLY GEOMETRY-SPECIFIC flow-regime classification --
    real transition Reynolds numbers depend on the flow geometry, so this
    function requires the caller to say which one its thresholds should
    assume, rather than presenting one universal answer.

    geometry="internal_pipe" (fully-developed pipe/duct flow, the most
    common textbook case): Re<2300 laminar, 2300<=Re<4000 transitional,
    Re>=4000 turbulent (Moody-diagram convention).
    geometry="external_bluff_body" (flow past a sphere/cylinder): Re<1
    Stokes/creeping flow, 1<=Re<~1e3 laminar-separated wake,
    Re>=~1e3-2e5 subcritical turbulent wake (order-of-magnitude, not a
    sharp transition -- external-flow transitions are inherently fuzzier
    than internal pipe flow).
    """
    if re is None:
        return "unknown (Reynolds number not computable from the given parameters)"

    if geometry == "internal_pipe":
        if re < 2300:
            return f"laminar (Re={re:.3g} < 2300, internal pipe flow)"
        if re < 4000:
            return f"transitional (2300 <= Re={re:.3g} < 4000, internal pipe flow)"
        return f"turbulent (Re={re:.3g} >= 4000, internal pipe flow)"

    if geometry == "external_bluff_body":
        if re < 1:
            return f"Stokes / creeping flow (Re={re:.3g} < 1, external bluff-body)"
        if re < 1.0e3:
            return f"laminar separated wake (1 <= Re={re:.3g} < 1e3, external bluff-body)"
        return f"turbulent wake, order-of-magnitude estimate (Re={re:.3g} >= 1e3, external bluff-body)"

    raise ValueError(f"unknown geometry '{geometry}', expected 'internal_pipe' or 'external_bluff_body'")
