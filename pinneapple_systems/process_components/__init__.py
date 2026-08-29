"""pinneapple_systems.process_components — physical unit-operation models
for process/plant digital twins: real-gas thermodynamic properties,
turbomachinery (polytropic compression/expansion + nondimensional
similarity maps), control valves, heat exchangers, and 1D pipe networks.

Unlike ``pinneapple_systems.component_modeling`` (generic ML/control
tooling that operates on any differentiable model, independent of what it
represents physically), everything here IS domain physics -- real
equations of state and unit-operation models a plant/process digital
twin assembles into a larger system. Each module is independently usable
and makes no assumption about the larger system it might be part of (no
shared "plant" or "train" base class, no registry) -- compose them
directly in application code.

Modules
-------
real_gas_eos      Multi-component real-gas mixture properties (GERG-2008
                   via CoolProp's HEOS backend), reference-condition
                   bookkeeping, and a central-difference derivative
                   helper for use outside an autograd graph.
polytropic_path    Real-gas polytropic compression/expansion path
                   integration (ASME PTC 10-consistent direct/reference
                   integration), works for compressors or expanders.
similarity_map     Nondimensional turbomachinery performance-map
                   utilities (flow/head coefficients, Mach/Reynolds
                   corrections, surge/choke margins, multistage scaling).
control_valve      IEC 60534-2-1 compressible/incompressible control-
                   valve sizing, actuator lag, closed-seat leakage.
heat_exchanger     Effectiveness-NTU heat exchanger (counter-flow /
                   parallel-flow), steady-state and lumped-capacitance
                   transient.
pipe_network_1d    1D real-gas pipe flow: quasi-steady momentum +
                   transient continuity (finite-volume), a rapid
                   steady-state scenario mode, Colebrook-White friction.
explicit_equation_system  A generic, sandboxed "define your model as
                   named equations" engine (safe AST expression
                   evaluation, symbol dependency-graph resolution,
                   bounded least-squares calibration against measured
                   data) -- any explicit-formula model (a rheology
                   curve, a valve-sizing correlation, ...) can be
                   expressed as symbols + equation strings against this
                   engine instead of hand-written evaluation code.
beam_statics       Closed-form static linear Euler-Bernoulli beam
                   deflection/slope/moment/shear/stress solutions for
                   the standard cantilever and simply-supported load
                   cases (uniform, partial-uniform, triangular loads).
beam_nonlinear_fem Static, geometrically nonlinear (Von Karman) beam FEM
                   via load-stepped Newton-Raphson -- the large-
                   deflection generalization of beam_statics, at the
                   static limit of the transient spectral-element engine
                   in pinneapple_simulation.numerical_solvers.
                   nonlinear_beam_fem.
reaction_kinetics  Generic multi-species mass-action reaction-network
                   engine (any chemistry -- disinfection, combustion,
                   biochemical pathways, ...), Arrhenius/quadratic-T and
                   Henderson-Hasselbalch/Bjerrum pH-speciation helpers,
                   a stiff-ODE integration wrapper, and a 1D advection-
                   dispersion-reaction transport solver built on the
                   same network object.
non_newtonian_pipe_flow  Herschel-Bulkley rheology, the generalized
                   (Metzner-Reed) Reynolds number and friction factor
                   for power-law/yield-stress fluids, and steady-state
                   1D pressure-drop integration along an inclined
                   conduit with a hydrostatic-equivalent-density result.
curved_path_geometry  The "build-and-hold" curved-conduit path
                   generator (vertical section, constant-curvature
                   build, tangent/hold section) -- true vertical depth
                   and horizontal displacement from an inclination
                   profile, for a wellbore, curved pipeline routing, or
                   any similarly-shaped conduit.
pipe_stress_mechanics  Combined-load stress analysis for a slender tube
                   (Lame thick-wall hoop stress, torsional shear,
                   curvature-based bending stress, triaxial Von Mises),
                   Euler column buckling, and constrained-rod (Paslay-
                   Dawson) buckling of a slender member confined within
                   a surrounding cylindrical clearance.
torsional_stickslip  1D torsional-wave finite-difference simulation of
                   a compliant rotating shaft with Stribeck friction at
                   one end -- the general mechanism behind torsional
                   stick-slip in any long driveline with a friction
                   load.
fatigue_analysis  S-N (Wohler) curve fatigue life, Goodman mean-stress
                   correction, and Miner's-rule cumulative damage
                   accumulation for cyclic structural loading.

None of these modules depends on any of the others except through the
shared ``real_gas_eos.GasComposition``/``GasState`` types -- use only the
subset a given application needs.
"""
from __future__ import annotations

from .beam_statics import (
    BeamResult,
    rectangular_section_properties,
    solve_beam,
    von_mises_stress_rectangular_section,
)
from .beam_nonlinear_fem import (
    StaticNonlinearBeamResult,
    solve_nonlinear_beam_static,
)
from .non_newtonian_pipe_flow import (
    PressureProfile,
    effective_viscosity as non_newtonian_effective_viscosity,
    generalized_reynolds_number,
    herschel_bulkley_stress,
    integrate_pressure_profile,
    metzner_reed_friction_factor,
    pressure_gradient as non_newtonian_pressure_gradient,
)
from .curved_path_geometry import (
    CurvedPathProfile,
    build_and_hold_profile,
    circular_arc_tvd_hd,
    inclination_at_depth,
)
from .pipe_stress_mechanics import (
    ConstrainedRodBucklingResult,
    RotatingBendingCycleResult,
    beam_column_moment_amplification_factor,
    bending_stress_from_curvature,
    classify_buckling_mode,
    constrained_rod_buckling_load,
    euler_critical_buckling_load,
    lame_hoop_stress_inner,
    lame_hoop_stress_outer,
    rotating_bending_stress_cycle,
    torsional_shear_stress,
    von_mises_triaxial,
)
from .torsional_stickslip import (
    StickSlipResult,
    simulate_torsional_stickslip,
    stribeck_friction_torque,
)
from .fatigue_analysis import (
    MinerDamageResult,
    goodman_equivalent_amplitude,
    goodman_safety_ratio,
    miners_rule_damage,
    sn_curve_cycles_to_failure,
)
from .control_valve import (
    ValveSpec,
    ValveFlowResult,
    compressible_mass_flow,
    incompressible_mass_flow,
    installed_cv,
    effective_cv,
    actuator_response_rhs,
)
from .heat_exchanger import (
    HeatExchangerSpec,
    HeatExchangerResult,
    steady_state as heat_exchanger_steady_state,
    transient_rhs as heat_exchanger_transient_rhs,
)
from .pipe_network_1d import (
    PipeSpec,
    PipeState,
    SteadyProfilePoint,
    TransientPipe,
    colebrook_white_f,
    rapid_steady_state_profile,
)
from .reaction_kinetics import (
    AdvectionDispersionReactionSolver,
    IntegrationResult,
    Reaction,
    ReactionNetwork,
    acid_fraction,
    arrhenius_rate_constant,
    base_fraction,
    diprotic_fractions,
    integrate_network,
    linear_combination_rate_constant,
    mass_action_rate,
    quadratic_in_T,
)
from .explicit_equation_system import (
    AnalysisResult,
    CalibrationResult,
    Definition,
    ExplicitEquationError,
    ParameterSpec,
    analyze,
    build_definitions,
    calibrate,
    evaluate,
    evaluation_order,
    safe_eval,
)
from .polytropic_path import (
    PolytropicPathResult,
    solve_path_from_pressure_ratio,
    solve_path_from_work,
)
from .real_gas_eos import (
    GasComposition,
    GasState,
    OutOfEnvelopeError,
    StandardConditions,
    ValidityEnvelope,
    central_difference,
    mass_flow_to_standard_volumetric_flow,
    standard_volumetric_flow_to_mass_flow,
    state_from_Ph,
    state_from_Ps,
    state_from_PT,
)
from .similarity_map import (
    MapCoefficients,
    MapEvaluation,
    evaluate_map,
    flow_coefficient,
    make_map,
    polytropic_head_from_psi,
    required_speed_for_head,
    tip_mach_number,
    tip_speed_m_s,
)

__all__ = [
    # beam_statics
    "BeamResult", "solve_beam", "rectangular_section_properties", "von_mises_stress_rectangular_section",
    # beam_nonlinear_fem
    "StaticNonlinearBeamResult", "solve_nonlinear_beam_static",
    # non_newtonian_pipe_flow
    "PressureProfile", "herschel_bulkley_stress", "non_newtonian_effective_viscosity",
    "generalized_reynolds_number", "metzner_reed_friction_factor",
    "non_newtonian_pressure_gradient", "integrate_pressure_profile",
    # curved_path_geometry
    "CurvedPathProfile", "inclination_at_depth", "build_and_hold_profile", "circular_arc_tvd_hd",
    # pipe_stress_mechanics
    "ConstrainedRodBucklingResult", "RotatingBendingCycleResult",
    "lame_hoop_stress_outer", "lame_hoop_stress_inner",
    "torsional_shear_stress", "bending_stress_from_curvature", "von_mises_triaxial",
    "euler_critical_buckling_load", "constrained_rod_buckling_load", "classify_buckling_mode",
    "beam_column_moment_amplification_factor", "rotating_bending_stress_cycle",
    # torsional_stickslip
    "StickSlipResult", "stribeck_friction_torque", "simulate_torsional_stickslip",
    # fatigue_analysis
    "MinerDamageResult", "sn_curve_cycles_to_failure", "goodman_equivalent_amplitude",
    "goodman_safety_ratio", "miners_rule_damage",
    # explicit_equation_system
    "ExplicitEquationError", "Definition", "AnalysisResult", "ParameterSpec", "CalibrationResult",
    "safe_eval", "build_definitions", "evaluation_order", "evaluate", "analyze", "calibrate",
    # real_gas_eos
    "GasComposition", "GasState", "OutOfEnvelopeError", "StandardConditions", "ValidityEnvelope",
    "state_from_PT", "state_from_Ph", "state_from_Ps", "central_difference",
    "standard_volumetric_flow_to_mass_flow", "mass_flow_to_standard_volumetric_flow",
    # polytropic_path
    "PolytropicPathResult", "solve_path_from_pressure_ratio", "solve_path_from_work",
    # similarity_map
    "MapCoefficients", "MapEvaluation", "make_map", "evaluate_map",
    "flow_coefficient", "tip_speed_m_s", "tip_mach_number",
    "polytropic_head_from_psi", "required_speed_for_head",
    # control_valve
    "ValveSpec", "ValveFlowResult", "installed_cv", "effective_cv",
    "compressible_mass_flow", "incompressible_mass_flow", "actuator_response_rhs",
    # heat_exchanger
    "HeatExchangerSpec", "HeatExchangerResult", "heat_exchanger_steady_state", "heat_exchanger_transient_rhs",
    # pipe_network_1d
    "PipeSpec", "PipeState", "SteadyProfilePoint", "TransientPipe",
    "colebrook_white_f", "rapid_steady_state_profile",
    # reaction_kinetics
    "Reaction", "ReactionNetwork", "IntegrationResult", "AdvectionDispersionReactionSolver",
    "mass_action_rate", "arrhenius_rate_constant", "quadratic_in_T",
    "linear_combination_rate_constant", "acid_fraction", "base_fraction",
    "diprotic_fractions", "integrate_network",
]
