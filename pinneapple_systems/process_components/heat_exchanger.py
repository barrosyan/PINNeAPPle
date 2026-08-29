"""pinneapple_systems.process_components.heat_exchanger --
effectiveness-NTU heat exchanger model (counter-flow and parallel-flow
closed-form effectiveness), plus a lumped-capacitance transient response.

SELECTED FORMULATION: effectiveness-NTU (Incropera & DeWitt, "Fundamentals
of Heat and Mass Transfer") -- the standard method when the outlet
temperature is unknown and must be solved directly (vs. LMTD, which needs
an outlet-temperature guess):

    C_min = min(C_hot, C_cold), C_r = C_min/C_max
    NTU   = UA * f_fouling / C_min
    eps   = effectiveness(NTU, C_r, flow_arrangement)
    Q     = eps * C_min * (T_hot_in - T_cold_in)

Counter-flow (exact closed form, constant properties):
    eps = (1 - exp(-NTU(1-Cr))) / (1 - Cr*exp(-NTU(1-Cr)))   for Cr < 1
    eps = NTU / (1 + NTU)                                       for Cr = 1

Parallel-flow (exact closed form):
    eps = (1 - exp(-NTU(1+Cr))) / (1 + Cr)

ASSUMPTIONS: side-averaged specific heats (adequate for a modest
per-pass temperature change; NOT adequate across a wide-temperature-
range real-gas process where cp varies strongly -- for that, integrate
the real-gas EOS directly along the path instead, as
`polytropic_path.py` does for compression). Transient response uses a
single lumped thermal mass (metal + fluid holdup) relaxing toward the
instantaneous steady-state effectiveness-NTU solution -- adequate when
the exchanger's own thermal time constant is much faster than the
forecast horizon of interest but not instantaneous either.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class HeatExchangerSpec:
    name: str
    UA_design_W_K: float
    design_dP_Pa: float
    design_m_dot_hot_kg_s: float
    design_rho_hot_kg_m3: float
    thermal_mass_J_K: float
    cold_side_capacity_rate_W_K: float
    flow_arrangement: Literal["counterflow", "parallelflow"] = "counterflow"


@dataclass(frozen=True)
class HeatExchangerResult:
    Q_W: float
    T_hot_out_K: float
    T_cold_out_K: float
    effectiveness: float
    NTU: float
    dP_Pa: float
    UA_effective_W_K: float


def _effectiveness(NTU: float, Cr: float, arrangement: str) -> float:
    if NTU <= 0:
        return 0.0
    if arrangement == "parallelflow":
        return (1.0 - np.exp(-NTU * (1.0 + Cr))) / (1.0 + Cr)
    if Cr >= 0.9999:
        return NTU / (1.0 + NTU)
    exp_term = np.exp(-NTU * (1.0 - Cr))
    return (1.0 - exp_term) / (1.0 - Cr * exp_term)


def steady_state(
    spec: HeatExchangerSpec, m_dot_hot_kg_s: float, T_hot_in_K: float, cp_hot_J_kgK: float,
    T_cold_in_K: float, rho_hot_kg_m3: float, UA_multiplier: float = 1.0,
) -> HeatExchangerResult:
    if m_dot_hot_kg_s <= 0:
        return HeatExchangerResult(0.0, T_hot_in_K, T_cold_in_K, 0.0, 0.0, 0.0, spec.UA_design_W_K * UA_multiplier)

    UA_eff = spec.UA_design_W_K * UA_multiplier
    C_hot = m_dot_hot_kg_s * cp_hot_J_kgK
    C_cold = spec.cold_side_capacity_rate_W_K
    C_min, C_max = min(C_hot, C_cold), max(C_hot, C_cold)
    Cr = C_min / C_max if C_max > 0 else 0.0
    NTU = UA_eff / C_min if C_min > 0 else 0.0
    eps = _effectiveness(NTU, Cr, spec.flow_arrangement)

    Q = eps * C_min * (T_hot_in_K - T_cold_in_K)
    T_hot_out = T_hot_in_K - Q / C_hot
    T_cold_out = T_cold_in_K + Q / C_cold if C_cold > 0 else T_cold_in_K

    K = spec.design_dP_Pa / (spec.design_m_dot_hot_kg_s ** 2 / spec.design_rho_hot_kg_m3) if spec.design_m_dot_hot_kg_s > 0 else 0.0
    dP = K * (m_dot_hot_kg_s ** 2 / rho_hot_kg_m3) if rho_hot_kg_m3 > 0 else 0.0

    return HeatExchangerResult(
        Q_W=float(Q), T_hot_out_K=float(T_hot_out), T_cold_out_K=float(T_cold_out),
        effectiveness=float(eps), NTU=float(NTU), dP_Pa=float(dP), UA_effective_W_K=float(UA_eff),
    )


def transient_rhs(
    T_hot_out_current_K: float, spec: HeatExchangerSpec, m_dot_hot_kg_s: float, T_hot_in_K: float,
    cp_hot_J_kgK: float, T_cold_in_K: float, rho_hot_kg_m3: float, UA_multiplier: float = 1.0,
) -> float:
    """d(T_hot_out)/dt -- relaxes toward the instantaneous steady-state
    effectiveness-NTU outlet temperature with a time constant set by
    thermal_mass_J_K / (m_dot * cp)."""
    ss = steady_state(spec, m_dot_hot_kg_s, T_hot_in_K, cp_hot_J_kgK, T_cold_in_K, rho_hot_kg_m3, UA_multiplier)
    C_hot = max(m_dot_hot_kg_s * cp_hot_J_kgK, 1e-6)
    tau = spec.thermal_mass_J_K / C_hot
    return (ss.T_hot_out_K - T_hot_out_current_K) / tau
