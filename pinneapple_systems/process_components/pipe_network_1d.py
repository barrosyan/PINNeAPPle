"""pinneapple_systems.process_components.pipe_network_1d — 1D real-gas
pipe flow: a rapid steady-state scenario mode and a transient
finite-volume mode, covering a single pipe segment.

SELECTED FORMULATION -- quasi-steady momentum, transient continuity
---------------------------------------------------------------------
The standard simplification in gas-pipeline transient/linepack
simulation practice (Wylie & Streeter, "Fluid Transients in Systems";
the same basis underlying industrial pipeline simulators for
slow-transient/linepack studies, as distinct from millisecond-scale
acoustic water-hammer studies): the full 1D momentum equation's local/
convective-acceleration term is negligible for subsonic flow unless
resolving millisecond-scale acoustic transients. Dropping that term
collapses momentum to its quasi-steady (algebraic) form at every
instant, while mass conservation (continuity) stays fully transient --
pressure and flow still propagate cell-to-cell at the speed set by the
continuity equation's own numerics, not instantaneously.

Governing equations, per finite-volume cell i (real-gas EOS evaluated
via `real_gas_eos`):

    Continuity (transient):        dM_i/dt = (G_{i-1/2} - G_{i+1/2}) * A
    Quasi-steady momentum:          P_i^2 - P_{i+1}^2 - 2*rho_avg*g*dz*P_avg
                                       = (f*L/D) * G|G| * Z*R*T
    Colebrook-White (Darcy f):      1/sqrt(f) = -2 log10(eps/(3.7D) + 2.51/(Re*sqrt(f)))
    Energy (lumped, per cell):      enthalpy advection + wall heat loss (see ASSUMPTIONS)

ASSUMPTIONS: no local/convective acceleration term (valid for subsonic
flow over a slow-transient horizon, not for acoustic/water-hammer
events); linearized elevation correction (adequate for modest elevation
change; a long, high-relief route needs the full exponential correction
instead); energy equation tracks enthalpy advection + wall heat loss
only (kinetic/flow-work terms neglected, standard for subsonic gas
pipelines); flow assumed unidirectional (the energy equation's upstream-
temperature lookup does not handle a face-flow sign reversal).

DISCRETIZATION NOTE (a correctness bug caught and fixed during this
module's own validation): the rapid steady-state marcher returns NODE
values (x=0, dx, 2dx, ..., L), while the transient finite-volume solver
needs CELL-CENTER values (cell i spans [i*dx, (i+1)*dx]). Naively reusing
a node value as if it were a cell-center value introduces a silent
half-cell discretization offset that shows up as a spurious mass-
conservation drift in transient runs -- `TransientPipe.
initialize_from_steady_state` averages each cell's two bounding node
values instead, and the outlet face's momentum closure uses half the
cell length (not the full cell length used for internal faces), since
the last cell's center sits only dx/2 from the true outlet boundary.

BOUNDARY CONDITIONS: inlet mass flow and outlet pressure -- a standard
well-posed pair for this hyperbolic-parabolic system.

NUMERICAL METHOD: the rapid mode marches cell-to-cell algebraically (no
time integration). The transient mode integrates with
scipy.integrate.solve_ivp (RK45), with the quasi-steady face flux
re-solved (fixed-point on the Colebrook friction factor) at every
evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from pinneapple_systems.process_components.real_gas_eos import GasComposition, state_from_PT

G_EARTH = 9.80665


@dataclass(frozen=True)
class PipeSpec:
    name: str
    length_m: float
    diameter_m: float
    roughness_m: float
    elevation_change_m: float = 0.0
    ambient_temperature_K: float = 288.15
    wall_heat_transfer_coeff_W_m2K: float = 5.0
    n_cells: int = 20
    friction_multiplier: float = 1.0


def colebrook_white_f(Re: float, relative_roughness: float) -> float:
    """Darcy friction factor -- laminar closed form below Re=2300, else
    Colebrook-White, solved via Newton iteration on 1/sqrt(f)."""
    Re = max(Re, 1.0)
    if Re < 2300.0:
        return 64.0 / Re

    x = 1.0 / np.sqrt(0.02)

    def g(x):
        return x + 2.0 * np.log10(relative_roughness / 3.7 + 2.51 * x / Re)

    def gprime(x):
        h = 1e-6
        return (g(x + h) - g(x - h)) / (2 * h)

    for _ in range(50):
        step = g(x) / gprime(x)
        x -= step
        if abs(step) < 1e-10:
            break
    return 1.0 / x ** 2


def face_reynolds_number(G_kg_m2s: float, D_m: float, viscosity_Pa_s: float) -> float:
    return abs(G_kg_m2s) * D_m / viscosity_Pa_s


def _face_mass_flux(P1, P2, L_m, D_m, roughness_m, Z_avg, T_avg_K, R_specific, viscosity_Pa_s, rho_avg_kg_m3, dz_m, friction_multiplier) -> tuple[float, float]:
    elevation_term = rho_avg_kg_m3 * G_EARTH * dz_m * 0.5 * (P1 + P2)
    driving = P1 ** 2 - P2 ** 2 - 2.0 * elevation_term
    sign = 1.0 if driving >= 0 else -1.0
    driving_abs = abs(driving)

    f = 0.02
    rel_rough = roughness_m / D_m
    for _ in range(20):
        denom = f * friction_multiplier * L_m * Z_avg * R_specific * T_avg_K / D_m
        if denom <= 0:
            return 0.0, f
        G = sign * np.sqrt(driving_abs / denom)
        Re = face_reynolds_number(G, D_m, viscosity_Pa_s)
        f_new = colebrook_white_f(Re, rel_rough)
        if abs(f_new - f) < 1e-8:
            f = f_new
            break
        f = f_new
    denom = f * friction_multiplier * L_m * Z_avg * R_specific * T_avg_K / D_m
    G = sign * np.sqrt(driving_abs / denom) if denom > 0 else 0.0
    return G, f


@dataclass(frozen=True)
class SteadyProfilePoint:
    x_m: float
    P_Pa: float
    T_K: float
    rho_kg_m3: float
    G_kg_m2s: float
    mach: float


def rapid_steady_state_profile(spec: PipeSpec, gas: GasComposition, m_dot_kg_s: float, P_in_Pa: float, T_in_K: float) -> list[SteadyProfilePoint]:
    """Marches downstream cell-by-cell from the known inlet, solving each
    face's momentum closure explicitly (no time integration)."""
    A = np.pi * spec.diameter_m ** 2 / 4.0
    dx = spec.length_m / spec.n_cells
    dz = spec.elevation_change_m / spec.n_cells
    G = m_dot_kg_s / A

    P, T = P_in_Pa, T_in_K
    st0 = state_from_PT(gas, P, T)
    profile = [SteadyProfilePoint(0.0, P, T, st0.rho_kg_m3, G, G / st0.rho_kg_m3 / st0.speed_of_sound_m_s)]

    for i in range(spec.n_cells):
        st = state_from_PT(gas, P, T)
        R_specific = 8.314462618 / st.molar_mass_kg_mol

        def residual(P2, P=P, T=T, st=st, R_specific=R_specific):
            G_pred, _f = _face_mass_flux(P, P2, dx, spec.diameter_m, spec.roughness_m, st.Z, T, R_specific, st.viscosity_Pa_s, st.rho_kg_m3, dz, spec.friction_multiplier)
            return G_pred - G

        # Search down to a small fraction of P, not just 0.30*P: at lower
        # absolute pressures the SAME mass flux needs a progressively
        # larger fractional pressure drop per cell to dissipate the same
        # friction loss (density falls as pressure falls, so velocity --
        # and friction loss -- rises for fixed mass flow), so a 70%-of-P
        # floor that is generous near a high-pressure inlet can be too
        # tight a few cells downstream even when a valid, physically
        # subsonic solution still exists. Caught during this module's own
        # validation.
        P_floor = max(P * 1.0e-4, 1.0e4)
        f_hi = residual(P * 0.999999)
        f_lo = residual(P_floor)
        if f_hi > 0 or f_lo < 0:
            # No P2 in (P_floor, P) reaches the target flux -- a genuine
            # physical result (the gas chokes before this point), not a
            # solver artifact. Surfaced as a clear, actionable message.
            raise ValueError(
                f"flow cannot be sustained past x={i * dx:.0f} m of this {spec.length_m:.0f} m pipe "
                f"at these conditions (local P={P/1e5:.1f} bara, mass flow {m_dot_kg_s:.1f} kg/s, "
                f"diameter {spec.diameter_m:.3f} m) -- the gas chokes before reaching the end of the "
                f"pipe; try a lower mass flow, a larger diameter, a higher inlet pressure, or a shorter "
                f"pipe length"
            )
        P2 = brentq(residual, P_floor, P * 0.999999, xtol=1.0)
        T2 = spec.ambient_temperature_K + (T - spec.ambient_temperature_K) * np.exp(
            -spec.wall_heat_transfer_coeff_W_m2K * np.pi * spec.diameter_m * dx / max(m_dot_kg_s * st.cp_J_kgK, 1e-6)
        )
        st2 = state_from_PT(gas, P2, T2)
        mach = (G / st2.rho_kg_m3) / st2.speed_of_sound_m_s
        profile.append(SteadyProfilePoint((i + 1) * dx, P2, T2, st2.rho_kg_m3, G, mach))
        P, T = P2, T2

    return profile


@dataclass
class PipeState:
    P_Pa: np.ndarray
    T_K: np.ndarray

    def copy(self) -> "PipeState":
        return PipeState(self.P_Pa.copy(), self.T_K.copy())


class TransientPipe:
    """Finite-volume transient solver -- see module docstring's
    DISCRETIZATION NOTE for why `initialize_from_steady_state` averages
    node values into cell-center values rather than reusing them directly."""

    def __init__(self, spec: PipeSpec, gas: GasComposition):
        self.spec = spec
        self.gas = gas
        self.A = np.pi * spec.diameter_m ** 2 / 4.0
        self.dx = spec.length_m / spec.n_cells
        self.dz_cell = spec.elevation_change_m / spec.n_cells
        self.cell_volume = self.A * self.dx

    def initialize_from_steady_state(self, m_dot_kg_s: float, P_in_Pa: float, T_in_K: float) -> PipeState:
        profile = rapid_steady_state_profile(self.spec, self.gas, m_dot_kg_s, P_in_Pa, T_in_K)
        P_nodes = np.array([p.P_Pa for p in profile])
        T_nodes = np.array([p.T_K for p in profile])
        return PipeState(0.5 * (P_nodes[:-1] + P_nodes[1:]), 0.5 * (T_nodes[:-1] + T_nodes[1:]))

    def steady_outlet_pressure(self, m_dot_kg_s: float, P_in_Pa: float, T_in_K: float) -> float:
        """The TRUE pipe-outlet node pressure (x=L) -- use this as the
        P_out boundary condition when holding a profile stationary, not a
        cell's own (cell-center) pressure, which sits dx/2 upstream."""
        profile = rapid_steady_state_profile(self.spec, self.gas, m_dot_kg_s, P_in_Pa, T_in_K)
        return profile[-1].P_Pa

    def _face_fluxes(self, state: PipeState, m_dot_in_kg_s: float, P_out_Pa: float) -> np.ndarray:
        n = self.spec.n_cells
        G_faces = np.zeros(n + 1)
        G_faces[0] = m_dot_in_kg_s / self.A

        P_ext = np.concatenate([[state.P_Pa[0]], state.P_Pa, [P_out_Pa]])
        T_ext = np.concatenate([[state.T_K[0]], state.T_K, [state.T_K[-1]]])

        for i in range(1, n + 1):
            P1, P2 = P_ext[i], P_ext[i + 1]
            face_length = self.dx if i < n else self.dx / 2.0
            T_avg = 0.5 * (T_ext[i] + T_ext[i + 1])
            st = state_from_PT(self.gas, max(P1, 1e4), T_avg, envelope=None)
            R_specific = 8.314462618 / st.molar_mass_kg_mol
            G, _f = _face_mass_flux(P1, P2, face_length, self.spec.diameter_m, self.spec.roughness_m, st.Z, T_avg, R_specific, st.viscosity_Pa_s, st.rho_kg_m3, self.dz_cell * (1.0 if i < n else 0.5), self.spec.friction_multiplier)
            G_faces[i] = G
        return G_faces

    def rhs(self, t, y, m_dot_in_fn, T_in_fn, P_out_fn) -> np.ndarray:
        n = self.spec.n_cells
        P, T = y[:n], y[n:]
        state = PipeState(P, T)
        m_dot_in, T_in, P_out = m_dot_in_fn(t), T_in_fn(t), P_out_fn(t)
        G_faces = self._face_fluxes(state, m_dot_in, P_out)

        dP_dt, dT_dt = np.zeros(n), np.zeros(n)
        for i in range(n):
            st = state_from_PT(self.gas, max(P[i], 1e4), T[i], envelope=None)
            drho_dt = (G_faces[i] - G_faces[i + 1]) / self.dx
            drho_dP = self._drho_dP_at_T(P[i], T[i])
            dP_dt[i] = drho_dt / max(drho_dP, 1e-12)

            T_upstream = T_in if i == 0 else T[i - 1]
            m_dot_face_in = G_faces[i] * self.A
            C_cell = max(st.rho_kg_m3 * self.cell_volume * st.cp_J_kgK, 1e-6)
            advection = m_dot_face_in * st.cp_J_kgK * (T_upstream - T[i])
            wall_loss = self.spec.wall_heat_transfer_coeff_W_m2K * np.pi * self.spec.diameter_m * self.dx * (T[i] - self.spec.ambient_temperature_K)
            dT_dt[i] = (advection - wall_loss) / C_cell

        return np.concatenate([dP_dt, dT_dt])

    def _drho_dP_at_T(self, P_Pa: float, T_K: float, rel_step: float = 1e-4) -> float:
        dP = max(P_Pa * rel_step, 1.0)
        plus = state_from_PT(self.gas, P_Pa + dP, T_K, envelope=None)
        minus = state_from_PT(self.gas, max(P_Pa - dP, 1e4), T_K, envelope=None)
        return (plus.rho_kg_m3 - minus.rho_kg_m3) / (plus.P_Pa - minus.P_Pa)

    def simulate(self, state0: PipeState, t_span_s, m_dot_in_fn, T_in_fn, P_out_fn, t_eval=None):
        from scipy.integrate import solve_ivp
        y0 = np.concatenate([state0.P_Pa, state0.T_K])
        return solve_ivp(self.rhs, t_span_s, y0, method="RK45", t_eval=t_eval, args=(m_dot_in_fn, T_in_fn, P_out_fn), rtol=1e-6, atol=1e-3, max_step=60.0)

    def total_mass_kg(self, state: PipeState) -> float:
        rhos = np.array([state_from_PT(self.gas, p, t, envelope=None).rho_kg_m3 for p, t in zip(state.P_Pa, state.T_K)])
        return float(np.sum(rhos) * self.cell_volume)
