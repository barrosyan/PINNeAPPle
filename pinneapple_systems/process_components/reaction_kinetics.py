"""pinneapple_systems.process_components.reaction_kinetics -- a generic
multi-species mass-action reaction-network engine, plus a generic 1D
advection-dispersion-reaction transport solver built on top of it.

SCOPE: this module knows nothing about any specific chemistry. It
provides the data structures and numerics a domain-specific reaction
network (water disinfection chemistry, combustion, biochemical pathways,
...) is built FROM: a `Species` list, `Reaction` objects (a rate-law
callable + a net stoichiometric effect per species), a `ReactionNetwork`
that sums reaction rates into a species-derivative vector, generic
temperature/pH-dependence helpers for building those rate-law callables,
and a stiff-ODE integration wrapper with the non-negativity/raise-on-
failure conventions that matter for concentration state.

SELECTED FORMULATION
---------------------
For a set of R reactions over S species, each reaction r has a scalar
rate rho_r(C, T, pH) (mol/(L*time)) and a stoichiometric coefficient
nu_{r,s} for each species s (positive = produced, negative = consumed,
zero = not involved). The species derivative vector is:

    dC_s/dt = sum_r nu_{r,s} * rho_r(C, T, pH)                         (1)

`mass_action_rate` builds the standard form of rho_r itself:

    rho_r(C) = k_r * prod_s C_s^(order_{r,s})                         (2)

where k_r may itself be a constant or a callable of (T, pH, C) --
this is exactly how a catalyzed or temperature-dependent rate constant
is expressed (see CATALYZED RATE CONSTANTS below), without changing
equation (1)/(2)'s structure at all.

TEMPERATURE DEPENDENCE: `arrhenius_rate_constant(A, Ea_over_R)` builds
`k(T_K) = A * exp(-Ea_over_R / T_K)` (the standard Arrhenius form, with
Ea_over_R in Kelvin so no gas-constant unit bookkeeping is needed at the
call site). `quadratic_in_T(a, b, c)` builds `f(T_K) = a*T_K^2 + b*T_K + c`,
the generic form used for e.g. a temperature-dependent equilibrium
constant when a literature source gives a polynomial fit rather than an
Arrhenius form.

pH DEPENDENCE / ACID-BASE SPECIATION: `acid_fraction(pH, pKa)` and
`base_fraction(pH, pKa)` are the two Henderson-Hasselbalch fractions for
a monoprotic conjugate acid-base pair (the fraction present as the
protonated/acid form and as the deprotonated/base form respectively --
each is one minus the other). `diprotic_fractions(pH, pKa1, pKa2)`
returns the three Bjerrum fractions (H2A, HA-, A2-) for a diprotic
system, generalizing to any two-pKa acid-base system (carbonate being
one example, not the only one).

CATALYZED RATE CONSTANTS: a rate constant that is itself a linear
combination of several catalyst-species concentrations (e.g. an
acid/base/buffer-catalyzed reaction whose observed rate constant is
`k_eff = kA*[cat_A] + kB*[cat_B] + kC*[cat_C]`) is built with
`linear_combination_rate_constant(terms)`, where `terms` is a list of
`(coefficient, catalyst_species_name)` pairs -- this is a general
pattern (not specific to any one catalyst set), evaluated against
whatever state dict is passed to the resulting rate constant callable.

NUMERICAL METHOD: `integrate_network` wraps `scipy.integrate.solve_ivp`
with `method="Radau"` (implicit, appropriate for the stiff kinetics
typical of multi-timescale reaction networks), non-negativity clamping
applied to the state before every RHS evaluation (a real reaction
network's concentrations cannot go negative; clamping prevents solver
overshoot into an unphysical region from producing garbage derivatives),
and raises on integration failure rather than returning a partial/
silently-wrong result.

TRANSPORT: `AdvectionDispersionReactionSolver` discretizes
    dC_s/dt + u * dC_s/dx = D * d2C_s/dx2 + R_s(C) + S_s(x,t)          (3)
via upwind advection + central dispersion finite differences on a
periodic 1D grid (open/reflecting boundary conditions are NOT
implemented -- periodic was the only boundary condition validated in
the source material this engine generalizes from; a one-way upwind
scheme cannot use a mirrored ghost node for a reflecting wall without
introducing an artificial phantom flux, so that mode is intentionally
left unimplemented rather than shipped silently wrong), reusing
`build_rhs` for R_s(C) at every grid point so the SAME reaction network
object drives both a well-mixed (0D) and a spatially-resolved (1D)
solve with zero duplicated chemistry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp

RateFn = Callable[..., float]  # called as rate_fn(C: dict[str, float], T_K: float | None = None, pH: float | None = None) -> float


@dataclass(frozen=True)
class Reaction:
    name: str
    rate_fn: RateFn
    stoichiometry: dict[str, float]  # species name -> net coefficient (produced positive, consumed negative)


@dataclass(frozen=True)
class ReactionNetwork:
    species: tuple[str, ...]
    reactions: tuple[Reaction, ...]

    def rhs(self, C: np.ndarray, T_K: float | None = None, pH: float | None = None) -> np.ndarray:
        """Equation (1): species derivative vector, C ordered as `species`."""
        state = dict(zip(self.species, C))
        d = np.zeros(len(self.species))
        for reaction in self.reactions:
            rate = reaction.rate_fn(state, T_K=T_K, pH=pH)
            for sp, nu in reaction.stoichiometry.items():
                d[self.species.index(sp)] += nu * rate
        return d


def mass_action_rate(k: float | Callable[..., float], orders: dict[str, int]) -> RateFn:
    """Builds equation (2): rho(C) = k * prod(C[s]^order[s]). `k` may be a
    plain float or a callable(T_K=.., pH=.., C=..) -> float for a
    temperature/pH/catalyst-dependent rate constant (see
    `arrhenius_rate_constant`, `linear_combination_rate_constant`)."""
    def rate(C: dict[str, float], T_K: float | None = None, pH: float | None = None) -> float:
        k_val = k(T_K=T_K, pH=pH, C=C) if callable(k) else k
        value = k_val
        for sp, order in orders.items():
            value *= C.get(sp, 0.0) ** order
        return value
    return rate


def arrhenius_rate_constant(A: float, Ea_over_R: float) -> Callable[..., float]:
    """k(T_K) = A * exp(-Ea_over_R / T_K). Raises if called without a
    temperature (a temperature-dependent rate constant used where no T
    is supplied is a caller bug, not something to silently default)."""
    def k(T_K: float | None = None, **_kwargs) -> float:
        if T_K is None:
            raise ValueError("arrhenius_rate_constant requires T_K")
        return A * np.exp(-Ea_over_R / T_K)
    return k


def quadratic_in_T(a: float, b: float, c: float) -> Callable[[float], float]:
    """f(T_K) = a*T_K^2 + b*T_K + c -- e.g. for a polynomial-fit pKa(T) or
    equilibrium constant, when a literature source gives that form
    instead of an Arrhenius one."""
    return lambda T_K: a * T_K ** 2 + b * T_K + c


def linear_combination_rate_constant(terms: Sequence[tuple[float, str]]) -> Callable[..., float]:
    """k_eff(C) = sum(coefficient_i * C[species_i]) -- for a rate
    constant that is itself a linear combination of catalyst-species
    concentrations (e.g. an acid/base/buffer-catalyzed reaction)."""
    def k(C: dict[str, float] | None = None, **_kwargs) -> float:
        if C is None:
            raise ValueError("linear_combination_rate_constant requires C")
        return sum(coeff * C.get(sp, 0.0) for coeff, sp in terms)
    return k


def acid_fraction(pH: float, pKa: float) -> float:
    """Henderson-Hasselbalch: fraction of a monoprotic acid-base pair
    present in the PROTONATED (acid) form."""
    return 1.0 / (1.0 + 10.0 ** (pH - pKa))


def base_fraction(pH: float, pKa: float) -> float:
    """Fraction present in the DEPROTONATED (base/conjugate-base) form
    -- exactly 1 - acid_fraction(pH, pKa)."""
    return 1.0 / (1.0 + 10.0 ** (pKa - pH))


def diprotic_fractions(pH: float, pKa1: float, pKa2: float) -> tuple[float, float, float]:
    """Bjerrum fractions (alpha_H2A, alpha_HA, alpha_A) for a diprotic
    acid-base system (H2A / HA- / A2-), generalizing the carbonate
    system (H2CO3/HCO3-/CO3^2-) to any two-pKa diprotic pair."""
    Ka1, Ka2 = 10.0 ** (-pKa1), 10.0 ** (-pKa2)
    H = 10.0 ** (-pH)
    denom = H ** 2 + Ka1 * H + Ka1 * Ka2
    return H ** 2 / denom, Ka1 * H / denom, Ka1 * Ka2 / denom


@dataclass(frozen=True)
class IntegrationResult:
    t: np.ndarray
    C: np.ndarray  # shape (n_species, n_times)
    success: bool


def integrate_network(
    network: ReactionNetwork, C0: np.ndarray, t_eval: np.ndarray,
    T_K: float | None = None, pH: float | None = None,
    *, method: str = "Radau", rtol: float = 1e-9, atol: float = 1e-14,
) -> IntegrationResult:
    """Integrates dC/dt = network.rhs(C) from C0 over t_eval, clamping
    concentrations non-negative before every RHS evaluation and raising
    RuntimeError (never silently returning a failed/partial solve) if
    the integrator does not report success."""
    def rhs(t, C):
        C_clamped = np.maximum(C, 0.0)
        return network.rhs(C_clamped, T_K=T_K, pH=pH)

    sol = solve_ivp(rhs, [t_eval[0], t_eval[-1]], C0, t_eval=t_eval, method=method, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"reaction-network integration failed: {sol.message}")
    return IntegrationResult(t=sol.t, C=sol.y, success=sol.success)


class AdvectionDispersionReactionSolver:
    """1D transport (equation 3) on a periodic grid, driven by a
    ReactionNetwork for R_s(C) -- see module docstring's TRANSPORT
    section for the discretization and the periodic-only scope note."""

    def __init__(self, network: ReactionNetwork, n_grid: int, length_m: float, velocity_m_s: float, dispersion_m2_s: float):
        self.network = network
        self.n_grid = n_grid
        self.dx = length_m / n_grid
        self.u = velocity_m_s
        self.D = dispersion_m2_s

    def _d_dx_upwind(self, C: np.ndarray) -> np.ndarray:
        # First-order upwind, assumes u >= 0 (flow in +x); periodic wrap via np.roll.
        return (C - np.roll(C, 1)) / self.dx

    def _d2_dx2_central(self, C: np.ndarray) -> np.ndarray:
        return (np.roll(C, -1) - 2.0 * C + np.roll(C, 1)) / self.dx ** 2

    def rhs(self, t: float, y: np.ndarray, source_fn: Callable[[float, float], dict[str, float]] | None, T_K: float | None, pH: float | None) -> np.ndarray:
        n_species = len(self.network.species)
        C = y.reshape(n_species, self.n_grid)
        dC = np.zeros_like(C)
        for j in range(self.n_grid):
            x = j * self.dx
            reaction_rate = self.network.rhs(C[:, j], T_K=T_K, pH=pH)
            source = np.zeros(n_species)
            if source_fn is not None:
                s = source_fn(x, t)
                for sp, val in s.items():
                    source[self.network.species.index(sp)] = val
            dC[:, j] = reaction_rate + source[:]
        for i in range(n_species):
            dC[i, :] += -self.u * self._d_dx_upwind(C[i, :]) + self.D * self._d2_dx2_central(C[i, :])
        return dC.reshape(-1)

    def integrate(
        self, C0_grid: np.ndarray, t_eval: np.ndarray,
        source_fn: Callable[[float, float], dict[str, float]] | None = None,
        T_K: float | None = None, pH: float | None = None,
        *, method: str = "Radau", rtol: float = 1e-6, atol: float = 1e-9,
    ) -> IntegrationResult:
        y0 = np.maximum(C0_grid, 0.0).reshape(-1)
        sol = solve_ivp(
            lambda t, y: self.rhs(t, np.maximum(y, 0.0), source_fn, T_K, pH),
            [t_eval[0], t_eval[-1]], y0, t_eval=t_eval, method=method, rtol=rtol, atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"transport integration failed: {sol.message}")
        n_species = len(self.network.species)
        return IntegrationResult(t=sol.t, C=sol.y.reshape(n_species, self.n_grid, -1), success=sol.success)
