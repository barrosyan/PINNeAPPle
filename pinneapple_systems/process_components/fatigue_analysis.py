"""pinneapple_systems.process_components.fatigue_analysis -- S-N
(Wohler) curve fatigue life, Goodman mean-stress correction, and Miner's
rule cumulative damage accumulation for cyclic structural loading.

SELECTED FORMULATION
---------------------
S-N curve: `log10(N_f) = A - m*log10(sigma_a)`, i.e.
`N_f = 10**(A - m*log10(sigma_a))` -- the standard power-law S-N fit
(API RP 7G uses this exact form for drill-pipe fatigue; the same
functional form is standard for any material with a power-law S-N
region), for a **fully-reversed** (zero mean stress) alternating stress
amplitude `sigma_a`.

Goodman mean-stress correction: real cyclic loading is rarely fully
reversed (a rotating shaft under a steady bending load, for instance,
has a nonzero mean stress from that steady load plus a cyclic component
from rotation). The modified-Goodman line maps a stress state
`(sigma_alt, sigma_mean)` onto an *equivalent fully-reversed amplitude*
that the S-N curve above can consume directly:

    sigma_a_equivalent = sigma_alt / (1 - sigma_mean/S_ut)

derived by solving the Goodman failure-boundary equation
`sigma_alt/S_f + sigma_mean/S_ut = 1` for `S_f` (the "failure stress"
lives on the S-N curve; `S_ut` is the material's ultimate tensile
strength). At `sigma_mean = 0` this correctly reduces to
`sigma_a_equivalent = sigma_alt` (no correction needed). This is the
standard combination of the Goodman diagram with an S-N curve (see e.g.
Shigley's *Mechanical Engineering Design*) -- NOT the same as computing
a Goodman safety ratio `sigma_alt/S_e + sigma_mean/S_ut` and inverting it
directly into a cycle count, which conflates a safety margin with a
life estimate and was not carried into this port.

Miner's rule: cumulative linear damage `D = sum(n_i / N_f,i)` across
load blocks/cycles; failure at `D = 1`. Applied here for a single
dominant cyclic loading condition (`D = n_cycles / N_f`) plus any
caller-supplied prior damage fraction, since linear (Miner's-rule)
damage is additive across service history by construction.

VALIDITY ENVELOPE: high-cycle fatigue in the S-N power-law regime (not
valid near the endurance limit "knee" or in the low-cycle/strain-based
regime). Linear (Miner's-rule) damage accumulation -- no load-sequence
or overload effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


def sn_curve_cycles_to_failure(sigma_a: ArrayLike, A: float, m: float) -> ArrayLike:
    """N_f = 10**(A - m*log10(sigma_a)) for a fully-reversed amplitude
    `sigma_a` (floored at 1.0 to keep log10 finite)."""
    sigma_safe = np.maximum(np.asarray(sigma_a, dtype=float), 1.0)
    return 10.0 ** (A - m * np.log10(sigma_safe))


def goodman_equivalent_amplitude(sigma_alt: ArrayLike, sigma_mean: ArrayLike, S_ut: float) -> ArrayLike:
    """Equivalent fully-reversed amplitude per the modified-Goodman line
    (see module docstring): sigma_alt / (1 - sigma_mean/S_ut). A mean
    stress approaching (or exceeding) S_ut drives this to infinity/
    negative, correctly signaling the Goodman boundary has already been
    crossed by static overload alone."""
    denom = 1.0 - np.asarray(sigma_mean, dtype=float) / S_ut
    return np.asarray(sigma_alt, dtype=float) / np.where(np.abs(denom) > 1e-9, denom, 1e-9)


def goodman_safety_ratio(sigma_alt: ArrayLike, sigma_mean: ArrayLike, S_e: float, S_ut: float) -> ArrayLike:
    """sigma_alt/S_e + sigma_mean/S_ut -- the Goodman-line safety ratio
    itself (< 1 inside the safe region, independent of any S-N life
    estimate); `S_e` is the endurance limit / target-life fully-reversed
    strength the diagram is drawn against."""
    return sigma_alt / S_e + np.asarray(sigma_mean, dtype=float) / S_ut


@dataclass(frozen=True)
class MinerDamageResult:
    N_f: np.ndarray
    damage_this_period: np.ndarray
    cumulative_damage: np.ndarray
    remaining_life_fraction: np.ndarray


def miners_rule_damage(
    sigma_alt: ArrayLike,
    sigma_mean: ArrayLike,
    n_cycles: float,
    A_sn: float,
    m_sn: float,
    S_ut: float,
    prior_damage_fraction: ArrayLike = 0.0,
) -> MinerDamageResult:
    """Goodman-corrected S-N life `N_f`, this-period damage
    `n_cycles/N_f`, and Miner's-rule cumulative damage (clipped to
    [0, 1]) combining it with `prior_damage_fraction`."""
    sigma_eq = goodman_equivalent_amplitude(sigma_alt, sigma_mean, S_ut)
    N_f = sn_curve_cycles_to_failure(sigma_eq, A_sn, m_sn)
    damage_this_period = n_cycles / N_f
    cumulative = np.clip(np.asarray(prior_damage_fraction, dtype=float) + damage_this_period, 0.0, 1.0)
    return MinerDamageResult(
        N_f=N_f, damage_this_period=damage_this_period,
        cumulative_damage=cumulative, remaining_life_fraction=1.0 - cumulative,
    )
