"""Validates pinneapple_systems.process_components.fatigue_analysis: the
Goodman-equivalent-amplitude reduction to the uncorrected S-N life at
zero mean stress, monotonic sensitivity to mean stress, and Miner's-rule
damage accumulation/clipping."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.process_components.fatigue_analysis import (
    goodman_equivalent_amplitude,
    goodman_safety_ratio,
    miners_rule_damage,
    sn_curve_cycles_to_failure,
)

A_SN, M_SN, S_UT = 30.0, 3.0, 1.0e9  # calibrated so N_f ~ 1e6 at sigma_a ~ 1e8 Pa (100 MPa), a
                                      # representative high-cycle-fatigue steel S-N point


def test_goodman_equivalent_amplitude_reduces_to_raw_amplitude_at_zero_mean_stress():
    sigma_alt = 5.0e7
    eq = goodman_equivalent_amplitude(sigma_alt, sigma_mean=0.0, S_ut=S_UT)
    assert eq == pytest.approx(sigma_alt, rel=1e-10)


def test_goodman_equivalent_amplitude_increases_with_mean_stress():
    sigma_alt = 5.0e7
    eq_low = goodman_equivalent_amplitude(sigma_alt, sigma_mean=1.0e8, S_ut=S_UT)
    eq_high = goodman_equivalent_amplitude(sigma_alt, sigma_mean=5.0e8, S_ut=S_UT)
    assert eq_high > eq_low > sigma_alt


def test_higher_equivalent_amplitude_gives_shorter_fatigue_life():
    N_low = sn_curve_cycles_to_failure(sigma_a=1.0e8, A=A_SN, m=M_SN)
    N_high = sn_curve_cycles_to_failure(sigma_a=2.0e8, A=A_SN, m=M_SN)
    assert N_high < N_low


def test_sn_curve_matches_direct_power_law_evaluation():
    sigma_a = 1.5e8
    N_f = sn_curve_cycles_to_failure(sigma_a, A=A_SN, m=M_SN)
    assert N_f == pytest.approx(10.0 ** (A_SN - M_SN * np.log10(sigma_a)), rel=1e-10)


def test_goodman_safety_ratio_below_one_inside_safe_region():
    S_e = 3.0e8
    ratio = goodman_safety_ratio(sigma_alt=1.0e8, sigma_mean=1.0e8, S_e=S_e, S_ut=S_UT)
    assert ratio < 1.0


def test_miners_rule_damage_accumulates_additively_with_prior_damage():
    result = miners_rule_damage(
        sigma_alt=1.2e8, sigma_mean=2.0e8, n_cycles=1.0e4,
        A_sn=A_SN, m_sn=M_SN, S_ut=S_UT, prior_damage_fraction=0.2,
    )
    assert result.damage_this_period < 0.5  # sanity: this case shouldn't clip on its own
    assert result.cumulative_damage == pytest.approx(0.2 + result.damage_this_period, rel=1e-9)
    assert result.remaining_life_fraction == pytest.approx(1.0 - result.cumulative_damage, rel=1e-9)


def test_miners_rule_damage_clips_at_one_when_exceeding_life():
    result = miners_rule_damage(
        sigma_alt=5.0e8, sigma_mean=1.0e8, n_cycles=1.0e12,
        A_sn=A_SN, m_sn=M_SN, S_ut=S_UT, prior_damage_fraction=0.0,
    )
    assert result.cumulative_damage == pytest.approx(1.0)
    assert result.remaining_life_fraction == pytest.approx(0.0)


def test_zero_mean_stress_miners_damage_uses_uncorrected_sn_life():
    result = miners_rule_damage(
        sigma_alt=1.0e8, sigma_mean=0.0, n_cycles=1.0e5,
        A_sn=A_SN, m_sn=M_SN, S_ut=S_UT, prior_damage_fraction=0.0,
    )
    N_f_expected = sn_curve_cycles_to_failure(1.0e8, A_SN, M_SN)
    assert result.N_f == pytest.approx(N_f_expected, rel=1e-9)
