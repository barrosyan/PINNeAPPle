"""Validates pinneapple_systems.process_components.curved_path_geometry
against its own closed-form circular-arc cross-check for the build
section, plus straightforward boundary-condition checks for the
vertical and hold sections."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.process_components.curved_path_geometry import (
    build_and_hold_profile,
    circular_arc_tvd_hd,
    inclination_at_depth,
)


def test_purely_vertical_path_has_tvd_equal_to_measured_depth_and_zero_horizontal_displacement():
    profile = build_and_hold_profile(
        total_length_m=1000.0, build_length_m=0.0, final_inclination_deg=90.0,
        hold_length_m=0.0, bin_width_m=1.0,
    )
    assert profile.tvd_m[-1] == pytest.approx(profile.s_m[-1], rel=1e-6)
    assert profile.hd_m[-1] == pytest.approx(0.0, abs=1e-6)


def test_inclination_is_90_throughout_the_vertical_section():
    s = np.linspace(0.0, 50.0, 20)
    inc = inclination_at_depth(s, vertical_length_m=100.0, build_length_m=200.0, final_inclination_deg=0.0)
    assert inc == pytest.approx(90.0)


def test_inclination_is_held_at_final_value_past_the_build_section():
    s = np.array([350.0, 500.0, 1000.0])
    inc = inclination_at_depth(s, vertical_length_m=100.0, build_length_m=200.0, final_inclination_deg=30.0)
    assert inc == pytest.approx(30.0)


def test_build_section_tvd_hd_matches_closed_form_circular_arc():
    build_length_m = 300.0
    final_inclination_deg = 20.0
    bin_width_m = 0.1  # fine bins so the midpoint Riemann sum tracks the arc closely

    profile = build_and_hold_profile(
        total_length_m=1000.0, build_length_m=build_length_m, final_inclination_deg=final_inclination_deg,
        hold_length_m=1000.0 - 400.0 - build_length_m, bin_width_m=bin_width_m,
    )
    vertical_length_m = profile.vertical_length_m
    in_build = (profile.s_m >= vertical_length_m) & (profile.s_m <= vertical_length_m + build_length_m)
    s_within_build = profile.s_m[in_build] - vertical_length_m

    tvd_arc, hd_arc = circular_arc_tvd_hd(s_within_build, build_length_m, final_inclination_deg)
    tvd_numeric = profile.tvd_m[in_build] - profile.tvd_m[in_build][0] + np.sin(np.pi / 2) * bin_width_m / 2
    hd_numeric = profile.hd_m[in_build] - profile.hd_m[in_build][0] + np.cos(np.pi / 2) * bin_width_m / 2

    assert tvd_numeric == pytest.approx(tvd_arc, abs=5e-3)
    assert hd_numeric == pytest.approx(hd_arc, abs=5e-3)


def test_horizontal_hold_section_accumulates_no_further_tvd():
    profile = build_and_hold_profile(
        total_length_m=1000.0, build_length_m=200.0, final_inclination_deg=0.0,
        hold_length_m=500.0, bin_width_m=1.0,
    )
    tvd_at_build_end = profile.tvd_m[profile.s_m >= profile.vertical_length_m + 200.0][0]
    assert profile.tvd_m[-1] == pytest.approx(tvd_at_build_end, abs=1e-3)


def test_lengths_must_be_internally_consistent():
    with pytest.raises(ValueError):
        build_and_hold_profile(
            total_length_m=100.0, build_length_m=60.0, final_inclination_deg=0.0,
            hold_length_m=60.0, bin_width_m=1.0,
        )


def test_final_inclination_must_be_in_valid_range():
    with pytest.raises(ValueError):
        build_and_hold_profile(
            total_length_m=100.0, build_length_m=20.0, final_inclination_deg=120.0,
            hold_length_m=20.0, bin_width_m=1.0,
        )
