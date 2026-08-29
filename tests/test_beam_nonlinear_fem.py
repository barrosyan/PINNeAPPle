"""Validates pinneapple_systems.process_components.beam_nonlinear_fem
against the independent closed-form linear beam solution (beam_statics)
at small load, and against the well-known qualitative Von Karman
stress-stiffening signature (axially-restrained beam deflects
sub-linearly with load) at larger load -- not just internal
self-consistency."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_systems.process_components.beam_nonlinear_fem import (
    solve_nonlinear_beam_static,
)
from pinneapple_systems.process_components.beam_statics import (
    rectangular_section_properties,
    solve_beam,
)

L = 3.0
E = 200e9
WIDTH, HEIGHT = 0.1, 0.05
I, _ = rectangular_section_properties(WIDTH, HEIGHT)
C = HEIGHT / 2.0


def _cantilever_boundary(num_elements: int) -> dict:
    return {
        "D": {
            "globalNode#": [[0], [3]],
            "Values": [[0, 0.0], [1, 0.0], [2, 0.0]],
        },
        "N": {"globalNode#": [[], []], "Values": []},
    }


def _fixed_fixed_boundary(num_elements: int) -> dict:
    last = num_elements
    return {
        "D": {
            "globalNode#": [[0, last], [3, 3]],
            "Values": [[0, 0.0], [1, 0.0], [2, 0.0], [0, 0.0], [1, 0.0], [2, 0.0]],
        },
        "N": {"globalNode#": [[], []], "Values": []},
    }


def test_small_load_cantilever_matches_linear_closed_form_tip_deflection():
    q0 = 1e-3  # N/m -- small enough that w_tip/L ~ 1e-5, negligible nonlinearity
    num_elements = 30
    result = solve_nonlinear_beam_static(
        boundary=_cantilever_boundary(num_elements),
        L_m=L, A_m2=WIDTH * HEIGHT, I_m4=I, E_Pa=E, fiber_distance_m=C,
        transverse_dist_load_N_per_m=q0,
        num_elements=num_elements, load_steps=2, newton_iterations=30,
    )
    assert not result.nonconverged

    x = np.array([L])
    linear = solve_beam("cantilever_uniform", x, L, E, I, q0)
    w_tip_nonlinear = result.w_m[-1]
    assert w_tip_nonlinear == pytest.approx(linear.deflection[0], rel=2e-3)


def test_cantilever_fixed_end_has_zero_deflection_and_slope():
    num_elements = 20
    result = solve_nonlinear_beam_static(
        boundary=_cantilever_boundary(num_elements),
        L_m=L, A_m2=WIDTH * HEIGHT, I_m4=I, E_Pa=E, fiber_distance_m=C,
        transverse_dist_load_N_per_m=50.0,
        num_elements=num_elements, load_steps=5,
    )
    assert result.w_m[0] == pytest.approx(0.0, abs=1e-12)
    assert result.theta_rad[0] == pytest.approx(0.0, abs=1e-12)
    assert result.u_m[0] == pytest.approx(0.0, abs=1e-12)


def test_bending_stress_matches_M_c_over_I_identity_at_every_node():
    num_elements = 20
    result = solve_nonlinear_beam_static(
        boundary=_cantilever_boundary(num_elements),
        L_m=L, A_m2=WIDTH * HEIGHT, I_m4=I, E_Pa=E, fiber_distance_m=C,
        transverse_dist_load_N_per_m=50.0,
        num_elements=num_elements, load_steps=5,
    )
    sigma_from_M = result.bending_moment * C / I
    assert result.bending_stress == pytest.approx(-sigma_from_M, rel=1e-9)


def test_axially_restrained_beam_stiffens_nonlinearly_with_increasing_load():
    """Fixed-fixed beam: membrane (stretching) coupling from the Von
    Karman term should make deflection grow SUB-linearly with load
    (doubling the load less than doubles the deflection) -- the standard
    qualitative signature of geometric stress-stiffening, absent from any
    linear model."""
    num_elements = 24
    mid = num_elements // 2

    def midspan_deflection(q0: float) -> float:
        result = solve_nonlinear_beam_static(
            boundary=_fixed_fixed_boundary(num_elements),
            L_m=L, A_m2=WIDTH * HEIGHT, I_m4=I, E_Pa=E, fiber_distance_m=C,
            transverse_dist_load_N_per_m=q0,
            num_elements=num_elements, load_steps=8, newton_iterations=40,
        )
        assert not result.nonconverged
        return abs(result.w_m[mid])

    w_q = midspan_deflection(2.0e4)
    w_2q = midspan_deflection(4.0e4)
    assert w_2q < 2.0 * w_q


def test_zero_load_gives_zero_deflection_everywhere():
    num_elements = 10
    result = solve_nonlinear_beam_static(
        boundary=_cantilever_boundary(num_elements),
        L_m=L, A_m2=WIDTH * HEIGHT, I_m4=I, E_Pa=E, fiber_distance_m=C,
        transverse_dist_load_N_per_m=0.0,
        num_elements=num_elements, load_steps=1,
    )
    assert result.w_m == pytest.approx(np.zeros_like(result.w_m), abs=1e-10)


def test_fiber_distance_must_be_positive():
    num_elements = 5
    with pytest.raises(ValueError):
        solve_nonlinear_beam_static(
            boundary=_cantilever_boundary(num_elements),
            L_m=L, A_m2=WIDTH * HEIGHT, I_m4=I, E_Pa=E, fiber_distance_m=0.0,
            num_elements=num_elements,
        )
