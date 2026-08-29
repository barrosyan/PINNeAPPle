"""Validates pinneapple_simulation.numerical_solvers.nonlinear_beam_fem
(the transient Von Karman beam engine) against two independent checks
neither of which touches its own Newton/Newmark code path directly: the
closed-form cantilever natural-frequency formula (linearized-at-rest
eigenvalue problem), and the known long-time settling behavior of a
damped beam under a suddenly-applied constant load (should relax to the
static linear-beam-theory deflection). This engine previously shipped
with no tests at all; the zero-load degenerate case exercised here
(`test_natural_frequencies...`, which runs before any load is applied)
caught a real 0/0 convergence-check bug, fixed at the source."""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_simulation.numerical_solvers.nonlinear_beam_fem import (
    cantilever_frequency_roots,
    solve_nonlinear_beam_transient,
)
from pinneapple_systems.process_components.beam_statics import (
    rectangular_section_properties,
    solve_beam,
)

L = 2.0
E = 200e9
RHO = 7850.0
WIDTH, HEIGHT = 0.05, 0.02
I, _ = rectangular_section_properties(WIDTH, HEIGHT)
A = WIDTH * HEIGHT


def _cantilever_boundary(num_elements: int) -> dict:
    return {
        "D": {
            "globalNode#": [[0], [3]],
            "Values": [[0, 0.0], [1, 0.0], [2, 0.0]],
        },
        "N": {"globalNode#": [[], []], "Values": []},
    }


def test_cantilever_first_natural_frequency_matches_closed_form():
    num_elements = 30
    out = solve_nonlinear_beam_transient(
        boundary=_cantilever_boundary(num_elements),
        L_m=L, A_m2=A, I_m4=I, E_Pa=E, rho_kg_m3=RHO,
        num_elements=num_elements, time_total_s=1e-6, timesteps=1,
        compute_natural_frequencies=True, num_modes=2,
    )
    assert not out["nonconverged"]

    beta1_L = cantilever_frequency_roots(1)[0]
    omega1_analytical = beta1_L ** 2 * np.sqrt(E * I / (RHO * A * L ** 4))
    f1_analytical = omega1_analytical / (2 * np.pi)

    f1_fem = out["natural_frequencies_hz"][0]
    assert f1_fem == pytest.approx(f1_analytical, rel=5e-3)


def test_undamped_step_load_first_peak_matches_dynamic_amplification_factor_of_2():
    """Classical SDOF step-response result: an undamped system suddenly
    loaded to a constant value overshoots to ~2x the static deflection at
    its first peak (t ~ T1/2) before oscillating -- independent of mesh/
    timestep choice, and independent of the engine's own damping/Newmark
    bookkeeping since damping is exactly zero here."""
    num_elements = 20
    q0 = 200.0  # N/m, small enough for negligible Von Karman correction

    boundary = _cantilever_boundary(num_elements)

    out = solve_nonlinear_beam_transient(
        boundary=boundary,
        L_m=L, A_m2=A, I_m4=I, E_Pa=E, rho_kg_m3=RHO,
        damping_coeff=0.0,
        transverse_dist_load_N_per_m=q0,
        num_elements=num_elements, time_total_s=0.15, timesteps=300,
        compute_natural_frequencies=False,
    )
    assert not out["nonconverged"]

    x = np.array([L])
    linear = solve_beam("cantilever_uniform", x, L, E, I, q0)
    peak_w_tip = out["w_m"][:, -1].max()
    assert peak_w_tip == pytest.approx(2.0 * linear.deflection[0], rel=0.1)
