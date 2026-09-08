"""Tests for the minimal NVE Lennard-Jones molecular dynamics simulator."""
from __future__ import annotations

import torch

from pinneapple_simulation.particle_dynamics.molecular_dynamics import (
    MDState,
    MolecularDynamicsSimulator,
    lj_force_magnitude,
    lj_potential,
)


def _lattice_state(spacing: float = 1.2, n_side: int = 2, dim: int = 3, seed: int = 0) -> MDState:
    """Small cubic lattice of particles near the LJ equilibrium separation,
    with a small random velocity perturbation (avoids the huge initial
    forces / energy blow-up of a random configuration)."""
    torch.manual_seed(seed)
    coords_1d = torch.arange(n_side, dtype=torch.float32) * spacing + 2.0
    grids = torch.meshgrid(*([coords_1d] * dim), indexing="ij")
    pos = torch.stack([g.reshape(-1) for g in grids], dim=-1)
    vel = 0.05 * torch.randn_like(pos)
    return MDState(pos=pos, vel=vel)


def test_lj_potential_matches_formula_and_has_a_zero_crossing_at_sigma():
    epsilon, sigma = 1.0, 1.0
    r = torch.tensor([sigma, 2.0 ** (1.0 / 6.0) * sigma])
    u = lj_potential(r, epsilon, sigma)
    # U(sigma) = 4*eps*(1 - 1) = 0
    assert torch.isclose(u[0], torch.tensor(0.0), atol=1e-6)
    # U(r_min) = -epsilon, the potential well minimum at r_min = 2^(1/6)*sigma
    assert torch.isclose(u[1], torch.tensor(-epsilon), atol=1e-5)


def test_lj_force_is_repulsive_at_short_range_and_attractive_at_long_range():
    epsilon, sigma = 1.0, 1.0
    f_short = lj_force_magnitude(torch.tensor([0.9 * sigma]), epsilon, sigma)
    f_long = lj_force_magnitude(torch.tensor([1.5 * sigma]), epsilon, sigma)
    assert f_short.item() > 0.0   # repulsive
    assert f_long.item() < 0.0    # attractive


def test_compute_forces_are_zero_for_a_single_isolated_particle():
    sim = MolecularDynamicsSimulator(epsilon=1.0, sigma=1.0)
    pos = torch.zeros(1, 3)
    F = sim.compute_forces(pos)
    assert torch.allclose(F, torch.zeros_like(F))


def test_energy_conservation_nve_lennard_jones():
    """The standard MD correctness check: for a small NVE LJ system,
    total energy should stay approximately constant (not drift wildly)
    over a short integration."""
    sim = MolecularDynamicsSimulator(
        epsilon=1.0, sigma=1.0, mass=1.0, cutoff=2.5, box_size=10.0, dim=3
    )
    state = _lattice_state(spacing=1.2, n_side=2, dim=3, seed=0)

    e0 = sim.total_energy(state).item()
    n_steps = 300
    dt = 1e-3
    energies = [e0]
    for _ in range(n_steps):
        state = sim.step(state, dt)
        energies.append(sim.total_energy(state).item())

    e_final = energies[-1]
    max_dev = max(abs(e - e0) for e in energies)

    # Energy should not drift wildly: bound the maximum deviation and the
    # final deviation relative to the (nonzero) initial energy scale.
    scale = max(abs(e0), 1.0)
    assert max_dev / scale < 0.1, f"energy drifted too much: max_dev={max_dev}, e0={e0}"
    assert abs(e_final - e0) / scale < 0.05, f"final energy drift too large: {e_final} vs {e0}"


def test_forward_matches_repeated_step_calls():
    sim = MolecularDynamicsSimulator(epsilon=1.0, sigma=1.0, box_size=10.0, dim=3)
    state = _lattice_state(spacing=1.2, n_side=2, dim=3, seed=2)
    state_a = state.clone()
    state_b = state.clone()

    for _ in range(5):
        state_a = sim.step(state_a, dt=1e-3)
    state_b = sim.forward(state_b, n_steps=5, dt=1e-3)

    assert torch.allclose(state_a.pos, state_b.pos)
    assert torch.allclose(state_a.vel, state_b.vel)


def test_periodic_boundary_conditions_wrap_particles_at_the_edge():
    box = 5.0
    sim = MolecularDynamicsSimulator(epsilon=1.0, sigma=1.0, box_size=box, dim=2, cutoff=1.0)
    # A single particle near the right edge, moving right fast enough to
    # cross the boundary in one step.
    pos = torch.tensor([[4.9, 2.5]])
    vel = torch.tensor([[5.0, 0.0]])
    state = MDState(pos=pos, vel=vel)

    new_state = sim.step(state, dt=0.05)  # would move to x=5.15 without wrapping

    assert torch.all(new_state.pos >= 0.0)
    assert torch.all(new_state.pos < box)
    # Confirms it actually wrapped around (didn't just clamp near the edge).
    assert new_state.pos[0, 0].item() < 1.0


def test_minimum_image_convention_gives_short_distance_across_boundary():
    box = 5.0
    sim = MolecularDynamicsSimulator(epsilon=1.0, sigma=1.0, box_size=box, dim=2, cutoff=2.0)
    # Two particles near opposite edges of the box -- under PBC they are
    # close neighbours (separation ~0.2), not far apart (separation ~4.8).
    pos = torch.tensor([[0.1, 2.5], [4.9, 2.5]])
    _, _, r_ij, dist = sim._find_pairs(pos)
    assert dist.numel() > 0
    assert torch.all(dist < 0.5)


def test_total_energy_is_differentiable_wrt_positions():
    sim = MolecularDynamicsSimulator(epsilon=1.0, sigma=1.0, box_size=10.0, dim=3)
    state = _lattice_state(spacing=1.2, n_side=2, dim=3, seed=3)
    pos = state.pos.clone().requires_grad_(True)
    energy = sim.potential_energy(pos)
    grad = torch.autograd.grad(energy, pos)[0]
    assert grad.shape == pos.shape
    assert torch.any(grad != 0)
