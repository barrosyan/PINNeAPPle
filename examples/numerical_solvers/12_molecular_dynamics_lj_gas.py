"""
Demo: differentiable NVE Lennard-Jones molecular dynamics

Showcases ``MolecularDynamicsSimulator`` / ``MDState``
(pinneapple_simulation/particle_dynamics/molecular_dynamics.py): pairwise
Lennard-Jones forces, periodic boundary conditions (minimum-image
convention), and velocity-Verlet time integration.

Sets up a small 2D LJ gas (particles on a square lattice near the LJ
equilibrium separation, spacing ~2^(1/6)*sigma, plus a small random velocity
kick) inside a periodic box, and runs a few hundred velocity-Verlet steps.
The standard NVE correctness check is approximate total-energy (kinetic +
potential) conservation -- printed at the start and end of the run.
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pinneapple_simulation.particle_dynamics.molecular_dynamics import (
    MDState,
    MolecularDynamicsSimulator,
)


def lattice_state(spacing: float, n_side: int, dim: int = 2, seed: int = 0) -> MDState:
    """Small square/cubic lattice of particles near the LJ equilibrium
    separation, with a small random velocity perturbation -- avoids the
    huge initial forces / energy blow-up of a purely random configuration."""
    torch.manual_seed(seed)
    coords_1d = torch.arange(n_side, dtype=torch.float32) * spacing + 2.0
    grids = torch.meshgrid(*([coords_1d] * dim), indexing="ij")
    pos = torch.stack([g.reshape(-1) for g in grids], dim=-1)
    vel = 0.05 * torch.randn_like(pos)
    return MDState(pos=pos, vel=vel)


def main():
    torch.manual_seed(0)

    dim = 2
    n_side = 6  # 6x6 = 36 particles
    sigma = 1.0
    epsilon = 1.0
    spacing = 2.0 ** (1.0 / 6.0) * sigma  # LJ potential-well minimum separation

    sim = MolecularDynamicsSimulator(
        epsilon=epsilon, sigma=sigma, mass=1.0, cutoff=2.5 * sigma,
        box_size=15.0, dim=dim,
    )
    state = lattice_state(spacing=spacing, n_side=n_side, dim=dim, seed=0)
    n_particles = state.n_particles

    print("NVE Lennard-Jones gas")
    print(f"  n_particles={n_particles}  dim={dim}  box_size=15.0  "
          f"epsilon={epsilon}  sigma={sigma}")

    e0 = sim.total_energy(state).item()
    ke0 = sim.kinetic_energy(state.vel).item()
    pe0 = sim.potential_energy(state.pos).item()
    print(f"\nInitial:  E_total={e0:.6f}  KE={ke0:.6f}  PE={pe0:.6f}")

    n_steps = 500
    dt = 1e-3
    energies = [e0]
    for step in range(1, n_steps + 1):
        state = sim.step(state, dt)
        if step % 100 == 0:
            e = sim.total_energy(state).item()
            energies.append(e)
            print(f"  step={step:04d}  E_total={e:.6f}")

    ke_f = sim.kinetic_energy(state.vel).item()
    pe_f = sim.potential_energy(state.pos).item()
    e_f = ke_f + pe_f
    max_dev = max(abs(e - e0) for e in energies)
    rel_drift = abs(e_f - e0) / max(abs(e0), 1.0)

    print(f"\nFinal:    E_total={e_f:.6f}  KE={ke_f:.6f}  PE={pe_f:.6f}")
    print(f"Max |E(t) - E0| over run: {max_dev:.6f}")
    print(f"Relative energy drift |E_final - E0| / max(|E0|,1): {rel_drift:.4%}")
    print("\nAll particle positions remain within the periodic box:",
          bool(torch.all(state.pos >= 0.0) and torch.all(state.pos < 15.0)))


if __name__ == "__main__":
    main()
