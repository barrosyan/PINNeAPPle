"""Differentiable dynamics simulation for PINNeAPPle.

Provides three simulation back-ends, all implemented in pure PyTorch so
every quantity is differentiable via autograd:

1. **Rigid body dynamics** – 2-D / 3-D rigid bodies with symplectic Euler
   integration; suitable for robot learning and PINN-coupling experiments.

2. **Material Point Method (MPM)** – MLS-MPM for elastic solids, viscous
   fluids, and snow/sand with Drucker-Prager plasticity.

3. **Particle-based simulations** – Smoothed Particle Hydrodynamics (SPH)
   for free-surface flows.

Quick start::

    import torch
    from pinneaple_dynamics import RigidBody, RigidBodyState
    from pinneaple_dynamics import MPMSimulator, MPMState
    from pinneaple_dynamics import SPHParticles

    # --- Rigid body ---
    body = RigidBody(mass=1.0, inertia=torch.tensor(0.1), dim=2)
    state = RigidBodyState(n_bodies=1, dim=2)
    force = torch.tensor([[0.0, -9.81]])   # gravity
    state = body.step(state, force, torque=torch.zeros(1), dt=1e-3)

    # --- MPM ---
    pos = torch.rand(128, 2) * 0.5 + 0.25
    mpm_state = MPMState(pos)
    sim = MPMSimulator(grid_resolution=32, material="elastic")
    mpm_state = sim(mpm_state, n_steps=10)

    # --- SPH ---
    sph = SPHParticles(n_particles=200, smoothing_length=0.05)
    pos = torch.rand(200, 2)
    vel = torch.zeros(200, 2)
    pos, vel = sph(pos, vel, dt=1e-3)
"""

from .rigid_body import RigidBody, RigidBodyState, RigidBodySystem
from .mpm import MPMSimulator, MPMState
from .particles import ParticleSystem, SPHParticles

__all__ = [
    # Rigid body dynamics
    "RigidBody",
    "RigidBodyState",
    "RigidBodySystem",
    # Material Point Method
    "MPMSimulator",
    "MPMState",
    # Particle systems
    "ParticleSystem",
    "SPHParticles",
]
