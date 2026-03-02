"""Example: SPH simulation in a box.

Run:
  python examples/solvers/02_sph_box.py
"""

import torch

from pinneaple_geom.sample import sample_box_particles
from pinneaple_solvers import SolverRegistry, register_all


def main():
    register_all()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fluid block
    pos = sample_box_particles(lo=(0.1, 0.1), hi=(0.4, 0.6), spacing=0.03, jitter=0.2, device=device)
    vel = torch.zeros_like(pos)

    sph = SolverRegistry.build(
        "sph",
        h=0.04,
        rho0=1000.0,
        c0=25.0,
        mu=0.02,
        gravity=[0.0, -9.81],
        boundary_lo=[0.0, 0.0],
        boundary_hi=[1.0, 1.0],
    ).to(device)

    out = sph(pos, vel, dt=2e-3, steps=50)
    traj = out.result  # (T,N,2)
    print("traj:", traj.shape)

    # quick sanity: final bounding box
    final = traj[-1]
    print("final min", final.min(dim=0).values, "max", final.max(dim=0).values)


if __name__ == "__main__":
    main()
