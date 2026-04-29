"""15_mpm_simulation.py — MPM fluid/solid simulation with PINN coupling.

Demonstrates:
- MPMSimulator with three material presets: elastic, snow, fluid
- MPMState initialization from particle positions
- Simulation step loop
- Coupling MPM particle positions to a PINN for force prediction
- Visualization of particle trajectories
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_dynamics import MPMSimulator, MPMState


# ---------------------------------------------------------------------------
# Helper: create a square block of particles
# ---------------------------------------------------------------------------

def particle_block(cx: float, cy: float, side: float, n: int,
                   dim: int = 2) -> torch.Tensor:
    """Return (n²,  dim) particle positions for a square block."""
    side_half = side / 2
    lin = torch.linspace(cx - side_half, cx + side_half, n)
    if dim == 2:
        gx, gy = torch.meshgrid(lin, lin, indexing="ij")
        pos = torch.stack([gx.ravel(), gy.ravel()], dim=1)
    else:
        gz = torch.linspace(cy - side_half, cy + side_half, n)
        gx, gy, gz3 = torch.meshgrid(lin, lin, gz, indexing="ij")
        pos = torch.stack([gx.ravel(), gy.ravel(), gz3.ravel()], dim=1)
    return pos.float()


# ---------------------------------------------------------------------------
# Simple PINN: map particle positions to a virtual "stress" scalar
# Used as a coupling example (e.g. to augment material response)
# ---------------------------------------------------------------------------

class StressPINN(nn.Module):
    def __init__(self, dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        return self.net(pos)


def run_mpm(material: str, n_steps: int, grid_res: int = 32, dim: int = 2) -> dict:
    """Run a short MPM simulation for the given material.

    Returns dict with 'positions': list of (n_p, dim) numpy arrays.
    """
    # Particle block centred at (0.5, 0.75) — above midpoint, falls under gravity
    n_side = 8
    pos_init = particle_block(cx=0.5, cy=0.75, side=0.2, n=n_side, dim=dim)
    n_p = pos_init.shape[0]

    # Initial downward velocity for fluid demo
    vel_init = torch.zeros_like(pos_init)
    if material == "fluid":
        vel_init[:, 1] = -0.5   # give a downward push

    state = MPMState(pos=pos_init, vel=vel_init)

    sim = MPMSimulator(
        grid_resolution=grid_res,
        dt=2e-4,
        material=material,
        dim=dim,
        E=1e4,
        nu=0.2,
        rho=1.0,
        gravity=(0.0, -9.8) if dim == 2 else (0.0, -9.8, 0.0),
    )

    positions = [pos_init.numpy().copy()]
    for step in range(n_steps):
        state = sim(state, n_steps=1)
        positions.append(state.pos.detach().numpy().copy())

    return {"positions": positions, "n_steps": n_steps}


def main():
    torch.manual_seed(11)
    device = "cpu"
    print("MPM simulation demo (elastic, snow, fluid)")

    n_sim_steps = 40   # keep small for fast demo

    results = {}
    for material in ["elastic", "snow", "fluid"]:
        print(f"  Running MPM ({material})...")
        results[material] = run_mpm(material, n_steps=n_sim_steps, grid_res=32)

    # --- PINN coupling example -------------------------------------------
    print("\nCoupling MPM particle positions to StressPINN...")
    stress_pinn = StressPINN(dim=2)

    # Pre-train PINN on a simple analytic stress field σ = sin(πx)sin(πy)
    optimizer = torch.optim.Adam(stress_pinn.parameters(), lr=1e-3)
    for ep in range(1000):
        optimizer.zero_grad()
        pos = torch.rand(300, 2)
        target = torch.sin(math.pi * pos[:, 0:1]) * torch.sin(math.pi * pos[:, 1:2])
        loss = (stress_pinn(pos) - target).pow(2).mean()
        loss.backward()
        optimizer.step()

    # Query PINN stress at final elastic simulation positions
    final_pos_elastic = torch.tensor(
        results["elastic"]["positions"][-1], dtype=torch.float32
    )
    with torch.no_grad():
        sigma = stress_pinn(final_pos_elastic).squeeze().numpy()
    print(f"  PINN stress at {len(sigma)} particles: "
          f"min={sigma.min():.4f}  max={sigma.max():.4f}")

    # --- Visualization ----------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    snapshot_steps = [0, n_sim_steps // 2, n_sim_steps]
    materials = list(results.keys())

    for row, step_idx in enumerate([0, n_sim_steps]):
        for col, mat in enumerate(materials):
            ax = axes[row, col]
            pos_snap = results[mat]["positions"][min(step_idx, n_sim_steps)]
            ax.scatter(pos_snap[:, 0], pos_snap[:, 1], s=8,
                       c="steelblue" if row == 0 else "crimson", alpha=0.7)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.set_title(f"{mat}  {'(initial)' if row==0 else '(final)'}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.grid(True, alpha=0.2)

    plt.suptitle("MPM simulation: particle positions", fontsize=13)
    plt.tight_layout()
    plt.savefig("15_mpm_simulation_result.png", dpi=120)
    print("Saved 15_mpm_simulation_result.png")

    # PINN stress overlay on final elastic state
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    sc = ax2.scatter(final_pos_elastic[:, 0].numpy(),
                     final_pos_elastic[:, 1].numpy(),
                     c=sigma, cmap="plasma", s=30, zorder=3)
    plt.colorbar(sc, ax=ax2, label="σ (PINN)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_aspect("equal")
    ax2.set_title("Elastic body: PINN stress field overlay")
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("15_mpm_stress_pinn.png", dpi=120)
    print("Saved 15_mpm_stress_pinn.png")


if __name__ == "__main__":
    main()
