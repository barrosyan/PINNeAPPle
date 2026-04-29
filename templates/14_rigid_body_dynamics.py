"""14_rigid_body_dynamics.py — Rigid body dynamics coupled with a PINN.

Demonstrates:
- RigidBody with 2D symplectic Euler integration
- RigidBodySystem for managing multiple bodies
- Coupling rigid body positions to PINN inputs (pressure/flow field)
- Visualizing the trajectory and PINN-predicted force
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_dynamics import RigidBody, RigidBodyState, RigidBodySystem


# ---------------------------------------------------------------------------
# PINN surrogate: maps (x, y, t) → pressure p at query points
# (trained to approximate a simple vortex potential)
# ---------------------------------------------------------------------------

class PressurePINN(nn.Module):
    """Surrogate pressure field: p(x, y, t) = sin(π x) cos(π y) exp(-0.1 t)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.net(xyt)

    @torch.no_grad()
    def pressure_force_2d(self, pos: torch.Tensor, t: float,
                          n_surface: int = 16) -> torch.Tensor:
        """Estimate net pressure force on a body at position *pos* at time *t*.

        Integrates p · n̂ around a small circle (trapezoid rule).

        Returns
        -------
        torch.Tensor shape (n_bodies, 2)
        """
        n_bodies = pos.shape[0]
        angles = torch.linspace(0, 2 * math.pi, n_surface + 1)[:-1]
        r = 0.1  # body radius
        force = torch.zeros(n_bodies, 2)

        for b in range(n_bodies):
            cx, cy = pos[b, 0].item(), pos[b, 1].item()
            xs = cx + r * torch.cos(angles)
            ys = cy + r * torch.sin(angles)
            t_col = torch.full((n_surface,), t)
            xyt = torch.stack([xs, ys, t_col], dim=1)
            p_vals = self.forward(xyt).squeeze()   # (n_surface,)

            # Normal directions: outward
            nx = torch.cos(angles)
            ny = torch.sin(angles)
            # Trapezoidal integration: Fx = -∫ p nx ds, Fy = -∫ p ny ds
            ds = 2 * math.pi * r / n_surface
            force[b, 0] = -(p_vals * nx).mean() * 2 * math.pi * r
            force[b, 1] = -(p_vals * ny).mean() * 2 * math.pi * r

        return force


def quick_pretrain_pinn(pinn: nn.Module, n_epochs: int = 1000):
    """Pre-train the pressure PINN on analytic samples."""
    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    for ep in range(1, n_epochs + 1):
        optimizer.zero_grad()
        xyt = torch.rand(500, 3)
        xyt[:, 2] *= 5.0   # t ∈ [0, 5]
        out = pinn(xyt)
        target = (torch.sin(math.pi * xyt[:, 0:1])
                  * torch.cos(math.pi * xyt[:, 1:2])
                  * torch.exp(-0.1 * xyt[:, 2:3]))
        loss = (out - target).pow(2).mean()
        loss.backward()
        optimizer.step()
    print(f"  PINN pre-train final loss: {loss.item():.4e}")


def main():
    torch.manual_seed(7)
    device = "cpu"
    print("Rigid body + PINN coupling demo")

    # --- PINN surrogate ---------------------------------------------------
    pinn = PressurePINN()
    print("Pre-training pressure PINN...")
    quick_pretrain_pinn(pinn, n_epochs=1500)

    # --- Rigid body setup -------------------------------------------------
    # Two circular bodies of mass 1.0, inertia 0.05 (thin disk)
    n_bodies = 2
    body = RigidBody(mass=1.0, inertia=torch.tensor(0.05), dim=2)

    state = RigidBodyState(n_bodies=n_bodies, dim=2, device=device)
    # Initial positions
    state.pos[0] = torch.tensor([0.3, 0.5])
    state.pos[1] = torch.tensor([0.7, 0.5])
    # Initial velocities
    state.vel[0] = torch.tensor([0.1,  0.05])
    state.vel[1] = torch.tensor([-0.1, -0.05])

    # --- RigidBodySystem --------------------------------------------------
    system = RigidBodySystem(bodies=[body] * n_bodies)

    # --- Simulation loop --------------------------------------------------
    dt = 0.02
    T = 4.0
    n_steps = int(T / dt)

    traj = np.zeros((n_steps + 1, n_bodies, 2))
    traj[0] = state.pos.detach().numpy()

    gravity = torch.tensor([[0.0, 0.0]] * n_bodies)   # no gravity

    forces_history = []

    for step in range(n_steps):
        t_curr = step * dt

        # Pressure force from PINN
        pressure_f = pinn.pressure_force_2d(state.pos.detach(), t=t_curr)
        total_force = gravity + pressure_f     # (n_bodies, 2)
        torque      = torch.zeros(n_bodies)

        # Integrate one step: symplectic Euler
        state = body.step(state, total_force, torque=torque, dt=dt)

        # Clip to domain [0,1]² (soft wall)
        state.pos = state.pos.clamp(0.05, 0.95)

        traj[step + 1] = state.pos.detach().numpy()
        forces_history.append(pressure_f.norm(dim=1).mean().item())

    print(f"Simulated {n_steps} steps  (T={T} s, dt={dt} s)")

    # --- Visualization ----------------------------------------------------
    t_arr = np.linspace(0, T, n_steps + 1)
    colors = ["steelblue", "crimson"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Trajectories in 2D space
    for b in range(n_bodies):
        axes[0].plot(traj[:, b, 0], traj[:, b, 1], "-", color=colors[b],
                     label=f"Body {b+1}", linewidth=1.5)
        axes[0].scatter(*traj[0, b], s=60, color=colors[b], marker="o", zorder=5)
        axes[0].scatter(*traj[-1, b], s=60, color=colors[b], marker="^", zorder=5)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_aspect("equal")
    axes[0].set_title("Body trajectories  (circle=start, triangle=end)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # x-position vs time
    for b in range(n_bodies):
        axes[1].plot(t_arr, traj[:, b, 0], color=colors[b], label=f"Body {b+1} x")
        axes[1].plot(t_arr, traj[:, b, 1], "--", color=colors[b], label=f"Body {b+1} y")
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("Position")
    axes[1].set_title("Position vs time")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # PINN force magnitude
    axes[2].plot(np.linspace(0, T, n_steps), forces_history, "k-")
    axes[2].set_xlabel("t (s)")
    axes[2].set_ylabel("Mean |F_pressure|")
    axes[2].set_title("PINN pressure force on bodies")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("14_rigid_body_dynamics_result.png", dpi=120)
    print("Saved 14_rigid_body_dynamics_result.png")


if __name__ == "__main__":
    main()
