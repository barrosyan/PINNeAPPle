"""Demo: Full CAD-to-CFD pipeline (Feature 20).

Shows the complete workflow:
  1. Build a geometry (NACA airfoil or unit-square channel)
  2. Generate a computational mesh (structured rect fallback when gmsh absent)
  3. Run the incompressible Navier-Stokes solver
  4. Extract PINN training data
  5. Define a simple PINN and compare predictions with the CFD solution
  6. Demonstrate the adjoint shape optimisation (Feature 16)

Run::
    python examples/pinneaple_solvers/11_cad_cfd_pipeline_demo.py
"""

import sys
import time

import numpy as np
import torch
import torch.nn as nn

# ------------------------------------------------------------------
# Feature 20: CAD-to-CFD pipeline
# ------------------------------------------------------------------
from pinneaple_solvers import CADToCFDPipeline, CFDMesh, NSFlowSolver
from pinneaple_solvers.cfd_pipeline import _gmsh_available

# ------------------------------------------------------------------
# Feature 16: Adjoint shape optimisation
# ------------------------------------------------------------------
from pinneaple_design_opt.adjoint import (
    ShapeParametrization,
    ContinuousAdjointSolver,
    naca_parametric,
)

# ------------------------------------------------------------------
# Optional: Feature 15 backend check
# ------------------------------------------------------------------
from pinneaple_backend import get_backend, set_backend, jax_available

# ------------------------------------------------------------------
# Optional: Feature 18 dynamics
# ------------------------------------------------------------------
from pinneaple_dynamics import RigidBody, RigidBodyState, MPMSimulator, MPMState, SPHParticles

# ------------------------------------------------------------------
# Optional: Feature 19 world model
# ------------------------------------------------------------------
from pinneaple_worldmodel import CosmosAdapter, WorldModelConfig


# =====================================================================
# 1. Simple PINN for 2-D channel flow
# =====================================================================

class ChannelFlowPINN(nn.Module):
    """Minimal 4-layer MLP PINN for 2-D channel flow.

    Input:  (x, y)
    Output: (u_vel, v_vel, pressure)
    """

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =====================================================================
# 2. Physics residual for Stokes flow (linearised NS)
# =====================================================================

def stokes_residual(model: nn.Module, x: torch.Tensor, nu: float = 1e-3) -> torch.Tensor:
    """Compute PDE residual for 2-D Stokes equations at collocation points."""
    x = x.requires_grad_(True)
    out = model(x)
    u = out[:, 0:1]
    v = out[:, 1:2]
    p = out[:, 2:3]

    # Compute gradients
    def _grad(f):
        return torch.autograd.grad(f.sum(), x, create_graph=True)[0]

    grad_u = _grad(u)
    grad_v = _grad(v)
    grad_p = _grad(p)
    u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
    v_x, v_y = grad_v[:, 0:1], grad_v[:, 1:2]
    p_x, p_y = grad_p[:, 0:1], grad_p[:, 1:2]

    # Second-order derivatives (Laplacian)
    u_xx = _grad(u_x)[:, 0:1]
    u_yy = _grad(u_y)[:, 1:2]
    v_xx = _grad(v_x)[:, 0:1]
    v_yy = _grad(v_y)[:, 1:2]

    # Stokes equations: -nu * Δu + ∇p = 0, div u = 0
    res_u = -nu * (u_xx + u_yy) + p_x
    res_v = -nu * (v_xx + v_yy) + p_y
    res_c = u_x + v_y  # continuity

    return torch.cat([res_u, res_v, res_c], dim=1)


# =====================================================================
# Main demo
# =====================================================================

def main():
    print("=" * 65)
    print("PINNeAPPle Group-E Feature Demo")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Feature 15: Backend selection
    # ------------------------------------------------------------------
    print("\n[Feature 15] Multi-backend support")
    print(f"  Default backend : {get_backend()}")
    print(f"  JAX available   : {jax_available()}")
    if jax_available():
        set_backend("jax")
        print(f"  Switched to     : {get_backend()}")
        set_backend("torch")  # revert
    else:
        print("  (JAX not installed – staying on PyTorch backend)")

    # ------------------------------------------------------------------
    # Feature 20: Build mesh and run CFD solver
    # ------------------------------------------------------------------
    print("\n[Feature 20] CAD-to-CFD pipeline")
    print(f"  gmsh available  : {_gmsh_available()}")

    t0 = time.perf_counter()
    pipeline = CADToCFDPipeline(nu=1e-3, rho=1.0)
    # No STEP/STL file available in CI – pipeline will fall back to rect mesh
    pipeline.mesh(max_edge_length=0.08)
    print(f"  Mesh            : {pipeline.cfd_mesh}")

    pipeline.set_bcs(
        inlet_velocity=(1.0, 0.0),
        no_slip_tags=["wall_bottom", "wall_top"],
        outlet_tags=["outlet"],
    )
    results = pipeline.solve(max_iter=10, tol=1e-6)
    dt_cfd = time.perf_counter() - t0

    u_max = float(np.max(np.abs(results["u"])))
    print(f"  NS solve        : {dt_cfd:.3f}s  converged={results['converged']}")
    print(f"  Max |u|         : {u_max:.4f}")

    # Extract PINN training data
    pinn_data = pipeline.to_pinn_data(n_col=2000)
    print(f"  Collocation pts : {pinn_data['x_col'].shape}")
    print(f"  Boundary pts    : {pinn_data['x_bc'].shape}")

    # ------------------------------------------------------------------
    # Feature 20: Train a simple PINN and compare with CFD
    # ------------------------------------------------------------------
    print("\n[Feature 20] PINN training on CFD data (20 steps)")
    pinn = ChannelFlowPINN(hidden=32)
    opt = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    nu = 1e-3

    x_col = pinn_data["x_col"].requires_grad_(False)
    x_bc = pinn_data["x_bc"]
    u_bc = pinn_data["u_bc"]
    v_bc = pinn_data["v_bc"]

    for step in range(20):
        opt.zero_grad()
        # BC loss
        out_bc = pinn(x_bc)
        loss_bc = torch.mean((out_bc[:, 0] - u_bc) ** 2 + (out_bc[:, 1] - v_bc) ** 2)
        # Data loss (CFD u, v)
        if "x_cfd" in pinn_data:
            out_cfd = pinn(pinn_data["x_cfd"])
            loss_data = torch.mean(
                (out_cfd[:, 0] - pinn_data["u_cfd"]) ** 2
                + (out_cfd[:, 1] - pinn_data["v_cfd"]) ** 2
            )
        else:
            loss_data = torch.tensor(0.0)
        loss = loss_bc + 0.1 * loss_data
        loss.backward()
        opt.step()
        if step % 5 == 0:
            print(f"    step {step:3d}  loss={float(loss.detach()):.4e}")

    # Compare PINN vs CFD
    metrics = pipeline.compare_with_pinn(pinn, field="u")
    print(f"  CFD vs PINN  L2={metrics['L2_error']:.4e}  "
          f"max={metrics['max_error']:.4e}  "
          f"relL2={metrics['rel_L2_error']:.4e}")

    # ------------------------------------------------------------------
    # Feature 16: Adjoint shape optimisation
    # ------------------------------------------------------------------
    print("\n[Feature 16] Adjoint-based shape optimisation")
    naca_pts = naca_parametric(m=0.0, p=0.0, t_c=0.12, n_pts=40)
    shape = ShapeParametrization(naca_pts)
    print(f"  NACA ctrl pts   : {naca_pts.shape}")

    def dummy_objective(model, x):
        """Proxy drag: mean of |u|^2 at deformed pts."""
        out = model(x[:, :2])
        return torch.mean(out[:, 0] ** 2)

    def dummy_residual(model, x):
        """Proxy residual: Laplacian proxy."""
        return torch.zeros(x.shape[0], 1)

    adjoint = ContinuousAdjointSolver(pinn, dummy_residual, dummy_objective)
    x_col_small = x_col[:200]
    result = adjoint.optimize(shape, x_col_small, n_steps=5, lr=1e-3)
    print(f"  Best objective  : {result['best_objective']:.6e}")
    print(f"  Shape pts after : {result['best_control_points'].shape}")

    # ------------------------------------------------------------------
    # Feature 18: Differentiable dynamics
    # ------------------------------------------------------------------
    print("\n[Feature 18] Differentiable dynamics")

    # Rigid body
    body = RigidBody(mass=1.0, inertia=torch.tensor(0.1), dim=2)
    state = RigidBodyState(n_bodies=1, dim=2)
    force = torch.zeros(1, 2)
    state = body.step(state, force, torque=torch.zeros(1), dt=1e-3)
    print(f"  RigidBody pos   : {state.pos.tolist()}")

    # MPM (small grid for speed)
    pos_mpm = torch.rand(32, 2) * 0.4 + 0.3
    mpm_state = MPMState(pos_mpm)
    sim = MPMSimulator(grid_resolution=16, material="elastic", dt=1e-4)
    t0 = time.perf_counter()
    mpm_state = sim(mpm_state, n_steps=5)
    dt_mpm = time.perf_counter() - t0
    print(f"  MPM (32 pts)    : {dt_mpm:.3f}s  pos range [{mpm_state.pos.min():.3f}, {mpm_state.pos.max():.3f}]")

    # SPH
    sph = SPHParticles(n_particles=50, smoothing_length=0.1, dim=2)
    sph_pos = torch.rand(50, 2)
    sph_vel = torch.zeros(50, 2)
    t0 = time.perf_counter()
    sph_pos, sph_vel = sph(sph_pos, sph_vel, dt=1e-3)
    dt_sph = time.perf_counter() - t0
    print(f"  SPH (50 pts)    : {dt_sph:.3f}s  pos range [{sph_pos.min():.3f}, {sph_pos.max():.3f}]")

    # ------------------------------------------------------------------
    # Feature 19: World Foundation Model
    # ------------------------------------------------------------------
    print("\n[Feature 19] World foundation model (physics fallback)")
    adapter = CosmosAdapter(WorldModelConfig(n_frames=4, resolution=(32, 32)))
    print(f"  Using Cosmos    : {adapter.using_cosmos}")

    initial_state = torch.rand(3, 32, 32)
    frames = adapter.generate(initial_state, "channel flow", n_frames=4)
    print(f"  Generated frames: {frames.shape}")

    state_dict = adapter.extract_state(frames)
    print(f"  Vel proxy shape : {state_dict['velocity_proxy'].shape}")

    prior_loss = adapter.physics_prior_loss(
        pinn(x_col[:4]).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 32, 32),
        frames,
    )
    print(f"  Physics prior L : {float(prior_loss.detach()):.4e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("All Group-E features demonstrated successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
