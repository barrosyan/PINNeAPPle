"""11_cad_to_cfd.py — CAD-to-CFD full pipeline.

Demonstrates:
- CADToCFDPipeline: load geometry, mesh, set BCs, solve NS, extract PINN data
- NSFlowSolver for incompressible Navier-Stokes
- CFDMesh construction (structured rectangular fallback when gmsh is absent)
- PINN refinement pass using CFD data
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_solvers.cfd_pipeline import CADToCFDPipeline, CFDMesh, NSFlowSolver


# ---------------------------------------------------------------------------
# Utility: build a simple structured 2D channel mesh without gmsh
# ---------------------------------------------------------------------------

def make_channel_mesh(nx: int = 40, ny: int = 20,
                      Lx: float = 2.0, Ly: float = 0.5) -> CFDMesh:
    """Rectangular channel mesh (nx × ny quad → triangular elements)."""
    x_lin = np.linspace(0, Lx, nx + 1)
    y_lin = np.linspace(0, Ly, ny + 1)
    xx, yy = np.meshgrid(x_lin, y_lin)
    nodes_2d = np.stack([xx.ravel(), yy.ravel()], axis=1)
    nodes = np.column_stack([nodes_2d, np.zeros(len(nodes_2d))])  # (N, 3)

    # Triangulate each quad cell into 2 triangles
    def node_id(i, j): return j * (nx + 1) + i
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = node_id(i,   j)
            n1 = node_id(i+1, j)
            n2 = node_id(i+1, j+1)
            n3 = node_id(i,   j+1)
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])
    elements = np.array(elements, dtype=np.int64)

    # Boundary tags
    left_mask  = nodes[:, 0] < 1e-8
    right_mask = nodes[:, 0] > Lx - 1e-8
    bot_mask   = nodes[:, 1] < 1e-8
    top_mask   = nodes[:, 1] > Ly - 1e-8

    boundary_tags = {
        "inlet":  np.where(left_mask)[0],
        "outlet": np.where(right_mask)[0],
        "bottom": np.where(bot_mask)[0],
        "top":    np.where(top_mask)[0],
    }
    return CFDMesh(nodes, elements, boundary_tags)


def main():
    torch.manual_seed(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- CFDMesh (channel geometry without STL/STEP file) ----------------
    mesh = make_channel_mesh(nx=40, ny=20, Lx=2.0, Ly=0.5)
    print(f"Mesh: {mesh.nodes.shape[0]} nodes, {mesh.elements.shape[0]} elements")
    for tag, idxs in mesh.boundary_tags.items():
        print(f"  Boundary '{tag}': {len(idxs)} nodes")

    # --- NSFlowSolver: Navier-Stokes on the mesh -------------------------
    nu = 0.01   # kinematic viscosity  (Re ≈ 100)
    ns_solver = NSFlowSolver(mesh=mesh, nu=nu)
    ns_solver.set_boundary_conditions(
        inlet_velocity=(1.0, 0.0),     # u_inlet = (1,0)
        no_slip_tags=["bottom", "top"],
    )
    print("\nSolving NS (Picard iteration or direct)...")
    cfd_result = ns_solver.solve(max_iter=50, tol=1e-4)
    print(f"NS solver converged in {cfd_result.get('n_iter', '?')} iterations.")
    print(f"  p_max = {cfd_result['p'].max():.4f}")
    print(f"  u_max = {np.sqrt((cfd_result['u']**2).sum(axis=1)).max():.4f}")

    # --- CADToCFDPipeline (high-level, wraps everything) -----------------
    # This shows the high-level API; uses the mesh we already built
    pipeline = CADToCFDPipeline(nu=nu)
    pipeline.load_mesh(mesh)             # inject pre-built mesh directly
    pipeline.set_bcs(inlet_velocity=(1.0, 0.0),
                     no_slip_tags=["bottom", "top"])
    results = pipeline.solve(max_iter=50, tol=1e-4)
    train_data = pipeline.to_pinn_data()   # dict: x_col, u_data, p_data
    print(f"\nPINN training data: {train_data['x_col'].shape[0]} points")

    # --- PINN refinement pass --------------------------------------------
    # Use CFD data as supervised signal to train a PINN surrogate
    pinn = nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 3),           # outputs: (u, v, p)
    ).to(device)

    x_col = torch.tensor(train_data["x_col"], dtype=torch.float32, device=device)
    u_data = torch.tensor(
        np.stack([train_data["u_data"][:, 0],
                  train_data["u_data"][:, 1]], axis=1),
        dtype=torch.float32, device=device
    )
    p_data = torch.tensor(
        train_data["p_data"].reshape(-1, 1),
        dtype=torch.float32, device=device
    )

    optimizer = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    n_epochs_pinn = 2000
    print("\nFitting PINN to CFD data...")
    for ep in range(1, n_epochs_pinn + 1):
        optimizer.zero_grad()
        out = pinn(x_col)
        loss = ((out[:, 0:2] - u_data).pow(2).mean()
                + (out[:, 2:3] - p_data).pow(2).mean())
        loss.backward()
        optimizer.step()
        if ep % 500 == 0:
            print(f"  epoch {ep:4d}  loss={loss.item():.4e}")

    print("PINN refinement complete.")

    # --- Visualization ----------------------------------------------------
    n_vis = mesh.nodes.shape[0]
    xy_vis = torch.tensor(mesh.nodes[:, :2], dtype=torch.float32, device=device)
    with torch.no_grad():
        pinn_out = pinn(xy_vis).cpu().numpy()

    u_pinn  = pinn_out[:, 0]
    v_pinn  = pinn_out[:, 1]
    p_pinn  = pinn_out[:, 2]
    speed   = np.sqrt(u_pinn**2 + v_pinn**2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sc0 = axes[0].scatter(mesh.nodes[:, 0], mesh.nodes[:, 1],
                          c=speed, cmap="viridis", s=3)
    plt.colorbar(sc0, ax=axes[0])
    axes[0].set_title("PINN speed |u|  (refined on CFD data)")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    sc1 = axes[1].scatter(mesh.nodes[:, 0], mesh.nodes[:, 1],
                          c=p_pinn, cmap="RdBu_r", s=3)
    plt.colorbar(sc1, ax=axes[1])
    axes[1].set_title("PINN pressure p")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x")

    plt.tight_layout()
    plt.savefig("11_cad_to_cfd_result.png", dpi=120)
    print("Saved 11_cad_to_cfd_result.png")


if __name__ == "__main__":
    main()
