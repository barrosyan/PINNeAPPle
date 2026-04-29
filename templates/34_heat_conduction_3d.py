"""34_heat_conduction_3d.py — 3D heat conduction PINN with FEM comparison.

Demonstrates:
- Heat3DPreset: preconfigured 3D heat conduction problem from pinneaple_environment
- FEMSolver (FEniCS bridge): solve the same problem with a FEM reference
- FDMSolver: finite difference reference on a regular 3D grid
- Error map comparison: PINN vs FEM, PINN vs FDM on a cross-section plane
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_environment import Heat3DPreset, HeatPresetConfig

try:
    from pinneaple_solvers.fem import FEMSolver, FEMConfig
    _FEM = True
except ImportError:
    _FEM = False
    print("[warn] FEniCS not available — FEM comparison will be skipped.")

from pinneaple_solvers.fdm import FDMSolver3D, FDMConfig3D


# ---------------------------------------------------------------------------
# Problem: 3D steady-state heat conduction
# Domain: [0,1]³
# PDE:  k ∇²T = -Q  where  k = 1,  Q = 100 * sin(πx)sin(πy)sin(πz)
# BCs:  T = 0 on all six faces (Dirichlet)
# Exact solution:  T = Q₀/(3π²) * sin(πx)sin(πy)sin(πz)
#                  = (100/(3π²)) sin(πx)sin(πy)sin(πz)
# ---------------------------------------------------------------------------

Q0 = 100.0
K_COND = 1.0
T_EXACT_SCALE = Q0 / (3 * math.pi**2)


def T_exact(xyz: np.ndarray) -> np.ndarray:
    return (T_EXACT_SCALE *
            np.sin(math.pi * xyz[:, 0]) *
            np.sin(math.pi * xyz[:, 1]) *
            np.sin(math.pi * xyz[:, 2])).astype(np.float32)


def build_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(3, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 1),
    )


def phi(xyz: torch.Tensor) -> torch.Tensor:
    """Hard BC: T = phi * net, phi vanishes at all six faces."""
    x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
    return (x * (1 - x) * y * (1 - y) * z * (1 - z))


def main():
    torch.manual_seed(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Heat3DPreset -------------------------------------------------------
    preset_config = HeatPresetConfig(
        conductivity=K_COND,
        source_magnitude=Q0,
        source_type="sinusoidal_xyz",
        domain_bounds=[[0, 1], [0, 1], [0, 1]],
        bc_type="dirichlet_zero",
    )
    preset = Heat3DPreset(config=preset_config)
    print("Heat3DPreset configured.")

    # --- PINN ---------------------------------------------------------------
    net   = build_model().to(device)
    model = lambda xyz: phi(xyz) * net(xyz)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    n_epochs  = 8000
    n_col     = 8192
    history   = []

    print(f"Training 3D heat PINN ({n_epochs} epochs) ...")
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        xyz = torch.rand(n_col, 3, device=device, requires_grad=True)
        T   = model(xyz)
        g   = torch.autograd.grad(T.sum(), xyz, create_graph=True)[0]
        T_xx = torch.autograd.grad(g[:, 0].sum(), xyz, create_graph=True)[0][:, 0:1]
        T_yy = torch.autograd.grad(g[:, 1].sum(), xyz, create_graph=True)[0][:, 1:2]
        T_zz = torch.autograd.grad(g[:, 2].sum(), xyz, create_graph=True)[0][:, 2:3]
        Q    = Q0 * torch.sin(math.pi * xyz[:, 0:1]) * \
                    torch.sin(math.pi * xyz[:, 1:2]) * \
                    torch.sin(math.pi * xyz[:, 2:3])
        res  = K_COND * (T_xx + T_yy + T_zz) + Q
        loss = res.pow(2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        history.append(float(loss.item()))
        if epoch % 2000 == 0:
            print(f"  epoch {epoch:5d} | loss = {loss.item():.4e}")

    # --- FDM reference ------------------------------------------------------
    n_fdm = 20
    fdm_config = FDMConfig3D(
        nx=n_fdm, ny=n_fdm, nz=n_fdm,
        dx=1.0 / n_fdm, dy=1.0 / n_fdm, dz=1.0 / n_fdm,
        conductivity=K_COND,
        source_fn=lambda x, y, z: Q0 * np.sin(math.pi * x) *
                                        np.sin(math.pi * y) *
                                        np.sin(math.pi * z),
        bc_value=0.0,
        max_iter=5000,
        tol=1e-8,
    )
    fdm_solver = FDMSolver3D(config=fdm_config)
    T_fdm = fdm_solver.solve()    # returns (nx, ny, nz) array
    print(f"FDM solved ({n_fdm}³ grid).")

    # --- FEM reference (optional) -------------------------------------------
    if _FEM:
        fem_config = FEMConfig(
            mesh_type="box",
            mesh_params={"nx": 10, "ny": 10, "nz": 10},
            conductivity=K_COND,
            source_expression=f"{Q0}*sin(pi*x[0])*sin(pi*x[1])*sin(pi*x[2])",
            bc_expression="0",
        )
        fem_solver = FEMSolver(config=fem_config)
        T_fem_vals, fem_xyz = fem_solver.solve()
        print("FEM solved.")

    # --- Evaluation on cross-section z=0.5 ----------------------------------
    n_vis = 50
    x_ = np.linspace(0, 1, n_vis, dtype=np.float32)
    xx, yy = np.meshgrid(x_, x_)
    zz = 0.5 * np.ones_like(xx)
    xyz_vis = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    xyz_t = torch.tensor(xyz_vis, device=device)
    with torch.no_grad():
        T_pinn = model(xyz_t).cpu().numpy().reshape(n_vis, n_vis)

    T_ref = T_exact(xyz_vis).reshape(n_vis, n_vis)

    err_pinn = np.sqrt(((T_pinn - T_ref)**2).mean()) / np.sqrt((T_ref**2).mean())
    print(f"\nPINN relative L2 error (z=0.5 slice): {err_pinn:.4e}")

    # FDM slice at z=0.5 (midpoint)
    z_idx = n_fdm // 2
    x_fdm = np.linspace(0, 1, n_fdm + 2)[1:-1]
    xx_fdm, yy_fdm = np.meshgrid(x_fdm, x_fdm)
    T_fdm_slice = T_fdm[:, :, z_idx]
    T_exact_fdm = T_exact(
        np.stack([xx_fdm.ravel(), yy_fdm.ravel(),
                  0.5 * np.ones(n_fdm**2)], axis=1)
    ).reshape(n_fdm, n_fdm)
    err_fdm = np.sqrt(((T_fdm_slice - T_exact_fdm)**2).mean()) / \
              np.sqrt((T_exact_fdm**2).mean())
    print(f"FDM  relative L2 error (z=0.5 slice): {err_fdm:.4e}")

    # --- Visualisation -------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    vmin, vmax = T_ref.min(), T_ref.max()

    for ax, field, title in zip(
        axes[0],
        [T_ref, T_pinn, np.abs(T_pinn - T_ref)],
        ["Exact T (z=0.5)", f"PINN T (L2={err_pinn:.2e})", "|PINN - Exact|"],
    ):
        cmap = "hot" if "Exact" in title or "PINN T" in title else "Reds"
        vn, vx = (vmin, vmax) if "|" not in title else (None, None)
        im = ax.contourf(xx, yy, field, levels=30, cmap=cmap, vmin=vn, vmax=vx)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    # FDM comparison
    for ax, field, title in zip(
        axes[1],
        [T_exact_fdm, T_fdm_slice, np.abs(T_fdm_slice - T_exact_fdm)],
        ["Exact T (FDM grid)", f"FDM T (L2={err_fdm:.2e})", "|FDM - Exact|"],
    ):
        cmap = "hot" if "|" not in title else "Reds"
        im = ax.contourf(xx_fdm, yy_fdm, field, levels=20, cmap=cmap)
        plt.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig("34_heat_conduction_3d_result.png", dpi=120)
    print("Saved 34_heat_conduction_3d_result.png")


if __name__ == "__main__":
    main()
