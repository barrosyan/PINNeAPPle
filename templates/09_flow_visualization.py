"""09_flow_visualization.py — Post-processing and flow visualization.

Demonstrates:
- FlowVisualizer for high-level figure generation
- compute_streamlines with RK4 integration on a 2D grid
- plot_streamlines_2d_model using a trained PINN
- compute_isosurface for a 3D scalar field (marching cubes)
"""

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_inference.postprocess import (
    FlowVisualizer,
    compute_streamlines,
    plot_streamlines_2d_model,
    compute_isosurface,
)


# ---------------------------------------------------------------------------
# Build a simple analytic-flow surrogate model (for demo purposes)
# Uses: u = -y, v = x  (solid-body rotation)
# ---------------------------------------------------------------------------

class AnalyticRotationModel(nn.Module):
    """Returns (u, v) = (-y, x)  — solid-body rotation at unit angular speed."""

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0:1]
        y = xy[:, 1:2]
        u = -y
        v =  x
        return torch.cat([u, v], dim=1)


# ---------------------------------------------------------------------------
# Build a 3D scalar model for isosurface demo: f = sin(πx)sin(πy)sin(πz)
# ---------------------------------------------------------------------------

class Analytic3DModel(nn.Module):
    """Scalar field sin(πx)sin(πy)sin(πz) on [0,1]³."""

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x, y, z = xyz[:, 0:1], xyz[:, 1:2], xyz[:, 2:3]
        return (torch.sin(math.pi * x)
                * torch.sin(math.pi * y)
                * torch.sin(math.pi * z))


def main():
    # --- 2D Flow: compute streamlines on a grid ---------------------------
    n_grid = 60
    x_lin = np.linspace(-1.5, 1.5, n_grid)
    y_lin = np.linspace(-1.5, 1.5, n_grid)
    xx, yy = np.meshgrid(x_lin, y_lin)

    # Analytic velocity field: rotation
    u_field = -yy.astype(np.float32)
    v_field =  xx.astype(np.float32)

    # Seed points along a ring
    angles = np.linspace(0, 2 * math.pi, 20, endpoint=False)
    seed_pts = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

    streamlines = compute_streamlines(
        u_field=u_field,
        v_field=v_field,
        x_grid=x_lin,
        y_grid=y_lin,
        seed_points=seed_pts,
        max_length=3.0,
        dt=0.02,
    )
    print(f"Computed {len(streamlines)} streamlines")

    # --- 2D Flow: model-driven plot ---------------------------------------
    rotation_model = AnalyticRotationModel()

    fig_2d = plot_streamlines_2d_model(
        model=rotation_model,
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        n_grid=50,
        velocity_channels=(0, 1),
        n_seeds=25,
        title="Solid-body rotation streamlines",
    )
    if fig_2d is not None:
        fig_2d.savefig("09_streamlines_2d.png", dpi=120)
        print("Saved 09_streamlines_2d.png")
    else:
        # Fallback manual plot if figure not returned
        fig, ax = plt.subplots(figsize=(6, 6))
        for sl in streamlines:
            ax.plot(sl[:, 0], sl[:, 1], "b-", linewidth=0.8, alpha=0.7)
        ax.quiver(xx[::8, ::8], yy[::8, ::8],
                  u_field[::8, ::8], v_field[::8, ::8],
                  alpha=0.5, scale=30)
        ax.set_aspect("equal")
        ax.set_title("Solid-body rotation streamlines")
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig("09_streamlines_2d.png", dpi=120)
        print("Saved 09_streamlines_2d.png (fallback)")

    # --- 3D Isosurface ----------------------------------------------------
    n3 = 32
    xyz_lin = np.linspace(0, 1, n3, dtype=np.float32)
    X3, Y3, Z3 = np.meshgrid(xyz_lin, xyz_lin, xyz_lin, indexing="ij")
    pts3 = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=1)

    model_3d = Analytic3DModel()
    with torch.no_grad():
        scalar_field = model_3d(
            torch.tensor(pts3, dtype=torch.float32)
        ).numpy().reshape(n3, n3, n3)

    iso_level = 0.5
    result = compute_isosurface(
        scalar_field=scalar_field,
        level=iso_level,
        spacing=(1.0 / n3, 1.0 / n3, 1.0 / n3),
    )

    if result is not None and len(result) == 2:
        verts, faces = result
        print(f"Isosurface: {len(verts)} vertices, {len(faces)} faces")

        fig3d = plt.figure(figsize=(7, 6))
        ax3d = fig3d.add_subplot(111, projection="3d")
        ax3d.plot_trisurf(
            verts[:, 0], verts[:, 1], faces, verts[:, 2],
            cmap="viridis", alpha=0.7, lw=0
        )
        ax3d.set_title(f"Isosurface of sin(πx)sin(πy)sin(πz) at level={iso_level}")
        ax3d.set_xlabel("x")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel("z")
        plt.tight_layout()
        plt.savefig("09_isosurface_3d.png", dpi=100)
        print("Saved 09_isosurface_3d.png")
    else:
        print("Isosurface extraction not available (install scikit-image).")

    # --- FlowVisualizer high-level API ------------------------------------
    fv = FlowVisualizer(model=rotation_model, device="cpu")
    fv_fig = fv.plot_2d_flow(
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        n_grid=60,
        velocity_channels=(0, 1),
        show_streamlines=True,
        title="FlowVisualizer: solid-body rotation",
    )
    if fv_fig is not None:
        fv_fig.savefig("09_flow_visualizer.png", dpi=120)
        print("Saved 09_flow_visualizer.png")


if __name__ == "__main__":
    main()
