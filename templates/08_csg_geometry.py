"""08_csg_geometry.py — CSG geometry pipeline for PINN domains.

Demonstrates:
- Building complex 2D domains: L-shape, annulus, channel-with-hole
- CSG operators: union, intersection, difference
- Sampling interior collocation and boundary points
- Visualizing all three geometries
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from pinneaple_geom.csg import (
    CSGRectangle, CSGCircle,
    lshape, csg_annulus, channel_with_hole,
    CSGUnion, CSGDifference,
)


def visualize_domain(ax, interior, boundary, title: str, color_int="steelblue",
                     color_bnd="crimson"):
    """Plot interior collocation and boundary points."""
    ax.scatter(interior[:, 0], interior[:, 1], s=1.5, c=color_int,
               alpha=0.4, label=f"Interior ({len(interior)})")
    ax.scatter(boundary[:, 0], boundary[:, 1], s=6, c=color_bnd,
               alpha=0.8, label=f"Boundary ({len(boundary)})")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, markerscale=3)
    ax.grid(True, alpha=0.2)


def main():
    np.random.seed(0)
    N_INT = 3000   # interior points
    N_BND = 400    # boundary points

    # --- Domain 1: L-shape (2×2 square minus 1×1 corner) -----------------
    # lshape(W, H, notch_w, notch_h)
    lshape_domain = lshape(W=2.0, H=2.0, notch_w=1.0, notch_h=1.0)
    lshape_int = lshape_domain.sample_interior(N_INT, seed=1)
    lshape_bnd = lshape_domain.sample_boundary(N_BND, seed=2)
    print(f"L-shape: {lshape_int.shape[0]} interior, {lshape_bnd.shape[0]} boundary")

    # --- Domain 2: Annulus (ring between two circles) --------------------
    # csg_annulus(cx, cy, r_inner, r_outer)
    annulus_domain = csg_annulus(cx=0.0, cy=0.0, r_inner=0.3, r_outer=1.0)
    annulus_int = annulus_domain.sample_interior(N_INT, seed=3)
    annulus_bnd = annulus_domain.sample_boundary(N_BND, seed=4)
    print(f"Annulus: {annulus_int.shape[0]} interior, {annulus_bnd.shape[0]} boundary")

    # --- Domain 3: Channel with circular hole ----------------------------
    # channel_with_hole(W, H, cx, cy, r)
    channel_domain = channel_with_hole(W=4.0, H=1.0, cx=1.0, cy=0.5, r=0.2)
    channel_int = channel_domain.sample_interior(N_INT, seed=5)
    channel_bnd = channel_domain.sample_boundary(N_BND, seed=6)
    print(f"Channel+hole: {channel_int.shape[0]} interior, {channel_bnd.shape[0]} boundary")

    # --- Custom CSG: two overlapping rectangles (union) ------------------
    rect1 = CSGRectangle(x_min=0.0, y_min=0.0, x_max=1.5, y_max=1.0)
    rect2 = CSGRectangle(x_min=0.7, y_min=0.3, x_max=2.0, y_max=1.3)
    union_domain = CSGUnion(rect1, rect2)
    union_int = union_domain.sample_interior(N_INT, seed=7)
    union_bnd = union_domain.sample_boundary(N_BND, seed=8)
    print(f"Union rect: {union_int.shape[0]} interior, {union_bnd.shape[0]} boundary")

    # --- Custom CSG: rectangle minus circle ------------------------------
    big_rect = CSGRectangle(x_min=-1.5, y_min=-1.5, x_max=1.5, y_max=1.5)
    hole     = CSGCircle(cx=0.0, cy=0.0, r=0.8)
    diff_domain = CSGDifference(big_rect, hole)
    diff_int = diff_domain.sample_interior(N_INT, seed=9)
    diff_bnd = diff_domain.sample_boundary(N_BND, seed=10)
    print(f"Rect-circle: {diff_int.shape[0]} interior, {diff_bnd.shape[0]} boundary")

    # --- Plot all five domains --------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    visualize_domain(axes[0, 0], lshape_int, lshape_bnd, "L-shape domain")
    visualize_domain(axes[0, 1], annulus_int, annulus_bnd, "Annulus domain")
    visualize_domain(axes[0, 2], channel_int, channel_bnd, "Channel with hole")
    visualize_domain(axes[1, 0], union_int, union_bnd, "Union of two rectangles")
    visualize_domain(axes[1, 1], diff_int, diff_bnd, "Rectangle minus circle")

    # SDF heatmap for the channel domain
    nx, ny = 200, 60
    x_lin = np.linspace(-0.2, 4.2, nx)
    y_lin = np.linspace(-0.2, 1.2, ny)
    xx, yy = np.meshgrid(x_lin, y_lin)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    sdf_vals = channel_domain.sdf(pts).reshape(ny, nx)
    im = axes[1, 2].contourf(xx, yy, sdf_vals, levels=30, cmap="RdBu_r")
    plt.colorbar(im, ax=axes[1, 2])
    axes[1, 2].contour(xx, yy, sdf_vals, levels=[0], colors="k", linewidths=2)
    axes[1, 2].set_title("Channel SDF (zero = boundary)")
    axes[1, 2].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("08_csg_geometry_result.png", dpi=120)
    print("Saved 08_csg_geometry_result.png")


if __name__ == "__main__":
    main()
