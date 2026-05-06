"""CSG domain demo for PINN collocation point generation.

Demonstrates L-shape, annulus, and channel-with-hole domains built with
the pinneaple_geom CSG module.  Each domain is sampled for interior
collocation points and boundary points, then visualised with matplotlib.

Run::

    python examples/pinneaple_geom/07_csg_domain_demo.py

Output images are written to examples/pinneaple_geom/_out/.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Import CSG API
# ---------------------------------------------------------------------------
from pinneaple_geom.csg import (
    CSGRectangle,
    CSGCircle,
    CSGEllipse,
    CSGPolygon,
    lshape,
    annulus,
    channel_with_hole,
    t_junction,
)

OUT_DIR = Path(__file__).parent / "_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_COL = 4096   # interior collocation points
N_BND = 512    # boundary points


# ---------------------------------------------------------------------------
# Helper: scatter plot coloured by signed distance
# ---------------------------------------------------------------------------

def _scatter_domain(
    ax,
    domain,
    pts_int: np.ndarray,
    pts_bnd: np.ndarray,
    title: str,
) -> None:
    """Plot interior + boundary points with SDF-based coloring."""
    sdf_int = domain.sdf(pts_int.astype(np.float64))
    sc = ax.scatter(
        pts_int[:, 0], pts_int[:, 1],
        c=sdf_int, cmap="RdBu_r", s=1.5, alpha=0.6,
        vmin=sdf_int.min(), vmax=max(-sdf_int.min() * 0.1, -sdf_int.min()),
    )
    ax.scatter(
        pts_bnd[:, 0], pts_bnd[:, 1],
        c="k", s=3, alpha=0.9, label="boundary",
    )
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7)
    return sc


# ---------------------------------------------------------------------------
# Domain 1: L-shape
# ---------------------------------------------------------------------------

def demo_lshape():
    """L-shaped domain via CSGDifference.

    Domain: big rectangle [0,2] x [0,2] minus upper-right [1,2] x [1,2].
    """
    domain = lshape(width1=2.0, height1=2.0, width2=1.0, height2=1.0)

    pts_int = domain.sample_interior(N_COL, seed=0)
    pts_bnd = domain.sample_boundary(N_BND, seed=1)

    print(f"[L-shape]  interior: {pts_int.shape}, boundary: {pts_bnd.shape}")
    print(f"           SDF mean interior: {domain.sdf(pts_int.astype(np.float64)).mean():.4f}")
    print(f"           BBox: {domain.bounds_min} -> {domain.bounds_max}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: collocation scatter coloured by SDF
    sc = _scatter_domain(axes[0], domain, pts_int, pts_bnd, "L-shape: collocation points (coloured by SDF)")
    fig.colorbar(sc, ax=axes[0], label="SDF (neg. inside)")

    # Right: 2D SDF heatmap using a regular grid
    nx, ny = 200, 200
    xs = np.linspace(domain.bounds_min[0], domain.bounds_max[0], nx)
    ys = np.linspace(domain.bounds_min[1], domain.bounds_max[1], ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    sdf_grid = domain.sdf(grid_pts).reshape(nx, ny)

    im = axes[1].pcolormesh(XX, YY, sdf_grid, cmap="seismic_r", shading="auto",
                             vmin=-max(abs(sdf_grid.min()), sdf_grid.max()),
                             vmax=max(abs(sdf_grid.min()), sdf_grid.max()))
    axes[1].contour(XX, YY, sdf_grid, levels=[0.0], colors="k", linewidths=1.5)
    fig.colorbar(im, ax=axes[1], label="SDF")
    axes[1].set_title("L-shape: SDF heatmap (zero contour = boundary)")
    axes[1].set_aspect("equal")

    fig.suptitle("CSG Demo: L-Shape Domain", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "07_csg_lshape.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"           Saved: {out}\n")


# ---------------------------------------------------------------------------
# Domain 2: Annulus (ring)
# ---------------------------------------------------------------------------

def demo_annulus():
    """Annular domain: disk with inner hole.

    Outer radius R=1.0, inner radius r=0.3, centred at origin.
    Useful for Taylor-Couette flow, concentric pipe problems.
    """
    domain = annulus(cx=0.0, cy=0.0, r_inner=0.3, r_outer=1.0)

    pts_int = domain.sample_interior(N_COL, seed=2)
    pts_bnd = domain.sample_boundary(N_BND, seed=3)

    print(f"[Annulus]  interior: {pts_int.shape}, boundary: {pts_bnd.shape}")
    print(f"           SDF mean interior: {domain.sdf(pts_int.astype(np.float64)).mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc = _scatter_domain(axes[0], domain, pts_int, pts_bnd, "Annulus: collocation points (coloured by SDF)")
    fig.colorbar(sc, ax=axes[0], label="SDF (neg. inside)")

    # SDF heatmap
    nx, ny = 200, 200
    xs = np.linspace(-1.1, 1.1, nx)
    ys = np.linspace(-1.1, 1.1, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    sdf_grid = domain.sdf(grid_pts).reshape(nx, ny)

    vabs = float(np.max(np.abs(sdf_grid)))
    im = axes[1].pcolormesh(XX, YY, sdf_grid, cmap="seismic_r", shading="auto",
                             vmin=-vabs, vmax=vabs)
    axes[1].contour(XX, YY, sdf_grid, levels=[0.0], colors="k", linewidths=1.5)
    fig.colorbar(im, ax=axes[1], label="SDF")
    axes[1].set_title("Annulus: SDF heatmap")
    axes[1].set_aspect("equal")

    # show radial lines for reference
    for theta in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        axes[1].plot(
            [0.3 * np.cos(theta), 1.0 * np.cos(theta)],
            [0.3 * np.sin(theta), 1.0 * np.sin(theta)],
            "w--", linewidth=0.5, alpha=0.4,
        )

    fig.suptitle("CSG Demo: Annular Domain", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "07_csg_annulus.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"           Saved: {out}\n")


# ---------------------------------------------------------------------------
# Domain 3: Channel with circular obstacle
# ---------------------------------------------------------------------------

def demo_channel_with_hole():
    """Rectangular channel [0,4] x [0,1] with a circular obstacle.

    Classic PINN benchmark: flow around a cylinder in a channel.
    """
    length, height = 4.0, 1.0
    hole_cx, hole_cy, hole_r = 1.0, 0.5, 0.15

    domain = channel_with_hole(length, height, hole_cx, hole_cy, hole_r)

    pts_int = domain.sample_interior(N_COL, seed=4)
    pts_bnd = domain.sample_boundary(N_BND, seed=5)

    print(f"[Channel+hole] interior: {pts_int.shape}, boundary: {pts_bnd.shape}")
    print(f"               SDF mean interior: {domain.sdf(pts_int.astype(np.float64)).mean():.4f}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: scatter coloured by SDF
    sdf_int = domain.sdf(pts_int.astype(np.float64))
    sc = axes[0].scatter(
        pts_int[:, 0], pts_int[:, 1],
        c=sdf_int, cmap="RdBu_r", s=2.0, alpha=0.7,
    )
    axes[0].scatter(pts_bnd[:, 0], pts_bnd[:, 1], c="k", s=4, alpha=0.8, label="boundary")
    fig.colorbar(sc, ax=axes[0], label="SDF (neg. inside)")
    circ = plt.Circle((hole_cx, hole_cy), hole_r, color="red", fill=False, linewidth=1.5)
    axes[0].add_patch(circ)
    axes[0].set_title(f"Channel with obstacle: {N_COL} collocation + {N_BND} boundary points")
    axes[0].set_aspect("equal")
    axes[0].legend(fontsize=8)

    # Bottom: SDF heatmap with boundary contour
    nx, ny = 300, 80
    xs = np.linspace(0.0, length, nx)
    ys = np.linspace(0.0, height, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)
    sdf_grid = domain.sdf(grid_pts).reshape(nx, ny)

    vabs = float(np.max(np.abs(sdf_grid)))
    im = axes[1].pcolormesh(XX, YY, sdf_grid, cmap="seismic_r", shading="auto",
                             vmin=-vabs, vmax=vabs)
    axes[1].contour(XX, YY, sdf_grid, levels=[0.0], colors="k", linewidths=1.5)
    fig.colorbar(im, ax=axes[1], label="SDF")
    axes[1].set_title("Channel with obstacle: SDF heatmap (boundary = zero contour)")
    axes[1].set_aspect("equal")

    fig.suptitle("CSG Demo: Channel with Circular Obstacle", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "07_csg_channel_hole.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"               Saved: {out}\n")


# ---------------------------------------------------------------------------
# Bonus: CSG operator showcase
# ---------------------------------------------------------------------------

def demo_csg_operators():
    """Show union, intersection, and difference with simple primitives."""
    r1 = CSGRectangle(0.0, 0.0, 1.0, 1.0)
    c1 = CSGCircle(0.8, 0.8, 0.5)

    ops = {
        "Union (r + c)":        r1 + c1,
        "Intersection (r * c)": r1 * c1,
        "Difference (r - c)":   r1 - c1,
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    xs = np.linspace(-0.6, 1.6, 200)
    ys = np.linspace(-0.6, 1.6, 200)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)

    for ax, (label, dom) in zip(axes, ops.items()):
        sdf_g = dom.sdf(grid_pts).reshape(200, 200)
        vabs = float(np.max(np.abs(sdf_g)))
        im = ax.pcolormesh(XX, YY, sdf_g, cmap="RdBu_r", shading="auto",
                            vmin=-vabs, vmax=vabs)
        ax.contour(XX, YY, sdf_g, levels=[0.0], colors="k", linewidths=1.5)
        fig.colorbar(im, ax=ax, label="SDF")
        # sample interior
        pts_i = dom.sample_interior(300, seed=0)
        ax.scatter(pts_i[:, 0], pts_i[:, 1], c="lime", s=3, alpha=0.5,
                   label="interior pts")
        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")
        ax.legend(fontsize=7)

    fig.suptitle("CSG Boolean Operators: Rectangle vs Circle", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "07_csg_operators.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[CSG Ops]  Saved: {out}\n")


# ---------------------------------------------------------------------------
# Bonus: Polygon domain
# ---------------------------------------------------------------------------

def demo_polygon():
    """Irregular polygon domain (hexagon with a rectangular notch)."""
    import math
    # regular hexagon vertices
    hex_verts = np.array([
        [math.cos(math.pi / 3 * i), math.sin(math.pi / 3 * i)]
        for i in range(6)
    ])
    hex_dom = CSGPolygon(hex_verts)
    notch   = CSGRectangle(-0.15, 0.5, 0.15, 1.1)
    domain  = hex_dom - notch

    pts_int = domain.sample_interior(2000, seed=7)
    pts_bnd = domain.sample_boundary(400, seed=8)

    print(f"[Hexagon-notch] interior: {pts_int.shape}, boundary: {pts_bnd.shape}")

    fig, ax = plt.subplots(figsize=(6, 6))
    sdf_int = domain.sdf(pts_int.astype(np.float64))
    sc = ax.scatter(pts_int[:, 0], pts_int[:, 1], c=sdf_int, cmap="RdBu_r",
                    s=3, alpha=0.7)
    ax.scatter(pts_bnd[:, 0], pts_bnd[:, 1], c="k", s=5, alpha=0.9, label="boundary")
    fig.colorbar(sc, ax=ax, label="SDF")
    ax.set_title("Hexagon minus rectangular notch")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)

    out = OUT_DIR / "07_csg_polygon.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"               Saved: {out}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PINNeAPPle CSG Domain Demo (Feature 12)")
    print("=" * 60)
    print()

    demo_lshape()
    demo_annulus()
    demo_channel_with_hole()
    demo_csg_operators()
    demo_polygon()

    print("All demos complete.  Figures written to:", OUT_DIR)
