"""Lesson 04 — Geometry Generation with CSG.

What you will learn
-------------------
  CSG (Constructive Solid Geometry) lets you build complex 2D domains
  from simple primitives: rectangles, circles, ellipses, polygons.

  Step 1 — Create primitive shapes (rectangle, circle, ellipse)
  Step 2 — Boolean operations: union (+), difference (-), intersection (*)
  Step 3 — Sample collocation and boundary points
  Step 4 — Build named Physics domains: channel, L-shape, annulus
  Step 5 — Visualise all domains and point distributions

Why geometry matters for PINNs
-------------------------------
  • The PINN collocation points come from the domain geometry.
  • Irregular domains (holes, thin channels) affect how much the PDE
    is enforced in each region.
  • PINNeAPPle's CSG handles sampling, SDF queries, and boundary
    normals automatically.

Run this lesson
---------------
    python -m pinneaple_learning.course.lesson_04_geometry
"""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# PINNeAPPle geometry imports — the entire lesson uses these
from pinneaple_geom import (
    CSGRectangle, CSGCircle, CSGEllipse, CSGPolygon,
    CSGUnion, CSGIntersection, CSGDifference,
    lshape, csg_annulus, channel_with_hole,
    PhysicsDomain2D, LShapeDomain2D, ChannelWithObstacleDomain2D,
    AnnularDomain2D,
)

N_INTERIOR = 1_200
N_BOUNDARY = 300


# ── Helper: plot a shape's sample points ─────────────────────────────────
def _plot_domain(ax, shape, title: str, n_int: int = N_INTERIOR,
                 n_bnd: int = N_BOUNDARY) -> None:
    pts_int = shape.sample_interior(n_int, seed=0)
    pts_bnd = shape.sample_boundary(n_bnd, seed=0)
    ax.scatter(pts_int[:, 0], pts_int[:, 1], s=1,  c="steelblue",  alpha=0.4, label="interior")
    ax.scatter(pts_bnd[:, 0], pts_bnd[:, 1], s=4,  c="crimson",    alpha=0.8, label="boundary")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, markerscale=3)
    ax.grid(True, alpha=0.2)


def main() -> None:
    print("─" * 60)
    print("  Lesson 04 — Geometry Generation with PINNeAPPle CSG")
    print("─" * 60)

    # ── Step 1: Primitives ─────────────────────────────────────────────────
    print("\n  [Step 1] Creating primitive shapes...")
    rect    = CSGRectangle(0, 0, 2, 1)
    circle  = CSGCircle(1.0, 0.5, 0.3)
    ellipse = CSGEllipse(1.0, 0.5, 0.6, 0.25)
    print(f"    Rectangle  interior sample shape: {rect.sample_interior(10).shape}")
    print(f"    Circle     interior sample shape: {circle.sample_interior(10).shape}")

    # ── Step 2: Boolean operations ────────────────────────────────────────
    print("  [Step 2] Boolean CSG operations...")
    rect_with_hole  = rect - circle           # difference
    two_overlapping = rect + circle           # union
    overlap_only    = rect * ellipse          # intersection
    print("    Created: rect-circle, rect+circle, rect∩ellipse")

    # ── Step 3: Sample points and inspect ─────────────────────────────────
    print("  [Step 3] Sampling collocation and boundary points...")
    pts_col = rect_with_hole.sample_interior(N_INTERIOR, seed=42)
    pts_bnd = rect_with_hole.sample_boundary(N_BOUNDARY, seed=42)
    print(f"    Interior collocation points: {pts_col.shape}")
    print(f"    Boundary points:             {pts_bnd.shape}")

    # ── Step 4: Named domains from PINNeAPPle ─────────────────────────────
    print("  [Step 4] Building named Physics domains...")
    l_domain  = LShapeDomain2D()
    ann_domain = AnnularDomain2D()
    ch_domain  = ChannelWithObstacleDomain2D()
    print("    Built: L-shape, annulus, channel-with-obstacle")

    # SDF query: negative inside, positive outside
    test_pts  = np.array([[0.5, 0.5], [3.0, 3.0]], dtype=np.float32)
    sdf_vals  = rect.sdf(test_pts)
    inside_mask = rect.contains(test_pts)
    print(f"\n    SDF at (0.5,0.5)={sdf_vals[0]:.3f}  (inside: {inside_mask[0]})")
    print(f"    SDF at (3.0,3.0)={sdf_vals[1]:.3f}  (inside: {inside_mask[1]})")

    # ── Step 5: Visualise ─────────────────────────────────────────────────
    print("  [Step 5] Plotting all domains...")

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    domains = [
        (rect,            "Rectangle [0,2]×[0,1]"),
        (circle,          "Circle  r=0.3"),
        (rect_with_hole,  "Rect − Circle  (hole)"),
        (two_overlapping, "Rect ∪ Circle"),
        (overlap_only,    "Rect ∩ Ellipse"),
        (l_domain,        "L-shape domain"),
        (ann_domain,      "Annular domain"),
        (ch_domain,       "Channel with obstacle"),
    ]
    for ax, (dom, title) in zip(axes.ravel(), domains):
        try:
            _plot_domain(ax, dom, title)
        except Exception as exc:
            ax.text(0.5, 0.5, f"(skip: {exc})", ha="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=9)

    plt.suptitle("Lesson 04 — PINNeAPPle CSG Geometry Library", fontsize=13)
    plt.tight_layout()
    out = "lesson_04_geometry.png"
    plt.savefig(out, dpi=120)
    print(f"  Saved {out}")

    # ── Show how to feed geometry into a PINN ─────────────────────────────
    print("""
  How to connect geometry to a PINN training loop:

      from pinneaple_geom import CSGRectangle, CSGCircle
      import torch

      domain = CSGRectangle(0, 0, 1, 1) - CSGCircle(0.5, 0.5, 0.2)

      # Collocation points (interior)
      pts_col = domain.sample_interior(2000, seed=42)
      x_col   = torch.tensor(pts_col, requires_grad=True)

      # Boundary points
      pts_bnd = domain.sample_boundary(500, seed=42)
      x_bc    = torch.tensor(pts_bnd)

      # SDF for hard BC enforcement:
      sdf_vals = torch.tensor(domain.sdf(pts_col))

  The SDF can be used with HardBC for exact zero-Dirichlet enforcement:
      hard_bc = HardBC(
          distance_fn = lambda x: domain_sdf(x),  # → 0 on boundary
          bc_value_fn = lambda x: torch.zeros(len(x), 1),
      )
      net_wrapped = hard_bc.wrap_model(net)
      # net_wrapped(x) = sdf(x) * net(x)  →  zero on every boundary point

  Next lesson:
    python -m pinneaple_learning.course.lesson_05_2d_poisson
""")


if __name__ == "__main__":
    main()
