"""Importance sampling on surfaces using a curvature proxy.

What this shows
--------------
- Compute face-level geometric features (normals, areas, curvature proxy).
- Use per-face weights to bias sampling towards high-curvature regions
  (edges, corners, tight features), which is often helpful for PINNs and
  Physics-AI boundary learning.

Run
---
python examples/pinneaple_geom/04_curvature_importance_sampling.py
"""

from __future__ import annotations

import numpy as np

from pinneaple_geom.gen.primitives import build_primitive
from pinneaple_geom.ops.features import compute_face_areas, compute_curvature_proxy
from pinneaple_geom.sample.points import sample_surface_points, sample_surface_points_weighted


def main() -> None:
    # A shape with edges (box) so the curvature proxy has something to pick up.
    mesh = build_primitive("box", extents=(2.0, 1.0, 0.5))

    # Uniform sampling (area-weighted)
    n = 50_000
    pts_u, _, face_u = sample_surface_points(mesh, n)

    # Curvature proxy (0..1) per face
    curv = compute_curvature_proxy(mesh)
    areas = compute_face_areas(mesh)

    # Importance weights: keep area weighting, amplify high curvature.
    # tweak exponent to taste.
    w = areas * (0.05 + curv) ** 2
    pts_c, _, face_c = sample_surface_points_weighted(mesh, n, face_weights=w)

    # Report: average curvature of sampled faces should be higher for importance sampling.
    mu_u = float(np.mean(curv[face_u]))
    mu_c = float(np.mean(curv[face_c]))
    p90_u = float(np.quantile(curv[face_u], 0.90))
    p90_c = float(np.quantile(curv[face_c], 0.90))

    print("uniform:    mean(curv_face) =", round(mu_u, 4), "p90 =", round(p90_u, 4))
    print("curvature:  mean(curv_face) =", round(mu_c, 4), "p90 =", round(p90_c, 4))
    print("points (uniform)   :", pts_u.shape)
    print("points (curvature) :", pts_c.shape)


if __name__ == "__main__":
    main()
