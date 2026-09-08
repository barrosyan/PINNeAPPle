# Geometry & Domain

The Geometry & Domain layer turns a domain description into two things a
training loop needs: a way to test "is this point inside/on the boundary?"
and a way to *sample* interior and boundary collocation points from it.

## Where it lives

`pinneapple_design.geometry` (re-exported at `pinneapple_design`), organized
into:

- **SDF primitives** — 2D (`sdf2d_circle`, `sdf2d_rectangle`, `sdf2d_ellipse`,
  `sdf2d_annulus`, `sdf2d_capsule`, `sdf2d_triangle`,
  `sdf2d_convex_polygon`) and 3D (`sdf3d_sphere`, `sdf3d_box`,
  `sdf3d_cylinder`, `sdf3d_torus`, `sdf3d_capsule`), plus combinators
  (`sdf_union`, `sdf_intersection`, `sdf_difference`,
  `sdf_smooth_union/intersection/difference`, `sdf_translate`, `sdf_scale`,
  `sdf_rotate_2d`, `sdf_onion`, `sdf_repeat_2d`).
- **CSG** (`pinneapple_design.geometry.csg`) — an object-oriented layer over
  the same idea: `CSGRectangle`, `CSGCircle`, `CSGEllipse`, `CSGPolygon`,
  combined with `CSGUnion`, `CSGIntersection`, `CSGDifference`.
- **Physics domains** (`pinneapple_design.geometry.gen.domains` /
  `domains3d`) — `PhysicsDomain2D` / `PhysicsDomain3D` base classes that
  wrap an SDF with named boundary regions and give you
  `sample_interior(n)`, `sample_boundary_region(name, n)`, and
  `get_pinn_batch(...)` directly. Built-ins include channel, channel-with-
  obstacle, lid-driven-cavity, L-shape, annular, and multi-obstacle domains
  in 2D, with analogous 3D domains.
- **Mesh** — structured/SDF-based mesh generation, `MeshCollocator` for
  sampling collocation points from a 3D mesh, and STL/STEP import.

## Building and sampling a domain

```python
from pinneapple_design import build_domain, sample_domain

domain = build_domain("lid_driven_cavity_2d")   # dim=2 by default
x_int, x_bnd = sample_domain(domain, n_interior=4096, n_boundary=512)
```

`build_domain` dispatches to the registry lookups `get_domain` (2D) or
`get_domain_3d` (3D) depending on `dim`; `sample_domain` is a thin wrapper
that calls `domain.sample_interior(n_interior)` and
`domain.sample_boundary(n_boundary)`. Use `list_domains()` /
`list_domains_3d()` to see every registered built-in name.

## Custom geometry

Any `PhysicsDomain2D`/`3D` subclass only needs to implement `sdf(p)` and
`sample_boundary_region(region_name, n, seed)`; the base class handles
interior sampling via rejection against the SDF and assembles PINN-ready
batches for you. For irregular or imported shapes, build a `GeometryAsset`
via `build_geometry_asset`/`load_geometry_asset` and drive collocation with
`MeshCollocator` instead of an analytic SDF.

## Relationship to the rest of the pipeline

A domain produces coordinate batches with no notion of which fields live on
them or which PDE governs them — that pairing happens when
[ProblemDefinition](problem_definition.md)'s conditions are matched against
the domain's named boundary regions, and residuals are compiled in
[PINN / Physics](pinn.md).
