"""
Parametric variants MVP: generate CAD -> trimesh -> MeshData -> sample surface points.

Requires:
  - cadquery (OCC stack)
  - trimesh
"""

from __future__ import annotations

try:
    import cadquery as cq  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "\n".join(
            [
                "[ERROR] cadquery is not available in this environment.",
                "This example depends on the OpenCascade (OCC) stack.",
                "",
                "Typical install options:",
                "  - conda install -c conda-forge cadquery",
                "  - or follow: https://cadquery.readthedocs.io/ (installation docs)",
                "",
                f"Original import error: {e}",
            ]
        )
    )

from pinneaple_geom.gen.cadquery_gen import cadquery_to_trimesh
from pinneaple_geom.io.trimesh_bridge import TrimeshBridge
from pinneaple_geom.sample.points import sample_surface_points


def make_part(width: float, height: float, thickness: float):
    return cq.Workplane("XY").box(width, height, thickness)


def main():
    # Generate a family of boxes and sample points from their surfaces.
    variants = [
        {"width": 1.0, "height": 1.0, "thickness": 0.2},
        {"width": 1.2, "height": 0.8, "thickness": 0.25},
        {"width": 0.8, "height": 1.3, "thickness": 0.15},
    ]

    bridge = TrimeshBridge()

    for i, v in enumerate(variants):
        solid = make_part(**v)

        # CADQuery -> trimesh.Trimesh
        tm = cadquery_to_trimesh(solid)

        # trimesh.Trimesh -> MeshData (Pinneaple internal)
        mesh = bridge.from_trimesh(tm, compute_normals=True)

        # Sample surface points + normals
        pts, nrm = sample_surface_points(mesh, n=5000, seed=42 + i)

        print(
            f"variant[{i}] params={v} | points={pts.shape} normals={nrm.shape} | bbox={mesh.bbox}"
        )


if __name__ == "__main__":
    main()