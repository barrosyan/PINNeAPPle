"""Parametric primitive + boolean ops + STL roundtrip.

What this shows
--------------
- Procedural geometry generation via `pinneaple_geom.gen.primitives`.
- Optional boolean cut/union/intersection (with robust fallback if no boolean engine is installed).
- Packaging into a `GeometryAsset` for consistent downstream use.
- Exporting to STL and loading back.

Run
---
python examples/pinneaple_geom/03_parametric_boolean_and_export_stl.py

Notes
-----
Trimesh booleans require a boolean backend (e.g. manifold3d / blender / cork).
If none is available, Pinneaple falls back to concatenation (not watertight).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pinneaple_geom.core.registry import build_geometry_asset
from pinneaple_geom.core.geometry import GeometrySpec


def _to_trimesh(asset):
    import trimesh

    md = asset.mesh
    return trimesh.Trimesh(vertices=md.vertices, faces=md.faces, process=False)


def main() -> None:
    out_dir = Path("examples/pinneaple_geom/_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # A box with a cylindrical hole (boolean cut).
    # If you have a real boolean engine installed, you get a watertight result.
    # Otherwise you'll still get a mesh (concatenated), good enough to demo the pipeline.
    spec = GeometrySpec(
        kind="primitive",
        name="box",
        params={
            "extents": (2.0, 1.0, 0.6),
            "boolean": {
                "op": "cut",
                "other": {
                    "name": "cylinder",
                    "radius": 0.25,
                    "height": 1.2,
                    "rotate": (0.0, np.pi / 2, 0.0),
                },
            },
        },
    )

    asset = build_geometry_asset(spec)
    print("asset bbox min/max:", asset.bounds[0], asset.bounds[1])
    print("faces/verts:", asset.mesh.n_faces, asset.mesh.n_vertices)

    # Export to STL
    stl_path = out_dir / "box_with_hole.stl"
    tm = _to_trimesh(asset)
    tm.export(stl_path)
    print("wrote:", stl_path)

    # Load back via registry (file kind)
    asset2 = build_geometry_asset({"kind": "file", "path": str(stl_path)})
    print("roundtrip faces/verts:", asset2.mesh.n_faces, asset2.mesh.n_vertices)


if __name__ == "__main__":
    main()
