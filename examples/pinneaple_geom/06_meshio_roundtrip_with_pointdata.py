"""MeshIO roundtrip + point_data/cell_data (useful for supervision / hybrid losses).

What this shows
--------------
- Save a triangle surface mesh to a meshio-supported format (VTU/VTP/VTK/etc.).
- Attach point_data (e.g., distance-to-center or signed distance approximations).
- Load it back with `meshio_to_upd` to get a `PhysicalSample` with fields.

Run
---
python examples/pinneaple_geom/06_meshio_roundtrip_with_pointdata.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pinneaple_geom.gen.primitives import build_primitive
from pinneaple_geom.io.meshio_bridge import save_meshio, meshio_to_upd


def main() -> None:
    out_dir = Path("examples/pinneaple_geom/_out")
    out_dir.mkdir(parents=True, exist_ok=True)

    # A smooth mesh so point_data looks clean.
    mesh = build_primitive("sphere", radius=1.0, subdivisions=3)

    # Make up a point feature: distance to origin
    r = np.linalg.norm(mesh.vertices, axis=1).astype(np.float64)

    # Save to VTU (triangles only)
    path = out_dir / "sphere_with_pointdata.vtu"
    save_meshio(mesh, path, point_data={"radius": r})
    print("wrote:", path)

    # Load back as a PhysicalSample (UPD-aligned)
    sample = meshio_to_upd(str(path))
    print("sample fields:", sorted(sample.fields.keys()))
    print("vertices:", tuple(sample.fields["vertices"].shape))
    print("faces:", tuple(sample.fields["faces"].shape))
    print("radius feature:", tuple(sample.fields["radius"].shape), "min/max:", float(sample.fields["radius"].min()), float(sample.fields["radius"].max()))


if __name__ == "__main__":
    main()
