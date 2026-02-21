"""Surface remeshing via pymeshlab isotropic explicit remeshing."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pinneaple_geom.core.mesh import MeshData


def remesh_surface(
    mesh: MeshData,
    *,
    max_edge_length: Optional[float] = None,
    target_edge_length: Optional[float] = None,
) -> MeshData:
    """
    Surface remeshing (best-effort).

    MVP behavior:
      - If pymeshlab is available, use isotropic remeshing.
      - Otherwise, return mesh unchanged.

    This is intentionally conservative to avoid breaking pipelines.
    """
    try:
        import pymeshlab
    except Exception:
        # MVP fallback: no remesh
        return mesh.copy()

    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=mesh.vertices,
            face_matrix=mesh.faces,
        )
    )

    if target_edge_length is not None:
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=float(target_edge_length),
            iterations=5,
        )
    elif max_edge_length is not None:
        ms.meshing_isotropic_explicit_remeshing(
            targetlen=float(max_edge_length),
            iterations=5,
        )
    else:
        return mesh.copy()

    m = ms.current_mesh()
    v = m.vertex_matrix()
    f = m.face_matrix()

    return MeshData(vertices=v, faces=f, normals=None)
