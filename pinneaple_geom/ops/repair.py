"""Best-effort mesh repair using trimesh (duplicates, degenerate, normals, holes)."""
from __future__ import annotations

import numpy as np

from pinneaple_geom.core.mesh import MeshData


def repair_mesh(
    mesh: MeshData,
    *,
    remove_duplicates: bool = True,
    remove_degenerate: bool = True,
    fix_normals: bool = True,
    fill_holes: bool = False,
) -> MeshData:
    """
    Best-effort mesh repair using trimesh.

    Returns a NEW MeshData (does not mutate input).
    """
    import trimesh

    tm = trimesh.Trimesh(
        vertices=mesh.vertices.copy(),
        faces=mesh.faces.copy(),
        process=False,
    )

    if remove_duplicates:
        tm.update_faces(tm.unique_faces())
        tm.remove_unreferenced_vertices()

    if remove_degenerate:
        tm.update_faces(tm.nondegenerate_faces())

    if fix_normals:
        try:
            tm.fix_normals()
        except Exception:
            pass

    if fill_holes:
        try:
            if not tm.is_watertight:
                tm.fill_holes()
        except Exception:
            pass

    normals = None
    try:
        normals = tm.face_normals.view(np.ndarray)
    except Exception:
        normals = None

    return MeshData(
        vertices=tm.vertices.view(np.ndarray),
        faces=tm.faces.view(np.ndarray),
        normals=normals,
    )
