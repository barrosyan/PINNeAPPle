"""Face/vertex normals, areas, and mesh feature computation."""
from __future__ import annotations

from typing import Optional

import numpy as np

from pinneaple_geom.core.mesh import MeshData


def compute_face_normals(mesh: MeshData) -> np.ndarray:
    """
    Compute face normals (M,3).
    """
    v = mesh.vertices
    f = mesh.faces
    p0 = v[f[:, 0]]
    p1 = v[f[:, 1]]
    p2 = v[f[:, 2]]

    n = np.cross(p1 - p0, p2 - p0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return n / norm


def compute_vertex_normals(mesh: MeshData) -> np.ndarray:
    """
    Compute vertex normals by area-weighted face normals.
    """
    fn = compute_face_normals(mesh)
    v = mesh.vertices
    f = mesh.faces

    vn = np.zeros_like(v)
    for i in range(3):
        np.add.at(vn, f[:, i], fn)

    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return vn / norm


def compute_face_areas(mesh: MeshData) -> np.ndarray:
    """
    Compute triangle face areas (M,).
    """
    v = mesh.vertices
    f = mesh.faces
    p0 = v[f[:, 0]]
    p1 = v[f[:, 1]]
    p2 = v[f[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def compute_curvature_proxy(mesh: MeshData) -> np.ndarray:
    """
    Very lightweight curvature proxy per face.

    Defined as variation of normals across neighboring faces.
    MVP proxy: |n - mean(n_neighbors)|.

    Note:
      - This is NOT true curvature.
      - It's useful as a feature or sampling weight.
    """
    import trimesh

    tm = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        process=False,
    )

    fn = compute_face_normals(mesh)
    adj = tm.face_adjacency

    curv = np.zeros((mesh.faces.shape[0],), dtype=np.float64)
    for a, b in adj:
        d = np.linalg.norm(fn[a] - fn[b])
        curv[a] += d
        curv[b] += d

    # normalize
    if curv.max() > 0:
        curv /= curv.max()
    return curv
