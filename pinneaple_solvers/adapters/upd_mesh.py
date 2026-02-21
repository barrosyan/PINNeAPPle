"""UPD mesh adapter for solver inputs (vertices, faces, topology)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def _to_torch(x, device=None, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.tensor(x)
    if device is not None:
        t = t.to(device=device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


def _extract_mesh_arrays(mesh: Any) -> Tuple[Any, Any]:
    """
    Duck-typed mesh extraction.

    Accepts:
      - trimesh.Trimesh: mesh.vertices, mesh.faces
      - meshio.Mesh: mesh.points, mesh.cells_dict (tri/tetra/quad/hex)
      - Pinneaple mesh wrapper with .vertices/.faces or .points/.cells_dict
    """
    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        return mesh.vertices, mesh.faces
    if hasattr(mesh, "points") and (hasattr(mesh, "cells_dict") or hasattr(mesh, "cells")):
        points = mesh.points
        if hasattr(mesh, "cells_dict"):
            cd = mesh.cells_dict
        else:
            # meshio Mesh has .cells as list of (cell_type, data)
            cd = {ct: data for (ct, data) in mesh.cells}
        # choose first supported surface cell type
        for k in ("triangle", "quad"):
            if k in cd:
                return points, cd[k]
        # fallback
        k0 = next(iter(cd.keys()))
        return points, cd[k0]
    raise TypeError("Unsupported mesh type. Expected trimesh.Trimesh, meshio.Mesh, or compatible wrapper.")


def mesh_to_fvm_topology(
    mesh: Any,
    *,
    device=None,
    dtype=None,
) -> Dict[str, torch.Tensor]:
    """
    Build a minimal FVM topology from a surface mesh (triangles/quads).

    MVP assumptions:
      - Treat each face as a “cell” (good for surface PDEs / transport on surfaces).
      - Build adjacency by shared edges between faces.
      - Areas per face computed; "volumes" is area (surface control-volume).

    Output keys:
      - faces: (F,2) edge-adjacent face indices (left,right), right=-1 for boundary edge
      - areas: (E,1) edge “area weight” (uses edge length as a proxy)
      - normals: (E,3) edge normals proxy (not exact; placeholder)
      - volumes: (F,1) face areas
    """
    V, F = _extract_mesh_arrays(mesh)
    Vt = _to_torch(V, device=device, dtype=dtype)
    Ft = _to_torch(F, device=device, dtype=torch.long)

    # face areas
    if Ft.shape[1] == 3:
        a = Vt[Ft[:, 0]]
        b = Vt[Ft[:, 1]]
        c = Vt[Ft[:, 2]]
        face_area = 0.5 * torch.linalg.norm(torch.cross(b - a, c - a), dim=-1)
    else:
        # quad: split into 2 tris
        a = Vt[Ft[:, 0]]
        b = Vt[Ft[:, 1]]
        c = Vt[Ft[:, 2]]
        d = Vt[Ft[:, 3]]
        area1 = 0.5 * torch.linalg.norm(torch.cross(b - a, c - a), dim=-1)
        area2 = 0.5 * torch.linalg.norm(torch.cross(d - a, c - a), dim=-1)
        face_area = area1 + area2
    volumes = face_area[:, None].clamp_min(1e-12)

    # build edges -> faces map
    # edges as sorted vertex pairs
    def face_edges(fi: torch.Tensor):
        idx = fi.tolist()
        eds = []
        for i in range(len(idx)):
            u = idx[i]
            v = idx[(i + 1) % len(idx)]
            if u < v:
                eds.append((u, v))
            else:
                eds.append((v, u))
        return eds

    edge_to_faces: Dict[tuple, list] = {}
    for fidx in range(Ft.shape[0]):
        eds = face_edges(Ft[fidx])
        for e in eds:
            edge_to_faces.setdefault(e, []).append(fidx)

    # Each unique edge becomes a "face" in FVM topology between two adjacent “cells” (faces)
    E = len(edge_to_faces)
    faces = torch.full((E, 2), -1, device=Vt.device, dtype=torch.long)
    areas = torch.zeros((E, 1), device=Vt.device, dtype=Vt.dtype)
    normals = torch.zeros((E, 3), device=Vt.device, dtype=Vt.dtype)

    # quick per-face normal (for proxy edge normal)
    if Ft.shape[1] >= 3:
        a = Vt[Ft[:, 0]]
        b = Vt[Ft[:, 1]]
        c = Vt[Ft[:, 2]]
        fn = torch.cross(b - a, c - a)
        fn = fn / (torch.linalg.norm(fn, dim=-1, keepdim=True).clamp_min(1e-12))
    else:
        fn = torch.zeros((Ft.shape[0], 3), device=Vt.device, dtype=Vt.dtype)

    for i, (e, flist) in enumerate(edge_to_faces.items()):
        u, v = e
        pu = Vt[u]
        pv = Vt[v]
        edge_vec = pv - pu
        edge_len = torch.linalg.norm(edge_vec).clamp_min(1e-12)
        areas[i, 0] = edge_len  # proxy weight

        left = flist[0]
        right = flist[1] if len(flist) > 1 else -1
        faces[i, 0] = left
        faces[i, 1] = right

        # proxy normal: perpendicular to edge within face plane ~ cross(face_normal, edge_dir)
        edge_dir = edge_vec / edge_len
        n = torch.cross(fn[left], edge_dir)
        normals[i] = n

    return {"faces": faces, "areas": areas, "normals": normals, "volumes": volumes}


def mesh_to_fem_assembly_inputs(
    mesh: Any,
    *,
    device=None,
    dtype=None,
) -> Dict[str, torch.Tensor]:
    """
    Minimal FEM assembly inputs (MVP):
      - nodes (N,3)
      - elements (E,k) connectivity
    The actual K,f assembly is handled by user-provided assemble_fn (in FEMSolver).
    """
    V, F = _extract_mesh_arrays(mesh)
    nodes = _to_torch(V, device=device, dtype=dtype)
    elems = _to_torch(F, device=device, dtype=torch.long)
    return {"nodes": nodes, "elements": elems}
