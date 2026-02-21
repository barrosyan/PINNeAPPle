"""Meshio mesh loading and UPD PhysicalSample packaging for VTK/VTU/MSH/XDMF."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import numpy as np

from pinneaple_geom.core.mesh import MeshData
from pinneaple_data.physical_sample import PhysicalSample


def load_meshio(
    path: Union[str, Path],
    *,
    require_triangles: bool = True,
) -> MeshData:
    """
    Load a mesh file using meshio and return a triangle MeshData.

    Works well for:
      .vtk, .vtu, .msh, .xdmf, .xmf, ...

    Notes:
      - Many CFD meshes are volumetric (tet/hex). MVP supports only surface triangles.
      - If the file contains triangles, we extract them.
      - If it has only quads, we can optionally triangulate in a later MVP.
    """
    import meshio

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    m = meshio.read(str(path))

    if require_triangles and "triangle" not in m.cells_dict:
        raise ValueError(
            f"Mesh '{path.name}' does not contain triangle cells. "
            "MVP expects surface triangle mesh."
        )

    pts = np.asarray(m.points, dtype=np.float64)
    if pts.shape[1] > 3:
        pts = pts[:, :3]

    faces = m.cells_dict.get("triangle")
    if faces is None:
        # if require_triangles is False, attempt best-effort fallback
        # (MVP: pick first cell block if it looks like triangles)
        for k, v in m.cells_dict.items():
            if v.ndim == 2 and v.shape[1] == 3:
                faces = v
                break

    if faces is None:
        raise ValueError(f"Could not find triangle faces in mesh '{path.name}'.")

    faces = np.asarray(faces, dtype=np.int64)

    return MeshData(vertices=pts, faces=faces, normals=None)


def save_meshio(
    mesh: MeshData,
    path: Union[str, Path],
    *,
    file_format: Optional[str] = None,
    point_data: Optional[Dict[str, Any]] = None,
    cell_data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save MeshData via meshio.

    By default, writes triangles only.

    file_format examples: "vtk", "vtu", "stl", "ply", ...
    """
    import meshio

    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    m = meshio.Mesh(
        points=np.asarray(mesh.vertices, dtype=np.float64),
        cells=[("triangle", np.asarray(mesh.faces, dtype=np.int64))],
        point_data=point_data,
        cell_data=cell_data,
    )
    meshio.write(str(path), m, file_format=file_format)

def meshio_to_upd(path: str) -> PhysicalSample:
    """
    Load mesh + point data from any meshio-supported format (VTK, Exodus, etc.)
    """
    import meshio  # optional

    m = meshio.read(path)

    vertices = torch.tensor(m.points, dtype=torch.float32)
    # pick first cell block as faces/cells
    faces = None
    if m.cells:
        # choose triangles if present
        tri = None
        for c in m.cells:
            if c.type in ("triangle", "tri"):
                tri = c.data
                break
        if tri is None:
            tri = m.cells[0].data
        faces = torch.tensor(tri, dtype=torch.long)

    fields: Dict[str, Any] = {"vertices": vertices}
    if faces is not None:
        fields["faces"] = faces

    # point_data → fields
    for k, v in (m.point_data or {}).items():
        fields[k] = torch.tensor(v, dtype=torch.float32)

    return PhysicalSample(
        fields=fields,
        coords={},
        meta={
            "upd": {"version": "0.1", "domain": "mesh", "source": "meshio"},
            "provenance": {"path": path},
            "units": {},
        },
    )
