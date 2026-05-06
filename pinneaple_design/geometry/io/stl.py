"""STL loading utilities returning MeshData.

For packaging into a UPD PhysicalSample use
``pinneaple_data.adapters.geom_adapter.stl_to_upd``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from pinneaple_design.geometry.core.mesh import MeshData
from pinneaple_design.geometry.io.trimesh_bridge import TrimeshBridge


def load_stl(
    path: Union[str, Path],
    *,
    repair: bool = True,
    compute_normals: bool = True,
) -> MeshData:
    """Convenience STL loader (via trimesh) returning MeshData."""
    bridge = TrimeshBridge()
    tm = bridge._load_trimesh(path)
    if repair:
        tm = bridge._repair_trimesh(tm)
    return bridge.from_trimesh(tm, compute_normals=compute_normals)
