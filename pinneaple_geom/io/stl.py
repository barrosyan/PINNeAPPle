"""STL loading utilities and UPD-aligned PhysicalSample packaging."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from pinneaple_geom.core.mesh import MeshData
from pinneaple_geom.io.trimesh_bridge import TrimeshBridge
from pinneaple_data.physical_sample import PhysicalSample


def load_stl(
    path: Union[str, Path],
    *,
    repair: bool = True,
    compute_normals: bool = True,
) -> MeshData:
    """
    Convenience STL loader (via trimesh) returning MeshData.

    STL is extremely common for geometry-only inputs.
    """
    bridge = TrimeshBridge()

    # bridge.load already can repair; keep logic explicit
    tm = bridge._load_trimesh(path)
    if repair:
        tm = bridge._repair_trimesh(tm)

    return bridge.from_trimesh(tm, compute_normals=compute_normals)


def stl_to_upd(
    path: Union[str, Path],
    *,
    repair: bool = True,
    compute_normals: bool = True,
    units: str = "m",
) -> PhysicalSample:
    """
    Load an STL and package it as a UPD-aligned PhysicalSample.

    Returns a PhysicalSample where:
      - state contains {"vertices": Tensor[V,3], "faces": Tensor[F,3]}
      - domain indicates a mesh sample
      - provenance tracks the source path + optional id
    """
    import trimesh

    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    # robust load: may return Trimesh or Scene depending on file
    obj = trimesh.load(str(path), force="mesh")

    if isinstance(obj, trimesh.Scene):
        if not obj.geometry:
            raise ValueError(f"STL scene has no geometry: {path}")
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))

    if not isinstance(obj, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not a trimesh.Trimesh: {type(obj).__name__}")

    bridge = TrimeshBridge()
    if repair:
        obj = bridge._repair_trimesh(obj)

    md = bridge.from_trimesh(obj, compute_normals=compute_normals)

    V = torch.as_tensor(md.vertices, dtype=torch.float32)
    F = torch.as_tensor(md.faces, dtype=torch.long)

    # stable-ish id
    geom_id = None
    try:
        geom_id = obj.md5()
    except Exception:
        geom_id = None

    return PhysicalSample(
        state={"vertices": V, "faces": F},
        geometry=None,  # optional: later you can wrap md into GeometryAsset if you want
        domain={"type": "mesh"},
        provenance={
            "source": "stl",
            "path": str(path),
            "geometry_id": geom_id,
        },
        schema={
            "units": {"vertices": units},
        },
        extras={
            "meshdata": md,  # handy for downstream ops without reconversion
        },
    )
