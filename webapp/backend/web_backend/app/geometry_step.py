"""STEP/mesh/point-cloud endpoints for the webapp.

This is an MVP wiring layer that exposes `pinneaple_geom` capabilities to the
FastAPI backend.

Capabilities
  - Upload STEP (.step/.stp)
  - Mesh via gmsh (through pinneaple_geom)
  - Sample a point cloud from the mesh

Notes
  - Meshing requires `gmsh` + `meshio`.
  - If these deps aren't installed, the endpoint returns a clear error.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel


router = APIRouter(prefix="/api/geom", tags=["geometry"])


DATA_DIR = Path(os.environ.get("PINNEAPLE_GEOM_DATA", "/tmp/pinneaple_geom"))
UPLOAD_DIR = DATA_DIR / "uploads"
MESH_DIR = DATA_DIR / "meshes"
PC_DIR = DATA_DIR / "pointclouds"
for d in (UPLOAD_DIR, MESH_DIR, PC_DIR):
    d.mkdir(parents=True, exist_ok=True)


class UploadResp(BaseModel):
    geom_id: str
    filename: str


class MeshReq(BaseModel):
    geom_id: str
    mesh_size: float = 0.02
    dim: int = 3


class MeshResp(BaseModel):
    mesh_id: str
    geom_id: str
    path: str
    n_points: int
    n_cells: int
    cell_types: list[str]


class PointCloudReq(BaseModel):
    mesh_id: str
    n: int = 8192
    seed: int = 0


class PointCloudResp(BaseModel):
    pc_id: str
    mesh_id: str
    n: int
    path: str
    bounds_min: list[float]
    bounds_max: list[float]


def _safe_suffix(filename: Optional[str]) -> str:
    name = filename or "model.step"
    suf = Path(name).suffix.lower()
    if suf not in (".step", ".stp"):
        raise HTTPException(status_code=400, detail="Only .step/.stp files are supported")
    return suf


@router.post("/step/upload", response_model=UploadResp)
async def upload_step(file: UploadFile = File(...)):
    suf = _safe_suffix(file.filename)
    geom_id = uuid.uuid4().hex
    out = UPLOAD_DIR / f"{geom_id}{suf}"
    out.write_bytes(await file.read())
    return UploadResp(geom_id=geom_id, filename=out.name)


@router.post("/step/mesh", response_model=MeshResp)
def mesh_step(req: MeshReq):
    src = None
    for suf in (".step", ".stp"):
        cand = UPLOAD_DIR / f"{req.geom_id}{suf}"
        if cand.exists():
            src = cand
            break
    if src is None:
        raise HTTPException(status_code=404, detail=f"geom_id not found: {req.geom_id}")

    try:
        from pinneaple_geom.io.step import step_to_mesh
        from pinneaple_geom.io.meshio_bridge import save_meshio
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "STEP meshing requires pinneaple_geom + gmsh + meshio. "
                f"Import error: {type(e).__name__}: {e}"
            ),
        )

    mesh_id = uuid.uuid4().hex
    msh_path = MESH_DIR / f"{mesh_id}.msh"
    # Let pinneaple_geom do the STEP→Gmsh pipeline, then re-save under our mesh_id.
    meshdata = step_to_mesh(str(src), mesh_size=float(req.mesh_size), dim=int(req.dim), cache_dir=None)
    save_meshio(meshdata, str(msh_path))

    n_points = int(meshdata.points.shape[0])
    n_cells = int(sum(c.data.shape[0] for c in meshdata.cells))
    cell_types = [c.type for c in meshdata.cells]

    return MeshResp(
        mesh_id=mesh_id,
        geom_id=req.geom_id,
        path=str(msh_path),
        n_points=n_points,
        n_cells=n_cells,
        cell_types=cell_types,
    )


@router.get("/step/mesh/{mesh_id}.msh")
def download_mesh(mesh_id: str):
    path = MESH_DIR / f"{mesh_id}.msh"
    if not path.exists():
        raise HTTPException(status_code=404, detail="mesh not found")
    return FileResponse(str(path), media_type="application/octet-stream", filename=path.name)


@router.post("/step/pointcloud", response_model=PointCloudResp)
def mesh_to_pointcloud(req: PointCloudReq):
    msh_path = MESH_DIR / f"{req.mesh_id}.msh"
    if not msh_path.exists():
        raise HTTPException(status_code=404, detail="mesh not found")

    try:
        import numpy as np
        from pinneaple_geom.io.meshio_bridge import load_meshio
        from pinneaple_geom.ops.pointcloud import sample_pointcloud_from_mesh
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Point-cloud deps missing: {type(e).__name__}: {e}")

    meshdata = load_meshio(str(msh_path))
    pc = sample_pointcloud_from_mesh(meshdata.points, meshdata.cells, n=int(req.n), seed=int(req.seed))

    pc_id = uuid.uuid4().hex
    out = PC_DIR / f"{pc_id}.npz"
    np.savez_compressed(out, points=pc)

    bmin = pc.min(axis=0).tolist()
    bmax = pc.max(axis=0).tolist()

    return PointCloudResp(
        pc_id=pc_id,
        mesh_id=req.mesh_id,
        n=int(req.n),
        path=str(out),
        bounds_min=bmin,
        bounds_max=bmax,
    )


@router.get("/step/pointcloud/{pc_id}.npz")
def download_pointcloud(pc_id: str):
    path = PC_DIR / f"{pc_id}.npz"
    if not path.exists():
        raise HTTPException(status_code=404, detail="pointcloud not found")
    return FileResponse(str(path), media_type="application/octet-stream", filename=path.name)
