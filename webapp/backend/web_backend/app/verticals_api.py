"""FastAPI routes for Verticals A-D.

These endpoints are MVP but functional, and are designed to be expanded.

Vertical A:
  - STL import -> boundary sampling
  - Design iteration optimization (random search)
  - Batch inference (multi-geometry)
  - Internal benchmark/ranking scaffold

Vertical B:
  - Streaming ingest
  - Online parameter updates + anomaly detection
  - Realtime monitoring endpoints

Vertical C:
  - Experiment bundle export + hash/version
  - Persistent leaderboard

Vertical D:
  - Hybrid physics+time-series forecast scaffold
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import zipfile
from typing import Any, Dict, List, Optional

import torch
from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from .vertical_a_operator import stl_to_boundary_points
from .streaming_bus import make_bus, StreamBusConfig
from .digital_twin import SimpleDigitalTwin, TwinConfig
from pinneaple_pdb import PinneaplePDB


router = APIRouter(prefix="/verticals", tags=["verticals"])


def _now() -> float:
    return float(time.time())


def _sha1(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


# --------------------------
# Vertical A
# --------------------------


class STLImportResponse(BaseModel):
    stl_hash: str
    n_points: int
    points_preview: List[List[float]]


@router.post("/a/import-stl", response_model=STLImportResponse)
async def import_stl(file: UploadFile = File(...), n_points: int = 2000):
    data = await file.read()
    pts = stl_to_boundary_points(data, n_points=int(n_points), normalize=True)
    h = _sha1(data)[:16]
    # store raw STL for later batch inference / optimization
    os.makedirs("./uploads/stl", exist_ok=True)
    with open(f"./uploads/stl/{h}.stl", "wb") as f:
        f.write(data)
    preview = pts[: min(10, pts.shape[0])].cpu().numpy().tolist()
    return STLImportResponse(stl_hash=h, n_points=int(pts.shape[0]), points_preview=preview)


class DesignIterRequest(BaseModel):
    stl_hash: str
    objective: str = "min_residual"
    iters: int = 20
    seed: int = 0


@router.post("/a/design-iterate")
async def design_iterate(req: DesignIterRequest):
    """Random-search design iteration loop.

    MVP objective is a proxy: minimize variance of boundary curvature-like signal.
    Replace with real surrogate objective (drag, max temp, etc.).
    """
    path = f"./uploads/stl/{req.stl_hash}.stl"
    if not os.path.exists(path):
        return {"ok": False, "error": "unknown stl_hash"}
    data = open(path, "rb").read()

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(req.seed))

    best = None
    hist = []
    for k in range(int(req.iters)):
        # simple geometry parameterization: scale + translation
        scale = float(torch.rand((), generator=rng).item() * 0.7 + 0.65)
        tx = float(torch.randn((), generator=rng).item() * 0.05)
        ty = float(torch.randn((), generator=rng).item() * 0.05)

        pts = stl_to_boundary_points(data, n_points=1500, normalize=True, seed=int(req.seed) + k)
        pts2 = pts.clone()
        pts2[:, 0] = pts2[:, 0] * scale + tx
        pts2[:, 1] = pts2[:, 1] * scale + ty

        # proxy objective
        r = torch.linalg.norm(pts2[:, :2], dim=-1)
        score = float(r.std().item())
        cand = {"iter": k, "scale": scale, "tx": tx, "ty": ty, "score": score}
        hist.append(cand)
        if best is None or score < best["score"]:
            best = cand

    return {"ok": True, "best": best, "history": hist}


class BatchInferenceRequest(BaseModel):
    stl_hashes: List[str]
    n_points: int = 2000


@router.post("/a/batch-boundary")
async def batch_boundary(req: BatchInferenceRequest):
    outs = []
    for h in req.stl_hashes:
        path = f"./uploads/stl/{h}.stl"
        if not os.path.exists(path):
            outs.append({"stl_hash": h, "ok": False, "error": "missing"})
            continue
        data = open(path, "rb").read()
        pts = stl_to_boundary_points(data, n_points=int(req.n_points), normalize=True)
        outs.append({"stl_hash": h, "ok": True, "n": int(pts.shape[0]), "bbox": pts.min(dim=0).values.tolist() + pts.max(dim=0).values.tolist()})
    return {"ok": True, "results": outs}


# --------------------------
# Vertical B (Streaming + Digital Twin)
# --------------------------


class SensorIngestRequest(BaseModel):
    asset_id: str
    u: float
    y: float
    ts: Optional[float] = None


@router.post("/b/ingest")
async def ingest_sensor(req: SensorIngestRequest):
    bus = make_bus(StreamBusConfig())
    msg_id = bus.publish(req.asset_id, {"u": req.u, "y": req.y}, ts=req.ts)

    twin = SimpleDigitalTwin(TwinConfig(asset_id=req.asset_id))
    upd = twin.update_online(u=req.u, y=req.y, ts=req.ts)
    return {"ok": True, "msg_id": msg_id, "update": upd}


@router.get("/b/monitor/{asset_id}")
async def monitor(asset_id: str, limit: int = 200):
    bus = make_bus(StreamBusConfig())
    pdb = PinneaplePDB()
    series = bus.read_latest(asset_id, limit=int(limit))
    params = pdb.list_params(asset_id)
    anomalies = pdb.recent_anomalies(asset_id, limit=50)
    return {"ok": True, "asset_id": asset_id, "params": params, "series": series, "anomalies": anomalies}


# --------------------------
# Vertical C (Arena: experiments + leaderboard)
# --------------------------


class ExperimentRecordRequest(BaseModel):
    task: str
    model: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]


@router.post("/c/record")
async def record_experiment(req: ExperimentRecordRequest):
    pdb = PinneaplePDB()
    exp_id = hashlib.sha256(json.dumps({"task": req.task, "model": req.model, "config": req.config, "metrics": req.metrics}, sort_keys=True).encode()).hexdigest()[:16]

    # score heuristic: prefer metric named 'loss' if present
    score = float(req.metrics.get("loss", req.metrics.get("mse", req.metrics.get("score", 0.0))))
    pdb.upsert_experiment(exp_id, {"task": req.task, "model": req.model, **req.config}, req.metrics)
    pdb.upsert_leaderboard(req.task, req.model, score, {"exp_id": exp_id, **req.metrics}, ts=_now())
    return {"ok": True, "exp_id": exp_id, "score": score}


@router.get("/c/leaderboard/{task}")
async def leaderboard(task: str, limit: int = 20, higher_is_better: bool = False):
    pdb = PinneaplePDB()
    top = pdb.top_leaderboard(task, limit=int(limit), higher_is_better=bool(higher_is_better))
    return {"ok": True, "task": task, "rows": top}


@router.get("/c/export/{exp_id}")
async def export_experiment(exp_id: str):
    """Export a reproducible bundle as a zip blob (base64 is avoided; return as bytes)."""
    pdb = PinneaplePDB()
    exp = pdb.get_experiment(exp_id)
    if exp is None:
        return {"ok": False, "error": "unknown exp_id"}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("experiment.json", json.dumps(exp, indent=2, sort_keys=True))
        z.writestr("README.txt", "This is a Pinneaple experiment bundle.\n\nContents:\n- experiment.json (config + metrics)\n")
    b = buf.getvalue()
    return {"ok": True, "exp_id": exp_id, "zip_bytes": list(b[:4096]), "zip_len": len(b), "sha256": hashlib.sha256(b).hexdigest()}


# --------------------------
# Vertical D (Physics + Time Series Fusion)
# --------------------------


class FusionForecastRequest(BaseModel):
    asset_id: str
    u_hist: List[float]
    y_hist: List[float]
    horizon: int = 16
    physics_lambda: float = 1.0


@router.post("/d/fusion-forecast")
async def fusion_forecast(req: FusionForecastRequest):
    """Very small hybrid forecast.

    Uses a simple autoregressive baseline + physics constraint from the twin model.
    """
    pdb = PinneaplePDB()
    twin = SimpleDigitalTwin(TwinConfig(asset_id=req.asset_id), pdb=pdb)

    u = torch.tensor(req.u_hist, dtype=torch.float32)
    y = torch.tensor(req.y_hist, dtype=torch.float32)
    H = int(req.horizon)

    # baseline: last value persistence
    y_hat = torch.zeros((H,), dtype=torch.float32)
    y_hat[:] = y[-1]

    # physics correction: encourage y_hat ~ twin.predict(u_future)
    # Use last u as future control (MVP)
    u_future = float(u[-1].item())
    y_phys = float(twin.predict(u_future))
    y_hat = (y_hat + float(req.physics_lambda) * y_phys) / (1.0 + float(req.physics_lambda))

    # physical violation metric: |y_hat - y_phys|
    violation = float(torch.mean(torch.abs(y_hat - y_phys)).item())

    return {"ok": True, "forecast": y_hat.tolist(), "physics_reference": y_phys, "physics_violation": violation, "params": pdb.list_params(req.asset_id)}
