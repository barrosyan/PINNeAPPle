from __future__ import annotations

import hashlib, json, os, datetime as dt
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from .db import init_db, SessionLocal, Job, JobEvent, JobArtifact
from .settings import settings
from .storage import ensure_bucket, presign_get
from .tasks import run_job, run_job_gpu
from .verticals_api import router as verticals_router
from .geometry_sdf import router as geom_sdf_router
from .geometry_step import router as geom_step_router

app = FastAPI(title="Pinneaple WebApp API (v6)", version="0.6.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Additional APIs for Verticals A-D (STL import, streaming, leaderboard, fusion)
app.include_router(verticals_router)

# Geometry Lab APIs (SDF + STEP/mesh/point-cloud)
app.include_router(geom_sdf_router)
app.include_router(geom_step_router)

class StartJobRequest(BaseModel):
    vertical: str
    config: Dict[str, Any] = {}
    queue: Optional[str] = None  # cpu|gpu
    use_cache: bool = True

def _cache_key(vertical: str, config: Dict[str, Any]) -> str:
    payload = {"vertical": vertical, "config": config, "code_version": settings.pinneaple_code_version}
    s = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()

def _job_id() -> str:
    ts = dt.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    rnd = hashlib.sha1(os.urandom(16)).hexdigest()[:10]
    return f"job_{ts}_{rnd}"

@app.on_event("startup")
def _startup():
    init_db()
    ensure_bucket()

@app.get("/")
def root():
    return {"ok": True, "service": "pinneaple-webapp-api", "version": app.version}

@app.post("/api/jobs/start")
def start_job(req: StartJobRequest):
    vertical = req.vertical.strip()
    if not vertical:
        raise HTTPException(400, "vertical is required")

    cache_key = _cache_key(vertical, req.config or {})
    session = SessionLocal()
    try:
        if req.use_cache:
            cached = session.query(Job).filter(Job.cache_key == cache_key, Job.status == "completed").order_by(Job.created_at.desc()).first()
            if cached:
                return {"job_id": cached.id, "cached": True}

        job_id = _job_id()
        queue = (req.queue or req.config.get("queue") or "cpu").strip().lower()
        if queue not in ("cpu", "gpu"):
            queue = "cpu"

        job = Job(
            id=job_id,
            vertical=vertical,
            status="queued",
            queue=queue,
            cache_key=cache_key,
            use_cache=bool(req.use_cache),
            config=req.config or {},
        )
        session.add(job)
        session.commit()

        if queue == "gpu":
            run_job_gpu.apply_async(args=[job_id], queue="gpu")
        else:
            run_job.apply_async(args=[job_id], queue="cpu")

        session.add(JobEvent(job_id=job_id, level="INFO", message="Job queued."))
        session.commit()
        return {"job_id": job_id, "cached": False}
    finally:
        session.close()

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    session = SessionLocal()
    try:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return {
            "job_id": job.id,
            "vertical": job.vertical,
            "status": job.status,
            "queue": job.queue,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "error": job.error,
        }
    finally:
        session.close()

@app.get("/api/jobs/{job_id}/logs")
def get_logs(job_id: str, tail: int = Query(200, ge=1, le=5000)):
    session = SessionLocal()
    try:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(404, "job not found")
        q = session.query(JobEvent).filter(JobEvent.job_id == job_id).order_by(JobEvent.id.desc()).limit(int(tail)).all()
        events = [{"ts": e.created_at.isoformat(), "level": e.level, "message": e.message} for e in reversed(q)]
        return {"job_id": job_id, "events": events}
    finally:
        session.close()

@app.get("/api/jobs/recent")
def recent_jobs(limit: int = Query(30, ge=1, le=200)):
    session = SessionLocal()
    try:
        jobs = session.query(Job).order_by(Job.created_at.desc()).limit(int(limit)).all()
        return [{
            "job_id": j.id,
            "vertical": j.vertical,
            "status": j.status,
            "queue": j.queue,
            "created_at": j.created_at.isoformat(),
        } for j in jobs]
    finally:
        session.close()

@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    session = SessionLocal()
    try:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(404, "job not found")
        return {"job_id": job_id, "result": job.result}
    finally:
        session.close()

@app.get("/api/jobs/{job_id}/artifacts")
def list_artifacts(job_id: str):
    session = SessionLocal()
    try:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(404, "job not found")
        arts = session.query(JobArtifact).filter(JobArtifact.job_id == job_id).order_by(JobArtifact.id.asc()).all()
        items = []
        for a in arts:
            items.append({"kind": a.kind, "s3_key": a.s3_key, "url": presign_get(a.s3_key, expires_sec=3600)})
        return {"job_id": job_id, "artifacts": items}
    finally:
        session.close()

@app.get("/api/jobs/{job_id}/artifacts.zip")
def download_zip(job_id: str):
    session = SessionLocal()
    try:
        a = session.query(JobArtifact).filter(JobArtifact.job_id == job_id, JobArtifact.kind == "artifacts_zip").order_by(JobArtifact.id.desc()).first()
        if not a:
            raise HTTPException(404, "artifacts.zip not available")
        return RedirectResponse(presign_get(a.s3_key, expires_sec=3600))
    finally:
        session.close()

@app.get("/api/models")
def list_models(
    family: str | None = Query(default=None),
    input_kind: str | None = Query(default=None),
    supports_physics_loss: bool | None = Query(default=None),
):
    try:
        from pinneaple_models.register_all import register_all as register_all_models
        from pinneaple_models.registry import ModelRegistry
        register_all_models()

        items = []
        for name in ModelRegistry.list():
            s = ModelRegistry.spec(name)
            if family and s.family != family:
                continue
            if input_kind and s.input_kind != input_kind:
                continue
            if supports_physics_loss is not None and s.supports_physics_loss != supports_physics_loss:
                continue
            items.append({
                "name": s.name,
                "family": s.family,
                "description": s.description,
                "tags": s.tags,
                "input_kind": s.input_kind,
                "supports_physics_loss": s.supports_physics_loss,
                "expects": s.expects,
                "predicts": s.predicts,
            })
        return {"models": items}
    except Exception as e:
        return {"models": [], "warning": f"Model registry not available: {e}"}

def main():
    import uvicorn
    uvicorn.run("web_backend.app.main:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
