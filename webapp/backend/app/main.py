from __future__ import annotations

import os
from typing import Any, Dict

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .pipelines.vertical_a_surrogate import run_vertical_a
from .pipelines.vertical_b_digital_twin import run_vertical_b
from .pipelines.vertical_c_arena import run_vertical_c
from .pipelines.vertical_d_physics_ts import run_vertical_d


class RunRequest(BaseModel):
    config: Dict[str, Any] = Field(default_factory=dict)


app = FastAPI(title="Pinneaple Web (v2)", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }


@app.post("/api/vertical-a/run")
def vertical_a(req: RunRequest) -> Dict[str, Any]:
    return run_vertical_a(req.config)


@app.post("/api/vertical-b/run")
def vertical_b(req: RunRequest) -> Dict[str, Any]:
    return run_vertical_b(req.config)


@app.post("/api/vertical-c/run")
def vertical_c(req: RunRequest) -> Dict[str, Any]:
    return run_vertical_c(req.config)


@app.post("/api/vertical-d/run")
def vertical_d(req: RunRequest) -> Dict[str, Any]:
    return run_vertical_d(req.config)


# Serve built frontend (Docker mode)
FRONTEND_DIST = os.getenv("FRONTEND_DIST", "")
if FRONTEND_DIST and os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="static")
