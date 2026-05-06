"""FastAPI application entry point for PINNeAPPle App."""
from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routers import problems, models, experiments

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PINNeAPPle App",
    description="Physics AI experimentation laboratory — benchmark PINN models on physics problems.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routers ───────────────────────────────────────────────────────────
app.include_router(problems.router)
app.include_router(models.router)
app.include_router(experiments.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "pinneaple_app"}


@app.get("/api/info")
def info():
    """Return library capabilities summary."""
    try:
        from pinneaple_neural.architectures import ModelRegistry
        n_models = len(ModelRegistry.list())
        families = ModelRegistry.families()
    except Exception:
        n_models = 0
        families = []

    try:
        from pinneaple_physics.pde_environment import list_presets
        n_problems = len(list_presets())
    except Exception:
        n_problems = 0

    return {
        "n_models":   n_models,
        "n_problems": n_problems,
        "families":   families,
        "features": [
            "Preset and custom problem definition",
            "Auto solver selection for data generation",
            "Optional geometry + collocation strategy (LHS, Sobol, Halton, Grid, Adaptive)",
            "Multi-model parallel benchmark",
            "Real-time WebSocket training progress",
            "Benchmark report with charts and leaderboard",
        ],
    }


# ── Serve React frontend in production ────────────────────────────────────
FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")

if os.path.isdir(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")),
              name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_spa(full_path: str):
        index = os.path.join(FRONTEND_DIST, "index.html")
        return FileResponse(index)
