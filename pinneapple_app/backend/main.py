"""FastAPI application entry point for PINNeAPPle App."""
from __future__ import annotations
import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse

from .routers import problems, models, experiments, admin

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
app.include_router(admin.router)

_START_TIME = time.time()
_REQUEST_COUNT = 0


@app.middleware("http")
async def _count_requests(request, call_next):
    global _REQUEST_COUNT
    _REQUEST_COUNT += 1
    return await call_next(request)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "pinneapple_app"}


@app.get("/api/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    """Prometheus text-exposition format — deliberately hand-rolled
    (no new dependency) rather than pulling in ``prometheus-client`` for
    two gauges. See ``pinneapple_app/deploy/monitoring/prometheus.yml``
    for the scrape config pointed at this endpoint."""
    uptime_s = time.time() - _START_TIME
    lines = [
        "# HELP pinneapple_app_uptime_seconds Seconds since process start.",
        "# TYPE pinneapple_app_uptime_seconds gauge",
        f"pinneapple_app_uptime_seconds {uptime_s:.3f}",
        "# HELP pinneapple_app_requests_total Total HTTP requests served since process start.",
        "# TYPE pinneapple_app_requests_total counter",
        f"pinneapple_app_requests_total {_REQUEST_COUNT}",
    ]
    return "\n".join(lines) + "\n"


@app.get("/api/info")
def info():
    """Return library capabilities summary."""
    try:
        from pinneapple_neural.architectures import ModelRegistry
        n_models = len(ModelRegistry.list())
        families = ModelRegistry.families()
    except Exception:
        n_models = 0
        families = []

    try:
        from pinneapple_physics.pde_environment import list_presets
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
