"""Experiments API router — launch, monitor, and retrieve experiments."""
from __future__ import annotations
import asyncio
import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

router = APIRouter(prefix="/api/experiments", tags=["experiments"])

# ── In-memory experiment store (replace with DB in production) ────────────
_EXPERIMENTS: Dict[str, Any] = {}
_RESULTS:     Dict[str, Any] = {}
_PROGRESS:    Dict[str, list] = {}


class CollocationRequest(BaseModel):
    strategy: str = "lhs"
    n_interior: int = 4096
    n_boundary: int = 512
    n_initial: int = 256
    seed: int = 42
    use_geometry: bool = False
    domain_name: Optional[str] = None


class DataRequest(BaseModel):
    solver_key: Optional[str] = None    # None = auto-select from problem registry
    n_snapshots: int = 5
    grid_resolution: int = 32
    t_end: float = 1.0
    use_solver: bool = True


class ModelRequest(BaseModel):
    name: str
    extra_kwargs: Dict[str, Any] = {}


class ExperimentRequest(BaseModel):
    # Problem
    problem_name: str                    # preset name OR "__custom__"
    custom_problem: Optional[Dict[str, Any]] = None  # required when problem_name="__custom__"
    # Data
    collocation: CollocationRequest = CollocationRequest()
    data: DataRequest = DataRequest()
    # Models
    models: List[ModelRequest]
    metrics: List[str] = ["l2_relative", "mse", "pde_residual", "bc_residual",
                          "train_time_s", "n_params"]
    # Training
    epochs: int = 2000
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 42


class ExperimentStatusResponse(BaseModel):
    experiment_id: str
    status: str      # "queued" | "running" | "done" | "failed"
    progress: float  # 0–100
    message: str = ""


@router.post("/launch", response_model=ExperimentStatusResponse)
async def launch_experiment(req: ExperimentRequest, background_tasks: BackgroundTasks):
    """Launch a new experiment asynchronously. Returns experiment_id immediately."""
    from ..core.experiment import ExperimentConfig, ModelRunConfig, ExperimentRunner
    from ..core.collocation import CollocationConfig
    from ..core.data_pipeline import DataConfig, run_data_pipeline
    from ..core.problem import load_preset, define_custom, EquationSpec, BoundaryConditionSpec

    exp_id = _new_id()
    _EXPERIMENTS[exp_id] = {"status": "queued", "progress": 0.0, "request": req.model_dump()}
    _PROGRESS[exp_id] = []

    # Build objects now so validation errors surface synchronously
    if req.problem_name == "__custom__" and req.custom_problem:
        cp = req.custom_problem
        problem = define_custom(
            name=cp.get("name", "custom"),
            equations=[EquationSpec(expression=e) for e in cp.get("equations", [])],
            bcs=[BoundaryConditionSpec(kind=b.get("kind", "dirichlet"),
                                       location=b.get("location", ""),
                                       value=b.get("value", 0.0))
                 for b in cp.get("boundary_conditions", [])],
            domain_bounds={k: tuple(v) for k, v in cp.get("domain_bounds", {"x": [0,1]}).items()},
            dim=cp.get("dim", 2),
            pde_family=cp.get("pde_family", "generic"),
            is_time_dependent=cp.get("is_time_dependent", False),
        )
    else:
        try:
            problem = load_preset(req.problem_name)
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))

    # Auto-select solver if not specified
    solver_key = req.data.solver_key
    if solver_key is None:
        from ..core.problem_registry import recommended_solver
        solver_key = recommended_solver(req.problem_name) if req.problem_name != "__custom__" else "fdm_2d_generic"

    col_cfg = CollocationConfig(
        strategy=req.collocation.strategy,
        n_interior=req.collocation.n_interior,
        n_boundary=req.collocation.n_boundary,
        n_initial=req.collocation.n_initial,
        seed=req.collocation.seed,
        use_geometry=req.collocation.use_geometry,
        domain_name=req.collocation.domain_name,
    )

    data_cfg = DataConfig(
        solver_key=solver_key,
        n_snapshots=req.data.n_snapshots,
        grid_resolution=req.data.grid_resolution,
        t_end=req.data.t_end,
        use_solver=req.data.use_solver,
    )

    exp_cfg = ExperimentConfig(
        experiment_id=exp_id,
        problem_name=req.problem_name,
        models=[ModelRunConfig(name=m.name, extra_kwargs=m.extra_kwargs) for m in req.models],
        metrics=req.metrics,
        epochs=req.epochs,
        lr=req.lr,
        device=req.device,
        seed=req.seed,
    )

    background_tasks.add_task(
        _run_experiment_bg, exp_id, problem, col_cfg, data_cfg, exp_cfg
    )

    return ExperimentStatusResponse(
        experiment_id=exp_id,
        status="queued",
        progress=0.0,
        message="Experiment queued.",
    )


@router.get("/{experiment_id}/status", response_model=ExperimentStatusResponse)
def get_status(experiment_id: str):
    if experiment_id not in _EXPERIMENTS:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    exp = _EXPERIMENTS[experiment_id]
    return ExperimentStatusResponse(
        experiment_id=experiment_id,
        status=exp["status"],
        progress=exp.get("progress", 0.0),
        message=exp.get("message", ""),
    )


@router.get("/{experiment_id}/results")
def get_results(experiment_id: str):
    if experiment_id not in _RESULTS:
        status = _EXPERIMENTS.get(experiment_id, {}).get("status", "unknown")
        raise HTTPException(status_code=404,
                            detail=f"Results not ready yet (status={status}).")
    return _RESULTS[experiment_id]


@router.get("")
def list_experiments():
    return [
        {"experiment_id": eid, **{k: v for k, v in meta.items() if k != "request"}}
        for eid, meta in _EXPERIMENTS.items()
    ]


@router.websocket("/{experiment_id}/ws")
async def experiment_ws(websocket: WebSocket, experiment_id: str):
    """WebSocket endpoint — streams training progress events as JSON."""
    await websocket.accept()
    last_sent = 0
    try:
        while True:
            events = _PROGRESS.get(experiment_id, [])
            if len(events) > last_sent:
                for ev in events[last_sent:]:
                    await websocket.send_json(ev)
                last_sent = len(events)

            status = _EXPERIMENTS.get(experiment_id, {}).get("status", "unknown")
            if status in ("done", "failed"):
                await websocket.send_json({"type": "done", "status": status})
                break

            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass


# ── Background task ────────────────────────────────────────────────────────

async def _run_experiment_bg(exp_id, problem, col_cfg, data_cfg, exp_cfg):
    from ..core.data_pipeline import run_data_pipeline
    from ..core.experiment import ExperimentRunner
    from ..core.benchmark import build_benchmark_payload

    try:
        _EXPERIMENTS[exp_id]["status"] = "running"

        # Data generation
        _push(exp_id, {"type": "progress", "stage": "data_generation", "progress": 5})
        data = run_data_pipeline(problem, col_cfg, data_cfg, device=exp_cfg.device, verbose=False)

        _push(exp_id, {"type": "progress", "stage": "training", "progress": 10})

        n_models = len(exp_cfg.models)
        epoch_unit = 80.0 / max(n_models, 1)

        def progress_cb(ev):
            # ExperimentRunner sends three distinct event shapes through
            # this one callback: plain per-epoch training events (no
            # "type" key, carry "epoch"/"total_epochs"), and the
            # auto-fix advisor loop's "advisor"/"retrain" events (no
            # "epoch"/"total_epochs" at all -- see
            # core/experiment.py's three progress_cb(...) call sites).
            # Unconditionally indexing ev["epoch"] here raised a bare
            # KeyError('epoch') and silently failed the WHOLE experiment
            # the instant an advisor/retrain event fired -- found via
            # tests/test_app_backend.py's TestExperiments failing.
            if ev.get("type") in ("advisor", "retrain"):
                _push(exp_id, {**ev, "overall_progress": _EXPERIMENTS[exp_id]["progress"]})
                return
            frac = ev["epoch"] / max(ev["total_epochs"], 1)
            model_idx = [m.name for m in exp_cfg.models].index(ev["model"])
            prog = 10 + model_idx * epoch_unit + frac * epoch_unit
            _EXPERIMENTS[exp_id]["progress"] = round(prog, 1)
            _push(exp_id, {"type": "training", **ev, "overall_progress": round(prog, 1)})

        runner = ExperimentRunner(exp_cfg, data, problem)
        result = await runner.run(progress_cb=progress_cb)

        _push(exp_id, {"type": "progress", "stage": "benchmark", "progress": 95})
        payload = build_benchmark_payload(result)
        _RESULTS[exp_id] = payload

        _EXPERIMENTS[exp_id].update({"status": "done", "progress": 100.0})
        _push(exp_id, {"type": "done", "status": "done"})

    except Exception as e:
        _EXPERIMENTS[exp_id].update({"status": "failed", "message": str(e)})
        _push(exp_id, {"type": "error", "message": str(e)})


def _push(exp_id: str, ev: dict):
    _PROGRESS.setdefault(exp_id, []).append(ev)


def _new_id() -> str:
    import uuid
    return str(uuid.uuid4())[:8]
