from __future__ import annotations

import os, json, traceback, datetime as dt, zipfile
from typing import Any, Dict

from .celery_app import celery_app
from .db import SessionLocal, Job, JobEvent, JobArtifact
from .settings import settings
from .storage import ensure_bucket, upload_file

from .vertical_a_operator import build_dataset_for_operator, train_operator_model, render_preview


def _emit(session, job_id: str, message: str, level: str = "INFO") -> None:
    session.add(JobEvent(job_id=job_id, level=level, message=message))
    session.commit()

def _job_dir(job_id: str) -> str:
    d = os.path.join(settings.jobs_local_dir, job_id)
    os.makedirs(d, exist_ok=True)
    return d

def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _zip_dir(src_dir: str, zip_path: str) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(src_dir):
            for fn in files:
                p = os.path.join(root, fn)
                rel = os.path.relpath(p, src_dir)
                z.write(p, rel)

def _finalize_artifacts(job_id: str, session) -> None:
    ensure_bucket()
    run_dir = _job_dir(job_id)

    zip_path = os.path.join(run_dir, "artifacts.zip")
    _zip_dir(run_dir, zip_path)

    s3_zip_key = f"jobs/{job_id}/artifacts.zip"
    upload_file(zip_path, s3_zip_key, content_type="application/zip")
    session.add(JobArtifact(job_id=job_id, kind="artifacts_zip", s3_key=s3_zip_key))
    session.commit()

    for fn, kind, ctype in [
        ("result.json", "result_json", "application/json"),
        ("preview.png", "preview_png", "image/png"),
        ("preview_ts.png", "preview_ts_png", "image/png"),
    ]:
        p = os.path.join(run_dir, fn)
        if os.path.exists(p):
            key = f"jobs/{job_id}/{fn}"
            upload_file(p, key, content_type=ctype)
            session.add(JobArtifact(job_id=job_id, kind=kind, s3_key=key))
            session.commit()

def _load_config_bundle(cfg: Dict[str, Any], session, job_id: str) -> Dict[str, Any]:
    config_name = cfg.get("config_name")
    if config_name:
        path = os.path.join("configs", "arena", f"{config_name}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    bundle = json.load(f)
                merged = dict(bundle)
                merged.update(cfg)
                _emit(session, job_id, f"Loaded config bundle: {path}")
                return merged
            except Exception as e:
                _emit(session, job_id, f"Failed to load config bundle: {e}", level="WARN")
    return cfg

def _run_vertical_a(job, session) -> Dict[str, Any]:
    from pinneaple_models.register_all import register_all as register_all_models
    from pinneaple_models.registry import ModelRegistry
    from pinneaple_models.adapters.operators import OperatorAdapter

    cfg = _load_config_bundle(job.config or {}, session, job.id)
    run_dir = _job_dir(job.id)
    _write_json(os.path.join(run_dir, "config_resolved.json"), cfg)

    register_all_models()

    model_name = cfg.get("model") or (cfg.get("models") or [None])[0]
    if not model_name:
        raise ValueError("Vertical A requires 'model' (or models[0]).")

    spec = ModelRegistry.spec(model_name)
    if spec.family != "neural_operators":
        raise ValueError(f"Vertical A expects neural_operators model, got family={spec.family}")

    _emit(session, job.id, f"Selected neural operator: {model_name} (input_kind={spec.input_kind})")

    model_kwargs = cfg.get("model_kwargs", {}) or {}

    # PINO special: allow operator: {name, kwargs}
    if model_name in ("pino", "physics_informed_neural_operator") and isinstance(model_kwargs.get("operator"), dict):
        inner = model_kwargs["operator"]
        inner_name = inner.get("name")
        inner_kwargs = inner.get("kwargs", {}) or {}
        if not inner_name:
            raise ValueError("PINO requires operator: {name, kwargs}")
        inner_model = ModelRegistry.build(inner_name, **inner_kwargs)
        model_kwargs = dict(model_kwargs)
        model_kwargs["operator"] = inner_model

    model = ModelRegistry.build(model_name, **model_kwargs)
    adapter = OperatorAdapter()

    device = cfg.get("device", "cpu")
    epochs = int(cfg.get("epochs", 250))
    lr = float(cfg.get("lr", 1e-3))

    ds_cfg = cfg.get("dataset", {}) or {}
    H = int(ds_cfg.get("H", 96))
    W = int(ds_cfg.get("W", 96))
    L = int(ds_cfg.get("L", 256))
    batch_size = int(ds_cfg.get("batch_size", 8))
    branch_dim = int(ds_cfg.get("branch_dim", model_kwargs.get("branch_dim", 128)))
    n_coords = int(ds_cfg.get("n_coords", 1024))
    n_points = int(ds_cfg.get("n_points", 2048))

    geometry_params = cfg.get("geometry_params", {}) or {}
    solver_cfg = cfg.get("solver_cfg", {"name": "fdm", "equation": "heat2d"}) or {"name": "fdm", "equation": "heat2d"}

    input_kind = spec.input_kind
    if input_kind == "grid_or_points":
        input_kind_eff = "grid_2d"
    elif input_kind == "grid_1d":
        input_kind_eff = "grid_1d"
    elif input_kind == "operator_branch_trunk":
        input_kind_eff = "operator_branch_trunk"
    elif input_kind == "points":
        input_kind_eff = "points"
    else:
        input_kind_eff = "grid_2d"

    dataset = build_dataset_for_operator(
        input_kind=input_kind_eff,
        batch_size=batch_size,
        H=H, W=W, L=L,
        branch_dim=branch_dim,
        n_coords=n_coords,
        n_points=n_points,
        geometry_params=geometry_params,
        solver_cfg=solver_cfg,
        device=device,
    )
    _emit(session, job.id, f"Built dataset kind={dataset.kind} batch={batch_size} solver={solver_cfg.get('name')}")

    train_info = train_operator_model(
        model, adapter, dataset,
        epochs=epochs, lr=lr, device=device,
        emit=lambda jid,msg:_emit(session,jid,msg),
        job_id=job.id
    )

    model.eval()
    with torch.no_grad():
        batch = {"y_true": dataset.y_true.to(device)}
        if dataset.kind == "grid_1d":
            batch["u_grid_1d"] = dataset.u_in.to(device)
        elif dataset.kind == "grid_2d":
            batch["u_grid"] = dataset.u_in.to(device)
        elif dataset.kind == "branch_trunk":
            batch["u_branch"] = dataset.u_in.to(device)
            batch["coords"] = dataset.coords.to(device)
        elif dataset.kind == "points":
            batch["u_points"] = dataset.u_points.to(device)
            batch["coords_points"] = dataset.coords_points.to(device)
        out = adapter.forward_batch(model, batch)
        pred = out.y if hasattr(out, "y") else out
        y_true = dataset.y_true.to(device)
        mse = float(((pred - y_true) ** 2).mean().detach().cpu())

    try:
        render_preview(dataset, pred, os.path.join(run_dir, "preview.png"))
    except Exception as e:
        _emit(session, job.id, f"Preview render failed: {e}", level="WARN")

    result = {
        "mode": "vertical_a_operator",
        "model": model_name,
        "family": spec.family,
        "input_kind": spec.input_kind,
        "dataset_kind": dataset.kind,
        "solver_cfg": solver_cfg,
        "geometry_params": geometry_params,
        "metrics": {"mse": mse},
        "train": train_info,
    }
    _write_json(os.path.join(run_dir, "result.json"), result)
    return result


def _run_vertical_b(job, session) -> Dict[str, Any]:
    """Digital Twin Builder MVP: create a streaming-like synthetic TS dataset and train selected TS models if present."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = _load_config_bundle(job.config or {}, session, job.id)
    run_dir = _job_dir(job.id)
    _write_json(os.path.join(run_dir, "config_resolved.json"), cfg)

    # Synthetic sensor stream: damped oscillator + drift
    T = int(cfg.get("T", 2000))
    dt = float(cfg.get("dt", 0.01))
    t = np.arange(T) * dt
    y = np.exp(-0.01 * t) * np.sin(2*np.pi*0.5*t) + 0.05*np.sin(2*np.pi*0.03*t) + 0.02*np.random.randn(T)

    # Minimal forecast: persistence + simple EMA baseline (placeholder for real pinneaple_timeseries)
    alpha = 0.2
    yhat = np.zeros_like(y)
    yhat[0] = y[0]
    for i in range(1, T):
        yhat[i] = alpha*y[i-1] + (1-alpha)*yhat[i-1]

    mse = float(np.mean((yhat - y)**2))

    plt.figure(figsize=(10,4))
    plt.title("Digital Twin Stream (MVP)")
    plt.plot(t[:600], y[:600], label="sensor")
    plt.plot(t[:600], yhat[:600], label="baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "preview_ts.png"), dpi=150)
    plt.close()

    result = {"mode":"vertical_b_digital_twin_mvp","metrics":{"mse":mse},"note":"Plug pinneaple_timeseries + pinneaple_pdb here for production."}
    _write_json(os.path.join(run_dir, "result.json"), result)
    return result


def _run_vertical_d(job, session) -> Dict[str, Any]:
    """Physics + Time Series Fusion MVP: forecasting with a physics-inspired constraint penalty placeholder."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = _load_config_bundle(job.config or {}, session, job.id)
    run_dir = _job_dir(job.id)
    _write_json(os.path.join(run_dir, "config_resolved.json"), cfg)

    T = int(cfg.get("T", 2000))
    dt = float(cfg.get("dt", 0.02))
    t = np.arange(T) * dt

    # Create a "grid load" signal with spikes (extremes)
    base = 0.6 + 0.15*np.sin(2*np.pi*0.03*t) + 0.08*np.sin(2*np.pi*0.12*t)
    spikes = (np.random.rand(T) < 0.01).astype(float) * (0.8*np.random.rand(T))
    y = base + spikes + 0.03*np.random.randn(T)

    # Baseline forecast (EMA)
    alpha = 0.15
    yhat = np.zeros_like(y)
    yhat[0] = y[0]
    for i in range(1, T):
        yhat[i] = alpha*y[i-1] + (1-alpha)*yhat[i-1]

    # Physics-like constraint: ramp-rate penalty (encourage bounded dy/dt)
    dy = np.diff(yhat, prepend=yhat[0])
    ramp_limit = float(cfg.get("ramp_limit", 0.08))
    physics_pen = float(np.mean(np.maximum(0.0, np.abs(dy)-ramp_limit)**2))

    mse = float(np.mean((yhat - y)**2))
    total = mse + float(cfg.get("lambda_physics", 1.0))*physics_pen

    plt.figure(figsize=(10,4))
    plt.title("Physics + Time Series Fusion (MVP)")
    plt.plot(t[:700], y[:700], label="true")
    plt.plot(t[:700], yhat[:700], label="forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "preview_ts.png"), dpi=150)
    plt.close()

    result = {"mode":"vertical_d_physics_ts_mvp","metrics":{"mse":mse,"physics_penalty":physics_pen,"total":total},
              "note":"Replace with pinneaple_timeseries model + pinneaple_pinn physics loss builder."}
    _write_json(os.path.join(run_dir, "result.json"), result)
    return result


def _run_vertical_c(job, session) -> Dict[str, Any]:
    cfg = _load_config_bundle(job.config or {}, session, job.id)
    run_dir = _job_dir(job.id)
    _write_json(os.path.join(run_dir, "config_resolved.json"), cfg)

    try:
        from pinneaple_models.register_all import register_all as register_all_models
        register_all_models()

        from pinneaple_arena.pipeline.dataset_builder import build_from_solver
        from pinneaple_arena.runner.run_sweep import run_sweep
        from pinneaple_arena.runner.compare import compare_runs

        models = cfg.get("models", [])
        if isinstance(models, str):
            models = [models]

        problem_spec = cfg.get("problem_spec", {})
        geometry = cfg.get("geometry", {})
        solver_cfg = cfg.get("solver_cfg", {"name": "fdm", "equation": "heat2d"})

        dataset = build_from_solver(problem_spec=problem_spec, geometry=geometry, solver_cfg=solver_cfg)

        out_dir = os.path.join(run_dir, "runs")
        os.makedirs(out_dir, exist_ok=True)

        _emit(session, job.id, f"Starting sweep with {len(models)} models")
        run_sweep(
            dataset=dataset,
            models=models,
            parallelism=cfg.get("parallelism", "process"),
            gpus=cfg.get("gpus", "auto"),
            ddp_per_model=bool(cfg.get("ddp_per_model", False)),
            out_dir=out_dir,
        )

        cmp_dir = os.path.join(run_dir, "compare")
        os.makedirs(cmp_dir, exist_ok=True)
        compare = compare_runs(runs_dir=out_dir, out_dir=cmp_dir)
        result = {"mode": "arena", "compare": compare}
        _write_json(os.path.join(run_dir, "result.json"), result)
        return result

    except Exception as e:
        _emit(session, job.id, f"Arena pipeline failed; fallback. Error: {e}", level="WARN")
        result = {"mode": "fallback", "error": str(e)}
        _write_json(os.path.join(run_dir, "result.json"), result)
        return result


@celery_app.task(name="web_backend.app.tasks.run_job")
def run_job(job_id: str) -> str:
    session = SessionLocal()
    try:
        job = session.get(Job, job_id)
        if not job:
            return "missing"

        job.status = "running"
        job.updated_at = dt.datetime.utcnow()
        session.commit()

        _emit(session, job_id, "Worker started job execution.")

        vertical = (job.vertical or "").strip().lower()
        if vertical in ("vertical_a", "surrogate"):
            _run_vertical_a(job, session)
        elif vertical in ("vertical_b", "digital_twin"):
            _run_vertical_b(job, session)
        elif vertical in ("vertical_d", "physics_ts", "fusion"):
            _run_vertical_d(job, session)
        elif vertical in ("vertical_c", "arena", "benchmark"):
            _run_vertical_c(job, session)
        else:
            run_dir = _job_dir(job.id)
            result = {"mode": "placeholder", "note": f"Unknown vertical '{job.vertical}'. Implement pipeline."}
            _write_json(os.path.join(run_dir, "result.json"), result)

        job = session.get(Job, job_id)
        job.status = "completed"
        job.updated_at = dt.datetime.utcnow()
        session.commit()

        _emit(session, job_id, "Job completed. Uploading artifacts.")
        _finalize_artifacts(job_id, session)
        _emit(session, job_id, "Artifacts uploaded.")
        return "ok"

    except Exception as e:
        tb = traceback.format_exc()
        try:
            _emit(session, job_id, f"Job failed: {e}\n{tb}", level="ERROR")
            job = session.get(Job, job_id)
            if job:
                job.status = "failed"
                job.error = f"{e}\n{tb}"
                job.updated_at = dt.datetime.utcnow()
                session.commit()
        except Exception:
            pass
        return "failed"
    finally:
        session.close()


@celery_app.task(name="web_backend.app.tasks.run_job_gpu")
def run_job_gpu(job_id: str) -> str:
    return run_job(job_id)
