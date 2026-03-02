# Pinneaple WebApp v7 (Celery) — Verticals A/B/C/D

Highlights:
- Celery queues (cpu + gpu) with Postgres job tracking and MinIO artifacts
- `/api/models` supports filters (family/input_kind/supports_physics_loss)
- **Vertical A (Surrogate Engineering Platform)**:
  - Genetic **SDF implicit geometry**
  - boundary sampling via **marching squares**
  - solver hook: uses `pinneaple_solvers.SolverRegistry` and adapts IO contracts for FDM (Poisson), LBM (D2Q9), and SPH (WCSPH) (calls `register_all()`), then `SolverRegistry.build(name, ...)` and runs a single step (batched wrapper); fallback otherwise
  - trains a selected **Neural Operator** to learn `u_t -> u_{t+1}`
  - generates `preview.png` with boundary overlay
- **Vertical B (Digital Twin Builder)** MVP:
  - generates a synthetic sensor stream and baseline forecast + `preview_ts.png`
  - TODO: plug pinneaple_timeseries + pinneaple_pdb for real streaming recalibration
- **Vertical C (Benchmark Arena)**:
  - calls your real Arena sweep/compare if those modules exist; fallback otherwise
- **Vertical D (Physics + TS Fusion)** MVP:
  - adds a physics-like constraint penalty (ramp-rate) on forecasting + `preview_ts.png`
  - TODO: plug pinneaple_pinn physics losses

Run:
```bash
docker compose -f webapp/docker-compose.yml up --build
```

Open:
- UI: http://localhost:8080
- API docs: http://localhost:8000/docs
- MinIO: http://localhost:9001 (minioadmin/minioadmin)
