"""
Background training worker.
All training runs in a daemon thread; progress is pushed to Django Channels.
"""
from __future__ import annotations
import logging
import threading
import time
import uuid
import math
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Global registry: ws_run_id → {status, history, result, stop_flag, thread}
_registry: Dict[str, Dict[str, Any]] = {}
_lock = threading.Lock()


# ── Public API ────────────────────────────────────────────────────────────────

def start_training(ws_run_id: str, db_run_id: str, model_type: str,
                   config: dict, problem: dict, ts_data=None) -> None:
    with _lock:
        _registry[ws_run_id] = {
            "status":    "running",
            "history":   [],
            "result":    None,
            "stop_flag": threading.Event(),
            "thread":    None,
        }
    stop_flag = _registry[ws_run_id]["stop_flag"]

    def _target():
        try:
            _dispatch(ws_run_id, db_run_id, model_type, config, problem,
                      ts_data, stop_flag)
        except Exception as exc:
            _push(ws_run_id, {"type": "error", "status": "error", "msg": str(exc)})
            _save_db(db_run_id, status="error", error_msg=str(exc))

    t = threading.Thread(target=_target, daemon=True)
    with _lock:
        _registry[ws_run_id]["thread"] = t
    t.start()


def stop_training(ws_run_id: str) -> None:
    with _lock:
        entry = _registry.get(ws_run_id)
    if entry:
        entry["stop_flag"].set()


def get_status(ws_run_id: str) -> Optional[Dict]:
    with _lock:
        e = _registry.get(ws_run_id)
        if e is None:
            return None
        return {"status": e["status"], "history": list(e["history"]),
                "result": e["result"]}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _push(ws_run_id: str, data: dict) -> None:
    """Send a WebSocket group message and update in-memory registry."""
    with _lock:
        e = _registry.get(ws_run_id)
        if e:
            if data.get("type") == "progress":
                e["history"].append({
                    "epoch": data.get("epoch", 0),
                    "loss":  data.get("loss", 0.0),
                    "pde":   data.get("pde"),
                    "bc":    data.get("bc"),
                })
            e["status"] = data.get("status", e["status"])
            if data.get("type") == "done":
                e["result"] = data.get("result")

    try:
        from channels.layers import get_channel_layer
        from asgiref.sync import async_to_sync
        cl = get_channel_layer()
        async_to_sync(cl.group_send)(
            f"training_{ws_run_id}",
            {"type": "training.update", "payload": data},
        )
    except Exception as exc:
        logger.warning("WebSocket push failed for run %s: %s", ws_run_id, exc)


def _save_db(db_run_id: str, **kwargs) -> None:
    try:
        from django.utils import timezone
        from api.models import TrainingRun
        run = TrainingRun.objects.get(id=db_run_id)
        for k, v in kwargs.items():
            setattr(run, k, v)
        if kwargs.get("status") in ("done", "error", "stopped"):
            run.completed_at = timezone.now()
        run.save()
    except Exception as exc:
        logger.error("DB save failed for run %s: %s", db_run_id, exc, exc_info=True)


def _dispatch(ws_run_id, db_run_id, model_type, config, problem, ts_data, stop_flag):
    _save_db(db_run_id, status="running")
    if   model_type == "pinn_mlp": _train_pinn(ws_run_id, db_run_id, config, problem, stop_flag)
    elif model_type == "lbm":      _run_lbm(ws_run_id, db_run_id, config, problem)
    elif model_type == "fdm":      _run_fdm(ws_run_id, db_run_id, config, problem)
    elif model_type == "fem":      _run_fem(ws_run_id, db_run_id, config, problem)
    elif model_type in ("tcn","lstm","tft"): _train_ts(ws_run_id, db_run_id, model_type, config, ts_data, stop_flag)
    elif model_type == "fft":      _fit_fft(ws_run_id, db_run_id, config, ts_data)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# ── PINN ──────────────────────────────────────────────────────────────────────

def _train_pinn(ws_run_id, db_run_id, config, problem, stop_flag):
    import torch
    import torch.nn as nn
    from api.problem_defs import generate_collocation_points, build_pinn_loss

    n_epochs   = int(config.get("n_epochs",   500))
    lr         = float(config.get("lr",        1e-3))
    hidden     = int(config.get("hidden",      64))
    n_layers   = int(config.get("n_layers",    4))
    n_interior = int(config.get("n_interior",  2000))

    domain = problem.get("domain", {"x": [0, 1], "y": [0, 1]})
    dim    = len(domain)

    layers = [nn.Linear(dim, hidden), nn.Tanh()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers.append(nn.Linear(hidden, 1))
    model = nn.Sequential(*layers)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    col = generate_collocation_points(problem, n_interior=n_interior, n_boundary=200)
    pts = torch.tensor(col["interior"], dtype=torch.float32)
    bnd = torch.tensor(col["boundary"], dtype=torch.float32)
    rf  = build_pinn_loss(problem)

    history      = []
    report_every = max(1, n_epochs // 100)

    for epoch in range(1, n_epochs + 1):
        if stop_flag.is_set():
            _push(ws_run_id, {"type": "stopped", "status": "stopped"})
            _save_db(db_run_id, status="stopped", history=history)
            return

        opt.zero_grad()
        pde_loss = rf(model, pts)
        bc_loss  = model(bnd).pow(2).mean()
        loss     = pde_loss + bc_loss
        loss.backward()
        opt.step()

        if epoch % report_every == 0:
            val = float(loss.item())
            entry = {"epoch": epoch, "loss": val,
                     "pde": float(pde_loss.item()), "bc": float(bc_loss.item())}
            history.append(entry)
            _push(ws_run_id, {**entry, "type": "progress", "total": n_epochs,
                               "status": "running"})

    # Inference on 64×64 grid for instant preview
    keys = list(domain.keys())
    x0, x1 = domain[keys[0]]; y0, y1 = domain[keys[1]]
    xg = np.linspace(x0, x1, 64); yg = np.linspace(y0, y1, 64)
    Xg, Yg = np.meshgrid(xg, yg, indexing="ij")
    gpts = torch.tensor(
        np.column_stack([Xg.ravel(), Yg.ravel()]).astype(np.float32)
    )
    model.eval()
    with torch.no_grad():
        u = model(gpts).numpy().reshape(64, 64)

    final_loss = history[-1]["loss"] if history else float("nan")
    result = {
        "final_loss": final_loss,
        "history":    history,
        "inference_grid": {
            "x": xg.tolist(), "y": yg.tolist(), "u": u.tolist(),
            "coord_keys": keys[:2],
        },
    }
    with _lock:
        _registry[ws_run_id]["result"] = result
        _registry[ws_run_id]["status"] = "done"

    _save_db(db_run_id, status="done", final_loss=final_loss,
             history=history, result_data=result)
    _push(ws_run_id, {"type": "done", "status": "done",
                      "final_loss": final_loss, "result": result})


# ── LBM ───────────────────────────────────────────────────────────────────────

def _run_lbm(ws_run_id, db_run_id, config, problem):
    from pinneaple_solvers.lbm import LBMSolver, cylinder_mask, rectangle_mask

    nx  = int(config.get("nx", 160)); ny  = int(config.get("ny", 64))
    Re  = float(config.get("Re", 200.0))
    u_in = float(config.get("u_in", 0.05))
    steps      = int(config.get("steps", 4000))
    save_every = int(config.get("save_every", 500))

    _push(ws_run_id, {"type": "progress", "status": "running",
                      "epoch": 0, "loss": 0, "msg": "Building LBM grid…"})

    obstacle = None
    obs_cfg  = config.get("obstacle") or problem.get("params", {}).get("obstacle")
    if obs_cfg:
        if obs_cfg.get("type") == "cylinder":
            obstacle = cylinder_mask(nx, ny,
                cx=int(obs_cfg.get("cx", nx // 4)),
                cy=int(obs_cfg.get("cy", ny // 2)),
                r=int(obs_cfg.get("r",   ny // 8)))
        elif obs_cfg.get("type") == "rectangle":
            obstacle = rectangle_mask(nx, ny,
                x0=int(obs_cfg.get("x0", 20)), x1=int(obs_cfg.get("x1", 30)),
                y0=int(obs_cfg.get("y0", 20)), y1=int(obs_cfg.get("y1", 44)))

    solver = LBMSolver(nx=nx, ny=ny, Re=Re, u_in=u_in, obstacle_mask=obstacle)
    _push(ws_run_id, {"type": "progress", "status": "running",
                      "epoch": 0, "loss": 0, "msg": f"Running {steps} LBM steps…"})

    out = solver.forward(steps=steps, save_every=save_every)
    e   = out.extras

    ux_np  = e["ux"].numpy()
    uy_np  = e["uy"].numpy()

    # Vorticity: ω = ∂ux/∂y − ∂uy/∂x  (central differences, boundary via forward/backward)
    dux_dy = np.gradient(ux_np, axis=1)   # ∂ux/∂y  (axis 1 = y)
    duy_dx = np.gradient(uy_np, axis=0)   # ∂uy/∂x  (axis 0 = x)
    vorticity = dux_dy - duy_dx

    # Q-criterion: Q = −½(S:S − Ω:Ω) where S is strain, Ω is rotation
    # In 2-D: Q = −(S₁₁² + 2S₁₂² + S₂₂²)/2 + Ω₁₂²
    dux_dx = np.gradient(ux_np, axis=0)
    duy_dy = np.gradient(uy_np, axis=1)
    S11, S22   = dux_dx, duy_dy
    S12        = 0.5 * (dux_dy + duy_dx)
    Omega12    = 0.5 * vorticity
    Q_crit     = Omega12**2 - 0.5 * (S11**2 + 2 * S12**2 + S22**2)

    result = {
        "type":          "lbm",
        "ux":            ux_np.tolist(),
        "uy":            uy_np.tolist(),
        "rho":           e["rho"].numpy().tolist(),
        "vel_mag":       e["vel_mag"].numpy().tolist(),
        "vorticity":     vorticity.tolist(),
        "Q":             Q_crit.tolist(),
        "trajectory_ux": [t.numpy().tolist() for t in e["trajectory_ux"]],
        "trajectory_uy": [t.numpy().tolist() for t in e["trajectory_uy"]],
        "obstacle":      obstacle.numpy().tolist() if obstacle is not None else None,
        "nx": nx, "ny": ny, "Re": Re,
    }
    with _lock:
        _registry[ws_run_id]["result"] = result
        _registry[ws_run_id]["status"] = "done"
    _save_db(db_run_id, status="done", result_data=result)
    _push(ws_run_id, {"type": "done", "status": "done", "result": result})


# ── FDM ───────────────────────────────────────────────────────────────────────

def _run_fdm(ws_run_id, db_run_id, config, problem):
    nx    = int(config.get("nx",    64))
    ny    = int(config.get("ny",    64))
    iters = int(config.get("iters", 8000))
    omega = float(config.get("omega", 1.5))   # SOR relaxation factor

    _push(ws_run_id, {"type": "progress", "status": "running",
                      "epoch": 0, "loss": 0, "msg": "Running FDM solver…"})

    domain    = problem.get("domain", {"x": [0, 1], "y": [0, 1]})
    params    = problem.get("params", {})
    pde_kind  = problem.get("_preset_key", "")
    keys      = list(domain.keys())[:2]
    x0, x1   = domain[keys[0]]
    y0, y1   = domain[keys[1]]
    x = np.linspace(x0, x1, nx)
    y = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = (x1 - x0) / (nx - 1)
    dy = (y1 - y0) / (ny - 1)

    # Build source term f and exact solution (where available) based on problem type
    if "poisson" in pde_kind or "poisson" in problem.get("name", "").lower():
        # ∇²u = −2π²sin(πx)sin(πy)  →  exact: sin(πx)sin(πy)
        f       = -2 * math.pi**2 * np.sin(math.pi * X) * np.sin(math.pi * Y)
        u_exact = np.sin(math.pi * X) * np.sin(math.pi * Y)
        label   = "u (Poisson)"
    elif "heat" in pde_kind or "heat" in problem.get("name", "").lower():
        # ∇²T = −Q/k  (uniform source; zero-BC → smooth field)
        Q_src   = float(params.get("Q", 1.0))
        k_cond  = float(params.get("k", 1.0))
        f       = np.full((nx, ny), -Q_src / k_cond)
        u_exact = None
        label   = "T (Heat)"
    else:
        # Generic Poisson: ∇²u = −2π²sin(πx)sin(πy) as default benchmark
        f       = -2 * math.pi**2 * np.sin(math.pi * X) * np.sin(math.pi * Y)
        u_exact = np.sin(math.pi * X) * np.sin(math.pi * Y)
        label   = "u (Poisson)"

    # SOR iteration:  (∇² u = f)  ↔  u[i,j] ≈ (neighbours + h²·f) / 4
    u = np.zeros((nx, ny))
    for _ in range(iters):
        u_prev = u[1:-1, 1:-1].copy()
        denom  = 2.0 / dx**2 + 2.0 / dy**2
        u[1:-1, 1:-1] = ((1 - omega) * u[1:-1, 1:-1]
                         + omega * (
                             (u[:-2, 1:-1] + u[2:, 1:-1]) / dx**2
                             + (u[1:-1, :-2] + u[1:-1, 2:]) / dy**2
                             - f[1:-1, 1:-1]
                         ) / denom)
        if np.max(np.abs(u[1:-1, 1:-1] - u_prev)) < 1e-9:
            break

    result: dict = {
        "type":       "fdm",
        "field":      u.tolist(),
        "x":          x.tolist(),
        "y":          y.tolist(),
        "coord_keys": keys,
        "label":      label,
    }
    if u_exact is not None:
        err  = u - u_exact
        result["exact"]      = u_exact.tolist()
        result["l2_error"]   = float(np.sqrt(np.mean(err**2)))
        result["linf_error"] = float(np.max(np.abs(err)))

    with _lock:
        _registry[ws_run_id]["result"] = result
        _registry[ws_run_id]["status"] = "done"
    _save_db(db_run_id, status="done", result_data=result)
    _push(ws_run_id, {"type": "done", "status": "done", "result": result})


# ── FEM ───────────────────────────────────────────────────────────────────────

def _run_fem(ws_run_id, db_run_id, config, problem):
    nx = int(config.get("nx", 20)); ny = int(config.get("ny", 20))
    _push(ws_run_id, {"type": "progress", "status": "running",
                      "epoch": 0, "loss": 0, "msg": "Running FEM solver…"})

    domain = problem.get("domain", {"x": [0, 1], "y": [0, 1]})
    keys   = list(domain.keys())[:2]
    x0, x1 = domain[keys[0]]; y0, y1 = domain[keys[1]]
    bounds  = (x0, x1, y0, y1)

    try:
        from pinneaple_solvers.fem import FEMSolver
        solver = FEMSolver(nx=nx, ny=ny)
        out    = solver.forward(mesh=(nx, ny, bounds), params={})
        nodes  = out.extras.get("nodes", np.zeros(((nx+1)*(ny+1), 2)))
        field  = out.result.numpy() if hasattr(out.result, "numpy") else np.zeros((nx+1)*(ny+1))
        nodes_list = nodes.tolist() if hasattr(nodes, "tolist") else nodes
        field_list = field.tolist() if hasattr(field, "tolist") else list(field)
        result = {"type": "fem", "nodes": nodes_list, "field": field_list,
                  "coord_keys": keys, "label": "u (FEM)"}
    except Exception as exc:
        nn = (nx + 1) * (ny + 1)
        result = {"type": "fem", "nodes": np.zeros((nn, 2)).tolist(),
                  "field": np.zeros(nn).tolist(), "label": "u (FEM)",
                  "error": str(exc)}

    with _lock:
        _registry[ws_run_id]["result"] = result
        _registry[ws_run_id]["status"] = "done"
    _save_db(db_run_id, status="done", result_data=result)
    _push(ws_run_id, {"type": "done", "status": "done", "result": result})


# ── Timeseries neural ─────────────────────────────────────────────────────────

def _train_ts(ws_run_id, db_run_id, model_type, config, ts_data, stop_flag):
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    if ts_data is None or len(ts_data) == 0:
        raise ValueError("No timeseries data provided.")

    y         = np.array(ts_data, dtype=np.float32)
    input_len = int(config.get("input_len", 32))
    horizon   = int(config.get("horizon",   16))
    epochs    = int(config.get("epochs",    50))
    lr        = float(config.get("lr",      1e-3))

    # Build sliding windows
    X_list, Y_list = [], []
    for i in range(len(y) - input_len - horizon + 1):
        X_list.append(y[i: i + input_len])
        Y_list.append(y[i + input_len: i + input_len + horizon])

    if not X_list:
        raise ValueError(f"Series too short ({len(y)}) for input_len={input_len}+horizon={horizon}.")

    X_t = torch.tensor(np.array(X_list)[:, :, None], dtype=torch.float32)
    Y_t = torch.tensor(np.array(Y_list)[:, :, None], dtype=torch.float32)
    dl  = DataLoader(TensorDataset(X_t, Y_t), batch_size=32, shuffle=True)

    try:
        from pinneaple_timeseries.models import TCNForecaster, LSTMForecaster, TFTForecaster
        mdl_map = {"tcn": TCNForecaster, "lstm": LSTMForecaster, "tft": TFTForecaster}
        model   = mdl_map[model_type](input_len=input_len, horizon=horizon, n_features=1)
    except (ImportError, KeyError):
        import torch.nn as nn

        class _FallbackLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(1, 64, 2, batch_first=True)
                self.fc   = nn.Linear(64, horizon)

            def forward(self, x):
                h, _ = self.lstm(x)
                pred  = self.fc(h[:, -1, :]).unsqueeze(-1)  # (B, H, 1)
                return type("Out", (), {"y_hat": pred})()

        model = _FallbackLSTM()

    opt     = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for ep in range(1, epochs + 1):
        if stop_flag.is_set():
            _push(ws_run_id, {"type": "stopped", "status": "stopped"})
            _save_db(db_run_id, status="stopped", history=history)
            return

        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            out  = model(xb)
            pred = out.y_hat if hasattr(out, "y_hat") else out
            loss = torch.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()

        avg = ep_loss / max(len(dl), 1)
        if ep % max(1, epochs // 20) == 0:
            entry = {"epoch": ep, "loss": avg}
            history.append(entry)
            _push(ws_run_id, {"type": "progress", "epoch": ep, "loss": avg,
                               "total": epochs, "status": "running"})

    # Produce one-shot forecast from last input_len observations
    model.eval()
    ctx  = torch.tensor(y[-input_len:][None, :, None], dtype=torch.float32)
    with torch.no_grad():
        out = model(ctx)
        forecast = (out.y_hat if hasattr(out, "y_hat") else out).squeeze().numpy().tolist()

    final_loss = history[-1]["loss"] if history else float("nan")
    result = {
        "type":       model_type,
        "history":    history,
        "final_loss": final_loss,
        "forecast":   forecast,
        "input_len":  input_len,
        "horizon":    horizon,
    }
    with _lock:
        _registry[ws_run_id]["result"] = result
        _registry[ws_run_id]["status"] = "done"
    _save_db(db_run_id, status="done", final_loss=final_loss,
             history=history, result_data=result)
    _push(ws_run_id, {"type": "done", "status": "done",
                      "final_loss": final_loss, "result": result})


# ── FFT ───────────────────────────────────────────────────────────────────────

def _fit_fft(ws_run_id, db_run_id, config, ts_data):
    if ts_data is None or len(ts_data) == 0:
        raise ValueError("No timeseries data provided.")

    y           = np.array(ts_data, dtype=np.float64)
    n_harmonics = int(config.get("n_harmonics", 5))
    detrend     = bool(config.get("detrend", True))
    horizon     = int(config.get("horizon",    50))

    _push(ws_run_id, {"type": "progress", "status": "running",
                      "epoch": 0, "loss": 0, "msg": "Fitting FFT model…"})

    try:
        from pinneaple_timeseries.decomposition.fft_forecaster import FFTForecaster
        m = FFTForecaster(n_harmonics=n_harmonics, detrend=detrend)
        m.fit(y)
        forecast = m.predict(horizon).tolist()
    except ImportError:
        N    = len(y)
        yf   = np.fft.rfft(y)
        freq = np.fft.rfftfreq(N)
        amp  = np.abs(yf) / N
        pha  = np.angle(yf)
        idx  = np.argsort(amp)[::-1][: n_harmonics + 1]
        t_f  = np.arange(N, N + horizon)
        fore = np.zeros(horizon)
        for i in idx:
            fore += 2 * amp[i] * np.cos(2 * math.pi * freq[i] * t_f + pha[i])
        forecast = fore.tolist()

    result = {"type": "fft", "forecast": forecast,
               "n_harmonics": n_harmonics, "final_loss": 0.0}
    with _lock:
        _registry[ws_run_id]["result"] = result
        _registry[ws_run_id]["status"] = "done"
    _save_db(db_run_id, status="done", result_data=result)
    _push(ws_run_id, {"type": "done", "status": "done", "result": result})
