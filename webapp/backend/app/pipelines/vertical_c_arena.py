from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# Try to use Pinneaple Arena multi-model pipeline if user has applied patches.
_HAS_ARENA = False
try:
    from pinneaple_arena.runner.run_sweep import run_sweep  # type: ignore
    from pinneaple_arena.runner.compare import compare_runs  # type: ignore
    from pinneaple_arena.pipeline.dataset_builder import build_from_solver  # type: ignore
    from pinneaple_pinn.compiler import compile_problem  # type: ignore
    _HAS_ARENA = True
except Exception:
    _HAS_ARENA = False


def _internal_make_heat_dataset(nx=48, ny=48, nt=30, dt=0.02, alpha=0.01, n=25000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, nx); y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    u = np.zeros((nt, ny, nx), dtype=np.float32)
    u[0] = np.exp(-80.0 * ((X-0.4)**2 + (Y-0.6)**2)).astype(np.float32)
    dx = x[1]-x[0]; dy = y[1]-y[0]
    cx = alpha*dt/(dx*dx); cy = alpha*dt/(dy*dy)
    for k in range(nt-1):
        uk=u[k]; un=uk.copy()
        un[1:-1,1:-1]=uk[1:-1,1:-1]+cx*(uk[1:-1,2:]-2*uk[1:-1,1:-1]+uk[1:-1,:-2])+cy*(uk[2:,1:-1]-2*uk[1:-1,1:-1]+uk[:-2,1:-1])
        un[0,:]=0; un[-1,:]=0; un[:,0]=0; un[:,-1]=0
        u[k+1]=un
    xs=np.stack([X.reshape(-1),Y.reshape(-1)],axis=-1).astype(np.float32)
    ts=(np.arange(nt)*dt).astype(np.float32)
    total=xs.shape[0]*nt
    idx=rng.choice(total,size=n,replace=False)
    t_idx=idx//xs.shape[0]
    x_idx=idx%xs.shape[0]
    X3=np.concatenate([xs[x_idx], ts[t_idx,None]], axis=1)
    yv=u[t_idx].reshape(nt,-1)[t_idx,x_idx][:,None]
    return X3, yv


def _fit_model(name: str, Xtr, Ytr, Xte, Yte, epochs=8, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xtr=Xtr.to(device); Ytr=Ytr.to(device); Xte=Xte.to(device); Yte=Yte.to(device)

    if name == "linear":
        model = torch.nn.Linear(3,1)
    elif name == "mlp":
        model = torch.nn.Sequential(torch.nn.Linear(3,128), torch.nn.GELU(), torch.nn.Linear(128,128), torch.nn.GELU(), torch.nn.Linear(128,1))
    elif name == "res_mlp":
        class Res(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1=torch.nn.Linear(3,128)
                self.b1=torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(128,128))
                self.b2=torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(128,128))
                self.out=torch.nn.Linear(128,1)
            def forward(self,x):
                h=torch.nn.functional.gelu(self.fc1(x))
                h=h+self.b1(h)
                h=h+self.b2(h)
                return self.out(torch.nn.functional.gelu(h))
        model=Res()
    elif name == "fourier_mlp_pinn":
        # physics-aware placeholder: use Fourier features + supervised (physics term omitted in fallback)
        class FF(torch.nn.Module):
            def __init__(self, m=16):
                super().__init__()
                B=torch.randn(3,m)*2.0
                self.register_buffer("B", B)
                self.net=torch.nn.Sequential(torch.nn.Linear(2*m,128), torch.nn.GELU(), torch.nn.Linear(128,128), torch.nn.GELU(), torch.nn.Linear(128,1))
            def forward(self,x):
                z=2*np.pi*(x@self.B)
                f=torch.cat([torch.sin(z), torch.cos(z)], dim=-1)
                return self.net(f)
        model=FF()
    else:  # siren_pinn
        class Sine(torch.nn.Module):
            def forward(self,x): return torch.sin(x)
        model=torch.nn.Sequential(torch.nn.Linear(3,128), Sine(), torch.nn.Linear(128,128), Sine(), torch.nn.Linear(128,1))

    model.to(device)
    opt=torch.optim.Adam(model.parameters(), lr=lr)
    bs=2048
    for ep in range(epochs):
        perm=torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0], bs):
            j=perm[i:i+bs]
            xb=Xtr[j]; yb=Ytr[j]
            pred=model(xb)
            loss=torch.mean((pred-yb)**2)
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred=model(Xte)
        mse=torch.mean((pred-Yte)**2).item()
        rel=(torch.linalg.norm(pred-Yte)/(torch.linalg.norm(Yte)+1e-12)).item()
    return model, {"mse": mse, "rel_l2": rel}


def run_vertical_c(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Vertical C (Scientific Benchmark Arena)
    Improved: tries to run Pinneaple Arena sweep+compare if present.
    Otherwise runs an internal 5-model benchmark with same UI contract.
    """
    models = cfg.get("models", ["linear", "mlp", "res_mlp", "fourier_mlp_pinn", "siren_pinn"])
    if len(models) < 5:
        # enforce at least 5 for the requested experience
        models = ["linear", "mlp", "res_mlp", "fourier_mlp_pinn", "siren_pinn"]

    out_root = Path(cfg.get("artifacts_root", "runs/webapp_v2"))
    out_root.mkdir(parents=True, exist_ok=True)

    if _HAS_ARENA:
        # Arena-based path (expects the user applied the earlier patches)
        task_name = str(cfg.get("task_name", "heat2d_genetic"))
        run_dir = out_root / "arena" / task_name / str(int(time.time()))
        run_dir.mkdir(parents=True, exist_ok=True)

        # build dataset via solver using build_from_solver() in patched pipeline
        # configs are passed through
        bundle = build_from_solver(
            problem_spec=cfg.get("problem_spec", {}),
            geometry=cfg.get("geometry", {}),
            solver_cfg=cfg.get("solver_cfg", {}),
        )

        # run sweep (multi-gpu aware inside run_sweep)
        results = run_sweep(
            models=models,
            bundle=bundle,
            out_dir=str(run_dir),
            parallelism=str(cfg.get("parallelism", "process")),
            gpus=str(cfg.get("gpus", "auto")),
            ddp_per_model=bool(cfg.get("ddp_per_model", False)),
            model_kwargs=cfg.get("model_kwargs", {}),
            train_cfg=cfg.get("train_cfg", {}),
            loss_cfg=cfg.get("loss_cfg", {}),
        )

        # compare
        comp_dir = run_dir / "compare"
        comp = compare_runs(results["run_dirs"], out_dir=str(comp_dir), fields=cfg.get("fields", None))

        # pick best model (by primary metric)
        ranking = comp.get("ranking", [])
        best = ranking[0]["model"] if ranking else None
        best_plot_b64 = None
        plot_path = comp.get("plots", {}).get("best")
        if plot_path and Path(plot_path).exists():
            best_plot_b64 = base64.b64encode(Path(plot_path).read_bytes()).decode("ascii")

        return {
            "status": "ok",
            "mode": "arena",
            "run_root": str(run_dir),
            "ranking": ranking,
            "table": comp.get("table", {}),
            "artifacts": {"best_plot_png_b64": best_plot_b64},
        }

    # Fallback (still 5 models + ranking)
    nx=int(cfg.get("nx",48)); ny=int(cfg.get("ny",48)); nt=int(cfg.get("nt",30))
    dt=float(cfg.get("dt",0.02)); alpha=float(cfg.get("alpha",0.01))
    n_total=int(cfg.get("n_total",30000))
    seed=int(cfg.get("seed",0))
    X, y = _internal_make_heat_dataset(nx,ny,nt,dt,alpha,n=n_total,seed=seed)
    X=torch.tensor(X, dtype=torch.float32)
    y=torch.tensor(y, dtype=torch.float32)
    n=int(0.8*X.shape[0])
    Xtr,Ytr=X[:n],y[:n]
    Xte,Yte=X[n:],y[n:]

    metrics={}
    models_fit={}
    for m in models:
        model, met = _fit_model(m, Xtr, Ytr, Xte, Yte, epochs=int(cfg.get("epochs",8)), lr=float(cfg.get("lr",1e-3)))
        metrics[m]=met
        models_fit[m]=model

    ranking=sorted([{"model":k, **v} for k,v in metrics.items()], key=lambda r: r["rel_l2"])
    best=ranking[0]["model"]

    # plot best prediction error slice
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xs = Xte[:5000].to(device)
    with torch.no_grad():
        pred=models_fit[best].to(device)(xs).detach().cpu().numpy().reshape(-1)
    ref=Yte[:5000].cpu().numpy().reshape(-1)
    err=pred-ref

    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(err, bins=40)
    ax.set_title(f"Best model error distribution: {best}")
    buf=io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    img_b64=base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "status": "ok",
        "mode": "fallback",
        "ranking": ranking,
        "table": metrics,
        "artifacts": {"best_plot_png_b64": img_b64},
        "notes": {"arena_available": False, "hint": "Apply Pinneaple Arena patches to enable sweep+compare+DDP."},
    }
