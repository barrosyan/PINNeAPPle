"""
Benchmark views — run standardised problems and record results.
"""
import math
import time
import numpy as np
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from api.models import BenchmarkResult
from api.serializers import BenchmarkResultSerializer

BENCHMARK_DEFS = {
    "poisson_fdm": {
        "name":        "Poisson 2D — FDM vs Analytical",
        "category":    "Classical PDE",
        "description": "Unit-square Poisson ∇²u = -2π²sin(πx)sin(πy). FDM Gauss-Seidel vs u=sin(πx)sin(πy). Reports L2 and L∞ errors.",
        "params":      {"nx": 64, "ny": 64, "iters": 5000},
    },
    "burgers_pinn": {
        "name":        "Burgers 1D — PINN",
        "category":    "PINN",
        "description": "Viscous Burgers ν=0.01, PINN MLP 4×64 trained 500 epochs. Reports final PDE residual.",
        "params":      {"n_epochs": 500, "hidden": 64, "n_layers": 4,
                        "lr": 1e-3, "n_interior": 2000},
    },
    "cylinder_lbm": {
        "name":        "Flow Past Cylinder — LBM Re=200",
        "category":    "CFD",
        "description": "Von Kármán vortex street, D2Q9 BGK, Zou-He BCs, cylinder bounce-back. Reports max velocity, density range, vortex cell count.",
        "params":      {"Re": 200, "steps": 4000, "save_every": 500,
                        "nx": 160, "ny": 64, "u_in": 0.05},
    },
    "cavity_lbm": {
        "name":        "Lid-Driven Cavity — LBM Re=100",
        "category":    "CFD",
        "description": "Classic 64×64 lid-driven cavity at Re=100.",
        "params":      {"Re": 100, "steps": 3000, "save_every": 500,
                        "nx": 64, "ny": 64, "u_in": 0.05},
    },
}


class BenchmarkListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response([{"key": k, **v} for k, v in BENCHMARK_DEFS.items()])


class BenchmarkRunView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        key    = request.data.get("benchmark_key")
        config = request.data.get("config", {})

        if key not in BENCHMARK_DEFS:
            return Response({"error": f"unknown benchmark: {key}"}, status=400)

        params = {**BENCHMARK_DEFS[key]["params"], **config}
        t0     = time.perf_counter()
        try:
            metrics = _run(key, params)
        except Exception as exc:
            return Response({"error": str(exc)}, status=500)
        elapsed = time.perf_counter() - t0
        metrics["time_s"] = round(elapsed, 3)

        result = BenchmarkResult.objects.create(
            owner=request.user, benchmark_key=key, config=params, metrics=metrics
        )
        return Response({
            "id":            str(result.id),
            "benchmark_key": key,
            "name":          BENCHMARK_DEFS[key]["name"],
            "config":        params,
            "metrics":       metrics,
            "created_at":    result.created_at.isoformat(),
        }, status=201)


class BenchmarkResultsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = BenchmarkResult.objects.filter(owner=request.user).order_by("-created_at")[:100]
        return Response(BenchmarkResultSerializer(qs, many=True).data)

    def delete(self, request):
        BenchmarkResult.objects.filter(owner=request.user).delete()
        return Response(status=204)


# ── Benchmark runners ─────────────────────────────────────────────────────────

def _run(key: str, params: dict) -> dict:
    if key == "poisson_fdm":    return _bench_poisson(params)
    if key == "burgers_pinn":   return _bench_burgers(params)
    if key in ("cylinder_lbm", "cavity_lbm"): return _bench_lbm(key, params)
    raise ValueError(f"No runner for {key}")


def _bench_poisson(p):
    nx    = int(p.get("nx", 64)); ny = int(p.get("ny", 64))
    iters = int(p.get("iters", 5000))
    x = np.linspace(0, 1, nx); y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = 1.0 / (nx - 1)
    f  = 2 * math.pi**2 * np.sin(math.pi * X) * np.sin(math.pi * Y)
    u  = np.zeros_like(f)
    for _ in range(iters):
        u[1:-1, 1:-1] = 0.25 * (
            u[:-2,1:-1] + u[2:,1:-1] + u[1:-1,:-2] + u[1:-1,2:]
            + dx**2 * f[1:-1, 1:-1]
        )
    u_exact = np.sin(math.pi * X) * np.sin(math.pi * Y)
    err = u - u_exact
    return {
        "l2_error":   float(np.sqrt(np.mean(err**2))),
        "linf_error": float(np.max(np.abs(err))),
        "grid":       f"{nx}x{ny}",
        "field":      u.tolist(),
        "exact":      u_exact.tolist(),
        "x":          x.tolist(),
        "y":          y.tolist(),
    }


def _bench_burgers(p):
    import torch
    import torch.nn as nn
    from api.problem_defs import get_problem, generate_collocation_points, build_pinn_loss

    prob     = {**get_problem("burgers_1d"), "_preset_key": "burgers_1d"}
    n_epochs = int(p.get("n_epochs", 500))
    hidden   = int(p.get("hidden", 64))
    n_layers = int(p.get("n_layers", 4))
    lr       = float(p.get("lr", 1e-3))
    n_col    = int(p.get("n_interior", 2000))

    layers = [nn.Linear(2, hidden), nn.Tanh()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers.append(nn.Linear(hidden, 1))
    model = nn.Sequential(*layers)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    col  = generate_collocation_points(prob, n_interior=n_col, n_boundary=200)
    pts  = torch.tensor(col["interior"], dtype=torch.float32)
    bnd  = torch.tensor(col["boundary"], dtype=torch.float32)
    rf   = build_pinn_loss(prob)

    history      = []
    report_every = max(1, n_epochs // 100)

    for epoch in range(1, n_epochs + 1):
        opt.zero_grad()
        loss = rf(model, pts) + model(bnd).pow(2).mean()
        loss.backward(); opt.step()
        if epoch % report_every == 0:
            history.append({"epoch": epoch, "loss": float(loss.item())})

    return {
        "final_loss": history[-1]["loss"] if history else float("nan"),
        "history":    history,
        "epochs":     n_epochs,
    }


def _bench_lbm(key: str, p: dict) -> dict:
    from pinneaple_solvers.lbm import LBMSolver, cylinder_mask

    nx         = int(p.get("nx",   160 if key == "cylinder_lbm" else 64))
    ny         = int(p.get("ny",   64))
    Re         = float(p.get("Re", 200 if key == "cylinder_lbm" else 100))
    u_in       = float(p.get("u_in", 0.05))
    steps      = int(p.get("steps", 4000))
    save_every = int(p.get("save_every", 500))

    obstacle = None
    if key == "cylinder_lbm":
        obstacle = cylinder_mask(nx, ny, cx=nx // 4, cy=ny // 2, r=ny // 8)

    solver = LBMSolver(nx=nx, ny=ny, Re=Re, u_in=u_in, obstacle_mask=obstacle)
    out    = solver.forward(steps=steps, save_every=save_every)
    e      = out.extras

    ux  = e["ux"].numpy(); uy = e["uy"].numpy()
    vm  = e["vel_mag"].numpy(); rho = e["rho"].numpy()
    obs = obstacle.numpy().astype(bool) if obstacle is not None else None

    vm_m = np.where(obs, np.nan, vm) if obs is not None else vm

    # Vorticity & Q-criterion
    dx = 1.0 / nx; dy = 1.0 / ny
    dux_dx = np.gradient(ux, dx, axis=0); duy_dy = np.gradient(uy, dy, axis=1)
    dux_dy = np.gradient(ux, dy, axis=1); duy_dx = np.gradient(uy, dx, axis=0)
    Q = -(dux_dx * duy_dy) - (duy_dx * dux_dy)
    if obs is not None: Q[obs] = 0.0  # frontend masks via obstacle array

    return {
        "max_vel":      float(np.nanmax(vm_m)),
        "min_rho":      float(np.nanmin(rho)),
        "max_rho":      float(np.nanmax(rho)),
        "vortex_cells": int(np.nansum(Q > 0)),
        "vel_mag":      vm.tolist(),
        "Q":            Q.tolist(),
        "vorticity":    (duy_dx - dux_dy).tolist(),
        "obstacle":     obs.tolist() if obs is not None else None,
        "trajectory_ux": [t.numpy().tolist() for t in e["trajectory_ux"]],
        "trajectory_uy": [t.numpy().tolist() for t in e["trajectory_uy"]],
        "nx": nx, "ny": ny, "Re": Re,
    }
