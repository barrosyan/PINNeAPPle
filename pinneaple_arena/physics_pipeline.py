"""PhysicsBenchmarkSpec — declarative physics/PINN benchmark pipeline.

Uses:
  - pinneaple_environment.presets  (get_preset, list_presets → ProblemSpec)
  - pinneaple_pinn                 (compile_problem, LossWeights → loss_fn)
  - pinneaple_models               (ModelRegistry.build → any registered model)
  - pinneaple_train.metrics        (MSE, MAE, RMSE, R2, RelL2, MaxError)

Any problem registered in pinneaple_environment can be used directly.
For custom ProblemSpec objects, compile_problem builds the loss automatically.
For the built-in fallback problems, a manual residual wrapper is provided.

Usage
-----
    from pinneaple_arena import PhysicsBenchmarkSpec

    spec = PhysicsBenchmarkSpec(
        problem  = "burgers_1d_default",   # any pinneaple_environment preset
        models   = ["vanilla_pinn", "siren", "modified_mlp"],
        metrics  = ["mse", "l2_rel"],
        epochs   = 3000,
        plots    = True,
    )
    report = spec.run()
    report.save("outputs/physics/burgers_report.json")
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .report import BenchmarkReport, ModelRunResult


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _unwrap(out: Any) -> torch.Tensor:
    """Extract tensor from ModelOutput / PINNOutput / plain tensor."""
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("y", "pred", "out", "recon", "x_hat", "y_hat"):
        v = getattr(out, attr, None)
        if isinstance(v, torch.Tensor):
            return v
    raise TypeError(f"Cannot extract tensor from {type(out)}")


def _pred(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return _unwrap(model(x))


def _l2_rel(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


# -----------------------------------------------------------------------------
# Metrics  (MetricBundle from pinneaple_train.metrics)
# -----------------------------------------------------------------------------

def _compute_metrics(pred: np.ndarray, true: np.ndarray,
                     requested: List[str]) -> Dict[str, float]:
    """Compute requested metrics using MetricBundle from pinneaple_train.metrics."""
    from pinneaple_train.metrics import (
        MSE, MAE, RMSE, R2, RelL2, MaxError, MetricBundle,
    )
    _map = {
        "mse":         MSE(),
        "rmse":        RMSE(),
        "mae":         MAE(),
        "r2":          R2(),
        "l2_rel":      RelL2(name="l2_rel"),
        "l2rel":       RelL2(name="l2rel"),
        "relative_l2": RelL2(name="relative_l2"),
        "max_err":     MaxError(name="max_err"),
        "linf":        MaxError(name="linf"),
        "max":         MaxError(name="max"),
    }
    metrics = [_map[m.lower()] for m in requested if m.lower() in _map]
    bundle  = MetricBundle(metrics=metrics)
    pred_t  = torch.tensor(pred.flatten(), dtype=torch.float32)
    true_t  = torch.tensor(true.flatten(), dtype=torch.float32)
    result  = bundle.compute(pred_t, true_t)
    # preserve the caller's requested key names (may differ from metric .name)
    return {m: result.get(m.lower(), result.get(_map[m.lower()].name, float("nan")))
            for m in requested if m.lower() in _map}


# -----------------------------------------------------------------------------
# Collocation sampling  (pinneaple_data.collocation.CollocationSampler)
# -----------------------------------------------------------------------------

def _col_sample_bounds(bounds_dict: Dict[str, Tuple[float, float]],
                        fields: Tuple[str, ...],
                        n_col: int, strategy: str = "lhs",
                        seed: int = 42) -> np.ndarray:
    """Sample interior collocation points from axis-aligned bounds."""
    from pinneaple_data.collocation import CollocationSampler
    sampler = CollocationSampler.from_bounds(bounds_dict, fields=fields,
                                              strategy=strategy, seed=seed)
    raw = sampler.sample(n_col=n_col, seed=seed)
    return raw["x_col"]


# -----------------------------------------------------------------------------
# Built-in fallback problems  (used when no pinneaple_environment preset found)
# -----------------------------------------------------------------------------

@dataclass
class _ProblemDef:
    id: str
    pde_str: str
    in_dim: int
    out_dim: int
    coord_names: List[str]
    field_names: List[str]
    domain: Dict[str, Tuple[float, float]]
    params: Dict[str, float]
    dataset_id: str
    has_time: bool


_BUILTIN_PROBLEMS: Dict[str, _ProblemDef] = {
    "burgers_1d": _ProblemDef(
        id="burgers_1d", pde_str="u_t + u*u_x = nu*u_xx",
        in_dim=2, out_dim=1, coord_names=["x", "t"], field_names=["u"],
        domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        params={"nu": 0.01 / math.pi}, dataset_id="burgers_1d", has_time=True,
    ),
    "heat_1d": _ProblemDef(
        id="heat_1d", pde_str="u_t = k*u_xx",
        in_dim=2, out_dim=1, coord_names=["x", "t"], field_names=["u"],
        domain={"x": (0.0, 1.0), "t": (0.0, 0.5)},
        params={"k": 0.4}, dataset_id="heat_1d", has_time=True,
    ),
    "heat_2d": _ProblemDef(
        id="heat_2d", pde_str="u_t = k*(u_xx + u_yy)",
        in_dim=3, out_dim=1, coord_names=["x", "y", "t"], field_names=["u"],
        domain={"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, 0.5)},
        params={"k": 0.1}, dataset_id="heat_2d", has_time=True,
    ),
    "poisson_2d": _ProblemDef(
        id="poisson_2d", pde_str="-Delta(u) = f",
        in_dim=2, out_dim=1, coord_names=["x", "y"], field_names=["u"],
        domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        params={}, dataset_id="poisson_2d", has_time=False,
    ),
    "wave_1d": _ProblemDef(
        id="wave_1d", pde_str="u_tt = c²*u_xx",
        in_dim=2, out_dim=1, coord_names=["x", "t"], field_names=["u"],
        domain={"x": (0.0, 1.0), "t": (0.0, 1.0)},
        params={"c": 1.0}, dataset_id="wave_1d", has_time=True,
    ),
    "kovasznay_ns": _ProblemDef(
        id="kovasznay_ns", pde_str="NS incompressible (Kovasznay)",
        in_dim=2, out_dim=3, coord_names=["x", "y"], field_names=["u", "v", "p"],
        domain={"x": (-0.5, 1.0), "y": (-0.5, 1.5)},
        params={"Re": 40.0}, dataset_id="kovasznay_ns", has_time=False,
    ),
    "helmholtz_2d": _ProblemDef(
        id="helmholtz_2d", pde_str="Delta(u) + k²u = q",
        in_dim=2, out_dim=1, coord_names=["x", "y"], field_names=["u"],
        domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        params={"k": 1.0, "a1": 1.0, "a2": 1.0}, dataset_id="helmholtz_2d", has_time=False,
    ),
    "allen_cahn_1d": _ProblemDef(
        id="allen_cahn_1d", pde_str="u_t - eps²u_xx + 5u³ - 5u = 0",
        in_dim=2, out_dim=1, coord_names=["x", "t"], field_names=["u"],
        domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        params={"eps": 0.01}, dataset_id="allen_cahn_1d", has_time=True,
    ),
}

_BUILTIN_ALIASES = {
    "burgers": "burgers_1d", "heat": "heat_1d",
    "poisson": "poisson_2d", "wave": "wave_1d",
    "kovasznay": "kovasznay_ns", "ns_2d": "kovasznay_ns",
    "helmholtz": "helmholtz_2d", "allen_cahn": "allen_cahn_1d",
}


def _physics_residual(prob: _ProblemDef, model: nn.Module,
                      pts: torch.Tensor,
                      inv_params: Optional[Dict[str, nn.Parameter]] = None
                      ) -> torch.Tensor:
    from pinneaple_pinn.compiler.autograd_ops import grad as _ag_grad
    u = _pred(model, pts)

    def _g(y, i=None):
        g = _ag_grad(y if y.dim() > 0 else y.unsqueeze(-1), pts)
        return g if i is None else g[:, i:i+1]

    pid = prob.id
    p = prob.params.copy()
    if inv_params:
        p.update({k: v for k, v in inv_params.items()})

    if pid.startswith("burgers"):
        nu = p.get("nu", 0.01 / math.pi)
        return _g(u, 1) + u * _g(u, 0) - nu * _g(_g(u, 0), 0)

    if pid.startswith("heat_1d"):
        k = p.get("k", 0.4)
        return _g(u, 1) - k * _g(_g(u, 0), 0)

    if pid.startswith("heat_2d"):
        k = p.get("k", 0.1)
        return _g(u, 2) - k * (_g(_g(u, 0), 0) + _g(_g(u, 1), 1))

    if pid.startswith("poisson"):
        f_pts = 2 * math.pi**2 * torch.sin(math.pi * pts[:, 0:1]) * torch.sin(math.pi * pts[:, 1:2])
        return -(_g(_g(u, 0), 0) + _g(_g(u, 1), 1)) - f_pts

    if pid.startswith("wave"):
        c = p.get("c", 1.0)
        return _g(_g(u, 1), 1) - c**2 * _g(_g(u, 0), 0)

    if pid == "kovasznay_ns":
        Re = p.get("Re", 40.0)
        u_, v_, pr = u[:, 0:1], u[:, 1:2], u[:, 2:3]
        u_x = _g(u_, 0); u_y = _g(u_, 1)
        v_x = _g(v_, 0); v_y = _g(v_, 1)
        p_x = _g(pr, 0); p_y = _g(pr, 1)
        cont = u_x + v_y
        mom_u = u_ * u_x + v_ * u_y + p_x - (_g(u_x, 0) + _g(u_y, 1)) / Re
        mom_v = u_ * v_x + v_ * v_y + p_y - (_g(v_x, 0) + _g(v_y, 1)) / Re
        return torch.cat([cont, mom_u, mom_v], dim=-1)

    if pid.startswith("helmholtz"):
        k = p.get("k", 1.0)
        a1, a2 = p.get("a1", 1.0), p.get("a2", 1.0)
        q = (-(a1**2 + a2**2) * math.pi**2 + k**2) * \
            torch.sin(a1 * math.pi * pts[:, 0:1]) * torch.sin(a2 * math.pi * pts[:, 1:2])
        return _g(_g(u, 0), 0) + _g(_g(u, 1), 1) + k**2 * u - q

    if pid.startswith("allen_cahn"):
        eps = p.get("eps", 0.01)
        return _g(u, 1) - eps**2 * _g(_g(u, 0), 0) + 5.0 * (u**3 - u)

    raise ValueError(f"No built-in residual for '{pid}'. "
                     "Use a pinneaple_environment preset instead.")


def _make_bc_ic_pts(prob: _ProblemDef, n_bc: int, n_ic: int,
                    seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor,
                                            Optional[torch.Tensor],
                                            Optional[torch.Tensor]]:
    rng = np.random.default_rng(seed)
    dom = prob.domain

    if prob.id.startswith("burgers") or prob.id.startswith("heat_1d") or prob.id.startswith("wave"):
        t_bc = rng.uniform(dom["t"][0], dom["t"][1], n_bc)
        x_bc = torch.tensor(
            np.vstack([np.column_stack([np.full(n_bc, dom["x"][0]), t_bc]),
                       np.column_stack([np.full(n_bc, dom["x"][1]), t_bc])]),
            dtype=torch.float32)
        u_bc = torch.zeros(2 * n_bc, prob.out_dim)
        x_ic_raw = rng.uniform(dom["x"][0], dom["x"][1], n_ic)
        x_ic = torch.tensor(np.column_stack([x_ic_raw, np.zeros(n_ic)]), dtype=torch.float32)
        u_ic_vals = (-np.sin(np.pi * x_ic_raw) if prob.id.startswith("burgers")
                     else np.sin(np.pi * x_ic_raw))
        u_ic = torch.tensor(u_ic_vals.reshape(-1, 1), dtype=torch.float32)

    elif prob.id.startswith("heat_2d"):
        pts_list = []
        for xv in [dom["x"][0], dom["x"][1]]:
            y_r = rng.uniform(dom["y"][0], dom["y"][1], n_bc // 4)
            pts_list.append(np.column_stack([np.full(n_bc//4, xv), y_r,
                                              rng.uniform(*dom["t"], n_bc//4)]))
        for yv in [dom["y"][0], dom["y"][1]]:
            x_r = rng.uniform(dom["x"][0], dom["x"][1], n_bc // 4)
            pts_list.append(np.column_stack([x_r, np.full(n_bc//4, yv),
                                              rng.uniform(*dom["t"], n_bc//4)]))
        x_bc = torch.tensor(np.vstack(pts_list), dtype=torch.float32)
        u_bc = torch.zeros(len(x_bc), 1)
        x_ic_r = rng.uniform(*dom["x"], n_ic)
        y_ic_r = rng.uniform(*dom["y"], n_ic)
        x_ic = torch.tensor(np.column_stack([x_ic_r, y_ic_r, np.zeros(n_ic)]), dtype=torch.float32)
        u_ic = torch.tensor(
            (np.sin(np.pi * x_ic_r) * np.sin(np.pi * y_ic_r)).reshape(-1, 1), dtype=torch.float32)

    elif prob.id in ("poisson_2d", "helmholtz_2d"):
        n_e = n_bc // 4
        pts_list = []
        for xv in [dom["x"][0], dom["x"][1]]:
            pts_list.append(np.column_stack([np.full(n_e, xv), rng.uniform(*dom["y"], n_e)]))
        for yv in [dom["y"][0], dom["y"][1]]:
            pts_list.append(np.column_stack([rng.uniform(*dom["x"], n_e), np.full(n_e, yv)]))
        x_bc = torch.tensor(np.vstack(pts_list), dtype=torch.float32)
        u_bc = torch.zeros(len(x_bc), 1)
        x_ic, u_ic = None, None

    elif prob.id == "kovasznay_ns":
        Re = prob.params.get("Re", 40.0)
        lam = Re / 2.0 - math.sqrt(Re**2 / 4.0 + 4.0 * math.pi**2)
        n_e = n_bc // 4
        pts_list, vals_list = [], []
        for xv in [dom["x"][0], dom["x"][1]]:
            yv = rng.uniform(*dom["y"], n_e)
            xr = np.full(n_e, xv)
            pts_list.append(np.column_stack([xr, yv]))
            vals_list.append(np.column_stack([
                1.0 - np.exp(lam*xr)*np.cos(2*math.pi*yv),
                lam/(2*math.pi)*np.exp(lam*xr)*np.sin(2*math.pi*yv),
                0.5*(1.0 - np.exp(2*lam*xr))]))
        for yv in [dom["y"][0], dom["y"][1]]:
            xr = rng.uniform(*dom["x"], n_e)
            yr = np.full(n_e, yv)
            pts_list.append(np.column_stack([xr, yr]))
            vals_list.append(np.column_stack([
                1.0 - np.exp(lam*xr)*np.cos(2*math.pi*yr),
                lam/(2*math.pi)*np.exp(lam*xr)*np.sin(2*math.pi*yr),
                0.5*(1.0 - np.exp(2*lam*xr))]))
        x_bc = torch.tensor(np.vstack(pts_list), dtype=torch.float32)
        u_bc = torch.tensor(np.vstack(vals_list), dtype=torch.float32)
        x_ic, u_ic = None, None

    else:
        try:
            from pinneaple_data.collocation import CollocationSampler
            sampler = CollocationSampler.from_bounds(
                dom, fields=tuple(prob.field_names), seed=seed)
            raw = sampler.sample(n_bc=n_bc, seed=seed)
            x_bc = torch.tensor(raw["x_bc"], dtype=torch.float32)
        except Exception:
            rng2 = np.random.default_rng(seed)
            lo = np.array([dom[c][0] for c in prob.coord_names])
            hi = np.array([dom[c][1] for c in prob.coord_names])
            x_bc = torch.tensor(
                rng2.uniform(lo, hi, (n_bc, prob.in_dim)).astype(np.float32))
        u_bc = torch.zeros(len(x_bc), prob.out_dim)
        x_ic, u_ic = None, None

    return x_bc, u_bc, x_ic, u_ic


# -----------------------------------------------------------------------------
# Batch builders
# -----------------------------------------------------------------------------

def _batch_from_builtin(prob: _ProblemDef, x_col_np: np.ndarray,
                        x_bc: torch.Tensor, u_bc: torch.Tensor,
                        x_ic: Optional[torch.Tensor],
                        u_ic: Optional[torch.Tensor],
                        x_obs: Optional[torch.Tensor] = None,
                        u_obs: Optional[torch.Tensor] = None,
                        ) -> Dict[str, Any]:
    batch: Dict[str, Any] = {
        "x_col": torch.tensor(x_col_np, dtype=torch.float32),
        "x_bc": x_bc,
        "y_bc": u_bc,
        "ctx": {},
    }
    if x_ic is not None and u_ic is not None:
        batch["x_ic"] = x_ic
        batch["y_ic"] = u_ic
    if x_obs is not None and u_obs is not None:
        batch["x_data"] = x_obs
        batch["y_data"] = u_obs
    return batch


def _raw_to_batch(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert CollocationSampler.sample() numpy output to a tensor batch dict."""
    batch: Dict[str, Any] = {
        "x_col": torch.tensor(np.asarray(raw["x_col"]), dtype=torch.float32),
        "ctx": raw.get("ctx", {}),
    }
    for key in ("x_bc", "y_bc", "x_ic", "y_ic"):
        val = raw.get(key)
        if val is not None and np.asarray(val).size > 0:
            batch[key] = torch.tensor(np.asarray(val), dtype=torch.float32)
    return batch


def _batch_from_spec(spec: Any, n_col: int, n_bc: int, n_ic: int,
                     strategy: str = "lhs", seed: int = 42) -> Dict[str, Any]:
    """Generate a training batch from a ProblemSpec via CollocationSampler."""
    from pinneaple_data.collocation import CollocationSampler
    sampler = CollocationSampler.from_problem_spec(spec, strategy=strategy, seed=seed)
    raw = sampler.sample(n_col=n_col, n_bc=n_bc, n_ic=n_ic, seed=seed)
    return _raw_to_batch(raw)


def _batch_from_geometry(geometry: Any, fields: List[str],
                          n_col: int, n_bc: int, n_ic: int,
                          strategy: str = "lhs", seed: int = 42) -> Dict[str, Any]:
    """Generate a training batch from an STL mesh via CollocationSampler.from_mesh."""
    from pinneaple_data.stl_import import load_stl, STLMesh
    from pinneaple_data.collocation import CollocationSampler

    if isinstance(geometry, str):
        mesh = load_stl(geometry)
    elif isinstance(geometry, STLMesh):
        mesh = geometry
    else:
        mesh = geometry  # already a compatible mesh object

    sampler = CollocationSampler.from_mesh(mesh, fields=tuple(fields),
                                            strategy=strategy, seed=seed)
    raw = sampler.sample(n_col=n_col, n_bc=n_bc, n_ic=n_ic, seed=seed)
    return _raw_to_batch(raw)


# -----------------------------------------------------------------------------
# Loss function builders
# -----------------------------------------------------------------------------

def _make_compiled_loss(spec: Any,
                        w_pde: float = 1.0, w_bc: float = 10.0,
                        w_ic: float = 10.0, w_data: float = 1.0
                        ) -> Callable:
    """Return compile_problem loss for a ProblemSpec (pinneaple_pinn)."""
    from pinneaple_pinn import compile_problem, LossWeights
    weights = LossWeights(w_pde=w_pde, w_bc=w_bc, w_ic=w_ic, w_data=w_data)
    return compile_problem(spec, weights=weights)


def _make_builtin_loss(prob: _ProblemDef,
                       w_pde: float = 1.0, w_bc: float = 10.0,
                       w_ic: float = 10.0, w_data: float = 20.0,
                       inv_params_getter: Optional[Callable] = None
                       ) -> Callable:
    """Return a loss_fn(model, y_hat, batch) for a built-in _ProblemDef."""

    def loss_fn(model: nn.Module, y_hat: Any,
                batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x_col = batch["x_col"]   # has requires_grad=True
        inv_p = inv_params_getter(model) if inv_params_getter else None

        res = _physics_residual(prob, model, x_col, inv_p)
        loss_pde = (res ** 2).mean()
        losses: Dict[str, torch.Tensor] = {"pde": loss_pde}
        total = w_pde * loss_pde

        if "x_bc" in batch and "y_bc" in batch:
            loss_bc = F.mse_loss(_pred(model, batch["x_bc"]), batch["y_bc"])
            losses["bc"] = loss_bc
            total = total + w_bc * loss_bc

        if "x_ic" in batch and "y_ic" in batch:
            loss_ic = F.mse_loss(_pred(model, batch["x_ic"]), batch["y_ic"])
            losses["ic"] = loss_ic
            total = total + w_ic * loss_ic

        if "x_data" in batch and "y_data" in batch:
            loss_data = F.mse_loss(_pred(model, batch["x_data"]), batch["y_data"])
            losses["data"] = loss_data
            total = total + w_data * loss_data

        losses["total"] = total
        return losses

    return loss_fn


# -----------------------------------------------------------------------------
# Model factory  (ModelRegistry — any registered model)
# -----------------------------------------------------------------------------

def _build_model(name: str, in_dim: int, out_dim: int,
                 hidden: List[int]) -> nn.Module:
    """Build any physics model via ModelRegistry / PINNCatalog / Group-B catalog."""
    from pinneaple_models import ModelRegistry, SIREN, ModifiedMLP, HashGridMLP
    from pinneaple_models.pinns import VanillaPINN, PINNCatalog

    name_l = name.lower().replace("-", "_").replace(" ", "_")
    h0 = hidden[0] if hidden else 64
    n_layers = len(hidden) + 1

    # Canonical shortcuts so users keep their short names
    _shortcuts = {
        "vanilla_pinn": lambda: VanillaPINN(in_dim=in_dim, out_dim=out_dim, hidden=hidden),
        "vanilla":      lambda: VanillaPINN(in_dim=in_dim, out_dim=out_dim, hidden=hidden),
        "siren":        lambda: SIREN(in_dim=in_dim, out_dim=out_dim,
                                      hidden_dim=h0, n_layers=n_layers),
        "modified_mlp": lambda: ModifiedMLP(in_dim=in_dim, out_dim=out_dim,
                                            hidden_dim=h0, n_layers=n_layers),
        "modmlp":       lambda: ModifiedMLP(in_dim=in_dim, out_dim=out_dim,
                                            hidden_dim=h0, n_layers=n_layers),
        "hash_grid":    lambda: HashGridMLP(in_dim=in_dim, out_dim=out_dim),
        "hash_grid_mlp":lambda: HashGridMLP(in_dim=in_dim, out_dim=out_dim),
    }
    if name_l in _shortcuts:
        return _shortcuts[name_l]()

    # Full ModelRegistry (75+ models)
    try:
        return ModelRegistry.build(name_l, in_dim=in_dim, out_dim=out_dim, hidden_dim=h0)
    except Exception:
        pass

    # PINNCatalog
    try:
        return PINNCatalog.build(name_l, in_dim=in_dim, out_dim=out_dim, hidden=hidden)
    except Exception:
        pass

    raise ValueError(
        f"Unknown model '{name}'. "
        "Any name registered in ModelRegistry or PINNCatalog is accepted.\n"
        f"Available families: {ModelRegistry.families()}"
    )


# -----------------------------------------------------------------------------
# Inverse wrapper
# -----------------------------------------------------------------------------

class _InverseWrapper(nn.Module):
    def __init__(self, base: nn.Module, var_names: List[str],
                 init_guesses: Dict[str, float]):
        super().__init__()
        self.base = base
        self.var_names = var_names
        self._log_params = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(math.log(max(float(init_guesses.get(k, 0.1)), 1e-9))))
            for k in var_names
        })

    def forward(self, x: torch.Tensor) -> Any:
        return self.base(x)

    @property
    def inv_params(self) -> Dict[str, torch.Tensor]:
        return {k: torch.exp(v) for k, v in self._log_params.items()}

    def param_estimates(self) -> Dict[str, float]:
        return {k: float(torch.exp(v).item()) for k, v in self._log_params.items()}


# -----------------------------------------------------------------------------
# Reference data
# -----------------------------------------------------------------------------

def _load_reference(dataset_id: str) -> Optional[Dict[str, np.ndarray]]:
    from pinneaple_data.datasets import load_dataset
    try:
        return load_dataset(dataset_id)
    except Exception:
        return None


def _make_eval_grid(coords: List[str], fields: List[str],
                    ref: Dict[str, np.ndarray]) -> Tuple[
        Optional[torch.Tensor], Optional[np.ndarray]]:
    arrays = []
    for c in coords:
        if c not in ref:
            return None, None
        arrays.append(ref[c])
    grid = np.meshgrid(*arrays, indexing="ij")
    pts = np.column_stack([g.flatten() for g in grid])
    x_eval = torch.tensor(pts, dtype=torch.float32)
    u_key = fields[0] if fields and fields[0] in ref else "u"
    if u_key not in ref:
        return x_eval, None
    return x_eval, ref[u_key].flatten()


# -----------------------------------------------------------------------------
# Unified training loop  (used for both compiled and builtin loss)
# -----------------------------------------------------------------------------

def _train_pinn(model: nn.Module,
                loss_fn: Callable,
                batch: Dict[str, Any],
                epochs: int, lr: float,
                log_every: int = 500,
                ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """PINN training loop.

    loss_fn signature: loss_fn(model, y_hat, batch) → Dict[str, Tensor] with 'total'.
    batch['x_col'] is the collocation tensor (detached); grad is re-enabled each step.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    x_col_base = batch["x_col"]
    history: List[Dict[str, float]] = []
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        # Fresh leaf with grad for autograd-based PDE residuals
        x_col = x_col_base.clone().detach().requires_grad_(True)
        b = {**batch, "x_col": x_col}

        y_hat = model(x_col)
        losses = loss_fn(model, y_hat, b)
        loss = losses["total"]

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            row: Dict[str, float] = {"epoch": epoch + 1,
                                     "loss_total": float(loss.detach())}
            for k, v in losses.items():
                if k != "total" and isinstance(v, torch.Tensor):
                    row[f"loss_{k}"] = float(v.detach())
            history.append(row)

    model.eval()
    return history[-1] if history else {}, history


# -----------------------------------------------------------------------------
# PhysicsBenchmarkSpec
# -----------------------------------------------------------------------------

class PhysicsBenchmarkSpec:
    """Declarative physics benchmark pipeline.

    Parameters
    ----------
    problem : str or ProblemSpec
        - Name of a pinneaple_environment preset (e.g. "burgers_1d_default").
        - Built-in alias (e.g. "burgers_1d", "heat_1d", "kovasznay_ns").
        - A ProblemSpec object from pinneaple_environment directly.
    models : list of str
        Any model name registered in ModelRegistry or PINNCatalog
        (e.g. "vanilla_pinn", "siren", "modified_mlp", "pinn_lstm",
        "deeponet", "fno", "hamiltonian_nn", …).
    metrics : list of str
        "mse", "rmse", "mae", "l2_rel", "max_err", "r2".
    collocation_points : str
        "sobol" (default), "lhs", "halton", "uniform".
    inverse : bool
        Enable inverse problem mode.
    inverse_variables : list of str
        Parameter names to identify.
    epochs : int
        Training epochs.
    lr : float
        Initial learning rate.
    n_col, n_bc, n_ic : int
        Number of collocation / BC / IC points.
    hidden : list of int
        Hidden layer widths passed to model constructors.
    seed : int
        Random seed.
    output_dir : str
        Directory for plots and JSON reports.
    """

    def __init__(
        self,
        problem: Union[str, Any],
        geometry: Optional[Any] = None,
        load_generate_data: str = "generate",
        source: Optional[str] = None,
        metrics: Sequence[str] = ("mse", "l2_rel", "max_err"),
        collocation_points: str = "sobol",
        models: Sequence[str] = ("vanilla_pinn",),
        inverse: bool = False,
        inverse_variables: Sequence[str] = (),
        plots: bool = True,
        epochs: int = 3000,
        lr: float = 1e-3,
        n_col: int = 3000,
        n_bc: int = 500,
        n_ic: int = 500,
        hidden: Optional[List[int]] = None,
        seed: int = 42,
        output_dir: str = "outputs",
    ):
        self.problem = problem
        self.geometry = geometry
        self.load_generate_data = load_generate_data
        self.source = source
        self.metrics = list(metrics)
        self.collocation_points = collocation_points
        self.models = list(models)
        self.inverse = inverse
        self.inverse_variables = list(inverse_variables)
        self.plots = plots
        self.epochs = epochs
        self.lr = lr
        self.n_col = n_col
        self.n_bc = n_bc
        self.n_ic = n_ic
        self.hidden = hidden if hidden is not None else [64, 64, 64, 64]
        self.seed = seed
        self.output_dir = Path(output_dir)

    # -------------------------------------------------------------------------
    # Problem resolution: preset → compile_problem; fallback → builtin residual
    # -------------------------------------------------------------------------

    def _resolve_problem(self) -> Tuple[Any, str]:
        """Returns (problem_obj, kind) where kind is 'preset' or 'builtin'."""
        prob = self.problem

        # Already a ProblemSpec (duck-type: has .coords and .conditions)
        if hasattr(prob, "coords") and hasattr(prob, "conditions"):
            return prob, "preset"

        if isinstance(prob, str):
            name = prob.strip()

            # 1. Try pinneaple_environment preset registry
            try:
                from pinneaple_environment.presets.registry import get_preset
                spec = get_preset(name)
                return spec, "preset"
            except Exception:
                pass

            # 2. Try built-in problems (with aliases)
            key = _BUILTIN_ALIASES.get(name.lower(), name.lower())
            if key in _BUILTIN_PROBLEMS:
                return _BUILTIN_PROBLEMS[key], "builtin"

            raise ValueError(
                f"Unknown problem '{name}'. "
                "Use a pinneaple_environment preset name or one of: "
                f"{list(_BUILTIN_PROBLEMS.keys())}"
            )

        # Already a _ProblemDef
        if isinstance(prob, _ProblemDef):
            return prob, "builtin"

        raise ValueError(f"Cannot parse problem: {type(prob)}")

    # -------------------------------------------------------------------------
    # Observation data (for inverse / data-informed)
    # -------------------------------------------------------------------------

    def _get_obs_data(self, coords: List[str], fields: List[str],
                      ref: Optional[Dict[str, np.ndarray]]
                      ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.load_generate_data != "load" and not self.inverse:
            return None, None
        if ref is None:
            return None, None
        x_eval, u_true = _make_eval_grid(coords, fields, ref)
        if x_eval is None or u_true is None:
            return None, None
        n = min(500, len(x_eval))
        idx = np.random.default_rng(self.seed).choice(len(x_eval), n, replace=False)
        x_obs = x_eval[idx]
        out_dim = len(fields)
        u_obs = torch.tensor(u_true[idx].reshape(-1, out_dim), dtype=torch.float32)
        return x_obs, u_obs

    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------

    def _plot_results(self, model: nn.Module,
                      coords: List[str], fields: List[str],
                      ref: Optional[Dict[str, np.ndarray]],
                      model_id: str,
                      history: List[Dict]) -> List[str]:
        paths: List[str] = []
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return paths

        self.output_dir.mkdir(parents=True, exist_ok=True)
        prob_name = getattr(self, "_current_prob_id", "problem")
        prefix = self.output_dir / f"physics_{prob_name}_{model_id}"

        # Loss curve
        fig, ax = plt.subplots(figsize=(7, 4))
        ep = [r["epoch"] for r in history]
        total_h = [r.get("loss_total", float("nan")) for r in history]
        ax.semilogy(ep, total_h, label="total", lw=2)
        for key in ("loss_pde", "loss_bc"):
            vals = [r.get(key, float("nan")) for r in history]
            if any(not math.isnan(v) for v in vals):
                ax.semilogy(ep, vals, label=key.split("_", 1)[1], lw=1.5, ls="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(f"{prob_name} | {model_id} — Training Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = str(prefix) + "_loss.png"
        fig.savefig(p, dpi=100, bbox_inches="tight"); plt.close(fig)
        paths.append(p)

        # Prediction heatmap (2-coord problems only)
        if ref is not None and len(coords) == 2:
            x_eval, u_true = _make_eval_grid(coords, fields, ref)
            if x_eval is not None and u_true is not None:
                with torch.no_grad():
                    u_pred_all = _pred(model, x_eval).cpu().numpy()
                u_pred = u_pred_all[:, 0].flatten() if u_pred_all.ndim == 2 else u_pred_all.flatten()
                c0, c1 = coords[0], coords[1]
                if c0 in ref and c1 in ref:
                    n0, n1 = len(ref[c0]), len(ref[c1])
                else:
                    n = int(math.sqrt(len(u_true))); n0 = n1 = n
                try:
                    shape = (n0, n1)
                    u_pg = u_pred[:np.prod(shape)].reshape(shape)
                    u_tg = u_true[:np.prod(shape)].reshape(shape)
                    arr0 = ref.get(c0, np.linspace(0, 1, n0))
                    arr1 = ref.get(c1, np.linspace(0, 1, n1))
                    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                    kw = dict(cmap="viridis", shading="auto", aspect="auto")
                    for ax, data, title in zip(axes, [u_tg, u_pg, np.abs(u_pg - u_tg)],
                                               ["Reference", "Prediction",
                                                f"|Error| L2={_l2_rel(u_pred, u_true):.4f}"]):
                        cm = "Oranges" if "Error" in title else "viridis"
                        im = ax.pcolormesh(arr1, arr0, data, cmap=cm, shading="auto", aspect="auto")
                        ax.set_title(title); plt.colorbar(im, ax=ax)
                        ax.set_xlabel(c1); ax.set_ylabel(c0)
                    fig.suptitle(f"{prob_name} | {model_id}", fontsize=11)
                    fig.tight_layout()
                    p2 = str(prefix) + "_pred.png"
                    fig.savefig(p2, dpi=100, bbox_inches="tight"); plt.close(fig)
                    paths.append(p2)
                except Exception:
                    pass

        return paths

    # -------------------------------------------------------------------------
    # run()
    # -------------------------------------------------------------------------

    def run(self) -> BenchmarkReport:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        report = BenchmarkReport(
            benchmark_type="physics",
            created_at=BenchmarkReport.now_timestamp(),
        )

        # 1. Resolve problem
        prob_obj, prob_kind = self._resolve_problem()

        # Extract metadata depending on kind
        if prob_kind == "preset":
            spec = prob_obj
            prob_id = getattr(spec, "problem_id", str(spec))
            pde_str = str(getattr(getattr(spec, "pde", None), "kind", ""))
            coords = list(spec.coords)
            fields = list(spec.fields)
            dom = spec.domain_bounds
            in_dim = len(coords)
            out_dim = len(fields)
        else:
            prob_def: _ProblemDef = prob_obj
            prob_id = prob_def.id
            pde_str = prob_def.pde_str
            coords = prob_def.coord_names
            fields = prob_def.field_names
            dom = prob_def.domain
            in_dim = prob_def.in_dim
            out_dim = prob_def.out_dim

        self._current_prob_id = prob_id

        report.problem_info = {
            "id": prob_id, "pde": pde_str,
            "in_dim": in_dim, "out_dim": out_dim,
            "kind": prob_kind,
            "domain": {k: list(v) for k, v in dom.items()},
        }
        report.config = {
            "collocation_points": self.collocation_points,
            "n_col": self.n_col, "n_bc": self.n_bc, "n_ic": self.n_ic,
            "epochs": self.epochs, "lr": self.lr,
            "metrics": self.metrics, "models": self.models,
            "inverse": self.inverse,
        }

        # 2. Build training batch
        if self.geometry is not None:
            # Geometry/STL-based domain: CollocationSampler.from_mesh()
            batch = _batch_from_geometry(
                self.geometry, fields,
                self.n_col, self.n_bc, self.n_ic,
                self.collocation_points, self.seed)
            loss_fn_base = _make_compiled_loss(spec) if prob_kind == "preset" else None
        elif prob_kind == "preset":
            batch = _batch_from_spec(spec, self.n_col, self.n_bc, self.n_ic,
                                     self.collocation_points, self.seed)
            loss_fn_base = _make_compiled_loss(spec)
        else:
            # Builtin problem: CollocationSampler.from_bounds() for interior points
            try:
                from pinneaple_data.collocation import CollocationSampler
                bounds_dict = {c: dom[c] for c in coords}
                sampler = CollocationSampler.from_bounds(
                    bounds_dict, fields=tuple(fields),
                    strategy=self.collocation_points, seed=self.seed)
                raw = sampler.sample(n_col=self.n_col, seed=self.seed)
                x_col_np = raw["x_col"]
            except Exception:
                rng = np.random.default_rng(self.seed)
                lo = np.array([dom[c][0] for c in coords])
                hi = np.array([dom[c][1] for c in coords])
                x_col_np = rng.uniform(lo, hi, (self.n_col, in_dim)).astype(np.float32)
            x_bc, u_bc, x_ic, u_ic = _make_bc_ic_pts(prob_def, self.n_bc, self.n_ic, self.seed)
            # Load obs for inverse/data-informed
            ref_data = _load_reference(prob_def.dataset_id)
            x_obs, u_obs = self._get_obs_data(coords, fields, ref_data)
            batch = _batch_from_builtin(prob_def, x_col_np, x_bc, u_bc,
                                        x_ic, u_ic, x_obs, u_obs)
            loss_fn_base = None  # built per model (may wrap inv_params)

        # 3. Reference data for metrics/plots
        ref_data_eval = _load_reference(prob_id)
        x_eval, u_true = (None, None)
        if ref_data_eval is not None:
            x_eval, u_true = _make_eval_grid(coords, fields, ref_data_eval)

        all_plots: List[str] = []
        print(f"\n{'-'*60}")
        print(f"  PhysicsBenchmarkSpec  ->  {prob_id}  [{prob_kind}]")
        print(f"  Models: {self.models}")
        print(f"  Epochs: {self.epochs}  |  n_col: {self.n_col}  |  seed: {self.seed}")
        print(f"{'-'*60}")

        # 4. Train each model
        for model_name in self.models:
            print(f"\n  > Model: {model_name}")
            t_start = time.time()
            try:
                base_model = _build_model(model_name, in_dim, out_dim, self.hidden)
                n_params = sum(p.numel() for p in base_model.parameters())

                # Wrap for inverse problems (builtin only)
                if self.inverse and self.inverse_variables and prob_kind == "builtin":
                    init_g = {v: prob_def.params.get(v, 0.1) * 0.5
                              for v in self.inverse_variables}
                    model = _InverseWrapper(base_model, self.inverse_variables, init_g)
                    n_params += len(self.inverse_variables)
                    loss_fn = _make_builtin_loss(
                        prob_def,
                        inv_params_getter=lambda m: m.inv_params
                        if isinstance(m, _InverseWrapper) else None
                    )
                elif prob_kind == "builtin":
                    model = base_model
                    loss_fn = _make_builtin_loss(prob_def)
                else:
                    model = base_model
                    loss_fn = loss_fn_base

                print(f"    params = {n_params:,}")

                final_losses, history = _train_pinn(
                    model, loss_fn, batch,
                    epochs=self.epochs, lr=self.lr,
                    log_every=max(1, self.epochs // 6),
                )

                elapsed = time.time() - t_start

                # Evaluate metrics
                if x_eval is not None and u_true is not None:
                    with torch.no_grad():
                        u_pred_all = _pred(model, x_eval).cpu().numpy()
                        u_pred_np = (u_pred_all[:, 0].flatten()
                                     if u_pred_all.ndim == 2 else u_pred_all.flatten())
                    metrics_out = _compute_metrics(u_pred_np, u_true, self.metrics)
                else:
                    metrics_out = {"loss_pde": final_losses.get("loss_pde", float("nan"))}

                param_est = None
                if self.inverse and isinstance(model, _InverseWrapper):
                    param_est = model.param_estimates()
                    for k, v in param_est.items():
                        true_v = prob_def.params.get(k, float("nan"))
                        metrics_out[f"param_{k}_err_pct"] = (
                            abs(v - true_v) / (abs(true_v) + 1e-12) * 100)
                    print(f"    identified: {param_est}")

                print(f"    metrics: {metrics_out}")
                print(f"    time: {elapsed:.1f}s")

                report.model_results[model_name] = ModelRunResult(
                    model_id=model_name, n_params=n_params,
                    training_time_s=elapsed, metrics=metrics_out,
                    history=history, param_estimates=param_est,
                )

                if self.plots:
                    plots = self._plot_results(model, coords, fields,
                                               ref_data_eval, model_name, history)
                    all_plots.extend(plots)

            except Exception as exc:
                elapsed = time.time() - t_start
                print(f"    ERROR: {exc}")
                import traceback; traceback.print_exc()
                report.model_results[model_name] = ModelRunResult(
                    model_id=model_name, n_params=0,
                    training_time_s=elapsed, metrics={},
                    history=[], error_message=str(exc),
                )

        # 5. Leaderboard
        primary = self.metrics[0] if self.metrics else "mse"
        scored = [
            (mid, r.metrics.get(primary, float("inf")))
            for mid, r in report.model_results.items()
            if not r.error_message
        ]
        scored.sort(key=lambda x: x[1])
        report.leaderboard = [
            {"rank": i+1, "model": mid, primary: score}
            for i, (mid, score) in enumerate(scored)
        ]
        for i, (mid, _) in enumerate(scored):
            report.model_results[mid].rank = i + 1
        report.best_model = scored[0][0] if scored else None
        report.plots_saved = all_plots

        report.print_summary()
        return report
