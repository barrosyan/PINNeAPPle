"""PhysicsBenchmarkSpec — declarative physics/PINN benchmark pipeline.

Usage
-----
    from pinneaple_arena import PhysicsBenchmarkSpec

    spec = PhysicsBenchmarkSpec(
        problem        = "burgers_1d",
        load_generate_data = "generate",
        source         = "analytical",
        metrics        = ["mse", "l2_rel", "max_err"],
        collocation_points = "sobol",
        models         = ["vanilla_pinn", "siren", "modified_mlp"],
        inverse        = False,
        inverse_variables = [],
        plots          = True,
    )
    report = spec.run()
    report.print_summary()
    report.save("outputs/my_benchmark.json")
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .report import BenchmarkReport, ModelRunResult


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _pred(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, torch.Tensor):
        return out
    return getattr(out, "y", out)


def _grad1(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y.sum(), x, create_graph=True, retain_graph=True)[0]


def _l2_rel(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def _compute_metrics(pred: np.ndarray, true: np.ndarray,
                     requested: List[str]) -> Dict[str, float]:
    diff = pred - true
    out: Dict[str, float] = {}
    for m in requested:
        m_lower = m.lower()
        if m_lower == "mse":
            out[m] = float(np.mean(diff**2))
        elif m_lower == "rmse":
            out[m] = float(np.sqrt(np.mean(diff**2)))
        elif m_lower == "mae":
            out[m] = float(np.mean(np.abs(diff)))
        elif m_lower in ("l2_rel", "l2rel", "relative_l2"):
            out[m] = _l2_rel(pred, true)
        elif m_lower in ("max_err", "linf", "max"):
            out[m] = float(np.max(np.abs(diff)))
        elif m_lower == "r2":
            ss_res = np.sum(diff**2)
            ss_tot = np.sum((true - true.mean())**2) + 1e-12
            out[m] = float(1.0 - ss_res / ss_tot)
        else:
            out[m] = float("nan")
    return out


# -----------------------------------------------------------------------------
# Collocation sampling strategies
# -----------------------------------------------------------------------------

def _sample_points(n: int, bounds: List[Tuple[float, float]],
                   strategy: str = "sobol", seed: int = 42) -> np.ndarray:
    """Sample n points in the hyper-rectangle defined by bounds.

    Returns ndarray of shape (n, len(bounds)).
    """
    d = len(bounds)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    if strategy in ("sobol", "quasi"):
        try:
            engine = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
            pts = engine.draw(n).numpy()
        except Exception:
            pts = np.random.default_rng(seed).random((n, d))
    elif strategy in ("lhs", "latin_hypercube"):
        try:
            from scipy.stats.qmc import LatinHypercube
            sampler = LatinHypercube(d=d, seed=seed)
            pts = sampler.random(n)
        except ImportError:
            pts = np.random.default_rng(seed).random((n, d))
    elif strategy == "halton":
        try:
            from scipy.stats.qmc import Halton
            sampler = Halton(d=d, scramble=True, seed=seed)
            pts = sampler.random(n)
        except ImportError:
            pts = np.random.default_rng(seed).random((n, d))
    else:   # "uniform" or fallback
        pts = np.random.default_rng(seed).random((n, d))

    return lo + (hi - lo) * pts


# -----------------------------------------------------------------------------
# Built-in problem definitions
# -----------------------------------------------------------------------------

@dataclass
class _ProblemDef:
    id: str
    pde_str: str
    in_dim: int
    out_dim: int
    coord_names: List[str]
    field_names: List[str]
    domain: Dict[str, Tuple[float, float]]   # coord -> (lo, hi)
    params: Dict[str, float]
    dataset_id: str
    has_time: bool


def _make_bc_ic_pts(prob: _ProblemDef, n_bc: int, n_ic: int,
                    seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor,
                                            Optional[torch.Tensor],
                                            Optional[torch.Tensor]]:
    """Generate (x_bc, u_bc, x_ic, u_ic) tensors for common problem types."""
    rng = np.random.default_rng(seed)
    coords = prob.coord_names
    dom = prob.domain

    # -- boundary ------------------------------------------------------------
    if prob.id.startswith("burgers") or prob.id.startswith("heat_1d") or prob.id.startswith("wave"):
        # x-boundaries: u=0 for all t
        t_bc = rng.uniform(dom["t"][0], dom["t"][1], n_bc)
        x_lo = np.full(n_bc, dom["x"][0])
        x_hi = np.full(n_bc, dom["x"][1])
        pts_lo = np.column_stack([x_lo, t_bc])
        pts_hi = np.column_stack([x_hi, t_bc])
        x_bc = torch.tensor(np.vstack([pts_lo, pts_hi]), dtype=torch.float32)
        u_bc = torch.zeros(2 * n_bc, prob.out_dim)

        # IC
        x_ic_raw = rng.uniform(dom["x"][0], dom["x"][1], n_ic)
        t_ic = np.zeros(n_ic)
        x_ic = torch.tensor(np.column_stack([x_ic_raw, t_ic]), dtype=torch.float32)
        if prob.id.startswith("burgers") or prob.id.startswith("allen"):
            u_ic_vals = -np.sin(np.pi * x_ic_raw)
        elif prob.id.startswith("heat_1d"):
            u_ic_vals = np.sin(np.pi * x_ic_raw)
        elif prob.id.startswith("wave"):
            u_ic_vals = np.sin(np.pi * x_ic_raw)
        else:
            u_ic_vals = np.zeros(n_ic)
        u_ic = torch.tensor(u_ic_vals.reshape(-1, 1), dtype=torch.float32)

    elif prob.id.startswith("heat_2d"):
        # All 4 edges: u=0
        t_bc = rng.uniform(dom["t"][0], dom["t"][1], n_bc // 4)
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
        # IC: u(x,y,0) = sin(πx)sin(πy)
        x_ic_r = rng.uniform(*dom["x"], n_ic)
        y_ic_r = rng.uniform(*dom["y"], n_ic)
        x_ic = torch.tensor(
            np.column_stack([x_ic_r, y_ic_r, np.zeros(n_ic)]), dtype=torch.float32
        )
        u_ic = torch.tensor(
            (np.sin(np.pi * x_ic_r) * np.sin(np.pi * y_ic_r)).reshape(-1, 1),
            dtype=torch.float32,
        )

    elif prob.id in ("poisson_2d", "helmholtz_2d"):
        # Dirichlet u=0 on all 4 edges
        n_e = n_bc // 4
        pts_list = []
        for xv in [dom["x"][0], dom["x"][1]]:
            yv = rng.uniform(*dom["y"], n_e)
            pts_list.append(np.column_stack([np.full(n_e, xv), yv]))
        for yv in [dom["y"][0], dom["y"][1]]:
            xv = rng.uniform(*dom["x"], n_e)
            pts_list.append(np.column_stack([xv, np.full(n_e, yv)]))
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
            u_val = 1.0 - np.exp(lam*xr)*np.cos(2*math.pi*yv)
            v_val = lam/(2*math.pi)*np.exp(lam*xr)*np.sin(2*math.pi*yv)
            p_val = 0.5*(1.0 - np.exp(2*lam*xr))
            pts_list.append(np.column_stack([xr, yv]))
            vals_list.append(np.column_stack([u_val, v_val, p_val]))
        for yv in [dom["y"][0], dom["y"][1]]:
            xr = rng.uniform(*dom["x"], n_e)
            yr = np.full(n_e, yv)
            u_val = 1.0 - np.exp(lam*xr)*np.cos(2*math.pi*yr)
            v_val = lam/(2*math.pi)*np.exp(lam*xr)*np.sin(2*math.pi*yr)
            p_val = 0.5*(1.0 - np.exp(2*lam*xr))
            pts_list.append(np.column_stack([xr, yr]))
            vals_list.append(np.column_stack([u_val, v_val, p_val]))
        x_bc = torch.tensor(np.vstack(pts_list), dtype=torch.float32)
        u_bc = torch.tensor(np.vstack(vals_list), dtype=torch.float32)
        x_ic, u_ic = None, None

    else:
        # Generic fallback: zero BC on boundaries
        bounds_list = [dom[c] for c in coords]
        x_bc = torch.tensor(
            _sample_points(n_bc, bounds_list, "uniform", seed), dtype=torch.float32
        )
        u_bc = torch.zeros(n_bc, prob.out_dim)
        x_ic, u_ic = None, None

    return x_bc, u_bc, x_ic, u_ic


def _physics_residual(prob: _ProblemDef, model: nn.Module,
                      pts: torch.Tensor,
                      inv_params: Optional[Dict[str, nn.Parameter]] = None
                      ) -> torch.Tensor:
    """Compute PDE residual at interior collocation points."""
    u = _pred(model, pts)

    def _g(y, i=None):
        g = _grad1(y if y.dim() > 0 else y.unsqueeze(-1), pts)
        return g if i is None else g[:, i:i+1]

    pid = prob.id
    p = prob.params.copy()
    if inv_params:
        for k, v in inv_params.items():
            p[k] = v   # replace with learnable tensor

    if pid.startswith("burgers"):
        nu = p.get("nu", 0.01 / math.pi)
        u_x = _g(u, 0)
        u_t = _g(u, 1)
        u_xx = _g(u_x, 0)
        return u_t + u * u_x - nu * u_xx

    elif pid.startswith("heat_1d"):
        k = p.get("k", 0.4)
        u_x = _g(u, 0)
        u_t = _g(u, 1)
        u_xx = _g(u_x, 0)
        return u_t - k * u_xx

    elif pid.startswith("heat_2d"):
        k = p.get("k", 0.1)
        u_x = _g(u, 0); u_y = _g(u, 1); u_t = _g(u, 2)
        u_xx = _g(u_x, 0); u_yy = _g(u_y, 1)
        return u_t - k * (u_xx + u_yy)

    elif pid.startswith("poisson"):
        u_x = _g(u, 0); u_y = _g(u, 1)
        u_xx = _g(u_x, 0); u_yy = _g(u_y, 1)
        f = p.get("f", 2.0 * math.pi**2)    # forcing magnitude for sin*sin
        # full forcing: 2π2sin(πx)sin(πy)
        f_pts = 2 * math.pi**2 * torch.sin(math.pi * pts[:, 0:1]) * torch.sin(math.pi * pts[:, 1:2])
        return -(u_xx + u_yy) - f_pts

    elif pid.startswith("wave"):
        c = p.get("c", 1.0)
        u_x = _g(u, 0)
        u_t = _g(u, 1)
        u_xx = _g(u_x, 0)
        u_tt = _g(u_t, 1)
        return u_tt - c**2 * u_xx

    elif pid == "kovasznay_ns":
        Re = p.get("Re", 40.0)
        u_ = u[:, 0:1]; v_ = u[:, 1:2]; pr = u[:, 2:3]
        u_x = _g(u_, 0); u_y = _g(u_, 1)
        v_x = _g(v_, 0); v_y = _g(v_, 1)
        p_x = _g(pr, 0); p_y = _g(pr, 1)
        u_xx = _g(u_x, 0); u_yy = _g(u_y, 1)
        v_xx = _g(v_x, 0); v_yy = _g(v_y, 1)
        cont = u_x + v_y
        mom_u = u_ * u_x + v_ * u_y + p_x - (u_xx + u_yy) / Re
        mom_v = u_ * v_x + v_ * v_y + p_y - (v_xx + v_yy) / Re
        return torch.cat([cont, mom_u, mom_v], dim=-1)

    elif pid.startswith("helmholtz"):
        k = p.get("k", 1.0)
        u_x = _g(u, 0); u_y = _g(u, 1)
        u_xx = _g(u_x, 0); u_yy = _g(u_y, 1)
        q_pts = (-(p.get("a1", 1.0)**2 + p.get("a2", 1.0)**2)*math.pi**2 + k**2) * \
                torch.sin(p.get("a1", 1.0)*math.pi*pts[:, 0:1]) * \
                torch.sin(p.get("a2", 1.0)*math.pi*pts[:, 1:2])
        return u_xx + u_yy + k**2 * u - q_pts

    elif pid.startswith("allen_cahn"):
        eps = p.get("eps", 0.01)
        u_x = _g(u, 0); u_t = _g(u, 1)
        u_xx = _g(u_x, 0)
        return u_t - eps**2 * u_xx + 5.0*(u**3 - u)

    else:
        raise ValueError(
            f"No built-in residual for problem '{pid}'. "
            "Provide a custom ProblemSpec and compile_problem."
        )


# -----------------------------------------------------------------------------
# Built-in problem registry
# -----------------------------------------------------------------------------

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
        id="poisson_2d", pde_str="-Deltau = 2π2sin(πx)sin(πy)",
        in_dim=2, out_dim=1, coord_names=["x", "y"], field_names=["u"],
        domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        params={}, dataset_id="poisson_2d", has_time=False,
    ),
    "wave_1d": _ProblemDef(
        id="wave_1d", pde_str="u_tt = c2*u_xx",
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
        id="helmholtz_2d", pde_str="Deltau + k2u = q",
        in_dim=2, out_dim=1, coord_names=["x", "y"], field_names=["u"],
        domain={"x": (0.0, 1.0), "y": (0.0, 1.0)},
        params={"k": 1.0, "a1": 1.0, "a2": 1.0}, dataset_id="helmholtz_2d", has_time=False,
    ),
    "allen_cahn_1d": _ProblemDef(
        id="allen_cahn_1d", pde_str="u_t − eps2u_xx + 5u3 − 5u = 0",
        in_dim=2, out_dim=1, coord_names=["x", "t"], field_names=["u"],
        domain={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
        params={"eps": 0.01}, dataset_id="allen_cahn_1d", has_time=True,
    ),
}


def _resolve_aliases(name: str) -> str:
    aliases = {
        "burgers": "burgers_1d",
        "heat": "heat_1d",
        "poisson": "poisson_2d",
        "wave": "wave_1d",
        "kovasznay": "kovasznay_ns",
        "ns_2d": "kovasznay_ns",
        "helmholtz": "helmholtz_2d",
        "allen_cahn": "allen_cahn_1d",
    }
    return aliases.get(name.lower(), name.lower())


# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------

def _build_model(name: str, in_dim: int, out_dim: int,
                 hidden: List[int]) -> nn.Module:
    name_l = name.lower().replace("-", "_").replace(" ", "_")

    if name_l in ("vanilla_pinn", "vanilla", "mlp"):
        try:
            from pinneaple_models.pinns.vanilla import VanillaPINN
            return VanillaPINN(in_dim=in_dim, out_dim=out_dim, hidden=hidden)
        except ImportError:
            pass
        # fallback: plain MLP
        layers: List[nn.Module] = []
        dims = [in_dim] + hidden + [out_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.Tanh())
        return nn.Sequential(*layers)

    if name_l == "siren":
        try:
            from pinneaple_models.siren import SIREN
            return SIREN(in_dim=in_dim, out_dim=out_dim,
                         hidden_dim=hidden[0] if hidden else 128,
                         n_layers=len(hidden) + 1)
        except ImportError:
            pass

    if name_l in ("modified_mlp", "modmlp", "mod_mlp"):
        try:
            from pinneaple_models.modified_mlp import ModifiedMLP
            return ModifiedMLP(in_dim=in_dim, out_dim=out_dim,
                               hidden_dim=hidden[0] if hidden else 128,
                               n_layers=len(hidden) + 1)
        except ImportError:
            pass

    if name_l in ("hash_grid", "hash_grid_mlp", "hashgrid"):
        try:
            from pinneaple_models.hash_grid import HashGridMLP
            return HashGridMLP(in_dim=in_dim, out_dim=out_dim)
        except ImportError:
            pass

    # Try ModelRegistry as last resort
    try:
        from pinneaple_models.registry import ModelRegistry
        return ModelRegistry.build(name_l, in_dim=in_dim, out_dim=out_dim,
                                   hidden_dim=hidden[0] if hidden else 64)
    except Exception:
        pass

    raise ValueError(
        f"Could not build model '{name}'. "
        "Supported: vanilla_pinn, siren, modified_mlp, hash_grid."
    )


# -----------------------------------------------------------------------------
# Inverse wrapper
# -----------------------------------------------------------------------------

class _InverseWrapper(nn.Module):
    """Wraps a model and adds log-parameterized learnable scalars."""

    def __init__(self, base: nn.Module, var_names: List[str],
                 init_guesses: Dict[str, float]):
        super().__init__()
        self.base = base
        self.var_names = var_names
        self._log_params = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(math.log(float(init_guesses.get(k, 0.1)))))
            for k in var_names
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _pred(self.base, x)

    @property
    def inv_params(self) -> Dict[str, torch.Tensor]:
        return {k: torch.exp(v) for k, v in self._log_params.items()}

    def param_estimates(self) -> Dict[str, float]:
        return {k: float(torch.exp(v).item()) for k, v in self._log_params.items()}


# -----------------------------------------------------------------------------
# Reference data loading
# -----------------------------------------------------------------------------

def _load_reference(prob: _ProblemDef, source: str,
                    load_generate: str) -> Optional[Dict[str, np.ndarray]]:
    """Try to load or generate reference data for evaluation."""
    if load_generate == "load" or (source and source not in
                                   ("generate", "sobol", "lhs", "uniform",
                                    "halton", "adaptive", None)):
        # Try as dataset ID
        try:
            from pinneaple_data.datasets import load_dataset
            return load_dataset(prob.dataset_id)
        except Exception:
            pass
        # Try as file path
        try:
            p = Path(source)
            if p.exists():
                if p.suffix == ".npz":
                    return dict(np.load(p))
                if p.suffix == ".npy":
                    return {"u": np.load(p)}
        except Exception:
            pass

    # Auto-load from datasets registry by problem ID
    try:
        from pinneaple_data.datasets import load_dataset
        return load_dataset(prob.dataset_id)
    except Exception:
        return None


def _make_eval_grid(prob: _ProblemDef, ref: Dict[str, np.ndarray]) -> Tuple[
        torch.Tensor, np.ndarray]:
    """Build flat (N, in_dim) eval grid and matching u_true from reference."""
    coords = prob.coord_names
    fields = prob.field_names

    arrays = []
    for c in coords:
        if c in ref:
            arrays.append(ref[c])
        else:
            return None, None

    # Build meshgrid
    grid = np.meshgrid(*arrays, indexing="ij")
    pts = np.column_stack([g.flatten() for g in grid])
    x_eval = torch.tensor(pts, dtype=torch.float32)

    # Reference field (first field)
    u_key = fields[0] if fields[0] in ref else "u"
    if u_key not in ref:
        return x_eval, None
    u_ref = ref[u_key].flatten()
    return x_eval, u_ref


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------

def _train(model: nn.Module,
           prob: _ProblemDef,
           x_col_np: np.ndarray,
           x_bc: torch.Tensor, u_bc: torch.Tensor,
           x_ic: Optional[torch.Tensor], u_ic: Optional[torch.Tensor],
           x_obs: Optional[torch.Tensor], u_obs: Optional[torch.Tensor],
           epochs: int, lr: float,
           w_pde: float = 1.0, w_bc: float = 10.0,
           w_ic: float = 10.0, w_data: float = 20.0,
           log_every: int = 500,
           inverse: bool = False,
           ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """Main PINN training loop. Returns (final_losses, history)."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history: List[Dict[str, float]] = []
    inv_params: Optional[Dict] = None
    if inverse and isinstance(model, _InverseWrapper):
        inv_params = model.inv_params

    x_col_t = torch.tensor(x_col_np, dtype=torch.float32)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Resample collocation points (or use fixed)
        x_col = x_col_t.clone().detach().requires_grad_(True)

        # PDE residual
        if inv_params and isinstance(model, _InverseWrapper):
            inv_params = model.inv_params
        try:
            res = _physics_residual(prob, model, x_col, inv_params)
            loss_pde = (res ** 2).mean()
        except Exception as e:
            loss_pde = torch.tensor(0.0, requires_grad=True)

        # BC
        u_bc_pred = _pred(model, x_bc)
        loss_bc = F.mse_loss(u_bc_pred, u_bc)

        # IC
        loss_ic = torch.tensor(0.0)
        if x_ic is not None and u_ic is not None:
            u_ic_pred = _pred(model, x_ic)
            loss_ic = F.mse_loss(u_ic_pred, u_ic)

        # Data (for inverse or data-informed)
        loss_data = torch.tensor(0.0)
        if x_obs is not None and u_obs is not None:
            u_obs_pred = _pred(model, x_obs)
            loss_data = F.mse_loss(u_obs_pred, u_obs)

        loss = w_pde * loss_pde + w_bc * loss_bc + w_ic * loss_ic
        if x_obs is not None:
            loss = loss + w_data * loss_data

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % log_every == 0 or epoch == 0:
            row = {
                "epoch": epoch + 1,
                "loss_total": float(loss.detach()),
                "loss_pde": float(loss_pde.detach()),
                "loss_bc": float(loss_bc.detach()),
                "loss_ic": float(loss_ic.detach()) if isinstance(loss_ic, torch.Tensor) else 0.0,
            }
            if x_obs is not None:
                row["loss_data"] = float(loss_data.detach())
            if inv_params and isinstance(model, _InverseWrapper):
                for k, v in model.param_estimates().items():
                    row[f"param_{k}"] = v
            history.append(row)

    model.eval()
    final: Dict[str, float] = history[-1] if history else {}
    return final, history


# -----------------------------------------------------------------------------
# PhysicsBenchmarkSpec
# -----------------------------------------------------------------------------

class PhysicsBenchmarkSpec:
    """Declarative physics benchmark pipeline.

    Parameters
    ----------
    problem : str or ProblemSpec
        Built-in ID (e.g. "burgers_1d") or a ProblemSpec object.
    geometry : optional
        MeshData or path to STL/STEP file.  Not required for 1D/2D grid problems.
    load_generate_data : "generate" | "load"
        "generate" — sample collocation points + use datasets for evaluation.
        "load"     — load external data as observations (data-informed PINN).
    source : str
        When "load": dataset ID (e.g. "burgers_1d") or file path.
        When "generate": ignored (collocation strategy comes from collocation_points).
    metrics : list of str
        Metrics to compute: "mse", "rmse", "mae", "l2_rel", "max_err", "r2".
    collocation_points : str
        Sampling strategy: "sobol", "lhs", "halton", "uniform".
    models : list of str
        Model names: "vanilla_pinn", "siren", "modified_mlp", "hash_grid".
    inverse : bool
        If True, add learnable parameters for each variable in inverse_variables.
    inverse_variables : list of str
        Variable names to identify (e.g. ["nu", "k"]).  Requires load != "generate".
    plots : bool
        Save prediction and error plots to output_dir.
    epochs : int
        Training epochs per model.
    lr : float
        Initial learning rate.
    n_col : int
        Number of interior collocation points.
    n_bc : int
        Number of boundary condition points.
    n_ic : int
        Number of initial condition points.
    hidden : list of int
        Hidden layer widths (shared across all models).
    seed : int
        Random seed for reproducibility.
    output_dir : str
        Directory for plots and report JSON.
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
        hidden: List[int] = None,
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

    # -- internal helpers -----------------------------------------------------

    def _resolve_problem(self) -> _ProblemDef:
        if isinstance(self.problem, _ProblemDef):
            return self.problem
        if isinstance(self.problem, str):
            key = _resolve_aliases(self.problem)
            if key in _BUILTIN_PROBLEMS:
                return _BUILTIN_PROBLEMS[key]
            # Try pinneaple_environment preset
            try:
                from pinneaple_environment.presets.registry import get_preset
                spec = get_preset(key)
                dom = {c: spec.domain_bounds.get(c, (0.0, 1.0))
                       for c in spec.coords}
                return _ProblemDef(
                    id=spec.problem_id, pde_str=spec.pde.kind,
                    in_dim=len(spec.coords), out_dim=len(spec.fields),
                    coord_names=list(spec.coords), field_names=list(spec.fields),
                    domain=dom, params=dict(spec.pde.params or {}),
                    dataset_id=spec.problem_id, has_time="t" in spec.coords,
                )
            except Exception:
                pass
            raise ValueError(
                f"Unknown problem '{self.problem}'. "
                f"Built-in IDs: {list(_BUILTIN_PROBLEMS.keys())}"
            )
        # Assume it's already a ProblemSpec-like object
        try:
            dom = {c: self.problem.domain_bounds.get(c, (0.0, 1.0))
                   for c in self.problem.coords}
            return _ProblemDef(
                id=self.problem.problem_id, pde_str=str(self.problem.pde.kind),
                in_dim=len(self.problem.coords), out_dim=len(self.problem.fields),
                coord_names=list(self.problem.coords),
                field_names=list(self.problem.fields),
                domain=dom, params=dict(self.problem.pde.params or {}),
                dataset_id=self.problem.problem_id,
                has_time="t" in self.problem.coords,
            )
        except Exception as e:
            raise ValueError(f"Cannot parse problem spec: {e}") from e

    def _get_obs_data(self, prob: _ProblemDef,
                      ref: Optional[Dict[str, np.ndarray]]
                      ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return observation tensors for data-informed / inverse training."""
        if self.load_generate_data != "load" and not self.inverse:
            return None, None
        if ref is None:
            return None, None
        x_eval, u_true = _make_eval_grid(prob, ref)
        if x_eval is None or u_true is None:
            return None, None
        # Subsample to at most 500 observations
        n = min(500, len(x_eval))
        idx = np.random.default_rng(self.seed).choice(len(x_eval), n, replace=False)
        x_obs = x_eval[idx]
        u_obs = torch.tensor(u_true[idx].reshape(-1, prob.out_dim), dtype=torch.float32)
        return x_obs, u_obs

    def _plot_results(self, model: nn.Module, prob: _ProblemDef,
                      ref: Optional[Dict[str, np.ndarray]],
                      model_id: str, history: List[Dict]) -> List[str]:
        paths: List[str] = []
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return paths

        self.output_dir.mkdir(parents=True, exist_ok=True)
        prefix = self.output_dir / f"physics_{prob.id}_{model_id}"

        # Loss curve
        fig, ax = plt.subplots(figsize=(7, 4))
        epochs_h = [r["epoch"] for r in history]
        total_h = [r.get("loss_total", float("nan")) for r in history]
        pde_h = [r.get("loss_pde", float("nan")) for r in history]
        ax.semilogy(epochs_h, total_h, label="total", lw=2)
        ax.semilogy(epochs_h, pde_h, label="PDE", lw=1.5, ls="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title(f"{prob.id} | {model_id} ? Training Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p = str(prefix) + "_loss.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

        # Prediction vs reference (2D heatmap for 2-coord problems)
        if ref is not None and prob.in_dim == 2:
            x_eval, u_true = _make_eval_grid(prob, ref)
            if x_eval is not None and u_true is not None:
                with torch.no_grad():
                    u_pred_all = _pred(model, x_eval).cpu().numpy()
                    u_pred = u_pred_all[:, 0].flatten() if u_pred_all.ndim == 2 else u_pred_all.flatten()

                # Determine grid shape
                c0, c1 = prob.coord_names[0], prob.coord_names[1]
                if c0 in ref and c1 in ref:
                    n0, n1 = len(ref[c0]), len(ref[c1])
                    shape = (n0, n1)
                else:
                    n = int(math.sqrt(len(u_true)))
                    shape = (n, n)

                try:
                    u_pred_g = u_pred[:np.prod(shape)].reshape(shape)
                    u_true_g = u_true[:np.prod(shape)].reshape(shape)
                    arr_c0 = ref.get(c0, np.linspace(0, 1, shape[0]))
                    arr_c1 = ref.get(c1, np.linspace(0, 1, shape[1]))

                    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                    kw = dict(cmap="viridis", shading="auto", aspect="auto")
                    im0 = axes[0].pcolormesh(arr_c1, arr_c0, u_true_g, **kw)
                    axes[0].set_title("Reference"); plt.colorbar(im0, ax=axes[0])
                    im1 = axes[1].pcolormesh(arr_c1, arr_c0, u_pred_g, **kw)
                    axes[1].set_title("Prediction"); plt.colorbar(im1, ax=axes[1])
                    err = np.abs(u_pred_g - u_true_g)
                    im2 = axes[2].pcolormesh(arr_c1, arr_c0, err, cmap="Oranges", shading="auto", aspect="auto")
                    axes[2].set_title(f"|Error|  L2={_l2_rel(u_pred, u_true):.4f}")
                    plt.colorbar(im2, ax=axes[2])
                    for ax in axes:
                        ax.set_xlabel(c1); ax.set_ylabel(c0)
                    fig.suptitle(f"{prob.id} | {model_id}", fontsize=11)
                    fig.tight_layout()
                    p2 = str(prefix) + "_pred.png"
                    fig.savefig(p2, dpi=100, bbox_inches="tight")
                    plt.close(fig)
                    paths.append(p2)
                except Exception:
                    pass

        return paths

    # -- public API -----------------------------------------------------------

    def run(self) -> BenchmarkReport:
        """Execute the full benchmark pipeline and return a BenchmarkReport."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        report = BenchmarkReport(
            benchmark_type="physics",
            created_at=BenchmarkReport.now_timestamp(),
        )

        # 1. Resolve problem
        prob = self._resolve_problem()
        report.problem_info = {
            "id": prob.id,
            "pde": prob.pde_str,
            "in_dim": prob.in_dim,
            "out_dim": prob.out_dim,
            "domain": {k: list(v) for k, v in prob.domain.items()},
            "params": {k: float(v) for k, v in prob.params.items()},
        }
        report.config = {
            "load_generate_data": self.load_generate_data,
            "source": self.source,
            "collocation_points": self.collocation_points,
            "n_col": self.n_col, "n_bc": self.n_bc, "n_ic": self.n_ic,
            "epochs": self.epochs, "lr": self.lr,
            "metrics": self.metrics,
            "models": self.models,
            "inverse": self.inverse,
            "inverse_variables": self.inverse_variables,
            "hidden": self.hidden,
            "seed": self.seed,
        }

        # 2. Generate collocation points
        bounds_list = [prob.domain[c] for c in prob.coord_names]
        x_col_np = _sample_points(self.n_col, bounds_list,
                                   self.collocation_points, self.seed)
        x_bc, u_bc, x_ic, u_ic = _make_bc_ic_pts(prob, self.n_bc, self.n_ic, self.seed)

        # 3. Load reference / observation data
        ref_data = _load_reference(prob, self.source or "", self.load_generate_data)
        x_obs, u_obs = self._get_obs_data(prob, ref_data)

        # 4. Prepare evaluation grid
        x_eval, u_true = (None, None)
        if ref_data is not None:
            x_eval, u_true = _make_eval_grid(prob, ref_data)

        all_plots: List[str] = []
        print(f"\n{'-'*60}")
        print(f"  PhysicsBenchmarkSpec  ->  {prob.id}")
        print(f"  Models: {self.models}")
        print(f"  Epochs: {self.epochs}  |  n_col: {self.n_col}  |  seed: {self.seed}")
        print(f"{'-'*60}")

        # 5. Train each model
        for model_name in self.models:
            print(f"\n  > Model: {model_name}")
            t_start = time.time()
            try:
                base_model = _build_model(model_name, prob.in_dim, prob.out_dim,
                                          self.hidden)
                n_params = sum(p.numel() for p in base_model.parameters())

                if self.inverse and self.inverse_variables:
                    # Wrap with learnable inverse parameters
                    init_g = {v: prob.params.get(v, 0.1) * 0.5
                              for v in self.inverse_variables}
                    model = _InverseWrapper(base_model, self.inverse_variables, init_g)
                    n_params += len(self.inverse_variables)
                else:
                    model = base_model

                print(f"    params = {n_params:,}")

                final_losses, history = _train(
                    model, prob, x_col_np, x_bc, u_bc, x_ic, u_ic,
                    x_obs, u_obs,
                    epochs=self.epochs, lr=self.lr,
                    log_every=max(1, self.epochs // 6),
                    inverse=self.inverse,
                )

                elapsed = time.time() - t_start

                # Evaluate metrics
                metrics_out: Dict[str, float] = {}
                if x_eval is not None and u_true is not None:
                    with torch.no_grad():
                        u_pred_all = _pred(model, x_eval).cpu().numpy()
                        # u_true corresponds to first output field only
                        u_pred_np = u_pred_all[:, 0].flatten() if u_pred_all.ndim == 2 else u_pred_all.flatten()
                    metrics_out = _compute_metrics(u_pred_np, u_true, self.metrics)
                else:
                    metrics_out = {"loss_pde": final_losses.get("loss_pde", float("nan"))}

                param_est = None
                if self.inverse and isinstance(model, _InverseWrapper):
                    param_est = model.param_estimates()
                    for k, v in param_est.items():
                        true_v = prob.params.get(k, float("nan"))
                        err_pct = abs(v - true_v) / (abs(true_v) + 1e-12) * 100
                        metrics_out[f"param_{k}_err_pct"] = err_pct
                    print(f"    params identified: {param_est}")

                print(f"    metrics: {metrics_out}")
                print(f"    time: {elapsed:.1f}s")

                result = ModelRunResult(
                    model_id=model_name,
                    n_params=n_params,
                    training_time_s=elapsed,
                    metrics=metrics_out,
                    history=history,
                    param_estimates=param_est,
                )
                report.model_results[model_name] = result

                # Plots
                if self.plots:
                    plots = self._plot_results(model, prob, ref_data,
                                               model_name, history)
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

        # 6. Build leaderboard
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
