"""PINN Arena — multi-architecture, multi-problem benchmark suite.

Usage
-----
>>> from pinneaple_tools.benchmark_suite.benchmark import PINNArenaBenchmark
>>> bench = PINNArenaBenchmark.default()
>>> results = bench.run(verbose=True)
>>> print(bench.leaderboard())
>>> bench.save_results("results.json")
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR


# ─────────────────────────────────────────────────────────────────────────────
# Config & result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    n_col: int = 4000
    n_bc: int = 800
    n_ic: int = 800
    epochs: int = 5000
    lr: float = 1e-3
    weight_pde: float = 1.0
    weight_bc: float = 10.0
    weight_ic: float = 10.0
    device: str = "auto"
    seed: int = 42
    n_eval: int = 10000
    convergence_threshold: float = 1e-3
    log_interval: int = 500


@dataclass
class ModelSpec:
    name: str
    factory: Callable[[int, int], nn.Module]
    description: str = ""


@dataclass
class BenchmarkResult:
    problem_id: str
    model_id: str
    metrics: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)
    elapsed_s: float = 0.0
    n_params: int = 0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "model_id": self.model_id,
            "metrics": self.metrics,
            "elapsed_s": self.elapsed_s,
            "n_params": self.n_params,
            "rank": self.rank,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark task base
# ─────────────────────────────────────────────────────────────────────────────

class BenchmarkTaskBase:
    """Abstract base for physics benchmark tasks."""
    task_id: str = "base"
    in_dim: int = 2
    out_dim: int = 1

    def sample_collocation(self, n: int, seed: int) -> np.ndarray:
        raise NotImplementedError

    def sample_boundary(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def sample_ic(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        return np.empty((0, self.in_dim)), np.empty((0, self.out_dim))

    def pde_residual(self, model: nn.Module, X: "torch.Tensor") -> "torch.Tensor":
        raise NotImplementedError

    def eval_grid(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (X_eval, U_exact) as numpy arrays."""
        raise NotImplementedError

    def evaluate(self, model: nn.Module, device: "torch.device") -> Dict[str, float]:
        X_eval, U_exact = self.eval_grid(self.n_eval if hasattr(self, 'n_eval') else 10000)
        X_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
        U_ref = torch.tensor(U_exact, dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            U_pred = model(X_t)
            if hasattr(U_pred, "y"):
                U_pred = U_pred.y

        # Ensure shape match
        if U_pred.shape != U_ref.shape:
            if U_pred.ndim == 1:
                U_pred = U_pred.unsqueeze(-1)
            if U_ref.ndim == 1:
                U_ref = U_ref.unsqueeze(-1)

        diff = U_pred - U_ref
        rel_l2 = float((diff.pow(2).sum() / (U_ref.pow(2).sum() + 1e-10)).sqrt().item())
        l_inf = float(diff.abs().max().item())
        mse = float(diff.pow(2).mean().item())

        # PDE residual on eval grid
        X_req = X_t.detach().requires_grad_(True)
        model.train()
        try:
            pde_res = float(self.pde_residual(model, X_req).item())
        except Exception:
            pde_res = float("nan")
        model.eval()

        return {
            "rel_l2": rel_l2,
            "l_inf": l_inf,
            "mse": mse,
            "pde_residual": pde_res,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Inline model architectures
# ─────────────────────────────────────────────────────────────────────────────

def _act_fn(name: str) -> nn.Module:
    return {"tanh": nn.Tanh(), "relu": nn.ReLU(), "silu": nn.SiLU(), "gelu": nn.GELU()}.get(
        name.lower(), nn.Tanh()
    )


def _build_mlp(in_dim: int, out_dim: int, hidden: Sequence[int], activation: str = "tanh") -> nn.Sequential:
    dims = [in_dim, *list(hidden), out_dim]
    layers: List[nn.Module] = []
    act = _act_fn(activation)
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(type(act)())
    return nn.Sequential(*layers)


class _ResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.fc2(torch.tanh(self.fc1(x))))


class _ResNetPINN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 128, n_blocks: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList([_ResBlock(width) for _ in range(n_blocks)])
        self.decoder = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.encoder(x))
        for blk in self.blocks:
            h = blk(h)
        return self.decoder(h)


class _FourierFeaturePINN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_fourier: int = 128,
        sigma: float = 1.0,
        hidden: Sequence[int] = (128, 128, 128, 128),
    ) -> None:
        super().__init__()
        B = torch.randn(in_dim, n_fourier) * sigma
        self.register_buffer("B", B)
        mlp_in = 2 * n_fourier
        self.mlp = _build_mlp(mlp_in, out_dim, hidden, activation="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B
        feats = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return self.mlp(feats)


class _SIRENLayer(nn.Module):
    def __init__(self, in_f: int, out_f: int, is_first: bool, omega_0: float) -> None:
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)
        self.omega_0 = omega_0
        w = 1.0 / in_f if is_first else math.sqrt(6.0 / in_f) / omega_0
        nn.init.uniform_(self.fc.weight, -w, w)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.fc(x))


class _SIREN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Sequence[int] = (128, 128, 128, 128),
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        dims = [in_dim, *list(hidden)]
        layers: List[nn.Module] = []
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            layers.append(_SIRENLayer(a, b, is_first=(i == 0), omega_0=omega_0))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Default model specs
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODELS: List[ModelSpec] = [
    ModelSpec(
        name="VanillaPINN_S",
        factory=lambda i, o: _build_mlp(i, o, [64, 64, 64, 64]),
        description="MLP 4x64 tanh",
    ),
    ModelSpec(
        name="VanillaPINN_M",
        factory=lambda i, o: _build_mlp(i, o, [128, 128, 128, 128, 128, 128]),
        description="MLP 6x128 tanh",
    ),
    ModelSpec(
        name="VanillaPINN_L",
        factory=lambda i, o: _build_mlp(i, o, [256, 256, 256, 256, 256, 256, 256, 256]),
        description="MLP 8x256 tanh",
    ),
    ModelSpec(
        name="ResNetPINN",
        factory=lambda i, o: _ResNetPINN(i, o, width=128, n_blocks=4),
        description="ResNet 4 blocks x128",
    ),
    ModelSpec(
        name="FourierPINN",
        factory=lambda i, o: _FourierFeaturePINN(i, o, n_fourier=128, sigma=1.0),
        description="Random Fourier features + MLP 4x128",
    ),
    ModelSpec(
        name="SIREN",
        factory=lambda i, o: _SIREN(i, o, hidden=[128, 128, 128, 128], omega_0=30.0),
        description="SIREN 4x128 omega=30",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Problem adapters  (wrap _ProblemDef / ProblemSpec into BenchmarkTaskBase)
# ─────────────────────────────────────────────────────────────────────────────

class _BuiltinProblemTask(BenchmarkTaskBase):
    """Adapt a physics_pipeline._ProblemDef to BenchmarkTaskBase."""

    def __init__(self, prob_def: Any) -> None:
        self._p = prob_def
        self.task_id = prob_def.id
        self.in_dim  = prob_def.in_dim
        self.out_dim = prob_def.out_dim

    def sample_collocation(self, n: int, seed: int) -> np.ndarray:
        try:
            from pinneaple_data.collocation import CollocationSampler
            dom = self._p.domain
            bounds = {c: dom[c] for c in self._p.coord_names}
            sampler = CollocationSampler.from_bounds(
                bounds, fields=tuple(self._p.field_names), seed=seed)
            return sampler.sample(n_col=n, seed=seed)["x_col"]
        except Exception:
            rng = np.random.default_rng(seed)
            dom = self._p.domain
            lo = np.array([dom[c][0] for c in self._p.coord_names])
            hi = np.array([dom[c][1] for c in self._p.coord_names])
            return rng.uniform(lo, hi, (n, self.in_dim)).astype(np.float32)

    def sample_boundary(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        from pinneaple_tools.benchmark_suite.physics_pipeline import _make_bc_ic_pts
        x_bc, u_bc, _, _ = _make_bc_ic_pts(self._p, n, 0, seed)
        return x_bc.numpy(), u_bc.numpy()

    def sample_ic(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._p.has_time:
            return np.empty((0, self.in_dim)), np.empty((0, self.out_dim))
        from pinneaple_tools.benchmark_suite.physics_pipeline import _make_bc_ic_pts
        _, _, x_ic, u_ic = _make_bc_ic_pts(self._p, 0, n, seed)
        if x_ic is None:
            return np.empty((0, self.in_dim)), np.empty((0, self.out_dim))
        return x_ic.numpy(), u_ic.numpy()

    def pde_residual(self, model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        from pinneaple_tools.benchmark_suite.physics_pipeline import _physics_residual
        res = _physics_residual(self._p, model, X)
        return (res ** 2).mean()

    def eval_grid(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        from pinneaple_tools.benchmark_suite.physics_pipeline import (
            _load_reference, _make_eval_grid,
        )
        ref = _load_reference(self._p.dataset_id)
        if ref is None:
            # fallback: uniform grid, no ground truth
            rng = np.random.default_rng(0)
            dom = self._p.domain
            lo = np.array([dom[c][0] for c in self._p.coord_names])
            hi = np.array([dom[c][1] for c in self._p.coord_names])
            X = rng.uniform(lo, hi, (n, self.in_dim)).astype(np.float32)
            return X, np.zeros((n, self.out_dim), dtype=np.float32)
        x_eval, u_true = _make_eval_grid(
            self._p.coord_names, self._p.field_names, ref)
        if x_eval is None or u_true is None:
            return np.zeros((1, self.in_dim), dtype=np.float32), np.zeros((1,), dtype=np.float32)
        return x_eval.numpy(), u_true.astype(np.float32)


class _PresetProblemTask(BenchmarkTaskBase):
    """Adapt a pinneaple_environment ProblemSpec to BenchmarkTaskBase."""

    def __init__(self, spec: Any, prob_id: str) -> None:
        self._spec   = spec
        self.task_id = prob_id
        self.in_dim  = len(spec.coords)
        self.out_dim = len(spec.fields)
        self._has_time = "t" in spec.coords or "time" in spec.coords
        self._loss_fn: Any = None  # compiled lazily

    def _get_loss_fn(self) -> Any:
        if self._loss_fn is None:
            from pinneaple_physics.pinn_solver import compile_problem, LossWeights
            self._loss_fn = compile_problem(
                self._spec, weights=LossWeights(w_pde=1.0, w_bc=10.0))
        return self._loss_fn

    def _sample_batch(self, n_col: int, n_bc: int, seed: int) -> Dict[str, Any]:
        try:
            from pinneaple_data.collocation import CollocationSampler
            sampler = CollocationSampler.from_problem_spec(
                self._spec, strategy="lhs", seed=seed)
            raw = sampler.sample(n_col=n_col, n_bc=n_bc, n_ic=0, seed=seed)
            return {
                "x_col": torch.tensor(np.asarray(raw["x_col"]), dtype=torch.float32),
                "x_bc":  torch.tensor(np.asarray(raw.get("x_bc", np.zeros((0, self.in_dim)))),
                                       dtype=torch.float32),
                "y_bc":  torch.tensor(np.asarray(raw.get("y_bc", np.zeros((0, self.out_dim)))),
                                       dtype=torch.float32),
                "ctx":   raw.get("ctx", {}),
            }
        except Exception:
            dom = self._spec.domain_bounds
            coords = list(self._spec.coords)
            rng = np.random.default_rng(seed)
            lo = np.array([dom[c][0] for c in coords])
            hi = np.array([dom[c][1] for c in coords])
            x = rng.uniform(lo, hi, (n_col, self.in_dim)).astype(np.float32)
            return {
                "x_col": torch.tensor(x),
                "x_bc":  torch.zeros(0, self.in_dim),
                "y_bc":  torch.zeros(0, self.out_dim),
                "ctx":   {},
            }

    def sample_collocation(self, n: int, seed: int) -> np.ndarray:
        return self._sample_batch(n, 0, seed)["x_col"].numpy()

    def sample_boundary(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        b = self._sample_batch(0, n, seed)
        return b["x_bc"].numpy(), b["y_bc"].numpy()

    def sample_ic(self, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._has_time:
            return np.empty((0, self.in_dim)), np.empty((0, self.out_dim))
        try:
            from pinneaple_data.collocation import CollocationSampler
            sampler = CollocationSampler.from_problem_spec(
                self._spec, strategy="lhs", seed=seed)
            raw = sampler.sample(n_col=0, n_bc=0, n_ic=n, seed=seed)
            x_ic = np.asarray(raw.get("x_ic", np.empty((0, self.in_dim))))
            y_ic = np.asarray(raw.get("y_ic", np.empty((0, self.out_dim))))
            return x_ic.astype(np.float32), y_ic.astype(np.float32)
        except Exception:
            return np.empty((0, self.in_dim)), np.empty((0, self.out_dim))

    def pde_residual(self, model: nn.Module, X: torch.Tensor) -> torch.Tensor:
        try:
            loss_fn = self._get_loss_fn()
            batch = {
                "x_col": X,
                "x_bc":  torch.zeros(1, self.in_dim, device=X.device),
                "y_bc":  torch.zeros(1, self.out_dim, device=X.device),
                "ctx":   {},
            }
            losses = loss_fn(model, None, batch)
            pde = losses.get("pde", losses.get("total", torch.tensor(float("nan"))))
            return pde if isinstance(pde, torch.Tensor) else torch.tensor(float(pde))
        except Exception:
            return torch.tensor(float("nan"))

    def eval_grid(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        # Try pinneaple_data reference first
        try:
            from pinneaple_tools.benchmark_suite.physics_pipeline import _load_reference, _make_eval_grid
            ref = _load_reference(self.task_id)
            if ref is not None:
                coords = list(self._spec.coords)
                fields = list(self._spec.fields)
                x_eval, u_true = _make_eval_grid(coords, fields, ref)
                if x_eval is not None and u_true is not None:
                    return x_eval.numpy(), u_true.astype(np.float32)
        except Exception:
            pass
        # Fallback: uniform grid, zeros (metrics computed vs loss only)
        dom = self._spec.domain_bounds
        coords = list(self._spec.coords)
        rng = np.random.default_rng(0)
        lo = np.array([dom[c][0] for c in coords])
        hi = np.array([dom[c][1] for c in coords])
        X = rng.uniform(lo, hi, (min(n, 1000), self.in_dim)).astype(np.float32)
        return X, np.zeros((len(X),), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic model catalogue  (all pinneaple_models registered architectures)
# ─────────────────────────────────────────────────────────────────────────────

def _make_registry_factory(model_name: str, description: str) -> "ModelSpec":
    """Create a benchmark ModelSpec for a model in ModelRegistry."""
    def factory(in_dim: int, out_dim: int, _n: str = model_name) -> nn.Module:
        from pinneaple_neural.architectures import ModelRegistry
        # Pass all plausible kwargs; filter_supported_kwargs in instantiate() drops unknowns
        attempts = [
            {"in_dim": in_dim, "out_dim": out_dim, "dim_q": in_dim},
            {"in_dim": in_dim, "out_dim": out_dim, "dim_q": in_dim, "hidden_dim": 64},
            {"in_dim": in_dim, "out_dim": out_dim, "dim_q": in_dim,
             "hidden_dim": 64, "n_layers": 4},
        ]
        last_exc: Exception = RuntimeError("no attempts")
        for kw in attempts:
            try:
                return ModelRegistry.build(_n, **kw)
            except Exception as e:
                last_exc = e
        raise last_exc
    return ModelSpec(name=model_name, factory=factory, description=description)


def _make_pinn_factory(model_name: str, pinn_cls: Any) -> "ModelSpec":
    """Create a benchmark ModelSpec for a model in PINNCatalog."""
    def factory(in_dim: int, out_dim: int, _cls: Any = pinn_cls) -> nn.Module:
        for kwargs in [
            {"in_dim": in_dim, "out_dim": out_dim, "hidden": [64, 64, 64, 64]},
            {"in_dim": in_dim, "out_dim": out_dim},
        ]:
            try:
                return _cls(**kwargs)
            except Exception:
                continue
        raise RuntimeError(f"Cannot build PINN {_cls.__name__}")
    return ModelSpec(name=model_name, factory=factory,
                     description=f"PINNCatalog: {model_name}")


def all_model_specs() -> List[ModelSpec]:
    """Return a ModelSpec for every model in ``pinneaple_models`` that can
    accept plain ``(N, in_dim)`` coordinate tensors.

    Importing ``pinneaple_models`` triggers ``register_all()`` automatically,
    so ``ModelRegistry`` is guaranteed to be fully populated here.

    Models with ``input_kind != 'pointwise_coords'`` (graph / grid / sequence
    architectures such as MeshGraphNet, AFNO, FNO, DeepONet) are excluded
    because the PINN benchmark loop passes raw coordinate tensors.
    """
    # Importing pinneaple_models calls register_all() — registry is now full.
    try:
        from pinneaple_neural.architectures import ModelRegistry
    except Exception:
        return list(DEFAULT_MODELS)

    seen_cls: set = set()
    specs: List[ModelSpec] = []

    for name in ModelRegistry.list():
        try:
            reg_spec = ModelRegistry.spec(name)
        except Exception:
            continue
        if reg_spec.input_kind != "pointwise_coords":
            continue
        if reg_spec.cls in seen_cls:       # skip aliases (same class, different key)
            continue
        seen_cls.add(reg_spec.cls)
        specs.append(_make_registry_factory(name, reg_spec.description))

    return specs if specs else list(DEFAULT_MODELS)


def _filter_model_specs(
    all_specs: List["ModelSpec"],
    names: Optional[List[str]],
) -> List["ModelSpec"]:
    if names is None:
        return all_specs
    by_name = {s.name: s for s in all_specs}
    result = []
    for n in names:
        if n not in by_name:
            available = list(by_name)
            raise ValueError(
                f"Unknown model '{n}'. Available ({len(available)}): "
                f"{available[:20]}{'...' if len(available) > 20 else ''}"
            )
        result.append(by_name[n])
    return result


def _filter_tasks(
    all_tasks: Dict[str, "BenchmarkTaskBase"],
    names: Optional[List[str]],
) -> List["BenchmarkTaskBase"]:
    if names is None:
        return list(all_tasks.values())
    result = []
    for pid in names:
        if pid not in all_tasks:
            raise ValueError(
                f"Unknown problem '{pid}'. Available ({len(all_tasks)}): "
                f"{list(all_tasks)[:20]}"
            )
        result.append(all_tasks[pid])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _train_benchmark(
    task: BenchmarkTaskBase,
    model: nn.Module,
    cfg: BenchmarkConfig,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[List[Dict[str, float]], float]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    model = model.to(device)

    # Sample training data once
    X_col_np = task.sample_collocation(cfg.n_col, cfg.seed)
    X_bc_np, Y_bc_np = task.sample_boundary(cfg.n_bc, cfg.seed)
    X_ic_np, Y_ic_np = task.sample_ic(cfg.n_ic, cfg.seed)

    X_col = torch.tensor(X_col_np, dtype=torch.float32, device=device)
    X_bc = torch.tensor(X_bc_np, dtype=torch.float32, device=device)
    Y_bc = torch.tensor(Y_bc_np, dtype=torch.float32, device=device)

    has_ic = X_ic_np.shape[0] > 0
    X_ic = torch.tensor(X_ic_np, dtype=torch.float32, device=device) if has_ic else None
    Y_ic = torch.tensor(Y_ic_np, dtype=torch.float32, device=device) if has_ic else None

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 1e-2)

    history: List[Dict[str, float]] = []
    convergence_epoch = -1
    t0 = time.time()

    for ep in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # PDE residual
        X_req = X_col.detach().requires_grad_(True)
        pde_loss = task.pde_residual(model, X_req)

        # BC loss
        if X_bc.shape[0] > 0:
            pred_bc = model(X_bc)
            if hasattr(pred_bc, "y"):
                pred_bc = pred_bc.y
            bc_loss = nn.functional.mse_loss(pred_bc, Y_bc)
        else:
            bc_loss = torch.zeros(1, device=device).squeeze()

        # IC loss
        if has_ic and X_ic is not None:
            pred_ic = model(X_ic)
            if hasattr(pred_ic, "y"):
                pred_ic = pred_ic.y
            ic_loss = nn.functional.mse_loss(pred_ic, Y_ic)
        else:
            ic_loss = torch.zeros(1, device=device).squeeze()

        total = cfg.weight_pde * pde_loss + cfg.weight_bc * bc_loss + cfg.weight_ic * ic_loss
        total.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        entry = {
            "epoch": ep,
            "loss": float(total.item()),
            "pde": float(pde_loss.item()),
            "bc": float(bc_loss.item()) if isinstance(bc_loss, torch.Tensor) else 0.0,
            "ic": float(ic_loss.item()) if isinstance(ic_loss, torch.Tensor) else 0.0,
        }
        history.append(entry)

        if convergence_epoch < 0 and entry["pde"] < cfg.convergence_threshold:
            convergence_epoch = ep

        if verbose and ep % cfg.log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"      ep {ep:5d}/{cfg.epochs}  "
                f"pde={entry['pde']:.3e}  bc={entry['bc']:.3e}  "
                f"ic={entry['ic']:.3e}  t={elapsed:.1f}s"
            )

    return history, convergence_epoch


# ─────────────────────────────────────────────────────────────────────────────
# PINNArenaBenchmark
# ─────────────────────────────────────────────────────────────────────────────

class PINNArenaBenchmark:
    """
    Multi-problem, multi-architecture PINN benchmark suite.

    Parameters
    ----------
    tasks       : list of BenchmarkTaskBase instances
    model_specs : list of ModelSpec instances
    config      : BenchmarkConfig

    Example
    -------
    >>> bench = PINNArenaBenchmark.default()
    >>> results = bench.run(verbose=True)
    >>> print(bench.leaderboard())
    """

    def __init__(
        self,
        tasks: List[BenchmarkTaskBase],
        model_specs: Optional[List[ModelSpec]] = None,
        config: Optional[BenchmarkConfig] = None,
    ) -> None:
        self.tasks = tasks
        self.model_specs = model_specs or DEFAULT_MODELS
        self.config = config or BenchmarkConfig()
        self.results: List[BenchmarkResult] = []

    # ── Running ───────────────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> List[BenchmarkResult]:
        """Run all model × problem combinations and store results."""
        device = _resolve_device(self.config.device)
        all_results: List[BenchmarkResult] = []

        n_tasks = len(self.tasks)
        n_models = len(self.model_specs)
        total = n_tasks * n_models

        if verbose:
            print("=" * 70)
            print(f"  PINN Arena Benchmark -- {n_tasks} problems x {n_models} models = {total} runs")
            print(f"  Device : {device}  |  Epochs per run : {self.config.epochs}")
            print("=" * 70)

        run_idx = 0
        for task in self.tasks:
            if verbose:
                print(f"\n  Problem: {task.task_id}  (in_dim={task.in_dim}, out_dim={task.out_dim})")
                print("  " + "-" * 66)

            for spec in self.model_specs:
                run_idx += 1
                if verbose:
                    print(f"  [{run_idx:2d}/{total}] {spec.name:<20} | {spec.description}")

                t0 = time.time()
                try:
                    model = spec.factory(task.in_dim, task.out_dim).to(device)
                    n_params = _count_params(model)
                    history, conv_ep = _train_benchmark(
                        task, model, self.config, device, verbose=verbose
                    )
                    eval_metrics = task.evaluate(model, device)
                except Exception as e:
                    if verbose:
                        print(f"      ERROR: {e}")
                    n_params = 0
                    history = []
                    conv_ep = -1
                    eval_metrics = {"rel_l2": float("nan"), "l_inf": float("nan"),
                                    "mse": float("nan"), "pde_residual": float("nan")}

                elapsed = time.time() - t0
                metrics = {
                    **eval_metrics,
                    "train_time_s": elapsed,
                    "n_params": float(n_params),
                    "convergence_epoch": float(conv_ep),
                }

                result = BenchmarkResult(
                    problem_id=task.task_id,
                    model_id=spec.name,
                    metrics=metrics,
                    history=history,
                    elapsed_s=elapsed,
                    n_params=n_params,
                )
                all_results.append(result)

                if verbose:
                    rl = metrics.get("rel_l2", float("nan"))
                    li = metrics.get("l_inf", float("nan"))
                    pr = metrics.get("pde_residual", float("nan"))
                    print(
                        f"      rel_l2={rl:.3e}  l_inf={li:.3e}  "
                        f"pde_res={pr:.3e}  t={elapsed:.1f}s  params={n_params:,}"
                    )

        self.results = all_results
        self._assign_ranks()

        if verbose:
            print("\n" + "=" * 70)
            print(self.leaderboard(by_problem=True))
            print("=" * 70)

        return all_results

    def _assign_ranks(self) -> None:
        """Assign per-problem rank by rel_l2 (lower = better)."""
        by_problem: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            by_problem.setdefault(r.problem_id, []).append(r)

        for pid, runs in by_problem.items():
            valid = [r for r in runs if not math.isnan(r.metrics.get("rel_l2", float("nan")))]
            sorted_runs = sorted(valid, key=lambda r: r.metrics.get("rel_l2", 1e9))
            for rank, r in enumerate(sorted_runs, start=1):
                r.rank = rank
            failed = [r for r in runs if r not in valid]
            for r in failed:
                r.rank = len(runs)

    # ── Reporting ─────────────────────────────────────────────────────────────

    def leaderboard(self, metric: str = "rel_l2", by_problem: bool = False) -> str:
        if not self.results:
            return "(no results)"

        if by_problem:
            lines: List[str] = []
            by_prob: Dict[str, List[BenchmarkResult]] = {}
            for r in self.results:
                by_prob.setdefault(r.problem_id, []).append(r)

            for pid, runs in by_prob.items():
                sorted_runs = sorted(runs, key=lambda r: r.metrics.get(metric, 1e9))
                lines.append(f"\n  Problem: {pid}")
                hdr = f"    {'Rank':<5} {'Model':<22} {metric:>10}  {'l_inf':>10}  {'time':>8}  {'params':>10}"
                lines.append(hdr)
                lines.append("    " + "-" * (len(hdr) - 4))
                for r in sorted_runs:
                    val = r.metrics.get(metric, float("nan"))
                    li = r.metrics.get("l_inf", float("nan"))
                    lines.append(
                        f"    #{r.rank:<4} {r.model_id:<22} {val:>10.3e}  {li:>10.3e}"
                        f"  {r.elapsed_s:>7.1f}s  {r.n_params:>10,}"
                    )
            return "\n".join(lines)

        # Global ranking
        all_models = {r.model_id for r in self.results}
        avg_by_model: Dict[str, float] = {}
        for mid in all_models:
            vals = [r.metrics.get(metric, float("nan")) for r in self.results if r.model_id == mid]
            vals = [v for v in vals if not math.isnan(v)]
            avg_by_model[mid] = float(np.mean(vals)) if vals else float("nan")

        ranked = sorted(avg_by_model.items(), key=lambda x: x[1])
        lines = [
            "\n  Global Leaderboard (avg across all problems)",
            f"  {'Rank':<5} {'Model':<22} {'avg ' + metric:>12}",
            "  " + "-" * 42,
        ]
        for i, (mid, val) in enumerate(ranked, start=1):
            lines.append(f"  #{i:<4} {mid:<22} {val:>12.4e}")
        return "\n".join(lines)

    def best_per_problem(self, metric: str = "rel_l2") -> Dict[str, str]:
        """Return {problem_id: best_model_id} by given metric."""
        by_prob: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            by_prob.setdefault(r.problem_id, []).append(r)
        best: Dict[str, str] = {}
        for pid, runs in by_prob.items():
            valid = [r for r in runs if not math.isnan(r.metrics.get(metric, float("nan")))]
            if valid:
                best[pid] = min(valid, key=lambda r: r.metrics.get(metric, 1e9)).model_id
        return best

    def summary_table(self) -> str:
        """Pivot table: problems as rows, models as cols, rel_l2 as values."""
        if not self.results:
            return "(no results)"

        problems = list(dict.fromkeys(r.problem_id for r in self.results))
        models = list(dict.fromkeys(r.model_id for r in self.results))

        idx: Dict[Tuple[str, str], float] = {
            (r.problem_id, r.model_id): r.metrics.get("rel_l2", float("nan"))
            for r in self.results
        }

        col_w = max(12, *(len(m) for m in models))
        prob_w = max(20, *(len(p) for p in problems))

        header = f"  {'Problem':<{prob_w}} " + " ".join(f"{m:>{col_w}}" for m in models)
        sep = "  " + "-" * len(header)
        lines = ["\n  rel_l2 Summary Table", header, sep]
        for prob in problems:
            row = f"  {prob:<{prob_w}} "
            for mod in models:
                val = idx.get((prob, mod), float("nan"))
                row += f"{val:>{col_w}.3e} " if not math.isnan(val) else f"{'N/A':>{col_w}} "
            lines.append(row)
        return "\n".join(lines)

    def plot_leaderboard(self, save_path: Optional[str] = None) -> None:
        """Bar chart of average rel_l2 per model across all problems."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[benchmark] matplotlib not available - skip plot")
            return

        models = list(dict.fromkeys(r.model_id for r in self.results))
        avgs = []
        for mid in models:
            vals = [
                r.metrics.get("rel_l2", float("nan"))
                for r in self.results
                if r.model_id == mid and not math.isnan(r.metrics.get("rel_l2", float("nan")))
            ]
            avgs.append(float(np.mean(vals)) if vals else float("nan"))

        order = sorted(range(len(models)), key=lambda i: avgs[i] if not math.isnan(avgs[i]) else 1e9)
        models_s = [models[i] for i in order]
        avgs_s = [avgs[i] for i in order]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(models_s, avgs_s, color="steelblue", edgecolor="white")
        ax.bar_label(bars, fmt="%.2e", fontsize=9, padding=3)
        ax.set_ylabel("Average Relative L2 Error")
        ax.set_title("PINN Arena — Model Leaderboard (lower is better)")
        ax.set_yscale("log")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"[benchmark] Leaderboard plot saved: {save_path}")
        else:
            plt.show()
        plt.close(fig)

    def plot_convergence(self, problem_id: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """Loss convergence curves for all models on one problem."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        runs = self.results
        if problem_id:
            runs = [r for r in runs if r.problem_id == problem_id]
        if not runs:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        for r in runs:
            if not r.history:
                continue
            epochs = [h["epoch"] for h in r.history]
            losses = [h["pde"] for h in r.history]
            ax.semilogy(epochs, losses, label=r.model_id, alpha=0.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("PDE Residual Loss")
        pid = problem_id or "All Problems"
        ax.set_title(f"PINN Convergence — {pid}")
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close(fig)

    # ── I/O ──────────────────────────────────────────────────────────────────

    def save_results(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = [r.to_dict() for r in self.results]
        p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[benchmark] Results saved: {p}  ({len(data)} entries)")

    @classmethod
    def load_results(cls, path: str) -> List[Dict[str, Any]]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    # ── Default ───────────────────────────────────────────────────────────────

    @classmethod
    def default(
        cls,
        problems: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        epochs: int = 5000,
        device: str = "auto",
    ) -> "PINNArenaBenchmark":
        """Create benchmark from ALL available problems and models.

        Problems are loaded from ``pinneaple_environment`` presets + built-in
        tasks. Models are loaded from ``pinneaple_models`` (ModelRegistry,
        PINNCatalog, and Group-B catalogue).

        Parameters
        ----------
        problems : list of problem ids to restrict the run, or None for all.
        models   : list of model names to restrict the run, or None for all.
        epochs   : training epochs per run.
        device   : "auto" | "cpu" | "cuda".
        """
        # ── Problems ──────────────────────────────────────────────────────────
        all_tasks: Dict[str, BenchmarkTaskBase] = {}

        # 1. Six original tasks (guaranteed working, own eval grids)
        try:
            from pinneaple_tools.benchmark_suite.tasks.burgers_1d import Burgers1DTask
            from pinneaple_tools.benchmark_suite.tasks.poisson_2d import Poisson2DTask
            from pinneaple_tools.benchmark_suite.tasks.heat_1d import Heat1DTask
            from pinneaple_tools.benchmark_suite.tasks.wave_1d import Wave1DTask
            from pinneaple_tools.benchmark_suite.tasks.allen_cahn_1d import AllenCahn1DTask
            from pinneaple_tools.benchmark_suite.tasks.navier_stokes_tgv_2d import NavierStokesTGV2DTask
            all_tasks.update({
                "burgers_1d": Burgers1DTask(),
                "poisson_2d": Poisson2DTask(),
                "heat_1d": Heat1DTask(),
                "wave_1d": Wave1DTask(),
                "allen_cahn_1d": AllenCahn1DTask(),
                "ns_tgv_2d": NavierStokesTGV2DTask(),
            })
        except Exception:
            pass

        # 2. All built-in _ProblemDef problems (from physics_pipeline)
        try:
            from pinneaple_tools.benchmark_suite.physics_pipeline import _BUILTIN_PROBLEMS
            for pid, prob_def in _BUILTIN_PROBLEMS.items():
                if pid not in all_tasks:
                    all_tasks[pid] = _BuiltinProblemTask(prob_def)
        except Exception:
            pass

        # 3. All pinneaple_environment presets
        try:
            from pinneaple_physics.pde_environment.presets.registry import list_presets, get_preset
            for pid in list_presets():
                if pid not in all_tasks:
                    try:
                        spec = get_preset(pid)
                        all_tasks[pid] = _PresetProblemTask(spec, pid)
                    except Exception:
                        pass
        except Exception:
            pass

        selected_tasks = _filter_tasks(all_tasks, problems)

        # ── Models ────────────────────────────────────────────────────────────
        all_specs = all_model_specs()
        selected_specs = _filter_model_specs(all_specs, models)

        cfg = BenchmarkConfig(epochs=epochs, device=device)
        return cls(tasks=selected_tasks, model_specs=selected_specs, config=cfg)

    @classmethod
    def list_problems(cls) -> List[str]:
        """Return names of all problems that ``default()`` would load."""
        names: List[str] = []
        for src in ("burgers_1d", "poisson_2d", "heat_1d", "wave_1d",
                    "allen_cahn_1d", "ns_tgv_2d"):
            names.append(src)
        try:
            from pinneaple_tools.benchmark_suite.physics_pipeline import _BUILTIN_PROBLEMS
            names += [k for k in _BUILTIN_PROBLEMS if k not in names]
        except Exception:
            pass
        try:
            from pinneaple_physics.pde_environment.presets.registry import list_presets
            names += [p for p in list_presets() if p not in names]
        except Exception:
            pass
        return names

    @classmethod
    def list_models(cls) -> List[str]:
        """Return names of all models that ``default()`` would load."""
        return [s.name for s in all_model_specs()]
