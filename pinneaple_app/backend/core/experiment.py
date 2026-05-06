"""Experiment runner: trains multiple models on the same problem and collects results."""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
import numpy as np


@dataclass
class ModelRunConfig:
    name: str                              # model registry name
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)  # forwarded to ModelRegistry.build
    weight_override: Optional[Dict[str, float]] = None           # physics loss weights


@dataclass
class ExperimentConfig:
    """Full configuration for one experimental run (one problem, N models)."""
    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    problem_name: str = ""
    models: List[ModelRunConfig] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    epochs: int = 2000
    lr: float = 1e-3
    device: str = "cpu"
    batch_size: int = 4096
    seed: int = 42


@dataclass
class ModelResult:
    name: str
    metrics: Dict[str, float]
    loss_history: List[float]               # per-epoch total loss
    physics_loss_history: List[float]
    bc_loss_history: List[float]
    train_time_s: float
    n_params: int
    model_state: Optional[Any] = None       # state_dict (serializable)
    error: Optional[str] = None             # non-None if training failed


@dataclass
class ExperimentResult:
    experiment_id: str
    config: ExperimentConfig
    model_results: Dict[str, ModelResult]
    completed_at: str = ""

    def leaderboard(self) -> list:
        """Return list of dicts sorted by l2_relative (ascending)."""
        rows = []
        for name, r in self.model_results.items():
            if r.error:
                continue
            row = {"model": name, **r.metrics, "n_params": r.n_params,
                   "train_time_s": round(r.train_time_s, 2)}
            rows.append(row)
        return sorted(rows, key=lambda x: x.get("l2_relative", float("inf")))


class ExperimentRunner:
    """Trains all configured models and streams progress via an async generator."""

    def __init__(self, config: ExperimentConfig, data_bundle, problem):
        self.config = config
        self.data = data_bundle
        self.problem = problem

    async def run(self, progress_cb: Optional[Callable] = None) -> ExperimentResult:
        """Run all models sequentially. Calls ``progress_cb(event_dict)`` after each epoch."""
        import asyncio
        results: Dict[str, ModelResult] = {}

        for model_cfg in self.config.models:
            result = await asyncio.to_thread(
                self._train_one, model_cfg, progress_cb
            )
            results[model_cfg.name] = result

        from datetime import datetime
        return ExperimentResult(
            experiment_id=self.config.experiment_id,
            config=self.config,
            model_results=results,
            completed_at=datetime.utcnow().isoformat(),
        )

    def _train_one(self, model_cfg: ModelRunConfig, progress_cb) -> ModelResult:
        """Synchronous training of a single model."""
        import torch
        from pinneaple_neural.architectures import ModelRegistry
        from pinneaple_neural.trainer import Trainer, TrainConfig
        from pinneaple_physics.pinn_solver import compile_problem

        name = model_cfg.name
        t0 = time.time()

        try:
            # ── Build model ───────────────────────────────────────────────
            in_dim = len(self.problem.domain_bounds)
            if self.problem.is_time_dependent:
                in_dim += 1
            out_dim = len(self.problem.field_names)

            model = _build_model(name, in_dim, out_dim, model_cfg.extra_kwargs)

            # ── Compile physics loss ──────────────────────────────────────
            if self.problem.kind == "preset" and self.problem.spec is not None:
                physics_fn = compile_problem(self.problem.spec)
            elif self.problem.kind == "custom" and self.problem.equations:
                physics_fn = _compile_custom(self.problem)
            else:
                physics_fn = None

            # ── Prepare tensors ───────────────────────────────────────────
            dev = torch.device(self.config.device)
            x_col = torch.as_tensor(self.data.x_col, dtype=torch.float32, device=dev)
            x_bnd = torch.as_tensor(self.data.x_bnd, dtype=torch.float32, device=dev)
            x_ic  = (torch.as_tensor(self.data.x_ic,  dtype=torch.float32, device=dev)
                     if self.data.x_ic is not None else None)
            u_ref = (torch.as_tensor(self.data.u_ref, dtype=torch.float32, device=dev)
                     if self.data.u_ref is not None else None)

            model = model.to(dev)

            # ── Training loop ─────────────────────────────────────────────
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
            loss_hist, phys_hist, bc_hist = [], [], []

            for epoch in range(self.config.epochs):
                optimizer.zero_grad()

                out = model(x_col)
                pred = out.y if hasattr(out, "y") else out

                # Physics loss
                phys_loss = torch.tensor(0.0, device=dev)
                if physics_fn is not None:
                    try:
                        loss_dict = physics_fn(model, x_col, {})
                        phys_loss = sum(loss_dict.values())
                    except Exception:
                        pass

                # BC loss
                bc_loss = torch.tensor(0.0, device=dev)
                if len(x_bnd) > 0:
                    out_bnd = model(x_bnd)
                    pred_bnd = out_bnd.y if hasattr(out_bnd, "y") else out_bnd
                    bc_loss = (pred_bnd ** 2).mean()

                # Data loss (if reference available)
                data_loss = torch.tensor(0.0, device=dev)
                if u_ref is not None and len(u_ref) == len(pred):
                    data_loss = ((pred - u_ref) ** 2).mean()

                total = phys_loss + bc_loss + data_loss
                total.backward()
                optimizer.step()

                loss_hist.append(float(total))
                phys_hist.append(float(phys_loss))
                bc_hist.append(float(bc_loss))

                if progress_cb and epoch % max(1, self.config.epochs // 50) == 0:
                    progress_cb({
                        "model": name,
                        "epoch": epoch,
                        "total_epochs": self.config.epochs,
                        "loss": float(total),
                        "phys_loss": float(phys_loss),
                        "bc_loss": float(bc_loss),
                    })

            # ── Compute metrics ───────────────────────────────────────────
            metrics = _compute_metrics(
                model, x_col, u_ref, phys_hist, bc_hist,
                self.config.metrics, dev
            )

            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            return ModelResult(
                name=name,
                metrics=metrics,
                loss_history=loss_hist,
                physics_loss_history=phys_hist,
                bc_loss_history=bc_hist,
                train_time_s=time.time() - t0,
                n_params=n_params,
                model_state={k: v.cpu().numpy().tolist()
                             for k, v in model.state_dict().items()},
            )

        except Exception as e:
            return ModelResult(
                name=name,
                metrics={},
                loss_history=[],
                physics_loss_history=[],
                bc_loss_history=[],
                train_time_s=time.time() - t0,
                n_params=0,
                error=str(e),
            )


# ── Helpers ───────────────────────────────────────────────────────────────

def _build_model(name: str, in_dim: int, out_dim: int, extra_kwargs: dict):
    from pinneaple_neural.architectures import ModelRegistry
    # Common overrides so the registry has what it needs
    kwargs = dict(in_dim=in_dim, out_dim=out_dim)
    kwargs.update(extra_kwargs)
    return ModelRegistry.build(name, **kwargs)


def _compile_custom(problem):
    """Attempt to compile custom EquationSpec list into a physics_fn."""
    try:
        from pinneaple_physics.symbolic_pde import SymbolicPDE, auto_residual
        import sympy as sp
        residuals = []
        for eq in problem.equations:
            if isinstance(eq.expression, str):
                expr = sp.sympify(eq.expression)
            else:
                expr = eq.expression
            residuals.append(expr)
        # Build a simple combined residual
        def _physics_fn(model, x, _):
            import torch
            out = model(x)
            pred = out.y if hasattr(out, "y") else out
            return {"custom_pde": (pred ** 2).mean() * 0.0}  # placeholder
        return _physics_fn
    except Exception:
        return None


def _compute_metrics(model, x_col, u_ref, phys_hist, bc_hist, metric_names, dev):
    import torch
    metrics: Dict[str, float] = {}

    with torch.no_grad():
        out = model(x_col)
        pred = out.y if hasattr(out, "y") else out

    if "pde_residual" in metric_names and phys_hist:
        metrics["pde_residual"] = float(np.mean(phys_hist[-10:]))
    if "bc_residual" in metric_names and bc_hist:
        metrics["bc_residual"] = float(np.mean(bc_hist[-10:]))

    if u_ref is not None:
        diff = (pred - u_ref).cpu().numpy()
        ref_np = u_ref.cpu().numpy()

        if "mse" in metric_names:
            metrics["mse"] = float(np.mean(diff ** 2))
        if "mae" in metric_names:
            metrics["mae"] = float(np.mean(np.abs(diff)))
        if "max_error" in metric_names:
            metrics["max_error"] = float(np.max(np.abs(diff)))
        if "l2_relative" in metric_names:
            norm = np.linalg.norm(ref_np) + 1e-10
            metrics["l2_relative"] = float(np.linalg.norm(diff) / norm)
        if "r2" in metric_names:
            ss_res = np.sum(diff ** 2)
            ss_tot = np.sum((ref_np - ref_np.mean()) ** 2) + 1e-10
            metrics["r2"] = float(1.0 - ss_res / ss_tot)
    else:
        if "l2_relative" in metric_names:
            metrics["l2_relative"] = float("nan")

    return metrics
