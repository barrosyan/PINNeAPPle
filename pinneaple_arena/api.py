"""Arena — high-level API for running PINNeAPPle resolution pipelines.

This module bridges the problem design phase (``ProblemSpec``) with the
resolution pipeline (dataset generation → training → metrics).

Three entry points
------------------
1. From a ProblemSpec (built with ProblemBuilder)::

       from pinneaple_environment import ProblemBuilder
       from pinneaple_arena import Arena

       spec = ProblemBuilder("heat_1d").domain(x=(0,1), t=(0,1)) ...build()
       result = Arena.from_spec(spec).run(model="VanillaPINN", epochs=5000)

2. From a registered preset::

       result = Arena.from_preset("burgers_1d", nu=0.01).run(...)

3. From a YAML config (existing workflow)::

       result = Arena.from_yaml("configs/experiment_burgers_1d.yaml").run()

Output contract
---------------
All ``.run()`` / ``.compare()`` calls return an ``ArenaResult`` dataclass with:
- ``spec``     : ProblemSpec used
- ``model``    : trained nn.Module
- ``history``  : list of loss dicts per epoch
- ``metrics``  : final evaluation metrics dict
- ``artifacts``: dict of paths to saved files
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ArenaResult:
    """Container returned by Arena.run() and Arena.compare()."""
    spec: Any                                    # ProblemSpec
    model: Optional[nn.Module] = None
    history: List[Dict[str, float]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    model_id: str = "model"
    elapsed_s: float = 0.0

    def summary(self) -> str:
        lines = [
            f"ArenaResult — {self.spec.name} / {self.model_id}",
            f"  Epochs   : {len(self.history)}",
            f"  Time     : {self.elapsed_s:.1f}s",
        ]
        if self.metrics:
            for k, v in self.metrics.items():
                lines.append(f"  {k:<12}: {v:.4e}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


@dataclass
class ArenaCompareResult:
    """Container returned by Arena.compare() with multiple models."""
    spec: Any
    results: List[ArenaResult] = field(default_factory=list)

    def leaderboard(self) -> str:
        header = f"{'Model':<20}  {'RMSE':>10}  {'Rel-L2':>10}  {'Time':>8}"
        sep = "─" * len(header)
        rows = [header, sep]
        for r in sorted(self.results, key=lambda x: x.metrics.get("rmse", 1e9)):
            rmse = r.metrics.get("rmse", float("nan"))
            rel  = r.metrics.get("rel_l2", float("nan"))
            rows.append(f"{r.model_id:<20}  {rmse:>10.4e}  {rel:>10.4e}  {r.elapsed_s:>7.1f}s")
        return "\n".join(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Device helper
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_batch(
    spec,
    n_col: int,
    n_bc: int,
    n_ic: int,
    seed: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Generate training batch from ProblemSpec using problem_runner."""
    try:
        from pinneaple_solvers.problem_runner import generate_pinn_dataset
        batch_np = generate_pinn_dataset(
            spec,
            n_col=n_col,
            n_bc=n_bc,
            n_ic=n_ic,
            n_data=0,
            seed=seed,
        )
        batch = {}
        for k, v in batch_np.items():
            if isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(v.astype(np.float32)).to(device)
            else:
                batch[k] = v
        if "x" not in batch:
            batch["x"] = batch.get("x_col") or batch.get("x_data")
        return batch
    except Exception as e:
        # Fallback: uniform collocation only
        rng = np.random.default_rng(seed)
        dims = len(spec.coords)
        pts = np.zeros((n_col, dims), dtype=np.float32)
        for i, c in enumerate(spec.coords):
            lo, hi = spec.domain_bounds.get(c, (0.0, 1.0))
            pts[:, i] = rng.uniform(lo, hi, n_col)
        x = torch.from_numpy(pts).to(device)
        return {"x": x, "x_col": x}


# ══════════════════════════════════════════════════════════════════════════════
# Model builder
# ══════════════════════════════════════════════════════════════════════════════

def _build_model(
    model: Union[str, nn.Module],
    spec,
    hidden: Sequence[int] = (64, 64, 64, 64),
    activation: str = "tanh",
) -> nn.Module:
    """Build or return a model for the given spec."""
    if isinstance(model, nn.Module):
        return model

    in_dim  = len(spec.coords)
    out_dim = len(spec.fields)

    if isinstance(model, str):
        name_lc = model.lower().replace("_", "").replace("-", "")

        # Try model registry
        try:
            from pinneaple_models.registry import ModelRegistry
            from pinneaple_models.register_all import register_all
            register_all()
            return ModelRegistry.build(model, in_dim=in_dim, out_dim=out_dim)
        except Exception:
            pass

        # VanillaPINN / vanilla_pinn
        if "vanilla" in name_lc or name_lc == "pinn":
            try:
                from pinneaple_models.pinns.vanilla import VanillaPINN
                return VanillaPINN(in_dim=in_dim, out_dim=out_dim, hidden=list(hidden))
            except Exception:
                pass

        # InversePINN
        if "inverse" in name_lc:
            try:
                from pinneaple_models.pinns.inverse import InversePINN
                return InversePINN(in_dim=in_dim, out_dim=out_dim, hidden=list(hidden))
            except Exception:
                pass

        # DeepONet
        if "deeponet" in name_lc or "onet" in name_lc:
            try:
                from pinneaple_models.neural_operators.deeponet import DeepONet
                return DeepONet(branch_in=in_dim, trunk_in=in_dim, out_dim=out_dim)
            except Exception:
                pass

    # Generic MLP fallback
    act_map = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}
    act_fn = act_map.get(activation.lower(), nn.Tanh)
    dims = [in_dim] + list(hidden) + [out_dim]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_fn())
    return nn.Sequential(*layers)


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def _default_physics_loss(model_out: Any, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """MSE against x_bc / y_bc targets when physics compiler isn't available."""
    loss = torch.zeros(1, device=next(iter(batch.values())).device)
    x_bc = batch.get("x_bc")
    y_bc = batch.get("y_bc")
    if x_bc is not None and y_bc is not None:
        if callable(model_out):
            pred = model_out(x_bc)
        elif isinstance(model_out, nn.Module):
            pred = model_out(x_bc)
        else:
            pred = torch.zeros_like(y_bc)
        if hasattr(pred, "y"):
            pred = pred.y
        loss = loss + nn.functional.mse_loss(pred, y_bc)
    return loss


def _train_loop(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    spec,
    epochs: int,
    lr: float,
    device: torch.device,
    verbose: bool,
    physics_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, float]]:
    """Minimal training loop. Uses physics loss compiler if available."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Try to compile physics loss
    physics_loss_fn = None
    try:
        from pinneaple_train.losses import build_loss
        physics_loss_fn = build_loss(
            problem_spec=spec,
            model_capabilities={"supports_physics_loss": True},
            weights=physics_weights,
            supervised_kind="mse",
        )
    except Exception:
        pass

    history: List[Dict[str, float]] = []
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        x = batch.get("x", batch.get("x_col"))
        if x is None:
            raise RuntimeError("Batch missing 'x' or 'x_col' key.")

        if hasattr(model, "forward"):
            try:
                y_hat = model(x)
            except Exception:
                y_hat = None
        else:
            y_hat = None

        if physics_loss_fn is not None:
            try:
                loss = physics_loss_fn(model, y_hat, batch)
                if isinstance(loss, dict):
                    total = sum(loss.values())
                    loss_dict = {k: float(v.item()) for k, v in loss.items()}
                else:
                    total = loss
                    loss_dict = {}
                total.backward()
            except Exception:
                # Fallback to simple MSE
                if y_hat is not None and hasattr(y_hat, "y"):
                    pred = y_hat.y
                elif y_hat is not None:
                    pred = y_hat
                else:
                    pred = model(x)
                y_bc = batch.get("y_bc")
                if y_bc is not None:
                    loss = nn.functional.mse_loss(pred, y_bc)
                else:
                    loss = pred.pow(2).mean()
                loss.backward()
                loss_dict = {}
                total = loss
        else:
            if y_hat is not None and hasattr(y_hat, "y"):
                pred = y_hat.y
            elif y_hat is not None:
                pred = y_hat
            else:
                pred = model(x)
            y_bc = batch.get("y_bc")
            if y_bc is not None:
                total = nn.functional.mse_loss(pred, y_bc)
            else:
                total = pred.pow(2).mean()
            total.backward()
            loss_dict = {}

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        entry = {"epoch": ep, "loss": float(total.item()), **loss_dict}
        history.append(entry)

        if verbose and ep % max(1, epochs // 10) == 0:
            elapsed = time.time() - t0
            print(f"    epoch {ep:5d}/{epochs}  loss={total.item():.4e}  {elapsed:.1f}s")

    return history


def _compute_metrics(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    spec,
) -> Dict[str, float]:
    """Compute basic regression metrics on data points if available."""
    metrics: Dict[str, float] = {}
    x_data = batch.get("x_data")
    y_data = batch.get("y_data")
    if x_data is None or y_data is None:
        return metrics

    model.eval()
    with torch.no_grad():
        pred = model(x_data)
        if hasattr(pred, "y"):
            pred = pred.y
        if not isinstance(pred, torch.Tensor):
            return metrics

        diff = pred - y_data
        mse  = float(diff.pow(2).mean().item())
        rmse = float(mse**0.5)
        ss_res = float(diff.pow(2).sum().item())
        ss_tot = float((y_data - y_data.mean()).pow(2).sum().item())
        r2 = 1.0 - ss_res / (ss_tot + 1e-10)
        rel_l2 = float((diff.pow(2).sum() / (y_data.pow(2).sum() + 1e-10)).sqrt().item())
        max_err = float(diff.abs().max().item())

    metrics = {
        "mse": mse, "rmse": rmse, "rel_l2": rel_l2, "r2": r2, "max_error": max_err
    }
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Arena
# ══════════════════════════════════════════════════════════════════════════════

class Arena:
    """
    High-level runner that connects ``ProblemSpec`` to the training pipeline.

    Create via one of the class-method constructors, then call ``.run()``
    or ``.compare()``.
    """

    def __init__(self, spec) -> None:
        self._spec = spec
        self._yaml_config: Optional[Dict[str, Any]] = None

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_spec(cls, spec) -> "Arena":
        """
        Create an Arena from a ``ProblemSpec`` (e.g. built with ProblemBuilder).

        Example::

            arena = Arena.from_spec(
                ProblemBuilder("heat_1d")
                .domain(x=(0,1), t=(0,1))
                .fields("u")
                .pde("heat_1d", alpha=0.01)
                .ic(field="u", fn=lambda X: np.sin(np.pi * X[:, 0:1]))
                .bc("dirichlet", field="u", value=0.0, on="x_boundary")
                .build()
            )
            result = arena.run(epochs=5000)
        """
        return cls(spec)

    @classmethod
    def from_preset(cls, preset_id: str, **params) -> "Arena":
        """
        Create an Arena from a registered preset.

        Example::

            arena = Arena.from_preset("burgers_1d", nu=0.005)
            result = arena.run(epochs=10000)
        """
        from pinneaple_environment.presets.registry import get_preset
        spec = get_preset(preset_id, **params)
        return cls(spec)

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "Arena":
        """
        Create an Arena from a YAML config file.

        The YAML must follow the Arena experiment schema.
        Delegates to ``run_arena_experiment`` when ``.run()`` is called.
        """
        from pinneaple_arena.io.yamlx import load_yaml
        cfg = load_yaml(str(config_path))
        problem_cfg = cfg.get("problem", {})
        from pinneaple_environment.presets.registry import get_preset
        preset_id = str(problem_cfg.get("id", "burgers_1d"))
        params = dict(problem_cfg.get("params", {}))
        spec = get_preset(preset_id, **params)
        arena = cls(spec)
        arena._yaml_config = cfg
        return arena

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(
        self,
        model: Union[str, nn.Module] = "VanillaPINN",
        *,
        epochs: int = 5000,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int = 42,
        n_col: Optional[int] = None,
        n_bc: Optional[int] = None,
        n_ic: Optional[int] = None,
        physics_weights: Optional[Dict[str, float]] = None,
        hidden: Sequence[int] = (64, 64, 64, 64),
        activation: str = "tanh",
        verbose: bool = True,
        save_dir: Optional[Union[str, Path]] = None,
        model_id: str = "model",
    ) -> ArenaResult:
        """
        Train a model on the problem and return results.

        Parameters
        ----------
        model   : model name string or nn.Module instance
        epochs  : training epochs
        lr      : initial learning rate (cosine decay)
        device  : "auto" | "cpu" | "cuda" | "cuda:0"
        seed    : random seed for reproducibility
        n_col   : collocation points (overrides spec default)
        n_bc    : boundary points (overrides spec default)
        n_ic    : initial condition points (overrides spec default)
        physics_weights : dict of loss term weights, e.g. {"pde": 1.0, "bc": 10.0}
        hidden  : hidden layer sizes for default MLP
        verbose : print progress
        save_dir: save model checkpoint here if given
        model_id: label for this run (used in compare / leaderboard)

        Returns
        -------
        ArenaResult
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        dev = _resolve_device(device)

        sd = self._spec.sample_defaults
        _n_col = n_col or sd.get("n_col", 4000)
        _n_bc  = n_bc  or sd.get("n_bc",  800)
        _n_ic  = n_ic  or sd.get("n_ic",  800)

        if verbose:
            print(f"\n[Arena] Problem : {self._spec.name}")
            print(f"[Arena] Model   : {model if isinstance(model, str) else type(model).__name__}")
            print(f"[Arena] Device  : {dev}  |  Epochs: {epochs}")
            print(f"[Arena] Points  : col={_n_col}  bc={_n_bc}  ic={_n_ic}")

        # Build dataset
        batch = _build_batch(self._spec, _n_col, _n_bc, _n_ic, seed, dev)

        # Build model
        net = _build_model(model, self._spec, hidden=hidden, activation=activation)

        # Train
        t0 = time.time()
        history = _train_loop(
            net, batch, self._spec, epochs, lr, dev, verbose, physics_weights
        )
        elapsed = time.time() - t0

        # Metrics
        metrics = _compute_metrics(net, batch, self._spec)

        # Save
        artifacts: Dict[str, str] = {}
        if save_dir is not None:
            save_path = Path(save_dir) / f"{model_id}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(save_path))
            artifacts["checkpoint"] = str(save_path)
            if verbose:
                print(f"[Arena] Checkpoint saved: {save_path}")

        if verbose:
            final_loss = history[-1]["loss"] if history else float("nan")
            print(f"[Arena] Done — loss={final_loss:.4e}  t={elapsed:.1f}s")
            if metrics:
                print(f"[Arena] Metrics: " + "  ".join(f"{k}={v:.4e}" for k, v in metrics.items()))

        return ArenaResult(
            spec=self._spec,
            model=net,
            history=history,
            metrics=metrics,
            artifacts=artifacts,
            model_id=model_id,
            elapsed_s=elapsed,
        )

    # ── Compare ───────────────────────────────────────────────────────────────

    def compare(
        self,
        models: Sequence[Union[str, nn.Module]],
        *,
        epochs: int = 5000,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int = 42,
        n_col: Optional[int] = None,
        n_bc: Optional[int] = None,
        n_ic: Optional[int] = None,
        verbose: bool = True,
    ) -> ArenaCompareResult:
        """
        Train multiple models on the same problem and compare results.

        Example::

            compare_result = arena.compare(
                ["VanillaPINN", "InversePINN", "VPINN"],
                epochs=3000,
            )
            print(compare_result.leaderboard())

        Returns
        -------
        ArenaCompareResult with .leaderboard() and .results list
        """
        results: List[ArenaResult] = []
        for m in models:
            model_id = m if isinstance(m, str) else type(m).__name__
            if verbose:
                print(f"\n{'─'*50}")
                print(f"  Running: {model_id}")
            r = self.run(
                model=m,
                epochs=epochs,
                lr=lr,
                device=device,
                seed=seed,
                n_col=n_col,
                n_bc=n_bc,
                n_ic=n_ic,
                verbose=verbose,
                model_id=model_id,
            )
            results.append(r)

        return ArenaCompareResult(spec=self._spec, results=results)

    # ── Spec access ───────────────────────────────────────────────────────────

    @property
    def spec(self):
        """The ProblemSpec this Arena was created from."""
        return self._spec

    def __repr__(self) -> str:
        return f"Arena(problem={self._spec.name!r}, dim={self._spec.dim})"
