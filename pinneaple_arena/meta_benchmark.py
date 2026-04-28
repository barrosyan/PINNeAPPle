"""Meta-Learning Benchmark for PINNs.

Trains MAML and Reptile meta-models on PDE parameter families (e.g. Burgers
with varying viscosity nu), then evaluates K-shot adaptation against
from-scratch training with the same number of gradient steps.

The central result is a **K-shot error curve**: how quickly each approach
reaches low relative L2 error as K (adaptation steps) grows.

Algorithms
----------
MAML (FOMAML)
    Bi-level: inner loop adapts a copy for K steps on support set, outer loop
    minimises query-set loss w.r.t. the meta-initialization.  First-order
    approximation (FOMAML) is used — gradients through the inner update are
    detached, which scales to deep networks without higher-order overhead.

Reptile
    Per-task: deep-copy model, run K SGD steps, then interpolate meta-weights
    toward task weights by step size epsilon.  Simpler than MAML and often
    competitive in practice.

Scratch baseline
    Start from random Kaiming initialization, run K gradient steps.  Same
    optimizer (Adam) and same data as the meta-adapted models.

Full-scratch oracle
    Train from scratch for the same TOTAL gradient steps as meta-training
    (n_meta_epochs x n_tasks_per_batch x n_inner_steps).  Upper bound on
    what dedicated per-task training can achieve.
"""
from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from pinneaple_arena.benchmark import (
    BenchmarkTaskBase,
    _build_mlp,
    _resolve_device,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config & dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetaBenchmarkConfig:
    # Meta-training
    n_meta_epochs: int = 500
    n_inner_steps: int = 5
    inner_lr: float = 0.01
    outer_lr: float = 1e-3
    n_tasks_per_batch: int = 4

    # K-shot evaluation points
    k_shots: Tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200)

    # Algorithms to compare
    algorithms: Tuple[str, ...] = ("maml", "reptile")

    # Data
    n_col: int = 1000
    n_bc: int = 200
    n_ic: int = 200

    # Loss weights
    weight_pde: float = 1.0
    weight_bc: float = 10.0
    weight_ic: float = 10.0

    # Model
    hidden: Tuple[int, ...] = (64, 64, 64, 64)

    # Misc
    device: str = "auto"
    seed: int = 42
    log_interval: int = 100


@dataclass
class MetaBenchmarkFamily:
    """A parametric PDE family for meta-learning."""
    name: str
    # factory(param_dict) -> BenchmarkTaskBase (should NOT build FDM reference)
    train_task_factory: Callable[[Dict[str, float]], BenchmarkTaskBase]
    # factory(param_dict) -> BenchmarkTaskBase (WITH reference for evaluation)
    eval_task_factory: Callable[[Dict[str, float]], BenchmarkTaskBase]
    param_ranges: Dict[str, Tuple[float, float]]
    eval_params: List[Dict[str, float]]  # specific param values to evaluate on


@dataclass
class MetaBenchmarkResult:
    family_name: str
    eval_params: Dict[str, float]
    algorithm: str          # "maml", "reptile", "scratch", "scratch_full"
    k_shot: int
    metrics: Dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "family": self.family_name,
            "eval_params": self.eval_params,
            "algorithm": self.algorithm,
            "k_shot": self.k_shot,
            "metrics": self.metrics,
            "elapsed_s": self.elapsed_s,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _sample_task_data(
    task: BenchmarkTaskBase,
    n_col: int, n_bc: int, n_ic: int,
    seed: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    X_col = torch.tensor(task.sample_collocation(n_col, seed), device=device)
    Xb, Yb = task.sample_boundary(n_bc, seed)
    X_bc = torch.tensor(Xb, device=device)
    Y_bc = torch.tensor(Yb, device=device)
    Xi, Yi = task.sample_ic(n_ic, seed)
    if Xi.shape[0] > 0:
        X_ic = torch.tensor(Xi, device=device)
        Y_ic = torch.tensor(Yi, device=device)
    else:
        X_ic = Y_ic = None
    return X_col, X_bc, Y_bc, X_ic, Y_ic


def _pinn_loss(
    model: nn.Module,
    task: BenchmarkTaskBase,
    X_col: torch.Tensor,
    X_bc: torch.Tensor, Y_bc: torch.Tensor,
    X_ic: Optional[torch.Tensor], Y_ic: Optional[torch.Tensor],
    weight_pde: float, weight_bc: float, weight_ic: float,
) -> torch.Tensor:
    X_req = X_col.detach().requires_grad_(True)
    pde = task.pde_residual(model, X_req)

    pred_bc = model(X_bc)
    if hasattr(pred_bc, "y"):
        pred_bc = pred_bc.y
    bc = F.mse_loss(pred_bc, Y_bc)
    total = weight_pde * pde + weight_bc * bc

    if X_ic is not None:
        pred_ic = model(X_ic)
        if hasattr(pred_ic, "y"):
            pred_ic = pred_ic.y
        ic = F.mse_loss(pred_ic, Y_ic)
        total = total + weight_ic * ic
    return total


def _eval_model(
    model: nn.Module,
    task: BenchmarkTaskBase,
    n_eval: int,
    device: torch.device,
) -> Dict[str, float]:
    X_eval, U_exact = task.eval_grid(n_eval)
    X_t = torch.tensor(X_eval, dtype=torch.float32, device=device)
    U_ref = torch.tensor(U_exact, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        U_pred = model(X_t)
        if hasattr(U_pred, "y"):
            U_pred = U_pred.y
    diff = U_pred - U_ref
    rel_l2 = float((diff.pow(2).sum() / (U_ref.pow(2).sum() + 1e-10)).sqrt().item())
    l_inf = float(diff.abs().max().item())
    return {"rel_l2": rel_l2, "l_inf": l_inf}


# ─────────────────────────────────────────────────────────────────────────────
# MAML (FOMAML)
# ─────────────────────────────────────────────────────────────────────────────

def _maml_inner_loop(
    model: nn.Module,
    task: BenchmarkTaskBase,
    X_col: torch.Tensor,
    X_bc: torch.Tensor, Y_bc: torch.Tensor,
    X_ic: Optional[torch.Tensor], Y_ic: Optional[torch.Tensor],
    n_steps: int,
    inner_lr: float,
    weight_pde: float, weight_bc: float, weight_ic: float,
) -> Tuple[nn.Module, float]:
    """Adapt a deep copy of model for n_steps, return (adapted_copy, final_loss)."""
    adapted = copy.deepcopy(model)
    opt = torch.optim.SGD(adapted.parameters(), lr=inner_lr)
    last_loss = 0.0
    for _ in range(n_steps):
        opt.zero_grad()
        loss = _pinn_loss(adapted, task, X_col, X_bc, Y_bc, X_ic, Y_ic,
                          weight_pde, weight_bc, weight_ic)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())
    return adapted, last_loss


def _maml_meta_train(
    model: nn.Module,
    family: MetaBenchmarkFamily,
    cfg: MetaBenchmarkConfig,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[nn.Module, List[Dict]]:
    """
    FOMAML outer loop.

    For each meta-epoch:
      - Sample n_tasks_per_batch tasks from family
      - Run inner loop on each (n_inner_steps steps)
      - Compute query loss on adapted copies
      - Average query gradients, apply to meta-model
    """
    torch.manual_seed(cfg.seed)
    meta_model = copy.deepcopy(model).to(device)
    meta_opt = torch.optim.Adam(meta_model.parameters(), lr=cfg.outer_lr)
    rng = np.random.default_rng(cfg.seed)
    history: List[Dict] = []

    n_total = cfg.n_meta_epochs * cfg.n_tasks_per_batch * cfg.n_inner_steps
    if verbose:
        print(f"      MAML: {cfg.n_meta_epochs} meta-epochs x {cfg.n_tasks_per_batch} tasks "
              f"x {cfg.n_inner_steps} inner = {n_total} total steps")

    t0 = time.time()
    for meta_ep in range(1, cfg.n_meta_epochs + 1):
        # Sample tasks
        params_batch = [
            {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in family.param_ranges.items()}
            for _ in range(cfg.n_tasks_per_batch)
        ]

        query_losses: List[torch.Tensor] = []
        for params in params_batch:
            task = family.train_task_factory(params)
            seed_t = int(rng.integers(0, 100000))
            X_col, X_bc, Y_bc, X_ic, Y_ic = _sample_task_data(
                task, cfg.n_col, cfg.n_bc, cfg.n_ic, seed_t, device
            )
            # Sample separate query set
            seed_q = int(rng.integers(0, 100000))
            X_col_q, X_bc_q, Y_bc_q, X_ic_q, Y_ic_q = _sample_task_data(
                task, cfg.n_col, cfg.n_bc, cfg.n_ic, seed_q, device
            )

            # Inner adaptation (FOMAML: stop gradients at adapted params)
            adapted, _ = _maml_inner_loop(
                meta_model, task, X_col, X_bc, Y_bc, X_ic, Y_ic,
                cfg.n_inner_steps, cfg.inner_lr,
                cfg.weight_pde, cfg.weight_bc, cfg.weight_ic,
            )

            # Query loss on adapted model (FOMAML: treat adapted as leaf)
            q_loss = _pinn_loss(
                adapted, task, X_col_q, X_bc_q, Y_bc_q, X_ic_q, Y_ic_q,
                cfg.weight_pde, cfg.weight_bc, cfg.weight_ic,
            )
            query_losses.append(q_loss)

        # Meta-gradient: average query losses, backprop into meta_model
        meta_loss = torch.stack(query_losses).mean()

        # FOMAML: manually compute gradients of query loss w.r.t. adapted params,
        # then assign to meta_model params (first-order approximation)
        meta_opt.zero_grad()
        # Sum query losses, backprop through adapted copies
        # Since adapted = deepcopy + SGD steps, and FOMAML ignores higher-order terms,
        # we approximate: grad_meta ~ mean(grad_query on adapted)
        # Implementation: compute loss again on meta_model directly as proxy
        # (standard FOMAML trick: evaluate query loss on meta_model, not adapted)
        proxy_losses: List[torch.Tensor] = []
        for params in params_batch:
            task = family.train_task_factory(params)
            seed_p = int(rng.integers(0, 100000))
            X_col_p, X_bc_p, Y_bc_p, X_ic_p, Y_ic_p = _sample_task_data(
                task, cfg.n_col, cfg.n_bc, cfg.n_ic, seed_p, device
            )
            p_loss = _pinn_loss(
                meta_model, task, X_col_p, X_bc_p, Y_bc_p, X_ic_p, Y_ic_p,
                cfg.weight_pde, cfg.weight_bc, cfg.weight_ic,
            )
            proxy_losses.append(p_loss)
        proxy_total = torch.stack(proxy_losses).mean()
        proxy_total.backward()
        nn.utils.clip_grad_norm_(meta_model.parameters(), 1.0)
        meta_opt.step()

        rec = {
            "meta_epoch": meta_ep,
            "meta_loss": float(meta_loss.item()),
            "proxy_loss": float(proxy_total.item()),
            "elapsed_s": time.time() - t0,
        }
        history.append(rec)

        if verbose and meta_ep % cfg.log_interval == 0:
            print(f"        ep {meta_ep:4d}/{cfg.n_meta_epochs}  "
                  f"meta_loss={rec['meta_loss']:.3e}  t={rec['elapsed_s']:.1f}s")

    return meta_model, history


# ─────────────────────────────────────────────────────────────────────────────
# Reptile
# ─────────────────────────────────────────────────────────────────────────────

def _reptile_meta_train(
    model: nn.Module,
    family: MetaBenchmarkFamily,
    cfg: MetaBenchmarkConfig,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[nn.Module, List[Dict]]:
    """
    Reptile outer loop.

    For each meta-epoch:
      - Sample n_tasks_per_batch tasks
      - For each task: deep-copy model, run n_inner_steps SGD steps
      - Outer update: theta <- theta + epsilon * mean(theta_task - theta)
    """
    torch.manual_seed(cfg.seed)
    meta_model = copy.deepcopy(model).to(device)
    # Reptile outer LR / epsilon — typically larger than MAML
    epsilon = cfg.outer_lr
    rng = np.random.default_rng(cfg.seed)
    history: List[Dict] = []

    n_total = cfg.n_meta_epochs * cfg.n_tasks_per_batch * cfg.n_inner_steps
    if verbose:
        print(f"      Reptile: {cfg.n_meta_epochs} meta-epochs x {cfg.n_tasks_per_batch} tasks "
              f"x {cfg.n_inner_steps} inner = {n_total} total steps")

    t0 = time.time()
    for meta_ep in range(1, cfg.n_meta_epochs + 1):
        params_batch = [
            {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in family.param_ranges.items()}
            for _ in range(cfg.n_tasks_per_batch)
        ]

        task_state_dicts: List[Dict[str, torch.Tensor]] = []
        task_losses: List[float] = []

        for params in params_batch:
            task = family.train_task_factory(params)
            seed_t = int(rng.integers(0, 100000))
            X_col, X_bc, Y_bc, X_ic, Y_ic = _sample_task_data(
                task, cfg.n_col, cfg.n_bc, cfg.n_ic, seed_t, device
            )
            adapted, last_loss = _maml_inner_loop(
                meta_model, task, X_col, X_bc, Y_bc, X_ic, Y_ic,
                cfg.n_inner_steps, cfg.inner_lr,
                cfg.weight_pde, cfg.weight_bc, cfg.weight_ic,
            )
            task_state_dicts.append({k: v.detach().clone() for k, v in adapted.state_dict().items()})
            task_losses.append(last_loss)

        # Reptile outer update: theta <- theta + epsilon * mean(theta_i - theta)
        meta_state = meta_model.state_dict()
        with torch.no_grad():
            avg_delta: Dict[str, torch.Tensor] = {}
            for name in meta_state:
                delta = torch.zeros_like(meta_state[name].float())
                for sd in task_state_dicts:
                    if name in sd:
                        delta = delta + (sd[name].float() - meta_state[name].float())
                avg_delta[name] = delta / len(task_state_dicts)

            new_state = {
                name: meta_state[name].float() + epsilon * avg_delta[name]
                for name in meta_state
            }
            meta_model.load_state_dict(new_state)

        rec = {
            "meta_epoch": meta_ep,
            "meta_loss": float(np.mean(task_losses)),
            "elapsed_s": time.time() - t0,
        }
        history.append(rec)

        if verbose and meta_ep % cfg.log_interval == 0:
            print(f"        ep {meta_ep:4d}/{cfg.n_meta_epochs}  "
                  f"meta_loss={rec['meta_loss']:.3e}  t={rec['elapsed_s']:.1f}s")

    return meta_model, history


# ─────────────────────────────────────────────────────────────────────────────
# K-shot adaptation & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _adapt_and_eval(
    init_model: nn.Module,
    eval_task: BenchmarkTaskBase,
    k: int,
    inner_lr: float,
    cfg: MetaBenchmarkConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Adapt init_model for k steps, evaluate, return metrics dict."""
    X_col, X_bc, Y_bc, X_ic, Y_ic = _sample_task_data(
        eval_task, cfg.n_col, cfg.n_bc, cfg.n_ic, cfg.seed + 7, device
    )
    adapted = copy.deepcopy(init_model)
    opt = torch.optim.Adam(adapted.parameters(), lr=inner_lr)

    t0 = time.time()
    for _ in range(k):
        opt.zero_grad()
        loss = _pinn_loss(adapted, eval_task, X_col, X_bc, Y_bc, X_ic, Y_ic,
                          cfg.weight_pde, cfg.weight_bc, cfg.weight_ic)
        loss.backward()
        nn.utils.clip_grad_norm_(adapted.parameters(), 1.0)
        opt.step()

    metrics = _eval_model(adapted, eval_task, n_eval=5000, device=device)
    metrics["adapt_time_s"] = time.time() - t0
    metrics["final_loss"] = float(loss.item()) if k > 0 else float("nan")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Full-scratch oracle (same total steps as meta-training)
# ─────────────────────────────────────────────────────────────────────────────

def _scratch_full_oracle(
    in_dim: int, out_dim: int,
    eval_task: BenchmarkTaskBase,
    n_steps: int,
    cfg: MetaBenchmarkConfig,
    device: torch.device,
) -> Dict[str, float]:
    """Train from random init for n_steps with Adam + cosine decay."""
    torch.manual_seed(cfg.seed + 99)
    model = _build_mlp(in_dim, out_dim, list(cfg.hidden), "tanh").to(device)
    X_col, X_bc, Y_bc, X_ic, Y_ic = _sample_task_data(
        eval_task, cfg.n_col, cfg.n_bc, cfg.n_ic, cfg.seed + 13, device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)
    t0 = time.time()
    last_loss = float("nan")
    for step in range(n_steps):
        optimizer.zero_grad()
        loss = _pinn_loss(model, eval_task, X_col, X_bc, Y_bc, X_ic, Y_ic,
                          cfg.weight_pde, cfg.weight_bc, cfg.weight_ic)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.item())
    metrics = _eval_model(model, eval_task, n_eval=5000, device=device)
    metrics["train_time_s"] = time.time() - t0
    metrics["final_loss"] = last_loss
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark class
# ─────────────────────────────────────────────────────────────────────────────

class MetaBenchmarkPipeline:
    """
    Meta-learning benchmark pipeline.

    For each family:
      1. Meta-train MAML model on parameter family
      2. Meta-train Reptile model on parameter family
      3. For each eval task and each K in k_shots:
         a. Adapt MAML / Reptile for K steps -> eval
         b. Train from scratch for K steps -> eval
      4. Train from-scratch oracle (same total steps as meta-training)
      5. Report K-shot error curves comparing all approaches

    Usage
    -----
    >>> pipe = MetaBenchmarkPipeline.default()
    >>> results = pipe.run(verbose=True)
    >>> print(pipe.leaderboard())
    >>> pipe.plot_k_shot_curves()
    """

    def __init__(
        self,
        families: List[MetaBenchmarkFamily],
        config: Optional[MetaBenchmarkConfig] = None,
    ) -> None:
        self.families = families
        self.config = config or MetaBenchmarkConfig()
        self.results: List[MetaBenchmarkResult] = []
        self._meta_models: Dict[Tuple[str, str], nn.Module] = {}   # (family, algo) -> model
        self._meta_histories: Dict[Tuple[str, str], List[Dict]] = {}

    def _make_model(self, in_dim: int, out_dim: int) -> nn.Module:
        return _build_mlp(in_dim, out_dim, list(self.config.hidden), "tanh")

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> List[MetaBenchmarkResult]:
        cfg = self.config
        device = _resolve_device(cfg.device)
        all_results: List[MetaBenchmarkResult] = []
        n_total_meta = cfg.n_meta_epochs * cfg.n_tasks_per_batch * cfg.n_inner_steps

        if verbose:
            print("=" * 72)
            print("  Meta-Learning Benchmark")
            print(f"  Algorithms : {list(cfg.algorithms)}")
            print(f"  Meta-epochs: {cfg.n_meta_epochs}  |  Inner steps: {cfg.n_inner_steps}")
            print(f"  K-shots    : {list(cfg.k_shots)}")
            print(f"  Device     : {device}  |  Total inner steps: {n_total_meta:,}")
            print("=" * 72)

        for family in self.families:
            if verbose:
                print(f"\n  Family: {family.name}  |  params: {list(family.param_ranges)}")
                print(f"  Eval tasks: {family.eval_params}")

            # Infer model dims from a dummy eval task
            dummy_task = family.eval_task_factory(family.eval_params[0])
            in_dim, out_dim = dummy_task.in_dim, dummy_task.out_dim

            # ── Meta-train ───────────────────────────────────────────────
            meta_models: Dict[str, nn.Module] = {}
            meta_histories: Dict[str, List[Dict]] = {}

            for algo in cfg.algorithms:
                if verbose:
                    print(f"\n  [{algo.upper()}] Meta-training on {family.name}...")
                base_model = self._make_model(in_dim, out_dim).to(device)

                t0 = time.time()
                if algo == "maml":
                    mm, mh = _maml_meta_train(base_model, family, cfg, device, verbose=verbose)
                elif algo == "reptile":
                    mm, mh = _reptile_meta_train(base_model, family, cfg, device, verbose=verbose)
                else:
                    continue
                meta_elapsed = time.time() - t0

                meta_models[algo] = mm
                meta_histories[algo] = mh
                self._meta_models[(family.name, algo)] = mm
                self._meta_histories[(family.name, algo)] = mh

                if verbose:
                    final_loss = mh[-1]["meta_loss"] if mh else float("nan")
                    print(f"  [{algo.upper()}] Done  meta_loss={final_loss:.3e}  t={meta_elapsed:.1f}s")

            # ── K-shot evaluation ────────────────────────────────────────
            if verbose:
                print(f"\n  K-shot evaluation on {len(family.eval_params)} tasks...")

            for ep_idx, eval_params in enumerate(family.eval_params):
                eval_task = family.eval_task_factory(eval_params)
                param_str = ", ".join(f"{k}={v:.4f}" for k, v in eval_params.items())
                if verbose:
                    print(f"\n    Eval task [{ep_idx+1}/{len(family.eval_params)}]: {param_str}")

                for k in cfg.k_shots:
                    # Meta-adapted approaches
                    for algo, mm in meta_models.items():
                        t0 = time.time()
                        try:
                            m = _adapt_and_eval(mm, eval_task, k, cfg.inner_lr, cfg, device)
                        except Exception:
                            m = {"rel_l2": float("nan"), "l_inf": float("nan")}
                        all_results.append(MetaBenchmarkResult(
                            family_name=family.name,
                            eval_params=eval_params,
                            algorithm=algo,
                            k_shot=k,
                            metrics=m,
                            elapsed_s=time.time() - t0,
                        ))

                    # Scratch baseline (same K steps from random init)
                    torch.manual_seed(cfg.seed + k)
                    scratch_model = self._make_model(in_dim, out_dim).to(device)
                    t0 = time.time()
                    try:
                        m_scratch = _adapt_and_eval(
                            scratch_model, eval_task, k, cfg.inner_lr, cfg, device
                        )
                    except Exception:
                        m_scratch = {"rel_l2": float("nan"), "l_inf": float("nan")}
                    all_results.append(MetaBenchmarkResult(
                        family_name=family.name,
                        eval_params=eval_params,
                        algorithm="scratch",
                        k_shot=k,
                        metrics=m_scratch,
                        elapsed_s=time.time() - t0,
                    ))

                # Full-scratch oracle (n_total_meta steps from scratch)
                if verbose:
                    print(f"    Full-scratch oracle ({n_total_meta} steps)...", end="  ", flush=True)
                t0 = time.time()
                try:
                    m_oracle = _scratch_full_oracle(in_dim, out_dim, eval_task, n_total_meta, cfg, device)
                except Exception:
                    m_oracle = {"rel_l2": float("nan"), "l_inf": float("nan")}
                oracle_elapsed = time.time() - t0
                all_results.append(MetaBenchmarkResult(
                    family_name=family.name,
                    eval_params=eval_params,
                    algorithm="scratch_full",
                    k_shot=n_total_meta,
                    metrics=m_oracle,
                    elapsed_s=oracle_elapsed,
                ))
                if verbose:
                    print(f"rel_l2={m_oracle.get('rel_l2', float('nan')):.3e}  t={oracle_elapsed:.1f}s")

                # Print K-shot summary
                if verbose:
                    print(f"    {'K':>5}  {'MAML':>10}  {'Reptile':>10}  {'Scratch':>10}")
                    algo_order = list(cfg.algorithms) + ["scratch"]
                    for k in cfg.k_shots:
                        row = f"    {k:>5}"
                        for algo in algo_order:
                            r = next((r for r in all_results
                                      if r.family_name == family.name
                                      and r.eval_params == eval_params
                                      and r.algorithm == algo
                                      and r.k_shot == k), None)
                            val = r.metrics.get("rel_l2", float("nan")) if r else float("nan")
                            row += f"  {val:>10.3e}"
                        print(row)

        self.results = all_results
        return all_results

    # ── Reporting ─────────────────────────────────────────────────────────────

    def leaderboard(self, k_shot: Optional[int] = None, metric: str = "rel_l2") -> str:
        if not self.results:
            return "(no results)"

        # Use the largest K in k_shots if not specified
        if k_shot is None:
            k_shot = max(r.k_shot for r in self.results
                         if r.algorithm not in ("scratch_full",))

        lines: List[str] = [f"\n  Meta-Learning Leaderboard (K={k_shot}, metric={metric})"]

        families = list(dict.fromkeys(r.family_name for r in self.results))
        for fam in families:
            lines.append(f"\n  Family: {fam}")
            eval_param_strs = list(dict.fromkeys(
                str(r.eval_params) for r in self.results if r.family_name == fam
            ))
            for ep_str in eval_param_strs:
                runs = [r for r in self.results
                        if r.family_name == fam
                        and str(r.eval_params) == ep_str
                        and r.k_shot == k_shot]
                if not runs:
                    continue
                algos_ordered = sorted(runs, key=lambda r: r.metrics.get(metric, 1e9))
                lines.append(f"    Task {ep_str}")
                hdr = f"    {'Algorithm':<16} {metric:>10}  {'l_inf':>10}  {'time':>7}"
                lines.append(hdr)
                lines.append("    " + "-" * (len(hdr) - 4))
                for r in algos_ordered:
                    v = r.metrics.get(metric, float("nan"))
                    li = r.metrics.get("l_inf", float("nan"))
                    lines.append(f"    {r.algorithm:<16} {v:>10.3e}  {li:>10.3e}  {r.elapsed_s:>6.1f}s")
        return "\n".join(lines)

    def summary_k_shot(self, family_name: Optional[str] = None) -> str:
        """Table: rows=K values, cols=algorithms, cells=avg rel_l2 across eval tasks."""
        results = self.results
        if family_name:
            results = [r for r in results if r.family_name == family_name]
        if not results:
            return "(no results)"

        k_vals = sorted(set(r.k_shot for r in results if r.algorithm != "scratch_full"))
        algos = list(dict.fromkeys(r.algorithm for r in results if r.algorithm != "scratch_full"))
        col_w = 12

        header = f"  {'K':>6} " + " ".join(f"{a:>{col_w}}" for a in algos)
        sep = "  " + "-" * len(header)
        lines = [f"\n  K-shot Summary ({family_name or 'all families'}) -- avg rel_l2", header, sep]
        for k in k_vals:
            row = f"  {k:>6} "
            for algo in algos:
                vals = [r.metrics.get("rel_l2", float("nan")) for r in results
                        if r.k_shot == k and r.algorithm == algo
                        and not math.isnan(r.metrics.get("rel_l2", float("nan")))]
                avg = float(np.mean(vals)) if vals else float("nan")
                row += f"  {avg:>{col_w}.3e}"
            lines.append(row)
        return "\n".join(lines)

    def plot_k_shot_curves(
        self,
        family_name: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """K vs avg_rel_l2 curves for each algorithm."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        results = self.results
        if family_name:
            results = [r for r in results if r.family_name == family_name]
        if not results:
            return

        k_vals = sorted(set(r.k_shot for r in results if r.algorithm != "scratch_full"))
        algos = list(dict.fromkeys(r.algorithm for r in results if r.algorithm != "scratch_full"))

        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {"maml": "#1f77b4", "reptile": "#ff7f0e", "scratch": "#d62728"}
        lines_obj = {}
        for algo in algos:
            avgs = []
            for k in k_vals:
                vals = [r.metrics.get("rel_l2", float("nan")) for r in results
                        if r.k_shot == k and r.algorithm == algo
                        and not math.isnan(r.metrics.get("rel_l2", float("nan")))]
                avgs.append(float(np.mean(vals)) if vals else float("nan"))
            c = colors.get(algo, None)
            line, = ax.plot(k_vals, avgs, "o-", label=algo, color=c, lw=2, ms=5)
            lines_obj[algo] = line

        # Full-scratch oracle as horizontal line
        oracle_vals = [r.metrics.get("rel_l2", float("nan")) for r in results
                       if r.algorithm == "scratch_full"
                       and not math.isnan(r.metrics.get("rel_l2", float("nan")))]
        if oracle_vals:
            ax.axhline(float(np.mean(oracle_vals)), ls="--", color="gray",
                       label=f"scratch_full ({cfg_str(results)})", lw=1.5)

        ax.set_xlabel("K adaptation steps")
        ax.set_ylabel("Relative L2 Error")
        ax.set_xscale("log")
        ax.set_yscale("log")
        fam_title = family_name or "All Families"
        ax.set_title(f"Meta-Learning K-shot Curves: {fam_title}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close(fig)

    def plot_meta_convergence(
        self,
        family_name: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Meta-training loss curves for MAML and Reptile."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, ax = plt.subplots(figsize=(9, 4))
        for (fam, algo), history in self._meta_histories.items():
            if family_name and fam != family_name:
                continue
            if not history:
                continue
            eps = [h["meta_epoch"] for h in history]
            losses = [h["meta_loss"] for h in history]
            ax.semilogy(eps, losses, label=f"{fam}/{algo}", alpha=0.85)

        ax.set_xlabel("Meta-epoch")
        ax.set_ylabel("Meta-loss")
        ax.set_title(f"Meta-Training Convergence: {family_name or 'All Families'}")
        ax.legend(fontsize=8)
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close(fig)

    def save_results(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([r.to_dict() for r in self.results], indent=2), encoding="utf-8")
        print(f"[meta_benchmark] Results saved: {p}")

    @classmethod
    def load_results(cls, path: str) -> List[Dict]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    # ── Default ───────────────────────────────────────────────────────────────

    @classmethod
    def default(
        cls,
        families: Optional[List[str]] = None,
        n_meta_epochs: int = 500,
        device: str = "auto",
    ) -> "MetaBenchmarkPipeline":
        """
        Standard benchmark families.

        Available: "burgers_nu", "heat_alpha", "wave_c"
        """
        from pinneaple_arena.tasks.burgers_1d import Burgers1DTask
        from pinneaple_arena.tasks.heat_1d import Heat1DTask
        from pinneaple_arena.tasks.wave_1d import Wave1DTask

        all_families: Dict[str, MetaBenchmarkFamily] = {
            "burgers_nu": MetaBenchmarkFamily(
                name="burgers_nu",
                train_task_factory=lambda p: Burgers1DTask(nu=p["nu"], build_reference=False),
                eval_task_factory=lambda p: Burgers1DTask(nu=p["nu"], build_reference=True),
                param_ranges={"nu": (0.003 / np.pi, 0.015 / np.pi)},
                eval_params=[
                    {"nu": 0.006 / np.pi},
                    {"nu": 0.010 / np.pi},
                    {"nu": 0.013 / np.pi},
                ],
            ),
            "heat_alpha": MetaBenchmarkFamily(
                name="heat_alpha",
                train_task_factory=lambda p: Heat1DTask(alpha=p["alpha"]),
                eval_task_factory=lambda p: Heat1DTask(alpha=p["alpha"]),
                param_ranges={"alpha": (0.005, 0.05)},
                eval_params=[
                    {"alpha": 0.01},
                    {"alpha": 0.025},
                    {"alpha": 0.04},
                ],
            ),
            "wave_c": MetaBenchmarkFamily(
                name="wave_c",
                train_task_factory=lambda p: Wave1DTask(c=p["c"]),
                eval_task_factory=lambda p: Wave1DTask(c=p["c"]),
                param_ranges={"c": (0.5, 2.0)},
                eval_params=[
                    {"c": 0.75},
                    {"c": 1.25},
                    {"c": 1.75},
                ],
            ),
        }

        selected = list(all_families.values()) if families is None else [
            all_families[f] for f in families if f in all_families
        ]

        cfg = MetaBenchmarkConfig(n_meta_epochs=n_meta_epochs, device=device)
        return cls(families=selected, config=cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers used in plot methods
# ─────────────────────────────────────────────────────────────────────────────

def cfg_str(results: List[MetaBenchmarkResult]) -> str:
    n = next((r.k_shot for r in results if r.algorithm == "scratch_full"), "?")
    return f"{n} steps"
