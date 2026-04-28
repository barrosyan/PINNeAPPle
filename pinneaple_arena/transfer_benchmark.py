"""Transfer Learning Benchmark for PINNs.

Trains a source model from scratch, then fine-tunes it to a family of target
problems using different strategies from ``pinneaple_transfer``.  Compares
each transfer approach against two from-scratch baselines.

Strategies
----------
scratch_budget : train from scratch with same epoch budget as fine-tuning
scratch_full   : train from scratch with source + fine-tune epochs (total budget)
finetune       : transfer all layers, low LR (TransferTrainer strategy="finetune")
partial_freeze : freeze first half of linear layers, fine-tune rest
feature_extract: freeze all except last linear layer
progressive    : frozen backbone for warmup_epochs, then full unfreeze

Key metrics
-----------
rel_l2            : relative L2 error vs exact/reference solution
speedup_factor    : steps_to_threshold(scratch_budget) / steps_to_threshold(transfer)
trainable_ratio   : trainable params / total params
convergence_epoch : first epoch where pde_residual < threshold
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

from pinneaple_transfer.freeze import (
    count_trainable,
    freeze_all_except,
    freeze_layers,
    layer_lr_groups,
)
from pinneaple_transfer.config import TransferConfig
from pinneaple_transfer.trainer import TransferTrainer

from pinneaple_arena.benchmark import (
    BenchmarkTaskBase,
    _build_mlp,
    _ResNetPINN,
    _resolve_device,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config & dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransferBenchmarkConfig:
    # Source model training
    n_source_epochs: int = 3000
    lr_source: float = 1e-3

    # Fine-tuning / target training budget
    n_finetune_epochs: int = 1000
    lr_finetune: float = 5e-5      # for "finetune" strategy
    lr_base: float = 1e-4          # for "partial_freeze" / "feature_extract"
    lr_scratch: float = 1e-3

    # Strategies to compare (always includes scratch baselines)
    strategies: List[str] = field(default_factory=lambda: [
        "finetune", "partial_freeze", "feature_extract", "progressive"
    ])

    # Progressive warmup (epochs before unfreezing)
    warmup_epochs: int = 200

    # Data sampling
    n_col: int = 3000
    n_bc: int = 500
    n_ic: int = 500

    # Loss weights
    weight_pde: float = 1.0
    weight_bc: float = 10.0
    weight_ic: float = 10.0

    # Model architecture (for MLP)
    hidden: Tuple[int, ...] = (128, 128, 128, 128)

    # Evaluation
    n_eval: int = 10000
    convergence_threshold: float = 5e-3

    # Misc
    device: str = "auto"
    seed: int = 42
    log_interval: int = 250


@dataclass
class TransferScenario:
    """One transfer scenario: one source task and N target tasks."""
    name: str
    source_task: BenchmarkTaskBase
    target_tasks: List[BenchmarkTaskBase]
    source_label: str = ""
    target_labels: List[str] = field(default_factory=list)


@dataclass
class TransferBenchmarkResult:
    scenario_name: str
    target_label: str
    strategy: str
    metrics: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, float]] = field(default_factory=list)
    elapsed_s: float = 0.0
    n_params_trainable: int = 0
    n_params_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "target": self.target_label,
            "strategy": self.strategy,
            "metrics": self.metrics,
            "elapsed_s": self.elapsed_s,
            "n_params_trainable": self.n_params_trainable,
            "n_params_total": self.n_params_total,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Layer-freezing helpers (architecture-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _get_linear_param_prefixes(model: nn.Module) -> List[str]:
    """Return sorted parameter-name prefixes for each Linear layer."""
    prefixes: List[str] = []
    seen: set = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prefix = name if name else "linear"
            if prefix not in seen:
                prefixes.append(prefix)
                seen.add(prefix)
    return prefixes


def _apply_freeze_strategy(model: nn.Module, strategy: str, cfg: TransferBenchmarkConfig) -> None:
    """Apply freezing strategy in-place."""
    # Start from all trainable
    for p in model.parameters():
        p.requires_grad_(True)

    prefixes = _get_linear_param_prefixes(model)
    n = len(prefixes)

    if strategy == "finetune":
        pass  # all trainable

    elif strategy == "partial_freeze":
        # Freeze first half of linear layers
        half = max(1, n // 2)
        freeze_layers(model, prefixes[:half])

    elif strategy == "feature_extract":
        # Freeze everything except last linear layer
        if n >= 2:
            freeze_layers(model, prefixes[:-1])

    elif strategy == "progressive":
        # Start fully frozen; TransferTrainer handles warmup unfreezing
        freeze_all_except(model, [])


# ─────────────────────────────────────────────────────────────────────────────
# Physics loss factory (closure over sampled data)
# ─────────────────────────────────────────────────────────────────────────────

def _make_physics_fn(
    task: BenchmarkTaskBase,
    X_col: torch.Tensor,
    X_bc: torch.Tensor,
    Y_bc: torch.Tensor,
    X_ic: Optional[torch.Tensor],
    Y_ic: Optional[torch.Tensor],
    weight_pde: float,
    weight_bc: float,
    weight_ic: float,
) -> Callable:
    """Return a closure-based physics_fn compatible with TransferTrainer."""
    def physics_fn(model: nn.Module, _batch: Dict) -> Dict[str, torch.Tensor]:
        # PDE residual
        X_req = X_col.detach().requires_grad_(True)
        pde_loss = task.pde_residual(model, X_req)

        # BC
        pred_bc = model(X_bc)
        if hasattr(pred_bc, "y"):
            pred_bc = pred_bc.y
        bc_loss = F.mse_loss(pred_bc, Y_bc)

        total = weight_pde * pde_loss + weight_bc * bc_loss

        # IC
        if X_ic is not None:
            pred_ic = model(X_ic)
            if hasattr(pred_ic, "y"):
                pred_ic = pred_ic.y
            ic_loss = F.mse_loss(pred_ic, Y_ic)
            total = total + weight_ic * ic_loss
        else:
            ic_loss = torch.zeros(1, device=X_col.device).squeeze()

        return {
            "total": total,
            "pde": pde_loss.detach(),
            "bc": bc_loss.detach(),
            "ic": ic_loss.detach() if isinstance(ic_loss, torch.Tensor) else ic_loss,
        }
    return physics_fn


def _sample_task_data(
    task: BenchmarkTaskBase,
    n_col: int, n_bc: int, n_ic: int,
    seed: int, device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    X_col = torch.tensor(task.sample_collocation(n_col, seed), device=device)
    Xb_np, Yb_np = task.sample_boundary(n_bc, seed)
    X_bc = torch.tensor(Xb_np, device=device)
    Y_bc = torch.tensor(Yb_np, device=device)
    Xi_np, Yi_np = task.sample_ic(n_ic, seed)
    if Xi_np.shape[0] > 0:
        X_ic = torch.tensor(Xi_np, device=device)
        Y_ic = torch.tensor(Yi_np, device=device)
    else:
        X_ic = Y_ic = None
    return X_col, X_bc, Y_bc, X_ic, Y_ic


# ─────────────────────────────────────────────────────────────────────────────
# Scratch training loop
# ─────────────────────────────────────────────────────────────────────────────

def _train_scratch(
    model: nn.Module,
    task: BenchmarkTaskBase,
    X_col: torch.Tensor,
    X_bc: torch.Tensor, Y_bc: torch.Tensor,
    X_ic: Optional[torch.Tensor], Y_ic: Optional[torch.Tensor],
    n_epochs: int,
    lr: float,
    cfg: TransferBenchmarkConfig,
    verbose: bool = False,
) -> Tuple[List[Dict], int]:
    """Train a model from scratch with Adam + cosine LR decay."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr * 1e-2)
    convergence_epoch = -1
    history: List[Dict] = []

    for ep in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()

        X_req = X_col.detach().requires_grad_(True)
        pde_loss = task.pde_residual(model, X_req)

        pred_bc = model(X_bc)
        if hasattr(pred_bc, "y"):
            pred_bc = pred_bc.y
        bc_loss = F.mse_loss(pred_bc, Y_bc)
        total = cfg.weight_pde * pde_loss + cfg.weight_bc * bc_loss

        if X_ic is not None:
            pred_ic = model(X_ic)
            if hasattr(pred_ic, "y"):
                pred_ic = pred_ic.y
            ic_loss = F.mse_loss(pred_ic, Y_ic)
            total = total + cfg.weight_ic * ic_loss
        else:
            ic_loss = torch.zeros(1)

        total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        rec = {
            "epoch": ep,
            "loss": float(total.item()),
            "pde": float(pde_loss.item()),
            "bc": float(bc_loss.item()),
        }
        history.append(rec)
        if convergence_epoch < 0 and rec["pde"] < cfg.convergence_threshold:
            convergence_epoch = ep

        if verbose and ep % cfg.log_interval == 0:
            print(f"        ep {ep:5d}/{n_epochs}  pde={rec['pde']:.3e}")

    return history, convergence_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark class
# ─────────────────────────────────────────────────────────────────────────────

class TransferBenchmarkPipeline:
    """
    Benchmark comparing transfer learning strategies against from-scratch baselines.

    For each scenario (source_task → target_tasks):
      1. Train a source model from scratch (n_source_epochs)
      2. For each target task and each strategy:
         a. Clone source model, apply freeze strategy
         b. Fine-tune on target task for n_finetune_epochs via TransferTrainer
         c. Evaluate vs exact/reference solution
      3. Also run scratch_budget (n_finetune_epochs from random init)
         and scratch_full (n_source + n_finetune epochs from random init)
      4. Report and compare all strategies.

    Usage
    -----
    >>> pipe = TransferBenchmarkPipeline.default()
    >>> results = pipe.run(verbose=True)
    >>> print(pipe.leaderboard())
    >>> pipe.save_results("transfer_results.json")
    """

    def __init__(
        self,
        scenarios: List[TransferScenario],
        config: Optional[TransferBenchmarkConfig] = None,
    ) -> None:
        self.scenarios = scenarios
        self.config = config or TransferBenchmarkConfig()
        self.results: List[TransferBenchmarkResult] = []

    # ── Model factory ─────────────────────────────────────────────────────────

    def _make_model(self, in_dim: int, out_dim: int) -> nn.Module:
        cfg = self.config
        return _build_mlp(in_dim, out_dim, list(cfg.hidden), activation="tanh")

    def _eval_model(self, model: nn.Module, task: BenchmarkTaskBase, device: torch.device) -> Dict[str, float]:
        X_eval, U_exact = task.eval_grid(self.config.n_eval)
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

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> List[TransferBenchmarkResult]:
        cfg = self.config
        device = _resolve_device(cfg.device)
        all_results: List[TransferBenchmarkResult] = []

        n_strategies = len(cfg.strategies) + 2  # +2 for scratch_budget, scratch_full

        if verbose:
            print("=" * 72)
            print("  Transfer Learning Benchmark")
            print(f"  Source epochs : {cfg.n_source_epochs}  |  Finetune epochs : {cfg.n_finetune_epochs}")
            print(f"  Strategies    : {cfg.strategies + ['scratch_budget','scratch_full']}")
            print(f"  Device        : {device}")
            print("=" * 72)

        for scenario in self.scenarios:
            if verbose:
                print(f"\n  Scenario: {scenario.name}  (source: {scenario.source_label})")

            src_task = scenario.source_task
            in_dim, out_dim = src_task.in_dim, src_task.out_dim

            # --- 1. Sample source data and train source model ---
            X_col_s, X_bc_s, Y_bc_s, X_ic_s, Y_ic_s = _sample_task_data(
                src_task, cfg.n_col, cfg.n_bc, cfg.n_ic, cfg.seed, device
            )
            source_model = self._make_model(in_dim, out_dim).to(device)
            torch.manual_seed(cfg.seed)

            if verbose:
                print(f"    [1/2] Training source model ({cfg.n_source_epochs} ep)...")
            t0 = time.time()
            src_history, _ = _train_scratch(
                source_model, src_task, X_col_s, X_bc_s, Y_bc_s, X_ic_s, Y_ic_s,
                cfg.n_source_epochs, cfg.lr_source, cfg,
                verbose=verbose,
            )
            src_elapsed = time.time() - t0
            src_metrics = self._eval_model(source_model, src_task, device)
            if verbose:
                print(f"    Source rel_l2={src_metrics['rel_l2']:.3e}  t={src_elapsed:.1f}s")

            # --- 2. For each target task, run all strategies ---
            for t_idx, tgt_task in enumerate(scenario.target_tasks):
                tgt_label = (
                    scenario.target_labels[t_idx]
                    if t_idx < len(scenario.target_labels)
                    else f"target_{t_idx}"
                )
                if verbose:
                    print(f"\n    [2/2] Target: {tgt_label}")

                X_col_t, X_bc_t, Y_bc_t, X_ic_t, Y_ic_t = _sample_task_data(
                    tgt_task, cfg.n_col, cfg.n_bc, cfg.n_ic, cfg.seed + t_idx + 1, device
                )

                physics_fn = _make_physics_fn(
                    tgt_task, X_col_t, X_bc_t, Y_bc_t, X_ic_t, Y_ic_t,
                    cfg.weight_pde, cfg.weight_bc, cfg.weight_ic,
                )

                # ── Transfer strategies ─────────────────────────────────────
                for strategy in cfg.strategies:
                    if verbose:
                        print(f"      {strategy:<20}", end="  ", flush=True)

                    # Build transfer config
                    if strategy == "finetune":
                        tr_cfg = TransferConfig(
                            strategy="finetune",
                            finetune_lr=cfg.lr_finetune,
                            base_lr=cfg.lr_finetune,
                            epochs=cfg.n_finetune_epochs,
                            physics_weight=1.0,
                            data_weight=0.0,
                            device=str(device),
                            seed=cfg.seed,
                        )
                    elif strategy == "partial_freeze":
                        prefixes = _get_linear_param_prefixes(source_model)
                        half = max(1, len(prefixes) // 2)
                        tr_cfg = TransferConfig(
                            strategy="partial_freeze",
                            freeze_prefix=prefixes[:half],
                            finetune_lr=cfg.lr_base,
                            base_lr=cfg.lr_base,
                            epochs=cfg.n_finetune_epochs,
                            physics_weight=1.0,
                            data_weight=0.0,
                            device=str(device),
                            seed=cfg.seed,
                        )
                    elif strategy == "feature_extract":
                        prefixes = _get_linear_param_prefixes(source_model)
                        tr_cfg = TransferConfig(
                            strategy="feature_extract",
                            unfreeze_prefix=prefixes[-1:],
                            finetune_lr=cfg.lr_base,
                            base_lr=cfg.lr_base,
                            epochs=cfg.n_finetune_epochs,
                            physics_weight=1.0,
                            data_weight=0.0,
                            device=str(device),
                            seed=cfg.seed,
                        )
                    elif strategy == "progressive":
                        tr_cfg = TransferConfig(
                            strategy="progressive",
                            warmup_epochs=cfg.warmup_epochs,
                            finetune_lr=cfg.lr_finetune,
                            base_lr=cfg.lr_base,
                            epochs=cfg.n_finetune_epochs,
                            physics_weight=1.0,
                            data_weight=0.0,
                            device=str(device),
                            seed=cfg.seed,
                        )
                    else:
                        continue

                    t_start = time.time()
                    try:
                        trainer = TransferTrainer(source_model, tr_cfg)
                        trainer.prepare()
                        param_info = count_trainable(trainer.model)
                        ft_result = trainer.finetune(target_physics_fn=physics_fn)
                        trained_model = ft_result["model"]
                        raw_history = ft_result["history"]
                        # Normalize history to match our format
                        history = [
                            {
                                "epoch": h["epoch"],
                                "loss": h["loss_total"],
                                "pde": h.get("phys_pde", h["loss_physics"]),
                                "bc": h.get("phys_bc", 0.0),
                            }
                            for h in raw_history
                        ]
                        conv_ep = next(
                            (h["epoch"] for h in history if h["pde"] < cfg.convergence_threshold), -1
                        )
                        metrics = self._eval_model(trained_model, tgt_task, device)
                        metrics["convergence_epoch"] = float(conv_ep)
                        metrics["trainable_ratio"] = param_info["trainable"] / max(param_info["total"], 1)
                    except Exception as e:
                        if verbose:
                            print(f"ERROR: {e}")
                        history, conv_ep = [], -1
                        metrics = {"rel_l2": float("nan"), "l_inf": float("nan"),
                                   "convergence_epoch": -1.0, "trainable_ratio": 1.0}
                        param_info = {"trainable": 0, "total": 0}

                    metrics["train_time_s"] = time.time() - t_start
                    if verbose:
                        print(
                            f"rel_l2={metrics.get('rel_l2', float('nan')):.3e}  "
                            f"conv_ep={conv_ep}  "
                            f"trainable={param_info.get('trainable',0):,}/"
                            f"{param_info.get('total',0):,}  "
                            f"t={metrics['train_time_s']:.1f}s"
                        )

                    all_results.append(TransferBenchmarkResult(
                        scenario_name=scenario.name,
                        target_label=tgt_label,
                        strategy=strategy,
                        metrics=metrics,
                        history=history,
                        elapsed_s=metrics["train_time_s"],
                        n_params_trainable=param_info.get("trainable", 0),
                        n_params_total=param_info.get("total", 0),
                    ))

                # ── Scratch baselines ───────────────────────────────────────
                for scratch_tag, n_ep, lr in [
                    ("scratch_budget", cfg.n_finetune_epochs, cfg.lr_scratch),
                    ("scratch_full", cfg.n_source_epochs + cfg.n_finetune_epochs, cfg.lr_scratch),
                ]:
                    if verbose:
                        print(f"      {scratch_tag:<20}", end="  ", flush=True)
                    torch.manual_seed(cfg.seed + 99)
                    scratch_model = self._make_model(in_dim, out_dim).to(device)
                    t_start = time.time()
                    try:
                        sc_history, sc_conv = _train_scratch(
                            scratch_model, tgt_task,
                            X_col_t, X_bc_t, Y_bc_t, X_ic_t, Y_ic_t,
                            n_ep, lr, cfg,
                        )
                        sc_metrics = self._eval_model(scratch_model, tgt_task, device)
                        sc_metrics["convergence_epoch"] = float(sc_conv)
                        sc_metrics["trainable_ratio"] = 1.0
                    except Exception as e:
                        if verbose:
                            print(f"ERROR: {e}")
                        sc_history, sc_conv = [], -1
                        sc_metrics = {"rel_l2": float("nan"), "l_inf": float("nan"),
                                      "convergence_epoch": -1.0, "trainable_ratio": 1.0}
                    sc_metrics["train_time_s"] = time.time() - t_start
                    n_tot = sum(p.numel() for p in scratch_model.parameters())
                    if verbose:
                        print(
                            f"rel_l2={sc_metrics.get('rel_l2', float('nan')):.3e}  "
                            f"conv_ep={sc_conv}  t={sc_metrics['train_time_s']:.1f}s"
                        )
                    all_results.append(TransferBenchmarkResult(
                        scenario_name=scenario.name,
                        target_label=tgt_label,
                        strategy=scratch_tag,
                        metrics=sc_metrics,
                        history=sc_history,
                        elapsed_s=sc_metrics["train_time_s"],
                        n_params_trainable=n_tot,
                        n_params_total=n_tot,
                    ))

        self.results = all_results
        self._compute_speedup()
        return all_results

    def _compute_speedup(self) -> None:
        """Compute speedup_factor = steps_scratch_budget / steps_transfer per (scenario, target)."""
        by_key: Dict[Tuple[str, str], List[TransferBenchmarkResult]] = {}
        for r in self.results:
            k = (r.scenario_name, r.target_label)
            by_key.setdefault(k, []).append(r)

        for runs in by_key.values():
            scratch_b = next((r for r in runs if r.strategy == "scratch_budget"), None)
            ref_ep = scratch_b.metrics.get("convergence_epoch", -1) if scratch_b else -1
            for r in runs:
                if r.strategy.startswith("scratch"):
                    r.metrics["speedup_factor"] = 1.0
                else:
                    ep = r.metrics.get("convergence_epoch", -1)
                    if ref_ep > 0 and ep > 0:
                        r.metrics["speedup_factor"] = ref_ep / ep
                    else:
                        r.metrics["speedup_factor"] = float("nan")

    # ── Reporting ─────────────────────────────────────────────────────────────

    def leaderboard(self, metric: str = "rel_l2") -> str:
        if not self.results:
            return "(no results)"

        by_key: Dict[Tuple[str, str], List[TransferBenchmarkResult]] = {}
        for r in self.results:
            by_key.setdefault((r.scenario_name, r.target_label), []).append(r)

        lines: List[str] = ["\n  Transfer Learning Benchmark Results"]
        all_strategies = list(dict.fromkeys(r.strategy for r in self.results))

        for (scene, target), runs in by_key.items():
            lines.append(f"\n  Scenario: {scene} -> {target}")
            header = f"    {'Strategy':<22} {metric:>10}  {'l_inf':>10}  {'conv_ep':>8}  {'speedup':>8}  {'time':>7}"
            lines.append(header)
            lines.append("    " + "-" * (len(header) - 4))
            sorted_runs = sorted(runs, key=lambda r: r.metrics.get(metric, 1e9))
            for r in sorted_runs:
                v = r.metrics.get(metric, float("nan"))
                li = r.metrics.get("l_inf", float("nan"))
                conv = r.metrics.get("convergence_epoch", -1)
                speed = r.metrics.get("speedup_factor", float("nan"))
                conv_str = f"{conv:.0f}" if conv > 0 else "N/A"
                speed_str = f"{speed:.1f}x" if not math.isnan(speed) else "N/A"
                lines.append(
                    f"    {r.strategy:<22} {v:>10.3e}  {li:>10.3e}  "
                    f"{conv_str:>8}  {speed_str:>8}  {r.elapsed_s:>6.1f}s"
                )
        return "\n".join(lines)

    def plot_comparison(
        self,
        scenario_name: Optional[str] = None,
        metric: str = "rel_l2",
        save_path: Optional[str] = None,
    ) -> None:
        """Bar chart comparing strategies on each target."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        runs = self.results
        if scenario_name:
            runs = [r for r in runs if r.scenario_name == scenario_name]
        if not runs:
            return

        targets = list(dict.fromkeys(r.target_label for r in runs))
        strategies = list(dict.fromkeys(r.strategy for r in runs))
        n_t, n_s = len(targets), len(strategies)

        x = np.arange(n_t)
        width = 0.8 / n_s

        fig, ax = plt.subplots(figsize=(max(8, n_t * 2), 5))
        for i, strat in enumerate(strategies):
            vals = []
            for tgt in targets:
                r = next((r for r in runs if r.target_label == tgt and r.strategy == strat), None)
                vals.append(r.metrics.get(metric, float("nan")) if r else float("nan"))
            ax.bar(x + i * width, vals, width, label=strat, alpha=0.85)

        ax.set_xticks(x + width * (n_s - 1) / 2)
        ax.set_xticklabels(targets, rotation=15, ha="right")
        ax.set_ylabel(metric)
        ax.set_yscale("log")
        title = f"Transfer Learning: {scenario_name or 'All Scenarios'}"
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=min(3, n_s))
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
        else:
            plt.show()
        plt.close(fig)

    def plot_convergence(
        self,
        scenario_name: Optional[str] = None,
        target_label: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Loss convergence curves for all strategies on a target."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        runs = self.results
        if scenario_name:
            runs = [r for r in runs if r.scenario_name == scenario_name]
        if target_label:
            runs = [r for r in runs if r.target_label == target_label]
        if not runs:
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        for r in runs:
            if not r.history:
                continue
            epochs = [h.get("epoch", i) for i, h in enumerate(r.history)]
            pde = [h.get("pde", h.get("loss_physics", 0)) for h in r.history]
            ax.semilogy(epochs, pde, label=r.strategy, alpha=0.85)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("PDE Residual")
        title_parts = [p for p in [scenario_name, target_label] if p]
        ax.set_title("Transfer Convergence: " + " / ".join(title_parts))
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
        print(f"[transfer_benchmark] Results saved: {p}")

    @classmethod
    def load_results(cls, path: str) -> List[Dict]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    # ── Default scenarios ─────────────────────────────────────────────────────

    @classmethod
    def default(
        cls,
        scenarios: Optional[List[str]] = None,
        epochs_source: int = 3000,
        epochs_finetune: int = 1000,
        device: str = "auto",
    ) -> "TransferBenchmarkPipeline":
        """
        Standard benchmark scenarios.

        Available: "burgers_nu", "heat_alpha", "wave_c"
        """
        from pinneaple_arena.tasks.burgers_1d import Burgers1DTask
        from pinneaple_arena.tasks.heat_1d import Heat1DTask
        from pinneaple_arena.tasks.wave_1d import Wave1DTask

        all_scenarios: Dict[str, TransferScenario] = {
            "burgers_nu": TransferScenario(
                name="burgers_nu",
                source_task=Burgers1DTask(nu=0.01 / np.pi, build_reference=False),
                target_tasks=[
                    Burgers1DTask(nu=0.005 / np.pi),   # sharper shock
                    Burgers1DTask(nu=0.02 / np.pi),    # smoother solution
                ],
                source_label="nu=0.01/pi",
                target_labels=["nu=0.005/pi (sharper)", "nu=0.02/pi (smoother)"],
            ),
            "heat_alpha": TransferScenario(
                name="heat_alpha",
                source_task=Heat1DTask(alpha=0.01),
                target_tasks=[
                    Heat1DTask(alpha=0.005),  # slower diffusion
                    Heat1DTask(alpha=0.05),   # faster diffusion
                ],
                source_label="alpha=0.01",
                target_labels=["alpha=0.005 (slower)", "alpha=0.05 (faster)"],
            ),
            "wave_c": TransferScenario(
                name="wave_c",
                source_task=Wave1DTask(c=1.0),
                target_tasks=[
                    Wave1DTask(c=0.5),   # slower wave
                    Wave1DTask(c=2.0),   # faster wave
                ],
                source_label="c=1.0",
                target_labels=["c=0.5 (slower)", "c=2.0 (faster)"],
            ),
        }

        selected = list(all_scenarios.values()) if scenarios is None else [
            all_scenarios[s] for s in scenarios if s in all_scenarios
        ]

        cfg = TransferBenchmarkConfig(
            n_source_epochs=epochs_source,
            n_finetune_epochs=epochs_finetune,
            device=device,
        )
        return cls(scenarios=selected, config=cfg)
