"""Advanced automatic loss weight optimisation for PINNs.

New methods added on top of the existing weight_scheduler.py:

ReLoBRaLo  — Relative Loss Balancing via Residuals and Losses (Bischof & Kraus, 2021).
             Very cheap (no extra backward pass), works excellently in practice.
             Uses per-step loss change ratios with EMA smoothing.

SoftAdapt  — Adaptive loss weighting based on rates-of-change of individual losses
             (Heydari et al., 2019).  Focuses effort on the term that is improving
             most slowly.

AugmentedLagrangian — Treats BC/IC losses as inequality constraints solved via
             dual ascent (λ_i += ρ · L_i).  Gradually tightens constraints.
             Strong theoretical basis for constrained PDE problems.

PCGrad     — Gradient Surgery: projects conflicting task gradients onto each
             other's normal plane.  Eliminates gradient cancellation between PDE
             and BC terms.  Requires per-loss backward passes (more expensive).

InverseDirichlet — Sets weights proportional to the inverse of per-layer
             gradient statistics, balancing training "stiffness" across terms.
             (Van der Meer et al., 2022). Very cheap, one backward pass.

AutoBalancer — Meta-scheduler: monitors training health, automatically switches
               between methods and adjusts hyper-parameters when progress stalls.

Usage
-----
    from pinneaple_neural.trainer.loss_balancer import (
        ReLoBRaLo, SoftAdapt, AugmentedLagrangian, PCGrad, AutoBalancer,
    )
    from pinneaple_neural.trainer.weight_scheduler import WeightScheduler, WeightSchedulerConfig

    # Drop-in replacement — just change method name:
    sched = WeightScheduler(pinn, ["pde", "bc", "ic"],
                            WeightSchedulerConfig(method="relobralo"))

    # Or use directly:
    balancer = ReLoBRaLo(["pde", "bc", "ic"])
    total = balancer.step({"pde": l_pde, "bc": l_bc, "ic": l_ic})
    total.backward()
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ReLoBRaLo
# ─────────────────────────────────────────────────────────────────────────────

class ReLoBRaLo:
    """
    Relative Loss Balancing via Residuals and Losses.

    Algorithm (simplified, Bischof & Kraus 2021):
      At step t, for each loss L_i:
        ρ_i(t) = L_i(t) / L_i(0)          # relative loss ratio (normalised to init)
        ρ̄_i    = EMA(ρ_i)                  # smoothed ratio
        w_i    = n * softmax(ρ̄ / τ)[i]    # temperature-scaled attention over ratios

    Weights are highest for terms that have *not* decreased as much relative
    to their initial value.  Very cheap: only scalar loss values needed.

    Parameters
    ----------
    loss_names     : list of loss term identifiers
    tau            : temperature (higher → more uniform; lower → winner-takes-all)
    alpha          : EMA smoothing factor for ρ_i
    update_every   : recompute weights every N steps
    clip_min/max   : hard weight clamps
    """

    def __init__(
        self,
        loss_names: List[str],
        *,
        tau: float = 0.1,
        alpha: float = 0.999,
        update_every: int = 1,
        clip_min: float = 1e-3,
        clip_max: float = 1e3,
        initial_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.loss_names = list(loss_names)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.update_every = int(update_every)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        self._weights: Dict[str, float] = {
            n: float((initial_weights or {}).get(n, 1.0)) for n in loss_names
        }
        self._L0: Dict[str, float] = {}          # initial loss values
        self._rho_ema: Dict[str, float] = {n: 1.0 for n in loss_names}
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._step = 0

    def step(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted total and update weights."""
        n = len(self.loss_names)

        # Record initial loss values on first call
        for name in self.loss_names:
            if name in losses and name not in self._L0:
                self._L0[name] = max(1e-12, float(losses[name].detach().item()))

        if self._step % self.update_every == 0 and self._L0:
            # Update EMA of relative ratios
            for name in self.loss_names:
                if name not in losses or name not in self._L0:
                    continue
                l_cur = max(1e-12, float(losses[name].detach().item()))
                rho = l_cur / self._L0[name]
                self._rho_ema[name] = (
                    self.alpha * self._rho_ema[name] + (1.0 - self.alpha) * rho
                )

            # Temperature-scaled softmax over rho_ema
            rho_vals = [self._rho_ema.get(nm, 1.0) for nm in self.loss_names]
            rho_t = [r / self.tau for r in rho_vals]
            rho_max = max(rho_t)
            exp_r = [math.exp(r - rho_max) for r in rho_t]
            sum_exp = sum(exp_r) + 1e-30
            for name, e in zip(self.loss_names, exp_r):
                w = n * e / sum_exp
                self._weights[name] = float(max(self.clip_min, min(self.clip_max, w)))
                self._history[name].append(self._weights[name])

        self._step += 1
        return self._apply(losses)

    def _apply(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for name, w in self._weights.items():
            if name not in losses:
                continue
            term = w * losses[name]
            total = term if total is None else total + term
        if total is None:
            ref = next(iter(losses.values()))
            total = ref.new_zeros(())
        return total

    @property
    def current_weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)


# ─────────────────────────────────────────────────────────────────────────────
# SoftAdapt
# ─────────────────────────────────────────────────────────────────────────────

class SoftAdapt:
    """
    SoftAdapt — Adaptive loss weighting via rates of change.

    (Heydari et al., 2019 — "SoftAdapt: Techniques for Adaptive Loss
    Weighting of Neural Networks with Multi-Part Loss Functions")

    Weight update:
      Δ_i  = L_i(t) - L_i(t-1)            # 1-step change (or EMA thereof)
      β_i  = softmax(Δ / τ)[i]            # attention over improvements
      w_i  = n * β_i                       # n = number of terms

    Terms that are *decreasing fastest* get lower weight; terms that are
    barely moving (hard) get higher weight.

    Parameters
    ----------
    loss_names   : loss term identifiers
    tau          : temperature; lower → sharper focus on hardest term
    beta         : EMA smoothing for loss change (0 = no EMA, 1 = never update)
    normalized   : if True, divide Δ by L (relative change rather than absolute)
    update_every : recompute weights every N steps
    """

    def __init__(
        self,
        loss_names: List[str],
        *,
        tau: float = 0.1,
        beta: float = 0.0,
        normalized: bool = True,
        update_every: int = 1,
        clip_min: float = 1e-3,
        clip_max: float = 1e3,
        initial_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.loss_names = list(loss_names)
        self.tau = float(tau)
        self.beta = float(beta)
        self.normalized = normalized
        self.update_every = int(update_every)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        self._weights: Dict[str, float] = {
            n: float((initial_weights or {}).get(n, 1.0)) for n in loss_names
        }
        self._prev: Dict[str, float] = {}
        self._delta_ema: Dict[str, float] = {n: 0.0 for n in loss_names}
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._step = 0

    def step(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        n = len(self.loss_names)

        for name in self.loss_names:
            if name not in losses:
                continue
            cur = float(losses[name].detach().item())
            if name in self._prev:
                delta = cur - self._prev[name]
                if self.normalized:
                    delta = delta / (abs(self._prev[name]) + 1e-12)
                if self.beta > 0:
                    self._delta_ema[name] = (
                        self.beta * self._delta_ema[name] + (1.0 - self.beta) * delta
                    )
                else:
                    self._delta_ema[name] = delta
            self._prev[name] = cur

        if self._step % self.update_every == 0 and len(self._prev) >= len(self.loss_names):
            deltas = [self._delta_ema.get(nm, 0.0) for nm in self.loss_names]
            # Temperature softmax over deltas
            d_t = [d / self.tau for d in deltas]
            d_max = max(d_t)
            exp_d = [math.exp(d - d_max) for d in d_t]
            sum_exp = sum(exp_d) + 1e-30
            for name, e in zip(self.loss_names, exp_d):
                w = n * e / sum_exp
                self._weights[name] = float(max(self.clip_min, min(self.clip_max, w)))
                self._history[name].append(self._weights[name])

        self._step += 1
        total = None
        for name, w in self._weights.items():
            if name not in losses:
                continue
            term = w * losses[name]
            total = term if total is None else total + term
        if total is None:
            ref = next(iter(losses.values()))
            total = ref.new_zeros(())
        return total

    @property
    def current_weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)


# ─────────────────────────────────────────────────────────────────────────────
# Augmented Lagrangian
# ─────────────────────────────────────────────────────────────────────────────

class AugmentedLagrangian:
    """
    Augmented Lagrangian method for PINN loss balancing.

    Formulation
    -----------
    Treats each non-PDE term (BC, IC, data) as a constraint L_i ≤ ε_i.
    The augmented objective is:

        J = L_pde + Σ_i [λ_i · L_i + (ρ/2) · L_i²]

    Dual ascent update (after each optimisation step):
        λ_i ← clip(λ_i + ρ · L_i, λ_min, λ_max)

    This is equivalent to having a weight that automatically grows whenever
    a constraint is violated and shrinks when it's satisfied.

    Parameters
    ----------
    loss_names      : all loss term identifiers
    pde_key         : key for the physics residual (no multiplier update)
    rho             : penalty parameter (step size for multiplier update)
    lambda_init     : initial multiplier values
    lambda_min/max  : hard clamps on multipliers
    update_every    : update multipliers every N steps
    warmup_steps    : steps before starting multiplier updates (stabilise first)
    """

    def __init__(
        self,
        loss_names: List[str],
        *,
        pde_key: str = "pde",
        rho: float = 0.1,
        lambda_init: float = 1.0,
        lambda_min: float = 0.0,
        lambda_max: float = 1e4,
        update_every: int = 1,
        warmup_steps: int = 100,
        initial_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.loss_names = list(loss_names)
        self.pde_key = pde_key
        self.rho = float(rho)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.update_every = int(update_every)
        self.warmup_steps = int(warmup_steps)

        init = initial_weights or {}
        self._lam: Dict[str, float] = {
            n: float(init.get(n, lambda_init))
            for n in loss_names if n != pde_key
        }
        self._pde_weight: float = float(init.get(pde_key, 1.0))
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._step = 0

    def step(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute augmented Lagrangian objective and update multipliers."""
        # ── Compute objective ────────────────────────────────────────────────
        total = None

        # PDE term (always included as-is)
        if self.pde_key in losses:
            total = self._pde_weight * losses[self.pde_key]

        for name, lam in self._lam.items():
            if name not in losses:
                continue
            L_i = losses[name]
            term = lam * L_i + (self.rho / 2.0) * L_i * L_i
            total = term if total is None else total + term

        if total is None:
            ref = next(iter(losses.values()))
            total = ref.new_zeros(())

        # ── Multiplier update (dual ascent) ───────────────────────────────────
        if (self._step >= self.warmup_steps
                and self._step % self.update_every == 0):
            for name in list(self._lam.keys()):
                if name not in losses:
                    continue
                L_val = float(losses[name].detach().item())
                new_lam = self._lam[name] + self.rho * L_val
                self._lam[name] = float(max(self.lambda_min, min(self.lambda_max, new_lam)))
                self._history[name].append(self._lam[name])

        self._step += 1
        return total

    @property
    def current_weights(self) -> Dict[str, float]:
        w = {self.pde_key: self._pde_weight}
        w.update(self._lam)
        return w

    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)


# ─────────────────────────────────────────────────────────────────────────────
# Inverse Dirichlet Weighting
# ─────────────────────────────────────────────────────────────────────────────

class InverseDirichlet:
    """
    Inverse Dirichlet (gradient statistics) weighting.

    (van der Meer et al., 2022 — "Optimally Weighted Loss Functions for
    Solving PDEs with Neural Networks")

    For each loss term, computes the *standard deviation of gradients* w.r.t.
    the first shared layer.  Weights are set proportional to 1 / std(grad).

    This balances the "learning stiffness" — terms with large gradient std
    need smaller weights, and vice versa.  Very cheap: single forward+backward
    per update, computes grads for all losses simultaneously.

    Parameters
    ----------
    model        : nn.Module (the PINN)
    loss_names   : list of loss term identifiers
    shared_layer : layer used for gradient statistics (default: last layer)
    update_every : update weights every N steps (20–100 is typical)
    alpha        : EMA decay for weight smoothing (0 = no EMA)
    clip_min/max : hard clamps
    """

    def __init__(
        self,
        model: nn.Module,
        loss_names: List[str],
        *,
        shared_layer: Optional[nn.Module] = None,
        update_every: int = 50,
        alpha: float = 0.9,
        clip_min: float = 1e-3,
        clip_max: float = 1e3,
        initial_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.model = model
        self.loss_names = list(loss_names)
        self.update_every = int(update_every)
        self.alpha = float(alpha)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)

        init = initial_weights or {}
        self._weights: Dict[str, float] = {
            n: float(init.get(n, 1.0)) for n in loss_names
        }
        self._history: Dict[str, List[float]] = defaultdict(list)
        self._step = 0

        # Determine shared layer
        if shared_layer is not None:
            self._layer = shared_layer
        else:
            self._layer = self._find_last_layer(model)

    @staticmethod
    def _find_last_layer(model: nn.Module) -> Optional[nn.Module]:
        last = None
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                last = m
        return last

    def _grad_std(self, loss: torch.Tensor) -> float:
        if self._layer is None:
            return 1.0
        params = [p for p in self._layer.parameters() if p.requires_grad]
        if not params:
            return 1.0
        try:
            grads = torch.autograd.grad(
                loss, params, retain_graph=True,
                create_graph=False, allow_unused=True
            )
            all_g = torch.cat([g.flatten() for g in grads if g is not None])
            return float(all_g.std().item()) if all_g.numel() > 1 else 1.0
        except Exception:
            return 1.0

    def step(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._step % self.update_every == 0:
            stds: Dict[str, float] = {}
            for name in self.loss_names:
                if name not in losses:
                    continue
                stds[name] = max(1e-12, self._grad_std(losses[name]))

            if stds:
                mean_std = sum(stds.values()) / len(stds)
                for name, s in stds.items():
                    new_w = mean_std / s   # inverse Dirichlet weight
                    new_w = max(self.clip_min, min(self.clip_max, new_w))
                    if self.alpha > 0:
                        new_w = self.alpha * self._weights.get(name, new_w) + (1.0 - self.alpha) * new_w
                    self._weights[name] = float(new_w)
                    self._history[name].append(self._weights[name])

        self._step += 1
        total = None
        for name, w in self._weights.items():
            if name not in losses:
                continue
            term = w * losses[name]
            total = term if total is None else total + term
        if total is None:
            ref = next(iter(losses.values()))
            total = ref.new_zeros(())
        return total

    @property
    def current_weights(self) -> Dict[str, float]:
        return dict(self._weights)

    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)


# ─────────────────────────────────────────────────────────────────────────────
# PCGrad (Gradient Surgery)
# ─────────────────────────────────────────────────────────────────────────────

class PCGrad:
    """
    PCGrad — Projecting Conflicting Gradients.

    (Yu et al., 2020 — "Gradient Surgery for Multi-Task Learning")

    For each pair of tasks (i, j) with conflicting gradients (cos < 0),
    project g_i onto the normal plane of g_j:
        g_i ← g_i - (g_i · g_j / ||g_j||²) · g_j

    This eliminates destructive interference between PDE and BC gradients.

    Note: requires one backward pass per loss term.  Use ``update_every``
    to amortise the cost (e.g. every 10 steps).

    Parameters
    ----------
    model        : nn.Module — the PINN
    loss_names   : list of loss term identifiers
    update_every : apply gradient surgery every N steps (default: every step)
    weights      : fixed multiplier for each term (PCGrad works with 1.0)
    """

    def __init__(
        self,
        model: nn.Module,
        loss_names: List[str],
        *,
        update_every: int = 1,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.model = model
        self.loss_names = list(loss_names)
        self.update_every = int(update_every)
        self._weights: Dict[str, float] = {
            n: float((weights or {}).get(n, 1.0)) for n in loss_names
        }
        self._step = 0

    def step(
        self,
        losses: Dict[str, torch.Tensor],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> torch.Tensor:
        """
        Apply gradient surgery and return the total loss for logging.

        If ``optimizer`` is provided, zeroes gradients, computes per-task
        gradients, applies surgery, and writes modified gradients to model
        parameters.  Caller should then only call ``optimizer.step()``.

        If ``optimizer`` is None, falls back to standard weighted sum
        (useful when called from a custom training loop that manages its
        own backward calls).
        """
        if optimizer is not None and self._step % self.update_every == 0:
            optimizer.zero_grad()
            self._apply_surgery(losses)
        else:
            # Standard fallback: weighted sum
            total = None
            for name, w in self._weights.items():
                if name not in losses:
                    continue
                term = w * losses[name]
                total = term if total is None else total + term
            if total is None:
                ref = next(iter(losses.values()))
                total = ref.new_zeros(())
            self._step += 1
            return total

        # Return scalar total for logging (detached)
        self._step += 1
        total = sum(
            self._weights.get(n, 1.0) * losses[n]
            for n in self.loss_names if n in losses
        )
        return total if not isinstance(total, int) else losses[self.loss_names[0]].new_zeros(())

    def _apply_surgery(self, losses: Dict[str, torch.Tensor]) -> None:
        """Compute per-task gradients, project, and accumulate into .grad."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return

        # Collect per-task gradients
        task_grads: Dict[str, List[torch.Tensor]] = {}
        for name in self.loss_names:
            if name not in losses:
                continue
            w = self._weights.get(name, 1.0)
            try:
                grads = torch.autograd.grad(
                    w * losses[name], params,
                    retain_graph=True, create_graph=False, allow_unused=True
                )
                task_grads[name] = [
                    g.detach().clone() if g is not None
                    else torch.zeros_like(p)
                    for g, p in zip(grads, params)
                ]
            except Exception:
                pass

        if not task_grads:
            return

        task_names = list(task_grads.keys())

        # Apply gradient surgery per task
        projected: Dict[str, List[torch.Tensor]] = {n: [g.clone() for g in task_grads[n]]
                                                    for n in task_names}

        for i, ni in enumerate(task_names):
            for j, nj in enumerate(task_names):
                if i == j:
                    continue
                # Project task i gradient onto normal plane of task j
                for k in range(len(params)):
                    gi = projected[ni][k]
                    gj = task_grads[nj][k]
                    dot = (gi * gj).sum()
                    if dot < 0:  # conflicting
                        norm_sq = (gj * gj).sum() + 1e-12
                        projected[ni][k] = gi - (dot / norm_sq) * gj

        # Accumulate projected gradients into .grad
        for p in params:
            p.grad = None

        for ni, pg in projected.items():
            for p, g in zip(params, pg):
                if p.grad is None:
                    p.grad = g.clone()
                else:
                    p.grad += g

    @property
    def current_weights(self) -> Dict[str, float]:
        return dict(self._weights)


# ─────────────────────────────────────────────────────────────────────────────
# AutoBalancer — meta-scheduler
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AutoBalancerConfig:
    """Configuration for the AutoBalancer meta-scheduler."""
    # Initial method
    method: str = "relobralo"           # starting method
    # How often to check health and possibly switch (steps)
    check_every: int = 500
    # Number of recent steps to evaluate health
    health_window: int = 200
    # Plateau: if loss hasn't decreased by this fraction → switch or escalate
    plateau_rel_thresh: float = 0.01
    # Divergence: if loss increased by this fraction → switch to safer method
    divergence_thresh: float = 0.05
    # Max switches before giving up and using safe fallback
    max_switches: int = 3
    # Fallback method when escalation limit reached
    fallback_method: str = "loss_ratio"
    # Passed to underlying balancers
    tau: float = 0.1
    alpha: float = 0.9
    update_every: int = 1
    clip_min: float = 1e-3
    clip_max: float = 1e3
    rho: float = 0.1                    # AugmentedLagrangian penalty
    warmup_steps: int = 100


class AutoBalancer:
    """
    Auto-balancing meta-scheduler for PINN loss weights.

    Monitors training health (plateau, divergence) and automatically
    switches between balancing methods when the current one is not working.

    Escalation ladder (default):
      relobralo → softadapt → augmented_lagrangian → loss_ratio (safe fallback)

    Parameters
    ----------
    model      : nn.Module (needed for grad-based methods)
    loss_names : list of loss term identifiers
    config     : AutoBalancerConfig
    pde_key    : key for the physics residual loss
    """

    _LADDER = ["relobralo", "softadapt", "augmented_lagrangian", "loss_ratio"]

    def __init__(
        self,
        model: nn.Module,
        loss_names: List[str],
        *,
        config: Optional[AutoBalancerConfig] = None,
        pde_key: str = "pde",
    ) -> None:
        self.model = model
        self.loss_names = list(loss_names)
        self.pde_key = pde_key
        self.cfg = config or AutoBalancerConfig()

        self._balancer: Any = self._build(self.cfg.method)
        self._current_method = self.cfg.method
        self._loss_window: deque = deque(maxlen=self.cfg.health_window)
        self._n_switches = 0
        self._step = 0
        self._switch_log: List[Tuple[int, str, str]] = []  # (step, from, to)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def _build(self, method: str) -> Any:
        cfg = self.cfg
        common = dict(
            loss_names=self.loss_names,
            clip_min=cfg.clip_min,
            clip_max=cfg.clip_max,
        )
        if method == "relobralo":
            return ReLoBRaLo(**common, tau=cfg.tau, alpha=1.0 - (1.0 - cfg.alpha) * 0.01,
                              update_every=cfg.update_every)
        if method == "softadapt":
            return SoftAdapt(**common, tau=cfg.tau, normalized=True,
                             update_every=cfg.update_every)
        if method == "augmented_lagrangian":
            return AugmentedLagrangian(
                self.loss_names, pde_key=self.pde_key,
                rho=cfg.rho, warmup_steps=cfg.warmup_steps,
            )
        if method == "loss_ratio":
            # Import from existing module
            from pinneaple_neural.trainer.weight_scheduler import (
                LossRatioBalancer, WeightSchedulerConfig,
            )
            wscfg = WeightSchedulerConfig(ema_decay=cfg.alpha, update_every=cfg.update_every)
            return _LossRatioWrapper(LossRatioBalancer(self.loss_names, config=wscfg))
        if method == "inverse_dirichlet":
            return InverseDirichlet(
                self.model, self.loss_names,
                update_every=max(cfg.update_every, 20),
                alpha=cfg.alpha, clip_min=cfg.clip_min, clip_max=cfg.clip_max,
            )
        raise ValueError(f"Unknown AutoBalancer method: '{method}'")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def _is_plateau(self) -> bool:
        if len(self._loss_window) < self.cfg.health_window:
            return False
        vals = list(self._loss_window)
        head = sum(vals[:len(vals) // 4]) / max(1, len(vals) // 4)
        tail = sum(vals[-len(vals) // 4:]) / max(1, len(vals) // 4)
        rel = (head - tail) / (abs(head) + 1e-12)
        return rel < self.cfg.plateau_rel_thresh

    def _is_diverging(self) -> bool:
        if len(self._loss_window) < 10:
            return False
        vals = list(self._loss_window)
        head = vals[0]
        tail = vals[-1]
        rel = (tail - head) / (abs(head) + 1e-12)
        return rel > self.cfg.divergence_thresh

    def _switch_to_next(self) -> None:
        if self._n_switches >= self.cfg.max_switches:
            next_method = self.cfg.fallback_method
        else:
            try:
                idx = self._LADDER.index(self._current_method)
                next_method = self._LADDER[min(idx + 1, len(self._LADDER) - 1)]
            except ValueError:
                next_method = self.cfg.fallback_method

        if next_method == self._current_method:
            return

        logger.info(
            f"AutoBalancer: switching {self._current_method} → {next_method} "
            f"at step {self._step} (switch #{self._n_switches + 1})"
        )
        self._switch_log.append((self._step, self._current_method, next_method))
        self._balancer = self._build(next_method)
        self._current_method = next_method
        self._n_switches += 1
        self._loss_window.clear()

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def step(
        self,
        losses: Dict[str, torch.Tensor],
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> torch.Tensor:
        """Apply current balancer, monitor health, switch if needed."""
        # Get total loss from current balancer
        if isinstance(self._balancer, PCGrad):
            total = self._balancer.step(losses, optimizer)
        elif hasattr(self._balancer, "step"):
            total = self._balancer.step(losses)
        else:
            total = self._balancer.apply(losses)

        # Track total loss for health monitoring
        try:
            self._loss_window.append(float(total.detach().item()))
        except Exception:
            pass

        # Health check
        if self._step % self.cfg.check_every == 0 and self._step > 0:
            if self._is_diverging():
                logger.warning(f"AutoBalancer: divergence detected at step {self._step}.")
                self._switch_to_next()
            elif self._is_plateau():
                logger.info(f"AutoBalancer: plateau detected at step {self._step}.")
                self._switch_to_next()

        self._step += 1
        return total

    @property
    def current_method(self) -> str:
        return self._current_method

    @property
    def current_weights(self) -> Dict[str, float]:
        b = self._balancer
        if hasattr(b, "current_weights"):
            return b.current_weights
        if hasattr(b, "_weights"):
            return dict(b._weights)
        return {}

    def switch_log(self) -> List[Tuple[int, str, str]]:
        """List of (step, from_method, to_method) switches."""
        return list(self._switch_log)

    def history(self) -> Dict[str, List[float]]:
        b = self._balancer
        return b.history() if hasattr(b, "history") else {}


class _LossRatioWrapper:
    """Thin wrapper to give LossRatioBalancer a .step() interface."""
    def __init__(self, balancer: Any) -> None:
        self._b = balancer
        self._step_count = 0

    def step(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        self._b.update(losses, self._step_count)
        self._step_count += 1
        return self._b.apply(losses)

    @property
    def current_weights(self) -> Dict[str, float]:
        return dict(self._b._weights)

    def history(self) -> Dict[str, Any]:
        return self._b.history()


# ─────────────────────────────────────────────────────────────────────────────
# JointAdaptiveWeights — jointly-trained softplus loss weights
# ─────────────────────────────────────────────────────────────────────────────

def _softplus_inv(x: float, beta: float = 1.0, max_raw: float = 20.0) -> float:
    """Inverse of softplus(z, beta) = (1/beta)*log(1+exp(beta*z)), clamped to [-max_raw, max_raw]."""
    import math
    x = max(float(x), 1e-6)
    try:
        raw = math.log(math.expm1(min(beta * x, 500.0))) / beta
    except (ValueError, OverflowError):
        raw = max_raw
    return max(-max_raw, min(max_raw, raw))


class JointAdaptiveWeights(nn.Module):
    """
    Jointly-trained adaptive loss weights via softplus parameterisation.

    Unlike ``SelfAdaptiveWeights`` (SA-PINN), these λ parameters are trained by
    gradient *descent* using the **same** optimiser as the model — no separate
    weight optimiser is needed.  Weights are kept strictly positive via softplus.

    This is the approach used in Rate-PINN and similar works: each λ_i is an
    unconstrained raw parameter initialised via ``softplus_inv(λ_init)``, so that
    ``softplus(raw_i) ≈ λ_init`` at the start of training.  The network can then
    increase or decrease λ_i freely while guaranteeing λ_i > 0.

    Optionally, a fixed ``scale`` dict multiplies the weighted term before
    combining (useful to pre-amplify rare but important losses like a BCF term).

    Parameters
    ----------
    loss_names      : ordered list of loss term keys
    initial_weights : starting λ values (default 1.0 per term)
    sp_beta         : softplus sharpness (1.0 = standard)
    max_raw_init    : clamp on initial raw parameter (prevents overflow at init)
    scale           : optional fixed per-term scale factors applied on top of λ

    Usage
    -----
    ::

        jaw = JointAdaptiveWeights(["pde", "bc", "ic"],
                                   initial_weights={"pde": 1.0, "bc": 10.0, "ic": 5.0})
        optimizer = torch.optim.Adam(list(pinn.parameters()) + jaw.weight_params(), lr=1e-4)

        for step in range(n_steps):
            optimizer.zero_grad()
            losses = compute_losses(pinn, batch)
            total = jaw(losses)
            total.backward()
            optimizer.step()
    """

    def __init__(
        self,
        loss_names: List[str],
        *,
        initial_weights: Optional[Dict[str, float]] = None,
        sp_beta: float = 1.0,
        max_raw_init: float = 20.0,
        scale: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.loss_names = list(loss_names)
        self.sp = nn.Softplus(beta=sp_beta)
        self._scale: Dict[str, float] = scale or {}

        init = initial_weights or {}
        # Sanitize keys for ParameterDict (no dots / dashes)
        self._name_to_key: Dict[str, str] = {
            n: n.replace(".", "_").replace("-", "_") for n in loss_names
        }
        raw_params = {}
        for name in loss_names:
            w0 = float(init.get(name, 1.0))
            raw = _softplus_inv(w0, beta=sp_beta, max_raw=max_raw_init)
            raw_params[self._name_to_key[name]] = nn.Parameter(
                torch.tensor(raw, dtype=torch.float32)
            )
        self.raw_weights = nn.ParameterDict(raw_params)
        self._history: Dict[str, List[float]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Weight access
    # ------------------------------------------------------------------

    def weights(self) -> Dict[str, torch.Tensor]:
        """Current λ values as differentiable tensors."""
        return {n: self.sp(self.raw_weights[self._name_to_key[n]]) for n in self.loss_names}

    def weight_dict(self) -> Dict[str, float]:
        """Current λ values as Python floats."""
        return {
            n: float(self.sp(self.raw_weights[self._name_to_key[n]]).detach().item())
            for n in self.loss_names
        }

    def weight_params(self) -> List[nn.Parameter]:
        """Raw parameters — include in the model optimiser param group."""
        return list(self.raw_weights.values())

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return Σ_i scale_i · λ_i · L_i."""
        total = None
        for name in self.loss_names:
            if name not in losses:
                continue
            lam = self.sp(self.raw_weights[self._name_to_key[name]])
            s = self._scale.get(name, 1.0)
            term = s * lam * losses[name]
            self._history[name].append(float(lam.detach().item()))
            total = term if total is None else total + term
        if total is None:
            ref = next(iter(losses.values()))
            total = ref.new_zeros(())
        return total

    # Alias so it fits the .step(losses) contract of the other balancers
    def step(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self(losses)

    @property
    def current_weights(self) -> Dict[str, float]:
        return self.weight_dict()

    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)
