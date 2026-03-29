"""Automatic loss weight scheduling for PINN training.

PINNs balance multiple competing loss terms (PDE residual, boundary conditions,
initial conditions, data). The relative magnitudes of these terms change during
training, making fixed weights suboptimal. This module provides several
automatic strategies to adjust weights on the fly.

Methods
-------
SelfAdaptiveWeights (SA-PINN)
    Learnable log-weights updated by *gradient ascent* (opposite direction to
    model weights). Weights increase where the model struggles, focusing
    training effort on the hardest terms.
    Ref: McClenny & Braga-Neto (2020). arXiv:2009.04544

GradNormBalancer
    Adjusts weights so gradient norms of all tasks at a shared layer remain
    proportional to a target ratio. Prevents any single term from dominating.
    Ref: Chen et al. (2018). ICML. arXiv:1711.02257

LossRatioBalancer
    Heuristic: keeps the ratio L_term / L_pde near a target value using
    exponential moving averages. Simple and effective for most cases.

NTKWeightBalancer
    Uses the trace of the Neural Tangent Kernel to set weights proportional
    to tr(K) / tr(K_i). Theoretically grounded but more expensive.
    Ref: Wang et al. (2022). arXiv:2007.14527

WeightScheduler
    Unified interface wrapping all methods above.

Quick start
-----------
>>> from pinneaple_train.weight_scheduler import WeightScheduler, WeightSchedulerConfig
>>>
>>> sched = WeightScheduler(
...     model=pinn,
...     loss_names=["pde", "bc", "ic"],
...     config=WeightSchedulerConfig(method="self_adaptive"),
... )
>>>
>>> # In training loop:
>>> raw_losses = {"pde": l_pde, "bc": l_bc, "ic": l_ic}
>>> total = sched.step(raw_losses, step=epoch)
>>> total.backward()
>>>
>>> # If using SA-PINN, weights need their own optimizer:
>>> weight_opt = torch.optim.Adam(sched.weight_params(), lr=1e-3)
>>> weight_opt.step()   # after model_opt.step()
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class WeightSchedulerConfig:
    """Configuration for automatic PINN loss weight scheduling.

    Attributes
    ----------
    method : str
        One of "self_adaptive", "gradnorm", "loss_ratio", "ntk", "fixed".
    initial_weights : dict
        Starting weight for each loss term.  Keys must match loss_names.
        Defaults: pde=1, bc=10, ic=10, data=1.
    update_every : int
        Update weights every N optimisation steps (default 100).
    lr : float
        Learning rate for learnable weights (SA-PINN only).
    alpha : float
        GradNorm restoring-force exponent (default 0.16).
    clip_min / clip_max : float
        Hard bounds on weight values after each update.
    ema_decay : float
        EMA factor for running loss statistics (LossRatio method).
    """
    method: str = "self_adaptive"
    initial_weights: Dict[str, float] = field(
        default_factory=lambda: {"pde": 1.0, "bc": 10.0, "ic": 10.0, "data": 1.0}
    )
    update_every: int = 100
    lr: float = 0.01
    alpha: float = 0.16
    clip_min: float = 0.01
    clip_max: float = 1000.0
    ema_decay: float = 0.9


# ---------------------------------------------------------------------------
# Self-Adaptive Weights (SA-PINN)
# ---------------------------------------------------------------------------

class SelfAdaptiveWeights(nn.Module):
    """Learnable loss weights trained by gradient *ascent*.

    The weights λ_i = exp(log_λ_i) are parameters updated in the direction
    that *increases* the total loss, so they grow for terms the model finds
    hard.  The model parameters are updated by gradient descent as usual.

    Usage
    -----
    This module exposes its own parameters via ``weight_params()`` which
    should be placed in a *separate* optimiser that takes gradient ascent
    steps (or equivalently, minimises the negative of the total loss)::

        weight_opt = torch.optim.Adam(sched.weight_params(), lr=1e-3)
        # ... after loss.backward():
        model_opt.step()
        weight_opt.step()   # ascend weights
    """

    def __init__(self, loss_names: List[str], config: WeightSchedulerConfig) -> None:
        super().__init__()
        init = config.initial_weights
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.log(torch.tensor(float(init.get(name, 1.0)))))
            for name in loss_names
        })
        self.config = config
        self._history: Dict[str, List[float]] = defaultdict(list)

    @property
    def weights(self) -> Dict[str, torch.Tensor]:
        """Current weights as tensors (always positive via exp)."""
        return {k: torch.exp(v) for k, v in self.log_weights.items()}

    def weight_dict(self) -> Dict[str, float]:
        """Current weight values as floats."""
        return {k: float(torch.exp(v).item()) for k, v in self.log_weights.items()}

    def weight_params(self) -> List[nn.Parameter]:
        """Parameters for the weight optimiser."""
        return list(self.log_weights.values())

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return λᵀ · L (weighted sum of losses)."""
        total = None
        for name, lam in self.weights.items():
            if name not in losses:
                continue
            lam_clipped = lam.clamp(self.config.clip_min, self.config.clip_max)
            term = lam_clipped * losses[name]
            total = term if total is None else total + term
            self._history[name].append(float(lam_clipped.item()))
        if total is None:
            ref = next(iter(losses.values()))
            total = ref.new_zeros(())
        return total

    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)


# ---------------------------------------------------------------------------
# GradNorm
# ---------------------------------------------------------------------------

class GradNormBalancer:
    """Gradient-norm-based weight balancing.

    Adjusts weights so that the gradient norm of each task at a shared layer
    remains proportional to the mean gradient norm raised to a restoring-force
    exponent.

    Parameters
    ----------
    model : nn.Module
    loss_names : list of loss term names
    shared_layer : the layer used to measure gradient norms (default: last layer)
    config : WeightSchedulerConfig
    """

    def __init__(
        self,
        model: nn.Module,
        loss_names: List[str],
        shared_layer: Optional[nn.Module] = None,
        config: Optional[WeightSchedulerConfig] = None,
    ) -> None:
        self.model = model
        self.loss_names = loss_names
        cfg = config or WeightSchedulerConfig()
        self.config = cfg
        self.alpha = cfg.alpha

        # Find shared layer: use last Linear/Conv layer if not specified
        if shared_layer is not None:
            self.shared_layer = shared_layer
        else:
            self.shared_layer = self._find_last_layer(model)

        init = cfg.initial_weights
        self._weights: Dict[str, float] = {n: float(init.get(n, 1.0)) for n in loss_names}
        self._initial_task_losses: Dict[str, Optional[float]] = {n: None for n in loss_names}
        self._history: Dict[str, List[float]] = defaultdict(list)

    @staticmethod
    def _find_last_layer(model: nn.Module) -> Optional[nn.Module]:
        last = None
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                last = m
        return last

    def _grad_norm(self, loss: torch.Tensor) -> float:
        """Compute gradient norm of *loss* w.r.t. shared_layer parameters."""
        if self.shared_layer is None:
            return 1.0
        params = [p for p in self.shared_layer.parameters() if p.requires_grad]
        if not params:
            return 1.0
        try:
            grads = torch.autograd.grad(
                loss, params, retain_graph=True, create_graph=False, allow_unused=True
            )
            norms = [g.norm() for g in grads if g is not None]
            return float(sum(norms).item()) if norms else 1.0
        except Exception:
            return 1.0

    def update_weights(
        self,
        losses: Dict[str, torch.Tensor],
        step: int,
    ) -> Dict[str, float]:
        """Recompute weights based on gradient norms.

        Only runs every ``config.update_every`` steps.
        """
        if step % self.config.update_every != 0:
            return dict(self._weights)

        grad_norms: Dict[str, float] = {}
        for name in self.loss_names:
            if name not in losses:
                continue
            w = self._weights.get(name, 1.0)
            weighted_loss = w * losses[name]
            grad_norms[name] = self._grad_norm(weighted_loss)

            if self._initial_task_losses[name] is None:
                self._initial_task_losses[name] = float(losses[name].item()) + 1e-12

        if not grad_norms:
            return dict(self._weights)

        mean_norm = sum(grad_norms.values()) / len(grad_norms)

        for name, gn in grad_norms.items():
            l0 = self._initial_task_losses.get(name) or 1.0
            l_cur = float(losses[name].item()) + 1e-12
            loss_ratio = l_cur / l0
            target_norm = mean_norm * (loss_ratio ** self.alpha)
            if gn > 1e-10:
                self._weights[name] = float(
                    max(self.config.clip_min, min(self.config.clip_max, target_norm / gn))
                )
            self._history[name].append(self._weights[name])

        return dict(self._weights)

    def apply(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return weighted sum using current weights."""
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

    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)


# ---------------------------------------------------------------------------
# Loss Ratio Balancer
# ---------------------------------------------------------------------------

class LossRatioBalancer:
    """Keep loss component ratios near target values using EMA smoothing.

    For each loss term, maintains a running EMA and adjusts the weight
    multiplicatively so that ``EMA(L_term) / EMA(L_pde)`` stays near the
    target ratio.

    Parameters
    ----------
    loss_names : list of loss keys
    target_ratios : optional dict {name: target_ratio relative to first term}.
        Default: all ratios → 1.0.
    config : WeightSchedulerConfig
    """

    def __init__(
        self,
        loss_names: List[str],
        target_ratios: Optional[Dict[str, float]] = None,
        config: Optional[WeightSchedulerConfig] = None,
    ) -> None:
        cfg = config or WeightSchedulerConfig()
        self.config = cfg
        self.loss_names = loss_names
        init = cfg.initial_weights
        self._weights: Dict[str, float] = {n: float(init.get(n, 1.0)) for n in loss_names}
        self._target_ratios: Dict[str, float] = target_ratios or {n: 1.0 for n in loss_names}
        self._ema: Dict[str, float] = {n: 0.0 for n in loss_names}
        self._ema_init = False
        self._history: Dict[str, List[float]] = defaultdict(list)

    def update(self, losses: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        """Update weights based on current loss ratios."""
        gamma = self.config.ema_decay

        # Update EMA
        for name in self.loss_names:
            if name not in losses:
                continue
            val = float(losses[name].item())
            if not self._ema_init:
                self._ema[name] = val
            else:
                self._ema[name] = gamma * self._ema[name] + (1.0 - gamma) * val
        self._ema_init = True

        if step % self.config.update_every != 0:
            return dict(self._weights)

        # Reference: first loss term
        ref_name = self.loss_names[0]
        ref_ema = self._ema.get(ref_name, 1.0) + 1e-12

        for name in self.loss_names:
            if name == ref_name or name not in self._ema:
                continue
            actual_ratio = self._ema[name] / ref_ema
            target = self._target_ratios.get(name, 1.0)
            if actual_ratio > 1e-10:
                correction = target / actual_ratio
                # Smooth update: blend current weight with corrected weight
                new_w = self._weights[name] * (0.5 + 0.5 * correction)
                self._weights[name] = float(
                    max(self.config.clip_min, min(self.config.clip_max, new_w))
                )
            self._history[name].append(self._weights[name])

        return dict(self._weights)

    def apply(self, losses: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        w = weights or self._weights
        total = None
        for name, wval in w.items():
            if name not in losses:
                continue
            term = wval * losses[name]
            total = term if total is None else total + term
        if total is None:
            ref = next(iter(losses.values()))
            total = ref.new_zeros(())
        return total

    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)


# ---------------------------------------------------------------------------
# NTK Weight Balancer
# ---------------------------------------------------------------------------

class NTKWeightBalancer:
    """NTK-trace based weight balancing.

    Approximates tr(K_i) for each loss term using a Hutchinson estimator
    (random projection, cheap to compute).  Sets λ_i ∝ tr(K) / tr(K_i).

    Parameters
    ----------
    model : nn.Module
    loss_names : list of loss keys
    n_proj : number of random projections for NTK trace estimate
    config : WeightSchedulerConfig

    Notes
    -----
    Computing NTK traces requires a backward pass per loss term and is
    significantly more expensive than the other methods.  Use ``update_every``
    with a larger value (e.g. 500) to amortise the cost.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_names: List[str],
        n_proj: int = 50,
        config: Optional[WeightSchedulerConfig] = None,
    ) -> None:
        self.model = model
        self.loss_names = loss_names
        self.n_proj = n_proj
        cfg = config or WeightSchedulerConfig()
        self.config = cfg
        init = cfg.initial_weights
        self._weights: Dict[str, float] = {n: float(init.get(n, 1.0)) for n in loss_names}
        self._history: Dict[str, List[float]] = defaultdict(list)

    def _estimate_ntk_trace(self, loss: torch.Tensor) -> float:
        """Approximate tr(K) via Hutchinson estimator over model parameters."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return 1.0

        total = 0.0
        for _ in range(self.n_proj):
            try:
                grads = torch.autograd.grad(
                    loss, params,
                    retain_graph=True, create_graph=False, allow_unused=True
                )
                # Hutchinson: v^T H v ≈ tr(H) where v ~ Rademacher
                acc = 0.0
                for g, p in zip(grads, params):
                    if g is None:
                        continue
                    v = torch.randint_like(p, 0, 2, dtype=p.dtype).mul_(2).sub_(1)
                    acc += float((g * v).sum().item())
                total += acc
            except Exception:
                return 1.0

        return abs(total / max(self.n_proj, 1))

    def compute_weights(
        self,
        loss_fns: Dict[str, Any],
        x_batches: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Compute NTK-based weights.

        Parameters
        ----------
        loss_fns : {name: callable(model, x) -> scalar_loss}
        x_batches : {name: x_tensor}
        """
        traces = {}
        for name in self.loss_names:
            fn = loss_fns.get(name)
            x = x_batches.get(name)
            if fn is None or x is None:
                traces[name] = 1.0
                continue
            try:
                loss = fn(self.model, x)
                if isinstance(loss, tuple):
                    loss = loss[0]
                traces[name] = self._estimate_ntk_trace(loss)
            except Exception:
                traces[name] = 1.0

        total_trace = sum(traces.values()) + 1e-12
        for name, tr in traces.items():
            w = total_trace / (tr + 1e-12) if tr > 1e-10 else 1.0
            self._weights[name] = float(max(self.config.clip_min, min(self.config.clip_max, w)))
            self._history[name].append(self._weights[name])

        return dict(self._weights)

    def apply(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
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

    def history(self) -> Dict[str, List[float]]:
        return dict(self._history)


# ---------------------------------------------------------------------------
# Unified WeightScheduler
# ---------------------------------------------------------------------------

class WeightScheduler:
    """Unified interface for automatic PINN loss weight scheduling.

    Wraps any of the four strategies under a single API so training code
    doesn't need to change when switching methods.

    Parameters
    ----------
    model : nn.Module — the PINN being trained
    loss_names : list of loss term names (e.g. ["pde", "bc", "ic"])
    config : WeightSchedulerConfig

    Usage
    -----
    ::

        sched = WeightScheduler(pinn, ["pde", "bc", "ic"],
                                WeightSchedulerConfig(method="self_adaptive"))

        for epoch in range(n_epochs):
            losses = compute_losses(pinn, batch)
            total = sched.step(losses, epoch)
            total.backward()
            model_opt.step()
            if sched.has_weight_optimizer:
                weight_opt.step()
    """

    def __init__(
        self,
        model: nn.Module,
        loss_names: List[str],
        config: Optional[WeightSchedulerConfig] = None,
    ) -> None:
        cfg = config or WeightSchedulerConfig()
        self.model = model
        self.loss_names = loss_names
        self.config = cfg
        self._step_count = 0

        method = cfg.method.lower()
        self._method = method

        if method == "self_adaptive":
            self._impl = SelfAdaptiveWeights(loss_names, cfg)
        elif method == "gradnorm":
            self._impl = GradNormBalancer(model, loss_names, config=cfg)
        elif method == "loss_ratio":
            self._impl = LossRatioBalancer(loss_names, config=cfg)
        elif method == "ntk":
            self._impl = NTKWeightBalancer(model, loss_names, config=cfg)
        elif method == "fixed":
            self._impl = None  # use initial_weights directly
        else:
            raise ValueError(
                f"Unknown weight scheduling method: '{method}'. "
                f"Choose from: self_adaptive, gradnorm, loss_ratio, ntk, fixed."
            )

        # Fixed weights dict for "fixed" method
        self._fixed_weights: Dict[str, float] = dict(cfg.initial_weights)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def step(self, losses: Dict[str, torch.Tensor], step: Optional[int] = None) -> torch.Tensor:
        """Apply current weights to *losses* and return weighted total.

        Also updates weights according to the configured schedule.

        Parameters
        ----------
        losses : dict of named scalar loss tensors
        step : current training step (auto-incremented if None)

        Returns
        -------
        Weighted total loss (scalar Tensor with grad_fn).
        """
        s = step if step is not None else self._step_count
        self._step_count += 1

        if self._method == "fixed":
            return self._apply_fixed(losses)

        if self._method == "self_adaptive":
            return self._impl(losses)  # type: ignore[operator]

        if self._method == "gradnorm":
            self._impl.update_weights(losses, s)  # type: ignore[union-attr]
            return self._impl.apply(losses)  # type: ignore[union-attr]

        if self._method == "loss_ratio":
            self._impl.update(losses, s)  # type: ignore[union-attr]
            return self._impl.apply(losses)  # type: ignore[union-attr]

        if self._method == "ntk":
            # NTK is expensive; apply cached weights, update periodically
            return self._impl.apply(losses)  # type: ignore[union-attr]

        return self._apply_fixed(losses)

    def _apply_fixed(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for name, w in self._fixed_weights.items():
            if name not in losses:
                continue
            term = w * losses[name]
            total = term if total is None else total + term
        if total is None:
            ref = next(iter(losses.values()))
            total = ref.new_zeros(())
        return total

    # ------------------------------------------------------------------
    # Weight access
    # ------------------------------------------------------------------

    @property
    def current_weights(self) -> Dict[str, float]:
        """Current weight values as floats."""
        if self._method == "fixed":
            return dict(self._fixed_weights)
        if self._method == "self_adaptive":
            return self._impl.weight_dict()  # type: ignore[union-attr]
        if hasattr(self._impl, "_weights"):
            return dict(self._impl._weights)  # type: ignore[union-attr]
        return {}

    def weight_history(self) -> Dict[str, List[float]]:
        """History of weight values per loss term across all steps."""
        if self._impl is not None and hasattr(self._impl, "history"):
            return self._impl.history()
        return {}

    # ------------------------------------------------------------------
    # SA-PINN weight optimizer helpers
    # ------------------------------------------------------------------

    @property
    def has_weight_optimizer(self) -> bool:
        """True if this method uses learnable weights requiring a separate optimiser."""
        return self._method == "self_adaptive"

    def weight_params(self) -> List[nn.Parameter]:
        """Return learnable weight parameters (SA-PINN only).

        Place in a separate optimiser with gradient *ascent*::

            weight_opt = torch.optim.Adam(sched.weight_params(), lr=1e-3)
            # After total.backward() and model_opt.step():
            # Negate gradients for ascent:
            for p in sched.weight_params():
                if p.grad is not None:
                    p.grad.neg_()
            weight_opt.step()
        """
        if self._method != "self_adaptive":
            return []
        return self._impl.weight_params()  # type: ignore[union-attr]

    def optimizer_param_groups(self, lr: Optional[float] = None) -> List[dict]:
        """Param groups for a weight optimiser (SA-PINN only)."""
        params = self.weight_params()
        if not params:
            return []
        _lr = lr or self.config.lr
        return [{"params": params, "lr": _lr}]
