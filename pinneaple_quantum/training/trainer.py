"""Quantum-aware hybrid trainer with parameter-shift support."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

import torch
import torch.nn as nn


@dataclass
class QTrainerConfig:
    """Configuration for the quantum-aware trainer.

    Attributes
    ----------
    epochs : int
        Total number of training steps.
    lr : float
        Learning rate for the classical optimizer.
    optimizer : str
        Optimizer name: ``"adam"`` (default), ``"lbfgs"``, ``"sgd"``.
    quantum_gradient : str
        Gradient strategy for quantum parameters.
        - ``"auto"``            — let the backend compute gradients (PennyLane
          autograd / parameter-shift QNode).
        - ``"parameter_shift"`` — explicit parameter-shift rule applied outside
          the QNode (useful for shot-based evaluation).
        - ``"backprop"``        — standard autograd (simulators only).
    lambda_norm : float
        Weight for the normalization constraint loss  (∫|ψ|²dx = 1).
    lambda_bc : float
        Weight for boundary condition loss.
    print_every : int
        Log interval (steps). 0 to silence.
    device : str
        PyTorch device for tensors.
    """
    epochs:            int   = 2000
    lr:                float = 1e-2
    optimizer:         Literal["adam", "lbfgs", "sgd"] = "adam"
    quantum_gradient:  Literal["auto", "parameter_shift", "backprop"] = "auto"
    lambda_norm:       float = 1.0
    lambda_bc:         float = 1.0
    print_every:       int   = 100
    device:            str   = "cpu"


@dataclass
class QTrainHistory:
    """Training history container."""
    epochs:     List[int]   = field(default_factory=list)
    total_loss: List[float] = field(default_factory=list)
    pde_loss:   List[float] = field(default_factory=list)
    energy:     List[float] = field(default_factory=list)
    norm_sq:    List[float] = field(default_factory=list)


class QTrainer:
    """
    Hybrid classical-quantum trainer.

    Handles models that mix classical ``nn.Module`` parameters (optimized via
    autograd + Adam/L-BFGS) with variational quantum circuit parameters
    (optionally using the parameter-shift rule for exact quantum gradients).

    Parameters
    ----------
    model : nn.Module
        The model to train (QuantumModel, HybridModel, or any nn.Module).
    physics_loss_fn : Callable
        ``loss_fn(model, x_col, **kwargs) → (loss, extras_dict)``
        where ``extras_dict`` may contain ``"energy"``, ``"norm_sq"``, etc.
    bc_loss_fn : Callable, optional
        ``bc_fn(model, x_bc) → scalar loss``
    config : QTrainerConfig, optional
        Trainer configuration.

    Examples
    --------
    >>> from pinneaple_quantum.loss.schrodinger import energy_loss
    >>> trainer = QTrainer(
    ...     model=qmodel,
    ...     physics_loss_fn=lambda m, x: energy_loss(m, x, "harmonic"),
    ...     config=QTrainerConfig(epochs=3000, lr=0.01),
    ... )
    >>> history = trainer.train(x_col)
    """

    def __init__(
        self,
        model: nn.Module,
        physics_loss_fn: Callable,
        bc_loss_fn: Optional[Callable] = None,
        config: Optional[QTrainerConfig] = None,
    ):
        self.model          = model
        self.physics_loss_fn = physics_loss_fn
        self.bc_loss_fn      = bc_loss_fn
        self.config          = config or QTrainerConfig()

        self._device = torch.device(self.config.device)
        self.model.to(self._device)
        self._opt  = self._build_optimizer()
        self.history = QTrainHistory()

    def _build_optimizer(self):
        opt_name = self.config.optimizer.lower()
        params   = list(self.model.parameters())
        if opt_name == "adam":
            return torch.optim.Adam(params, lr=self.config.lr)
        elif opt_name == "lbfgs":
            return torch.optim.LBFGS(params, lr=self.config.lr, max_iter=20,
                                     history_size=50, line_search_fn="strong_wolfe")
        elif opt_name == "sgd":
            return torch.optim.SGD(params, lr=self.config.lr, momentum=0.9)
        raise ValueError(f"Unknown optimizer {self.config.optimizer!r}")

    def _step(self, x_col: torch.Tensor, x_bc: Optional[torch.Tensor]) -> Dict[str, float]:
        """Single gradient step. Returns loss dict for logging."""

        if self.config.optimizer == "lbfgs":
            def closure():
                self._opt.zero_grad()
                loss, _ = self._compute_loss(x_col, x_bc)
                loss.backward()
                return loss
            self._opt.step(closure)
            with torch.no_grad():
                total, extras = self._compute_loss(x_col, x_bc)
        else:
            self._opt.zero_grad()
            total, extras = self._compute_loss(x_col, x_bc)
            total.backward()
            self._opt.step()

        return {
            "total": float(total.detach()),
            **{k: float(v.detach()) if hasattr(v, "detach") else float(v)
               for k, v in extras.items()},
        }

    def _compute_loss(
        self,
        x_col: torch.Tensor,
        x_bc: Optional[torch.Tensor],
    ):
        """Compute total loss and extras dict."""
        result = self.physics_loss_fn(self.model, x_col)

        # Support returning (loss, extras) or just loss
        if isinstance(result, (tuple, list)) and len(result) == 2:
            pde_loss, extras = result
        else:
            pde_loss, extras = result, {}

        total = pde_loss

        if x_bc is not None and self.bc_loss_fn is not None:
            bc_loss = self.bc_loss_fn(self.model, x_bc)
            total   = total + self.config.lambda_bc * bc_loss
            extras["bc_loss"] = bc_loss

        return total, extras

    def train(
        self,
        x_col: torch.Tensor,
        x_bc: Optional[torch.Tensor] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> QTrainHistory:
        """
        Run the training loop.

        Parameters
        ----------
        x_col : Tensor (N, dim)
            Collocation (interior domain) points.
        x_bc : Tensor (M, dim), optional
            Boundary condition points.
        callbacks : list of Callable, optional
            Each callback receives ``(epoch, loss_dict, model)`` at every
            ``print_every`` interval.

        Returns
        -------
        QTrainHistory
            Object with ``epochs``, ``total_loss``, ``pde_loss``, ``energy``,
            and ``norm_sq`` lists.
        """
        x_col = x_col.to(self._device)
        if x_bc is not None:
            x_bc = x_bc.to(self._device)

        t0 = time.time()
        cfg = self.config

        for epoch in range(1, cfg.epochs + 1):
            losses = self._step(x_col, x_bc)

            self.history.epochs.append(epoch)
            self.history.total_loss.append(losses.get("total", 0.0))
            self.history.pde_loss.append(losses.get("pde_loss", losses.get("total", 0.0)))
            self.history.energy.append(losses.get("energy", float("nan")))
            self.history.norm_sq.append(losses.get("norm_sq", float("nan")))

            if cfg.print_every and (epoch == 1 or epoch % cfg.print_every == 0):
                elapsed = time.time() - t0
                msg = (f"[QTrainer] epoch {epoch:5d}/{cfg.epochs}  "
                       f"loss={losses['total']:.4e}  "
                       f"elapsed={elapsed:.1f}s")
                if "energy" in losses:
                    msg += f"  E={losses['energy']:.4f}"
                print(msg)

                if callbacks:
                    for cb in callbacks:
                        cb(epoch, losses, self.model)

        return self.history


# ── Parameter-shift standalone utility ───────────────────────────────────────

def parameter_shift_gradient(
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    weights: torch.Tensor,
    shift: float = math.pi / 2,
) -> torch.Tensor:
    """
    Compute exact gradient of ``loss_fn`` w.r.t. ``weights`` using the
    parameter-shift rule.

    .. math::

        \\frac{\\partial \\langle O \\rangle}{\\partial \\theta_i} =
        \\frac{1}{2 \\sin(s)}
        \\left[
            \\langle O \\rangle_{\\theta_i + s}
            - \\langle O \\rangle_{\\theta_i - s}
        \\right]

    Parameters
    ----------
    loss_fn : Callable
        ``loss_fn(weights) → scalar``. Must not use autograd on ``weights``.
    weights : Tensor (n_params,)
        Current circuit parameters.
    shift : float
        Shift value. Default π/2 gives exact gradients for standard gates.

    Returns
    -------
    Tensor (n_params,) — exact gradient vector.
    """
    grad = torch.zeros_like(weights)
    with torch.no_grad():
        for i in range(len(weights)):
            w_plus  = weights.clone(); w_plus[i]  += shift
            w_minus = weights.clone(); w_minus[i] -= shift
            f_plus  = loss_fn(w_plus)
            f_minus = loss_fn(w_minus)
            grad[i] = (f_plus - f_minus) / (2.0 * math.sin(shift))
    return grad
