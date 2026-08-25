"""Inverse PINN: jointly optimize network weights and unknown scalar
physical parameters from sparse sensor observations plus a PDE/ODE residual.

Convention: the caller supplies `residual_fn(model, x_col, param_values) ->
Tensor | Sequence[Tensor]` that assembles whatever residual their governing
equation needs, using `param_values[name]` (a dict of the current
`InverseParam` values, differentiable) wherever an unknown parameter
belongs -- the same "supply the physics, we handle the training loop"
pattern used elsewhere in this package (e.g. `pinneapple_neural.trainer.
graybox`'s `known_residual_fn`, or the `reaction_kinetics_network` PDE
kind's `rate_fn`). No equation is hardcoded here, so this works for any
inverse problem with one or more unknown scalar parameters, not a fixed
catalog of problem types.

Robustness features (built in, not opt-in extras):
- Gradient clipping applied to both network weights and inverse parameters
- Early stopping on loss plateau
- Adaptive PDE-vs-data loss weighting via an EMA of each loss's own
  magnitude (mirrors `pinneapple_neural.trainer.graybox`'s balancing rule --
  prevents whichever term is larger from dominating training)
- Pluggable collocation sampling (uniform / Latin hypercube / Sobol)
- Optional log-space parameterization for positivity-constrained parameters
  (e.g. a diffusivity or rate constant that must stay > 0)

Typical workflow
-----------------
>>> from pinneapple_neural.trainer.inverse_pinn import InverseParam, InversePINN
>>> def residual_fn(model, x_col, params):
...     x_col.requires_grad_(True)
...     y = model(x_col)
...     dydx = torch.autograd.grad(y, x_col, torch.ones_like(y), create_graph=True)[0]
...     return dydx + params["k"] * y   # dy/dx = -k*y
>>> k_param = InverseParam(initial_value=1.0, log_space=True, name="k")
>>> ipinn = InversePINN(model, [k_param], residual_fn, sensor_x, sensor_y, domain_bounds=[(0.0, 5.0)])
>>> history = ipinn.train(n_epochs=3000)
>>> ipinn.get_identified_params()  # {"k": <recovered value>}
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class InverseParam(nn.Module):
    """Wraps a scalar as a learnable parameter. Supports a log-space
    transform for positivity-constrained parameters (the raw trainable
    value is log(param); `.value` exponentiates it, so it is always > 0
    and unconstrained-Adam-friendly even near 0)."""

    def __init__(self, initial_value: float, log_space: bool = False, name: str = "param"):
        super().__init__()
        self.log_space = log_space
        self.name = name
        if log_space:
            self.raw = nn.Parameter(torch.tensor(float(np.log(initial_value + 1e-12))))
        else:
            self.raw = nn.Parameter(torch.tensor(float(initial_value)))

    @property
    def value(self) -> torch.Tensor:
        if self.log_space:
            return torch.exp(self.raw)
        return self.raw

    def __repr__(self):
        return f"InverseParam({self.name}={float(self.value.item()):.4e}, log_space={self.log_space})"


class InversePINN:
    """Jointly optimizes network weights (theta) and unknown physical
    parameters (lambda) from sparse sensor observations plus a caller-
    supplied PDE/ODE residual (see module docstring)."""

    def __init__(
        self,
        model: nn.Module,
        inverse_params: List[InverseParam],
        residual_fn: Any,
        sensor_points: torch.Tensor,
        sensor_values: torch.Tensor,
        domain_bounds: list,
        n_col: int = 2048,
        lr: float = 1e-3,
        lr_params: float = 1e-2,
        device: str = "cpu",
        sampling_method: str = "uniform",
        patience: int = 300,
    ):
        self.model = model
        self.inverse_params = nn.ModuleList(inverse_params)
        self.residual_fn = residual_fn
        self.sensor_points = sensor_points.to(device)
        self.sensor_values = sensor_values.to(device)
        self.domain_bounds = domain_bounds
        self.n_col = n_col
        self.device = device
        self.sampling_method = sampling_method
        self.patience = patience

        self.optimizer = torch.optim.Adam([
            {"params": model.parameters(), "lr": lr},
            {"params": list(self.inverse_params.parameters()), "lr": lr_params},
        ])

        self.rng = np.random.default_rng(42)
        self.dim = len(domain_bounds)

    def _sample_collocation(self) -> torch.Tensor:
        method = self.sampling_method
        n, dim = self.n_col, self.dim

        if method == "lhs":
            from scipy.stats.qmc import LatinHypercube
            s = LatinHypercube(d=dim, seed=int(self.rng.integers(0, 2**31)))
            pts = s.random(n).astype(np.float32)
        elif method == "sobol":
            import math
            from scipy.stats.qmc import Sobol
            n_pow2 = 2 ** math.ceil(math.log2(max(n, 2)))
            s = Sobol(d=dim, scramble=True, seed=int(self.rng.integers(0, 2**31)))
            pts = s.random(n_pow2).astype(np.float32)[:n]
        else:
            pts = self.rng.random((n, dim)).astype(np.float32)

        for i, (lo, hi) in enumerate(self.domain_bounds):
            pts[:, i] = pts[:, i] * (hi - lo) + lo
        return torch.tensor(pts, device=self.device)

    def train(self, n_epochs: int = 3000) -> Dict:
        history: Dict = {
            "total": [], "pde": [], "data": [],
            "params": {p.name: [] for p in self.inverse_params},
        }

        best_loss = float("inf")
        no_improve = 0
        ema_data, ema_pde, ema_alpha = 1.0, 1.0, 0.98

        for epoch in range(n_epochs):
            self.optimizer.zero_grad(set_to_none=True)

            x_col = self._sample_collocation().requires_grad_(True)
            param_values = {p.name: p.value for p in self.inverse_params}

            L_pde = torch.tensor(0.0, device=self.device)
            try:
                res = self.residual_fn(self.model, x_col, param_values)
                if isinstance(res, (list, tuple)):
                    L_pde = sum(r.pow(2).mean() for r in res)
                else:
                    L_pde = res.pow(2).mean()
            except Exception:
                logger.warning("PDE residual failed at epoch %d", epoch, exc_info=True)

            pred_sensor = self.model(self.sensor_points)
            L_data = ((pred_sensor - self.sensor_values) ** 2).mean()

            with torch.no_grad():
                ema_data = ema_alpha * ema_data + (1 - ema_alpha) * float(L_data.item() + 1e-12)
                ema_pde = ema_alpha * ema_pde + (1 - ema_alpha) * float(L_pde.item() + 1e-12)
                scale = ema_data / (ema_pde + 1e-12)
                w_pde_eff = min(max(scale, 0.01), 100.0) * 100.0

            total = w_pde_eff * L_pde + 100.0 * L_data
            total.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            for p in self.inverse_params:
                if p.raw.grad is not None:
                    p.raw.grad.data.clamp_(-1.0, 1.0)

            self.optimizer.step()

            loss_val = float(total.item())
            if loss_val < best_loss - 1e-7:
                best_loss = loss_val
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

            history["total"].append(loss_val)
            history["pde"].append(float(L_pde.item()))
            history["data"].append(float(L_data.item()))
            for p in self.inverse_params:
                history["params"][p.name].append(float(p.value.item()))

        return history

    def get_identified_params(self) -> Dict[str, float]:
        return {p.name: float(p.value.item()) for p in self.inverse_params}
