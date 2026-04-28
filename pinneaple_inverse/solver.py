"""High-level InverseProblemSolver combining all inverse-problem components.

Wires together:
- A physics model (PINN or any nn.Module)
- Observation operator H
- Data-misfit / noise model D
- Regularization R
- Optimizer choice: gradient-based (Adam / L-BFGS) or derivative-free (EKI)

Quick start
-----------
>>> from pinneaple_inverse.solver import InverseProblemSolver, InverseSolverConfig
>>> from pinneaple_inverse.noise_models import HuberMisfit
>>> from pinneaple_inverse.regularization import TikhonovRegularizer
>>> from pinneaple_inverse.obs_operator import PointObsOperator

>>> cfg = InverseSolverConfig(method="adam", n_iters=2000, lr=5e-4)
>>> solver = InverseProblemSolver(
...     model=pinn,
...     obs_operator=PointObsOperator(sensor_locs),
...     data_misfit=HuberMisfit(delta=0.5),
...     regularizer=TikhonovRegularizer(lambda_reg=1e-3),
...     config=cfg,
... )
>>> result = solver.solve(y_obs, theta_init)
>>> print(result.theta_opt, result.final_misfit)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .noise_models import DataMisfitBase, GaussianMisfit
from .regularization import RegularizerBase
from .obs_operator import ObsOperatorBase, PointObsOperator


# ──────────────────────────────────────────────────────────────────────────────
# Config and result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InverseSolverConfig:
    """Configuration for InverseProblemSolver.

    Parameters
    ----------
    method : str
        Optimisation method:
        - "adam"   — gradient-based, Adam optimiser.
        - "lbfgs"  — gradient-based, L-BFGS (good for few parameters).
        - "eki"    — derivative-free Ensemble Kalman Inversion.
        - "teki"   — derivative-free Tikhonov-regularised EKI.
    n_iters : int
        Gradient iterations (adam/lbfgs) or EKI update steps.
    lr : float
        Learning rate (adam only).
    lbfgs_max_iter : int
        Inner iterations per L-BFGS step.
    device : str
    grad_clip : float
        Gradient clip norm (0 = disabled).
    print_every : int
        Print progress every N iterations (0 = silent).
    # EKI-specific
    eki_n_ensemble : int
        Ensemble size for EKI/TEKI.
    eki_noise_std : float
        Observation noise std for EKI.
    eki_lambda_reg : float
        Tikhonov weight for TEKI.
    eki_init_spread : float
        Ensemble spread initialisation.
    seed : int, optional
    """
    method: str = "adam"
    n_iters: int = 1000
    lr: float = 1e-3
    lbfgs_max_iter: int = 20
    device: str = "cpu"
    grad_clip: float = 0.0
    print_every: int = 100
    # EKI
    eki_n_ensemble: int = 50
    eki_noise_std: float = 0.01
    eki_lambda_reg: float = 0.0
    eki_init_spread: float = 0.1
    seed: Optional[int] = None


@dataclass
class InverseSolverResult:
    """Result of an inverse problem solve.

    Attributes
    ----------
    theta_opt : np.ndarray or dict
        Optimal parameters.  If ``theta_init`` was a ``nn.ParameterDict``,
        returns a dict of numpy arrays.
    final_misfit : float
    final_regularization : float
    final_total_loss : float
    loss_history : list[float]
    n_iters : int
    method : str
    converged : bool
    """
    theta_opt: Any
    final_misfit: float
    final_regularization: float
    final_total_loss: float
    loss_history: List[float]
    n_iters: int
    method: str
    converged: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Main solver
# ──────────────────────────────────────────────────────────────────────────────

class InverseProblemSolver:
    """Unified solver for physics-informed inverse problems.

    Parameters
    ----------
    model : nn.Module
        Forward PINN or surrogate model.  Must expose learnable parameters
        either via ``model.inverse_params`` (nn.ParameterDict) or via
        explicit ``theta`` argument.
    obs_operator : ObsOperatorBase
        Maps model output to observable space.
    data_misfit : DataMisfitBase, optional
        Noise model / negative log-likelihood.  Defaults to Gaussian MSE.
    regularizer : RegularizerBase, optional
        Regularization term R(θ).  Optional; set to None for no regularization.
    config : InverseSolverConfig, optional
    extra_loss_fn : callable, optional
        Additional physics or constraint loss ``fn(model) → scalar``.
    """

    def __init__(
        self,
        model: nn.Module,
        obs_operator: ObsOperatorBase,
        data_misfit: Optional[DataMisfitBase] = None,
        regularizer: Optional[RegularizerBase] = None,
        config: Optional[InverseSolverConfig] = None,
        extra_loss_fn: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.obs_operator = obs_operator
        self.data_misfit = data_misfit or GaussianMisfit()
        self.regularizer = regularizer
        self.cfg = config or InverseSolverConfig()
        self.extra_loss_fn = extra_loss_fn

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_inverse_params(self) -> Optional[nn.ParameterDict]:
        """Return the nn.ParameterDict if the model exposes one."""
        if hasattr(self.model, "inverse_params"):
            return self.model.inverse_params
        return None

    def _params_to_numpy(self) -> np.ndarray:
        """Flatten model inverse_params to a numpy vector."""
        pd = self._get_inverse_params()
        if pd is not None:
            return np.concatenate([p.detach().cpu().numpy().ravel() for p in pd.values()])
        raise RuntimeError(
            "Model has no inverse_params. Pass theta_init explicitly and use "
            "set_theta_fn to update the model."
        )

    def _set_params_from_numpy(self, theta: np.ndarray) -> None:
        """Write numpy vector back into model inverse_params."""
        pd = self._get_inverse_params()
        if pd is None:
            return
        offset = 0
        for param in pd.values():
            n = param.numel()
            param.data.copy_(
                torch.tensor(theta[offset: offset + n], dtype=param.dtype)
                .reshape(param.shape)
            )
            offset += n

    def _build_total_loss(
        self,
        x_obs: torch.Tensor,
        y_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate total loss: misfit + regularization + physics."""
        # Forward model through observation operator
        y_pred = self.obs_operator(self.model, x_obs)
        misfit = self.data_misfit(y_pred, y_obs)

        total = misfit

        if self.regularizer is not None:
            pd = self._get_inverse_params()
            if pd is not None:
                theta_vec = torch.cat([p.reshape(-1) for p in pd.values()])
                total = total + self.regularizer(theta_vec)

        if self.extra_loss_fn is not None:
            total = total + self.extra_loss_fn(self.model)

        return total

    # ── Gradient-based: Adam ──────────────────────────────────────────────────

    def _solve_adam(
        self,
        x_obs: torch.Tensor,
        y_obs: torch.Tensor,
    ) -> InverseSolverResult:
        cfg = self.cfg
        device = torch.device(cfg.device)
        self.model.to(device)
        x_obs = x_obs.to(device)
        y_obs = y_obs.to(device)

        # Determine which parameters to optimise
        pd = self._get_inverse_params()
        params = list(pd.values()) if pd is not None else list(self.model.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr)

        loss_history: List[float] = []

        for i in range(1, cfg.n_iters + 1):
            opt.zero_grad()
            loss = self._build_total_loss(x_obs, y_obs)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()

            lv = float(loss.detach())
            loss_history.append(lv)

            if cfg.print_every > 0 and i % cfg.print_every == 0:
                param_str = ""
                if pd is not None:
                    vals = {k: float(v.detach()) for k, v in pd.items()}
                    param_str = "  params=" + str({k: f"{v:.4f}" for k, v in vals.items()})
                print(f"  [Adam  {i:5d}/{cfg.n_iters}]  loss={lv:.4e}{param_str}")

        return self._build_result("adam", loss_history)

    # ── Gradient-based: L-BFGS ───────────────────────────────────────────────

    def _solve_lbfgs(
        self,
        x_obs: torch.Tensor,
        y_obs: torch.Tensor,
    ) -> InverseSolverResult:
        cfg = self.cfg
        device = torch.device(cfg.device)
        self.model.to(device)
        x_obs = x_obs.to(device)
        y_obs = y_obs.to(device)

        pd = self._get_inverse_params()
        params = list(pd.values()) if pd is not None else list(self.model.parameters())
        opt = torch.optim.LBFGS(
            params,
            lr=cfg.lr,
            max_iter=cfg.lbfgs_max_iter,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        loss_history: List[float] = []
        _last_loss: List[float] = [float("inf")]

        for i in range(1, cfg.n_iters + 1):
            def closure():
                opt.zero_grad()
                loss = self._build_total_loss(x_obs, y_obs)
                loss.backward()
                _last_loss[0] = float(loss.detach())
                return loss

            opt.step(closure)
            loss_history.append(_last_loss[0])

            if cfg.print_every > 0 and i % cfg.print_every == 0:
                print(f"  [LBFGS {i:5d}/{cfg.n_iters}]  loss={_last_loss[0]:.4e}")

        return self._build_result("lbfgs", loss_history)

    # ── Derivative-free: EKI / TEKI ──────────────────────────────────────────

    def _solve_eki(
        self,
        x_obs: torch.Tensor,
        y_obs: torch.Tensor,
        method: str,
    ) -> InverseSolverResult:
        from .ensemble_kalman import EKIConfig, EnsembleKalmanInversion, IteratedEKI

        cfg = self.cfg
        y_np = y_obs.detach().cpu().numpy().ravel()
        k = len(y_np)

        # Wrap the PINN forward model for numpy-based EKI
        def _forward_fn(theta_batch: np.ndarray) -> np.ndarray:
            results = []
            for theta_row in theta_batch:
                self._set_params_from_numpy(theta_row)
                with torch.no_grad():
                    y_pred = self.obs_operator(self.model, x_obs.to(cfg.device))
                results.append(y_pred.detach().cpu().numpy().ravel())
            return np.array(results)

        eki_cfg = EKIConfig(
            n_ensemble=cfg.eki_n_ensemble,
            n_iterations=cfg.n_iters,
            noise_std=cfg.eki_noise_std,
            lambda_reg=cfg.eki_lambda_reg,
            init_spread=cfg.eki_init_spread,
            seed=cfg.seed,
            verbose=(cfg.print_every > 0),
            print_every=max(1, cfg.print_every),
        )

        theta_init = self._params_to_numpy()

        SolverClass = IteratedEKI if method == "teki" else EnsembleKalmanInversion
        eki = SolverClass(_forward_fn, eki_cfg)
        history = eki.run(y_np, theta_init)

        # Update model with final ensemble mean
        self._set_params_from_numpy(eki.theta_mean)

        return InverseSolverResult(
            theta_opt=eki.theta_mean.copy(),
            final_misfit=history.final_misfit(),
            final_regularization=0.0,
            final_total_loss=history.final_misfit(),
            loss_history=history.data_misfit,
            n_iters=len(history.iterations),
            method=method,
            converged=history.converged(cfg.eki_noise_std ** 2),
        )

    # ── Result builder ────────────────────────────────────────────────────────

    def _build_result(self, method: str, loss_history: List[float]) -> InverseSolverResult:
        pd = self._get_inverse_params()
        if pd is not None:
            theta_opt = {k: v.detach().cpu().numpy() for k, v in pd.items()}
        else:
            theta_opt = self._params_to_numpy()

        final = loss_history[-1] if loss_history else float("nan")
        return InverseSolverResult(
            theta_opt=theta_opt,
            final_misfit=final,
            final_regularization=0.0,
            final_total_loss=final,
            loss_history=loss_history,
            n_iters=len(loss_history),
            method=method,
            converged=(final < 1e-6),
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def solve(
        self,
        y_obs: torch.Tensor,
        x_obs: Optional[torch.Tensor] = None,
        *,
        theta_init: Optional[np.ndarray] = None,
    ) -> InverseSolverResult:
        """Solve the inverse problem.

        Parameters
        ----------
        y_obs : torch.Tensor, shape (N_obs,) or (N_obs, k)
            Observed measurements.
        x_obs : torch.Tensor, optional
            Sensor locations (shape N_obs × d).  If None, uses operator defaults.
        theta_init : np.ndarray, optional
            Initial parameter values.  If None, reads from model.inverse_params.

        Returns
        -------
        InverseSolverResult
        """
        if theta_init is not None:
            self._set_params_from_numpy(theta_init)

        if x_obs is None:
            # Try to get sensor locations from obs_operator
            if hasattr(self.obs_operator, "sensor_locations"):
                x_obs = self.obs_operator.sensor_locations
            else:
                raise ValueError(
                    "x_obs is required when obs_operator has no default sensor_locations."
                )

        method = self.cfg.method.lower()
        if method == "adam":
            return self._solve_adam(x_obs, y_obs)
        if method == "lbfgs":
            return self._solve_lbfgs(x_obs, y_obs)
        if method in ("eki", "teki"):
            return self._solve_eki(x_obs, y_obs, method)
        raise ValueError(f"Unknown method: {method!r}. Choose 'adam', 'lbfgs', 'eki', 'teki'.")


__all__ = [
    "InverseSolverConfig",
    "InverseSolverResult",
    "InverseProblemSolver",
]
