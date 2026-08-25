from __future__ import annotations
"""Bayesian Physics-Informed Neural Networks (B-PINNs).

Implements the variational-inference (VI) formulation from:
  Yang, L. et al. (2021) "B-PINNs: Bayesian physics-informed neural networks
  for forward and inverse PDE problems with noisy data." J. Comput. Phys.
  arXiv:2003.06097

Architecture
------------
Each weight in the network is treated as a Gaussian random variable.
Training maximises the ELBO:

    ELBO = E_q[log p(data | θ)] − KL(q(θ) ‖ p(θ))

where:
  - p(θ) = N(0, σ²_prior I)  — weight prior
  - q(θ) = N(μ, diag(σ²))   — mean-field variational posterior
  - p(data | θ) combines PDE-residual and observation likelihoods

Uncertainty Quantification
--------------------------
Call ``predict_ensemble(inputs, n_samples)`` to draw n_samples forward
passes from the posterior, yielding a mean and std over the field.

Usage
-----
    model = BayesianPINN(in_dim=2, out_dim=1, hidden=[64, 64])
    loss  = model.elbo_loss(inputs, pde_residual_fn, obs_x=..., obs_y=...)
    mean, std = model.predict_ensemble(inputs, n_samples=100)
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PINNBase, PINNOutput


# ---------------------------------------------------------------------------
# Bayesian linear layer (reparameterisation trick)
# ---------------------------------------------------------------------------

class BayesianLinear(nn.Module):
    """Linear layer whose weights are Gaussian random variables.

    Parameters
    ----------
    in_features, out_features:
        Standard layer dimensions.
    prior_std:
        Standard deviation of the isotropic Gaussian weight prior.
    init_rho:
        Initial value of ρ (log-variance = 2 * log(softplus(ρ))).
        A more negative value → smaller initial posterior variance.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        prior_std: float = 1.0,
        init_rho: float = -3.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Posterior mean
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu   = nn.Parameter(torch.zeros(out_features)) if bias else None

        # Posterior log-ρ  (σ = softplus(ρ))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), init_rho))
        self.bias_rho   = nn.Parameter(torch.full((out_features,), init_rho)) if bias else None

        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))

    # ------------------------------------------------------------------

    def _sigma(self, rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single stochastic forward pass (reparameterisation trick)."""
        w_eps = torch.randn_like(self.weight_mu)
        w = self.weight_mu + self._sigma(self.weight_rho) * w_eps

        bias = None
        if self.bias_mu is not None:
            b_eps = torch.randn_like(self.bias_mu)
            bias  = self.bias_mu + self._sigma(self.bias_rho) * b_eps

        return F.linear(x, w, bias)

    def deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic forward using posterior means only."""
        return F.linear(x, self.weight_mu, self.bias_mu)

    # ------------------------------------------------------------------
    # KL divergence  KL(q ‖ p)

    def kl(self) -> torch.Tensor:
        """Analytical KL(N(μ,σ²) ‖ N(0, σ²_prior)) summed over all weights."""
        prior_var = self.prior_std ** 2

        def _kl_gaussian(mu: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
            sigma_sq = self._sigma(rho) ** 2
            return 0.5 * (
                sigma_sq / prior_var
                + mu.pow(2) / prior_var
                - 1.0
                + math.log(prior_var)
                - torch.log(sigma_sq)
            ).sum()

        kl = _kl_gaussian(self.weight_mu, self.weight_rho)
        if self.bias_mu is not None:
            kl = kl + _kl_gaussian(self.bias_mu, self.bias_rho)
        return kl


# ---------------------------------------------------------------------------
# Full Bayesian PINN
# ---------------------------------------------------------------------------

@dataclass
class BPINNConfig:
    """Configuration for BayesianPINN."""
    in_dim: int = 2
    out_dim: int = 1
    hidden: List[int] = field(default_factory=lambda: [64, 64, 64])
    activation: str = "tanh"
    prior_std: float = 1.0
    init_rho: float = -3.0
    # Noise levels for likelihood terms (as standard deviations)
    noise_pde: float = 0.01      # σ_f  — PDE-residual noise
    noise_obs: float = 0.01      # σ_y  — observation noise
    # KL annealing (β-VAE style): weight on KL term
    kl_weight: float = 1.0
    # Inverse PDE parameters
    inverse_params: Optional[List[str]] = None
    initial_guesses: Optional[Dict[str, float]] = None


class BayesianPINN(PINNBase):
    """Bayesian Physics-Informed Neural Network via mean-field VI.

    The network can be used for:
    - **Forward problems**: solve a PDE given BCs/ICs.
    - **Inverse problems**: infer PDE parameters from noisy data.
    - **Uncertainty quantification**: posterior predictive mean and std.

    Parameters
    ----------
    config : BPINNConfig
        Full configuration (use the dataclass for all options).
    in_dim, out_dim, hidden, activation :
        Shortcut kwargs (override config if both provided).

    Examples
    --------
    >>> model = BayesianPINN(in_dim=1, out_dim=1, hidden=[32, 32])
    >>> # ELBO training step
    >>> loss = model.elbo_loss(
    ...     inputs=(x_col,),
    ...     pde_residual_fn=lambda u, x: torch.autograd.grad(u.sum(), x, create_graph=True)[0] + u,
    ...     obs_inputs=(x_obs,), obs_targets=u_obs,
    ... )
    >>> # Posterior predictive
    >>> mean, std = model.predict_ensemble((x_test,), n_samples=200)
    """

    def __init__(
        self,
        in_dim: int = 2,
        out_dim: int = 1,
        hidden: List[int] = (64, 64, 64),
        activation: str = "tanh",
        *,
        config: Optional[BPINNConfig] = None,
        prior_std: float = 1.0,
        init_rho: float = -3.0,
        noise_pde: float = 0.01,
        noise_obs: float = 0.01,
        kl_weight: float = 1.0,
        inverse_params_names: Optional[List[str]] = None,
        initial_guesses: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        cfg = config or BPINNConfig(
            in_dim=in_dim, out_dim=out_dim,
            hidden=list(hidden), activation=activation,
            prior_std=prior_std, init_rho=init_rho,
            noise_pde=noise_pde, noise_obs=noise_obs,
            kl_weight=kl_weight,
            inverse_params=inverse_params_names,
            initial_guesses=initial_guesses,
        )
        self.cfg = cfg

        # -- activation
        acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        act_cls = acts.get(cfg.activation.lower(), nn.Tanh)

        # -- Bayesian MLP
        dims = [cfg.in_dim, *cfg.hidden, cfg.out_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(
                BayesianLinear(dims[i], dims[i + 1],
                               prior_std=cfg.prior_std, init_rho=cfg.init_rho)
            )
            if i < len(dims) - 2:
                layers.append(act_cls())
        self.bayes_layers = nn.ModuleList(layers)

        # -- inverse parameters (optional)
        self.inverse_params = nn.ParameterDict()
        if cfg.inverse_params:
            ig = cfg.initial_guesses or {}
            for name in cfg.inverse_params:
                val = float(ig.get(name, 0.1))
                self.inverse_params[name] = nn.Parameter(torch.tensor(val))

    # ------------------------------------------------------------------
    # Forward passes

    def _apply_layers(self, x: torch.Tensor, *, deterministic: bool = False) -> torch.Tensor:
        for layer in self.bayes_layers:
            if isinstance(layer, BayesianLinear):
                x = layer.deterministic(x) if deterministic else layer(x)
            else:
                x = layer(x)
        return x

    def forward(
        self,
        *inputs: torch.Tensor,
        physics_fn: Optional[Callable] = None,
        physics_data: Optional[Dict] = None,
        deterministic: bool = False,
    ) -> PINNOutput:
        x = self._concat_inputs(inputs)
        y = self._apply_layers(x, deterministic=deterministic)

        losses: Dict[str, torch.Tensor] = {"total": self._zeros()}

        if physics_fn is not None and physics_data is not None:
            total_phys, comps = physics_fn(self, physics_data)
            losses["physics"] = total_phys
            for k, v in comps.items():
                if k != "total":
                    losses[k] = torch.as_tensor(v, device=y.device, dtype=y.dtype)
            losses["total"] = losses["total"] + losses["physics"]

        return PINNOutput(y=y, losses=losses, extras={"deterministic": deterministic})

    def _concat_inputs(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(inputs) == 1:
            x = inputs[0]
            return x.unsqueeze(-1) if x.ndim == 1 else x
        cols = [t.unsqueeze(-1) if t.ndim == 1 else t for t in inputs]
        return torch.cat(cols, dim=1)

    # ------------------------------------------------------------------
    # KL divergence (summed over all Bayesian layers)

    def kl_divergence(self) -> torch.Tensor:
        kl = torch.zeros((), device=next(self.parameters()).device)
        for layer in self.bayes_layers:
            if isinstance(layer, BayesianLinear):
                kl = kl + layer.kl()
        return kl

    # ------------------------------------------------------------------
    # ELBO loss

    def elbo_loss(
        self,
        inputs: Tuple[torch.Tensor, ...],
        pde_residual_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        obs_inputs: Optional[Tuple[torch.Tensor, ...]] = None,
        obs_targets: Optional[torch.Tensor] = None,
        n_mc: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """Compute the (negative) ELBO loss.

        ELBO = E_q[log p(R=0|θ) + log p(y_obs|θ)] − β · KL(q‖p)

        Parameters
        ----------
        inputs :
            Collocation / interior points — tuple of tensors passed to forward().
        pde_residual_fn :
            Callable (u_pred, x_cat) → residual tensor.
            Should return a tensor of shape (N,) or (N, d).
        obs_inputs :
            Observation / boundary points.
        obs_targets :
            Ground-truth values at observation points.
        n_mc :
            Number of MC samples for the expectation (1 is typical for SGD).

        Returns
        -------
        Dict with keys: ``"elbo"``, ``"kl"``, ``"nll_pde"``, ``"nll_obs"``, ``"total"``.
        """
        cfg = self.cfg
        x_col = self._concat_inputs(inputs)
        if not x_col.requires_grad and pde_residual_fn is not None:
            x_col = x_col.requires_grad_(True)

        nll_pde = torch.zeros((), device=x_col.device)
        nll_obs = torch.zeros((), device=x_col.device)

        for _ in range(n_mc):
            # -- PDE residual likelihood
            if pde_residual_fn is not None:
                u_col = self._apply_layers(x_col)
                R = pde_residual_fn(u_col, x_col)
                nll_pde = nll_pde + (R.pow(2).mean() / (2.0 * cfg.noise_pde ** 2))

            # -- Observation likelihood
            if obs_inputs is not None and obs_targets is not None:
                x_obs = self._concat_inputs(obs_inputs)
                u_obs_pred = self._apply_layers(x_obs)
                nll_obs = nll_obs + (
                    (u_obs_pred - obs_targets).pow(2).mean()
                    / (2.0 * cfg.noise_obs ** 2)
                )

        nll_pde = nll_pde / n_mc
        nll_obs = nll_obs / n_mc

        kl = self.kl_divergence()
        total = nll_pde + nll_obs + cfg.kl_weight * kl

        return {
            "total":   total,
            "elbo":    -(nll_pde + nll_obs) - cfg.kl_weight * kl,
            "kl":       kl,
            "nll_pde":  nll_pde,
            "nll_obs":  nll_obs,
        }

    # ------------------------------------------------------------------
    # Posterior predictive

    @torch.no_grad()
    def predict_ensemble(
        self,
        inputs: Tuple[torch.Tensor, ...],
        n_samples: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw *n_samples* posterior predictive samples.

        Returns
        -------
        mean : Tensor  shape (N, out_dim)
        std  : Tensor  shape (N, out_dim)
        """
        x = self._concat_inputs(inputs)
        preds = []
        for _ in range(n_samples):
            preds.append(self._apply_layers(x))
        stack = torch.stack(preds, dim=0)   # (S, N, out_dim)
        return stack.mean(0), stack.std(0)

    @torch.no_grad()
    def predict_mean(self, inputs: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Deterministic prediction using posterior mean weights."""
        x = self._concat_inputs(inputs)
        return self._apply_layers(x, deterministic=True)

    # ------------------------------------------------------------------
    # Factory-compatible predict

    def predict(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.predict_mean(inputs)
