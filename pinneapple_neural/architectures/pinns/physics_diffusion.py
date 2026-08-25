from __future__ import annotations
"""Physics-Informed Diffusion Model (PIDiff).

Composable generative model for physics-AI inspired by the PhysicsNeMo
Diffusion Module architecture (NVIDIA, 2025):

  "Introducing the PhysicsNeMo Diffusion Module: Composable, Extensible
   Generative Modeling for Physics-AI"

Design
------
A diffusion model is split into five clear abstractions (all swappable):

  1. NoiseScheduler    — forward process + reverse schedule (VP-DDPM / EDM)
  2. Denoiser network  — time-conditioned MLP / U-Net backbone
  3. DenoisingLoss     — denoising score matching (DSM) objective
  4. Sampler           — DDIM or Euler–Heun ODE integrator
  5. PhysicsGuidance   — DPS guidance: enforce PDE residuals at inference time
                         without retraining

Capabilities
------------
- **Large ensembles**: draw N independent samples → UQ at zero marginal cost.
- **Inverse problems**: steer sampling with DPS toward sparse observations.
- **Physics-constrained generation**: add a PDE-residual guidance term at
  inference; the physics constraint requires no retraining.

Quick start
-----------
    # Build model
    sched   = VPNoiseScheduler(T=1000)
    net     = MLPDenoiser(x_dim=64, t_emb_dim=64, hidden=[256, 256])
    loss_fn = DSMLoss(net, sched)
    model   = PhysicsInformedDiffusion(net, sched)

    # Train (one step)
    x0_batch = ...           # shape (B, x_dim)
    loss = loss_fn(x0_batch)
    loss.backward()

    # Ensemble sampling (pure generative)
    ensemble = model.sample(n_samples=64, x_dim=64, n_steps=50)

    # Physics-constrained sampling
    def pde_residual(x_hat):
        # e.g. Laplacian residual
        return x_hat.pow(2).mean(dim=-1, keepdim=True)

    physics_guidance = PDEResidualGuidance(pde_residual, weight=0.1)
    constrained = model.sample(
        n_samples=8, x_dim=64, n_steps=50,
        guidance=physics_guidance,
    )

    # Data assimilation (DPS toward sparse observations)
    dps = DataConsistencyGuidance(mask=mask, y_obs=y_obs, weight=0.5)
    posterior_samples = model.sample(
        n_samples=32, x_dim=64, n_steps=50,
        guidance=dps,
    )
"""

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PINNBase, PINNOutput


# ===========================================================================
# 1.  Noise schedulers
# ===========================================================================

class NoiseSchedulerBase:
    """Protocol for noise schedulers.

    A concrete scheduler must implement:
        alpha(t), sigma(t)    — signal/noise scaling at continuous time t ∈ [0,1]
        timesteps(T)           — discrete reverse schedule  (T → 0)
        add_noise(x0, t)       — sample from q(x_t | x_0)
        x0_from_pred(pred, xt, t) — recover x_0 from network prediction
    """

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def timesteps(self, T: int) -> torch.Tensor:
        raise NotImplementedError

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x_t, noise) where x_t ~ q(x_t | x_0)."""
        raise NotImplementedError

    def x0_from_pred(
        self, pred: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class VPNoiseScheduler(NoiseSchedulerBase):
    """Variance-preserving (VP-DDPM) linear beta schedule.

    Forward: q(x_t | x_0) = N(√ᾱ_t x_0, (1-ᾱ_t) I)
    """

    def __init__(
        self,
        T: int = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
    ):
        self.T = T
        betas = torch.linspace(beta_min, beta_max, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars)
        self._alpha_bars = alpha_bars           # plain tensor (no nn.Module)

    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)

    def _ab(self, t_idx: torch.Tensor) -> torch.Tensor:
        """Gather ᾱ_t using integer indices in [0, T-1]."""
        idx = t_idx.long().clamp(0, self.T - 1)
        return self._alpha_bars.to(idx.device)[idx]

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return self._ab(t).sqrt()

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return (1.0 - self._ab(t)).sqrt()

    def timesteps(self, T: int) -> torch.Tensor:
        return torch.linspace(self.T - 1, 0, T).long()

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a = self.alpha(t).view(-1, *([1] * (x0.ndim - 1)))
        s = self.sigma(t).view(-1, *([1] * (x0.ndim - 1)))
        eps = torch.randn_like(x0)
        xt = a * x0 + s * eps
        return xt, eps

    def x0_from_pred(
        self, pred: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Recover x_0 from predicted noise (pred = ε_θ)."""
        a = self.alpha(t).view(-1, *([1] * (xt.ndim - 1)))
        s = self.sigma(t).view(-1, *([1] * (xt.ndim - 1)))
        return (xt - s * pred) / a.clamp(min=1e-8)


class EDMNoiseScheduler(NoiseSchedulerBase):
    """Elucidated Diffusion Model (EDM) schedule.

    Karras et al. (2022) "Elucidating the Design Space of Diffusion-Based
    Generative Models". Forward: x_t = x_0 + σ_t * ε.

    σ range: [σ_min, σ_max] on a log-linear schedule.
    """

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def sigma_schedule(self, T: int, device="cpu") -> torch.Tensor:
        steps = torch.arange(T, device=device)
        inv_rho = 1.0 / self.rho
        s = (
            self.sigma_max ** inv_rho
            + steps / (T - 1) * (self.sigma_min ** inv_rho - self.sigma_max ** inv_rho)
        ) ** self.rho
        return s

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return t          # EDM uses σ directly as "time"

    def timesteps(self, T: int) -> torch.Tensor:
        return torch.arange(T)

    def add_noise(
        self, x0: torch.Tensor, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = torch.randn_like(x0)
        s = sigma.view(-1, *([1] * (x0.ndim - 1)))
        return x0 + s * eps, eps

    def x0_from_pred(
        self, pred: torch.Tensor, xt: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        return pred       # EDM networks predict x_0 directly


# ===========================================================================
# 2.  Denoiser network (time-conditioned MLP)
# ===========================================================================

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for noise level / timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation — conditions activations on noise level."""

    def __init__(self, feat_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta  = nn.Linear(cond_dim, feat_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x * (1 + self.gamma(cond)) + self.beta(cond)


class MLPDenoiser(nn.Module):
    """Time-conditioned MLP denoiser for 1-D physics fields.

    Parameters
    ----------
    x_dim :
        Spatial dimension of the field (number of grid points / features).
    t_emb_dim :
        Sinusoidal embedding dimension for the noise level.
    hidden :
        List of hidden layer widths.
    activation :
        Nonlinearity (tanh / relu / gelu / silu).
    predict_x0 :
        If True, the network predicts x_0 (EDM style).
        If False, it predicts ε (VP / DDPM style).
    """

    def __init__(
        self,
        x_dim: int,
        t_emb_dim: int = 64,
        hidden: List[int] = (256, 256, 256),
        activation: str = "gelu",
        predict_x0: bool = False,
    ):
        super().__init__()
        self.x_dim       = x_dim
        self.predict_x0  = predict_x0

        self.t_emb = SinusoidalEmbedding(t_emb_dim)
        self.t_proj = nn.Sequential(
            nn.Linear(t_emb_dim, t_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(t_emb_dim * 2, t_emb_dim),
        )

        acts = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        act_cls = acts.get(activation.lower(), nn.GELU)

        dims = [x_dim, *hidden]
        self.in_proj  = nn.Linear(x_dim, hidden[0])
        self.blocks   = nn.ModuleList()
        self.films    = nn.ModuleList()
        for i in range(len(hidden)):
            in_d  = hidden[i]
            out_d = hidden[i + 1] if i + 1 < len(hidden) else hidden[-1]
            self.blocks.append(nn.Sequential(nn.Linear(in_d, out_d), act_cls()))
            self.films.append(FiLMLayer(out_d, t_emb_dim))

        self.out_proj = nn.Linear(hidden[-1], x_dim)

    def forward(
        self,
        xt: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        xt : (B, x_dim) — noisy field at noise level t
        t  : (B,)       — noise level / timestep index
        cond : (B, c)   — optional external conditioning (ignored in base)

        Returns
        -------
        pred : (B, x_dim) — predicted noise ε or x_0
        """
        t_emb = self.t_proj(self.t_emb(t))
        h = self.in_proj(xt)
        for blk, film in zip(self.blocks, self.films):
            h = film(blk(h), t_emb)
        return self.out_proj(h)


# ===========================================================================
# 3.  Training loss (denoising score matching)
# ===========================================================================

class DSMLoss(nn.Module):
    """Denoising score-matching loss.

    L = E_{t,x_0,ε}[ λ(t) · ‖ε_θ(x_t, t) − ε ‖² ]

    where x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε  (VP schedule).
    """

    def __init__(
        self,
        denoiser: nn.Module,
        scheduler: NoiseSchedulerBase,
        loss_weight_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.scheduler = scheduler
        self.loss_weight_fn = loss_weight_fn or (lambda t: torch.ones_like(t.float()))

    def forward(
        self,
        x0: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x0.shape[0]
        device = x0.device

        # Sample a random noise level per sample
        if isinstance(self.scheduler, EDMNoiseScheduler):
            T_steps = 1000
            sigma_sched = self.scheduler.sigma_schedule(T_steps, device=device)
            idx = torch.randint(0, T_steps, (B,), device=device)
            t = sigma_sched[idx]
        else:
            t = torch.randint(0, self.scheduler.T, (B,), device=device).float()

        xt, eps = self.scheduler.add_noise(x0, t)
        pred = self.denoiser(xt, t, cond)

        w = self.loss_weight_fn(t).view(-1, *([1] * (x0.ndim - 1)))

        if isinstance(self.scheduler, EDMNoiseScheduler):
            target = x0             # EDM denoiser predicts x_0
        else:
            target = eps             # VP denoiser predicts ε

        return (w * (pred - target).pow(2)).mean()


# ===========================================================================
# 4.  Guidance (DPS — plug-and-play at inference)
# ===========================================================================

class GuidanceBase:
    """Protocol for DPS-style guidance."""
    def __call__(
        self,
        x_hat: torch.Tensor,    # predicted x_0
        xt: torch.Tensor,       # current noisy state
        t: torch.Tensor,        # noise level
    ) -> torch.Tensor:
        """Return gradient ∂G/∂x_t to add to the denoising update."""
        raise NotImplementedError


class PDEResidualGuidance(GuidanceBase):
    """Physics-constrained guidance via PDE residual.

    Nudges each denoising step toward lower PDE residual.

    Parameters
    ----------
    residual_fn :
        Callable x_hat → R(x_hat) — must be differentiable via autograd.
    weight :
        Step size / guidance strength  (γ in DPS paper).
    """

    def __init__(
        self,
        residual_fn: Callable[[torch.Tensor], torch.Tensor],
        weight: float = 0.1,
    ):
        self.residual_fn = residual_fn
        self.weight = weight

    def __call__(
        self,
        x_hat: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        with torch.enable_grad():
            x_hat = x_hat.detach().requires_grad_(True)
            R = self.residual_fn(x_hat)
            loss = R.pow(2).mean()
            grad = torch.autograd.grad(loss, x_hat)[0]
        return -self.weight * grad.detach()


class DataConsistencyGuidance(GuidanceBase):
    """DPS guidance for data assimilation / inverse problems.

    Steers sampling toward agreement with sparse observations y_obs
    at masked locations (Diffusion Posterior Sampling, Chung et al. 2022).

    Parameters
    ----------
    mask : BoolTensor (N,) or (B, N)
        True where observations are available.
    y_obs : Tensor (N_obs,) or (B, N_obs)
        Observed values.
    weight : float
        Guidance strength γ.
    """

    def __init__(
        self,
        mask: torch.Tensor,
        y_obs: torch.Tensor,
        weight: float = 0.5,
    ):
        self.mask   = mask
        self.y_obs  = y_obs
        self.weight = weight

    def __call__(
        self,
        x_hat: torch.Tensor,
        xt: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        with torch.enable_grad():
            x_hat = x_hat.detach().requires_grad_(True)
            mask = self.mask.to(x_hat.device)
            y    = self.y_obs.to(x_hat.device)

            pred_obs = x_hat[:, mask] if mask.ndim == 1 else x_hat[mask]
            if pred_obs.shape[-1] != y.shape[-1]:
                y = y[:pred_obs.shape[0]] if y.ndim > 0 else y

            loss = (pred_obs - y).pow(2).mean()
            grad = torch.autograd.grad(loss, x_hat)[0]
        return -self.weight * grad.detach()


class ComposedGuidance(GuidanceBase):
    """Sum of multiple guidance terms."""

    def __init__(self, *guidances: GuidanceBase):
        self.guidances = list(guidances)

    def __call__(self, x_hat, xt, t) -> torch.Tensor:
        total = torch.zeros_like(xt)
        for g in self.guidances:
            total = total + g(x_hat, xt, t)
        return total


# ===========================================================================
# 5.  Sampler (DDIM + Euler–Heun ODE)
# ===========================================================================

def _ddim_step(
    denoiser: nn.Module,
    scheduler: VPNoiseScheduler,
    xt: torch.Tensor,
    t_curr: torch.Tensor,
    t_next: torch.Tensor,
    cond: Optional[torch.Tensor],
    guidance: Optional[GuidanceBase],
    eta: float = 0.0,
) -> torch.Tensor:
    """One DDIM step (η=0 → deterministic, η=1 → DDPM)."""
    B = xt.shape[0]
    tc = t_curr.expand(B).to(xt.device)
    tn = t_next.expand(B).to(xt.device)

    pred = denoiser(xt, tc, cond)
    x0_hat = scheduler.x0_from_pred(pred, xt, tc)

    # Optional DPS guidance
    if guidance is not None:
        grad = guidance(x0_hat, xt, tc)
        x0_hat = x0_hat + grad

    a_c = scheduler.alpha(tc).view(-1, *([1] * (xt.ndim - 1)))
    a_n = scheduler.alpha(tn).view(-1, *([1] * (xt.ndim - 1)))
    s_n = scheduler.sigma(tn).view(-1, *([1] * (xt.ndim - 1)))

    # DDIM update
    eps_hat = (xt - a_c * x0_hat) / scheduler.sigma(tc).view(-1, *([1] * (xt.ndim - 1))).clamp(1e-8)

    # σ_t = η·√((1-ᾱ_{t-1})/(1-ᾱ_t))·√(1-ᾱ_t/ᾱ_{t-1}); η=1 reproduces the DDPM
    # posterior variance, η=0 is deterministic DDIM.
    ab_c = a_c ** 2
    ab_n = a_n ** 2
    sigma_t = eta * torch.sqrt(
        ((1.0 - ab_n) / (1.0 - ab_c).clamp(min=1e-8)).clamp(min=0)
        * (1.0 - ab_c / ab_n.clamp(min=1e-8)).clamp(min=0)
    )
    noise = torch.randn_like(xt)
    return a_n * x0_hat + (s_n ** 2 - sigma_t ** 2).clamp(0).sqrt() * eps_hat + sigma_t * noise


def _edm_euler_step(
    denoiser: nn.Module,
    scheduler: EDMNoiseScheduler,
    xt: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    cond: Optional[torch.Tensor],
    guidance: Optional[GuidanceBase],
) -> torch.Tensor:
    """One EDM Euler step (Karras et al., Algorithm 1)."""
    B = xt.shape[0]
    s = sigma.expand(B).to(xt.device)

    x0_hat = denoiser(xt, s, cond)

    if guidance is not None:
        grad = guidance(x0_hat, xt, s)
        x0_hat = x0_hat + grad

    s_val = s.view(-1, *([1] * (xt.ndim - 1)))
    sn_val = sigma_next.view(-1, *([1] * (xt.ndim - 1)))
    d = (xt - x0_hat) / s_val.clamp(1e-8)
    return xt + (sn_val - s_val) * d


# ===========================================================================
# 6.  Main model — PhysicsInformedDiffusion
# ===========================================================================

@dataclass
class PIDiffConfig:
    x_dim: int = 64
    t_emb_dim: int = 64
    hidden: List[int] = None
    activation: str = "gelu"
    scheduler_type: str = "vp"          # "vp" | "edm"
    T: int = 1000
    beta_min: float = 1e-4
    beta_max: float = 0.02
    sigma_min: float = 0.002
    sigma_max: float = 80.0

    def __post_init__(self):
        if self.hidden is None:
            self.hidden = [256, 256, 256]


class PhysicsInformedDiffusion(PINNBase):
    """Physics-Informed Diffusion Model.

    Combines a learnable denoiser with:
    - DSM training objective
    - Physics guidance at inference (no retraining needed)
    - Data-assimilation guidance (DPS)
    - Large-ensemble generation

    Parameters
    ----------
    denoiser :
        Time-conditioned network (e.g. MLPDenoiser, or any backbone).
        If None, an MLPDenoiser is built from config.
    scheduler :
        Noise scheduler. If None, built from config.scheduler_type.
    config :
        Full configuration dataclass.
    x_dim, hidden, activation :
        Shortcuts for building a default MLPDenoiser.

    Examples
    --------
    >>> model   = PhysicsInformedDiffusion(x_dim=32, hidden=[128, 128])
    >>> loss_fn = DSMLoss(model.denoiser, model.scheduler)
    >>> loss     = loss_fn(x0_batch)
    >>>
    >>> # Physics-constrained ensemble
    >>> physics  = PDEResidualGuidance(lambda u: u.diff(dim=-1), weight=0.05)
    >>> ensemble = model.sample(n_samples=32, n_steps=50, guidance=physics)
    """

    def __init__(
        self,
        x_dim: int = 64,
        hidden: List[int] = (256, 256, 256),
        activation: str = "gelu",
        *,
        denoiser: Optional[nn.Module] = None,
        scheduler: Optional[NoiseSchedulerBase] = None,
        config: Optional[PIDiffConfig] = None,
        scheduler_type: str = "vp",
        T: int = 1000,
    ):
        super().__init__()

        cfg = config or PIDiffConfig(
            x_dim=x_dim, hidden=list(hidden), activation=activation,
            scheduler_type=scheduler_type, T=T,
        )
        self.cfg = cfg

        # Build scheduler
        if scheduler is not None:
            self.scheduler = scheduler
        elif cfg.scheduler_type == "edm":
            self.scheduler = EDMNoiseScheduler(cfg.sigma_min, cfg.sigma_max)
        else:
            self.scheduler = VPNoiseScheduler(cfg.T, cfg.beta_min, cfg.beta_max)

        # Build denoiser
        if denoiser is not None:
            self.denoiser = denoiser
        else:
            self.denoiser = MLPDenoiser(
                x_dim=cfg.x_dim,
                t_emb_dim=cfg.t_emb_dim,
                hidden=cfg.hidden,
                activation=cfg.activation,
                predict_x0=(cfg.scheduler_type == "edm"),
            )

        # Inverse params slot (unused by default — kept for API consistency)
        self.inverse_params = nn.ParameterDict()

    # ------------------------------------------------------------------
    # Forward (denoising — used during training)

    def forward(
        self,
        *inputs: torch.Tensor,
        physics_fn: Optional[Callable] = None,
        physics_data: Optional[Dict] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> PINNOutput:
        """
        When called with a single noisy-field tensor and a noise-level tensor:
            forward(xt, t) → PINNOutput with y = predicted noise/x0.
        """
        if len(inputs) == 2:
            xt, t = inputs
            y = self.denoiser(xt, t, cond)
        elif len(inputs) == 1:
            y = inputs[0]
        else:
            raise ValueError("PhysicsInformedDiffusion.forward expects (xt, t) or (xt,).")

        return PINNOutput(y=y, losses={"total": self._zeros()}, extras={})

    # ------------------------------------------------------------------
    # Sampling

    @torch.no_grad()
    def sample(
        self,
        n_samples: int = 1,
        x_dim: Optional[int] = None,
        n_steps: int = 50,
        guidance: Optional[GuidanceBase] = None,
        cond: Optional[torch.Tensor] = None,
        eta: float = 0.0,
        device: Optional[torch.device] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Generate samples from the learned distribution.

        Parameters
        ----------
        n_samples :
            Number of independent samples (ensemble size).
        x_dim :
            Field dimension (defaults to cfg.x_dim).
        n_steps :
            Number of reverse diffusion steps.
        guidance :
            Optional GuidanceBase (PDEResidualGuidance / DataConsistencyGuidance).
        cond :
            External conditioning tensor (B, c) — passed to denoiser.
        eta :
            DDIM noise parameter (0 = deterministic, 1 = DDPM).
        return_trajectory :
            If True, return list of intermediate states.

        Returns
        -------
        Tensor (n_samples, x_dim) or list thereof.
        """
        dev = device or next(self.denoiser.parameters()).device
        xd  = x_dim or self.cfg.x_dim

        traj = []

        if isinstance(self.scheduler, EDMNoiseScheduler):
            sigma_sched = self.scheduler.sigma_schedule(n_steps, device=dev)
            xt = torch.randn(n_samples, xd, device=dev) * sigma_sched[0]

            for i in range(n_steps - 1):
                s_curr = sigma_sched[i]
                s_next = sigma_sched[i + 1]
                xt = _edm_euler_step(
                    self.denoiser, self.scheduler, xt,
                    s_curr.unsqueeze(0), s_next.unsqueeze(0), cond, guidance
                )
                if return_trajectory:
                    traj.append(xt.clone())
        else:
            timesteps = self.scheduler.timesteps(n_steps)
            xt = torch.randn(n_samples, xd, device=dev)

            for i in range(len(timesteps) - 1):
                t_curr = timesteps[i].float().unsqueeze(0)
                t_next = timesteps[i + 1].float().unsqueeze(0)
                xt = _ddim_step(
                    self.denoiser, self.scheduler, xt,
                    t_curr, t_next, cond, guidance, eta=eta
                )
                if return_trajectory:
                    traj.append(xt.clone())

        return traj if return_trajectory else xt

    # ------------------------------------------------------------------
    # Ensemble UQ

    @torch.no_grad()
    def sample_ensemble(
        self,
        n_samples: int = 64,
        x_dim: Optional[int] = None,
        n_steps: int = 50,
        guidance: Optional[GuidanceBase] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw large ensemble → (mean, std).

        Returns
        -------
        mean : (x_dim,)
        std  : (x_dim,)
        """
        samples = self.sample(
            n_samples=n_samples, x_dim=x_dim, n_steps=n_steps,
            guidance=guidance, **kwargs,
        )
        return samples.mean(0), samples.std(0)

    # ------------------------------------------------------------------
    # Convenience: build DSMLoss

    def make_loss(self) -> DSMLoss:
        """Return a DSMLoss bound to this model's denoiser and scheduler."""
        return DSMLoss(self.denoiser, self.scheduler)

    # ------------------------------------------------------------------
    # Factory-compatible predict (mean prediction, n_steps=20 fast)

    def predict(self, *inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        xd = inputs[0].shape[-1] if inputs else self.cfg.x_dim
        return self.sample(n_samples=inputs[0].shape[0] if inputs else 1,
                           x_dim=xd, n_steps=20)
