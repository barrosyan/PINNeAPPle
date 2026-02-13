from __future__ import annotations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ContinuousModelBase, ContOutput
from .neural_ode import NeuralODE


# -----------------------------
# Encoder: GRU-ODE (minimal)
# -----------------------------
class _GRUODECell(nn.Module):
    """
    Minimal GRU-ODE-style encoder:
      - Between observations: evolve hidden state h(t) with ODE dh/dt = f(h, t)
      - At observation times: apply GRU update with x_t

    This is a simplified, literature-aligned "GRU-ODE" recognition model.
    """
    def __init__(self, hidden: int, ode_hidden: int = 128):
        super().__init__()
        self.hidden = int(hidden)

        # ODE dynamics for hidden state
        self.dyn = nn.Sequential(
            nn.Linear(hidden + 1, ode_hidden),
            nn.Tanh(),
            nn.Linear(ode_hidden, ode_hidden),
            nn.Tanh(),
            nn.Linear(ode_hidden, hidden),
        )

    def f(self, h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        h: (B,H)
        t: scalar tensor or (B,) -> we broadcast to (B,1)
        """
        if t.dim() == 0:
            t_in = t.expand(h.shape[0]).unsqueeze(-1)
        elif t.dim() == 1:
            t_in = t.unsqueeze(-1)
        else:
            # (B,1) already
            t_in = t
        return self.dyn(torch.cat([h, t_in], dim=-1))


class _GRUODEEncoder(nn.Module):
    """
    Recognition model that encodes an irregularly-sampled sequence (x, t)
    into a final hidden state, then outputs (mu, logvar) for z0.

    Inputs:
      x: (B,T,D)
      t: (T,) increasing (can be irregular)

    NOTE:
      - Handles irregular times via adaptive ODE steps between t[i-1] and t[i].
      - Uses a GRUCell update at each observation.
    """
    def __init__(self, obs_dim: int, hidden: int, latent_dim: int, method: str = "dopri5"):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        self.latent_dim = int(latent_dim)

        self.gru_cell = nn.GRUCell(self.obs_dim, self.hidden)
        self.h_ode = NeuralODE(state_dim=self.hidden, hidden=self.hidden, num_layers=2, method=method)

        self.mu = nn.Linear(self.hidden, self.latent_dim)
        self.logvar = nn.Linear(self.hidden, self.latent_dim)

        # optional stabilization
        self.logvar_min = -20.0
        self.logvar_max = 5.0

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        device = x.device
        dtype = x.dtype

        # init hidden state at time t[0]
        h = torch.zeros(B, self.hidden, device=device, dtype=dtype)

        # If T==1, just update once and return
        if T == 1:
            h = self.gru_cell(x[:, 0], h)
            mu = self.mu(h)
            logvar = self.logvar(h).clamp(self.logvar_min, self.logvar_max)
            return mu, logvar

        # Iterate observations; between each pair, evolve h with ODE
        # We treat the ODE integration as producing h(t_i) from h(t_{i-1})
        for i in range(T):
            if i > 0:
                # integrate from t[i-1] to t[i]
                t_span = torch.stack([t[i - 1], t[i]]).to(device=device, dtype=dtype)  # (2,)
                h = self.h_ode(h, t_span).y[:, -1, :]  # (B, H)
            # discrete update at observation i
            h = self.gru_cell(x[:, i], h)

        mu = self.mu(h)
        logvar = self.logvar(h).clamp(self.logvar_min, self.logvar_max)
        return mu, logvar


# -----------------------------
# Latent ODE with improvements
# -----------------------------
class LatentODE(ContinuousModelBase):
    """
    Latent ODE (improved, literature-aligned):
      - Encoder: GRU-ODE recognition model (handles irregular t)
      - Latent dynamics: NeuralODE with adaptive solver (default dopri5)
      - Decoder: Gaussian likelihood (mu_x, log_sigma_x) -> NLL
    """
    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 32,
        hidden: int = 128,
        *,
        ode_method: str = "dopri5",      # adaptive solver
        enc_method: str = "dopri5",
        min_log_sigma: float = -6.0,
        max_log_sigma: float = 2.0,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.latent_dim = int(latent_dim)
        self.hidden = int(hidden)

        # Encoder incorporates time/irregularity
        self.encoder_rnn = _GRUODEEncoder(obs_dim=self.obs_dim, hidden=self.hidden, latent_dim=self.latent_dim, method=enc_method)

        # Latent ODE dynamics (adaptive by default)
        self.ode = NeuralODE(state_dim=self.latent_dim, hidden=self.hidden, num_layers=2, method=ode_method)

        # Decoder outputs Gaussian parameters per time step
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, 2 * self.obs_dim),  # -> [mu_x, log_sigma_x]
        )
        self.min_log_sigma = float(min_log_sigma)
        self.max_log_sigma = float(max_log_sigma)

    def _reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def _gaussian_nll(self, mu_x: torch.Tensor, log_sigma_x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Elementwise Gaussian NLL, averaged over batch/time/dim.
        NLL = 0.5*((y-mu)^2 / sigma^2 + 2*log(sigma) + log(2*pi))
        """
        log_sigma_x = log_sigma_x.clamp(self.min_log_sigma, self.max_log_sigma)
        inv_var = torch.exp(-2.0 * log_sigma_x)
        nll = 0.5 * ((y - mu_x) ** 2) * inv_var + log_sigma_x + 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=y.device, dtype=y.dtype))
        return nll.mean()

    def forward(
        self,
        x: torch.Tensor,  # (B,T,obs_dim) observed sequence (can be irregularly sampled)
        t: torch.Tensor,  # (T,) timestamps (monotonic increasing; irregular ok)
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,T,obs_dim)
        beta_kl: float = 1.0,
        return_loss: bool = False,
    ) -> ContOutput:
        B, T, D = x.shape
        assert D == self.obs_dim, f"Expected obs_dim={self.obs_dim}, got {D}"
        assert t.dim() == 1 and t.numel() == T, f"t must be (T,), got {tuple(t.shape)}"

        # Encode with time-aware recognition model
        mu, logvar = self.encoder_rnn(x, t)
        z0 = self._reparam(mu, logvar)

        # Integrate latent path at the provided timestamps
        z_path = self.ode(z0, t).y  # (B,T,latent_dim)

        # Decode to Gaussian parameters
        dec = self.decoder(z_path)  # (B,T,2*obs_dim)
        mu_x, log_sigma_x = dec[..., : self.obs_dim], dec[..., self.obs_dim :]

        # For API compatibility: y is the mean prediction
        y_hat = mu_x

        losses: Dict[str, torch.Tensor] = {}
        if return_loss and y_true is not None:
            # Reconstruction as Gaussian NLL (more paper-like than MSE)
            rec = self._gaussian_nll(mu_x, log_sigma_x, y_true)

            # KL(q(z0|x) || p(z0)), with p=N(0,I)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())

            total = rec + float(beta_kl) * kl
            losses = {"rec": rec, "kl": kl, "total": total}
        else:
            # still provide a scalar tensor for consistency
            losses = {"total": x.new_zeros(())}

        return ContOutput(
            y=y_hat,
            losses=losses,
            extras={
                "mu": mu,
                "logvar": logvar,
                "mu_x": mu_x,
                "log_sigma_x": log_sigma_x.clamp(self.min_log_sigma, self.max_log_sigma),
            },
        )

    @torch.no_grad()
    def forecast(self, x: torch.Tensor, t_obs: torch.Tensor, t_new: torch.Tensor):
        """
        Encode using observed (x, t_obs) then decode on t_new.
        x:     (B,T_obs,obs_dim)
        t_obs: (T_obs,)
        t_new: (T_new,)
        returns: ContOutput with y shape (B,T_new,obs_dim)
        """
        B, T_obs, D = x.shape
        assert t_obs.dim() == 1 and t_obs.numel() == T_obs, f"t_obs must be (T_obs,), got {tuple(t_obs.shape)}"
        assert t_new.dim() == 1, f"t_new must be (T_new,), got {tuple(t_new.shape)}"

        mu, logvar = self.encoder_rnn(x, t_obs)
        z0 = self._reparam(mu, logvar)

        z_path = self.ode(z0, t_new).y  # (B,T_new,latent_dim)

        dec = self.decoder(z_path)      # (B,T_new,2*obs_dim)
        mu_x, log_sigma_x = dec[..., : self.obs_dim], dec[..., self.obs_dim :]

        return ContOutput(
            y=mu_x,
            losses={"total": x.new_zeros(())},
            extras={
                "mu": mu,
                "logvar": logvar,
                "mu_x": mu_x,
                "log_sigma_x": log_sigma_x.clamp(self.min_log_sigma, self.max_log_sigma),
            },
        )
