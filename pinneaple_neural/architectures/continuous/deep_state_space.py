from __future__ import annotations
"""Deep state-space model for probabilistic dynamics."""

from dataclasses import dataclass
from typing import Dict, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContOutput:
    y: torch.Tensor
    losses: Dict[str, torch.Tensor]
    extras: Dict[str, Any]


def _mlp(in_dim: int, hidden: int, out_dim: int, depth: int = 2) -> nn.Module:
    layers = []
    d = in_dim
    for _ in range(max(1, depth)):
        layers += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def gaussian_nll_diag(mu: torch.Tensor, logvar: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # mu, logvar, y: (..., D)
    # returns scalar mean NLL
    return 0.5 * (logvar + (y - mu) ** 2 / torch.exp(logvar) + torch.log(torch.tensor(2.0 * torch.pi, device=y.device))).sum(dim=-1).mean()


def kl_diag_gaussians(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    # KL( N(mu_q, var_q) || N(mu_p, var_p) ), diagonal
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-12) - 1.0)
    return kl.sum(dim=-1).mean()


class DeepStateSpaceModel(nn.Module):
    """
    Variational Deep State Space Model (DSSM / deep SSM) - MVP (filtering):

    Prior/transition:    p(z_t | z_{t-1}, x_t) = N(mu_p, diag(exp(logvar_p)))
    Inference posterior: q(z_t | z_{t-1}, x_t, y_t) = N(mu_q, diag(exp(logvar_q)))
    Emission:            p(y_t | z_t) = N(mu_y, diag(exp(logvar_y)))

    Inputs:
      x:      (B, T, input_dim)
      y_true: (B, T, out_dim)  (needed for training / ELBO)

    Output:
      y_hat = mu_y: (B, T, out_dim)
    """

    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        mlp_depth: int = 2,
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
        beta_kl: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.out_dim = int(out_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.mlp_depth = int(mlp_depth)

        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.beta_kl = float(beta_kl)

        # Prior p(z_t | z_{t-1}, x_t)
        self.prior_net = _mlp(in_dim=self.latent_dim + self.input_dim, hidden=self.hidden_dim,
                              out_dim=2 * self.latent_dim, depth=self.mlp_depth)

        # Posterior q(z_t | z_{t-1}, x_t, y_t)
        self.post_net = _mlp(in_dim=self.latent_dim + self.input_dim + self.out_dim, hidden=self.hidden_dim,
                             out_dim=2 * self.latent_dim, depth=self.mlp_depth)

        # Emission p(y_t | z_t)
        self.emit_net = _mlp(in_dim=self.latent_dim, hidden=self.hidden_dim,
                             out_dim=2 * self.out_dim, depth=self.mlp_depth)

        # Optional: learnable initial z0 prior and posterior anchors (simple)
        self.z0_mu = nn.Parameter(torch.zeros(self.latent_dim))
        self.z0_logvar = nn.Parameter(torch.zeros(self.latent_dim))

    def _clamp_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        return torch.clamp(logvar, self.min_logvar, self.max_logvar)

    def _reparam(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def forward(
        self,
        x: torch.Tensor,                     # (B,T,input_dim)
        *,
        y_true: Optional[torch.Tensor] = None,  # (B,T,out_dim)
        return_loss: bool = False,
        sample: bool = False,                # if True, sample y using predicted variance (not only mean)
    ) -> ContOutput:
        if x.dim() != 3:
            raise ValueError("x must have shape (B,T,input_dim)")
        B, T, D = x.shape
        if D != self.input_dim:
            raise ValueError(f"Expected x dim {self.input_dim}, got {D}")

        if return_loss and y_true is None:
            raise ValueError("y_true is required when return_loss=True")

        # init z_{0}
        z_prev_mu = self.z0_mu.unsqueeze(0).expand(B, -1)                 # (B,latent)
        z_prev_logvar = self._clamp_logvar(self.z0_logvar.unsqueeze(0).expand(B, -1))
        z_prev = self._reparam(z_prev_mu, z_prev_logvar)                  # sample initial

        y_mus = []
        y_logvars = []
        z_mus_q = []
        z_logvars_q = []
        z_mus_p = []
        z_logvars_p = []
        z_samples = []

        for t in range(T):
            x_t = x[:, t, :]  # (B,input_dim)

            # prior params from (z_{t-1}, x_t)
            prior_in = torch.cat([z_prev, x_t], dim=-1)
            prior_out = self.prior_net(prior_in)
            mu_p, logvar_p = prior_out.chunk(2, dim=-1)
            logvar_p = self._clamp_logvar(logvar_p)

            if y_true is not None:
                y_t = y_true[:, t, :]  # (B,out_dim)
                post_in = torch.cat([z_prev, x_t, y_t], dim=-1)
                post_out = self.post_net(post_in)
                mu_q, logvar_q = post_out.chunk(2, dim=-1)
                logvar_q = self._clamp_logvar(logvar_q)

                # sample z_t from posterior
                z_t = self._reparam(mu_q, logvar_q)

                z_mus_q.append(mu_q)
                z_logvars_q.append(logvar_q)
            else:
                # inference-free mode: sample from prior
                mu_q, logvar_q = None, None
                z_t = self._reparam(mu_p, logvar_p)

            z_mus_p.append(mu_p)
            z_logvars_p.append(logvar_p)
            z_samples.append(z_t)

            # emission params from z_t
            emit_out = self.emit_net(z_t)
            mu_y, logvar_y = emit_out.chunk(2, dim=-1)
            logvar_y = self._clamp_logvar(logvar_y)

            y_mus.append(mu_y)
            y_logvars.append(logvar_y)

            # roll
            z_prev = z_t

        mu_y = torch.stack(y_mus, dim=1)         # (B,T,out_dim)
        logvar_y = torch.stack(y_logvars, dim=1) # (B,T,out_dim)
        z_s = torch.stack(z_samples, dim=1)      # (B,T,latent)

        # Optionally sample y
        if sample:
            y_hat = mu_y + torch.exp(0.5 * logvar_y) * torch.randn_like(mu_y)
        else:
            y_hat = mu_y

        losses: Dict[str, torch.Tensor] = {"total": x.new_zeros(())}

        if return_loss and (y_true is not None):
            # reconstruction term: sum/mean across time
            nll = gaussian_nll_diag(mu_y.reshape(-1, self.out_dim),
                                    logvar_y.reshape(-1, self.out_dim),
                                    y_true.reshape(-1, self.out_dim))

            # KL term: q(z_t|...) || p(z_t|...)
            mu_p = torch.stack(z_mus_p, dim=1)          # (B,T,latent)
            logvar_p = torch.stack(z_logvars_p, dim=1)
            mu_q = torch.stack(z_mus_q, dim=1)          # (B,T,latent)
            logvar_q = torch.stack(z_logvars_q, dim=1)

            kl = kl_diag_gaussians(mu_q.reshape(-1, self.latent_dim),
                                   logvar_q.reshape(-1, self.latent_dim),
                                   mu_p.reshape(-1, self.latent_dim),
                                   logvar_p.reshape(-1, self.latent_dim))

            total = nll + self.beta_kl * kl
            losses = {"total": total, "nll": nll, "kl": kl}

        extras = {
            "logvar_y": logvar_y,
            "z": z_s,
            "z0_mu": self.z0_mu,
            "z0_logvar": self.z0_logvar,
        }
        return ContOutput(y=y_hat, losses=losses, extras=extras)
