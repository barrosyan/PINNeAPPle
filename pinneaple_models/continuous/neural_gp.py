from __future__ import annotations
"""Neural Gaussian process for Bayesian continuous dynamics."""

from typing import Dict, Optional, Literal, Tuple

import torch
import torch.nn as nn

from .base import ContinuousModelBase, ContOutput


class _RandomFourierFeatures(nn.Module):
    """
    RBF kernel approximation with random Fourier features:
      k(x,x') ~ phi(x)^T phi(x')

    phi(x) = sqrt(2/m) * cos(Wx + b), W ~ N(0, 1/l^2), b ~ U(0,2pi)
    """
    def __init__(self, in_dim: int, num_features: int = 512, lengthscale: float = 1.0, freeze: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.m = int(num_features)
        self.lengthscale = float(lengthscale)

        W = torch.randn(self.m, self.in_dim) * (1.0 / max(self.lengthscale, 1e-12))
        b = torch.rand(self.m) * (2.0 * torch.pi)

        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)

        if freeze:
            self.W.requires_grad_(False)
            self.b.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.W.t() + self.b
        return (2.0 / self.m) ** 0.5 * torch.cos(proj)


class NeuralGaussianProcess(ContinuousModelBase):
    """
    Neural Gaussian Process (RFF + Bayesian Linear head):

    - Feature extractor (MLP) -> embedding
    - Random Fourier Features (RBF approx) on embedding
    - Bayesian Linear Regression head => GP-consistent predictive mean/variance
      (for the GP induced by the RFF kernel approximation)

    Conditioning:
      call .condition(x_train, y_train) to compute posterior over head weights.

    Output:
      mu: (..., out_dim)
      extras["logvar"]: (..., out_dim)  (posterior predictive variance)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        embed_dim: int = 64,
        mlp_hidden: int = 128,
        mlp_layers: int = 2,
        rff_features: int = 512,
        rff_lengthscale: float = 1.0,
        freeze_rff: bool = True,
        noise_mode: Literal["learned", "fixed"] = "learned",
        fixed_noise: float = 1e-3,
        min_logvar: float = -10.0,
        max_logvar: float = 2.0,
        # GP/BLR prior precision (alpha). Bigger => stronger regularization.
        alpha: float = 1.0,
        # numerical jitter for Cholesky
        jitter: float = 1e-6,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)
        self.noise_mode = noise_mode
        self.fixed_noise = float(fixed_noise)
        self.alpha = float(alpha)
        self.jitter = float(jitter)

        # MLP feature extractor
        layers = [nn.Linear(self.in_dim, mlp_hidden), nn.GELU()]
        for _ in range(mlp_layers - 1):
            layers += [nn.Linear(mlp_hidden, mlp_hidden), nn.GELU()]
        layers += [nn.Linear(mlp_hidden, embed_dim)]
        self.phi_nn = nn.Sequential(*layers)

        self.rff = _RandomFourierFeatures(
            embed_dim,
            num_features=rff_features,
            lengthscale=rff_lengthscale,
            freeze=freeze_rff,
        )

        # Keep a deterministic head for init / fallback (optional),
        # but the GP-consistent path uses posterior buffers below.
        self.head = nn.Linear(rff_features, out_dim, bias=True)

        if noise_mode == "learned":
            # noise variance sigma^2 = exp(log_noise)
            self.log_noise = nn.Parameter(torch.tensor(-6.0))
        else:
            self.register_buffer(
                "log_noise_buf",
                torch.log(torch.tensor(self.fixed_noise)),
            )

        # Posterior state (computed by condition):
        # A_chol: (m,m) lower-triangular Cholesky of A
        # M:      (m,out_dim) posterior mean of weights
        self.register_buffer("_A_chol", torch.empty(0))
        self.register_buffer("_M", torch.empty(0))
        self._posterior_ready: bool = False

    def _flatten(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        return x2, shape[:-1]

    def _noise_var(self, ref: torch.Tensor) -> torch.Tensor:
        # returns sigma^2 as scalar tensor on ref's device/dtype
        if self.noise_mode == "learned":
            return torch.exp(self.log_noise.to(device=ref.device, dtype=ref.dtype))
        return torch.exp(self.log_noise_buf.to(device=ref.device, dtype=ref.dtype))

    @torch.no_grad()
    def reset_posterior(self) -> None:
        self._A_chol = torch.empty(0, device=self._A_chol.device)
        self._M = torch.empty(0, device=self._M.device)
        self._posterior_ready = False

    @torch.no_grad()
    def condition(
        self,
        x_train: torch.Tensor,          # (..., in_dim)
        y_train: torch.Tensor,          # (..., out_dim)
        *,
        alpha: Optional[float] = None,
        noise_var: Optional[float] = None,
    ) -> None:
        """
        Computes posterior for Bayesian linear head given training data.
        This is the GP-consistent step (for the GP induced by RFF features).

        Posterior:
          A = alpha I + (1/sigma^2) Phi^T Phi
          M = (1/sigma^2) A^{-1} Phi^T Y
        """
        x2, _ = self._flatten(x_train)
        y2 = y_train.reshape(-1, self.out_dim)

        # compute features
        emb = self.phi_nn(x2)
        Phi = self.rff(emb)  # (N, m)
        N, m = Phi.shape

        a = float(self.alpha if alpha is None else alpha)
        ref = Phi
        sigma2 = self._noise_var(ref) if noise_var is None else torch.tensor(noise_var, device=ref.device, dtype=ref.dtype)

        # A = alpha I + (1/sigma^2) Phi^T Phi
        I = torch.eye(m, device=ref.device, dtype=ref.dtype)
        A = a * I + (Phi.t() @ Phi) / sigma2

        # jitter for stability
        A = A + self.jitter * I

        A_chol = torch.linalg.cholesky(A)  # (m,m)
        # Compute RHS: (1/sigma^2) Phi^T Y
        RHS = (Phi.t() @ y2) / sigma2  # (m,out_dim)

        # Solve for M: A M = RHS  -> M = A^{-1} RHS via Cholesky
        M = torch.cholesky_solve(RHS, A_chol)  # (m,out_dim)

        self._A_chol = A_chol
        self._M = M
        self._posterior_ready = True

    def _predict_posterior(self, Phi: torch.Tensor, sigma2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Phi: (N, m)
        returns:
          mu: (N, out_dim)
          var: (N, 1)  (same for all outputs under independent heads with same prior/noise)
        """
        # mean
        mu = Phi @ self._M  # (N,out_dim)

        # var = sigma^2 + diag(Phi A^{-1} Phi^T)
        # compute v = A^{-1} Phi^T via cholesky_solve:
        # v = solve(A, Phi^T) => (m,N)
        v = torch.cholesky_solve(Phi.t(), self._A_chol)  # (m,N)
        quad = torch.sum(Phi.t() * v, dim=0, keepdim=True).t()  # (N,1)
        var = sigma2 + quad
        return mu, var

    def forward(
        self,
        x: torch.Tensor,  # (..., in_dim)
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        use_nll: bool = True,
        # if posterior not set, optionally condition on (x,y_true) inside forward
        condition_on_batch: bool = False,
    ) -> ContOutput:
        x2, prefix = self._flatten(x)

        emb = self.phi_nn(x2)
        Phi = self.rff(emb)  # (N, m)
        sigma2 = self._noise_var(Phi)

        # Ensure GP-consistent posterior is available
        if (not self._posterior_ready) and return_loss and (y_true is not None) and condition_on_batch:
            # condition on current batch (simple training loop convenience)
            self.condition(x, y_true)

        if self._posterior_ready:
            mu2, var2 = self._predict_posterior(Phi, sigma2)
        else:
            # Fallback (not GP-consistent): deterministic head with fixed/learned noise
            mu2 = self.head(Phi)
            var2 = sigma2.expand(mu2.shape[0], 1)

        mu = mu2.reshape(*prefix, self.out_dim)
        var = var2.reshape(*prefix, 1).expand(*prefix, self.out_dim)

        # clamp logvar
        logvar = torch.log(var.clamp_min(1e-12))
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            if use_nll:
                # NLL under the posterior predictive Normal(mu, var)
                losses["nll"] = self.gaussian_nll(mu, logvar, y_true)
                losses["total"] = losses["nll"]
            else:
                losses["mse"] = self.mse(mu, y_true)
                losses["total"] = losses["mse"]

        return ContOutput(y=mu, losses=losses, extras={"logvar": logvar})
