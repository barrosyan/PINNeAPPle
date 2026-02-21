from __future__ import annotations
"""Neural stochastic differential equation models."""

from typing import Dict, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ContinuousModelBase, ContOutput


class NeuralSDE(ContinuousModelBase):
    """
    Neural SDE (Euler–Maruyama) for *solving* problems via simulation:

        dY = f(t, y) dt + G(t, y) dW

    Improvements vs MVP:
      - diag diffusion: sigma = softplus(raw) + eps (more stable than exp(clamp))
      - full diffusion: parameterize as lower-triangular Cholesky factor L (PSD covariance)
      - optional multi-path Monte Carlo and antithetic sampling for variance reduction
      - monotonicity check for t without dt.item() inside the loop

    Inputs:
      y0: (B, D)
      t:  (T,) strictly increasing
    Outputs:
      y_path: (B, T, D) if n_paths=1
              (B, K, T, D) if n_paths=K>1 and keep_paths=True
              (B, T, D) mean over paths if n_paths>1 and keep_paths=False
    """

    def __init__(
        self,
        state_dim: int,
        *,
        hidden: int = 128,
        num_layers: int = 3,
        diffusion: Literal["diag", "full"] = "diag",
        activation: str = "tanh",
        # diag diffusion stabilization
        sigma_eps: float = 1e-6,
        sigma_max: Optional[float] = None,  # optionally cap sigma
        # full diffusion stabilization (Cholesky factor)
        chol_diag_eps: float = 1e-6,
        # Monte Carlo simulation control
        n_paths: int = 1,
        antithetic: bool = False,
        keep_paths: bool = False,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.diffusion = diffusion

        self.sigma_eps = float(sigma_eps)
        self.sigma_max = float(sigma_max) if sigma_max is not None else None
        self.chol_diag_eps = float(chol_diag_eps)

        self.n_paths = int(n_paths)
        self.antithetic = bool(antithetic)
        self.keep_paths = bool(keep_paths)

        act = (activation or "tanh").lower()
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}.get(act, nn.Tanh)

        # drift f(t,y)
        f_layers = [nn.Linear(self.state_dim + 1, hidden), act_fn()]
        for _ in range(max(0, num_layers - 1)):
            f_layers += [nn.Linear(hidden, hidden), act_fn()]
        f_layers += [nn.Linear(hidden, self.state_dim)]
        self.f = nn.Sequential(*f_layers)

        # diffusion parameter head
        if diffusion == "diag":
            out = self.state_dim
        else:
            # output lower-triangular entries of L (including diagonal): D*(D+1)/2
            out = self.state_dim * (self.state_dim + 1) // 2

        g_layers = [nn.Linear(self.state_dim + 1, hidden), act_fn()]
        for _ in range(max(0, num_layers - 1)):
            g_layers += [nn.Linear(hidden, hidden), act_fn()]
        g_layers += [nn.Linear(hidden, out)]
        self.g = nn.Sequential(*g_layers)

        # optional: encourage small initial diffusion (helps stability early)
        # (safe default: slight negative bias for diag; for full, diagonal params slightly negative)
        with torch.no_grad():
            last = self.g[-1]
            if isinstance(last, nn.Linear):
                if diffusion == "diag":
                    last.bias.fill_(-2.0)  # softplus(-2) ~ 0.127
                else:
                    last.bias.fill_(0.0)

    # -------------------------
    # Helpers
    # -------------------------
    def _cat(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # y: (..., D), t can be scalar or (...,)
        if t.ndim == 0:
            tt = t.view(1).expand(y.shape[0])
        else:
            tt = t
        tt = tt.to(device=y.device, dtype=y.dtype)
        if tt.ndim == 1:
            tt = tt[..., None]
        return torch.cat([tt, y], dim=-1)

    def drift(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.f(self._cat(t, y))

    def _sigma_diag(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        raw = self.g(self._cat(t, y))
        sigma = F.softplus(raw) + self.sigma_eps
        if self.sigma_max is not None:
            sigma = torch.clamp(sigma, max=self.sigma_max)
        return sigma  # (..., D)

    def _chol_full(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Build lower-triangular Cholesky factor L: (..., D, D),
        with positive diagonal via softplus + eps.
        """
        D = self.state_dim
        raw = self.g(self._cat(t, y))  # (..., D*(D+1)/2)
        # fill lower-triangular
        L = torch.zeros(*raw.shape[:-1], D, D, device=y.device, dtype=y.dtype)
        tril_i, tril_j = torch.tril_indices(D, D, offset=0, device=y.device)
        L[..., tril_i, tril_j] = raw

        # enforce positive diagonal
        diag = torch.diagonal(L, dim1=-2, dim2=-1)
        diag_pos = F.softplus(diag) + self.chol_diag_eps
        # write back diagonal
        L = L.clone()
        idx = torch.arange(D, device=y.device)
        L[..., idx, idx] = diag_pos
        return L

    def _effective_paths(self) -> int:
        if self.n_paths <= 1:
            return 1
        # if antithetic, we will generate half and mirror, requiring even K
        if self.antithetic and (self.n_paths % 2 != 0):
            raise ValueError("antithetic=True requires n_paths to be even.")
        return self.n_paths

    # -------------------------
    # Forward / solve
    # -------------------------
    def forward(
        self,
        y0: torch.Tensor,                      # (B, D)
        t: torch.Tensor,                       # (T,)
        *,
        y_true: Optional[torch.Tensor] = None,  # (B, T, D) expected (mean path supervision)
        return_loss: bool = False,
    ) -> ContOutput:
        if y0.ndim != 2:
            raise ValueError(f"Expected y0 shape (B,D), got {tuple(y0.shape)}")
        B, D = y0.shape
        if D != self.state_dim:
            raise ValueError(f"Expected y0 dim {self.state_dim}, got {D}")

        if t.ndim != 1:
            raise ValueError(f"Expected t shape (T,), got {tuple(t.shape)}")
        if t.numel() < 2:
            raise ValueError("t must have at least 2 points for integration.")
        # monotonicity check once
        if not torch.all(t[1:] > t[:-1]):
            raise ValueError("t must be strictly increasing for SDE integration.")

        T = int(t.numel())
        K = self._effective_paths()

        # Expand y0 across paths: (B, K, D)
        if K == 1:
            y = y0
        else:
            y = y0[:, None, :].expand(B, K, D).contiguous()

        ys = []
        ys.append(y)

        # precompute dt and sqrt(dt) in correct dtype/device
        t_ = t.to(device=y0.device, dtype=y0.dtype)
        dt = (t_[1:] - t_[:-1])  # (T-1,)
        sqrt_dt = torch.sqrt(dt)

        for i in range(T - 1):
            ti = t_[i]
            dti = dt[i]
            sdt = sqrt_dt[i]

            if K == 1:
                f = self.drift(ti, y)  # (B, D)
                if self.diffusion == "diag":
                    sigma = self._sigma_diag(ti, y)  # (B, D)
                    dW = torch.randn_like(y) * sdt
                    y = y + f * dti + sigma * dW
                else:
                    L = self._chol_full(ti, y)  # (B, D, D)
                    dW = torch.randn((B, D), device=y.device, dtype=y.dtype) * sdt
                    noise = torch.einsum("bij,bj->bi", L, dW)
                    y = y + f * dti + noise
            else:
                # vectorize across K paths: treat (B*K, D)
                y_flat = y.reshape(B * K, D)
                f = self.drift(ti, y_flat).reshape(B, K, D)

                if self.diffusion == "diag":
                    sigma = self._sigma_diag(ti, y_flat).reshape(B, K, D)
                    # antithetic sampling: generate half and mirror
                    if self.antithetic:
                        half = K // 2
                        dW_half = torch.randn((B, half, D), device=y.device, dtype=y.dtype) * sdt
                        dW = torch.cat([dW_half, -dW_half], dim=1)
                    else:
                        dW = torch.randn((B, K, D), device=y.device, dtype=y.dtype) * sdt

                    y = y + f * dti + sigma * dW
                else:
                    L = self._chol_full(ti, y_flat).reshape(B, K, D, D)
                    if self.antithetic:
                        half = K // 2
                        dW_half = torch.randn((B, half, D), device=y.device, dtype=y.dtype) * sdt
                        dW = torch.cat([dW_half, -dW_half], dim=1)
                    else:
                        dW = torch.randn((B, K, D), device=y.device, dtype=y.dtype) * sdt

                    noise = torch.einsum("bkij,bkj->bki", L, dW)
                    y = y + f * dti + noise

            ys.append(y)

        y_path = torch.stack(ys, dim=-2)  # K==1: (B, T, D) ; K>1: (B, K, T, D)

        # Decide output aggregation
        if K == 1:
            y_out = y_path  # (B, T, D)
            extras = {"t": t_}
        else:
            if self.keep_paths:
                y_out = y_path  # (B, K, T, D)
            else:
                y_out = y_path.mean(dim=1)  # mean over paths -> (B, T, D)
            extras = {"t": t_, "n_paths": K, "antithetic": self.antithetic, "keep_paths": self.keep_paths}

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=y0.device, dtype=y0.dtype)}
        if return_loss and y_true is not None:
            # supervise either:
            # - if keep_paths=False: compare mean path (B,T,D) with y_true
            # - if keep_paths=True: compare mean over paths with y_true (common choice)
            if K == 1:
                pred = y_out  # (B,T,D)
            else:
                pred = y_path.mean(dim=1)  # (B,T,D)
            losses["mse"] = self.mse(pred, y_true)
            losses["total"] = losses["mse"]

        return ContOutput(y=y_out, losses=losses, extras=extras)
