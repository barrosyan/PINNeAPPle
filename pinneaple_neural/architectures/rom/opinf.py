from __future__ import annotations
"""Operator inference for data-driven reduced-order modeling."""
from typing import Dict, Optional, Literal, Tuple

import torch
import torch.nn as nn

from .base import ROMBase, ROMOutput


class OperatorInference(ROMBase):
    """
    Operator Inference (OpInf) for latent dynamics.

    Discrete-time (map):
        a_{t+1} = A a_t + H(a_t ⊗ a_t) + b

    Continuous-time (ODE):
        da/dt = A a + H(a ⊗ a) + b

    You provide latent trajectories a (e.g. POD encoder output) and fit by ridge regression.

    Shapes:
      a: (B,T,r)
      A: (r,r)
      H: (r, r*r)   (acts on vec(a⊗a))
      b: (r,)
      W: (F,r) where F = r + r*r (+1)
    """

    def __init__(
        self,
        r: int,
        *,
        use_quadratic: bool = True,
        use_bias: bool = True,
        # block-wise ridge
        l2_linear: float = 1e-6,
        l2_quad: float = 1e-6,
        l2_bias: float = 0.0,
        # preprocessing
        center: bool = True,
        scale: bool = False,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.r = int(r)
        self.use_quadratic = bool(use_quadratic)
        self.use_bias = bool(use_bias)

        self.l2_linear = float(l2_linear)
        self.l2_quad = float(l2_quad)
        self.l2_bias = float(l2_bias)

        self.center = bool(center)
        self.scale = bool(scale)
        self.eps = float(eps)

        self._F = self.r + (self.r * self.r if self.use_quadratic else 0) + (1 if self.use_bias else 0)

        # Train-free buffers (saved in state_dict)
        self.register_buffer("W", torch.zeros(self._F, self.r))  # (F, r)
        self.register_buffer("a_mean", torch.zeros(self.r))
        self.register_buffer("a_std", torch.ones(self.r))

        self._fitted = False

    # ----------------------------
    # Helpers: normalization
    # ----------------------------
    def _compute_norm_stats(self, a0: torch.Tensor) -> None:
        # a0: (N, r)
        if self.center:
            mean = a0.mean(dim=0)
        else:
            mean = torch.zeros(self.r, device=a0.device, dtype=a0.dtype)

        if self.scale:
            std = a0.std(dim=0, unbiased=False).clamp_min(self.eps)
        else:
            std = torch.ones(self.r, device=a0.device, dtype=a0.dtype)

        self.a_mean.copy_(mean)
        self.a_std.copy_(std)

    def _normalize(self, a: torch.Tensor) -> torch.Tensor:
        return (a - self.a_mean) / self.a_std

    def _denormalize(self, a: torch.Tensor) -> torch.Tensor:
        return a * self.a_std + self.a_mean

    # ----------------------------
    # Feature map
    # ----------------------------
    def _features(self, a: torch.Tensor) -> torch.Tensor:
        # a: (N, r) assumed normalized
        feats = [a]
        if self.use_quadratic:
            feats.append(torch.einsum("ni,nj->nij", a, a).reshape(a.shape[0], -1))  # (N, r*r)
        if self.use_bias:
            feats.append(torch.ones((a.shape[0], 1), device=a.device, dtype=a.dtype))
        return torch.cat(feats, dim=-1)  # (N, F)

    def _lambda_diag(self, device, dtype) -> torch.Tensor:
        parts = [torch.full((self.r,), self.l2_linear, device=device, dtype=dtype)]
        if self.use_quadratic:
            parts.append(torch.full((self.r * self.r,), self.l2_quad, device=device, dtype=dtype))
        if self.use_bias:
            parts.append(torch.full((1,), self.l2_bias, device=device, dtype=dtype))
        return torch.cat(parts, dim=0)  # (F,)

    @staticmethod
    def _solve_ridge(X: torch.Tensor, Y: torch.Tensor, lam_diag: torch.Tensor) -> torch.Tensor:
        """
        Solve ridge: (X^T X + diag(lam)) W = X^T Y
        Prefer Cholesky; fallback to solve.
        """
        K = X.transpose(0, 1) @ X
        K = K + torch.diag(lam_diag.to(device=X.device, dtype=X.dtype))
        RHS = X.transpose(0, 1) @ Y

        try:
            L = torch.linalg.cholesky(K)
            return torch.cholesky_solve(RHS, L)
        except RuntimeError:
            return torch.linalg.solve(K, RHS)

    # ----------------------------
    # Parameter accessors: A, H, b
    # ----------------------------
    def _split_W(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
          A_T: (r, r)  (because W is (F,r), first r rows correspond to A^T)
          H_T: (r*r, r) or None  (rows correspond to H^T blocks)
          b_T: (1, r) or None
        """
        idx = 0
        A_T = self.W[idx : idx + self.r, :]  # (r, r)
        idx += self.r

        H_T = None
        if self.use_quadratic:
            H_T = self.W[idx : idx + self.r * self.r, :]  # (r*r, r)
            idx += self.r * self.r

        b_T = None
        if self.use_bias:
            b_T = self.W[idx : idx + 1, :]  # (1, r)
            idx += 1

        return A_T, H_T, b_T

    @property
    def A(self) -> torch.Tensor:
        """Linear operator A with shape (r, r)."""
        A_T, _, _ = self._split_W()
        return A_T.transpose(0, 1).contiguous()

    @property
    def H(self) -> Optional[torch.Tensor]:
        """Quadratic operator H with shape (r, r*r), or None if disabled."""
        _, H_T, _ = self._split_W()
        if H_T is None:
            return None
        return H_T.transpose(0, 1).contiguous()

    @property
    def b(self) -> Optional[torch.Tensor]:
        """Bias b with shape (r,), or None if disabled."""
        _, _, b_T = self._split_W()
        if b_T is None:
            return None
        return b_T.view(-1).contiguous()

    # ----------------------------
    # Discrete-time fit/rollout
    # ----------------------------
    @torch.no_grad()
    def fit(self, a: torch.Tensor) -> "OperatorInference":
        """
        Fit discrete map using pairs (a_t -> a_{t+1}).
        a: (B, T, r)
        """
        if a.ndim != 3 or a.shape[-1] != self.r:
            raise ValueError(f"Expected a with shape (B,T,{self.r}), got {tuple(a.shape)}")

        B, T, r = a.shape
        a0 = a[:, :-1, :].reshape(-1, r)  # (N, r)
        a1 = a[:, 1:, :].reshape(-1, r)   # (N, r)

        self._compute_norm_stats(a0)

        a0n = self._normalize(a0)
        a1n = self._normalize(a1)

        X = self._features(a0n)  # (N, F)
        lam = self._lambda_diag(device=X.device, dtype=X.dtype)

        W = self._solve_ridge(X, a1n, lam)  # (F, r)
        self.W.copy_(W)
        self._fitted = True
        return self

    @torch.no_grad()
    def rollout(self, a0: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Discrete rollout using the learned 1-step map.
        a0: (B, r) -> (B, steps+1, r)
        """
        if not self._fitted:
            raise RuntimeError("OperatorInference is not fitted yet. Call .fit(...) or .fit_continuous(...).")
        if a0.ndim != 2 or a0.shape[-1] != self.r:
            raise ValueError(f"Expected a0 with shape (B,{self.r}), got {tuple(a0.shape)}")

        cur = self._normalize(a0)
        out = [cur]
        for _ in range(int(steps)):
            X = self._features(cur)
            cur = X @ self.W
            out.append(cur)

        y = torch.stack(out, dim=1)  # normalized
        return self._denormalize(y)

    # ----------------------------
    # Continuous-time fit/rollout
    # ----------------------------
    @staticmethod
    def _finite_diff(a: torch.Tensor, dt: float, method: Literal["forward", "central"] = "central") -> torch.Tensor:
        """
        a: (B, T, r) -> adot: (B, T, r) (same length; endpoints handled simply)
        """
        if dt <= 0:
            raise ValueError("dt must be > 0")
        if method == "forward":
            adot = (a[:, 1:, :] - a[:, :-1, :]) / dt
            # pad last with last forward diff
            last = adot[:, -1:, :]
            return torch.cat([adot, last], dim=1)
        elif method == "central":
            # interior central diff, endpoints forward/backward
            adot = torch.empty_like(a)
            adot[:, 1:-1, :] = (a[:, 2:, :] - a[:, :-2, :]) / (2.0 * dt)
            adot[:, 0:1, :] = (a[:, 1:2, :] - a[:, 0:1, :]) / dt
            adot[:, -1:, :] = (a[:, -1:, :] - a[:, -2:-1, :]) / dt
            return adot
        else:
            raise ValueError("method must be 'forward' or 'central'")

    @torch.no_grad()
    def fit_continuous(
        self,
        a: torch.Tensor,
        *,
        dt: float,
        diff: Literal["forward", "central"] = "central",
        drop_ends: bool = True,
    ) -> "OperatorInference":
        """
        Fit continuous-time ODE using regression on da/dt.

        We estimate adot from a using finite differences, then fit:
            adot = A a + H(a⊗a) + b

        Params:
          dt: sampling interval
          diff: finite difference scheme
          drop_ends: if True and diff=='central', drop first/last time points (cleaner targets)

        a: (B, T, r)
        """
        if a.ndim != 3 or a.shape[-1] != self.r:
            raise ValueError(f"Expected a with shape (B,T,{self.r}), got {tuple(a.shape)}")
        if dt <= 0:
            raise ValueError("dt must be > 0")

        adot = self._finite_diff(a, dt=float(dt), method=diff)  # (B,T,r)

        if diff == "central" and drop_ends and a.shape[1] >= 3:
            a_use = a[:, 1:-1, :]
            adot_use = adot[:, 1:-1, :]
        else:
            a_use = a
            adot_use = adot

        B, T, r = a_use.shape
        a0 = a_use.reshape(-1, r)        # (N, r)
        y = adot_use.reshape(-1, r)      # (N, r)

        self._compute_norm_stats(a0)
        a0n = self._normalize(a0)

        # IMPORTANT: adot should be in same normalized coordinates:
        # if a_norm = (a - mean)/std, then d(a_norm)/dt = (1/std) * da/dt
        yn = y / self.a_std  # broadcast (r,)

        X = self._features(a0n)
        lam = self._lambda_diag(device=X.device, dtype=X.dtype)

        W = self._solve_ridge(X, yn, lam)  # (F, r) mapping features -> d(a_norm)/dt
        self.W.copy_(W)
        self._fitted = True
        return self

    def _f_continuous_norm(self, a_norm: torch.Tensor) -> torch.Tensor:
        """
        Vector field in normalized coordinates:
          d(a_norm)/dt = phi(a_norm) @ W
        a_norm: (B, r)
        """
        X = self._features(a_norm)
        return X @ self.W  # (B, r)

    @torch.no_grad()
    def rollout_continuous(
        self,
        a0: torch.Tensor,
        *,
        dt: float,
        steps: int,
        integrator: Literal["euler", "rk4"] = "rk4",
    ) -> torch.Tensor:
        """
        Rollout continuous-time model with simple time integration.

        a0: (B, r)
        returns: (B, steps+1, r) in original (denormalized) space
        """
        if not self._fitted:
            raise RuntimeError("OperatorInference is not fitted yet. Call .fit_continuous(...) first.")
        if a0.ndim != 2 or a0.shape[-1] != self.r:
            raise ValueError(f"Expected a0 with shape (B,{self.r}), got {tuple(a0.shape)}")
        if dt <= 0:
            raise ValueError("dt must be > 0")

        dt = float(dt)
        cur = self._normalize(a0)  # normalized
        out = [cur]

        for _ in range(int(steps)):
            if integrator == "euler":
                k1 = self._f_continuous_norm(cur)
                cur = cur + dt * k1
            elif integrator == "rk4":
                k1 = self._f_continuous_norm(cur)
                k2 = self._f_continuous_norm(cur + 0.5 * dt * k1)
                k3 = self._f_continuous_norm(cur + 0.5 * dt * k2)
                k4 = self._f_continuous_norm(cur + dt * k3)
                cur = cur + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                raise ValueError("integrator must be 'euler' or 'rk4'")
            out.append(cur)

        y_norm = torch.stack(out, dim=1)
        return self._denormalize(y_norm)

    # ----------------------------
    # Forward
    # ----------------------------
    def forward(self, a: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        """
        Default forward: discrete-time rollout matching the given horizon.
        a: (B, T, r)
        """
        if a.ndim != 3 or a.shape[-1] != self.r:
            raise ValueError(f"Expected a with shape (B,T,{self.r}), got {tuple(a.shape)}")

        B, T, r = a.shape
        yhat = self.rollout(a[:, 0, :], steps=T - 1)

        losses: Dict[str, torch.Tensor] = {}
        if return_loss:
            losses["mse"] = self.mse(yhat, a)
            losses["total"] = losses["mse"]
        else:
            losses["total"] = torch.tensor(0.0, device=a.device, dtype=a.dtype)

        return ROMOutput(y=yhat, losses=losses, extras={"fitted": self._fitted})