from __future__ import annotations
"""
SINDy — Sparse Identification of Nonlinear Dynamics.

Identifies the governing equations of a dynamical system from data by
selecting a sparse set of terms from a nonlinear feature library:

    d a_i / dt  =  sum_j  xi_ij  * theta_j(a)

where theta is a library of candidate functions (polynomials, trig, etc.)
and xi is a sparse coefficient matrix found via STLSQ (sequential
thresholded least squares) or ridge regression.

Reference
---------
Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016).
Discovering governing equations from data by sparse identification of
nonlinear dynamical systems. PNAS.
"""
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn

from .base import ROMBase, ROMOutput


_LibraryKind = Literal["poly", "poly_trig", "poly_exp"]


class SINDy(ROMBase):
    """
    SINDy for latent dynamics.

    Inputs:  a (B, T, r)  — latent state trajectories
    Outputs: fitted coefficient matrix Xi (F, r), governing  da/dt = theta(a) Xi

    Feature library
    ---------------
    "poly"       : 1, a_i, a_i a_j  (degree-1 + degree-2 cross-terms)
    "poly_trig"  : poly + sin(a_i), cos(a_i)
    "poly_exp"   : poly + exp(a_i)

    Fitting
    -------
    Two solvers are available:
    - "ridge"  : closed-form ridge regression  (fast, not sparse)
    - "stlsq"  : sequential thresholded least squares  (induces sparsity)

    Parameters
    ----------
    r           : latent dimension
    library     : feature library type
    poly_degree : maximum polynomial degree (1 or 2)
    threshold   : sparsity threshold for STLSQ (coefficients below this
                  value are zeroed and refitted)
    n_iter      : number of STLSQ iterations
    l2          : ridge regularisation strength
    solver      : "ridge" or "stlsq"
    integrator  : ODE integrator for rollout: "euler" or "rk4"
    """

    def __init__(
        self,
        r: int,
        *,
        library: _LibraryKind = "poly",
        poly_degree: int = 2,
        threshold: float = 0.05,
        n_iter: int = 10,
        l2: float = 1e-4,
        solver: Literal["ridge", "stlsq"] = "stlsq",
        integrator: Literal["euler", "rk4"] = "rk4",
    ):
        super().__init__()
        self.r = int(r)
        self.library = library
        self.poly_degree = min(max(int(poly_degree), 1), 2)
        self.threshold = float(threshold)
        self.n_iter = int(n_iter)
        self.l2 = float(l2)
        self.solver = solver
        self.integrator = integrator

        # Derive feature count
        n_feat = self._feature_count()
        self._n_feat = n_feat
        self.feature_names: List[str] = self._build_feature_names()

        self.register_buffer("Xi",     torch.zeros(n_feat, r))
        self.register_buffer("a_mean", torch.zeros(r))
        self.register_buffer("a_std",  torch.ones(r))
        self._fitted = False

    # ------------------------------------------------------------------
    # Feature library
    # ------------------------------------------------------------------

    def _feature_count(self) -> int:
        r = self.r
        n = 1 + r                               # const + linear
        if self.poly_degree >= 2:
            n += r * (r + 1) // 2               # unique quadratic terms
        if self.library == "poly_trig":
            n += 2 * r                           # sin + cos
        elif self.library == "poly_exp":
            n += r                               # exp
        return n

    def _build_feature_names(self) -> List[str]:
        names = ["1"]
        names += [f"a{i}" for i in range(self.r)]
        if self.poly_degree >= 2:
            for i in range(self.r):
                for j in range(i, self.r):
                    names.append(f"a{i}*a{j}")
        if self.library == "poly_trig":
            names += [f"sin(a{i})" for i in range(self.r)]
            names += [f"cos(a{i})" for i in range(self.r)]
        elif self.library == "poly_exp":
            names += [f"exp(a{i})" for i in range(self.r)]
        return names

    def _theta(self, a: torch.Tensor) -> torch.Tensor:
        """
        Build feature matrix theta from state a of shape (N, r).
        Returns (N, n_feat).
        """
        N = a.shape[0]
        feats = [torch.ones(N, 1, device=a.device, dtype=a.dtype)]
        feats.append(a)                             # linear
        if self.poly_degree >= 2:
            for i in range(self.r):
                for j in range(i, self.r):
                    feats.append((a[:, i:i+1] * a[:, j:j+1]))
        if self.library == "poly_trig":
            feats.append(torch.sin(a))
            feats.append(torch.cos(a))
        elif self.library == "poly_exp":
            feats.append(torch.exp(a.clamp(-10, 10)))
        return torch.cat(feats, dim=1)

    # ------------------------------------------------------------------
    # Solvers
    # ------------------------------------------------------------------

    def _ridge(self, Theta: torch.Tensor, Adot: torch.Tensor) -> torch.Tensor:
        """Xi = (Theta^T Theta + l2 I)^{-1} Theta^T Adot  (shape: F, r)"""
        K = Theta.t() @ Theta
        K = K + self.l2 * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
        RHS = Theta.t() @ Adot
        try:
            L = torch.linalg.cholesky(K)
            return torch.cholesky_solve(RHS, L)
        except RuntimeError:
            return torch.linalg.solve(K, RHS)

    def _stlsq(self, Theta: torch.Tensor, Adot: torch.Tensor) -> torch.Tensor:
        """Sequential thresholded least squares.

        Each output state equation (column of Xi) is thresholded and refit
        independently, since different state variables generally have
        different active terms in their governing equation.
        """
        Xi = self._ridge(Theta, Adot)
        F, r = Xi.shape

        for k in range(r):
            xi_k = Xi[:, k:k+1]
            adot_k = Adot[:, k:k+1]
            active = torch.ones(F, dtype=torch.bool, device=Xi.device)

            for _ in range(self.n_iter):
                small = xi_k.abs().reshape(-1) < self.threshold
                xi_k = xi_k.clone()
                xi_k[small] = 0.0
                active = ~small & active
                if active.sum() == 0:
                    break
                # re-fit on active subset
                Th_sub = Theta[:, active]
                xi_sub = self._ridge(Th_sub, adot_k)
                xi_k = torch.zeros((F, 1), device=Xi.device, dtype=Xi.dtype)
                xi_k[active] = xi_sub

            Xi[:, k:k+1] = xi_k

        return Xi

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    @staticmethod
    def _fdiff(a: torch.Tensor, dt: float) -> torch.Tensor:
        """Central finite difference da/dt; shape (B, T, r) -> (B, T-2, r)."""
        adot = (a[:, 2:] - a[:, :-2]) / (2.0 * dt)
        a_mid = a[:, 1:-1]
        return a_mid, adot

    @torch.no_grad()
    def fit(
        self,
        a: torch.Tensor,
        *,
        dt: float = 1.0,
    ) -> "SINDy":
        """
        Fit SINDy coefficients Xi from latent trajectories.

        a  : (B, T, r)
        dt : time step between snapshots
        """
        if a.ndim != 3 or a.shape[-1] != self.r:
            raise ValueError(f"Expected a with shape (B, T, {self.r}), got {tuple(a.shape)}")

        a_mid, adot = self._fdiff(a, dt)
        N = a_mid.shape[0] * a_mid.shape[1]
        a_flat    = a_mid.reshape(N, self.r)
        adot_flat = adot.reshape(N, self.r)

        # Normalise
        self.a_mean.copy_(a_flat.mean(0))
        self.a_std.copy_(a_flat.std(0, unbiased=False).clamp_min(1e-8))
        a_norm = (a_flat - self.a_mean) / self.a_std

        Theta = self._theta(a_norm)                # (N, F)
        adot_norm = adot_flat / self.a_std         # (N, r)

        if self.solver == "stlsq":
            Xi = self._stlsq(Theta, adot_norm)
        else:
            Xi = self._ridge(Theta, adot_norm)

        self.Xi.copy_(Xi)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def _f(self, a_norm: torch.Tensor) -> torch.Tensor:
        """Vector field in normalised coordinates: da_norm/dt = theta @ Xi."""
        return self._theta(a_norm) @ self.Xi

    @torch.no_grad()
    def rollout(self, a0: torch.Tensor, *, dt: float, steps: int) -> torch.Tensor:
        """
        Integrate the SINDy model forward.

        a0    : (B, r) initial latent state (original space)
        dt    : integration time step
        steps : number of steps
        Returns (B, steps+1, r).
        """
        if not self._fitted:
            raise RuntimeError("SINDy not fitted. Call fit() first.")

        cur = (a0 - self.a_mean) / self.a_std
        out = [cur]
        for _ in range(steps):
            if self.integrator == "euler":
                cur = cur + dt * self._f(cur)
            else:  # rk4
                k1 = self._f(cur)
                k2 = self._f(cur + 0.5 * dt * k1)
                k3 = self._f(cur + 0.5 * dt * k2)
                k4 = self._f(cur + dt * k3)
                cur = cur + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(cur)

        y_norm = torch.stack(out, dim=1)
        return y_norm * self.a_std + self.a_mean

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def active_terms(self) -> Dict[str, list]:
        """Return dict {equation_i: [active feature names]} for each latent dim."""
        result = {}
        for i in range(self.r):
            col = self.Xi[:, i]
            active = [(self.feature_names[j], float(col[j])) for j in range(len(col)) if col[j].abs() > 1e-10]
            result[f"a{i}"] = active
        return result

    def sparsity(self) -> float:
        """Fraction of zero coefficients in Xi."""
        total = self.Xi.numel()
        zeros = (self.Xi.abs() < 1e-10).sum().item()
        return float(zeros) / max(total, 1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, a: torch.Tensor, *, dt: float = 1.0, return_loss: bool = False) -> ROMOutput:
        """
        Default forward: rollout matching the given horizon.
        a : (B, T, r)
        """
        if a.ndim != 3 or a.shape[-1] != self.r:
            raise ValueError(f"Expected a with shape (B, T, {self.r}), got {tuple(a.shape)}")
        B, T, r = a.shape
        ahat = self.rollout(a[:, 0], dt=dt, steps=T - 1)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=a.device)}
        if return_loss:
            losses["mse"]   = self.mse(ahat, a)
            losses["total"] = losses["mse"]

        return ROMOutput(
            y=ahat,
            losses=losses,
            extras={
                "fitted":   self._fitted,
                "sparsity": self.sparsity(),
                "Xi":       self.Xi,
            },
        )
