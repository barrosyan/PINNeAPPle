from __future__ import annotations
"""Echo State Network (ESN) with leaky reservoir dynamics + ridge readout.

Supports:
- Teacher-forced training (driven mode)
- 1-step-ahead training for autoregressive forecasting
- Autoregressive multi-step generation with flexible feedback mapping:
  * y -> x directly when in_dim == out_dim
  * partial feedback into selected input dims when in_dim != out_dim
  * optional future exogenous inputs during rollout
"""
from typing import Dict, Optional, Sequence, Tuple, List

import torch
import torch.nn as nn

from .base import RCBase, RCOutput


def _spectral_radius_power(W: torch.Tensor, iters: int = 80) -> torch.Tensor:
    """Cheap approximation. For non-symmetric W this can be imperfect."""
    v = torch.randn(W.shape[0], 1, device=W.device, dtype=W.dtype)
    for _ in range(iters):
        v = W @ v
        v = v / (v.norm() + 1e-12)
    lam = (v.t() @ (W @ v)).squeeze()
    return lam.abs().clamp_min(1e-12)


def _spectral_radius_eig(W: torch.Tensor) -> torch.Tensor:
    """More faithful spectral radius via eigvals (costlier)."""
    ev = torch.linalg.eigvals(W)  # complex
    return ev.abs().max().real.clamp_min(1e-12)


def _make_sparse_mask(n: int, density: float, device, dtype) -> torch.Tensor:
    """Binary mask with approximately density fraction of ones."""
    density = float(density)
    if density >= 1.0:
        return torch.ones((n, n), device=device, dtype=dtype)
    if density <= 0.0:
        return torch.zeros((n, n), device=device, dtype=dtype)
    # Bernoulli mask
    return (torch.rand((n, n), device=device) < density).to(dtype=dtype)


class EchoStateNetwork(RCBase):
    """
    Echo State Network (ESN):
      h_t = (1-leak)*h_{t-1} + leak*tanh(x_t W_in + h_{t-1} W)
      y_t = phi(h_t, x_t) W_out  (W_out trained by ridge)

    Tensors:
      x: (B,T,in_dim)
      y: (B,T,out_dim)   (optional in fit; can train to predict x_{t+1} when y is None)
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        reservoir_dim: int = 1024,
        spectral_radius: float = 0.9,
        leak: float = 1.0,
        input_scale: float = 1.0,
        l2: float = 1e-6,
        use_skip: bool = True,   # include x in readout features
        use_bias: bool = True,
        freeze_random: bool = True,
        reservoir_density: float = 1.0,  # set <1.0 to make W sparse (common in ESN practice)
        spectral_method: str = "auto",   # "auto" | "eig" | "power"
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.reservoir_dim = int(reservoir_dim)

        self.spectral_radius = float(spectral_radius)
        self.leak = float(leak)
        self.input_scale = float(input_scale)
        self.l2 = float(l2)

        self.use_skip = bool(use_skip)
        self.use_bias = bool(use_bias)

        self.reservoir_density = float(reservoir_density)
        self.spectral_method = str(spectral_method)

        if not (0.0 < self.leak <= 1.0):
            raise ValueError("leak should be in (0, 1].")

        # Random weights
        Win = torch.randn(self.in_dim, self.reservoir_dim) * (
            self.input_scale / (max(self.in_dim, 1) ** 0.5)
        )
        W = torch.randn(self.reservoir_dim, self.reservoir_dim) * (
            1.0 / (max(self.reservoir_dim, 1) ** 0.5)
        )

        # Optional sparsity (common ESN setting)
        if self.reservoir_density < 1.0:
            mask = _make_sparse_mask(self.reservoir_dim, self.reservoir_density, W.device, W.dtype)
            W = W * mask

        # Scale to desired spectral radius
        with torch.no_grad():
            Wf = W.float()
            method = self.spectral_method
            if method == "auto":
                # eig is accurate but can be heavier; power is cheaper
                method = "eig" if self.reservoir_dim <= 2048 else "power"

            if method == "eig":
                sr = _spectral_radius_eig(Wf)
            elif method == "power":
                sr = _spectral_radius_power(Wf)
            else:
                raise ValueError("spectral_method must be 'auto', 'eig', or 'power'.")

            W = (W * (self.spectral_radius / sr.to(W.dtype))).to(W.dtype)

        self.W_in = nn.Parameter(Win)
        self.W = nn.Parameter(W)

        if freeze_random:
            self.W_in.requires_grad_(False)
            self.W.requires_grad_(False)

        readout_dim = self.reservoir_dim + (self.in_dim if self.use_skip else 0) + (1 if self.use_bias else 0)
        self.W_out = nn.Parameter(torch.zeros(readout_dim, self.out_dim), requires_grad=False)

        self._fitted = False

    def _step(self, x_t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # x_t: (B,in_dim), h: (B,reservoir_dim)
        pre = x_t @ self.W_in + h @ self.W
        h_tilde = torch.tanh(pre)
        return (1.0 - self.leak) * h + self.leak * h_tilde

    def _readout_features(self, h: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        feats = [h]
        if self.use_skip:
            feats.append(x_t)
        if self.use_bias:
            feats.append(torch.ones((h.shape[0], 1), device=h.device, dtype=h.dtype))
        return torch.cat(feats, dim=-1)

    @staticmethod
    def _apply_feedback(
        x_base: torch.Tensor,
        y_pred: torch.Tensor,
        *,
        feedback_idx: Optional[Sequence[int]] = None,
        y_to_x_idx: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """
        Put predicted y back into the next x.

        - If feedback_idx is None: require x_dim == y_dim and overwrite all x with y.
        - If feedback_idx is given: overwrite x[:, feedback_idx] with y[:, y_to_x_idx] (or y directly if same length).
        """
        if feedback_idx is None:
            if x_base.shape[-1] != y_pred.shape[-1]:
                raise ValueError(
                    "in_dim != out_dim. Provide feedback_idx (and optionally y_to_x_idx) to map y -> x."
                )
            return y_pred

        x_next = x_base.clone()
        fb = list(map(int, feedback_idx))

        if y_to_x_idx is None:
            if len(fb) != y_pred.shape[-1]:
                raise ValueError(
                    "feedback_idx length must match out_dim if y_to_x_idx is None."
                )
            x_next[:, fb] = y_pred
        else:
            ymap = list(map(int, y_to_x_idx))
            if len(fb) != len(ymap):
                raise ValueError("feedback_idx and y_to_x_idx must have the same length.")
            x_next[:, fb] = y_pred[:, ymap]

        return x_next

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,                 # (B,T,in_dim)
        y: Optional[torch.Tensor] = None, # (B,T,out_dim) optional
        *,
        washout: int = 0,
        one_step_ahead: bool = True,
        # If y is None, train to predict x_{t+1} for autoregressive forecasting (requires in_dim == out_dim)
        target_from_input_if_y_none: bool = True,
    ) -> "EchoStateNetwork":
        """
        Training:
        - If one_step_ahead=True: learn mapping from state/features at t to target at t+1.
        - If one_step_ahead=False: learn mapping to target at same t.
        """
        if y is None and target_from_input_if_y_none:
            if self.in_dim != self.out_dim:
                raise ValueError("y is None and in_dim != out_dim; provide y or change dims.")
            y = x  # predict next step of x

        if y is None:
            raise ValueError("y is required unless target_from_input_if_y_none=True and in_dim==out_dim.")

        B, T, _ = x.shape
        h = torch.zeros((B, self.reservoir_dim), device=x.device, dtype=x.dtype)

        X_list: List[torch.Tensor] = []
        Y_list: List[torch.Tensor] = []

        if one_step_ahead:
            # Need t -> t+1, so last usable t is T-2
            for t in range(T - 1):
                h = self._step(x[:, t, :], h)
                if t >= int(washout):
                    X_list.append(self._readout_features(h, x[:, t, :]))
                    Y_list.append(y[:, t + 1, :])
        else:
            for t in range(T):
                h = self._step(x[:, t, :], h)
                if t >= int(washout):
                    X_list.append(self._readout_features(h, x[:, t, :]))
                    Y_list.append(y[:, t, :])

        Xr = torch.cat(X_list, dim=0)  # (B*(T-w'), F)
        Yr = torch.cat(Y_list, dim=0)  # (B*(T-w'), O)

        W_out = self.ridge_solve(Xr, Yr, l2=self.l2)
        self.W_out.copy_(W_out)
        self._fitted = True
        return self

    @torch.no_grad()
    def predict_autoregressive(
        self,
        x_context: torch.Tensor,                 # (B,Tc,in_dim) observed context
        steps: int,                              # horizon
        *,
        future_exog: Optional[torch.Tensor] = None,  # (B,steps,in_dim) optional full inputs for each future step
        feedback_idx: Optional[Sequence[int]] = None,
        y_to_x_idx: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """
        Multi-step autoregressive rollout.

        Cases:
        A) Pure autoregressive (no exog), in_dim == out_dim:
           - call with feedback_idx=None (default) and future_exog=None
        B) Inputs include exogenous dims:
           - provide future_exog with known exogenous values for each future step
           - provide feedback_idx pointing to the target dims inside x to be overwritten by y_pred
           - optionally provide y_to_x_idx if out_dim doesn't match len(feedback_idx) ordering
        """
        if not self._fitted:
            raise RuntimeError("Call fit(...) before predict_autoregressive(...).")

        B, Tc, _ = x_context.shape
        device, dtype = x_context.device, x_context.dtype
        h = torch.zeros((B, self.reservoir_dim), device=device, dtype=dtype)

        # warm-up on context
        for t in range(Tc):
            h = self._step(x_context[:, t, :], h)

        # initial x_t for the first generated step
        x_t = x_context[:, -1, :]

        ys: List[torch.Tensor] = []
        for k in range(int(steps)):
            # If we have future exogenous inputs, start from them as base for this step
            if future_exog is not None:
                x_base = future_exog[:, k, :]
            else:
                x_base = x_t

            # advance reservoir and produce y
            h = self._step(x_base, h)
            feats = self._readout_features(h, x_base)
            y_t = feats @ self.W_out
            ys.append(y_t)

            # build next x by feeding back prediction
            x_t = self._apply_feedback(
                x_base,
                y_t,
                feedback_idx=feedback_idx,
                y_to_x_idx=y_to_x_idx,
            )

        return torch.stack(ys, dim=1)  # (B,steps,out_dim)

    def forward(
        self,
        x: torch.Tensor,  # (B,T,in_dim)
        *,
        y_true: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        washout: int = 0,
    ) -> RCOutput:
        """
        Driven-mode forward (teacher forcing):
        - Produces y_hat for each timestep using the provided x sequence.
        - washout affects loss computation only (common ESN evaluation).
        """
        B, T, _ = x.shape
        h = torch.zeros((B, self.reservoir_dim), device=x.device, dtype=x.dtype)

        ys: List[torch.Tensor] = []
        for t in range(T):
            h = self._step(x[:, t, :], h)
            feats = self._readout_features(h, x[:, t, :])
            y_t = feats @ self.W_out
            ys.append(y_t)

        y_hat = torch.stack(ys, dim=1)  # (B,T,out_dim)

        losses: Dict[str, torch.Tensor] = {"total": torch.tensor(0.0, device=x.device)}
        if return_loss and y_true is not None:
            if washout > 0:
                losses["mse"] = torch.mean((y_hat[:, washout:, :] - y_true[:, washout:, :]) ** 2)
            else:
                losses["mse"] = self.mse(y_hat, y_true)
            losses["total"] = losses["mse"]

        return RCOutput(y=y_hat, losses=losses, extras={"fitted": self._fitted})