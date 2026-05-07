from __future__ import annotations
"""
NeuralROM — Fully neural reduced-order model.

Architecture
------------
  Encoder  : x_t (D) -------> z_t (r)          [nonlinear projection]
  Dynamics : z_t (r) -------> z_{t+1} (r)       [recurrent or MLP 1-step map]
  Decoder  : z_t (r) -------> x_hat_t (D)       [nonlinear reconstruction]

Three dynamics options:
  "mlp"    : feed-forward 1-step map  z_{t+1} = f_theta(z_t)
  "lstm"   : LSTM in latent space  (best for long trajectories)
  "gru"    : GRU in latent space

The encoder/decoder are trained end-to-end with the dynamics module via:
  L_recon + L_pred (multi-step rollout MSE)

Difference from KoopmanAutoencoder
-----------------------------------
No linearity constraint — the latent dynamics can be fully nonlinear.
Koopman is a special case when the dynamics module is a linear layer.

Parameters
----------
in_dim      : physical state dimension D
r           : latent dimension
hidden_enc  : encoder/decoder hidden width
n_layers_enc: encoder/decoder depth
dynamics    : "mlp", "lstm", or "gru"
hidden_dyn  : hidden width of dynamics module
n_layers_dyn: depth of dynamics MLP (only for "mlp")
dropout     : dropout probability in dynamics
pred_steps  : prediction horizon for training loss
w_recon     : reconstruction loss weight
w_pred      : prediction loss weight
"""
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ROMBase, ROMOutput


class _MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, n_layers, activation="tanh", dropout=0.0):
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        layers: list = [nn.Linear(in_dim, hidden), act()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), act()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _MLPDynamics(nn.Module):
    """1-step MLP map: z_t -> z_{t+1}."""
    def __init__(self, r, hidden, n_layers, dropout=0.0):
        super().__init__()
        self.net = _MLP(r, r, hidden, n_layers, "tanh", dropout)

    def forward(self, z: torch.Tensor, h=None) -> Tuple[torch.Tensor, None]:
        return self.net(z), None

    def init_hidden(self, B: int, device) -> None:
        return None


class _RNNDynamics(nn.Module):
    """Recurrent dynamics (LSTM or GRU) in latent space."""
    def __init__(self, r, hidden, rnn_type="lstm", dropout=0.0):
        super().__init__()
        self.r = r
        self.hidden = hidden
        cls = nn.LSTM if rnn_type == "lstm" else nn.GRU
        self.rnn = cls(r, hidden, num_layers=1, batch_first=True,
                       dropout=0.0)
        self.proj = nn.Linear(hidden, r)
        self._is_lstm = (rnn_type == "lstm")

    def forward(self, z: torch.Tensor, h) -> Tuple[torch.Tensor, object]:
        # z: (B, r)
        z_in = z.unsqueeze(1)  # (B, 1, r)
        out, h_new = self.rnn(z_in, h)
        return self.proj(out.squeeze(1)), h_new

    def init_hidden(self, B: int, device) -> object:
        w = next(self.parameters())
        h0 = torch.zeros(1, B, self.hidden, device=device, dtype=w.dtype)
        if self._is_lstm:
            return h0, torch.zeros_like(h0)
        return h0


class NeuralROM(ROMBase):
    """Fully neural ROM: nonlinear encoder + latent dynamics + nonlinear decoder."""

    def __init__(
        self,
        in_dim: int,
        r: int = 32,
        hidden_enc: int = 128,
        n_layers_enc: int = 3,
        dynamics: Literal["mlp", "lstm", "gru"] = "mlp",
        hidden_dyn: int = 128,
        n_layers_dyn: int = 2,
        dropout: float = 0.0,
        pred_steps: int = 5,
        w_recon: float = 1.0,
        w_pred: float = 1.0,
        activation: str = "tanh",
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.r = int(r)
        self.pred_steps = int(pred_steps)
        self.w_recon = float(w_recon)
        self.w_pred = float(w_pred)
        self.dynamics_kind = dynamics

        self.encoder = _MLP(in_dim, r,      hidden_enc, n_layers_enc, activation, dropout)
        self.decoder = _MLP(r,      in_dim, hidden_enc, n_layers_enc, activation, dropout)

        if dynamics == "mlp":
            self.dyn_module: nn.Module = _MLPDynamics(r, hidden_dyn, n_layers_dyn, dropout)
        elif dynamics in ("lstm", "gru"):
            self.dyn_module = _RNNDynamics(r, hidden_dyn, dynamics, dropout)
        else:
            raise ValueError(f"dynamics must be 'mlp', 'lstm', or 'gru', got '{dynamics}'")

    # ------------------------------------------------------------------
    # Core ops
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., D) -> z: (..., r)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., r) -> x: (..., D)"""
        return self.decoder(z)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def rollout(
        self,
        x0: torch.Tensor,
        steps: int,
        *,
        teacher_forcing: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Roll out the model from x0.

        x0             : (B, D)
        steps          : number of future steps
        teacher_forcing: (B, T, D) ground-truth sequence for teacher forcing
                         (used during training; if None, free-running rollout)
        Returns (B, steps+1, D).
        """
        B = x0.shape[0]
        z = self.encode(x0)
        h = self.dyn_module.init_hidden(B, x0.device)
        xs = [self.decode(z)]

        for s in range(steps):
            z, h = self.dyn_module(z, h)
            xs.append(self.decode(z))
            if teacher_forcing is not None and s < teacher_forcing.shape[1] - 1:
                z = self.encode(teacher_forcing[:, s + 1])

        return torch.stack(xs, dim=1)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_losses(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute reconstruction + multi-step prediction loss from X (B, T, D)."""
        B, T, D = X.shape

        # Reconstruction at every time step
        Z = self.encode(X.reshape(B * T, D)).reshape(B, T, self.r)
        X_recon = self.decode(Z.reshape(B * T, self.r)).reshape(B, T, D)
        l_recon = F.mse_loss(X_recon, X)

        # Multi-step prediction (teacher-forcing rollout)
        steps = min(self.pred_steps, T - 1)
        X_pred = self.rollout(X[:, 0], steps=steps, teacher_forcing=X)
        l_pred = F.mse_loss(X_pred[:, :steps + 1], X[:, :steps + 1])

        total = self.w_recon * l_recon + self.w_pred * l_pred
        return {"recon": l_recon, "pred": l_pred, "total": total}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        """
        X : (B, T, D) or (T, D)
        """
        if X.ndim == 2:
            X = X.unsqueeze(0)
        if X.ndim != 3:
            raise ValueError(f"Expected X with shape (B,T,D) or (T,D), got {tuple(X.shape)}")
        B, T, D = X.shape

        losses: Dict[str, torch.Tensor] = {}
        if return_loss:
            losses = self.compute_losses(X)
        else:
            losses["total"] = torch.tensor(0.0, device=X.device)

        with torch.set_grad_enabled(self.training):
            x_hat = self.rollout(X[:, 0], steps=T - 1)

        return ROMOutput(
            y=x_hat,
            losses=losses,
            extras={
                "r":        self.r,
                "dynamics": self.dynamics_kind,
            },
        )
