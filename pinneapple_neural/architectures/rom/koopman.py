from __future__ import annotations
"""
Deep Koopman Autoencoder.

Learns an encoder g: R^D -> R^K that maps the physical state into a
space where the dynamics are (approximately) linear:

    z_{t+1}  =  K  z_t          (Koopman linear operator)
    x_hat_t  =  h(z_t)          (decoder)

Loss terms
----------
L_recon     : ||h(g(x)) - x||^2
L_pred      : ||h(K z_t) - x_{t+1}||^2       (multi-step)
L_linear    : ||K z_t - g(x_{t+1})||^2        (linearity in latent space)
L_metric    : penalty if encoder Jacobian deviates from identity (optional)

The Koopman operator K is parametrised as a learnable (K_dim x K_dim) matrix.
For large systems, K can be block-diagonal (num_blocks x block_size x block_size)
to reduce parameter count while keeping structured dynamics.

Reference
---------
Lusch, B., Kutz, J. N., & Brunton, S. L. (2018).
Deep learning for universal linear embeddings of nonlinear dynamics. Nature Comm.
"""
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ROMBase, ROMOutput


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, n_layers: int, activation: str = "tanh"):
        super().__init__()
        act = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        layers: list = [nn.Linear(in_dim, hidden), act()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KoopmanOperator(nn.Module):
    """
    Learnable Koopman operator K.

    Structure options:
    - "full"   : unconstrained (K_dim x K_dim) dense matrix
    - "block"  : block-diagonal with num_blocks blocks of size block_size
    - "stable" : full matrix constrained to have spectral radius <= 1
                 via normalisation (for bounded dynamics)
    """

    def __init__(
        self,
        k_dim: int,
        structure: Literal["full", "block", "stable"] = "full",
        num_blocks: int = 4,
    ):
        super().__init__()
        self.k_dim = k_dim
        self.structure = structure
        self.num_blocks = num_blocks

        if structure == "block":
            assert k_dim % num_blocks == 0, "k_dim must be divisible by num_blocks"
            bs = k_dim // num_blocks
            self.K_blocks = nn.Parameter(torch.eye(bs).unsqueeze(0).repeat(num_blocks, 1, 1))
        else:
            self.K = nn.Parameter(torch.eye(k_dim) + 0.01 * torch.randn(k_dim, k_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply K to z of shape (..., k_dim)."""
        if self.structure == "block":
            bs = self.k_dim // self.num_blocks
            z_blocks = z.reshape(*z.shape[:-1], self.num_blocks, bs)
            out = torch.einsum("...bi,bji->...bj", z_blocks, self.K_blocks)
            return out.reshape(*z.shape[:-1], self.k_dim)
        elif self.structure == "stable":
            # spectral normalisation: K / max(1, rho(K))
            sv = torch.linalg.matrix_norm(self.K, ord=2)
            K_stable = self.K / sv.clamp_min(1.0)
            return z @ K_stable.t()
        else:
            return z @ self.K.t()


class KoopmanAutoencoder(ROMBase):
    """
    Deep Koopman autoencoder: learned lifting + linear dynamics.

    Parameters
    ----------
    in_dim       : physical state dimension D
    k_dim        : Koopman embedding dimension K (>= in_dim recommended)
    hidden       : hidden width of encoder/decoder MLPs
    n_layers     : depth of encoder/decoder
    activation   : activation function ("tanh", "relu", "gelu", "silu")
    k_structure  : Koopman operator structure ("full", "block", "stable")
    num_blocks   : for block-diagonal K, number of blocks
    pred_steps   : number of future steps used in prediction loss during forward
    w_recon      : weight for reconstruction loss
    w_pred       : weight for prediction loss
    w_linear     : weight for linearity-in-latent loss
    """

    def __init__(
        self,
        in_dim: int,
        k_dim: int = 128,
        hidden: int = 128,
        n_layers: int = 3,
        activation: str = "tanh",
        k_structure: Literal["full", "block", "stable"] = "full",
        num_blocks: int = 4,
        pred_steps: int = 3,
        w_recon: float = 1.0,
        w_pred: float = 1.0,
        w_linear: float = 1.0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.k_dim = int(k_dim)
        self.pred_steps = int(pred_steps)
        self.w_recon = float(w_recon)
        self.w_pred = float(w_pred)
        self.w_linear = float(w_linear)

        self.encoder = _MLP(in_dim, k_dim,  hidden, n_layers, activation)
        self.decoder = _MLP(k_dim,  in_dim, hidden, n_layers, activation)
        self.K_op    = KoopmanOperator(k_dim, structure=k_structure, num_blocks=num_blocks)

    # ------------------------------------------------------------------
    # Core ops
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., D) -> z: (..., K)"""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., K) -> x_hat: (..., D)"""
        return self.decoder(z)

    def step(self, z: torch.Tensor) -> torch.Tensor:
        """One Koopman step: z -> K z, shape (..., K)."""
        return self.K_op(z)

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def rollout(self, x0: torch.Tensor, steps: int) -> torch.Tensor:
        """
        Roll out the Koopman model from initial state x0.

        x0    : (B, D)
        steps : number of future steps
        Returns (B, steps+1, D).
        """
        z = self.encode(x0)
        xs = [self.decode(z)]
        for _ in range(steps):
            z = self.step(z)
            xs.append(self.decode(z))
        return torch.stack(xs, dim=1)

    def rollout_latent(self, z0: torch.Tensor, steps: int) -> torch.Tensor:
        """Roll out in latent space. z0: (B, K). Returns (B, steps+1, K)."""
        zs = [z0]
        cur = z0
        for _ in range(steps):
            cur = self.step(cur)
            zs.append(cur)
        return torch.stack(zs, dim=1)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_losses(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all training losses from trajectory X of shape (B, T, D).
        """
        B, T, D = X.shape

        # Reconstruction loss at each time step
        X_flat = X.reshape(B * T, D)
        Z_flat = self.encode(X_flat)
        X_hat  = self.decode(Z_flat)
        l_recon = F.mse_loss(X_hat, X_flat)

        # Prediction + linearity losses over pred_steps future steps
        l_pred   = torch.tensor(0.0, device=X.device)
        l_linear = torch.tensor(0.0, device=X.device)
        n_pairs  = 0

        Z = Z_flat.reshape(B, T, self.k_dim)
        for s in range(1, min(self.pred_steps + 1, T)):
            # Multi-step Koopman prediction
            Z_pred_s = self.rollout_latent(Z[:, 0], steps=s)[:, s]  # (B, K)
            # Linearity: K^s z_0 should equal g(x_s)
            Z_true_s = Z[:, s]
            l_linear = l_linear + F.mse_loss(Z_pred_s, Z_true_s)
            # Prediction: decode(K^s z_0) should equal x_s
            X_pred_s = self.decode(Z_pred_s)
            l_pred = l_pred + F.mse_loss(X_pred_s, X[:, s])
            n_pairs += 1

        if n_pairs > 0:
            l_linear = l_linear / n_pairs
            l_pred   = l_pred   / n_pairs

        total = self.w_recon * l_recon + self.w_pred * l_pred + self.w_linear * l_linear
        return {
            "recon":   l_recon,
            "pred":    l_pred,
            "linear":  l_linear,
            "total":   total,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, X: torch.Tensor, *, return_loss: bool = False) -> ROMOutput:
        """
        X : (B, T, D)  — trajectory snapshots
        Returns ROMOutput with y = reconstructed trajectory (B, T, D).
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

        # Reconstruct via encode->rollout->decode
        with torch.set_grad_enabled(self.training):
            x_hat = self.rollout(X[:, 0], steps=T - 1)

        return ROMOutput(
            y=x_hat,
            losses=losses,
            extras={
                "k_dim":      self.k_dim,
                "k_structure": self.K_op.structure,
                "pred_steps": self.pred_steps,
            },
        )
