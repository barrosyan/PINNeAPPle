from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class SurrogateConfig:
    """Configuration for a :class:`PhysicsSurrogate`.

    Attributes
    ----------
    model_name:
        Registered model name or ``"MLP"`` for a built-in multi-layer
        perceptron.
    in_dim:
        Number of design parameters (input dimensionality).
    out_channels:
        Number of output field channels.
    hidden_dim:
        Width of each hidden layer.
    n_layers:
        Number of hidden layers.
    device:
        Torch device string.
    checkpoint_path:
        Optional path to a saved surrogate checkpoint to load at startup.
    """

    model_name: str = "MLP"
    in_dim: int = 4
    out_channels: int = 2
    hidden_dim: int = 128
    n_layers: int = 4
    device: str = "cpu"
    checkpoint_path: Optional[str] = None


class PhysicsSurrogate:
    """Thin wrapper that exposes a consistent interface over any ``nn.Module``.

    Parameters
    ----------
    model:
        PyTorch module that maps ``theta -> u``.
    cfg:
        Optional surrogate configuration; stored for reference and checkpointing.
    """

    def __init__(self, model: nn.Module, cfg: Optional[SurrogateConfig] = None) -> None:
        self.model = model
        self.cfg = cfg or SurrogateConfig()
        self._device = torch.device(self.cfg.device)
        self.model.to(self._device)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, theta: Union[np.ndarray, Tensor]) -> Tensor:
        """Run a single forward pass and return the output tensor.

        Parameters
        ----------
        theta:
            Design parameters; converted to a float tensor automatically.

        Returns
        -------
        Tensor
            Model output on the configured device.
        """
        if isinstance(theta, np.ndarray):
            theta = torch.from_numpy(theta.astype(np.float32))
        theta = theta.float().to(self._device)
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        with torch.no_grad():
            return self.model(theta).squeeze(0)

    def predict_batch(self, theta_batch: np.ndarray) -> np.ndarray:
        """Batch predict; numpy in, numpy out — compatible with non-differentiable callers.

        Parameters
        ----------
        theta_batch:
            Array of shape (B, p).

        Returns
        -------
        np.ndarray
            Array of shape (B, out_channels).
        """
        t = torch.from_numpy(theta_batch.astype(np.float32)).to(self._device)
        with torch.no_grad():
            out = self.model(t)
        return out.cpu().numpy()

    # ------------------------------------------------------------------
    # Gradient computation
    # ------------------------------------------------------------------

    def jacobian_theta(self, theta: Tensor) -> Tensor:
        """Compute the Jacobian of the model output w.r.t. *theta*.

        Uses ``torch.autograd.functional.jacobian`` which requires the model
        to be differentiable and in training mode with no ``torch.no_grad``
        context active.

        Parameters
        ----------
        theta:
            1-D design parameter tensor, shape (p,).

        Returns
        -------
        Tensor
            Jacobian of shape (k, p) where k is the output dimensionality.
        """
        theta = theta.float().to(self._device).requires_grad_(True)

        def _forward(t: Tensor) -> Tensor:
            return self.model(t.unsqueeze(0)).squeeze(0)

        jac = torch.autograd.functional.jacobian(_forward, theta)
        # jacobian returns shape (k, p) when output is 1-D and input is 1-D.
        return jac  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------

    def train_on_dataset(
        self,
        dataset: Dict[str, Tensor],
        *,
        epochs: int = 200,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> None:
        """Train the surrogate on ``theta -> u`` pairs using Adam + MSE.

        Parameters
        ----------
        dataset:
            Dictionary with at least keys ``"theta"`` (shape (B, p)) and
            ``"u"`` (shape (B, ...)), both ``torch.Tensor``.
        epochs:
            Number of full-dataset passes.
        lr:
            Adam learning rate.
        verbose:
            Print loss every 10 % of epochs.
        """
        theta_data = dataset["theta"].float().to(self._device)
        u_data = dataset["u"].float().to(self._device)

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        log_every = max(1, epochs // 10)
        for epoch in range(1, epochs + 1):
            opt.zero_grad()
            pred = self.model(theta_data)
            loss = loss_fn(pred, u_data.reshape(pred.shape))
            loss.backward()
            opt.step()

            if verbose and epoch % log_every == 0:
                print(f"  [Surrogate] epoch {epoch:4d}/{epochs}  loss={loss.item():.4e}")

        self.model.eval()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model weights and config to *path*."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "cfg": self.cfg,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, model: nn.Module) -> "PhysicsSurrogate":
        """Load a :class:`PhysicsSurrogate` from a checkpoint saved by :meth:`save`.

        Parameters
        ----------
        path:
            Path to the ``.pt`` checkpoint file.
        model:
            An instantiated (architecture-matching) ``nn.Module`` to receive
            the weights.  The architecture must match what was saved.

        Returns
        -------
        PhysicsSurrogate
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        cfg: SurrogateConfig = ckpt.get("cfg", SurrogateConfig())
        surrogate = cls(model, cfg)
        surrogate.model.eval()
        return surrogate

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build_mlp(
        cls,
        in_dim: int,
        out_dim: int,
        hidden: int = 128,
        layers: int = 4,
        device: str = "cpu",
    ) -> "PhysicsSurrogate":
        """Build a fully-connected MLP surrogate with Tanh activations.

        Parameters
        ----------
        in_dim:
            Number of design parameters.
        out_dim:
            Output field dimensionality.
        hidden:
            Width of every hidden layer.
        layers:
            Number of hidden layers.
        device:
            Torch device string.

        Returns
        -------
        PhysicsSurrogate
        """
        parts: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            parts.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        parts.append(nn.Linear(hidden, out_dim))
        mlp = nn.Sequential(*parts)

        cfg = SurrogateConfig(
            model_name="MLP",
            in_dim=in_dim,
            out_channels=out_dim,
            hidden_dim=hidden,
            n_layers=layers,
            device=device,
        )
        surrogate = cls(mlp, cfg)
        surrogate.model.eval()
        return surrogate
