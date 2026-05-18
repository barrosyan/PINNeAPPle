"""Causal training for temporal PDEs.

Enforces that the network learns early times before late times by applying
exponentially decaying weights to later time collocation points, based on the
cumulative residual at earlier times.

Reference: Wang et al. 2022 "Respecting causality is all you need for training
physics-informed neural networks"
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# CausalWeightScheduler
# ---------------------------------------------------------------------------

class CausalWeightScheduler:
    """Compute per-point causal weights for temporal PDE training.

    The causality weight for time chunk k is::

        w_k = exp( -epsilon * sum_{j < k} L_j )

    where L_j is the mean squared residual in chunk j.  Early chunks receive
    weight ≈ 1 and late chunks receive exponentially smaller weights, which
    prevents the optimiser from fitting late times before early times have
    converged.

    Parameters
    ----------
    n_time_chunks : number of equal-width time chunks.
    epsilon       : causal decay strength (larger = stronger causality bias).
    update_every  : recompute chunk-mean residuals every this many epochs
                    (costly because it requires a forward + residual pass).
    """

    def __init__(
        self,
        n_time_chunks: int = 100,
        epsilon: float = 1.0,
        update_every: int = 100,
    ):
        self.n_time_chunks = n_time_chunks
        self.epsilon = epsilon
        self.update_every = update_every

        # Cached chunk losses (initialised to zero → uniform weights initially)
        self._chunk_losses: torch.Tensor = torch.zeros(n_time_chunks)

    # ------------------------------------------------------------------
    # Core computations
    # ------------------------------------------------------------------

    def _assign_chunks(self, t: torch.Tensor) -> torch.Tensor:
        """Map each time point to a chunk index in [0, n_time_chunks).

        Parameters
        ----------
        t : time values, shape [N] or [N, 1], in [0, 1] (normalised).

        Returns
        -------
        chunk_ids : LongTensor [N].
        """
        t_flat = t.flatten().clamp(0.0, 1.0 - 1e-7)
        chunk_ids = (t_flat * self.n_time_chunks).long()
        return chunk_ids

    def update_chunk_losses(
        self, t: torch.Tensor, residuals: torch.Tensor
    ) -> None:
        """Recompute per-chunk mean squared residuals and cache them.

        Parameters
        ----------
        t         : time values [N] or [N, 1].
        residuals : squared or raw residuals [N] or [N, out_dim].
        """
        chunk_ids = self._assign_chunks(t)
        r2 = residuals.detach()
        if r2.ndim > 1:
            r2 = r2.pow(2).mean(dim=1)
        else:
            r2 = r2.pow(2)

        chunk_losses = torch.zeros(
            self.n_time_chunks, device=r2.device, dtype=r2.dtype
        )
        counts = torch.zeros(
            self.n_time_chunks, device=r2.device, dtype=r2.dtype
        )
        chunk_losses.scatter_add_(0, chunk_ids, r2)
        counts.scatter_add_(0, chunk_ids, torch.ones_like(r2))
        # Avoid division by zero for empty chunks
        counts = counts.clamp(min=1.0)
        self._chunk_losses = (chunk_losses / counts).cpu()

    def compute_weights(
        self, t: torch.Tensor, residuals: torch.Tensor
    ) -> torch.Tensor:
        """Compute causal weights given current residuals at time points t.

        Calls ``update_chunk_losses`` first, then returns a weight tensor of
        shape [N] with the same device as ``t``.

        Parameters
        ----------
        t         : time values [N] or [N, 1].
        residuals : PDE residuals [N] or [N, out_dim].

        Returns
        -------
        weights : Tensor [N], values in (0, 1].
        """
        self.update_chunk_losses(t, residuals)
        return self._weights_from_cache(t)

    def _weights_from_cache(self, t: torch.Tensor) -> torch.Tensor:
        """Use cached chunk losses to build per-point weights."""
        device = t.device
        chunk_losses = self._chunk_losses.to(device=device, dtype=t.dtype)

        # Cumulative sum: sum_{j < k} L_j
        # For chunk k, the weight is exp(-eps * cumsum[k])
        # cumsum[0] = 0 (no earlier chunks)
        cumsum = torch.zeros_like(chunk_losses)
        cumsum[1:] = torch.cumsum(chunk_losses[:-1], dim=0)

        chunk_weights = torch.exp(-self.epsilon * cumsum)  # [n_chunks]

        chunk_ids = self._assign_chunks(t)
        weights = chunk_weights[chunk_ids]  # [N]
        return weights

    def weighted_loss(
        self, t: torch.Tensor, residuals: torch.Tensor
    ) -> torch.Tensor:
        """Mean residual loss with causal weighting.

        Updates weights from current residuals, then returns::

            sum_i w_i * r_i^2 / sum_i w_i

        Parameters
        ----------
        t         : time values [N] or [N, 1].
        residuals : raw PDE residuals [N] or [N, out_dim].

        Returns
        -------
        Scalar Tensor.
        """
        weights = self.compute_weights(t, residuals)  # detached weights [N]

        r2 = residuals
        if r2.ndim > 1:
            r2 = r2.pow(2).mean(dim=1)
        else:
            r2 = r2.pow(2)

        # weights are detached (treated as constants for gradient purposes)
        w = weights.detach()
        return (w * r2).sum() / (w.sum() + 1e-12)

    def causality_metric(self) -> float:
        """Causality violation metric.

        Returns the standard deviation of the log-weights across chunks.
        When training is causality-respecting, early chunks converge first
        and the weight distribution becomes more uniform → metric decreases.

        Returns
        -------
        float : std of log(w_k + 1e-8) across all chunks.
        """
        chunk_losses = self._chunk_losses
        cumsum = torch.zeros_like(chunk_losses)
        cumsum[1:] = torch.cumsum(chunk_losses[:-1], dim=0)
        log_weights = -self.epsilon * cumsum
        return float(log_weights.std())


# ---------------------------------------------------------------------------
# CausalPINNTrainer
# ---------------------------------------------------------------------------

class CausalPINNTrainer:
    """PINN trainer with causal weighting for temporal PDEs.

    Usage
    -----
    ::

        trainer = CausalPINNTrainer(model, epsilon=1.0, n_time_chunks=50)
        history = trainer.train(
            pde_residual_fn=burgers_residual,
            ic_loss_fn=lambda model: ic_mse(model, x_ic, u0),
            t_range=(0, 1),
            n_epochs=10000,
        )

    Parameters
    ----------
    model        : nn.Module that maps [N, D] → [N, out_dim].
    epsilon      : causal decay strength (passed to CausalWeightScheduler).
    n_time_chunks: number of time chunks (passed to CausalWeightScheduler).
    update_every : how often to recompute causal weights.
    """

    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 1.0,
        n_time_chunks: int = 50,
        update_every: int = 100,
    ):
        self.model = model
        self.scheduler = CausalWeightScheduler(
            n_time_chunks=n_time_chunks,
            epsilon=epsilon,
            update_every=update_every,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        pde_residual_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        ic_loss_fn: Callable[[nn.Module], torch.Tensor],
        t_range: Tuple[float, float] = (0.0, 1.0),
        n_epochs: int = 10000,
        n_col: int = 2000,
        lr: float = 1e-3,
        bc_loss_fn: Optional[Callable[[nn.Module], torch.Tensor]] = None,
        x_spatial_range: Tuple[float, float] = (0.0, 1.0),
        spatial_dim: int = 1,
        ic_weight: float = 10.0,
        bc_weight: float = 10.0,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler=None,
        print_every: int = 1000,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
    ) -> Dict[str, List[float]]:
        """Run causal PINN training.

        Parameters
        ----------
        pde_residual_fn : callable(model, x_col) -> residuals [N, ...].
                          ``x_col`` has ``requires_grad=True``; the last column
                          is assumed to be the time coordinate t.
        ic_loss_fn      : callable(model) -> scalar Tensor for IC enforcement.
        t_range         : (t_min, t_max) time domain.
        n_epochs        : total gradient steps.
        n_col           : number of random collocation points per step.
        lr              : Adam learning rate (if ``optimizer`` is None).
        bc_loss_fn      : optional callable(model) -> scalar Tensor for BCs.
        x_spatial_range : (x_min, x_max) for random spatial sampling.
        spatial_dim     : number of spatial dimensions.
        ic_weight       : weight on IC loss.
        bc_weight       : weight on BC loss.
        optimizer       : optional pre-built optimizer.
        scheduler       : optional LR scheduler.
        print_every     : log interval (0 to suppress).
        device          : torch device (defaults to CUDA if available).
        dtype           : float dtype.
        seed            : random seed for collocation sampling.

        Returns
        -------
        History dict with lists: ``total``, ``pde``, ``ic``, ``bc``,
        ``causality_metric``.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(device=device, dtype=dtype)

        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=lr)

        rng = np.random.default_rng(seed)

        history: Dict[str, List[float]] = {
            "total": [], "pde": [], "ic": [], "bc": [], "causality_metric": []
        }

        t_lo, t_hi = float(t_range[0]), float(t_range[1])
        x_lo, x_hi = float(x_spatial_range[0]), float(x_spatial_range[1])

        update_every = self.scheduler.update_every

        for epoch in range(1, n_epochs + 1):
            optimizer.zero_grad(set_to_none=True)

            # Sample collocation: [x_1, ..., x_d, t]
            x_sp = rng.uniform(x_lo, x_hi, (n_col, spatial_dim)).astype(np.float32)
            t_col = rng.uniform(t_lo, t_hi, (n_col, 1)).astype(np.float32)
            xt_np = np.concatenate([x_sp, t_col], axis=1)
            x_col = torch.from_numpy(xt_np).to(device=device, dtype=dtype)
            x_col.requires_grad_(True)

            # PDE residual
            res = pde_residual_fn(self.model, x_col)

            # Time tensor for causal weighting (last column)
            t_pts = x_col[:, -1].detach()

            # Recompute weights periodically; otherwise use cached
            if epoch % update_every == 1:
                loss_pde = self.scheduler.weighted_loss(t_pts, res.detach())
            else:
                loss_pde = self.scheduler.weighted_loss(t_pts, res)

            # IC loss
            loss_ic = ic_loss_fn(self.model)

            # BC loss (optional)
            loss_bc = (
                bc_loss_fn(self.model)
                if bc_loss_fn is not None
                else torch.tensor(0.0, device=device, dtype=dtype)
            )

            total = loss_pde + ic_weight * loss_ic + bc_weight * loss_bc
            total.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            history["total"].append(float(total.detach()))
            history["pde"].append(float(loss_pde.detach()))
            history["ic"].append(float(loss_ic.detach()))
            history["bc"].append(float(loss_bc.detach()))
            history["causality_metric"].append(self.scheduler.causality_metric())

            if print_every > 0 and (epoch % print_every == 0 or epoch == 1):
                print(
                    f"[Causal] epoch={epoch:05d}  "
                    f"total={float(total.detach()):.4e}  "
                    f"pde={float(loss_pde.detach()):.4e}  "
                    f"ic={float(loss_ic.detach()):.4e}  "
                    f"bc={float(loss_bc.detach()):.4e}  "
                    f"causality={self.scheduler.causality_metric():.4f}"
                )

        return history

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_causality_metric(self) -> float:
        """Return current causality violation metric.

        The metric is the standard deviation of log-weights across time chunks.
        It should decrease during training as the network learns causal order.

        Returns
        -------
        float
        """
        return self.scheduler.causality_metric()

    def plot_causal_weights(self, t_range: Tuple[float, float] = (0.0, 1.0)) -> None:
        """Visualise the current causal weight distribution.

        Requires matplotlib.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[CausalPINNTrainer.plot_causal_weights] matplotlib not available.")
            return

        n = self.scheduler.n_time_chunks
        t_centers = np.linspace(t_range[0], t_range[1], n)
        chunk_losses = self.scheduler._chunk_losses.numpy()
        cumsum = np.zeros(n)
        cumsum[1:] = np.cumsum(chunk_losses[:-1])
        weights = np.exp(-self.scheduler.epsilon * cumsum)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(t_centers, chunk_losses, "b-o", markersize=3)
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("Chunk mean residual L_k")
        axes[0].set_title("Per-chunk residuals")

        axes[1].plot(t_centers, weights, "r-o", markersize=3)
        axes[1].set_xlabel("t")
        axes[1].set_ylabel("Causal weight w_k")
        axes[1].set_title(f"Causal weights (epsilon={self.scheduler.epsilon})")

        plt.tight_layout()
        plt.show()
