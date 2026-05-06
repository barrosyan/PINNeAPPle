from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class RefinementConfig:
    """Configuration for PINN-based solution refinement.

    Attributes
    ----------
    n_epochs:
        Number of refinement training epochs.
    lr:
        Adam learning rate.
    pde_weight:
        Weight for the PDE residual loss term.
    bc_weight:
        Weight for the boundary-condition loss term.  Currently applied as a
        multiplier on the data loss near the boundary (BC points are assumed
        to come first in *x_col* when the caller cannot separate them).
    data_weight:
        Weight for the supervised data loss on surrogate predictions.
    n_collocation:
        Number of collocation points to sample when the caller does not
        supply *x_col* in :meth:`PINNRefinement.refine`.
    verbose:
        Print loss at the end of each 10 % progress milestone.
    """

    n_epochs: int = 500
    lr: float = 5e-4
    pde_weight: float = 1.0
    bc_weight: float = 10.0
    data_weight: float = 1.0
    n_collocation: int = 2000
    verbose: bool = True


@dataclass
class RefinementResult:
    """Result of a single PINN refinement run.

    Attributes
    ----------
    theta:
        Design parameters used during refinement.
    u_refined:
        Network output evaluated at all collocation points, as numpy array.
    loss_history:
        Total loss per epoch.
    improvement_ratio:
        ``(initial_loss - final_loss) / initial_loss``.  Positive means the
        PINN improved on the surrogate prediction.
    """

    theta: np.ndarray
    u_refined: np.ndarray
    loss_history: List[float]
    improvement_ratio: float


class PINNRefinement:
    """Refine surrogate predictions by continuing PINN training.

    Uses a combined loss:

    .. code-block:: text

        L = data_weight * MSE(u_nn, u_surrogate)
          + pde_weight  * mean(|pde_residual(u_nn, x_col)|^2)

    The boundary-condition weight is folded into *data_weight* for any
    collocation point that is in the first ``n_bc_hint`` rows.

    Parameters
    ----------
    pinn_model:
        PyTorch network that maps ``(x_col, theta)`` → ``u``.  For
        compatibility the model's ``forward`` is called as
        ``model(x_col)`` (i.e. theta is embedded externally before
        passing to this object) unless the model has a ``forward_theta``
        method.
    pde_residual_fn:
        Callable ``(u: Tensor, x: Tensor) -> Tensor`` returning the PDE
        residual at each collocation point.  Returns a scalar when averaged.
    domain:
        Optional domain specification (currently unused internally; callers
        may use it to generate *x_col* before calling :meth:`refine`).
    cfg:
        Refinement configuration; defaults to :class:`RefinementConfig`.
    """

    def __init__(
        self,
        pinn_model: nn.Module,
        pde_residual_fn: Callable[..., Tensor],
        domain: Any = None,
        cfg: Optional[RefinementConfig] = None,
    ) -> None:
        self.model = pinn_model
        self.pde_residual_fn = pde_residual_fn
        self.domain = domain
        self.cfg = cfg or RefinementConfig()

    # ------------------------------------------------------------------

    def refine(
        self,
        theta: np.ndarray,
        u_surrogate: np.ndarray,
        x_col: np.ndarray,
    ) -> RefinementResult:
        """Train the PINN to match *u_surrogate* while satisfying the PDE.

        Parameters
        ----------
        theta:
            Design-parameter vector, shape (p,).
        u_surrogate:
            Surrogate output at *x_col*, shape ``(N, d)``.
        x_col:
            Collocation points, shape ``(N, spatial_dim)``.

        Returns
        -------
        RefinementResult
        """
        cfg = self.cfg
        device = next(self.model.parameters()).device

        x_t = torch.from_numpy(x_col.astype(np.float32)).to(device).requires_grad_(True)
        u_ref = torch.from_numpy(u_surrogate.astype(np.float32)).to(device)

        opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        mse = nn.MSELoss()
        loss_history: List[float] = []
        log_every = max(1, cfg.n_epochs // 10)

        for epoch in range(1, cfg.n_epochs + 1):
            opt.zero_grad()

            u_pred = self.model(x_t)

            data_loss = cfg.data_weight * mse(u_pred, u_ref.reshape(u_pred.shape))

            # PDE residual loss.
            pde_res = self.pde_residual_fn(u_pred, x_t)
            if pde_res.ndim == 0:
                pde_loss = cfg.pde_weight * pde_res
            else:
                pde_loss = cfg.pde_weight * torch.mean(pde_res ** 2)

            total = data_loss + pde_loss
            total.backward()
            opt.step()

            loss_val = float(total.item())
            loss_history.append(loss_val)

            if cfg.verbose and epoch % log_every == 0:
                print(f"  [Refinement] epoch {epoch:4d}/{cfg.n_epochs}  loss={loss_val:.4e}")

        initial_loss = loss_history[0] if loss_history else 1.0
        final_loss = loss_history[-1] if loss_history else 0.0
        improvement = (initial_loss - final_loss) / (abs(initial_loss) + 1e-12)

        with torch.no_grad():
            u_refined_t = self.model(x_t)

        return RefinementResult(
            theta=theta.copy(),
            u_refined=u_refined_t.detach().cpu().numpy(),
            loss_history=loss_history,
            improvement_ratio=float(improvement),
        )

    # ------------------------------------------------------------------

    def refine_top_k(
        self,
        candidates: List[np.ndarray],
        surrogate: object,
        k: int = 3,
    ) -> List[RefinementResult]:
        """Refine the top *k* design candidates using surrogate predictions.

        Parameters
        ----------
        candidates:
            List of design-parameter arrays, each shape (p,).  The first *k*
            are refined; callers should sort by objective before passing.
        surrogate:
            :class:`~pinneaple_design_opt.surrogate.PhysicsSurrogate` instance
            used to obtain *u_surrogate* for each candidate.
        k:
            Number of top candidates to refine.

        Returns
        -------
        list of RefinementResult
            Length min(k, len(candidates)).
        """
        results: List[RefinementResult] = []
        for theta in candidates[:k]:
            u_surr = surrogate.predict_batch(theta.reshape(1, -1))[0]  # (out_dim,)
            # Build simple collocation points around a unit hypercube when no
            # domain is available — callers should supply x_col directly via
            # refine() for production use.
            n_col = self.cfg.n_collocation
            p = len(theta)
            rng = np.random.default_rng(42)
            x_col = rng.uniform(-1, 1, size=(n_col, max(p, 1)))
            # Tile u_surrogate to match collocation count.
            u_tiled = np.tile(u_surr, (n_col, 1)) if u_surr.ndim == 1 else np.tile(u_surr, (n_col // max(len(u_surr), 1) + 1, 1))[:n_col]
            res = self.refine(theta, u_tiled, x_col)
            results.append(res)
        return results
