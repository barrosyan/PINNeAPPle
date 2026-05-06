"""DoMINO: Domain Decomposition for Physics-Informed Neural Networks.

Splits a large domain into overlapping subdomains, trains a separate PINN per
subdomain, and enforces interface continuity conditions.

Reference: Li et al. 2020 "D3M: A Deep Domain Decomposition Method for PDE"
"""
from __future__ import annotations

import itertools
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


class Subdomain:
    """Rectangular subdomain with overlap.

    Parameters
    ----------
    bounds : list of (lo, hi) tuples, one per spatial dimension.
    overlap : fractional overlap added on each side of the subdomain boundary.
              E.g. overlap=0.1 means 10 % of each axis width is added on each
              side to form the extended/interface region.
    """

    def __init__(self, bounds: List[Tuple[float, float]], overlap: float = 0.1):
        self.bounds = bounds  # [(lo0, hi0), (lo1, hi1), ...]
        self.overlap = overlap
        # extended bounds (for contains check — points in overlap count as IN)
        self._dim = len(bounds)
        self._widths = [hi - lo for lo, hi in bounds]
        self._ext_bounds = [
            (lo - overlap * w, hi + overlap * w)
            for (lo, hi), w in zip(bounds, self._widths)
        ]

    # ------------------------------------------------------------------
    # Core geometric predicates
    # ------------------------------------------------------------------

    def contains(self, x: torch.Tensor) -> torch.Tensor:
        """Boolean mask: which rows of x (shape [N, D]) are in this subdomain.

        Uses the *extended* bounds (including the overlap strip) so that every
        point in the domain belongs to at least one subdomain.
        """
        mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        for dim_idx, (lo, hi) in enumerate(self._ext_bounds):
            col = x[:, dim_idx]
            mask = mask & (col >= lo) & (col <= hi)
        return mask

    def contains_strict(self, x: torch.Tensor) -> torch.Tensor:
        """Boolean mask using the *non-extended* (strict) bounds."""
        mask = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        for dim_idx, (lo, hi) in enumerate(self.bounds):
            col = x[:, dim_idx]
            mask = mask & (col >= lo) & (col <= hi)
        return mask

    def is_interface(self, x: torch.Tensor, tol: float = 1e-3) -> torch.Tensor:
        """Points that lie inside the overlap/interface strip.

        A point is in the interface if it is inside the extended bounds but
        within ``overlap * width + tol`` of any subdomain edge.
        """
        in_extended = self.contains(x)
        in_strict = self.contains_strict(x)
        # interface = in extended but not in strict (i.e. in the overlap strip)
        interface = in_extended & (~in_strict)
        return interface

    def __repr__(self) -> str:
        return f"Subdomain(bounds={self.bounds}, overlap={self.overlap})"


# ---------------------------------------------------------------------------
# SubdomainPINN
# ---------------------------------------------------------------------------

class SubdomainPINN(nn.Module):
    """PINN for a single subdomain.

    Parameters
    ----------
    subdomain : Subdomain
    model     : any nn.Module that maps (N, D) inputs → (N, out_dim) outputs.
    """

    def __init__(self, subdomain: Subdomain, model: nn.Module):
        super().__init__()
        self.subdomain = subdomain
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# DoMINO
# ---------------------------------------------------------------------------

class DoMINO(nn.Module):
    """Domain-decomposition PINN trainer.

    Each subdomain gets its own network (created by ``model_factory``).
    In the overlap region the outputs are combined using a distance-weighted
    average (soft-partition-of-unity blending).

    Usage
    -----
    ::

        subdomains = DoMINO.partition(bounds=[(0,1),(0,1)], n_splits=(2,2), overlap=0.1)
        domino = DoMINO(subdomains, model_factory=lambda: MLP(2, 1, 64, 4))
        history = domino.train(residual_fn, bc_fn, n_epochs=5000)

    Parameters
    ----------
    subdomains        : list of Subdomain instances.
    model_factory     : zero-argument callable that returns a fresh nn.Module.
    interface_weight  : weight multiplier applied to the interface continuity loss.
    """

    def __init__(
        self,
        subdomains: List[Subdomain],
        model_factory: Callable[[], nn.Module],
        interface_weight: float = 10.0,
    ):
        super().__init__()
        self.subdomains = subdomains
        self.interface_weight = interface_weight

        # Create one SubdomainPINN per subdomain and register with ModuleList
        self.pinn_list = nn.ModuleList(
            [SubdomainPINN(sd, model_factory()) for sd in subdomains]
        )

    # ------------------------------------------------------------------
    # Partition helper
    # ------------------------------------------------------------------

    @staticmethod
    def partition(
        bounds: List[Tuple[float, float]],
        n_splits: Tuple[int, ...],
        overlap: float = 0.1,
    ) -> List[Subdomain]:
        """Create a regular grid of overlapping subdomains.

        Parameters
        ----------
        bounds   : [(lo0, hi0), (lo1, hi1), ...] — full domain extent.
        n_splits : number of splits along each axis, e.g. (2, 3).
        overlap  : fractional overlap (same for all dimensions).

        Returns
        -------
        List of Subdomain objects covering the domain.
        """
        if len(bounds) != len(n_splits):
            raise ValueError(
                f"len(bounds)={len(bounds)} must equal len(n_splits)={len(n_splits)}"
            )

        # Build tick marks along each axis
        axis_edges: List[List[Tuple[float, float]]] = []
        for (lo, hi), ns in zip(bounds, n_splits):
            width = (hi - lo) / ns
            edges = [(lo + i * width, lo + (i + 1) * width) for i in range(ns)]
            axis_edges.append(edges)

        # Cartesian product of per-axis edge pairs → one subdomain per cell
        subdomains: List[Subdomain] = []
        for cell in itertools.product(*axis_edges):
            subdomains.append(Subdomain(list(cell), overlap=overlap))
        return subdomains

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _subdomain_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Soft partition-of-unity weights for blending overlapping subdomains.

        Returns a tensor of shape [N, K] where K = number of subdomains.
        For points not in any subdomain's extended region, weight is 0.
        """
        N = x.shape[0]
        K = len(self.subdomains)
        weights = torch.zeros(N, K, device=x.device, dtype=x.dtype)

        for k, sd in enumerate(self.subdomains):
            mask = sd.contains(x)  # [N]
            if mask.any():
                # Distance-based weight: product of distance to each face
                w = torch.ones(N, device=x.device, dtype=x.dtype)
                for dim_idx, (lo, hi) in enumerate(sd._ext_bounds):
                    col = x[:, dim_idx]
                    # clamp so that masked-out points get 0 later
                    dist_lo = (col - lo).clamp(min=0.0)
                    dist_hi = (hi - col).clamp(min=0.0)
                    w = w * (dist_lo * dist_hi + 1e-12)
                weights[:, k] = w * mask.float()

        # Normalize so weights sum to 1 per point
        row_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
        weights = weights / row_sum
        return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate PINN across all subdomains (weighted average in overlap).

        Parameters
        ----------
        x : Tensor of shape [N, D].

        Returns
        -------
        Tensor of shape [N, out_dim].
        """
        weights = self._subdomain_weights(x)  # [N, K]
        outputs: Optional[torch.Tensor] = None

        for k, pinn in enumerate(self.pinn_list):
            w_k = weights[:, k]  # [N]
            active = w_k > 0.0
            if not active.any():
                continue
            y_k = pinn(x)  # [N, out_dim]
            weighted_k = y_k * w_k.unsqueeze(1)
            outputs = weighted_k if outputs is None else outputs + weighted_k

        if outputs is None:
            raise RuntimeError("No subdomain covers any input point — check bounds.")
        return outputs

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def interface_loss(self, x_interface: torch.Tensor) -> torch.Tensor:
        """Continuity loss at subdomain interfaces.

        For every pair of overlapping subdomains we compute the MSE between
        their individual predictions at the shared interface points.
        """
        K = len(self.pinn_list)
        total_loss = torch.tensor(0.0, device=x_interface.device, dtype=x_interface.dtype)
        n_pairs = 0

        for i in range(K):
            mask_i = self.subdomains[i].contains(x_interface)
            if not mask_i.any():
                continue
            pred_i = self.pinn_list[i](x_interface[mask_i])

            for j in range(i + 1, K):
                mask_j = self.subdomains[j].contains(x_interface)
                if not mask_j.any():
                    continue

                # Shared points: in both subdomains
                shared = mask_i & mask_j
                if not shared.any():
                    continue

                pred_i_shared = self.pinn_list[i](x_interface[shared])
                pred_j_shared = self.pinn_list[j](x_interface[shared])
                total_loss = total_loss + torch.mean((pred_i_shared - pred_j_shared) ** 2)
                n_pairs += 1

        return total_loss / max(n_pairs, 1)

    def train_step(
        self,
        x_col: torch.Tensor,
        residual_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        x_bc: torch.Tensor,
        bc_fn: Callable[[torch.Tensor], torch.Tensor],
        optimizer: optim.Optimizer,
    ) -> Dict[str, float]:
        """Single training step with physics + BC + interface losses.

        Parameters
        ----------
        x_col       : collocation points, shape [N_col, D].
        residual_fn : callable(model, x) -> residual tensor [N_col, ...].
                      The model passed is the full DoMINO (forward combines all
                      subdomains), but gradients flow through each subnet.
        x_bc        : boundary points, shape [N_bc, D].
        bc_fn       : callable(x_bc) -> target values [N_bc, out_dim] or scalar.
        optimizer   : any torch optimizer wrapping self.parameters().

        Returns
        -------
        dict with scalar loss values (detached).
        """
        optimizer.zero_grad(set_to_none=True)

        # Physics residual
        x_col_r = x_col.detach().requires_grad_(True)
        res = residual_fn(self, x_col_r)
        loss_pde = torch.mean(res ** 2)

        # Boundary condition
        u_bc = self(x_bc)
        bc_target = bc_fn(x_bc)
        if isinstance(bc_target, torch.Tensor):
            loss_bc = torch.mean((u_bc - bc_target) ** 2)
        else:
            loss_bc = torch.mean(u_bc ** 2)  # homogeneous

        # Interface continuity (use collocation pts in overlap zones)
        loss_iface = self.interface_loss(x_col)

        total = loss_pde + loss_bc + self.interface_weight * loss_iface
        total.backward()
        optimizer.step()

        return {
            "total": float(total.detach()),
            "pde": float(loss_pde.detach()),
            "bc": float(loss_bc.detach()),
            "interface": float(loss_iface.detach()),
        }

    # ------------------------------------------------------------------
    # Convenience training loop
    # ------------------------------------------------------------------

    def train_domino(
        self,
        residual_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        bc_fn: Callable[[torch.Tensor], torch.Tensor],
        x_col: torch.Tensor,
        x_bc: torch.Tensor,
        n_epochs: int = 5000,
        lr: float = 1e-3,
        print_every: int = 500,
        optimizer_kwargs: Optional[Dict] = None,
    ) -> Dict[str, List[float]]:
        """Convenience full training loop.

        Parameters
        ----------
        residual_fn : callable(model, x) -> residuals.
        bc_fn       : callable(x_bc) -> BC target tensor.
        x_col       : collocation tensor [N, D].
        x_bc        : boundary tensor [N_bc, D].
        n_epochs    : number of training steps.
        lr          : Adam learning rate.
        print_every : log interval (0 to suppress).

        Returns
        -------
        History dict with lists for 'total', 'pde', 'bc', 'interface'.
        """
        kwargs = optimizer_kwargs or {}
        optimizer = optim.Adam(self.parameters(), lr=lr, **kwargs)

        history: Dict[str, List[float]] = {
            "total": [], "pde": [], "bc": [], "interface": []
        }

        for epoch in range(1, n_epochs + 1):
            losses = self.train_step(x_col, residual_fn, x_bc, bc_fn, optimizer)
            for k, v in losses.items():
                history[k].append(v)

            if print_every > 0 and (epoch % print_every == 0 or epoch == 1):
                print(
                    f"[DoMINO] epoch={epoch:05d}  "
                    f"total={losses['total']:.4e}  "
                    f"pde={losses['pde']:.4e}  "
                    f"bc={losses['bc']:.4e}  "
                    f"iface={losses['interface']:.4e}"
                )

        return history
