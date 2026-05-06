"""Sequential time-marching for stiff/long-horizon PDE problems.

Instead of training over the full time domain at once, divides [0, T] into
mini-windows and trains sequentially, using the previous window's solution as
the IC for the next window.

Reference: Wight & Zhao 2020 "Solving Allen-Cahn and Cahn-Hilliard equations
using the adaptive physics informed neural networks"
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TimeMarchingTrainer:
    """Trains a PINN by marching sequentially through time windows.

    Each window ``[t_k, t_{k+1}]`` gets a fresh copy of the network produced
    by ``model_factory``.  After training window ``k``, the model's prediction
    at ``t_{k+1}`` is frozen and used as the initial condition for window
    ``k+1``.

    Parameters
    ----------
    model_factory : zero-arg callable that returns an untrained ``nn.Module``.
    t_start       : start of the time domain.
    t_end         : end of the time domain.
    n_windows     : number of time windows to split [t_start, t_end] into.
    epochs_per_window : training epochs per window.
    n_col         : collocation points sampled per window (in space × time).
    ic_weight     : loss weight applied to the initial-condition term.
    optimizer_kwargs : extra kwargs forwarded to ``torch.optim.Adam``.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        t_start: float,
        t_end: float,
        n_windows: int = 10,
        epochs_per_window: int = 2000,
        n_col: int = 1000,
        ic_weight: float = 10.0,
        optimizer_kwargs: Optional[Dict] = None,
    ):
        self.model_factory = model_factory
        self.t_start = float(t_start)
        self.t_end = float(t_end)
        self.n_windows = n_windows
        self.epochs_per_window = epochs_per_window
        self.n_col = n_col
        self.ic_weight = ic_weight
        self.optimizer_kwargs = optimizer_kwargs or {}

        # Computed after march()
        self.models: List[nn.Module] = []
        self.window_edges: List[Tuple[float, float]] = []
        self._t_breaks: List[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_window_edges(self) -> List[Tuple[float, float]]:
        """Uniform partition of [t_start, t_end] into n_windows intervals."""
        breaks = np.linspace(self.t_start, self.t_end, self.n_windows + 1)
        return [(float(breaks[i]), float(breaks[i + 1])) for i in range(self.n_windows)]

    def _sample_collocation(
        self,
        t_lo: float,
        t_hi: float,
        x_domain: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        seed: int = 0,
    ) -> torch.Tensor:
        """Sample collocation points inside a time window.

        If ``x_domain`` is provided it is used as the spatial grid;
        otherwise spatial coordinates are sampled uniformly in [0, 1].
        """
        rng = np.random.default_rng(seed)

        if x_domain is not None:
            x_np = x_domain.cpu().numpy()
            n_x = x_np.shape[0]
            # Repeat spatial points with random t in window
            idx = rng.integers(0, n_x, size=self.n_col)
            x_col = x_np[idx]
            t_col = rng.uniform(t_lo, t_hi, size=(self.n_col, 1)).astype(np.float32)
        else:
            # 1-D spatial fallback
            x_col = rng.uniform(0.0, 1.0, size=(self.n_col, 1)).astype(np.float32)
            t_col = rng.uniform(t_lo, t_hi, size=(self.n_col, 1)).astype(np.float32)

        xt = np.concatenate([x_col, t_col], axis=1)
        return torch.from_numpy(xt).to(device=device, dtype=dtype)

    def _sample_ic_points(
        self,
        t_ic: float,
        x_domain: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        n_ic: int = 256,
        seed: int = 0,
    ) -> torch.Tensor:
        """Sample spatial points at t = t_ic for IC enforcement."""
        rng = np.random.default_rng(seed + 1000)
        if x_domain is not None:
            x_np = x_domain.cpu().numpy()
            idx = rng.integers(0, x_np.shape[0], size=n_ic)
            x_ic_pts = x_np[idx]
        else:
            x_ic_pts = rng.uniform(0.0, 1.0, size=(n_ic, 1)).astype(np.float32)

        t_ic_col = np.full((n_ic, 1), t_ic, dtype=np.float32)
        xt_ic = np.concatenate([x_ic_pts, t_ic_col], axis=1)
        return torch.from_numpy(xt_ic).to(device=device, dtype=dtype)

    def _train_window(
        self,
        model: nn.Module,
        t_lo: float,
        t_hi: float,
        pde_residual_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        ic_target_fn: Callable[[torch.Tensor], torch.Tensor],
        bc_fns: Optional[List[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]],
        x_domain: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        window_idx: int,
    ) -> Dict[str, List[float]]:
        """Train a single model for time window [t_lo, t_hi]."""
        optimizer = optim.Adam(model.parameters(), lr=1e-3, **self.optimizer_kwargs)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs_per_window
        )

        x_col = self._sample_collocation(
            t_lo, t_hi, x_domain, device, dtype, seed=window_idx * 17
        )
        x_ic = self._sample_ic_points(
            t_lo, x_domain, device, dtype, n_ic=max(256, self.n_col // 4),
            seed=window_idx * 17
        )

        history: Dict[str, List[float]] = {"total": [], "pde": [], "ic": []}
        print_every = max(1, self.epochs_per_window // 4)

        for ep in range(1, self.epochs_per_window + 1):
            optimizer.zero_grad(set_to_none=True)

            # PDE residual
            x_col_r = x_col.clone().requires_grad_(True)
            res = pde_residual_fn(model, x_col_r)
            loss_pde = torch.mean(res ** 2)

            # IC
            u_ic_pred = model(x_ic)
            u_ic_true = ic_target_fn(x_ic)
            loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

            # BC (optional)
            loss_bc = torch.tensor(0.0, device=device, dtype=dtype)
            if bc_fns:
                for bc_fn in bc_fns:
                    x_bc_pts, u_bc_true = bc_fn(t_lo, t_hi, device, dtype)
                    u_bc_pred = model(x_bc_pts)
                    loss_bc = loss_bc + torch.mean((u_bc_pred - u_bc_true) ** 2)

            total = loss_pde + self.ic_weight * loss_ic + loss_bc
            total.backward()
            optimizer.step()
            scheduler.step()

            history["total"].append(float(total.detach()))
            history["pde"].append(float(loss_pde.detach()))
            history["ic"].append(float(loss_ic.detach()))

            if ep % print_every == 0 or ep == 1:
                print(
                    f"  [Window {window_idx + 1}/{self.n_windows}] "
                    f"ep={ep:04d}  total={float(total.detach()):.4e}  "
                    f"pde={float(loss_pde.detach()):.4e}  "
                    f"ic={float(loss_ic.detach()):.4e}"
                )

        return history

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def march(
        self,
        pde_residual_fn: Callable[[nn.Module, torch.Tensor], torch.Tensor],
        ic_fn: Callable[[torch.Tensor], torch.Tensor],
        bc_fns: Optional[List[Callable]] = None,
        x_domain: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ) -> List[nn.Module]:
        """Execute time-marching. Returns list of trained models, one per window.

        Parameters
        ----------
        pde_residual_fn : callable(model, x_col) -> residual tensor [N, ...].
                          ``x_col`` has ``requires_grad=True``.
        ic_fn           : callable(x) -> target IC values [N, out_dim].
                          For window 0 this is the true IC; for later windows
                          it is replaced by the previous model's prediction.
        bc_fns          : list of callables(t_lo, t_hi, device, dtype) ->
                          (x_bc [N_bc, D], u_bc [N_bc, out_dim]).
        x_domain        : optional spatial collocation grid [N_x, D_x].
                          If None, 1-D [0,1] is assumed.
        device          : torch device (defaults to CUDA if available).
        dtype           : float dtype (default float32).

        Returns
        -------
        List of trained nn.Module, one per window.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.window_edges = self._make_window_edges()
        self._t_breaks = [lo for lo, _ in self.window_edges] + [self.window_edges[-1][1]]
        self.models = []

        current_ic_fn = ic_fn  # will be overridden after each window

        for w_idx, (t_lo, t_hi) in enumerate(self.window_edges):
            print(f"\n[TimeMarch] Window {w_idx + 1}/{self.n_windows}: t in [{t_lo:.4f}, {t_hi:.4f}]")

            model = self.model_factory().to(device=device, dtype=dtype)
            model.train()

            self._train_window(
                model=model,
                t_lo=t_lo,
                t_hi=t_hi,
                pde_residual_fn=pde_residual_fn,
                ic_target_fn=current_ic_fn,
                bc_fns=bc_fns,
                x_domain=x_domain,
                device=device,
                dtype=dtype,
                window_idx=w_idx,
            )

            model.eval()
            self.models.append(model)

            # Next window's IC = current model's prediction at t = t_hi
            _prev_model = model
            _t_hi_val = t_hi

            def _make_next_ic(prev_model: nn.Module, t_next: float) -> Callable:
                def next_ic_fn(x: torch.Tensor) -> torch.Tensor:
                    # x has shape [N, D]; replace time column with t_next
                    x_eval = x.clone()
                    x_eval[:, -1] = t_next  # last column = time
                    with torch.no_grad():
                        return prev_model(x_eval)
                return next_ic_fn

            current_ic_fn = _make_next_ic(_prev_model, _t_hi_val)

        return self.models

    def evaluate(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the full solution by routing (x, t) to the correct window.

        Parameters
        ----------
        x : spatial points [N, D_x].
        t : time values [N, 1] or [N].

        Returns
        -------
        u : [N, out_dim].
        """
        if not self.models:
            raise RuntimeError("Call march() before evaluate().")

        t_flat = t.flatten()
        N = x.shape[0]
        device = x.device
        dtype = x.dtype

        # Build input xt = [x | t]
        xt = torch.cat([x, t_flat.unsqueeze(1)], dim=1)
        out = torch.zeros(N, 1, device=device, dtype=dtype)

        for w_idx, (t_lo, t_hi) in enumerate(self.window_edges):
            is_last = w_idx == len(self.window_edges) - 1
            if is_last:
                mask = (t_flat >= t_lo) & (t_flat <= t_hi)
            else:
                mask = (t_flat >= t_lo) & (t_flat < t_hi)
            if not mask.any():
                continue
            with torch.no_grad():
                pred = self.models[w_idx](xt[mask])
            # Dynamically resize out if needed
            if out.shape[1] != pred.shape[1]:
                out = torch.zeros(N, pred.shape[1], device=device, dtype=dtype)
            out[mask] = pred

        return out

    @staticmethod
    def plot_spacetime(
        models: List[nn.Module],
        window_edges: List[Tuple[float, float]],
        x_range: Tuple[float, float] = (0, 1),
        t_range: Tuple[float, float] = (0, 1),
        n: int = 100,
    ) -> None:
        """Plot full space-time solution across all windows.

        Requires matplotlib. If not available, prints a warning and returns.

        Parameters
        ----------
        models       : list of trained models (one per window).
        window_edges : list of (t_lo, t_hi) tuples.
        x_range      : (x_min, x_max) for the spatial grid.
        t_range      : (t_min, t_max) for the time grid.
        n            : grid resolution per axis.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[TimeMarchingTrainer.plot_spacetime] matplotlib not available.")
            return

        device = next(models[0].parameters()).device
        dtype = next(models[0].parameters()).dtype

        xs = torch.linspace(x_range[0], x_range[1], n, device=device, dtype=dtype)
        ts = torch.linspace(t_range[0], t_range[1], n, device=device, dtype=dtype)
        XX, TT = torch.meshgrid(xs, ts, indexing="ij")
        x_flat = XX.reshape(-1, 1)
        t_flat = TT.reshape(-1)
        xt = torch.cat([x_flat, t_flat.unsqueeze(1)], dim=1)

        u_out = torch.zeros(n * n, 1, device=device, dtype=dtype)

        for w_idx, (t_lo, t_hi) in enumerate(window_edges):
            is_last = w_idx == len(window_edges) - 1
            if is_last:
                mask = (t_flat >= t_lo) & (t_flat <= t_hi)
            else:
                mask = (t_flat >= t_lo) & (t_flat < t_hi)
            if not mask.any():
                continue
            with torch.no_grad():
                pred = models[w_idx](xt[mask])
            if u_out.shape[1] < pred.shape[1]:
                u_out = u_out.expand(-1, pred.shape[1]).clone()
            u_out[mask] = pred

        u_grid = u_out[:, 0].reshape(n, n).cpu().numpy()
        x_np = xs.cpu().numpy()
        t_np = ts.cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 5))
        cmap = ax.contourf(t_np, x_np, u_grid, levels=50, cmap="viridis")
        fig.colorbar(cmap, ax=ax)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        ax.set_title("Time-marching PINN: full space-time solution")

        # Draw window boundaries
        for _, t_hi in window_edges[:-1]:
            ax.axvline(x=t_hi, color="white", linestyle="--", linewidth=0.8, alpha=0.6)

        plt.tight_layout()
        plt.show()
