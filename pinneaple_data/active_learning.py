"""Active Learning for adaptive collocation in PINN training.

Strategies:
- ResidualBasedAL: sample where PDE residual is highest
- VarianceBasedAL: sample where model output variance is highest (needs UQ)
- ExpectedImprovementAL: Bayesian-style expected improvement
- CombinedAL: weighted combination of strategies
- RAR (Residual-based Adaptive Refinement): the classic PINN method
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ActiveLearningConfig:
    strategy: str = "residual"         # "residual" | "variance" | "combined" | "rar"
    n_candidates: int = 50_000         # candidate pool size
    n_select: int = 1_000              # points to add per AL iteration
    n_initial: int = 5_000             # initial collocation points
    n_iterations: int = 5              # number of AL rounds
    exploitation_weight: float = 0.7   # weight for high-residual vs exploration
    seed: int = 0


# ---------------------------------------------------------------------------
# Internal helpers (mirror collocation.py style)
# ---------------------------------------------------------------------------

def _lhs(bounds_arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Latin Hypercube Sampling — D-dimensional, returns float32."""
    d = bounds_arr.shape[0]
    lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
    result = np.zeros((n, d), dtype=np.float32)
    for j in range(d):
        perm = rng.permutation(n)
        result[:, j] = (lo[j] + (perm + rng.random(n)) / n * (hi[j] - lo[j])).astype(np.float32)
    return result


def _uniform(bounds_arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    d = bounds_arr.shape[0]
    lo, hi = bounds_arr[:, 0], bounds_arr[:, 1]
    return (lo + rng.random((n, d)) * (hi - lo)).astype(np.float32)


def _bounds_to_arr(bounds: Dict[str, Tuple[float, float]]) -> np.ndarray:
    return np.array([[lo, hi] for lo, hi in bounds.values()], dtype=np.float64)


# ---------------------------------------------------------------------------
# ResidualBasedAL
# ---------------------------------------------------------------------------

class ResidualBasedAL:
    """Sample collocation points where PDE residual is highest (RAR/RAD).

    Supports two selection modes:

    - ``"top_k"``: take the top-k highest residual points (RAR — Residual-based
      Adaptive Refinement).
    - ``"weighted"``: sample proportional to |residual|^k, k=1 (RAD — Residual-based
      Adaptive Distribution).

    Parameters
    ----------
    config:
        :class:`ActiveLearningConfig` controlling pool size, selection count, etc.
    bounds:
        Coordinate bounds ``{name: (lo, hi)}``.  The same format used by
        :class:`~pinneaple_data.collocation.CollocationSampler`.
    """

    def __init__(
        self,
        config: ActiveLearningConfig,
        bounds: Dict[str, Tuple[float, float]],
    ) -> None:
        self.config = config
        self.bounds = bounds
        self._bounds_arr = _bounds_to_arr(bounds)
        self._coord_names: List[str] = list(bounds.keys())
        self._rng = np.random.default_rng(config.seed)

        # Candidate pool — populated lazily / on demand
        self._pool: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_pool(
        self,
        model=None,           # unused, kept for API symmetry
        coord_names=None,     # unused — we already have self._coord_names
        n_candidates: Optional[int] = None,
    ) -> None:
        """Refresh the candidate pool using LHS sampling from bounds."""
        n = n_candidates or self.config.n_candidates
        self._pool = _lhs(self._bounds_arr, n, self._rng)

    def select(
        self,
        residual_fn: Callable[[np.ndarray], np.ndarray],
        n_select: Optional[int] = None,
        mode: str = "weighted",  # "top_k" | "weighted"
    ) -> np.ndarray:
        """Select new collocation points based on PDE residuals.

        Parameters
        ----------
        residual_fn:
            Callable ``(N, D) -> (N,)`` returning per-point residual magnitudes.
            It should return ``|residual|`` (non-negative).
        n_select:
            How many new points to return.  Defaults to ``config.n_select``.
        mode:
            ``"weighted"`` for RAD (sample proportional to |r|); ``"top_k"``
            for RAR (take the highest-residual points directly).

        Returns
        -------
        np.ndarray
            Shape ``(n_select, D)``, dtype ``float32``.
        """
        k = n_select or self.config.n_select

        # Ensure we have a candidate pool
        if self._pool is None:
            self.update_pool()

        candidates = self._pool  # (N_cand, D)

        # Evaluate residuals
        res = np.abs(np.asarray(residual_fn(candidates), dtype=np.float64)).ravel()
        if res.shape[0] != candidates.shape[0]:
            raise ValueError(
                f"residual_fn returned {res.shape[0]} values for "
                f"{candidates.shape[0]} candidates."
            )

        if mode == "top_k":
            idx = np.argsort(res)[::-1][:k]
        elif mode == "weighted":
            # RAD: sample ∝ |r|  (shift so minimum weight is 1e-10)
            weights = res - res.min() + 1e-10
            probs = weights / weights.sum()
            replace = k > len(candidates)
            idx = self._rng.choice(len(candidates), size=k, replace=replace, p=probs)
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'top_k' or 'weighted'.")

        selected = candidates[idx].astype(np.float32)

        # Refresh pool for next call
        self.update_pool()

        return selected


# ---------------------------------------------------------------------------
# VarianceBasedAL
# ---------------------------------------------------------------------------

class VarianceBasedAL:
    """Sample where model output variance is highest (requires a UQ model).

    This is useful when the PINN carries uncertainty estimates — e.g. a
    Bayesian neural network or an ensemble — that expose a ``variance_fn``
    interface.

    Parameters
    ----------
    config:
        :class:`ActiveLearningConfig`.
    bounds:
        Coordinate bounds ``{name: (lo, hi)}``.
    """

    def __init__(
        self,
        config: ActiveLearningConfig,
        bounds: Dict[str, Tuple[float, float]],
    ) -> None:
        self.config = config
        self.bounds = bounds
        self._bounds_arr = _bounds_to_arr(bounds)
        self._rng = np.random.default_rng(config.seed + 1)

    def select(
        self,
        variance_fn: Callable[[np.ndarray], np.ndarray],
        n_select: Optional[int] = None,
    ) -> np.ndarray:
        """Select points where output variance is highest.

        Parameters
        ----------
        variance_fn:
            Callable ``(N, D) -> (N,)`` returning per-point variance estimates.
        n_select:
            Number of points to return.  Defaults to ``config.n_select``.

        Returns
        -------
        np.ndarray
            Shape ``(n_select, D)``, dtype ``float32``.
        """
        k = n_select or self.config.n_select
        n_cand = self.config.n_candidates

        candidates = _lhs(self._bounds_arr, n_cand, self._rng)
        var = np.abs(np.asarray(variance_fn(candidates), dtype=np.float64)).ravel()

        if var.shape[0] != candidates.shape[0]:
            raise ValueError(
                f"variance_fn returned {var.shape[0]} values for "
                f"{candidates.shape[0]} candidates."
            )

        # Sample proportional to variance
        weights = var - var.min() + 1e-10
        probs = weights / weights.sum()
        replace = k > n_cand
        idx = self._rng.choice(n_cand, size=k, replace=replace, p=probs)
        return candidates[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# CombinedAL
# ---------------------------------------------------------------------------

class CombinedAL:
    """Weighted combination: exploit high residual + explore uncertain regions.

    Scores are computed as::

        score = w_res * r̂ + w_var * v̂ + w_div * d̂

    where ``r̂``, ``v̂``, ``d̂`` are normalised residual, variance, and
    diversity (distance to nearest already-selected point) scores.

    Parameters
    ----------
    config:
        :class:`ActiveLearningConfig`.
    bounds:
        Coordinate bounds.
    residual_weight, variance_weight, diversity_weight:
        Mixing coefficients (need not sum to 1 — they are re-normalised).
    """

    def __init__(
        self,
        config: ActiveLearningConfig,
        bounds: Dict[str, Tuple[float, float]],
        residual_weight: float = 0.6,
        variance_weight: float = 0.3,
        diversity_weight: float = 0.1,
    ) -> None:
        self.config = config
        self.bounds = bounds
        self._bounds_arr = _bounds_to_arr(bounds)
        self._rng = np.random.default_rng(config.seed + 2)

        total = residual_weight + variance_weight + diversity_weight
        if total <= 0:
            raise ValueError("At least one of the weights must be positive.")
        self.w_res = residual_weight / total
        self.w_var = variance_weight / total
        self.w_div = diversity_weight / total

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        """Min-max normalise to [0, 1], returning 1/N if all equal."""
        span = arr.max() - arr.min()
        if span < 1e-12:
            return np.full_like(arr, 1.0 / len(arr))
        return (arr - arr.min()) / span

    @staticmethod
    def _diversity_scores(candidates: np.ndarray) -> np.ndarray:
        """Approximate diversity: distance of each point to the nearest other point.

        Uses a random-sample approximation (O(N)) rather than full pairwise
        O(N²) computation.  The farthest points from each other get highest
        scores.
        """
        # Subsample reference set to keep cost low
        n = len(candidates)
        ref_size = min(n, 2048)
        ref_idx = np.arange(n) if n <= ref_size else np.random.choice(n, ref_size, replace=False)
        ref = candidates[ref_idx]

        # For each candidate, find distance to nearest reference point
        # Compute in chunks to avoid huge memory allocations
        chunk = 4096
        min_dists = np.empty(n, dtype=np.float64)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            diff = candidates[start:end, None, :] - ref[None, :, :]  # (B, R, D)
            dists = np.sqrt((diff ** 2).sum(axis=-1))                # (B, R)
            min_dists[start:end] = dists.min(axis=-1)

        return min_dists

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        residual_fn: Callable,
        variance_fn: Optional[Callable] = None,
        n_select: Optional[int] = None,
    ) -> np.ndarray:
        """Select new collocation points via the combined score.

        Parameters
        ----------
        residual_fn:
            Callable ``(N, D) -> (N,)`` returning per-point |residual|.
        variance_fn:
            Optional callable ``(N, D) -> (N,)`` returning variance estimates.
            If ``None``, the variance component is zeroed out and the weights
            are redistributed between residual and diversity.
        n_select:
            Number of points to return.  Defaults to ``config.n_select``.

        Returns
        -------
        np.ndarray
            Shape ``(n_select, D)``, dtype ``float32``.
        """
        k = n_select or self.config.n_select
        n_cand = self.config.n_candidates

        candidates = _lhs(self._bounds_arr, n_cand, self._rng)

        # --- Residual score --------------------------------------------------
        res_raw = np.abs(np.asarray(residual_fn(candidates), dtype=np.float64)).ravel()
        res_score = self._normalise(res_raw)

        # --- Variance score --------------------------------------------------
        if variance_fn is not None:
            var_raw = np.abs(np.asarray(variance_fn(candidates), dtype=np.float64)).ravel()
            var_score = self._normalise(var_raw)
            w_res, w_var = self.w_res, self.w_var
        else:
            var_score = np.zeros(n_cand, dtype=np.float64)
            # Redistribute variance weight to residual
            w_res = self.w_res + self.w_var
            w_var = 0.0

        # --- Diversity score -------------------------------------------------
        div_score = self._normalise(self._diversity_scores(candidates))

        # --- Combined score --------------------------------------------------
        score = w_res * res_score + w_var * var_score + self.w_div * div_score

        # Sample proportional to combined score
        score = score - score.min() + 1e-10
        probs = score / score.sum()
        replace = k > n_cand
        idx = self._rng.choice(n_cand, size=k, replace=replace, p=probs)
        return candidates[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# AdaptiveCollocationTrainer
# ---------------------------------------------------------------------------

class AdaptiveCollocationTrainer:
    """Wraps a PINN model with active-learning collocation updates.

    The trainer holds its own collocation set, expanding it with new points
    selected by the chosen AL strategy every ``al_every`` epochs.

    Usage::

        trainer = AdaptiveCollocationTrainer(
            model=pinn,
            physics_fn=burgers_loss,
            sampler=CollocationSampler.from_problem_spec(spec),
            al_config=ActiveLearningConfig(strategy="residual"),
        )
        history = trainer.train(n_epochs=2000, al_every=500)

    Parameters
    ----------
    model:
        A PyTorch ``nn.Module`` (or anything callable that takes a float32
        tensor and returns predictions).
    physics_fn:
        Callable ``(model, x_col) -> (loss, residuals)`` where ``residuals``
        is a 1-D array / tensor of per-point residual magnitudes.  May also
        return just a scalar loss (in which case residuals are approximated
        via the output norm).
    sampler:
        A :class:`~pinneaple_data.collocation.CollocationSampler` instance
        used to generate initial boundary/IC points.
    al_config:
        :class:`ActiveLearningConfig`.
    optimizer:
        Optional pre-built PyTorch optimiser.  If ``None``, Adam with
        ``lr=1e-3`` is created during :meth:`train`.
    device:
        ``"cpu"`` or ``"cuda"``.
    """

    def __init__(
        self,
        model,
        physics_fn: Callable,
        sampler,                              # CollocationSampler
        al_config: Optional[ActiveLearningConfig] = None,
        optimizer=None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.physics_fn = physics_fn
        self.sampler = sampler
        self.al_config = al_config or ActiveLearningConfig()
        self.optimizer = optimizer
        self.device = device

        # Determine bounds from sampler
        self._bounds: Dict[str, Tuple[float, float]] = dict(sampler.bounds)
        self._bounds_arr = _bounds_to_arr(self._bounds)

        # Build the strategy object
        cfg = self.al_config
        if cfg.strategy in ("residual", "rar"):
            self._al_strategy = ResidualBasedAL(cfg, self._bounds)
        elif cfg.strategy == "variance":
            self._al_strategy = VarianceBasedAL(cfg, self._bounds)
        elif cfg.strategy == "combined":
            self._al_strategy = CombinedAL(
                cfg,
                self._bounds,
                residual_weight=cfg.exploitation_weight,
                variance_weight=max(0.0, 1.0 - cfg.exploitation_weight - 0.1),
                diversity_weight=0.1,
            )
        else:
            raise ValueError(
                f"Unknown AL strategy '{cfg.strategy}'. "
                "Choose from: 'residual', 'rar', 'variance', 'combined'."
            )

        # Current collocation set (set during train())
        self._x_col: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_residuals(self, x_col: np.ndarray) -> np.ndarray:
        """Run model + physics_fn to get per-point residual magnitudes.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` — absolute residuals, float64.
        """
        import torch

        x_t = torch.from_numpy(x_col.astype(np.float32)).to(self.device)
        x_t.requires_grad_(True)

        with torch.enable_grad():
            try:
                result = self.physics_fn(self.model, x_t)
                if isinstance(result, (tuple, list)) and len(result) >= 2:
                    _, residuals = result[0], result[1]
                    if isinstance(residuals, torch.Tensor):
                        res_np = residuals.detach().cpu().numpy().ravel()
                    elif isinstance(residuals, np.ndarray):
                        res_np = residuals.ravel()
                    else:
                        # dict of named residuals — concatenate
                        parts = []
                        for v in residuals.values():
                            if isinstance(v, torch.Tensor):
                                parts.append(v.detach().cpu().numpy().ravel())
                        res_np = np.concatenate(parts) if parts else np.ones(len(x_col))
                else:
                    # physics_fn returned only a scalar loss — fall back to
                    # output-norm heuristic
                    out = self.model(x_t)
                    if isinstance(out, torch.Tensor):
                        res_np = out.norm(dim=-1).detach().cpu().numpy().ravel()
                    else:
                        res_np = np.ones(len(x_col), dtype=np.float64)
            except Exception as exc:
                warnings.warn(
                    f"AdaptiveCollocationTrainer._compute_residuals: "
                    f"physics_fn raised {type(exc).__name__}: {exc}. "
                    "Falling back to uniform residuals.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                res_np = np.ones(len(x_col), dtype=np.float64)

        return np.abs(res_np).astype(np.float64)

    def _refine_collocation(self) -> None:
        """Select new collocation points and append them to the current set."""
        if self._x_col is None:
            return

        residual_fn = self._compute_residuals

        strategy = self._al_strategy
        if isinstance(strategy, ResidualBasedAL):
            mode = "top_k" if self.al_config.strategy == "rar" else "weighted"
            new_pts = strategy.select(residual_fn, mode=mode)
        elif isinstance(strategy, VarianceBasedAL):
            # Variance-based: use model output variance as proxy
            def _var_fn(x: np.ndarray) -> np.ndarray:
                return self._compute_residuals(x)  # residual as variance proxy
            new_pts = strategy.select(_var_fn)
        elif isinstance(strategy, CombinedAL):
            new_pts = strategy.select(residual_fn)
        else:
            return

        self._x_col = np.concatenate([self._x_col, new_pts], axis=0)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        n_epochs: int = 2000,
        al_every: int = 500,    # refine collocation every N epochs
        n_col: int = 4096,
        n_bc: int = 512,
        lr: float = 1e-3,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Train the PINN with periodic active-learning collocation refinement.

        Parameters
        ----------
        n_epochs:
            Total number of training epochs.
        al_every:
            Perform an AL refinement step every this many epochs.
        n_col:
            Number of initial interior collocation points.
        n_bc:
            Number of boundary condition points (resampled each epoch).
        lr:
            Learning rate (used only if no external optimiser was provided).
        verbose:
            Print progress to stdout.

        Returns
        -------
        dict
            ``{"loss_history": [...], "n_collocation_history": [...], "al_events": [...]}``
        """
        import torch

        # --- Optimiser -------------------------------------------------------
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # --- Initial collocation set -----------------------------------------
        rng = np.random.default_rng(self.al_config.seed)
        n_initial = max(n_col, self.al_config.n_initial)
        self._x_col = _lhs(self._bounds_arr, n_initial, rng)

        # --- Tracking --------------------------------------------------------
        loss_history: List[float] = []
        n_col_history: List[int] = []
        al_events: List[Dict[str, Any]] = []

        model = self.model
        model.to(self.device)
        optim = self.optimizer

        for epoch in range(1, n_epochs + 1):
            model.train()
            optim.zero_grad()

            x_t = torch.from_numpy(self._x_col.astype(np.float32)).to(self.device)
            x_t.requires_grad_(True)

            # Physics (PDE) loss
            try:
                result = self.physics_fn(model, x_t)
                if isinstance(result, (tuple, list)):
                    physics_loss = result[0]
                    if not isinstance(physics_loss, torch.Tensor):
                        physics_loss = torch.tensor(float(physics_loss), device=self.device, requires_grad=True)
                else:
                    physics_loss = result
                    if not isinstance(physics_loss, torch.Tensor):
                        physics_loss = torch.tensor(float(physics_loss), device=self.device, requires_grad=True)
            except Exception as exc:
                warnings.warn(
                    f"Epoch {epoch}: physics_fn error — {exc}. Loss set to 0.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                physics_loss = torch.zeros(1, device=self.device, requires_grad=True)

            # BC loss (sample each epoch for randomness)
            try:
                batch = self.sampler.sample(n_col=0, n_bc=n_bc, seed=epoch)
                x_bc_np = batch.get("x_bc", np.zeros((0, self._x_col.shape[1]), dtype=np.float32))
                y_bc_np = batch.get("y_bc", np.zeros((0, 1), dtype=np.float32))

                if len(x_bc_np) > 0:
                    x_bc = torch.from_numpy(x_bc_np).to(self.device)
                    y_bc = torch.from_numpy(y_bc_np).to(self.device)
                    pred_bc = model(x_bc)
                    if isinstance(pred_bc, torch.Tensor) and pred_bc.shape == y_bc.shape:
                        bc_loss = torch.mean((pred_bc - y_bc) ** 2)
                    else:
                        bc_loss = torch.zeros(1, device=self.device)
                else:
                    bc_loss = torch.zeros(1, device=self.device)
            except Exception:
                bc_loss = torch.zeros(1, device=self.device)

            total_loss = physics_loss + bc_loss
            total_loss.backward()
            optim.step()

            loss_val = float(total_loss.detach().cpu().item())
            loss_history.append(loss_val)
            n_col_history.append(len(self._x_col))

            # --- AL refinement -----------------------------------------------
            if epoch % al_every == 0 and epoch < n_epochs:
                prev_n = len(self._x_col)
                model.eval()
                with torch.no_grad():
                    self._refine_collocation()
                model.train()
                new_n = len(self._x_col)
                event = {"epoch": epoch, "n_before": prev_n, "n_after": new_n, "loss": loss_val}
                al_events.append(event)
                if verbose:
                    print(
                        f"[AL] epoch={epoch:>5d}  "
                        f"collocation points: {prev_n} -> {new_n}  "
                        f"loss={loss_val:.4e}"
                    )

            if verbose and epoch % max(1, n_epochs // 10) == 0:
                print(f"  epoch={epoch:>5d}  loss={loss_val:.4e}  n_col={len(self._x_col)}")

        return {
            "loss_history": loss_history,
            "n_collocation_history": n_col_history,
            "al_events": al_events,
        }
