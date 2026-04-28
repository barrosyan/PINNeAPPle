from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DesignOptimizerConfig:
    """Unified configuration for all design optimizer back-ends.

    Attributes
    ----------
    method:
        Optimization engine: ``"gradient"``, ``"bayesian"``, or
        ``"evolutionary"``.
    n_iters:
        Total number of optimization iterations.
    lr:
        Learning rate (used by the gradient optimizer).
    grad_optimizer:
        Sub-method for gradient optimization: ``"adam"`` or ``"lbfgs"``.
    grad_clip:
        Gradient norm clipping threshold (0 = disabled).
    n_initial_random:
        Number of random evaluations before Bayesian optimization starts.
    acquisition:
        Acquisition function: ``"ei"`` (expected improvement) or ``"ucb"``.
    xi:
        Exploration parameter for the EI acquisition.
    kappa:
        UCB trade-off coefficient.
    population_size:
        Evolutionary optimizer population size.
    sigma0:
        CMA-ES initial step size.
    mutation_std:
        Standard deviation for GA mutation (fallback when ``cma`` is absent).
    crossover_rate:
        Probability of crossover for GA fallback.
    """

    method: str = "gradient"
    n_iters: int = 100
    lr: float = 1e-3
    # Gradient
    grad_optimizer: str = "adam"
    grad_clip: float = 1.0
    # Bayesian
    n_initial_random: int = 10
    acquisition: str = "ei"
    xi: float = 0.01
    kappa: float = 2.0
    # Evolutionary
    population_size: int = 30
    sigma0: float = 0.3
    mutation_std: float = 0.1
    crossover_rate: float = 0.7


# ---------------------------------------------------------------------------
# Gradient optimizer
# ---------------------------------------------------------------------------


class GradientDesignOptimizer:
    """Single-step gradient-based design optimizer.

    Maintains ``theta`` as an ``nn.Parameter`` and applies one Adam or
    L-BFGS update per call to :meth:`step`.

    Parameters
    ----------
    theta_init:
        Initial design-parameter vector (1-D numpy array).
    cfg:
        Optimizer configuration.
    """

    def __init__(self, theta_init: np.ndarray, cfg: DesignOptimizerConfig) -> None:
        import torch
        import torch.nn as nn

        self.cfg = cfg
        self._theta = nn.Parameter(
            torch.from_numpy(theta_init.astype(np.float32)).clone()
        )

        if cfg.grad_optimizer == "lbfgs":
            self._opt = torch.optim.LBFGS(
                [self._theta],
                lr=cfg.lr,
                max_iter=20,
                line_search_fn="strong_wolfe",
            )
        else:
            self._opt = torch.optim.Adam([self._theta], lr=cfg.lr)

    # ------------------------------------------------------------------

    def step(
        self,
        surrogate: object,
        objective: object,
        constraints: Optional[object],
        theta_np: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Perform one gradient update and return the new theta and loss value.

        Parameters
        ----------
        surrogate:
            :class:`~pinneaple_design_opt.surrogate.PhysicsSurrogate` instance.
        objective:
            :class:`~pinneaple_design_opt.objective.ObjectiveBase` instance.
        constraints:
            Optional :class:`~pinneaple_design_opt.constraints.ConstraintSet`.
        theta_np:
            Current parameter vector; used to warm-start ``self._theta``
            when called from the pipeline.

        Returns
        -------
        tuple
            ``(new_theta_np, objective_value)``
        """
        import torch

        with torch.no_grad():
            self._theta.copy_(torch.from_numpy(theta_np.astype(np.float32)))

        cfg = self.cfg

        def _closure() -> "torch.Tensor":
            self._opt.zero_grad()
            u = surrogate.model(self._theta.unsqueeze(0)).squeeze(0)
            loss = objective(self._theta, u)
            if constraints is not None:
                loss = loss + constraints.total_penalty(self._theta, u)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([self._theta], cfg.grad_clip)
            return loss

        if isinstance(self._opt, torch.optim.LBFGS):
            loss_val = self._opt.step(_closure)
        else:
            loss_val = _closure()
            self._opt.step()

        new_theta = self._theta.detach().cpu().numpy().copy()
        obj_val = float(loss_val.item()) if loss_val is not None else float("nan")
        return new_theta, obj_val


# ---------------------------------------------------------------------------
# Pure-numpy squared-exponential GP (no sklearn required)
# ---------------------------------------------------------------------------


class _NumpyGP:
    """Minimal squared-exponential Gaussian process regressor.

    Uses a scalar length-scale estimated from the data range and a nugget
    for numerical stability.  Sufficient for acquisition-function evaluation
    when sklearn is unavailable.
    """

    def __init__(self, noise: float = 1e-4) -> None:
        self.noise = noise
        self._X: Optional[np.ndarray] = None
        self._alpha: Optional[np.ndarray] = None
        self._length_scale: float = 1.0

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # Squared-exponential kernel; broadcast over rows.
        diff = A[:, None, :] - B[None, :, :]          # (n, m, d)
        sq_dist = np.sum(diff ** 2, axis=-1)           # (n, m)
        return np.exp(-0.5 * sq_dist / self._length_scale ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape
        # Median heuristic for the length scale.
        if n > 1:
            diffs = X[:, None, :] - X[None, :, :]
            sq_dists = np.sum(diffs ** 2, axis=-1)
            median_sq = float(np.median(sq_dists[sq_dists > 0])) if np.any(sq_dists > 0) else 1.0
            self._length_scale = max(math.sqrt(median_sq / 2.0), 1e-3)

        self._X = X.copy()
        K = self._kernel(X, X) + self.noise * np.eye(n)

        # Cholesky solve for numerical stability.
        try:
            L = np.linalg.cholesky(K)
            self._alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            self._L = L
        except np.linalg.LinAlgError:
            # Fall back to direct solve when Cholesky fails.
            self._alpha = np.linalg.solve(K, y)
            self._L = None

    def predict(self, X_new: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) predictions for *X_new*."""
        K_s = self._kernel(X_new, self._X)           # (m, n)
        mu = K_s @ self._alpha

        K_ss = np.ones(len(X_new))                    # diag of k(X*, X*)
        if self._L is not None:
            v = np.linalg.solve(self._L, K_s.T)       # (n, m)
            var = K_ss - np.sum(v ** 2, axis=0)
        else:
            K = self._kernel(self._X, self._X) + self.noise * np.eye(len(self._X))
            var = K_ss - np.sum(K_s @ np.linalg.solve(K, K_s.T), axis=1)

        std = np.sqrt(np.maximum(var, 0.0))
        return mu, std


# ---------------------------------------------------------------------------
# Bayesian optimizer
# ---------------------------------------------------------------------------


class BayesianDesignOptimizer:
    """Sequential model-based optimizer using a Gaussian process surrogate.

    Prefers ``sklearn.gaussian_process.GaussianProcessRegressor`` when
    available; falls back to :class:`_NumpyGP` otherwise.

    Parameters
    ----------
    cfg:
        Optimizer configuration.
    seed:
        Random seed for reproducible random proposals.
    """

    def __init__(self, cfg: DesignOptimizerConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))
        self.X_obs: List[np.ndarray] = []
        self.y_obs: List[float] = []
        self._gp: Optional[object] = None
        self._use_sklearn = False

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
            from sklearn.gaussian_process.kernels import Matern  # type: ignore

            self._sklearn_gpr_cls = GaussianProcessRegressor
            self._sklearn_kernel = Matern(nu=2.5)
            self._use_sklearn = True
        except ImportError:
            self._numpy_gp = _NumpyGP()

    # ------------------------------------------------------------------

    def _fit_gp(self) -> None:
        X = np.stack(self.X_obs)
        y = np.array(self.y_obs)
        if self._use_sklearn:
            from sklearn.gaussian_process import GaussianProcessRegressor  # type: ignore
            from sklearn.gaussian_process.kernels import Matern  # type: ignore

            gpr = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                n_restarts_optimizer=3,
                normalize_y=True,
            )
            gpr.fit(X, y)
            self._gp = gpr
        else:
            self._numpy_gp.fit(X, y)

    def _gp_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._use_sklearn:
            return self._gp.predict(X, return_std=True)  # type: ignore[union-attr]
        else:
            return self._numpy_gp.predict(X)

    def _acquisition(self, x: np.ndarray) -> float:
        """Evaluate acquisition (negated because we minimise objectives)."""
        mu, sigma = self._gp_predict(x.reshape(1, -1))
        mu, sigma = float(mu[0]), float(sigma[0])
        y_best = min(self.y_obs)

        if self.cfg.acquisition == "ucb":
            # UCB for minimization: lower mu − kappa * sigma is better.
            return float(-(mu - self.cfg.kappa * sigma))

        # Expected Improvement for minimization.
        from scipy.stats import norm  # deferred: only needed for BO

        xi = self.cfg.xi
        z = (y_best - mu - xi) / (sigma + 1e-9)
        ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return float(ei)

    def propose(self, bounds: np.ndarray) -> np.ndarray:
        """Suggest the next design candidate to evaluate.

        During the random-exploration phase (fewer than *n_initial_random*
        observations), samples uniformly at random.  Afterwards selects the
        candidate that maximises the acquisition over a random multi-start
        search.

        Parameters
        ----------
        bounds:
            Array of shape (p, 2) with ``[lo, hi]`` per parameter.

        Returns
        -------
        np.ndarray
            1-D candidate array of shape (p,).
        """
        n = len(self.X_obs)
        lo, hi = bounds[:, 0], bounds[:, 1]

        if n < self.cfg.n_initial_random:
            return self.rng.uniform(lo, hi)

        self._fit_gp()

        # Random multi-start maximisation of the acquisition.
        n_candidates = max(500, 20 * len(lo))
        Xs = self.rng.uniform(lo, hi, size=(n_candidates, len(lo)))
        acq_vals = np.array([self._acquisition(x) for x in Xs])
        return Xs[int(np.argmax(acq_vals))]

    def update(self, x: np.ndarray, y: float) -> None:
        """Record a new design/objective observation."""
        self.X_obs.append(x.copy())
        self.y_obs.append(float(y))


# ---------------------------------------------------------------------------
# Evolutionary optimizer
# ---------------------------------------------------------------------------


class EvolutionaryDesignOptimizer:
    """Population-based optimizer with CMA-ES or simple GA fallback.

    When the ``cma`` package is installed, delegates to CMA-ES.  Otherwise
    runs a ``(μ+λ)`` genetic algorithm with tournament selection, uniform
    crossover, and Gaussian mutation.

    Parameters
    ----------
    cfg:
        Optimizer configuration.
    seed:
        Random seed.
    """

    def __init__(self, cfg: DesignOptimizerConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(int(seed))
        self._use_cma = False
        self._cma_es = None
        self._population: List[np.ndarray] = []
        self._fitness: List[float] = []

        try:
            import cma  # type: ignore
            self._cma_mod = cma
            self._use_cma = True
        except ImportError:
            self._use_cma = False

    def _init_cma(self, x0: np.ndarray) -> None:
        opts = {
            "seed": int(self.rng.integers(0, 2**31)),
            "popsize": self.cfg.population_size,
            "verbose": -9,
        }
        self._cma_es = self._cma_mod.CMAEvolutionStrategy(x0.tolist(), self.cfg.sigma0, opts)

    # ------------------------------------------------------------------

    def ask(self, bounds: np.ndarray, n: int) -> List[np.ndarray]:
        """Return *n* candidate design vectors.

        Parameters
        ----------
        bounds:
            Shape (p, 2) array of ``[lo, hi]`` per parameter.
        n:
            Number of candidates to generate.

        Returns
        -------
        list of np.ndarray
        """
        lo, hi = bounds[:, 0], bounds[:, 1]

        if self._use_cma:
            if self._cma_es is None:
                x0 = self.rng.uniform(lo, hi)
                self._init_cma(x0)
            raw = self._cma_es.ask(number=n)
            return [np.clip(np.array(x, dtype=np.float64), lo, hi) for x in raw]

        # GA: bootstrap with random population if not yet initialised.
        if not self._population:
            self._population = [self.rng.uniform(lo, hi) for _ in range(self.cfg.population_size)]
            return self._population[:n]

        candidates: List[np.ndarray] = []
        pop = self._population
        fit = self._fitness if self._fitness else [0.0] * len(pop)

        for _ in range(n):
            # Tournament selection of two parents.
            def _tournament() -> np.ndarray:
                i1, i2 = self.rng.integers(0, len(pop), size=2)
                return pop[i1] if fit[i1] <= fit[i2] else pop[i2]

            p1, p2 = _tournament(), _tournament()

            # Uniform crossover.
            if self.rng.random() < self.cfg.crossover_rate:
                mask = self.rng.random(size=p1.shape) < 0.5
                child = np.where(mask, p1, p2)
            else:
                child = p1.copy()

            # Gaussian mutation.
            child = child + self.rng.normal(0, self.cfg.mutation_std, size=child.shape)
            child = np.clip(child, lo, hi)
            candidates.append(child)

        return candidates

    def tell(self, xs: List[np.ndarray], ys: List[float]) -> None:
        """Inform the optimizer of objective values for the last candidates.

        Parameters
        ----------
        xs:
            Candidate design vectors (same list returned by :meth:`ask`).
        ys:
            Corresponding scalar objective values.
        """
        if self._use_cma and self._cma_es is not None:
            self._cma_es.tell(xs, ys)
            return

        # GA (μ+λ): merge candidates with existing population, keep best μ.
        combined_x = list(self._population) + list(xs)
        combined_y = list(self._fitness if self._fitness else [float("inf")] * len(self._population)) + list(ys)

        mu = self.cfg.population_size
        idx = np.argsort(combined_y)[:mu]
        self._population = [combined_x[i] for i in idx]
        self._fitness = [combined_y[i] for i in idx]
