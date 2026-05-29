"""Ensemble Kalman Inversion (EKI) for derivative-free inverse problems.

EKI is a derivative-free iterative method for PDE-constrained parameter
identification.  It maintains an ensemble of parameter candidates and updates
them via Kalman-like cross-covariance computations, requiring only forward
model evaluations (no gradients of G).

Implemented methods
-------------------
EnsembleKalmanInversion   standard EKI (Iglesias et al. 2013)
IteratedEKI               Tikhonov-regularized EKI / TEKI (Chada et al. 2020)

Both support:
- Parallel ensemble evaluation (via vectorised forward_fn)
- Configurable ensemble size (J ≈ 2–3 × p is often sufficient)
- Early stopping via ensemble collapse detection
- Full convergence history

Mathematical formulation
------------------------
Forward model:      G : R^p → R^k        (parameter → observations)
Observations:       y ∈ R^k               (noisy measurement vector)
Noise covariance:   Γ ∈ R^{k×k} (s.p.d.)

EKI update rule (for ensemble member j):

    g^(j) = G(θ^(j))
    C_θg  = (1/J) Σ_j (θ^(j) − θ̄)(g^(j) − ḡ)^T   ∈ R^{p×k}
    C_gg  = (1/J) Σ_j (g^(j) − ḡ)(g^(j) − ḡ)^T   ∈ R^{k×k}
    K     = C_θg (C_gg + Γ)^{-1}                    (Kalman gain)
    θ^(j) ← θ^(j) + K (y + η^(j) − g^(j)),   η^(j) ~ N(0, Γ)

For IteratedEKI, artificial parameters are appended to the state vector so
Tikhonov regularisation is incorporated analytically.

References
----------
- Iglesias, Law & Stuart (2013) "Ensemble Kalman Methods for Inverse Problems"
  Inverse Problems 29, 045001.
- Chada, Stuart & Tong (2020) "Tikhonov Regularisation in Ensemble Kalman
  Inversion" arXiv:1901.11493.
- Garbuno-Inigo et al. (2020) "Interacting Langevin Diffusions" SIAM J Appl
  Dyn Syst.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Configuration and history
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EKIConfig:
    """Configuration for Ensemble Kalman Inversion.

    Parameters
    ----------
    n_ensemble : int
        Ensemble size J.  Rule of thumb: J ≈ 3 × p (number of parameters).
        Larger J → more accurate covariance estimates, higher cost.
    n_iterations : int
        Maximum number of EKI update steps.
    noise_cov : np.ndarray, optional
        Observation noise covariance Γ ∈ R^{k×k}.  If None, uses noise_std²·I.
    noise_std : float
        Noise standard deviation σ (used when noise_cov is None).
    init_spread : float
        Standard deviation used to perturb the initial ensemble around
        ``theta_init``.  Set to 0 to start all members at the same point.
    lambda_reg : float
        Tikhonov regularisation strength λ ≥ 0.  Only used by IteratedEKI.
        λ = 0 recovers standard EKI.
    prior_mean : np.ndarray, optional
        Prior mean θ₀ for regularisation.  Defaults to ``theta_init``.
    prior_cov : np.ndarray, optional
        Prior covariance C₀ for regularisation.  If None uses λ·I.
    tol_collapse : float
        Stop early when ensemble spread < tol_collapse (collapsed ensemble).
    seed : int, optional
    verbose : bool
    print_every : int
    """
    n_ensemble: int = 100
    n_iterations: int = 50
    noise_cov: Optional[np.ndarray] = None
    noise_std: float = 0.01
    init_spread: float = 0.1
    lambda_reg: float = 0.0
    prior_mean: Optional[np.ndarray] = None
    prior_cov: Optional[np.ndarray] = None
    tol_collapse: float = 1e-8
    seed: Optional[int] = None
    verbose: bool = True
    print_every: int = 10


@dataclass
class EKIHistory:
    """Convergence history of an EKI run.

    Attributes
    ----------
    iterations : list[int]
    data_misfit : list[float]
        ‖G(θ̄) − y‖²_{Γ⁻¹} at each iteration (ensemble mean).
    ensemble_spread : list[float]
        Mean pairwise Euclidean distance in θ-space (collapse diagnostic).
    theta_mean_history : list[np.ndarray]
        Ensemble mean at each stored iteration.
    """
    iterations: List[int] = field(default_factory=list)
    data_misfit: List[float] = field(default_factory=list)
    ensemble_spread: List[float] = field(default_factory=list)
    theta_mean_history: List[np.ndarray] = field(default_factory=list)

    def final_misfit(self) -> float:
        return self.data_misfit[-1] if self.data_misfit else float("nan")

    def converged(self, tol: float = 1e-8) -> bool:
        return bool(self.ensemble_spread and self.ensemble_spread[-1] < tol)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_noise_cov(cfg: EKIConfig, k: int) -> np.ndarray:
    """Return the observation noise covariance Γ ∈ R^{k×k}."""
    if cfg.noise_cov is not None:
        Gamma = np.asarray(cfg.noise_cov, dtype=np.float64)
        if Gamma.shape != (k, k):
            raise ValueError(f"noise_cov shape {Gamma.shape} does not match obs dim {k}")
        return Gamma
    return cfg.noise_std ** 2 * np.eye(k)


def _ensemble_covariances(
    theta: np.ndarray,   # (J, p)
    g: np.ndarray,       # (J, k)
) -> Tuple[np.ndarray, np.ndarray]:
    """Return C_θg ∈ (p, k) and C_gg ∈ (k, k)."""
    J = theta.shape[0]
    theta_bar = theta.mean(axis=0, keepdims=True)   # (1, p)
    g_bar = g.mean(axis=0, keepdims=True)            # (1, k)
    dtheta = theta - theta_bar                        # (J, p)
    dg = g - g_bar                                    # (J, k)
    C_theta_g = (dtheta.T @ dg) / J                  # (p, k)
    C_gg = (dg.T @ dg) / J                            # (k, k)
    return C_theta_g, C_gg


def _data_misfit(g_mean: np.ndarray, y: np.ndarray, Gamma_inv: np.ndarray) -> float:
    r = g_mean - y
    return float(r @ Gamma_inv @ r)


def _ensemble_spread(theta: np.ndarray) -> float:
    """Mean pairwise distance (approximated via std across ensemble)."""
    return float(np.mean(np.std(theta, axis=0)))


# ──────────────────────────────────────────────────────────────────────────────
# Standard EKI
# ──────────────────────────────────────────────────────────────────────────────

class EnsembleKalmanInversion:
    """Standard Ensemble Kalman Inversion.

    Parameters
    ----------
    forward_fn : callable
        Maps a **batch** of parameters to observations.
        Signature: ``(theta_batch: np.ndarray) -> np.ndarray``
        Input shape (J, p), output shape (J, k).
        Batching is mandatory so the method can parallelise ensemble evals.
    config : EKIConfig, optional
    """

    def __init__(
        self,
        forward_fn: Callable[[np.ndarray], np.ndarray],
        config: Optional[EKIConfig] = None,
    ) -> None:
        self.forward_fn = forward_fn
        self.cfg = config or EKIConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self.theta: Optional[np.ndarray] = None   # (J, p) current ensemble
        self.history = EKIHistory()

    # ── Public interface ──────────────────────────────────────────────────────

    def initialize(
        self,
        theta_init: np.ndarray,
        *,
        ensemble: Optional[np.ndarray] = None,
    ) -> None:
        """Set up the ensemble.

        Parameters
        ----------
        theta_init : np.ndarray, shape (p,)
            Nominal starting point.
        ensemble : np.ndarray, shape (J, p), optional
            Custom initial ensemble.  If None, draws from
            N(theta_init, init_spread² I).
        """
        if ensemble is not None:
            self.theta = np.asarray(ensemble, dtype=np.float64)
        else:
            p = len(theta_init)
            J = self.cfg.n_ensemble
            noise = self._rng.normal(0.0, self.cfg.init_spread, size=(J, p))
            self.theta = theta_init[np.newaxis, :] + noise

        self.history = EKIHistory()

    @property
    def theta_mean(self) -> np.ndarray:
        """Current ensemble mean θ̄."""
        if self.theta is None:
            raise RuntimeError("Call initialize() before accessing theta_mean.")
        return self.theta.mean(axis=0)

    @property
    def theta_cov(self) -> np.ndarray:
        """Current ensemble covariance C ∈ R^{p×p}."""
        if self.theta is None:
            raise RuntimeError("Call initialize() before accessing theta_cov.")
        J = self.theta.shape[0]
        d = self.theta - self.theta_mean[np.newaxis, :]
        return (d.T @ d) / J

    def step(self, y: np.ndarray, Gamma: np.ndarray) -> float:
        """Perform one EKI update.

        Parameters
        ----------
        y : np.ndarray, shape (k,)
            Observed data vector.
        Gamma : np.ndarray, shape (k, k)
            Observation noise covariance.

        Returns
        -------
        float
            Data misfit ‖G(θ̄) − y‖²_{Γ⁻¹}.
        """
        J, p = self.theta.shape
        k = len(y)

        # Evaluate forward model for entire ensemble
        g = np.asarray(self.forward_fn(self.theta), dtype=np.float64)  # (J, k)
        if g.ndim == 1:
            g = g[:, np.newaxis]
        if g.shape != (J, k):
            raise ValueError(f"forward_fn output shape {g.shape}, expected ({J}, {k})")

        C_theta_g, C_gg = _ensemble_covariances(self.theta, g)

        # Kalman gain K = C_θg (C_gg + Γ)^{-1}   shape (p, k)
        S = C_gg + Gamma  # innovation covariance
        try:
            K = C_theta_g @ np.linalg.solve(S.T, np.eye(k)).T
        except np.linalg.LinAlgError:
            K = C_theta_g @ np.linalg.pinv(S)

        # Perturbed observations  η^(j) ~ N(0, Γ)
        eta = self._rng.multivariate_normal(np.zeros(k), Gamma, size=J)  # (J, k)

        # Update ensemble
        innovation = y[np.newaxis, :] + eta - g   # (J, k)
        self.theta = self.theta + innovation @ K.T   # (J, p)

        # Diagnostics
        Gamma_inv = np.linalg.pinv(Gamma)
        g_mean = g.mean(axis=0)
        misfit = _data_misfit(g_mean, y, Gamma_inv)
        return misfit

    def run(
        self,
        y: np.ndarray,
        theta_init: np.ndarray,
        *,
        ensemble: Optional[np.ndarray] = None,
    ) -> EKIHistory:
        """Run full EKI until convergence or max iterations.

        Parameters
        ----------
        y : np.ndarray, shape (k,)
            Observed data.
        theta_init : np.ndarray, shape (p,)
            Nominal initial parameter values.
        ensemble : np.ndarray, shape (J, p), optional
            Custom initial ensemble.

        Returns
        -------
        EKIHistory
        """
        self.initialize(theta_init, ensemble=ensemble)
        cfg = self.cfg
        k = len(y)
        Gamma = _build_noise_cov(cfg, k)

        for t in range(1, cfg.n_iterations + 1):
            misfit = self.step(y, Gamma)
            spread = _ensemble_spread(self.theta)

            self.history.iterations.append(t)
            self.history.data_misfit.append(misfit)
            self.history.ensemble_spread.append(spread)
            self.history.theta_mean_history.append(self.theta_mean.copy())

            if cfg.verbose and t % cfg.print_every == 0:
                print(f"  [EKI  t={t:4d}/{cfg.n_iterations}]  "
                      f"misfit={misfit:.4e}  spread={spread:.4e}")

            if spread < cfg.tol_collapse:
                if cfg.verbose:
                    print(f"  [EKI] Ensemble collapsed (spread={spread:.2e}) "
                          f"at iteration {t}. Stopping.")
                break

        return self.history


# ──────────────────────────────────────────────────────────────────────────────
# Tikhonov-regularized EKI (IteratedEKI / TEKI)
# ──────────────────────────────────────────────────────────────────────────────

class IteratedEKI(EnsembleKalmanInversion):
    """Tikhonov-regularized Ensemble Kalman Inversion (TEKI).

    Adds Tikhonov regularisation by extending the state space:

        G̃(θ) = [G(θ);  √λ (θ − θ₀)]   ∈ R^{k+p}
        ỹ    = [y;     0_p]
        Γ̃    = diag(Γ, I)

    Running standard EKI on the augmented system minimises:

        J(θ) = ‖G(θ) − y‖²_{Γ⁻¹} + λ ‖θ − θ₀‖²

    For λ → 0 recovers standard EKI; for λ → ∞ shrinks to prior mean.

    Parameters
    ----------
    forward_fn : callable
        Same as EnsembleKalmanInversion.
    config : EKIConfig
        Set ``lambda_reg`` > 0 and optionally ``prior_mean``.
    """

    def run(
        self,
        y: np.ndarray,
        theta_init: np.ndarray,
        *,
        ensemble: Optional[np.ndarray] = None,
    ) -> EKIHistory:
        self.initialize(theta_init, ensemble=ensemble)
        cfg = self.cfg
        k = len(y)
        p = theta_init.shape[0]
        Gamma = _build_noise_cov(cfg, k)
        lam = cfg.lambda_reg

        theta_0 = (
            np.asarray(cfg.prior_mean, dtype=np.float64)
            if cfg.prior_mean is not None
            else theta_init.copy()
        )

        if lam > 0.0:
            # Augmented noise covariance Γ̃ = diag(Γ, I/λ) ∈ R^{(k+p)×(k+p)}
            Gamma_aug = np.block([
                [Gamma,                np.zeros((k, p))],
                [np.zeros((p, k)),     np.eye(p) / lam ],
            ])
            y_aug = np.concatenate([y, np.zeros(p)])
        else:
            Gamma_aug = Gamma
            y_aug = y

        # Capture original fn in a local variable BEFORE replacing self.forward_fn
        orig_fn = self.forward_fn

        def _aug_forward(theta_batch: np.ndarray) -> np.ndarray:
            g_orig = np.asarray(orig_fn(theta_batch), dtype=np.float64)   # closure over orig_fn
            if g_orig.ndim == 1:
                g_orig = g_orig[:, np.newaxis]
            if lam > 0.0:
                reg_term = np.sqrt(lam) * (theta_batch - theta_0[np.newaxis, :])
                return np.concatenate([g_orig, reg_term], axis=1)
            return g_orig

        self.forward_fn = _aug_forward

        for t in range(1, cfg.n_iterations + 1):
            misfit_full = self.step(y_aug, Gamma_aug)

            # Compute real data misfit (not augmented) for reporting
            g_real = np.asarray(orig_fn(self.theta), dtype=np.float64)
            if g_real.ndim == 1:
                g_real = g_real[:, np.newaxis]
            Gamma_inv = np.linalg.pinv(Gamma)
            misfit = _data_misfit(g_real.mean(axis=0), y, Gamma_inv)
            spread = _ensemble_spread(self.theta)

            self.history.iterations.append(t)
            self.history.data_misfit.append(misfit)
            self.history.ensemble_spread.append(spread)
            self.history.theta_mean_history.append(self.theta_mean.copy())

            if cfg.verbose and t % cfg.print_every == 0:
                print(f"  [TEKI t={t:4d}/{cfg.n_iterations}]  "
                      f"misfit={misfit:.4e}  spread={spread:.4e}  lam={lam:.2e}")

            if spread < cfg.tol_collapse:
                if cfg.verbose:
                    print(f"  [TEKI] Ensemble collapsed at iteration {t}.")
                break

        self.forward_fn = orig_fn
        return self.history


__all__ = [
    "EKIConfig",
    "EKIHistory",
    "EnsembleKalmanInversion",
    "IteratedEKI",
]
