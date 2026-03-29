"""Kalman filter and Ensemble Kalman Filter (EnKF) for data assimilation.

Used by the DigitalTwin to fuse model predictions with real-time
sensor observations and maintain a calibrated state estimate.

References
----------
- Evensen (2009) "Data Assimilation: The Ensemble Kalman Filter"
- Asch, Bocquet, Nodet (2016) "Data Assimilation"
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extended Kalman Filter (EKF) — linearized, good for low-dimensional states
# ---------------------------------------------------------------------------

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for state estimation.

    State: x ∈ R^n
    Observation: y ∈ R^m

    Dynamics:   x_k = f(x_{k-1}) + w_k,  w ~ N(0, Q)
    Observation: y_k = h(x_k)    + v_k,  v ~ N(0, R)

    Parameters
    ----------
    n_state : int        state dimension
    n_obs : int          observation dimension
    f : callable         state transition x_{k-1} -> x_k
    h : callable         observation function x_k -> y_k
    Q : np.ndarray       process noise covariance (n_state, n_state)
    R : np.ndarray       observation noise covariance (n_obs, n_obs)
    F_jac : callable     Jacobian of f wrt x (optional; uses finite diff if None)
    H_jac : callable     Jacobian of h wrt x (optional; uses finite diff if None)
    """

    def __init__(
        self,
        n_state: int,
        n_obs: int,
        f: Callable[[np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        F_jac: Optional[Callable] = None,
        H_jac: Optional[Callable] = None,
        *,
        eps_fd: float = 1e-5,
    ) -> None:
        self.n_state = int(n_state)
        self.n_obs = int(n_obs)
        self.f = f
        self.h = h
        self.Q = Q if Q is not None else np.eye(n_state, dtype=np.float64) * 1e-4
        self.R = R if R is not None else np.eye(n_obs, dtype=np.float64) * 1e-2
        self._F_jac = F_jac
        self._H_jac = H_jac
        self.eps_fd = float(eps_fd)

        # State and covariance
        self.x: np.ndarray = np.zeros(self.n_state, dtype=np.float64)
        self.P: np.ndarray = np.eye(self.n_state, dtype=np.float64)

    def _jacobian_fd(
        self, fn: Callable, x: np.ndarray, eps: float
    ) -> np.ndarray:
        """Finite-difference Jacobian of fn at x."""
        f0 = fn(x)
        n = len(x)
        m = len(f0)
        J = np.zeros((m, n), dtype=np.float64)
        for i in range(n):
            dx = np.zeros(n)
            dx[i] = eps
            J[:, i] = (fn(x + dx) - f0) / eps
        return J

    def _F(self, x: np.ndarray) -> np.ndarray:
        if self._F_jac is not None:
            return self._F_jac(x)
        return self._jacobian_fd(self.f, x, self.eps_fd)

    def _H(self, x: np.ndarray) -> np.ndarray:
        if self._H_jac is not None:
            return self._H_jac(x)
        return self._jacobian_fd(self.h, x, self.eps_fd)

    def initialize(self, x0: np.ndarray, P0: Optional[np.ndarray] = None) -> None:
        """Set initial state estimate and covariance."""
        self.x = np.asarray(x0, dtype=np.float64)
        if P0 is not None:
            self.P = np.asarray(P0, dtype=np.float64)

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """Predict step: propagate state and covariance."""
        x_pred = self.f(self.x)
        F = self._F(self.x)
        P_pred = F @ self.P @ F.T + self.Q
        self.x = x_pred
        self.P = P_pred
        return x_pred, P_pred

    def update(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Update step: fuse observation y with the prior.

        Returns
        -------
        dict with keys: x (updated state), P (updated covariance),
                        K (Kalman gain), innovation, innovation_cov
        """
        y = np.asarray(y, dtype=np.float64)
        H = self._H(self.x)
        innovation = y - self.h(self.x)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.solve(S, np.eye(len(y)))
        self.x = self.x + K @ innovation
        self.P = (np.eye(self.n_state) - K @ H) @ self.P
        return {
            "x": self.x.copy(),
            "P": self.P.copy(),
            "K": K,
            "innovation": innovation,
            "innovation_cov": S,
        }

    def step(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict + update in one call."""
        self.predict()
        return self.update(y)


# ---------------------------------------------------------------------------
# Ensemble Kalman Filter (EnKF) — Monte Carlo, handles non-linear dynamics
# ---------------------------------------------------------------------------

class EnsembleKalmanFilter:
    """
    Stochastic Ensemble Kalman Filter.

    Better for high-dimensional or non-linear systems than EKF.
    Internally maintains an ensemble of N_ens state samples.

    Parameters
    ----------
    n_state : int         state dimension
    n_obs : int           observation dimension
    f : callable          state transition x -> x'
    h : callable          observation operator x -> y
    Q : np.ndarray        process noise covariance
    R : np.ndarray        observation noise covariance
    n_ens : int           ensemble size (default 100)
    inflation : float     covariance inflation factor (>1 prevents filter divergence)
    """

    def __init__(
        self,
        n_state: int,
        n_obs: int,
        f: Callable[[np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
        n_ens: int = 100,
        inflation: float = 1.02,
        seed: int = 0,
    ) -> None:
        self.n_state = int(n_state)
        self.n_obs = int(n_obs)
        self.f = f
        self.h = h
        self.Q = Q if Q is not None else np.eye(n_state) * 1e-4
        self.R = R if R is not None else np.eye(n_obs) * 1e-2
        self.n_ens = int(n_ens)
        self.inflation = float(inflation)
        self.rng = np.random.default_rng(seed)

        # Ensemble: (n_state, n_ens)
        self.ensemble: np.ndarray = np.zeros((self.n_state, self.n_ens))

    def initialize(
        self,
        x0: np.ndarray,
        P0: Optional[np.ndarray] = None,
    ) -> None:
        """Initialize ensemble around x0."""
        x0 = np.asarray(x0, dtype=np.float64)
        cov = P0 if P0 is not None else self.Q
        self.ensemble = (
            x0[:, None]
            + np.linalg.cholesky(cov + np.eye(self.n_state) * 1e-12)
            @ self.rng.standard_normal((self.n_state, self.n_ens))
        )

    @property
    def mean(self) -> np.ndarray:
        return self.ensemble.mean(axis=1)

    @property
    def covariance(self) -> np.ndarray:
        X = self.ensemble - self.mean[:, None]
        return (X @ X.T) / (self.n_ens - 1)

    def predict(self) -> None:
        """Propagate each ensemble member through the dynamics."""
        noise = (
            np.linalg.cholesky(self.Q + np.eye(self.n_state) * 1e-12)
            @ self.rng.standard_normal((self.n_state, self.n_ens))
        )
        for i in range(self.n_ens):
            self.ensemble[:, i] = self.f(self.ensemble[:, i]) + noise[:, i]

        # Covariance inflation
        mu = self.mean
        self.ensemble = mu[:, None] + self.inflation * (self.ensemble - mu[:, None])

    def update(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Stochastic EnKF analysis step.

        Returns mean state, innovation, and ensemble.
        """
        y = np.asarray(y, dtype=np.float64)

        # Perturbed observations
        obs_noise = (
            np.linalg.cholesky(self.R + np.eye(self.n_obs) * 1e-12)
            @ self.rng.standard_normal((self.n_obs, self.n_ens))
        )
        Y = y[:, None] + obs_noise  # (n_obs, n_ens)

        # Observation ensemble
        HX = np.column_stack([self.h(self.ensemble[:, i]) for i in range(self.n_ens)])

        # Cross-covariance
        A = self.ensemble - self.mean[:, None]
        HA = HX - HX.mean(axis=1, keepdims=True)
        PHT = A @ HA.T / (self.n_ens - 1)
        HPHT = HA @ HA.T / (self.n_ens - 1)
        S = HPHT + self.R
        K = PHT @ np.linalg.solve(S, np.eye(self.n_obs))

        # Update
        for i in range(self.n_ens):
            self.ensemble[:, i] += K @ (Y[:, i] - HX[:, i])

        innovation = y - HX.mean(axis=1)
        return {
            "x": self.mean.copy(),
            "ensemble": self.ensemble.copy(),
            "innovation": innovation,
            "K": K,
        }

    def step(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict + update in one call."""
        self.predict()
        return self.update(y)
