"""pinneapple_analysis.state_estimation -- Sequential state estimation
(data assimilation) building blocks.

Relocated from ``pinneapple_systems/digital_twin/assimilation/kalman.py``
(implementation code coupled to ``DigitalTwin`` only by its location in
the tree, not by its API) so any use case -- ``examples/``, other
``pinneapple_*`` packages, or standalone scripts -- can fuse a plain
physics model with streaming/batch sensor observations directly,
without building a full ``DigitalTwin``.
``pinneapple_systems/digital_twin/assimilation/kalman.py`` now contains
a thin re-export shim pointing back here (the same convention as
``pinneapple_models/registry.py``), so every existing caller keeps
working unchanged.

Sub-modules
-----------
kalman
    ``ExtendedKalmanFilter`` (linearized EKF, finite-difference or
    user-supplied Jacobians) and ``EnsembleKalmanFilter`` (stochastic
    EnKF for higher-dimensional / non-linear systems). Both drive a
    user-supplied state-transition ``f`` and observation operator ``h``
    through the classic predict -> observe -> update recursion over a
    time series of plain ``np.ndarray`` states and observations.

Note
----
This is sequential/recursive *state* filtering, not to be confused
with ``pinneapple_analysis.inverse_problems.ensemble_kalman``'s
``EnsembleKalmanInversion`` / ``IteratedEKI`` (Ensemble Kalman
**Inversion**), a derivative-free *parameter* estimation technique that
iterates an ensemble to a converged parameter estimate rather than
filtering a state through time.
"""
from __future__ import annotations

from pinneapple_analysis.state_estimation.kalman import (
    ExtendedKalmanFilter,
    EnsembleKalmanFilter,
)

__all__ = [
    "ExtendedKalmanFilter",
    "EnsembleKalmanFilter",
]
