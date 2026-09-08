"""``pinneapple_systems.digital_twin.assimilation.kalman`` compatibility
shim -- the real ``ExtendedKalmanFilter`` / ``EnsembleKalmanFilter``
implementation moved to ``pinneapple_analysis.state_estimation.kalman``
(a capability audit found no general-purpose, standalone
data-assimilation module usable outside ``DigitalTwin``; the code
itself never referenced ``DigitalTwin``/``SystemState``/``Observation``,
so the move is a straight relocation). This module re-exports the same
public names (same convention as ``pinneapple_models/registry.py``) so
every existing caller -- ``DigitalTwin._build_default_filter`` included
-- keeps working unchanged.
"""
from __future__ import annotations

from pinneapple_analysis.state_estimation.kalman import (
    ExtendedKalmanFilter,
    EnsembleKalmanFilter,
)

__all__ = ["ExtendedKalmanFilter", "EnsembleKalmanFilter"]
