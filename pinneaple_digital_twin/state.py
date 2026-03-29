"""System state representation for digital twins.

``SystemState`` holds the current (fused) estimate of the physical system
together with timestamps and uncertainty estimates.  It is the central
data structure passed between the data streams, assimilation layer, and
the predictive model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Observation:
    """A single raw observation from a sensor stream."""

    timestamp: float                           # UNIX seconds
    sensor_id: str                            # identifies the sensor
    values: Dict[str, float]                  # field_name -> measured value
    coords: Optional[Dict[str, float]] = None  # x, y, z, t where measurement was taken
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(
        cls,
        sensor_id: str,
        values: Dict[str, float],
        coords: Optional[Dict[str, float]] = None,
        **metadata: Any,
    ) -> "Observation":
        return cls(
            timestamp=time.time(),
            sensor_id=sensor_id,
            values=values,
            coords=coords,
            metadata=metadata,
        )


@dataclass
class SystemState:
    """
    Current estimate of the physical system state.

    Fields
    ------
    timestamp : float
        UNIX timestamp of the state estimate.
    fields : dict[str, np.ndarray]
        Named field arrays, e.g. {"u": ..., "v": ..., "p": ...}.
        Each array has shape ``(N,)`` or ``(nx, ny, ...)`` depending on
        the discretisation.
    coords : dict[str, np.ndarray]
        Coordinate arrays matching the field arrays, e.g. {"x": ..., "y": ...}.
    covariance : np.ndarray or None
        State covariance (NxN) from the Kalman filter, if available.
    residuals : dict[str, float]
        Observation-minus-background residuals (innovation statistics).
    metadata : dict
        Arbitrary extra info (run_id, problem_id, geometry hash, …).
    """

    timestamp: float = field(default_factory=time.time)
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    coords: Dict[str, np.ndarray] = field(default_factory=dict)
    covariance: Optional[np.ndarray] = None
    residuals: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # History: list of (timestamp, field_snapshots)
    history: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def snapshot(self) -> Dict[str, Any]:
        """Return a lightweight dict snapshot (no history, no covariance)."""
        return {
            "timestamp": self.timestamp,
            "fields": {k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in self.fields.items()},
            "coords": {k: v.tolist() if isinstance(v, np.ndarray) else v
                       for k, v in self.coords.items()},
            "residuals": dict(self.residuals),
            "metadata": dict(self.metadata),
        }

    def push_to_history(self, max_history: int = 500) -> None:
        """Append current snapshot to history, trimming if needed."""
        self.history.append(self.snapshot())
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]

    def get_field_array(self, field_name: str) -> np.ndarray:
        """Return the named field array, raising KeyError if absent."""
        if field_name not in self.fields:
            raise KeyError(f"Field '{field_name}' not in SystemState.fields. "
                           f"Available: {list(self.fields)}")
        return self.fields[field_name]

    def update_field(self, field_name: str, values: np.ndarray) -> None:
        """Update a named field array."""
        self.fields[field_name] = np.asarray(values, dtype=np.float32)
        self.timestamp = time.time()

    def to_flat_vector(self, field_names: Optional[Sequence[str]] = None) -> np.ndarray:
        """Concatenate requested fields into a flat 1D state vector."""
        names = list(field_names) if field_names else sorted(self.fields)
        return np.concatenate([self.fields[k].ravel() for k in names], axis=0)

    def from_flat_vector(
        self,
        vec: np.ndarray,
        field_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Unpack a flat state vector back into fields (in-place)."""
        names = list(field_names) if field_names else sorted(self.fields)
        idx = 0
        for k in names:
            n = self.fields[k].size
            self.fields[k] = vec[idx: idx + n].reshape(self.fields[k].shape)
            idx += n

    @classmethod
    def from_inference_result(cls, result: Any) -> "SystemState":
        """Build a SystemState from a ``pinneaple_inference.InferenceResult``."""
        import numpy as np
        coords = {k: np.asarray(v) for k, v in result.coords.items()}
        fields = {k: np.asarray(v) for k, v in result.fields.items()}
        return cls(
            fields=fields,
            coords=coords,
            metadata={"model_id": getattr(result, "model_id", None)},
        )
