"""Sensor management for digital twins.

A ``SensorRegistry`` keeps track of sensor positions, calibration offsets,
and expected field mappings.  Each ``Sensor`` converts a raw reading into a
calibrated ``Observation``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..state import Observation


@dataclass
class Sensor:
    """
    Represents a single physical sensor.

    Parameters
    ----------
    sensor_id : str
        Unique identifier (e.g. "pressure_01").
    coords : dict
        Spatial position {"x": ..., "y": ..., "z": ...} (subset allowed).
    field_names : list[str]
        Names of the physical fields this sensor measures.
    calibration_offset : dict[str, float]
        Additive calibration offsets per field (applied to raw readings).
    calibration_scale : dict[str, float]
        Multiplicative calibration factors per field.
    noise_std : dict[str, float]
        Expected measurement noise std per field (used for filtering).
    transform : callable, optional
        Extra transform applied *after* calibration: raw_val -> calibrated_val.
    active : bool
        If False the sensor is ignored during data assimilation.
    """

    sensor_id: str
    coords: Dict[str, float]
    field_names: List[str]
    calibration_offset: Dict[str, float] = field(default_factory=dict)
    calibration_scale: Dict[str, float] = field(default_factory=dict)
    noise_std: Dict[str, float] = field(default_factory=dict)
    transform: Optional[Callable[[str, float], float]] = field(
        default=None, repr=False
    )
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def calibrate(self, raw: Dict[str, float]) -> Dict[str, float]:
        """
        Apply calibration offsets and scales to raw field readings.

        raw : dict field_name -> raw value
        returns : dict field_name -> calibrated value
        """
        out: Dict[str, float] = {}
        for fname in self.field_names:
            v = raw.get(fname, float("nan"))
            scale = self.calibration_scale.get(fname, 1.0)
            offset = self.calibration_offset.get(fname, 0.0)
            v = v * scale + offset
            if self.transform is not None:
                v = self.transform(fname, v)
            out[fname] = v
        return out

    def read(self, raw: Dict[str, float]) -> Observation:
        """Convert raw readings into a calibrated Observation."""
        calibrated = self.calibrate(raw)
        return Observation(
            timestamp=time.time(),
            sensor_id=self.sensor_id,
            values=calibrated,
            coords=dict(self.coords),
            metadata={"noise_std": dict(self.noise_std)},
        )


class SensorRegistry:
    """
    Registry of all sensors in the digital twin.

    Usage
    -----
    >>> reg = SensorRegistry()
    >>> reg.add(Sensor("p01", coords={"x": 0.5, "y": 0.5}, field_names=["p"]))
    >>> obs = reg.read("p01", raw={"p": 101325.0})
    """

    def __init__(self) -> None:
        self._sensors: Dict[str, Sensor] = {}

    def add(self, sensor: Sensor) -> None:
        """Register a sensor."""
        self._sensors[sensor.sensor_id] = sensor

    def remove(self, sensor_id: str) -> None:
        self._sensors.pop(sensor_id, None)

    def get(self, sensor_id: str) -> Sensor:
        if sensor_id not in self._sensors:
            raise KeyError(f"Sensor '{sensor_id}' not registered.")
        return self._sensors[sensor_id]

    def list_active(self) -> List[str]:
        return [sid for sid, s in self._sensors.items() if s.active]

    def read(self, sensor_id: str, raw: Dict[str, float]) -> Observation:
        return self.get(sensor_id).read(raw)

    def bulk_read(
        self, raw_readings: Dict[str, Dict[str, float]]
    ) -> List[Observation]:
        """
        Read multiple sensors at once.

        raw_readings : {sensor_id: {field_name: raw_value}}
        """
        obs: List[Observation] = []
        for sid, raw in raw_readings.items():
            if sid in self._sensors and self._sensors[sid].active:
                obs.append(self.read(sid, raw))
        return obs

    def to_observation_array(
        self,
        observations: List[Observation],
        field_names: Optional[Sequence[str]] = None,
    ) -> "np.ndarray":
        """
        Stack observations into a numpy array of shape (n_obs, n_fields).
        Missing fields are filled with NaN.
        """
        if not observations:
            return np.empty((0, 0), dtype=np.float32)
        all_fields = field_names or sorted(
            {k for obs in observations for k in obs.values}
        )
        rows = []
        for obs in observations:
            row = [obs.values.get(f, float("nan")) for f in all_fields]
            rows.append(row)
        return np.array(rows, dtype=np.float32)
