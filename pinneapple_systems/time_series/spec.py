"""Time series specification (input_len, horizon, stride, target_offset)."""
from dataclasses import dataclass


@dataclass
class TimeSeriesSpec:
    """Specification for time series input length, horizon, stride, and target offset."""

    input_len: int = 64
    horizon: int = 16
    stride: int = 1
    target_offset: int = 0
