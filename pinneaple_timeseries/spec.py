from dataclasses import dataclass


@dataclass
class TimeSeriesSpec:
    input_len: int = 64
    horizon: int = 16
    stride: int = 1
    target_offset: int = 0
