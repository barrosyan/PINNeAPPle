from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Sequence

HorizonType = Literal["direct", "recursive", "dirrec"]
ObjectiveType = Literal["point", "quantile", "distribution"]
FrequencyType = Optional[str]  # e.g. "D", "M", "H"; None for irregular timestamps


@dataclass(frozen=True)
class ForecastProblemSpec:
    """Formal forecasting problem definition.

    Define this BEFORE modeling to avoid the common failure mode:
    training a model without a well-defined forecasting target/evaluation setup.
    """

    # Data semantics
    freq: FrequencyType = None
    time_col: Optional[str] = None
    target_cols: Sequence[str] = ("y",)
    feature_cols: Sequence[str] = tuple()      # past-known features
    exog_past_cols: Sequence[str] = tuple()    # only known up to time t
    exog_future_cols: Sequence[str] = tuple()  # known for the forecast horizon

    # Forecasting setup
    input_len: int = 64
    horizon: int = 16
    horizon_type: HorizonType = "direct"       # direct multi-horizon vs recursive
    target_offset: int = 0
    stride: int = 1

    # Evaluation / business
    objective: ObjectiveType = "point"
    metrics: Sequence[str] = ("mae", "rmse", "smape", "mase")
    business_costs: Dict[str, float] = field(default_factory=dict)

    # Probabilistic forecasting
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)

    # Constraints
    allow_random_split: bool = False
    notes: str = ""

    def to_timeseries_spec(self):
        """Convert to the windowing spec used by `WindowedTimeSeriesDataset`."""
        from .spec import TimeSeriesSpec
        return TimeSeriesSpec(
            input_len=int(self.input_len),
            horizon=int(self.horizon),
            stride=int(self.stride),
            target_offset=int(self.target_offset),
        )