# Authoritative implementation lives in pinneapple_uq.quantile.
# This file re-exports for backwards compatibility with pinneapple_timeseries users.
from pinneapple_analysis.uncertainty.quantile import (  # noqa: F401
    pinball_loss_torch,
    QuantileConfig,
    QuantileHead,
    QuantileLoss,
)