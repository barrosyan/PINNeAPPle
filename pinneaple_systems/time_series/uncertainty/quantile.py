# Authoritative implementation lives in pinneaple_uq.quantile.
# This file re-exports for backwards compatibility with pinneaple_timeseries users.
from pinneaple_analysis.uncertainty.quantile import (  # noqa: F401
    pinball_loss_torch,
    QuantileConfig,
    QuantileHead,
    QuantileLoss,
)