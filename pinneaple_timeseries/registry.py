from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch.nn as nn

from pinneaple_models.registry import ModelRegistry
from .models.fno_forecaster import FNOForecaster, FNOForecastConfig


# ---------------------------------------------------------
# Register default TS model (idempotent)
# ---------------------------------------------------------
def _ensure_ts_fno_registered() -> None:
    try:
        @ModelRegistry.register(
            name="ts_fno",
            family="timeseries",
            description="Default TS forecaster using FNO-1D with history+zero padding (direct multi-horizon).",
            tags=["timeseries", "forecast", "fno", "default"],
        )
        class _TSFNO(FNOForecaster):
            pass

    except KeyError:
        pass


_ensure_ts_fno_registered()


# ---------------------------------------------------------
# Catalog
# ---------------------------------------------------------
@dataclass
class TSModelCatalog:
    families: Optional[List[str]] = None
    default_name: str = "ts_fno"

    def __post_init__(self):
        if self.families is None:
            self.families = ["timeseries", "neural_operators", "recurrent", "transformers", "classical_ts", "pinns"]

    def list(self) -> List[str]:
        out = []
        for fam in self.families:
            try:
                out.extend(ModelRegistry.list(family=fam))
            except Exception:
                pass
        return sorted(set(out))

    def build(self, name: str, **kwargs) -> nn.Module:
        return ModelRegistry.build(name, **kwargs)

    def spec(self, name: str):
        return ModelRegistry.spec(name)

    def build_default_fno(
        self,
        *,
        input_len: int,
        horizon: int,
        n_features: int,
        n_targets: int,
        width: int = 64,
        modes: int = 16,
        layers: int = 4,
    ) -> nn.Module:
        cfg = FNOForecastConfig(
            input_len=int(input_len),
            horizon=int(horizon),
            n_features=int(n_features),
            n_targets=int(n_targets),
            width=int(width),
            modes=int(modes),
            layers=int(layers),
        )
        return self.build(self.default_name, cfg=cfg)
