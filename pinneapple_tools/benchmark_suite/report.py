"""Benchmark report: structured output for PhysicsBenchmarkSpec and TimeSeriesBenchmarkSpec.

The report is JSON-serializable and can be saved to disk, printed, or
post-processed programmatically.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# Per-model result
# -----------------------------------------------------------------------------

@dataclass
class ModelRunResult:
    model_id: str
    n_params: int
    training_time_s: float
    metrics: Dict[str, float]
    history: List[Dict[str, float]]        # e.g. [{"epoch":100, "loss":0.1}, ...]
    param_estimates: Optional[Dict[str, float]] = None   # inverse problems only
    rank: int = 0
    error_message: Optional[str] = None   # set if training failed


# -----------------------------------------------------------------------------
# Top-level benchmark report
# -----------------------------------------------------------------------------

@dataclass
class BenchmarkReport:
    benchmark_type: str                      # "physics" | "timeseries"
    created_at: str
    version: str = "1.0"
    problem_info: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    model_results: Dict[str, ModelRunResult] = field(default_factory=dict)
    leaderboard: List[Dict[str, Any]] = field(default_factory=list)
    best_model: Optional[str] = None
    plots_saved: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # -- serialization --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a fully JSON-serializable dict."""
        def _clean(obj):
            if isinstance(obj, dict):
                return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_clean(v) for v in obj]
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return round(obj, 8)
            if hasattr(obj, "__dict__"):
                return _clean(asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj.__dict__)
            return obj

        raw = {
            "benchmark_type": self.benchmark_type,
            "version": self.version,
            "created_at": self.created_at,
            "problem_info": self.problem_info,
            "config": self.config,
            "models": {
                mid: _clean(asdict(r))
                for mid, r in self.model_results.items()
            },
            "leaderboard": self.leaderboard,
            "best_model": self.best_model,
            "plots_saved": self.plots_saved,
            "metadata": self.metadata,
        }
        return _clean(raw)

    def to_json(self, indent: int = 2) -> str:
        """Return JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path: str | Path) -> Path:
        """Save report as JSON file. Returns the resolved path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")
        return p.resolve()

    # -- console output -------------------------------------------------------

    def print_summary(self) -> None:
        SEP = "=" * 72
        sep = "-" * 72
        btype = self.benchmark_type.upper()
        print(f"\n{SEP}")
        print(f"  PINNeAPPle {btype} Benchmark Report")
        print(f"  {self.created_at}")
        print(SEP)

        if self.problem_info:
            print(f"  Problem : {self.problem_info.get('id', '?')}")
            if "pde" in self.problem_info:
                print(f"  PDE     : {self.problem_info['pde']}")
            if "source" in self.problem_info:
                print(f"  Source  : {self.problem_info['source']}")
        print(sep)

        # Leaderboard
        print(f"  {'Rank':<6}{'Model':<28}{'#Params':>10}", end="")
        if self.model_results:
            first = next(iter(self.model_results.values()))
            for k in sorted(first.metrics.keys()):
                print(f"  {k:>12}", end="")
        print(f"  {'Time(s)':>8}")
        print(sep)

        for entry in self.leaderboard:
            mid = entry["model"]
            r = self.model_results.get(mid)
            if r is None:
                continue
            flag = " *" if mid == self.best_model else "  "
            print(f"  {entry['rank']:<6}{mid + flag:<28}{r.n_params:>10}", end="")
            for k in sorted(r.metrics.keys()):
                v = r.metrics[k]
                print(f"  {v:>12.5f}", end="")
            print(f"  {r.training_time_s:>8.1f}")

        print(sep)
        if self.best_model:
            print(f"  Best model: {self.best_model}")
        if self.plots_saved:
            print(f"  Plots saved ({len(self.plots_saved)}):")
            for p in self.plots_saved:
                print(f"    {p}")
        print(f"{SEP}\n")

    # -- convenience ----------------------------------------------------------

    @classmethod
    def now_timestamp(cls) -> str:
        return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
