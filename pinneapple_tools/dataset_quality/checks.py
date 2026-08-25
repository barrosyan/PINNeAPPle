"""
Generic dataset quality-assessment checks.

Operates on a plain `(coords, fields)` representation of a point-cloud
dataset (`coords`: (N, d) array of sample locations; `fields`: {name: array}
of per-point field values, each array's leading dimension of length N) — the
same shape any PDE/PINN dataset naturally takes, independent of where it came
from. No framework dependency (no orchestration/task-runner coupling).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def analyze_completeness(fields: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """NaN/missing-value fraction per field, plus an overall completeness score."""
    per_field: Dict[str, float] = {}
    for name, arr in fields.items():
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating):
            nan_frac = float(np.isnan(arr).mean()) if arr.size else 1.0
        else:
            # Non-float fields (e.g. int/bool) can't be NaN — treat as complete.
            nan_frac = 0.0
        per_field[name] = nan_frac

    overall_completeness = 1.0 - (sum(per_field.values()) / len(per_field) if per_field else 0.0)
    return {
        "missing_fraction_per_field": per_field,
        "overall_completeness": overall_completeness,
    }


_MONOTONIC_NAME_HINTS = (
    "cumulative", "cum_", "damage", "fatigue", "wear", "degradation",
    "age", "time", "elapsed", "runtime", "accumulat",
)


def _looks_monotonic_field(name: str) -> bool:
    lname = name.lower()
    return any(hint in lname for hint in _MONOTONIC_NAME_HINTS)


def validate_consistency(
    coords: np.ndarray,
    fields: Dict[str, np.ndarray],
    monotonic_name_hints: tuple = _MONOTONIC_NAME_HINTS,
) -> Dict[str, Any]:
    """Generic cross-field consistency checks:
      - shape consistency: every field array's leading dim must match coords.shape[0]
      - monotonicity heuristic: fields whose *name* suggests a cumulative/
        time-like quantity are checked for (near) non-decreasing behavior.
    This is intentionally generic/heuristic — it has no per-problem physical
    knowledge (a real physics-residual check is a separate concern).
    """
    coords = np.asarray(coords)
    n_points = coords.shape[0]

    shape_ok: Dict[str, bool] = {}
    for name, arr in fields.items():
        arr = np.asarray(arr)
        shape_ok[name] = bool(arr.shape[0] == n_points) if arr.ndim >= 1 else False

    monotonic_checks: Dict[str, Dict[str, Any]] = {}
    for name, arr in fields.items():
        lname = name.lower()
        if not any(hint in lname for hint in monotonic_name_hints):
            continue
        arr = np.asarray(arr)
        arr = arr.reshape(len(arr), -1)[:, 0] if arr.ndim > 1 else arr
        if arr.size < 2 or not np.issubdtype(arr.dtype, np.number):
            continue
        diffs = np.diff(arr.astype(float))
        # Small negative steps are tolerated as numerical noise (heuristic, not exact).
        tol = 1e-6 * (np.abs(arr).max() + 1e-12)
        violation_frac = float((diffs < -tol).mean()) if diffs.size else 0.0
        monotonic_checks[name] = {
            "expected_monotonic_nondecreasing": True,
            "violation_fraction": violation_frac,
            "passed": violation_frac == 0.0,
        }

    all_shapes_ok = all(shape_ok.values()) if shape_ok else True
    all_monotonic_ok = all(c["passed"] for c in monotonic_checks.values()) if monotonic_checks else True

    return {
        "shape_consistency": {"per_field": shape_ok, "passed": all_shapes_ok, "n_points": n_points},
        "monotonicity": {"checks": monotonic_checks, "passed": all_monotonic_ok},
        "passed": all_shapes_ok and all_monotonic_ok,
    }


def analyze_distribution(fields: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """scipy.stats.describe per field (mean/variance/skewness/kurtosis)."""
    from scipy import stats

    per_field: Dict[str, Any] = {}
    for name, arr in fields.items():
        arr = np.asarray(arr).reshape(-1).astype(float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            per_field[name] = {"note": "too few finite values to describe"}
            continue
        desc = stats.describe(arr)
        per_field[name] = {
            "nobs": int(desc.nobs),
            "min": float(desc.minmax[0]),
            "max": float(desc.minmax[1]),
            "mean": float(desc.mean),
            "variance": float(desc.variance),
            "skewness": float(desc.skewness),
            "kurtosis": float(desc.kurtosis),
        }
    return {"per_field": per_field}


def detect_outliers(fields: Dict[str, np.ndarray], random_state: int = 42) -> Dict[str, Any]:
    """IsolationForest fit jointly across all numeric fields (stacked as a
    feature matrix over points); report the fraction flagged as outliers."""
    from sklearn.ensemble import IsolationForest

    numeric_names = []
    columns = []
    n_points: Optional[int] = None
    for name, arr in fields.items():
        arr = np.asarray(arr)
        if not np.issubdtype(arr.dtype, np.number):
            continue
        col = arr.reshape(len(arr), -1)[:, 0] if arr.ndim > 1 else arr.reshape(-1)
        if n_points is None:
            n_points = col.shape[0]
        if col.shape[0] != n_points:
            continue  # inconsistent shape — validate_consistency already flags this
        numeric_names.append(name)
        columns.append(col.astype(float))

    if not columns or n_points is None or n_points < 2:
        return {"outlier_fraction": 0.0, "n_flagged": 0, "n_points": n_points or 0,
                "fields_used": numeric_names, "note": "insufficient numeric data — skipped"}

    X = np.column_stack(columns)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    forest = IsolationForest(random_state=random_state, contamination="auto", n_estimators=100)
    labels = forest.fit_predict(X)  # -1 == outlier, 1 == inlier
    n_flagged = int((labels == -1).sum())
    outlier_fraction = n_flagged / len(labels)

    return {
        "outlier_fraction": float(outlier_fraction),
        "n_flagged": n_flagged,
        "n_points": int(len(labels)),
        "fields_used": numeric_names,
    }


def field_range_issues(fields: Dict[str, np.ndarray], ranges: Dict[str, Any]) -> list:
    """Generic per-field physical-range check, keyed by the field's literal
    name: {field_name: (lo, hi)}. Returns a list of human-readable issue
    strings (empty if everything is within bounds)."""
    issues: list = []
    for name, bounds in (ranges or {}).items():
        if name not in fields:
            continue
        try:
            lo, hi = float(bounds[0]), float(bounds[1])
        except Exception:
            continue
        arr = np.asarray(fields[name], dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        amin, amax = float(arr.min()), float(arr.max())
        if amin < lo or amax > hi:
            issues.append(
                f"[range] '{name}' outside physical bounds ({lo},{hi}): min={amin:.4g} max={amax:.4g}"
            )
    return issues


def build_1d_linear_interpolant(z: np.ndarray, y: np.ndarray) -> "Any":
    """Build a differentiable (torch) piecewise-linear interpolant from a set
    of 1D-coordinate samples `z` and corresponding output samples `y`
    (shape (N,) and (N, k)). Useful for evaluating a differentiable
    PDE-residual function against static point-cloud data when no trained
    model is available — the interpolant stands in as a rough, cheap
    surrogate. This is an approximation (the true field is only piecewise-
    linear between samples), appropriate for a rough consistency signal, not
    a rigorous PDE check.
    """
    import torch

    order = np.argsort(z)
    z_sorted = torch.tensor(z[order], dtype=torch.float32)
    y_sorted = torch.tensor(y[order], dtype=torch.float32)
    dz = z_sorted[1:] - z_sorted[:-1]
    dz = torch.where(dz.abs() < 1e-12, torch.full_like(dz, 1e-12), dz)
    slopes = (y_sorted[1:] - y_sorted[:-1]) / dz.unsqueeze(-1)

    class _LinearInterp(torch.nn.Module):
        def forward(self, coords: "torch.Tensor") -> "torch.Tensor":
            zc = coords[:, 0]
            idx = torch.bucketize(zc.detach(), z_sorted, right=False)
            idx = idx.clamp(1, len(z_sorted) - 1) - 1
            z0 = z_sorted[idx]
            y0 = y_sorted[idx]
            slope = slopes[idx]
            return y0 + slope * (zc - z0).unsqueeze(-1)

    return _LinearInterp()
