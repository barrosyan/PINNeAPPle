"""PhysicalSample validation: units, ranges, non-negativity, and monotonic coordinate checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


@dataclass
class ValidationIssue:
    """
    Single validation issue from validate_physical_sample.

    Attributes
    ----------
    level : str
        Severity: "error" or "warning".
    message : str
        Human-readable description.
    field : Optional[str]
        Optional field name the issue relates to.
    """
    level: str  # "error" | "warning"
    message: str
    field: Optional[str] = None


def _is_number(x: Any) -> bool:
    """Return True if x is an int or float."""
    return isinstance(x, (int, float))


def validate_physical_sample(
    sample,
    *,
    units_policy: str = "warn",  # "strict"|"warn"|"off"
    required_units: Optional[Iterable[str]] = None,
    ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    non_negative: Optional[Iterable[str]] = None,
    monotonic_dims: Optional[Iterable[str]] = None,  # e.g. ["pressure:z"]
) -> List[ValidationIssue]:
    """
    Validate a PhysicalSample and return a list of issues.

    Parameters
    ----------
    sample : Any
        PhysicalSample-like object with fields, coords, meta.
    units_policy : str, optional
        "strict" | "warn" | "off". Default is "warn".
    required_units : Optional[Iterable[str]], optional
        Field names that must have units metadata.
    ranges : Optional[Dict[str, Tuple[float, float]]], optional
        Allowed (lo, hi) ranges per field.
    non_negative : Optional[Iterable[str]], optional
        Field names that must be non-negative.
    monotonic_dims : Optional[Iterable[str]], optional
        Coordinate specs (e.g. "pressure:z") that must be monotonic increasing.

    Returns
    -------
    List[ValidationIssue]
        List of validation issues found.
    """
    issues: List[ValidationIssue] = []

    required_units = list(required_units or [])
    ranges = dict(ranges or {})
    non_negative = list(non_negative or [])
    monotonic_dims = list(monotonic_dims or [])

    # --- UPD version existence
    upd = getattr(sample, "meta", {}).get("upd", {})
    if "version" not in upd:
        issues.append(ValidationIssue("warning", "Missing meta.upd.version"))

    units = getattr(sample, "meta", {}).get("units", {})

    if units_policy.lower() != "off":
        # enforce that required_units fields have units metadata
        for key in required_units:
            if key not in units:
                lvl = "error" if units_policy.lower() == "strict" else "warning"
                issues.append(ValidationIssue(lvl, f"Missing units for field '{key}'", field=key))

    # --- ranges
    for k, (lo, hi) in ranges.items():
        if k not in sample.fields:
            issues.append(ValidationIssue("warning", f"Range check: field '{k}' not found", field=k))
            continue
        v = sample.fields[k]
        if torch.is_tensor(v):
            vmin = float(v.min().item()) if v.numel() else 0.0
            vmax = float(v.max().item()) if v.numel() else 0.0
            if vmin < lo or vmax > hi:
                issues.append(ValidationIssue("error", f"Field '{k}' out of range [{lo},{hi}] -> [{vmin},{vmax}]", field=k))
        elif _is_number(v):
            if v < lo or v > hi:
                issues.append(ValidationIssue("error", f"Field '{k}' out of range [{lo},{hi}] -> {v}", field=k))
        else:
            issues.append(ValidationIssue("warning", f"Range check skipped for non-tensor field '{k}'", field=k))

    # --- non-negative
    for k in non_negative:
        if k not in sample.fields:
            issues.append(ValidationIssue("warning", f"Non-negativity: field '{k}' not found", field=k))
            continue
        v = sample.fields[k]
        if torch.is_tensor(v):
            if (v < 0).any().item():
                issues.append(ValidationIssue("error", f"Field '{k}' has negative values", field=k))
        elif _is_number(v):
            if v < 0:
                issues.append(ValidationIssue("error", f"Field '{k}' is negative ({v})", field=k))

    # --- monotonic checks (simple: coordinate dimension must be monotonic increasing)
    # Format: "field:coord" or ":coord" (just coord)
    for spec in monotonic_dims:
        if ":" in spec:
            _, coord_name = spec.split(":", 1)
        else:
            coord_name = spec
        coord_name = coord_name.strip()
        if not coord_name:
            continue
        if coord_name not in sample.coords:
            issues.append(ValidationIssue("warning", f"Monotonic coord '{coord_name}' not found"))
            continue
        c = sample.coords[coord_name]
        if torch.is_tensor(c) and c.numel() > 1:
            d = c[1:] - c[:-1]
            if (d < 0).any().item():
                issues.append(ValidationIssue("error", f"Coord '{coord_name}' is not monotonic increasing"))
        # non-tensor coords are ignored

    return issues


def assert_valid_physical_sample(*args, **kwargs) -> None:
    """
    Validate a PhysicalSample and raise ValueError if any errors are found.

    Parameters
    ----------
    *args
        Positional arguments passed to validate_physical_sample.
    **kwargs
        Keyword arguments passed to validate_physical_sample.

    Raises
    ------
    ValueError
        If validation produces any errors.
    """
    issues = validate_physical_sample(*args, **kwargs)
    errors = [i for i in issues if i.level == "error"]
    if errors:
        msg = "\n".join([f"- {e.message}" for e in errors])
        raise ValueError(f"PhysicalSample validation failed:\n{msg}")
