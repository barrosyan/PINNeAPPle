from __future__ import annotations
"""Comparison of trained PINN models to analytical solutions and solver data."""

from typing import Any, Callable, Dict, List, Optional

import numpy as np


def compare_to_analytical(
    model,
    analytical_fn: Callable[[np.ndarray], np.ndarray],
    coord_names: List[str],
    domain_bounds: Dict[str, tuple],
    n_points: int = 5_000,
    field_names: Optional[List[str]] = None,
    device: str = "cpu",
) -> dict:
    """Compare model predictions to an analytical solution.

    Parameters
    ----------
    model : trained nn.Module or callable (numpy -> numpy)
    analytical_fn : callable (N, D) -> (N, F) giving true field values
    coord_names : list of coordinate dimension names
    domain_bounds : dict {coord: (lo, hi)}
    n_points : number of test points (LHS sampled)
    field_names : field names for per-field error reporting
    device : torch device for model inference

    Returns
    -------
    dict with keys:
        rmse, rel_l2, max_error, mean_error, field_errors (per-field if multi-output)
    """
    import torch

    # LHS sampling
    rng = np.random.default_rng(42)
    D = len(coord_names)
    lo = np.array([domain_bounds[c][0] for c in coord_names], dtype=np.float64)
    hi = np.array([domain_bounds[c][1] for c in coord_names], dtype=np.float64)
    pts = np.zeros((n_points, D), dtype=np.float32)
    for j in range(D):
        perm = rng.permutation(n_points)
        pts[:, j] = (lo[j] + (perm + rng.random(n_points)) / n_points * (hi[j] - lo[j])).astype(np.float32)

    # Analytical reference
    y_ref = np.asarray(analytical_fn(pts), dtype=np.float32)
    if y_ref.ndim == 1:
        y_ref = y_ref[:, None]

    # Model prediction
    x_t = torch.from_numpy(pts).to(device)
    with torch.no_grad():
        out = model(x_t)
        if hasattr(out, "y"):
            out = out.y
        if not isinstance(out, torch.Tensor):
            out = torch.tensor(out, dtype=torch.float32)
        y_pred = out.cpu().numpy()
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    # Trim to same shape
    F = min(y_pred.shape[1], y_ref.shape[1])
    y_pred = y_pred[:, :F]
    y_ref = y_ref[:, :F]

    err = y_pred - y_ref
    rmse = float(np.sqrt(np.mean(err ** 2)))
    rel_l2 = float(np.linalg.norm(err) / (np.linalg.norm(y_ref) + 1e-12))
    max_error = float(np.abs(err).max())
    mean_error = float(np.abs(err).mean())

    result: dict = {
        "rmse": rmse,
        "rel_l2": rel_l2,
        "max_error": max_error,
        "mean_error": mean_error,
        "n_points": n_points,
    }

    if F > 1:
        names = field_names or [f"field_{i}" for i in range(F)]
        result["field_errors"] = {}
        for i, name in enumerate(names[:F]):
            e = y_pred[:, i] - y_ref[:, i]
            result["field_errors"][name] = {
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "max_error": float(np.abs(e).max()),
            }

    return result


def validate_against_solver(
    model,
    solver_data: Dict[str, Any],
    coord_names: List[str],
    field_names: List[str],
    device: str = "cpu",
) -> dict:
    """Compare model predictions to solver reference data.

    Parameters
    ----------
    model : trained nn.Module
    solver_data : dict with coord arrays and field arrays.
        Expected keys: coord_names (e.g. "x", "y") + field_names (e.g. "u", "p")
    coord_names : list of coordinate keys present in solver_data
    field_names : list of field keys present in solver_data
    device : torch device

    Returns
    -------
    dict with per-field and aggregate errors
    """
    import torch

    # Build input tensor
    cols = []
    for c in coord_names:
        arr = np.asarray(solver_data[c], dtype=np.float32).ravel()
        cols.append(arr[:, None])
    x_np = np.concatenate(cols, axis=1)
    N = x_np.shape[0]

    x_t = torch.from_numpy(x_np).to(device)
    with torch.no_grad():
        out = model(x_t)
        if hasattr(out, "y"):
            out = out.y
        if not isinstance(out, torch.Tensor):
            out = torch.tensor(out, dtype=torch.float32)
        y_pred = out.cpu().numpy()
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    results: dict = {"n_points": N, "field_errors": {}}
    all_rmse = []

    for i, fname in enumerate(field_names):
        if fname not in solver_data:
            continue
        y_ref = np.asarray(solver_data[fname], dtype=np.float32).ravel()
        if i >= y_pred.shape[1]:
            break
        y_p = y_pred[:, i]
        e = y_p - y_ref
        rmse = float(np.sqrt(np.mean(e ** 2)))
        rel_l2 = float(np.linalg.norm(e) / (np.linalg.norm(y_ref) + 1e-12))
        results["field_errors"][fname] = {
            "rmse": rmse,
            "rel_l2": rel_l2,
            "max_error": float(np.abs(e).max()),
            "mean_error": float(np.abs(e).mean()),
        }
        all_rmse.append(rmse)

    results["rmse"] = float(np.mean(all_rmse)) if all_rmse else float("nan")
    results["rel_l2"] = float(np.mean([
        v["rel_l2"] for v in results["field_errors"].values()
    ])) if results["field_errors"] else float("nan")

    return results
