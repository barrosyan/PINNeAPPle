"""
Generic model-evaluation metrics: regression performance, bootstrap
uncertainty, one-at-a-time sensitivity, and calibration — independent of any
particular model storage/registry format. Callers supply plain arrays
(`y_true`, `y_pred`, `coords`) or a plain callable (for sensitivity), not a
framework-specific artifact object.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    field_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """RMSE/MAE/relative-error/R2, overall and per output column.

    `y_true`/`y_pred`: (N,) or (N, k) arrays. `field_names`: optional length-k
    labels for the per-field breakdown (defaults to "field_0", "field_1", ...).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}")

    k = y_true.shape[1]
    names = list(field_names) if field_names is not None else [f"field_{i}" for i in range(k)]
    if len(names) != k:
        raise ValueError(f"field_names has {len(names)} entries but data has {k} columns")

    residuals = y_pred - y_true
    overall = {
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "relative_error": float(np.linalg.norm(residuals) / (np.linalg.norm(y_true) + 1e-12)),
        "r2": float(r2_score(y_true, y_pred)),
    }
    per_field = {
        names[i]: {
            "rmse": float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))),
            "mae": float(mean_absolute_error(y_true[:, i], y_pred[:, i])),
            "r2": float(r2_score(y_true[:, i], y_pred[:, i])),
        }
        for i in range(k)
    }
    return {"overall": overall, "per_field": per_field, "residuals": residuals}


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error, as a fraction (multiply by 100 for %).
    `eps` guards division by near-zero targets — MAPE dominated by that guard
    rather than genuine relative error is a sign `y_true` has values near
    zero, not a bug in this function."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs((y_pred - y_true) / (np.abs(y_true) + eps))))


def worst_n_points(
    coords: np.ndarray,
    abs_residuals: np.ndarray,
    field_names: Sequence[str],
    n: int = 20,
) -> List[Dict[str, Any]]:
    """Indices/coordinates of the n points with the largest total absolute
    error (summed across output fields) — useful for spotting where a model
    fits worst."""
    abs_residuals = np.asarray(abs_residuals)
    total = abs_residuals.sum(axis=1)
    worst_idx = np.argsort(-total)[:n]
    return [
        {
            "index": int(i),
            "coord": np.asarray(coords[i]).tolist() if coords is not None else None,
            "abs_error_per_field": {name: float(abs_residuals[i, j]) for j, name in enumerate(field_names)},
        }
        for i in worst_idx
    ]


def bootstrap_confidence_interval(
    residuals: np.ndarray,
    n_boot: int = 1000,
    ci: float = 90.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Bootstrap confidence interval on the mean residual — a simple,
    dependency-free uncertainty estimate when no dedicated UQ model is
    available."""
    residuals = np.asarray(residuals).reshape(-1)
    rng = np.random.default_rng(seed)
    boot_means = np.array(
        [rng.choice(residuals, size=residuals.size, replace=True).mean() for _ in range(n_boot)]
    )
    half = (100.0 - ci) / 2.0
    lo, hi = np.percentile(boot_means, [half, 100.0 - half])
    return {
        "method": "bootstrap", "n_boot": n_boot, "ci": ci,
        "interval": [float(lo), float(hi)], "residual_std": float(np.std(residuals)),
    }


def one_at_a_time_sensitivity(
    model_fn: Callable[[np.ndarray], np.ndarray],
    base_inputs: np.ndarray,
    param_columns: Dict[str, int],
    perturb_frac: float = 0.1,
    default_nominal: float = 1.0,
) -> Dict[str, Any]:
    """First-order, one-at-a-time sensitivity: for each named input column,
    perturb it +/- `perturb_frac` around its mean value in `base_inputs` and
    measure |model(plus) - model(minus)| / |delta_input|.

    This is a simple, hand-rolled approximation — NOT full Sobol/Saltelli
    variance-based sensitivity (see `pinneapple_data.parameter_sampling`'s
    `saltelli_perturbation_sweep` if that's what's needed instead).

    `model_fn`: callable mapping an (N, D) array to an (N, k) (or (N,)) array.
    `param_columns`: {display_name: column_index_into base_inputs}.
    """
    base_inputs = np.asarray(base_inputs, dtype=np.float32)
    sensitivities: Dict[str, Any] = {}
    for name, col in param_columns.items():
        nominal = float(np.mean(base_inputs[:, col]))
        if nominal == 0.0:
            nominal = default_nominal

        plus, minus = base_inputs.copy(), base_inputs.copy()
        plus[:, col] = nominal * (1.0 + perturb_frac)
        minus[:, col] = nominal * (1.0 - perturb_frac)

        y_plus = np.asarray(model_fn(plus))
        y_minus = np.asarray(model_fn(minus))
        delta_output = float(np.mean(np.abs(y_plus - y_minus)))
        delta_input = abs(nominal * 2.0 * perturb_frac) or 1e-12
        sensitivities[name] = {
            "nominal": nominal,
            "delta_output": delta_output,
            "sensitivity_delta_output_over_delta_input": delta_output / delta_input,
        }
    return {
        "sensitivities": sensitivities,
        "note": f"one-at-a-time +/-{perturb_frac*100:.0f}% perturbation approximation, not full Sobol/Saltelli.",
    }


def gradient_sensitivity(model: Callable, x: Any, n_out: Optional[int] = None) -> Dict[str, Any]:
    """Exact autograd-based sensitivity: mean |d output_j / d input_i| over
    the rows of `x`, computed via a single vectorized backward pass per
    output channel — not a finite-difference approximation like
    `one_at_a_time_sensitivity` above. Requires `model` to be a
    differentiable ``torch.nn.Module`` (or any callable built from
    differentiable torch ops) and `x` a ``(N, D)`` torch.Tensor.

    Complements `one_at_a_time_sensitivity`: use this when the model is
    differentiable and you want the exact local gradient rather than a
    perturbation-based estimate; use OAT when the model is a black box
    (only callable on numpy arrays).
    """
    import torch

    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    out = out.y if hasattr(out, "y") else out
    if out.ndim == 1:
        out = out.unsqueeze(1)
    in_dim, out_dim = x.shape[1], out.shape[1]
    if n_out is not None:
        out_dim = min(out_dim, n_out)

    sens = torch.zeros(in_dim, out_dim)
    for j in range(out_dim):
        retain = j < out_dim - 1
        grad_j = torch.autograd.grad(out[:, j].sum(), x, retain_graph=retain)[0]
        sens[:, j] = grad_j.abs().mean(dim=0)

    return {
        "sensitivity_matrix": sens.tolist(),  # [input_i][output_j] = mean |d out_j / d in_i|
        "most_sensitive_input_per_output": {
            f"output_{j}": int(torch.argmax(sens[:, j]).item()) for j in range(out_dim)
        },
    }


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residual_std: float,
    levels: Sequence[float] = (0.5, 0.7, 0.8, 0.9, 0.95),
) -> Dict[str, Any]:
    """Empirical coverage of a Gaussian(0, residual_std) predictive interval
    at each nominal confidence level — compares against the ideal diagonal to
    show whether a model's uncertainty estimate is well-calibrated."""
    from scipy.stats import norm

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    empirical = []
    for lvl in levels:
        z = norm.ppf(0.5 + lvl / 2.0)
        half_width = z * residual_std
        within = np.abs(y_pred - y_true) <= half_width
        empirical.append(float(np.mean(within)))
    return {"levels": list(levels), "empirical_coverage": empirical}
