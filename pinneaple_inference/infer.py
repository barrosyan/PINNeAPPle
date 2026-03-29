"""Inference utilities for trained PINN and neural network models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


@dataclass
class InferenceResult:
    """Container for model inference output.

    Attributes
    ----------
    coords : dict mapping coord name → 1-D numpy array (e.g. {"x": ..., "t": ...})
    fields : dict mapping field name → numpy array shaped (n_points, 1) or grid-shaped
    grid_shape : optional tuple giving the grid shape (e.g. (nx, nt)) when inferring on a grid
    model_id : optional string identifier for the model
    metadata : extra key/value info
    """
    coords: Dict[str, np.ndarray]
    fields: Dict[str, np.ndarray]
    grid_shape: Optional[Tuple[int, ...]] = None
    model_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def infer(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: Union[str, torch.device] = "cpu",
    field_names: Optional[List[str]] = None,
) -> np.ndarray:
    """Run a forward pass on numpy input.

    Parameters
    ----------
    model : trained torch.nn.Module
    x : (N, D) numpy array of input coordinates
    device : torch device string or object
    field_names : unused (for API consistency); raw prediction returned

    Returns
    -------
    y_hat : (N, F) numpy array of predictions
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    x_t = torch.from_numpy(x.astype(np.float32)).to(device)
    with torch.no_grad():
        out = model(x_t)
    if not isinstance(out, torch.Tensor):
        for attr in ("y", "pred", "out", "logits"):
            if hasattr(out, attr):
                out = getattr(out, attr)
                break
    return out.detach().cpu().numpy()


def infer_on_grid_1d(
    model: nn.Module,
    x_range: Tuple[float, float],
    t_range: Tuple[float, float],
    *,
    nx: int = 200,
    nt: int = 200,
    device: Union[str, torch.device] = "cpu",
    field_names: Optional[List[str]] = None,
    coord_names: Tuple[str, str] = ("x", "t"),
) -> InferenceResult:
    """Inference on a 1D+time rectangular grid.

    Returns an InferenceResult with coords["x"] and coords["t"] as 1-D arrays
    and each field reshaped to (nx, nt).
    """
    x_vals = np.linspace(x_range[0], x_range[1], nx, dtype=np.float32)
    t_vals = np.linspace(t_range[0], t_range[1], nt, dtype=np.float32)
    XX, TT = np.meshgrid(x_vals, t_vals, indexing="ij")  # shape (nx, nt) each
    pts = np.stack([XX.ravel(), TT.ravel()], axis=1)  # (nx*nt, 2)

    y_hat = infer(model, pts, device=device)  # (nx*nt, F)
    n_fields = y_hat.shape[1] if y_hat.ndim > 1 else 1
    if field_names is None:
        field_names = [f"field_{i}" for i in range(n_fields)]

    fields = {}
    for i, fname in enumerate(field_names):
        if y_hat.ndim > 1 and i < y_hat.shape[1]:
            fields[fname] = y_hat[:, i].reshape(nx, nt)
        elif y_hat.ndim == 1:
            fields[fname] = y_hat.reshape(nx, nt)

    return InferenceResult(
        coords={coord_names[0]: x_vals, coord_names[1]: t_vals},
        fields=fields,
        grid_shape=(nx, nt),
    )


def infer_on_grid_2d(
    model: nn.Module,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    *,
    nx: int = 100,
    ny: int = 100,
    device: Union[str, torch.device] = "cpu",
    field_names: Optional[List[str]] = None,
    coord_names: Tuple[str, str] = ("x", "y"),
) -> InferenceResult:
    """Inference on a 2D rectangular grid.

    Returns InferenceResult with coords["x"], coords["y"] as 1-D arrays
    and each field reshaped to (nx, ny).
    """
    x_vals = np.linspace(x_range[0], x_range[1], nx, dtype=np.float32)
    y_vals = np.linspace(y_range[0], y_range[1], ny, dtype=np.float32)
    XX, YY = np.meshgrid(x_vals, y_vals, indexing="ij")
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)

    y_hat = infer(model, pts, device=device)
    n_fields = y_hat.shape[1] if y_hat.ndim > 1 else 1
    if field_names is None:
        field_names = [f"field_{i}" for i in range(n_fields)]

    fields = {}
    for i, fname in enumerate(field_names):
        if y_hat.ndim > 1 and i < y_hat.shape[1]:
            fields[fname] = y_hat[:, i].reshape(nx, ny)
        elif y_hat.ndim == 1:
            fields[fname] = y_hat.reshape(nx, ny)

    return InferenceResult(
        coords={coord_names[0]: x_vals, coord_names[1]: y_vals},
        fields=fields,
        grid_shape=(nx, ny),
    )
