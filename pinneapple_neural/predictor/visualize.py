"""Visualization utilities for PINN inference results."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .infer import InferenceResult


def _get_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib") from e


def plot_field_1d(
    result: InferenceResult,
    field_name: str,
    *,
    ax=None,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Any:
    """2D colormesh plot of a 1D+time field (x vs t).

    Expects result.coords has keys matching the two coord dimensions,
    and result.fields[field_name] has shape (nx, nt).
    """
    plt = _get_mpl()
    coord_keys = list(result.coords.keys())

    if field_name not in result.fields:
        raise KeyError(f"Field '{field_name}' not found. Available: {list(result.fields.keys())}")

    Z = result.fields[field_name]  # (nx, nt) or (ny, nx)
    c0, c1 = coord_keys[0], coord_keys[1]
    x_vals = result.coords[c0]
    t_vals = result.coords[c1]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    else:
        fig = ax.figure

    XX, TT = np.meshgrid(x_vals, t_vals, indexing="ij")
    im = ax.pcolormesh(XX, TT, Z, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax, label=field_name)
    ax.set_xlabel(c0)
    ax.set_ylabel(c1)
    ax.set_title(title or f"{field_name} — {result.model_id}")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if own_fig and not show:
        plt.close(fig)
    return fig


def plot_field_2d(
    result: InferenceResult,
    field_name: str,
    *,
    ax=None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Any:
    """Colormesh plot of a 2D field (x vs y).

    Expects result.coords has 'x' and 'y' keys,
    and result.fields[field_name] has shape (nx, ny).
    """
    plt = _get_mpl()
    coord_keys = list(result.coords.keys())
    if field_name not in result.fields:
        raise KeyError(f"Field '{field_name}' not found. Available: {list(result.fields.keys())}")

    Z = result.fields[field_name]
    c0, c1 = coord_keys[0], coord_keys[1]
    x_vals = result.coords[c0]
    y_vals = result.coords[c1]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure

    XX, YY = np.meshgrid(x_vals, y_vals, indexing="ij")
    im = ax.pcolormesh(XX, YY, Z, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax, label=field_name)
    ax.set_xlabel(c0)
    ax.set_ylabel(c1)
    ax.set_title(title or f"{field_name} — {result.model_id}")
    ax.set_aspect("equal")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if own_fig and not show:
        plt.close(fig)
    return fig


def plot_error_map_1d(
    result: InferenceResult,
    reference: InferenceResult,
    field_name: str,
    *,
    ax=None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Any:
    """Plot absolute error |prediction - reference| on a 1D+time grid."""
    plt = _get_mpl()
    coord_keys = list(result.coords.keys())

    pred = result.fields[field_name]
    ref = reference.fields[field_name]
    error = np.abs(pred - ref)

    c0, c1 = coord_keys[0], coord_keys[1]
    x_vals = result.coords[c0]
    t_vals = result.coords[c1]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    else:
        fig = ax.figure

    XX, TT = np.meshgrid(x_vals, t_vals, indexing="ij")
    im = ax.pcolormesh(XX, TT, error, cmap="hot_r", shading="auto")
    fig.colorbar(im, ax=ax, label=f"|error| {field_name}")
    ax.set_xlabel(c0)
    ax.set_ylabel(c1)
    ax.set_title(title or f"Error map — {field_name} — {result.model_id}")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if own_fig and not show:
        plt.close(fig)
    return fig


def plot_error_map_2d(
    result: InferenceResult,
    reference: InferenceResult,
    field_name: str,
    *,
    ax=None,
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Any:
    """Plot absolute error for a 2D field."""
    plt = _get_mpl()
    coord_keys = list(result.coords.keys())

    pred = result.fields[field_name]
    ref = reference.fields[field_name]
    error = np.abs(pred - ref)

    c0, c1 = coord_keys[0], coord_keys[1]
    x_vals = result.coords[c0]
    y_vals = result.coords[c1]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure

    XX, YY = np.meshgrid(x_vals, y_vals, indexing="ij")
    im = ax.pcolormesh(XX, YY, error, cmap="hot_r", shading="auto")
    fig.colorbar(im, ax=ax, label=f"|error| {field_name}")
    ax.set_xlabel(c0)
    ax.set_ylabel(c1)
    ax.set_title(title or f"Error map — {field_name} — {result.model_id}")
    ax.set_aspect("equal")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if own_fig and not show:
        plt.close(fig)
    return fig


def plot_loss_curve(
    history: List[Dict[str, float]],
    *,
    keys: Optional[List[str]] = None,
    ax=None,
    title: str = "Training loss",
    log_scale: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Any:
    """Plot training/validation loss curves from a history list.

    Parameters
    ----------
    history : list of dicts, each entry is one epoch's metrics
    keys : which keys to plot (default: train_total and val_total)
    """
    plt = _get_mpl()
    if keys is None:
        keys = ["train_total", "val_total"]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    else:
        fig = ax.figure

    epochs = list(range(len(history)))
    for k in keys:
        vals = [h.get(k) for h in history if h.get(k) is not None]
        if vals:
            ep = [i for i, h in enumerate(history) if h.get(k) is not None]
            ax.plot(ep, vals, label=k)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if own_fig and not show:
        plt.close(fig)
    return fig


def plot_model_comparison_1d(
    results: Dict[str, InferenceResult],
    field_name: str,
    *,
    reference: Optional[InferenceResult] = None,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Any:
    """Side-by-side comparison of multiple models on a 1D+time field.

    If `reference` is provided, an extra column shows the reference solution.
    """
    plt = _get_mpl()
    model_ids = list(results.keys())
    ncols = len(model_ids) + (1 if reference is not None else 0)

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), squeeze=False)
    axes = axes[0]

    col = 0
    if reference is not None:
        plot_field_1d(reference, field_name, ax=axes[0], title="Reference", cmap=cmap)
        col = 1

    for mid, res in results.items():
        plot_field_1d(res, field_name, ax=axes[col], title=mid, cmap=cmap)
        col += 1

    fig.suptitle(title or f"Model comparison — {field_name}", y=1.01)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if not show:
        plt.close(fig)
    return fig


def plot_model_comparison_2d(
    results: Dict[str, InferenceResult],
    field_name: str,
    *,
    reference: Optional[InferenceResult] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> Any:
    """Side-by-side comparison for 2D fields."""
    plt = _get_mpl()
    model_ids = list(results.keys())
    ncols = len(model_ids) + (1 if reference is not None else 0)

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False)
    axes = axes[0]

    col = 0
    if reference is not None:
        plot_field_2d(reference, field_name, ax=axes[0], title="Reference", cmap=cmap)
        col = 1

    for mid, res in results.items():
        plot_field_2d(res, field_name, ax=axes[col], title=mid, cmap=cmap)
        col += 1

    fig.suptitle(title or f"Model comparison — {field_name}", y=1.01)
    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if not show:
        plt.close(fig)
    return fig


def render_visualizations(
    viz_cfgs: List[Dict[str, Any]],
    *,
    model_results: Dict[str, Dict[str, Any]],  # model_id -> {result, history, ...}
    problem_spec: Any = None,
    reference_result: Optional[InferenceResult] = None,
    out_dir: Union[str, Path] = ".",
) -> Dict[str, str]:
    """Dispatch visualization from YAML-driven config list.

    Each entry in viz_cfgs has at minimum a `type` key. Supported types:

    - ``solution_field``: calls plot_field_1d / plot_field_2d for each model
    - ``error_map``: calls plot_error_map_1d / plot_error_map_2d (requires reference)
    - ``loss_curve``: calls plot_loss_curve for each model
    - ``model_comparison``: calls plot_model_comparison_1d / _2d

    Returns a dict mapping description → saved file path.
    """
    out_dir = Path(out_dir)
    saved: Dict[str, str] = {}

    # detect dimensionality from first available inference result
    is_2d = False
    for mid, mdata in model_results.items():
        res = mdata.get("inference_result")
        if res is not None and res.grid_shape is not None:
            coord_keys = list(res.coords.keys())
            if "y" in coord_keys:
                is_2d = True
            break

    for cfg in viz_cfgs:
        vtype = str(cfg.get("type", "")).lower()
        fields = cfg.get("fields", None)
        field_list = fields if isinstance(fields, list) else (list(model_results[list(model_results.keys())[0]]["inference_result"].fields.keys()) if model_results else [])

        if vtype == "solution_field":
            for mid, mdata in model_results.items():
                res = mdata.get("inference_result")
                if res is None:
                    continue
                res.model_id = mid
                for fname in field_list:
                    if fname not in res.fields:
                        continue
                    sp = out_dir / f"solution_{fname}_{mid}.png"
                    if is_2d:
                        plot_field_2d(res, fname, save_path=sp)
                    else:
                        plot_field_1d(res, fname, save_path=sp)
                    saved[f"solution_{fname}_{mid}"] = str(sp)

        elif vtype == "error_map":
            if reference_result is None:
                continue
            for mid, mdata in model_results.items():
                res = mdata.get("inference_result")
                if res is None:
                    continue
                for fname in field_list:
                    if fname not in res.fields or fname not in reference_result.fields:
                        continue
                    sp = out_dir / f"error_{fname}_{mid}.png"
                    if is_2d:
                        plot_error_map_2d(res, reference_result, fname, save_path=sp)
                    else:
                        plot_error_map_1d(res, reference_result, fname, save_path=sp)
                    saved[f"error_{fname}_{mid}"] = str(sp)

        elif vtype == "loss_curve":
            for mid, mdata in model_results.items():
                history = mdata.get("loss_history", [])
                if not history:
                    continue
                sp = out_dir / f"loss_curve_{mid}.png"
                plot_loss_curve(history, save_path=sp, title=f"Loss — {mid}")
                saved[f"loss_curve_{mid}"] = str(sp)

        elif vtype == "model_comparison":
            all_res = {mid: mdata["inference_result"] for mid, mdata in model_results.items() if mdata.get("inference_result") is not None}
            for fname in field_list:
                sp = out_dir / f"comparison_{fname}.png"
                if is_2d:
                    plot_model_comparison_2d(all_res, fname, reference=reference_result, save_path=sp)
                else:
                    plot_model_comparison_1d(all_res, fname, reference=reference_result, save_path=sp)
                saved[f"comparison_{fname}"] = str(sp)

    return saved


# ---------------------------------------------------------------------------
# 3D internal-flow visualization helpers
# ---------------------------------------------------------------------------

def plot_velocity_slice(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    slice_axis: str = "z",
    slice_idx: Optional[int] = None,
    component: int = 0,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    ax=None,
    save_path=None,
    show: bool = False,
) -> Any:
    """2D colormesh of a velocity slice through a 3D field.

    Extracts the midplane (or specified index) slice and plots it.

    Parameters
    ----------
    u:
        Velocity array, shape ``(nx, ny, nz)`` for a single component or
        ``(nx, ny, nz, 3)`` for all three components.
    x, y, z:
        1-D coordinate arrays of lengths nx, ny, nz respectively.
    slice_axis:
        Axis to slice along — ``"x"``, ``"y"``, or ``"z"``.
    slice_idx:
        Index along *slice_axis*; defaults to the midplane when ``None``.
    component:
        Which velocity component to plot when *u* has shape ``(nx, ny, nz, 3)``.
        Ignored when *u* is 3-D.
    """
    plt = _get_mpl()
    u = np.asarray(u)

    # If 4-D input, extract the requested component
    if u.ndim == 4:
        u = u[..., component]

    axis_map = {"x": 0, "y": 1, "z": 2}
    if slice_axis not in axis_map:
        raise ValueError(f"slice_axis must be 'x', 'y', or 'z', got '{slice_axis}'")
    ax_idx = axis_map[slice_axis]

    coords = [x, y, z]
    dim_size = u.shape[ax_idx]
    if slice_idx is None:
        slice_idx = dim_size // 2
    slice_idx = int(np.clip(slice_idx, 0, dim_size - 1))

    # Extract 2-D slice
    if ax_idx == 0:
        field_2d = u[slice_idx, :, :]   # (ny, nz)
        h_coord, v_coord = z, y
        xlabel, ylabel = "z", "y"
    elif ax_idx == 1:
        field_2d = u[:, slice_idx, :]   # (nx, nz)
        h_coord, v_coord = z, x
        xlabel, ylabel = "z", "x"
    else:
        field_2d = u[:, :, slice_idx]   # (nx, ny)
        h_coord, v_coord = y, x
        xlabel, ylabel = "y", "x"

    HH, VV = np.meshgrid(h_coord, v_coord, indexing="ij")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.pcolormesh(HH, VV, field_2d, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    component_labels = ["u", "v", "w"]
    comp_label = component_labels[component] if 0 <= component < 3 else f"comp{component}"
    ax.set_title(title or f"{comp_label} — {slice_axis}={slice_idx}")
    ax.set_aspect("equal")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if own_fig and not show:
        plt.close(fig)
    return fig


def plot_velocity_magnitude_slice(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    slice_axis: str = "z",
    slice_idx: Optional[int] = None,
    title: Optional[str] = None,
    cmap: str = "viridis",
    ax=None,
    save_path=None,
    show: bool = False,
) -> Any:
    """Plot |u| = sqrt(u^2 + v^2 + w^2) on a 2D slice.

    Parameters
    ----------
    u:
        All three velocity components, shape ``(nx, ny, nz, 3)``.
    """
    plt = _get_mpl()
    u = np.asarray(u)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError("u must have shape (nx, ny, nz, 3)")

    mag = np.sqrt(np.sum(u ** 2, axis=-1))  # (nx, ny, nz)

    axis_map = {"x": 0, "y": 1, "z": 2}
    if slice_axis not in axis_map:
        raise ValueError(f"slice_axis must be 'x', 'y', or 'z', got '{slice_axis}'")
    ax_idx = axis_map[slice_axis]

    dim_size = mag.shape[ax_idx]
    if slice_idx is None:
        slice_idx = dim_size // 2
    slice_idx = int(np.clip(slice_idx, 0, dim_size - 1))

    if ax_idx == 0:
        field_2d = mag[slice_idx, :, :]
        h_coord, v_coord = z, y
        xlabel, ylabel = "z", "y"
    elif ax_idx == 1:
        field_2d = mag[:, slice_idx, :]
        h_coord, v_coord = z, x
        xlabel, ylabel = "z", "x"
    else:
        field_2d = mag[:, :, slice_idx]
        h_coord, v_coord = y, x
        xlabel, ylabel = "y", "x"

    HH, VV = np.meshgrid(h_coord, v_coord, indexing="ij")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure

    im = ax.pcolormesh(HH, VV, field_2d, cmap=cmap, shading="auto")
    fig.colorbar(im, ax=ax, label="|u|")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"|u| — {slice_axis}={slice_idx}")
    ax.set_aspect("equal")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if own_fig and not show:
        plt.close(fig)
    return fig


def plot_streamlines_2d(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    density: float = 1.5,
    title: Optional[str] = None,
    background_field: Optional[np.ndarray] = None,
    cmap: str = "viridis",
    ax=None,
    save_path=None,
    show: bool = False,
) -> Any:
    """Streamline plot for 2D velocity field.

    Uses matplotlib streamplot.  Optionally overlays a scalar background field
    (e.g. pressure, vorticity, velocity magnitude).

    Parameters
    ----------
    u, v:
        Velocity components, shape ``(nx, ny)``.  *u* is the x-component, *v*
        is the y-component.
    x, y:
        1-D coordinate arrays of lengths nx and ny respectively.
    density:
        Streamline density passed to ``ax.streamplot``.
    background_field:
        Optional scalar field ``(nx, ny)`` drawn as a pcolormesh beneath the
        streamlines.
    """
    plt = _get_mpl()
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # streamplot expects (Y, X) meshgrids when indexing='xy'
    # u/v must be shaped (ny, nx) for streamplot's default 'xy' convention
    # Our arrays are (nx, ny); transpose for streamplot.
    XX, YY = np.meshgrid(x, y, indexing="xy")  # both (ny, nx)
    u_plot = u.T   # (ny, nx)
    v_plot = v.T   # (ny, nx)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure

    if background_field is not None:
        bg = np.asarray(background_field, dtype=float)
        im = ax.pcolormesh(XX, YY, bg.T, cmap=cmap, shading="auto", alpha=0.6)
        fig.colorbar(im, ax=ax)

    ax.streamplot(XX, YY, u_plot, v_plot, density=density, color="k", linewidth=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title or "Streamlines")
    ax.set_aspect("equal")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if own_fig and not show:
        plt.close(fig)
    return fig


def plot_centerline_velocity(
    u_centerline: np.ndarray,
    coord: np.ndarray,
    *,
    reference_coord: Optional[np.ndarray] = None,
    reference_u: Optional[np.ndarray] = None,
    reference_label: str = "Ghia et al. (1982)",
    xlabel: str = "y/L",
    ylabel: str = "u/U_lid",
    title: str = "Centerline velocity profile",
    ax=None,
    save_path=None,
    show: bool = False,
) -> Any:
    """1D centerline velocity profile with optional literature reference overlay.

    Standard validation plot for the Lid-Driven Cavity benchmark.

    Parameters
    ----------
    u_centerline:
        Predicted velocity values along the centreline, shape ``(N,)``.
    coord:
        Coordinate values (e.g. y/L positions), shape ``(N,)``.
    reference_coord, reference_u:
        Optional reference data arrays for comparison (e.g. Ghia et al.).
    """
    plt = _get_mpl()
    u_centerline = np.asarray(u_centerline, dtype=float)
    coord = np.asarray(coord, dtype=float)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    else:
        fig = ax.figure

    ax.plot(u_centerline, coord, "-", color="steelblue", linewidth=1.8, label="Prediction")

    if reference_coord is not None and reference_u is not None:
        ref_c = np.asarray(reference_coord, dtype=float)
        ref_u = np.asarray(reference_u, dtype=float)
        ax.plot(ref_u, ref_c, "o", color="firebrick", markersize=5,
                markerfacecolor="none", label=reference_label)

    ax.set_xlabel(ylabel)   # u on x-axis (standard LDC plot orientation)
    ax.set_ylabel(xlabel)   # y on y-axis
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if own_fig and not show:
        plt.close(fig)
    return fig


def plot_vorticity_slice(
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    slice_axis: str = "z",
    slice_idx: Optional[int] = None,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    ax=None,
    save_path=None,
    show: bool = False,
) -> Any:
    """Plot z-component of vorticity (omega_z = dv/dx - du/dy) on a 2D slice.

    Useful for visualizing vortex structures in LDC and channel flows.
    Partial derivatives are computed via second-order central finite differences.

    Parameters
    ----------
    u, v:
        x- and y-velocity components, shape ``(nx, ny, nz)``.
    """
    plt = _get_mpl()
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    axis_map = {"x": 0, "y": 1, "z": 2}
    if slice_axis not in axis_map:
        raise ValueError(f"slice_axis must be 'x', 'y', or 'z', got '{slice_axis}'")
    ax_idx = axis_map[slice_axis]
    dim_size = u.shape[ax_idx]
    if slice_idx is None:
        slice_idx = dim_size // 2
    slice_idx = int(np.clip(slice_idx, 0, dim_size - 1))

    # Extract 2-D slices of u and v
    if ax_idx == 0:
        u2 = u[slice_idx, :, :]   # (ny, nz)
        v2 = v[slice_idx, :, :]
        h_coord, v_coord = z, y
        xlabel, ylabel = "z", "y"
        # dv/dx not computable (sliced); approximate with zeros
        dv_dx = np.zeros_like(u2)
        du_dy = np.gradient(u2, y, axis=0)
    elif ax_idx == 1:
        u2 = u[:, slice_idx, :]   # (nx, nz)
        v2 = v[:, slice_idx, :]
        h_coord, v_coord = z, x
        xlabel, ylabel = "z", "x"
        dv_dx = np.zeros_like(u2)
        du_dy = np.zeros_like(u2)
    else:
        u2 = u[:, :, slice_idx]   # (nx, ny)
        v2 = v[:, :, slice_idx]
        h_coord, v_coord = y, x
        xlabel, ylabel = "y", "x"
        # omega_z = dv/dx - du/dy
        dv_dx = np.gradient(v2, x, axis=0)
        du_dy = np.gradient(u2, y, axis=1)

    omega_z = dv_dx - du_dy
    HH, VV = np.meshgrid(h_coord, v_coord, indexing="ij")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure

    vmax = float(np.max(np.abs(omega_z))) or 1.0
    im = ax.pcolormesh(HH, VV, omega_z, cmap=cmap, shading="auto",
                       vmin=-vmax, vmax=vmax)
    fig.colorbar(im, ax=ax, label="omega_z")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Vorticity omega_z — {slice_axis}={slice_idx}")
    ax.set_aspect("equal")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if own_fig and not show:
        plt.close(fig)
    return fig


def plot_internal_flow_summary(
    solver_output: Any,
    *,
    time_step: int = -1,
    slice_axis: str = "z",
    title: str = "Internal Flow Summary",
    save_path=None,
    show: bool = False,
) -> Any:
    """Multi-panel summary figure for 3D internal flow (LDC or channel).

    4-panel layout:

    - Panel 1: velocity magnitude |u| on midplane slice
    - Panel 2: streamlines of (u, v) on midplane
    - Panel 3: pressure field on midplane (or blank if unavailable)
    - Panel 4: centreline u-velocity profile

    Parameters
    ----------
    solver_output:
        ``SolverOutput3D`` with:

        - ``.u``  shape ``(nt+1, 3, nx, ny, nz)`` — velocity ``[vx, vy, vz]``
        - ``.coords`` dict with keys ``"x"``, ``"y"``, ``"z"``, ``"t"``
        - ``.meta`` may contain ``"p_final"`` of shape ``(nx, ny, nz)``
    time_step:
        Which time index to visualise; ``-1`` selects the last step.
    slice_axis:
        Axis for the midplane slice (``"x"``, ``"y"``, or ``"z"``).
    """
    plt = _get_mpl()

    coords = solver_output.coords
    x = np.asarray(coords["x"], dtype=float)
    y = np.asarray(coords["y"], dtype=float)
    z = np.asarray(coords["z"], dtype=float)

    # u shape: (nt+1, 3, nx, ny, nz) — select time step
    vel_all = np.asarray(solver_output.u, dtype=float)
    vel = vel_all[time_step]  # (3, nx, ny, nz)

    # Rearrange to (nx, ny, nz, 3)
    vel_xyzc = np.moveaxis(vel, 0, -1)  # (nx, ny, nz, 3)

    ux = vel_xyzc[..., 0]  # (nx, ny, nz)
    uy = vel_xyzc[..., 1]
    # uz = vel_xyzc[..., 2]  # not used in 2-D panels below

    # Determine midplane index along slice_axis
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax_idx = axis_map.get(slice_axis, 2)
    mid_idx = ux.shape[ax_idx] // 2

    # Extract 2-D slice of (ux, uy) for streamlines and pressure
    if ax_idx == 0:
        ux2 = ux[mid_idx, :, :]   # (ny, nz) — use y,z plane
        uy2 = uy[mid_idx, :, :]
        h_coord, v_coord = z, y
        cl_u = ux[mid_idx, :, mid_idx]   # u along y-axis at x=mid, z=mid
        cl_coord = y / (y[-1] if y[-1] != 0 else 1.0)
    elif ax_idx == 1:
        ux2 = ux[:, mid_idx, :]   # (nx, nz)
        uy2 = uy[:, mid_idx, :]
        h_coord, v_coord = z, x
        cl_u = ux[:, mid_idx, mid_idx]
        cl_coord = x / (x[-1] if x[-1] != 0 else 1.0)
    else:
        ux2 = ux[:, :, mid_idx]   # (nx, ny)
        uy2 = uy[:, :, mid_idx]
        h_coord, v_coord = y, x
        cl_u = ux[:, ux.shape[1] // 2, mid_idx]   # u along x at y=mid, z=mid
        cl_coord = x / (x[-1] if x[-1] != 0 else 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=14)

    # --- Panel 1: velocity magnitude ---
    ax1 = axes[0, 0]
    mag2 = np.sqrt(ux2 ** 2 + uy2 ** 2)
    HH, VV = np.meshgrid(h_coord, v_coord, indexing="ij")
    im1 = ax1.pcolormesh(HH, VV, mag2, cmap="viridis", shading="auto")
    fig.colorbar(im1, ax=ax1, label="|u|")
    ax1.set_title("Velocity magnitude")
    ax1.set_aspect("equal")

    # --- Panel 2: streamlines ---
    ax2 = axes[0, 1]
    # streamplot needs (ny_plot, nx_plot) arrays with meshgrid(h, v, indexing='xy')
    XX_s, YY_s = np.meshgrid(h_coord, v_coord, indexing="xy")
    try:
        ax2.streamplot(XX_s, YY_s, ux2.T, uy2.T, density=1.5,
                       color="k", linewidth=0.7)
    except Exception:
        ax2.text(0.5, 0.5, "Streamplot unavailable", ha="center", va="center",
                 transform=ax2.transAxes)
    ax2.set_title("Streamlines (u, v)")
    ax2.set_aspect("equal")

    # --- Panel 3: pressure ---
    ax3 = axes[1, 0]
    meta = getattr(solver_output, "meta", {}) or {}
    p_final = meta.get("p_final", None)
    if p_final is not None:
        p_arr = np.asarray(p_final, dtype=float)
        if ax_idx == 0:
            p2 = p_arr[mid_idx, :, :]
        elif ax_idx == 1:
            p2 = p_arr[:, mid_idx, :]
        else:
            p2 = p_arr[:, :, mid_idx]
        im3 = ax3.pcolormesh(HH, VV, p2, cmap="RdBu_r", shading="auto")
        fig.colorbar(im3, ax=ax3, label="p")
        ax3.set_title("Pressure")
        ax3.set_aspect("equal")
    else:
        ax3.set_axis_off()
        ax3.text(0.5, 0.5, "No pressure data", ha="center", va="center",
                 fontsize=12, transform=ax3.transAxes)
        ax3.set_title("Pressure")

    # --- Panel 4: centreline u-velocity ---
    ax4 = axes[1, 1]
    u_lid = float(np.max(np.abs(ux))) or 1.0
    ax4.plot(cl_u / u_lid, cl_coord, "-", color="steelblue", linewidth=1.8,
             label="Prediction")
    ax4.set_xlabel("u/U_lid")
    ax4.set_ylabel("coordinate/L")
    ax4.set_title("Centreline u-velocity")
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if not show:
        plt.close(fig)
    return fig


def plot_design_opt_convergence(
    history_objectives: List[float],
    *,
    title: str = "Design Optimization Convergence",
    xlabel: str = "Iteration",
    ylabel: str = "Objective",
    log_scale: bool = False,
    ax=None,
    save_path=None,
    show: bool = False,
) -> Any:
    """Plot convergence curve of objective function over optimization iterations.

    Parameters
    ----------
    history_objectives:
        Sequence of objective values, one per iteration.
    log_scale:
        If ``True``, use a logarithmic y-axis.
    """
    plt = _get_mpl()
    objectives = np.asarray(history_objectives, dtype=float)
    iterations = np.arange(len(objectives))

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    else:
        fig = ax.figure

    ax.plot(iterations, objectives, "-o", color="steelblue", linewidth=1.5,
            markersize=3, markevery=max(1, len(objectives) // 50))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if log_scale:
        ax.set_yscale("log")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if own_fig and not show:
        plt.close(fig)
    return fig


def plot_pareto_front_2d(
    objectives: np.ndarray,
    pareto_mask: np.ndarray,
    *,
    xlabel: str = "Objective 1",
    ylabel: str = "Objective 2",
    title: str = "Pareto Front",
    ax=None,
    save_path=None,
    show: bool = False,
) -> Any:
    """Scatter plot with Pareto-optimal points highlighted.

    Parameters
    ----------
    objectives:
        Array of shape ``(N, 2)`` containing pairs of objective values.
    pareto_mask:
        Boolean array of shape ``(N,)``; ``True`` marks Pareto-optimal points.
    """
    plt = _get_mpl()
    objectives = np.asarray(objectives, dtype=float)
    pareto_mask = np.asarray(pareto_mask, dtype=bool)

    if objectives.ndim != 2 or objectives.shape[1] != 2:
        raise ValueError("objectives must have shape (N, 2)")
    if pareto_mask.shape[0] != objectives.shape[0]:
        raise ValueError("pareto_mask length must match objectives row count")

    dominated = ~pareto_mask
    pareto = pareto_mask

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    else:
        fig = ax.figure

    if dominated.any():
        ax.scatter(
            objectives[dominated, 0], objectives[dominated, 1],
            color="lightsteelblue", edgecolors="none", alpha=0.6,
            label="Dominated", s=25,
        )
    if pareto.any():
        # Sort Pareto points by first objective for a connected front line
        pareto_pts = objectives[pareto]
        order = np.argsort(pareto_pts[:, 0])
        pareto_sorted = pareto_pts[order]
        ax.scatter(
            pareto_sorted[:, 0], pareto_sorted[:, 1],
            color="firebrick", edgecolors="darkred", alpha=0.9,
            label="Pareto-optimal", s=40, zorder=3,
        )
        ax.plot(pareto_sorted[:, 0], pareto_sorted[:, 1],
                color="firebrick", linewidth=1.2, zorder=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    if own_fig and not show:
        plt.close(fig)
    return fig
