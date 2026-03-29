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
