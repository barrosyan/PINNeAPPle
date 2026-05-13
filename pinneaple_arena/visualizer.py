"""Dark-theme visualization for Arena benchmark results.

Produces the same style as 03_kovasznay_ns_benchmark.py:
  - Field comparison grids (reference vs predictions + error)
  - Training loss curves
  - Optional streamline overlay
  - Saves PNG files to the configured output directory
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# matplotlib imported lazily to respect dark_theme flag before pyplot import
_DARK_SETUP_DONE = False


def _setup_dark():
    global _DARK_SETUP_DONE
    if _DARK_SETUP_DONE:
        return
    import matplotlib
    matplotlib.rcParams.update({
        "figure.facecolor":  "#0d1117",
        "axes.facecolor":    "#161b22",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#c9d1d9",
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "text.color":        "#c9d1d9",
        "grid.color":        "#21262d",
        "grid.linestyle":    "--",
        "figure.titlesize":  14,
        "axes.titlesize":    11,
        "font.family":       "DejaVu Sans",
    })
    _DARK_SETUP_DONE = True


def _setup_light():
    import matplotlib
    matplotlib.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "black",
        "axes.labelcolor":   "black",
        "xtick.color":       "black",
        "ytick.color":       "black",
        "text.color":        "black",
    })


# ── colour maps ───────────────────────────────────────────────────────────────

FIELD_CMAPS  = ["RdBu_r", "PuOr_r", "BrBG_r", "PRGn_r"]
ERROR_CMAP   = "hot"
LOSS_COLORS  = ["#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff",
                "#ffa657", "#79c0ff", "#56d364"]


# ── helpers ───────────────────────────────────────────────────────────────────

def _grid_shape(n: int) -> Tuple[int, int]:
    side = int(np.round(np.sqrt(n)))
    if side * side == n:
        return side, side
    # fallback: nearest square
    return side, side


def _to_2d(arr: np.ndarray, grid_n: int) -> Optional[np.ndarray]:
    """Reshape flat (N,) array into (grid_n, grid_n) if possible."""
    if arr.size == grid_n * grid_n:
        return arr.reshape(grid_n, grid_n)
    side = int(np.round(np.sqrt(arr.size)))
    if side * side == arr.size:
        return arr.reshape(side, side)
    return None


# ── main figure: field grids ───────────────────────────────────────────────────

def plot_field_comparison(
    eval_results: List[Dict[str, Any]],
    field_names: List[str],
    xy_eval: np.ndarray,
    grid_n: int,
    problem_name: str,
    dark_theme: bool = True,
    dpi: int = 150,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Multi-panel field comparison: reference | model1 | model2 | … | error."""
    if dark_theme:
        _setup_dark()
    else:
        _setup_light()

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    is_1d = xy_eval.ndim == 1 or xy_eval.shape[1] == 1
    x_vals = xy_eval.ravel() if is_1d else xy_eval[:, 0]

    n_models = len(eval_results)
    n_fields = len(field_names)
    # columns: Reference + n_models pred + n_models error
    n_cols = 1 + 2 * n_models
    n_rows = n_fields

    fig_w = max(4 * n_cols, 16)
    fig_h = max(3.5 * n_rows, 7)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                           hspace=0.4, wspace=0.3)

    if not is_1d:
        xmin, xmax = xy_eval[:, 0].min(), xy_eval[:, 0].max()
        ymin, ymax = xy_eval[:, 1].min(), xy_eval[:, 1].max()
        extent = [xmin, xmax, ymin, ymax]

    sort_idx = np.argsort(x_vals) if is_1d else None

    for fi, fname in enumerate(field_names):
        ref_col = eval_results[0]["ref"]
        ref_f = ref_col[:, fi] if ref_col.ndim > 1 else ref_col.ravel()

        cmap = FIELD_CMAPS[fi % len(FIELD_CMAPS)]
        vmin, vmax = ref_f.min(), ref_f.max()

        # Reference
        ax_ref = fig.add_subplot(gs[fi, 0])
        if is_1d:
            ax_ref.plot(x_vals[sort_idx], ref_f[sort_idx], color="#58a6ff", linewidth=1.5)
            ax_ref.set_ylabel(fname)
        else:
            ref_2d = _to_2d(ref_f, grid_n)
            if ref_2d is not None:
                im = ax_ref.imshow(ref_2d, origin="lower", extent=extent,
                                   cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            else:
                im = ax_ref.scatter(xy_eval[:, 0], xy_eval[:, 1], c=ref_f,
                                    cmap=cmap, vmin=vmin, vmax=vmax, s=2)
            ax_ref.set_ylabel("y")
            plt.colorbar(im, ax=ax_ref, fraction=0.046, pad=0.04)
        ax_ref.set_title(f"Reference  {fname}", color=ax_ref.title.get_color())
        ax_ref.set_xlabel("x")

        for mi, res in enumerate(eval_results):
            pred = res["pred"]
            pred_f = pred[:, fi] if pred.ndim > 1 else pred.ravel()
            err_f = np.abs(pred_f - ref_f)

            model_name = res.get("name", f"Model{mi+1}")
            col_pred = 1 + 2 * mi
            col_err  = 2 + 2 * mi

            ax_p = fig.add_subplot(gs[fi, col_pred])
            ax_e = fig.add_subplot(gs[fi, col_err])

            if is_1d:
                ax_p.plot(x_vals[sort_idx], ref_f[sort_idx],
                          color="#58a6ff", linewidth=1.5, linestyle="--", label="ref")
                ax_p.plot(x_vals[sort_idx], pred_f[sort_idx],
                          color="#3fb950", linewidth=1.5, label="pred")
                ax_p.legend(fontsize=7)
                ax_p.set_ylabel(fname)

                emax = err_f.max() or 1.0
                ax_e.plot(x_vals[sort_idx], err_f[sort_idx],
                          color="#ff7b72", linewidth=1.5)
                ax_e.set_ylim(0, emax * 1.1)
                ax_e.set_ylabel("|error|")
            else:
                pred_2d = _to_2d(pred_f, grid_n)
                err_2d  = _to_2d(err_f, grid_n)
                emax = err_f.max()

                if pred_2d is not None:
                    im_p = ax_p.imshow(pred_2d, origin="lower", extent=extent,
                                       cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
                else:
                    im_p = ax_p.scatter(xy_eval[:, 0], xy_eval[:, 1], c=pred_f,
                                        cmap=cmap, vmin=vmin, vmax=vmax, s=2)
                plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)

                if err_2d is not None:
                    im_e = ax_e.imshow(err_2d, origin="lower", extent=extent,
                                       cmap=ERROR_CMAP, vmin=0, vmax=emax, aspect="auto")
                else:
                    im_e = ax_e.scatter(xy_eval[:, 0], xy_eval[:, 1], c=err_f,
                                        cmap=ERROR_CMAP, vmin=0, vmax=emax, s=2)
                ax_e.set_ylabel("y")
                plt.colorbar(im_e, ax=ax_e, fraction=0.046, pad=0.04)

            ax_p.set_title(f"{model_name}  {fname}")
            ax_p.set_xlabel("x")
            ax_e.set_title(f"|Error|  {model_name}  {fname}")
            ax_e.set_xlabel("x")

    fig.suptitle(f"Arena Benchmark — {problem_name}", fontsize=15, y=1.01)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── training loss curves ───────────────────────────────────────────────────────

def plot_loss_curves(
    train_results: List[Any],
    dark_theme: bool = True,
    dpi: int = 150,
    save_path: Optional[str] = None,
    show: bool = False,
):
    if dark_theme:
        _setup_dark()
    else:
        _setup_light()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    for i, res in enumerate(train_results):
        color = LOSS_COLORS[i % len(LOSS_COLORS)]
        losses = res.train_losses
        epochs = range(1, len(losses) + 1)
        ax.semilogy(epochs, losses, color=color, linewidth=1.5, label=res.name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── metrics table ──────────────────────────────────────────────────────────────

def plot_metrics_table(
    eval_results: List[Dict[str, Any]],
    field_names: List[str],
    train_results: List[Any],
    dark_theme: bool = True,
    dpi: int = 150,
    save_path: Optional[str] = None,
    show: bool = False,
):
    if dark_theme:
        _setup_dark()
    else:
        _setup_light()

    import matplotlib.pyplot as plt

    model_names = [r["name"] for r in eval_results]
    n_m = len(model_names)
    n_f = len(field_names)

    cols = ["Model"] + [f"L2 {f}" for f in field_names] + \
           [f"Linf {f}" for f in field_names] + ["Time (s)"]
    rows = []
    for mi, res in enumerate(eval_results):
        m = res["metrics"]
        tr = train_results[mi]
        row = [res["name"]]
        for f in field_names:
            row.append(f"{m.get(f'L2_{f}', float('nan')):.3e}")
        for f in field_names:
            row.append(f"{m.get(f'Linf_{f}', float('nan')):.3e}")
        row.append(f"{tr.train_time:.1f}")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(cols)), 1.5 + 0.5 * n_m), dpi=dpi)
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1.0, 1.8)
    for (r, c), cell in t.get_celld().items():
        cell.set_facecolor("#161b22" if dark_theme else "white")
        cell.set_edgecolor("#30363d" if dark_theme else "#cccccc")
        cell.set_text_props(color="#c9d1d9" if dark_theme else "black")
        if r == 0:
            cell.set_facecolor("#21262d" if dark_theme else "#dddddd")
    ax.set_title("Evaluation Metrics", pad=12)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── streamline overlay (2D NS flows) ──────────────────────────────────────────

def plot_streamlines(
    eval_results: List[Dict[str, Any]],
    xy_eval: np.ndarray,
    grid_n: int,
    u_field: str = "u",
    v_field: str = "v",
    field_names: List[str] = None,
    problem_name: str = "",
    dark_theme: bool = True,
    dpi: int = 150,
    save_path: Optional[str] = None,
    show: bool = False,
):
    if field_names is None:
        return
    try:
        u_idx = field_names.index(u_field)
        v_idx = field_names.index(v_field)
    except ValueError:
        return

    if dark_theme:
        _setup_dark()
    else:
        _setup_light()

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from scipy.interpolate import griddata

    n_cols = len(eval_results) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5), dpi=dpi)
    if n_cols == 1:
        axes = [axes]

    xg = np.linspace(xy_eval[:, 0].min(), xy_eval[:, 0].max(), grid_n)
    yg = np.linspace(xy_eval[:, 1].min(), xy_eval[:, 1].max(), grid_n)
    GX, GY = np.meshgrid(xg, yg)

    def _stream(ax, u_flat, v_flat, title):
        speed = np.sqrt(u_flat ** 2 + v_flat ** 2)
        u2d = griddata(xy_eval, u_flat, (GX, GY), method="linear")
        v2d = griddata(xy_eval, v_flat, (GX, GY), method="linear")
        sp  = griddata(xy_eval, speed,  (GX, GY), method="linear")
        ax.contourf(GX, GY, sp, levels=30, cmap="magma")
        ax.streamplot(xg, yg, u2d, v2d, color="white", density=1.5,
                      linewidth=0.8, arrowsize=1.0)
        ax.set_title(title); ax.set_xlabel("x"); ax.set_ylabel("y")

    ref = eval_results[0]["ref"]
    _stream(axes[0],
            ref[:, u_idx] if ref.ndim > 1 else ref.ravel(),
            ref[:, v_idx] if ref.ndim > 1 else np.zeros_like(ref.ravel()),
            "Reference")

    for i, res in enumerate(eval_results):
        pred = res["pred"]
        _stream(axes[i + 1],
                pred[:, u_idx] if pred.ndim > 1 else pred.ravel(),
                pred[:, v_idx] if pred.ndim > 1 else np.zeros_like(pred.ravel()),
                res.get("name", f"Model{i+1}"))

    fig.suptitle(f"Streamlines  {problem_name}", fontsize=14)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


# ── UQ visualization ──────────────────────────────────────────────────────────

def plot_uq(
    uq_result: Any,
    xy_eval: np.ndarray,
    field_names: List[str],
    grid_n: Optional[int] = None,
    title: str = "UQ",
    dark_theme: bool = True,
    dpi: int = 150,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot UQ mean +/- std from a pinneaple_analysis UQResult."""
    if dark_theme:
        _setup_dark()
    else:
        _setup_light()

    import matplotlib.pyplot as plt

    try:
        mean = uq_result.mean.cpu().numpy() if hasattr(uq_result.mean, "cpu") \
               else np.asarray(uq_result.mean)
        std  = uq_result.std.cpu().numpy()  if hasattr(uq_result.std, "cpu") \
               else np.asarray(uq_result.std)
    except Exception:
        return

    n_fields = len(field_names)
    fig, axes = plt.subplots(2, n_fields, figsize=(5 * n_fields, 8), dpi=dpi)
    if n_fields == 1:
        axes = axes.reshape(2, 1)

    gn = grid_n or int(np.round(np.sqrt(len(xy_eval))))
    extent = [xy_eval[:, 0].min(), xy_eval[:, 0].max(),
              xy_eval[:, 1].min(), xy_eval[:, 1].max()]

    for fi, fname in enumerate(field_names):
        m_f = mean[:, fi] if mean.ndim > 1 else mean.ravel()
        s_f = std[:, fi]  if std.ndim  > 1 else std.ravel()
        m2d = _to_2d(m_f, gn); s2d = _to_2d(s_f, gn)
        ax_m = axes[0, fi]; ax_s = axes[1, fi]
        if m2d is not None:
            im_m = ax_m.imshow(m2d, origin="lower", extent=extent, cmap="RdBu_r", aspect="auto")
            im_s = ax_s.imshow(s2d, origin="lower", extent=extent, cmap="hot",    aspect="auto")
        else:
            im_m = ax_m.scatter(xy_eval[:, 0], xy_eval[:, 1], c=m_f, cmap="RdBu_r", s=2)
            im_s = ax_s.scatter(xy_eval[:, 0], xy_eval[:, 1], c=s_f, cmap="hot",    s=2)
        ax_m.set_title(f"Mean {fname}")
        plt.colorbar(im_m, ax=ax_m, fraction=0.046, pad=0.04)
        ax_s.set_title(f"Std  {fname}")
        plt.colorbar(im_s, ax=ax_s, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14)
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor=fig.get_facecolor())
        print(f"  Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)
