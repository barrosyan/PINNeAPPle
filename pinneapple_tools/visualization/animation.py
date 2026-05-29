"""Time-evolution animations for PDE solutions — CFD style."""
from __future__ import annotations
from typing import Optional, Sequence, Tuple, Union
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as mplanim
from matplotlib.figure import Figure

from .style import get_cmap, make_figure, DEFAULT_CMAP


def _to_np(t):
    try:
        return t.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(t)


def animate_scalar_field(
    trajectory: "ArrayLike",
    x: Optional["ArrayLike"] = None,
    y: Optional["ArrayLike"] = None,
    *,
    t_vals: Optional["ArrayLike"] = None,
    title: str = "Field evolution",
    field_label: str = "u",
    cmap: str = DEFAULT_CMAP,
    interval_ms: int = 80,
    repeat: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 100,
    writer: str = "pillow",
) -> mplanim.FuncAnimation:
    """
    Animate a time series of 2-D scalar fields.

    Parameters
    ----------
    trajectory : (nt, nx, ny) or (nt, N) tensor/array
    x, y       : coordinate grids (nx,ny) or 1-D arrays; inferred if None
    t_vals     : time labels for each frame; if None uses frame index
    save_path  : if given, saves animation to file (.gif or .mp4)

    Returns
    -------
    FuncAnimation object (call .to_jshtml() in Jupyter)
    """
    traj = _to_np(trajectory)
    nt = traj.shape[0]
    is_2d = traj.ndim == 3
    if is_2d:
        _, nx, ny = traj.shape
    else:
        nx = traj.shape[1]; ny = 1

    if x is None:
        x_ = np.linspace(0, 1, nx)
        y_ = np.linspace(0, 1, ny)
    else:
        x_ = _to_np(x)
        y_ = _to_np(y) if y is not None else np.linspace(0, 1, ny)

    if x_.ndim == 1 and y_.ndim == 1 and is_2d:
        X, Y = np.meshgrid(x_, y_, indexing="ij")
    else:
        X, Y = x_, y_

    v_min = vmin if vmin is not None else traj.min()
    v_max = vmax if vmax is not None else traj.max()
    cm = get_cmap(cmap)

    fig, ax = make_figure(figsize=(7, 5))
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")

    # Initial frame
    if is_2d:
        cf = ax.contourf(X, Y, traj[0], levels=64, cmap=cm, vmin=v_min, vmax=v_max)
    else:
        cf = ax.plot(x_, traj[0], lw=1.5)[0]

    cbar = fig.colorbar(cf if is_2d else plt.cm.ScalarMappable(cmap=cm), ax=ax, label=field_label)
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=9, va="top", color="white")
    fig.tight_layout()

    def _update(frame):
        ax.clear()
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        if is_2d:
            ax.contourf(X, Y, traj[frame], levels=64, cmap=cm, vmin=v_min, vmax=v_max)
        else:
            ax.plot(x_, traj[frame], lw=1.5, color="#4c9be8")
            ax.set_ylim(v_min, v_max)
        t_label = t_vals[frame] if t_vals is not None else frame
        ax.text(0.02, 0.96, f"t = {t_label:.4g}" if isinstance(t_label, float) else f"frame {t_label}",
                transform=ax.transAxes, fontsize=9, va="top", color="white")
        return []

    anim = mplanim.FuncAnimation(fig, _update, frames=nt, interval=interval_ms, repeat=repeat, blit=False)

    if save_path is not None:
        save_path = Path(save_path)
        ext = save_path.suffix.lower()
        w = writer if ext != ".mp4" else "ffmpeg"
        anim.save(str(save_path), writer=w, dpi=dpi)

    return anim


def animate_streamlines(
    u_traj: "ArrayLike",
    v_traj: "ArrayLike",
    x: Optional["ArrayLike"] = None,
    y: Optional["ArrayLike"] = None,
    *,
    speed_traj: Optional["ArrayLike"] = None,
    t_vals: Optional["ArrayLike"] = None,
    title: str = "Velocity evolution",
    cmap: str = "plasma",
    density: float = 1.2,
    interval_ms: int = 120,
    save_path: Optional[Union[str, Path]] = None,
    writer: str = "pillow",
    dpi: int = 90,
) -> mplanim.FuncAnimation:
    """
    Animated streamline plot for a time series of 2-D velocity fields.

    u_traj, v_traj : (nt, nx, ny)
    """
    import scipy.interpolate as si

    ut = _to_np(u_traj)
    vt = _to_np(v_traj)
    nt, nx, ny = ut.shape

    if x is None:
        xi = np.linspace(0, 1, 60)
        yi = np.linspace(0, 1, 60)
    else:
        x_ = _to_np(x)
        y_ = _to_np(y)
        xi = np.linspace(x_.min(), x_.max(), 60)
        yi = np.linspace(y_.min(), y_.max(), 60)

    Xi, Yi = np.meshgrid(xi, yi, indexing="ij")

    if x is None:
        xsrc = np.linspace(0, 1, nx)
        ysrc = np.linspace(0, 1, ny)
        Xs, Ys = np.meshgrid(xsrc, ysrc, indexing="ij")
    else:
        Xs, Ys = _to_np(x), _to_np(y)

    pts_src = np.column_stack([Xs.ravel(), Ys.ravel()])

    cm = get_cmap(cmap)
    fig, ax = make_figure(figsize=(7, 5))

    def _update(frame):
        ax.clear()
        Ui = si.griddata(pts_src, ut[frame].ravel(), (Xi, Yi), method="linear", fill_value=0.0)
        Vi = si.griddata(pts_src, vt[frame].ravel(), (Xi, Yi), method="linear", fill_value=0.0)
        mag = np.sqrt(Ui ** 2 + Vi ** 2)
        strm = ax.streamplot(xi, yi, Ui.T, Vi.T, color=mag.T, cmap=cm, density=density,
                             linewidth=0.8, arrowsize=1.0)
        t_label = t_vals[frame] if t_vals is not None else frame
        lbl = f"t = {t_label:.4g}" if isinstance(t_label, float) else f"frame {t_label}"
        ax.set_title(f"{title}  |  {lbl}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        return []

    anim = mplanim.FuncAnimation(fig, _update, frames=nt, interval=interval_ms, repeat=True, blit=False)
    if save_path is not None:
        anim.save(str(Path(save_path)), writer=writer, dpi=dpi)
    return anim


def make_gif(
    frames: Sequence["ArrayLike"],
    x: "ArrayLike",
    y: "ArrayLike",
    save_path: Union[str, Path],
    *,
    t_vals: Optional[Sequence] = None,
    cmap: str = DEFAULT_CMAP,
    dpi: int = 80,
    fps: int = 10,
) -> None:
    """Convenience wrapper: save a list of 2-D field frames as a GIF."""
    traj = np.stack([_to_np(f) for f in frames], axis=0)
    anim = animate_scalar_field(
        traj, x, y, t_vals=t_vals, cmap=cmap,
        interval_ms=int(1000 / fps),
        save_path=save_path, dpi=dpi, writer="pillow",
    )
    plt.close("all")
