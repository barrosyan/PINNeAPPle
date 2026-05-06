"""Animated rolling forecast visualization with optional MP4 export."""
from __future__ import annotations
from typing import Optional, Sequence

import numpy as np


def animate_rolling_forecast(
    y_true:    np.ndarray,
    y_pred_seq: np.ndarray,
    y_std_seq:  Optional[np.ndarray] = None,
    horizon:    int = 16,
    context_len: int = 64,
    interval:   int = 80,
    sigmas:     Sequence[float] = (1.0, 2.0),
    title:      str = "Rolling Forecast",
    save_path:  Optional[str] = None,
    fps:        int = 10,
    dpi:        int = 150,
    figsize:    tuple = (12, 4),
):
    """
    Create an animated rolling forecast.

    Parameters
    ----------
    y_true      : full ground truth series  (N,)
    y_pred_seq  : array of shape (n_steps, horizon) — each row is a forecast
                  issued at step i, covering y_true[i : i+horizon]
    y_std_seq   : array of shape (n_steps, horizon) — forecast std (optional)
    interval    : milliseconds between frames
    save_path   : if given, save as .mp4 (requires ffmpeg) or .gif
    fps         : frames per second for saved video

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as manimation
    except ImportError:
        raise ImportError("pip install matplotlib")

    y_true    = np.asarray(y_true, dtype=float)
    y_pred_seq = np.asarray(y_pred_seq, dtype=float)
    n_steps   = y_pred_seq.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlim(0, context_len + horizon)
    ymin = y_true.min() - 0.1 * abs(y_true.ptp())
    ymax = y_true.max() + 0.1 * abs(y_true.ptp())
    ax.set_ylim(ymin, ymax)
    ax.grid(True, alpha=0.3)

    line_true,  = ax.plot([], [], color="#2196F3", lw=1.5, label="Actual")
    line_pred,  = ax.plot([], [], color="#FF5722", lw=1.5, ls="--", label="Forecast")
    fill_1s     = ax.fill_between([], [], [], alpha=0.20, color="#FF5722", label="±1σ")
    fill_2s     = ax.fill_between([], [], [], alpha=0.10, color="#FF5722", label="±2σ")
    vline       = ax.axvline(x=0, color="gray", ls=":", lw=1)
    ax.legend(loc="upper left", fontsize=8)

    def _init():
        line_true.set_data([], [])
        line_pred.set_data([], [])
        return line_true, line_pred

    def _update(frame):
        nonlocal fill_1s, fill_2s
        i_start = max(0, frame - context_len)
        i_end   = frame
        t_ctx   = np.arange(i_start, i_end)
        t_fwd   = np.arange(i_end, i_end + horizon)

        line_true.set_data(t_ctx, y_true[i_start:i_end])

        if frame < n_steps:
            pred = y_pred_seq[frame]
            line_pred.set_data(t_fwd[:len(pred)], pred)
            vline.set_xdata([i_end - 0.5])

            # Remove old fills and redraw
            for coll in [fill_1s, fill_2s]:
                coll.remove()

            if y_std_seq is not None:
                std = y_std_seq[frame]
                fill_1s = ax.fill_between(t_fwd[:len(pred)],
                                          pred - std, pred + std,
                                          alpha=0.20, color="#FF5722")
                fill_2s = ax.fill_between(t_fwd[:len(pred)],
                                          pred - 2*std, pred + 2*std,
                                          alpha=0.10, color="#FF5722")
            else:
                fill_1s = ax.fill_between([], [], [])
                fill_2s = ax.fill_between([], [], [])
        return line_true, line_pred, fill_1s, fill_2s

    anim = manimation.FuncAnimation(
        fig, _update, frames=n_steps, init_func=_init,
        interval=interval, blit=False,
    )

    if save_path is not None:
        if save_path.endswith(".gif"):
            writer = manimation.PillowWriter(fps=fps)
        else:
            writer = manimation.FFMpegWriter(fps=fps, bitrate=1800)
        anim.save(save_path, writer=writer, dpi=dpi)
        print(f"Saved animation to {save_path}")

    return anim
