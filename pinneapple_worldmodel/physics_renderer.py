# -*- coding: utf-8 -*-
"""Stage 4 — Physics Renderer.

Converts physical field arrays (velocity, pressure, temperature, concentration)
into multi-channel visual observations:

  - RGB frames      : matplotlib scientific colormap (viridis / plasma)
  - Thermal frames  : FLIR-style inferno colormap (temperature channel)
  - Depth frames    : synthetic depth map (flow magnitude as proxy)
  - IR frames       : infrared simulation (T + emissivity model)

Assembles frame sequences into video files (.mp4) via imageio / matplotlib
animation.  Falls back to PNG sequences if ffmpeg is not available.

Public API
----------
  PhysicsRenderer      — main rendering class
  RendererConfig       — dataclass for render settings
  RenderResult         — container with frame arrays + file paths
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.animation import FFMpegWriter, PillowWriter
    _MPL = True
except ImportError:
    _MPL = False

try:
    import imageio
    _IMAGEIO = True
except ImportError:
    _IMAGEIO = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RendererConfig:
    """Configuration for the PhysicsRenderer.

    Parameters
    ----------
    resolution : (H, W)
        Output frame size in pixels.
    fps : int
        Frames per second for video output.
    sensors : list of str
        Which sensor channels to render: ``"rgb"``, ``"thermal"``, ``"depth"``,
        ``"ir"``.
    colormap_rgb : str
        Matplotlib colormap for the RGB channel (default: ``"viridis"``).
    colormap_thermal : str
        Colormap for thermal frames (default: ``"inferno"``).
    colormap_depth : str
        Colormap for depth frames (default: ``"gray"``).
    dpi : int
        DPI for matplotlib rendering (affects resolution).
    dark_background : bool
        Use a dark (#0d1117) background.
    video_quality : int
        ffmpeg quality (CRF value, lower = better, 18–28 typical).
    """
    resolution:        Tuple[int, int] = (256, 256)
    fps:               int             = 24
    sensors:           List[str]       = field(default_factory=lambda: ["rgb", "thermal", "depth"])
    colormap_rgb:      str             = "viridis"
    colormap_thermal:  str             = "inferno"
    colormap_depth:    str             = "gray"
    colormap_ir:       str             = "hot"
    dpi:               int             = 72
    dark_background:   bool            = True
    video_quality:     int             = 23
    field_for_rgb:     str             = "auto"   # "auto" = pick by PDE kind
    add_colorbar:      bool            = False
    add_timestamp:     bool            = False
    bg_color:          str             = "#0d1117"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RenderResult:
    """Output of PhysicsRenderer.render().

    Attributes
    ----------
    frames : dict[sensor_name -> ndarray (T, H, W, C)]
        Raw frame sequences (uint8, RGB or single-channel).
    video_paths : dict[sensor_name -> Path]
        Saved video file paths.
    png_dirs : dict[sensor_name -> Path]
        Directories with individual PNG frames (if video failed).
    field_stats : dict
        Min/max/mean of each rendered physical field.
    """
    frames:      Dict[str, np.ndarray] = field(default_factory=dict)
    video_paths: Dict[str, Path]       = field(default_factory=dict)
    png_dirs:    Dict[str, Path]       = field(default_factory=dict)
    field_stats: Dict[str, Any]        = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PhysicsRenderer
# ---------------------------------------------------------------------------

class PhysicsRenderer:
    """Render physical field trajectories into video frames.

    Parameters
    ----------
    config : RendererConfig

    Examples
    --------
    ::

        renderer = PhysicsRenderer(RendererConfig(resolution=(256, 256), fps=24))
        result   = renderer.render(states, field_names, output_dir=Path("sample_001"))
    """

    def __init__(self, config: Optional[RendererConfig] = None) -> None:
        self.cfg = config or RendererConfig()

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def render(
        self,
        states:      np.ndarray,       # (T, C, Ny, Nx)  physical fields
        field_names: List[str],        # names per channel
        output_dir:  Path,
        sample_id:   str = "000",
    ) -> RenderResult:
        """Render all configured sensor channels.

        Parameters
        ----------
        states : ndarray (T, C, Ny, Nx)
            Time-major array of physical field snapshots.
        field_names : list of str
            Channel names (``"u"``, ``"v"``, ``"p"``, ``"T"``, …).
        output_dir : Path
            Directory where videos/frames are saved.
        sample_id : str
            Sample identifier used in filenames.

        Returns
        -------
        RenderResult
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = RenderResult()
        result.field_stats = _compute_stats(states, field_names)

        # Compute the scalar field to display for each sensor
        rgb_field    = self._select_rgb_field(states, field_names)
        temp_field   = self._select_temperature_field(states, field_names)
        depth_field  = self._select_depth_field(states, field_names)
        ir_field     = self._build_ir_field(temp_field, rgb_field)

        field_map = {
            "rgb":     (rgb_field,   self.cfg.colormap_rgb),
            "thermal": (temp_field,  self.cfg.colormap_thermal),
            "depth":   (depth_field, self.cfg.colormap_depth),
            "ir":      (ir_field,    self.cfg.colormap_ir),
        }

        for sensor in self.cfg.sensors:
            if sensor not in field_map:
                continue
            fld, cmap = field_map[sensor]
            frames = self._fields_to_frames(fld, cmap, sensor)
            result.frames[sensor] = frames

            # Save video
            vid_path = output_dir / f"video_{sensor}.mp4"
            png_dir  = output_dir / f"frames_{sensor}"
            try:
                _save_video(frames, vid_path, self.cfg.fps, self.cfg.video_quality)
                result.video_paths[sensor] = vid_path
            except Exception:
                _save_png_sequence(frames, png_dir)
                result.png_dirs[sensor] = png_dir

        return result

    # ------------------------------------------------------------------
    # Field selection helpers
    # ------------------------------------------------------------------

    def _select_rgb_field(self, states: np.ndarray, names: List[str]) -> np.ndarray:
        """Select / compute the primary scalar field for RGB rendering."""
        # Prefer velocity magnitude, then first channel
        if len(names) >= 2:
            u_idx = _find_channel(names, ["u", "ux", "vx", "velocity_x"])
            v_idx = _find_channel(names, ["v", "uy", "vy", "velocity_y"])
            if u_idx is not None and v_idx is not None:
                return np.sqrt(states[:, u_idx]**2 + states[:, v_idx]**2)
        return states[:, 0]

    def _select_temperature_field(self, states: np.ndarray, names: List[str]) -> np.ndarray:
        idx = _find_channel(names, ["T", "temperature", "temp", "theta"])
        if idx is not None:
            return states[:, idx]
        # Fall back: pressure (proxy for thermal energy)
        p_idx = _find_channel(names, ["p", "pressure"])
        if p_idx is not None:
            return states[:, p_idx]
        return states[:, 0]

    def _select_depth_field(self, states: np.ndarray, names: List[str]) -> np.ndarray:
        """Synthetic depth: use vorticity magnitude as proxy."""
        if states.shape[1] >= 2:
            u_idx = _find_channel(names, ["u", "ux"])
            v_idx = _find_channel(names, ["v", "uy"])
            if u_idx is not None and v_idx is not None:
                u = states[:, u_idx]  # (T, Ny, Nx)
                v = states[:, v_idx]  # (T, Ny, Nx)
                # Vorticity: dv/dx - du/dy  (axis 2=x, axis 1=y for T,Ny,Nx)
                dvdx = np.gradient(v, axis=2)
                dudy = np.gradient(u, axis=1)
                return np.abs(dvdx - dudy)
        return np.abs(states[:, 0])

    def _build_ir_field(self, temp_field: np.ndarray, vel_mag: np.ndarray) -> np.ndarray:
        """Simulate IR: temperature + emissivity modulation from velocity."""
        eps = 0.9   # emissivity constant (blackbody-like)
        t_n = _norm_01(temp_field)
        v_n = _norm_01(vel_mag)
        # Stefan-Boltzmann approximation: I_IR ~ eps * T^4 (normalised)
        ir  = eps * t_n**4 + 0.05 * v_n
        return ir

    # ------------------------------------------------------------------
    # Frame assembly
    # ------------------------------------------------------------------

    def _fields_to_frames(
        self,
        field:  np.ndarray,    # (T, Ny, Nx)
        cmap:   str,
        sensor: str,
    ) -> np.ndarray:
        """Convert scalar field sequence to uint8 RGB frames (T, H, W, 3)."""
        H, W = self.cfg.resolution
        T    = field.shape[0]
        f_n  = _norm_01(field)

        cmap_fn  = plt.get_cmap(cmap) if _MPL else _fallback_cmap(cmap)
        frames   = np.zeros((T, H, W, 3), dtype=np.uint8)

        for t in range(T):
            img = _apply_colormap(f_n[t], cmap_fn, H, W)
            if self.cfg.add_timestamp and _MPL:
                img = _burn_timestamp(img, t, T)
            frames[t] = img

        return frames


# ---------------------------------------------------------------------------
# Video / PNG I/O helpers
# ---------------------------------------------------------------------------

def _save_video(frames: np.ndarray, path: Path, fps: int, crf: int = 23) -> None:
    """Write (T, H, W, 3) uint8 array to .mp4.  Tries imageio then matplotlib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if _IMAGEIO:
        try:
            writer = imageio.get_writer(str(path), fps=fps, quality=8,
                                        macro_block_size=1)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            return
        except Exception:
            pass
    # Fallback: matplotlib FFMpeg writer
    if _MPL:
        try:
            fig, ax = plt.subplots(figsize=(frames.shape[2]/72, frames.shape[1]/72))
            ax.axis("off")
            im = ax.imshow(frames[0])
            fig.subplots_adjust(0, 0, 1, 1)

            def _update(t):
                im.set_data(frames[t])
                return [im]

            from matplotlib.animation import FuncAnimation
            anim   = FuncAnimation(fig, _update, frames=len(frames), interval=1000/fps)
            writer = FFMpegWriter(fps=fps, metadata={"crf": str(crf)})
            anim.save(str(path), writer=writer)
            plt.close(fig)
            return
        except Exception:
            pass
    raise RuntimeError("No video writer available (install imageio[ffmpeg] or ffmpeg)")


def _save_png_sequence(frames: np.ndarray, out_dir: Path) -> None:
    """Save each frame as a PNG (fallback when video writer is unavailable)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if _MPL:
        for t, frame in enumerate(frames):
            path = out_dir / f"frame_{t:06d}.png"
            plt.imsave(str(path), frame)
    else:
        try:
            import imageio
            for t, frame in enumerate(frames):
                imageio.imwrite(str(out_dir / f"frame_{t:06d}.png"), frame)
        except ImportError:
            np.save(str(out_dir / "frames.npy"), frames)


# ---------------------------------------------------------------------------
# Field utilities
# ---------------------------------------------------------------------------

def _find_channel(names: List[str], candidates: List[str]) -> Optional[int]:
    for cand in candidates:
        for i, nm in enumerate(names):
            if nm.lower() == cand.lower():
                return i
    return None


def _norm_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _compute_stats(states: np.ndarray, names: List[str]) -> Dict[str, Any]:
    stats = {}
    for i, nm in enumerate(names):
        if i >= states.shape[1]:
            break
        ch = states[:, i]
        stats[nm] = {"min": float(ch.min()), "max": float(ch.max()),
                     "mean": float(ch.mean()), "std": float(ch.std())}
    return stats


def _apply_colormap(
    field_2d: np.ndarray,   # (Ny, Nx)  float32 in [0, 1]
    cmap_fn,
    H: int, W: int,
) -> np.ndarray:             # (H, W, 3) uint8
    # Resize by simple nearest-neighbour if resolution differs
    Ny, Nx = field_2d.shape
    if Ny != H or Nx != W:
        iy = (np.arange(H) * Ny / H).astype(int).clip(0, Ny - 1)
        ix = (np.arange(W) * Nx / W).astype(int).clip(0, Nx - 1)
        field_2d = field_2d[np.ix_(iy, ix)]
    rgba = (cmap_fn(field_2d)[:, :, :3] * 255).astype(np.uint8)
    return rgba


def _burn_timestamp(frame: np.ndarray, t: int, T: int) -> np.ndarray:
    """Burn a simple 't/T' text into the top-left corner (in-place)."""
    if _MPL:
        H, W, _ = frame.shape
        fig, ax  = plt.subplots(figsize=(W/72, H/72), dpi=72)
        ax.imshow(frame)
        ax.text(5, 15, f"t={t}/{T}", color="white", fontsize=6,
                bbox=dict(facecolor="black", alpha=0.4, pad=1))
        ax.axis("off")
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = buf.reshape(H, W, 3).copy()
        plt.close(fig)
    return frame


def _fallback_cmap(name: str):
    """Simple fallback colormap when matplotlib is unavailable."""
    def cmap(x):
        x = np.clip(x, 0.0, 1.0)
        r = x
        g = x * 0.5
        b = 1.0 - x
        return np.stack([r, g, b, np.ones_like(r)], axis=-1)
    return cmap
