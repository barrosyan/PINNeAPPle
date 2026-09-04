"""Export a PINNeAPPle result (a point cloud + one or more scalar/vector
fields, optionally over several time steps) to a sequence of ``.ply``
files Blender can import natively (built-in importer, no add-on required
just to look at one frame) -- the PINNeAPPle-side half of the Blender
bridge; ``blender_addon/`` is the other half, running *inside* Blender.

Why PLY and not Alembic
------------------------
Alembic (the usual sim-to-DCC interchange format) would need the
``alembic``/``pyalembic`` Python bindings, which are not a normal
``pip install``-able package (built against Blender's/Maya's own bundled
libraries in practice) -- adding it as a dependency here would make this
module unusable for most people who don't already have a full VFX-pipeline
Python environment. PLY (a simple, well-specified, ASCII-or-binary mesh
format supporting arbitrary per-vertex properties, including colour) is
trivially writable with nothing beyond ``numpy``, and Blender has a
built-in PLY importer -- zero extra dependencies on either side of the
bridge, at the cost of "a directory of numbered files" instead of one
Alembic cache. See ``blender_addon/import_pinneapple_sequence.py`` for how
the numbered sequence becomes a Blender animation.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence

import numpy as np


def _colormap(values: np.ndarray, cmap_name: str = "viridis") -> np.ndarray:
    """values (N,) -> (N,3) uint8 RGB. Uses matplotlib (a base PINNeAPPle
    dependency, `matplotlib>=3.8` in pyproject.toml) if available; falls
    back to a plain grayscale ramp if not, rather than hard-failing on a
    dependency that should already be present."""
    vmin, vmax = float(values.min()), float(values.max())
    span = (vmax - vmin) or 1.0
    norm = np.clip((values - vmin) / span, 0.0, 1.0)
    try:
        import matplotlib.cm as cm
        colors = cm.get_cmap(cmap_name)(norm)[:, :3]
        return (colors * 255).astype(np.uint8)
    except Exception:
        gray = (norm * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=1)


def write_ply(path: str, points: np.ndarray, colors: Optional[np.ndarray] = None,
              scalar_fields: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Write a point cloud (no faces -- a vertex-only PLY, which Blender's
    importer turns into a mesh with only vertices, ready for a Point Cloud
    Visualizer-style geometry-nodes setup, or straightforward as a
    particle/point render). ``colors``: (N,3) uint8 RGB, optional.
    ``scalar_fields``: extra named (N,) float properties beyond colour
    (kept as raw values, not just their colour-mapped visualisation, so a
    downstream Blender geometry-nodes setup can still access the real
    numbers)."""
    n = points.shape[0]
    props = ["x", "y", "z"]
    if colors is not None:
        props += ["red", "green", "blue"]
    field_names = list((scalar_fields or {}).keys())
    props += field_names

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        for name in field_names:
            f.write(f"property float {name}\n")
        f.write("end_header\n")
        for i in range(n):
            row = [f"{points[i, 0]:.6f}", f"{points[i, 1]:.6f}", f"{points[i, 2]:.6f}"]
            if colors is not None:
                row += [str(int(colors[i, 0])), str(int(colors[i, 1])), str(int(colors[i, 2]))]
            for name in field_names:
                row.append(f"{scalar_fields[name][i]:.6f}")
            f.write(" ".join(row) + "\n")


def export_scene(
    points: np.ndarray,
    field: np.ndarray,
    out_dir: str,
    *,
    field_name: str = "field",
    cmap: str = "viridis",
    frame_index: int = 0,
    extra_fields: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    """Export one frame (one time instant / one realisation) as
    ``<out_dir>/frame_<NNNN>.ply``. Call once per frame with increasing
    ``frame_index`` to build a sequence -- see
    :func:`export_trajectory` for the common "I already have every
    frame's arrays" case in one call.

    ``points``: (N, 3). ``field``: (N,) -- the scalar mapped to vertex
    colour via ``cmap``. ``extra_fields``: additional (N,) arrays kept as
    raw PLY vertex properties (not colour-mapped) for downstream use in
    Blender (e.g. driving a geometry-nodes displacement).
    """
    colors = _colormap(np.asarray(field), cmap)
    scalars = {field_name: np.asarray(field, dtype=np.float32)}
    if extra_fields:
        scalars.update({k: np.asarray(v, dtype=np.float32) for k, v in extra_fields.items()})
    path = os.path.join(out_dir, f"frame_{frame_index:04d}.ply")
    write_ply(path, np.asarray(points, dtype=np.float32), colors, scalars)
    return path


def export_trajectory(
    points: np.ndarray,
    fields_over_time: Sequence[np.ndarray],
    out_dir: str,
    *,
    field_name: str = "field",
    cmap: str = "viridis",
) -> list:
    """Export a whole sequence at once: ``points`` fixed (N, 3), one
    ``field`` array (N,) per frame in ``fields_over_time``. This is the
    natural shape for a PINN surrogate's own output -- evaluate the model
    at the same spatial points across a range of ``t`` values and pass the
    resulting list of (N,) arrays straight in.

    Returns the list of written file paths, in frame order.
    """
    paths = []
    for i, field in enumerate(fields_over_time):
        paths.append(export_scene(points, field, out_dir, field_name=field_name, cmap=cmap, frame_index=i))
    return paths
