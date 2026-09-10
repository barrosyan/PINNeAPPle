"""Loader for a Splash-CFD ``.splash`` case (a zipped OpenFOAM case
directory) into plain numpy arrays -- both a flat point-cloud form (for
PINN-style training) and a dense structured-grid volume time series (for
grid-based neural operators like :class:`~pinneapple_neural.
architectures.neural_operators.fno.FNO3d`).

Promoted from the splash-pinneapple downstream project's own
``pipeline/splash_dataset.py`` / ``pipeline/splash_mesh.py`` (real,
already-used-in-production ingestion code, not a rewrite) so any project
depending on PINNeAPPle can ``pip install`` it and load a ``.splash``
archive directly -- no need to vendor this ingestion layer per-project.
Built on ``pinneapple_simulation.external_solvers.openfoam.binary_reader``/
``.mesh_reader`` (the real binary FoamFile / polyMesh readers) rather than
duplicating their parsing logic.

Why this is a separate adapter from
``pinneapple_simulation.external_solvers.openfoam.field_reader
.openfoam_case_to_upd``: that function solves a different problem -- it
reads *one* time directory from a *case directory on disk* into a single
``pinneapple_data.PhysicalSample``. This adapter reads, from a *zipped*
``.splash`` archive, several time directories at once, with
per-time-directory random subsampling, ``transportProperties``/manifest
parsing, a mesh cache to skip repeated (expensive) polyMesh
reconstruction, and (new here) reshaping into dense structured-grid
volumes for a structured single-block mesh -- none of which
``openfoam_case_to_upd`` does.
"""
from __future__ import annotations

import io
import json
import os
import re
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pinneapple_simulation.external_solvers.openfoam import binary_reader as _binary_reader
from pinneapple_simulation.external_solvers.openfoam import mesh_reader as _mesh_reader

__all__ = [
    "SplashMesh",
    "FieldSnapshot",
    "SplashCase",
    "open_case",
    "load_mesh",
    "save_mesh_cache",
    "load_mesh_cache",
    "load_dense_volumes",
]

_TIME_DIR_RE = re.compile(r"^\d+(\.\d+)?$")


# ---------------------------------------------------------------------------
# Mesh: cell centers / sizes from the binary polyMesh inside a .splash zip
# ---------------------------------------------------------------------------

@dataclass
class SplashMesh:
    n_cells: int
    n_points: int
    n_faces: int
    n_internal_faces: int
    cell_centers: np.ndarray  # (n_cells, 3)
    cell_size: np.ndarray  # (n_cells, 3) -- bounding-box (dx, dy, dz)
    cell_delta: np.ndarray  # (n_cells,) -- cubeRootVol = (dx*dy*dz)**(1/3)
    bounds_min: np.ndarray  # (3,)
    bounds_max: np.ndarray  # (3,)


def _from_geometry(geo: "_mesh_reader.MeshGeometry") -> SplashMesh:
    return SplashMesh(
        n_cells=geo.nodes.shape[0],
        n_points=geo.n_points,
        n_faces=geo.n_faces,
        n_internal_faces=geo.n_internal_faces,
        cell_centers=geo.nodes,
        cell_size=geo.cell_size,
        cell_delta=geo.cell_delta,
        bounds_min=geo.bounds_min,
        bounds_max=geo.bounds_max,
    )


def load_mesh(zf: zipfile.ZipFile, prefix: str = "constant/polyMesh/") -> SplashMesh:
    """Build cell centers/sizes from the binary polyMesh inside a .splash
    zip. ``zf`` is an open ``zipfile.ZipFile``."""
    return _from_geometry(_mesh_reader.load_mesh(zf, prefix=prefix))


def save_mesh_cache(mesh: SplashMesh, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(
        path,
        cell_centers=mesh.cell_centers, cell_size=mesh.cell_size, cell_delta=mesh.cell_delta,
        bounds_min=mesh.bounds_min, bounds_max=mesh.bounds_max,
        n_cells=mesh.n_cells, n_points=mesh.n_points, n_faces=mesh.n_faces,
        n_internal_faces=mesh.n_internal_faces,
    )


def load_mesh_cache(path: str) -> SplashMesh:
    d = np.load(path)
    return SplashMesh(
        n_cells=int(d["n_cells"]), n_points=int(d["n_points"]), n_faces=int(d["n_faces"]),
        n_internal_faces=int(d["n_internal_faces"]), cell_centers=d["cell_centers"],
        cell_size=d["cell_size"], cell_delta=d["cell_delta"],
        bounds_min=d["bounds_min"], bounds_max=d["bounds_max"],
    )


# ---------------------------------------------------------------------------
# Case: fields at each time directory
# ---------------------------------------------------------------------------

@dataclass
class FieldSnapshot:
    time: float
    fields: Dict[str, np.ndarray]  # name -> (n_cells,) or (n_cells, k)


@dataclass
class SplashCase:
    path: str
    mesh: SplashMesh
    time_dirs: List[str] = field(default_factory=list)
    transport: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    def _open(self) -> zipfile.ZipFile:
        return zipfile.ZipFile(self.path)

    def read_field(self, time_dir: str, name: str) -> np.ndarray:
        """Read one field's internalField at one time directory, broadcast
        to (n_cells,) or (n_cells, k)."""
        with self._open() as zf:
            member = f"{time_dir}/{name}"
            with zf.open(member) as fh:
                data = fh.read()
        arr, is_uniform, n_components = _binary_reader.read_internal_field(data, member)
        if is_uniform:
            if n_components == 1:
                return np.full((self.n_cells,), arr[0], dtype=np.float64)
            return np.tile(arr[None, :], (self.n_cells, 1))
        if arr.shape[0] != self.n_cells:
            raise ValueError(f"{member}: field has {arr.shape[0]} cells, mesh has {self.n_cells}")
        return arr

    def available_fields(self, time_dir: str) -> List[str]:
        with self._open() as zf:
            prefix = f"{time_dir}/"
            out = []
            for n in zf.namelist():
                if n.startswith(prefix) and "/" not in n[len(prefix):]:
                    out.append(n[len(prefix):])
            return sorted(out)

    def read_snapshot(self, time_dir: str, fields: Sequence[str]) -> FieldSnapshot:
        avail = set(self.available_fields(time_dir))
        out = {}
        for name in fields:
            if name not in avail:
                continue
            out[name] = self.read_field(time_dir, name)
        return FieldSnapshot(time=float(time_dir), fields=out)

    def build_point_cloud(
        self, times: Sequence[str], fields: Sequence[str],
        max_points_per_time: Optional[int] = None, seed: int = 0,
    ) -> Dict[str, np.ndarray]:
        """Concatenate cell-center coordinates + requested fields across
        several time directories into flat training arrays.

        Returns a dict with keys ``"xyzt"`` (N,4), ``"delta"`` (N,), and one
        entry per successfully-read field name (N,) or (N,k). Fields not
        present at every requested time are omitted for the snapshots that
        lack them.
        """
        rng = np.random.default_rng(seed)
        xyzt_parts, delta_parts = [], []
        field_parts: Dict[str, List[np.ndarray]] = {f: [] for f in fields}

        for t in times:
            snap = self.read_snapshot(t, fields)
            n = self.n_cells
            idx = np.arange(n)
            if max_points_per_time is not None and max_points_per_time < n:
                idx = rng.choice(n, size=max_points_per_time, replace=False)

            xyz = self.mesh.cell_centers[idx]
            tcol = np.full((idx.shape[0], 1), snap.time, dtype=np.float64)
            xyzt_parts.append(np.concatenate([xyz, tcol], axis=1))
            delta_parts.append(self.mesh.cell_delta[idx])

            for name in fields:
                if name in snap.fields:
                    field_parts[name].append(snap.fields[name][idx])

        out: Dict[str, np.ndarray] = {
            "xyzt": np.concatenate(xyzt_parts, axis=0),
            "delta": np.concatenate(delta_parts, axis=0),
        }
        for name, parts in field_parts.items():
            if parts and len(parts) == len(times):
                out[name] = np.concatenate(parts, axis=0)
        return out

    def read_dense_volume(self, time_dir: str, field: str, comp: Optional[int], nx: int, ny: int, nz: int) -> np.ndarray:
        """Read one field (optionally one vector component) at one time
        directory and reshape to a dense (nz, ny, nx) volume. Only valid
        for a structured single hex block (blockMesh, x fastest / y next /
        z slowest) whose cell count equals ``nx*ny*nz`` -- raises
        ``ValueError`` otherwise rather than silently misreshaping."""
        arr = self.read_field(time_dir, field)
        val = arr[:, comp] if (arr.ndim == 2 and comp is not None) else (arr[:, 0] if arr.ndim == 2 else arr)
        if val.shape[0] != nx * ny * nz:
            raise ValueError(
                f"read_dense_volume: field {field!r} has {val.shape[0]} cells, expected "
                f"nx*ny*nz={nx*ny*nz} -- this mesh is not a structured {nx}x{ny}x{nz} single block."
            )
        return val.reshape(nz, ny, nx)


def _read_text_member(zf: zipfile.ZipFile, name: str) -> str:
    with zf.open(name) as fh:
        return fh.read().decode("utf-8", errors="replace")


def _parse_transport_properties(text: str) -> Dict[str, float]:
    out = {}
    m = re.search(r"\bnu\s+([-\d.eE+]+)\s*;", text)
    if m:
        out["nu"] = float(m.group(1))
    return out


def open_case(path: str, mesh_cache_dir: Optional[str] = None) -> SplashCase:
    """Open a ``.splash`` file, reconstructing (or loading a cached) mesh."""
    cache_path = None
    if mesh_cache_dir is not None:
        base = os.path.splitext(os.path.basename(path))[0]
        cache_path = os.path.join(mesh_cache_dir, base + ".mesh.npz")

    if cache_path is not None and os.path.exists(cache_path):
        mesh = load_mesh_cache(cache_path)
    else:
        with zipfile.ZipFile(path) as zf:
            mesh = load_mesh(zf)
        if cache_path is not None:
            save_mesh_cache(mesh, cache_path)

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        time_dirs = sorted(
            {n.split("/")[0] for n in names if _TIME_DIR_RE.match(n.split("/")[0])}, key=float,
        )
        transport = {}
        if "constant/transportProperties" in names:
            transport = _parse_transport_properties(_read_text_member(zf, "constant/transportProperties"))
        meta = {}
        if "splash-manifest.json" in names:
            try:
                meta = json.loads(_read_text_member(zf, "splash-manifest.json"))
            except Exception:
                meta = {}

    return SplashCase(path=path, mesh=mesh, time_dirs=time_dirs, transport=transport, meta=meta)


# ---------------------------------------------------------------------------
# Dense volume time series -- for grid-based neural operators (FNO, etc.)
# ---------------------------------------------------------------------------

def load_dense_volumes(
    splash_path: str,
    field_components: Sequence[Tuple[str, Optional[int]]],
    nx: int, ny: int, nz: int,
    *, mesh_cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, List[float]]:
    """Load every time directory of a ``.splash`` archive as a dense
    volume time series, for a structured single hex block mesh.

    Parameters
    ----------
    field_components : e.g. ``[("U", 0), ("U", 1), ("U", 2), ("p", None)]``
        -- field name + component index (``None`` for a scalar field),
        one entry per output channel.
    nx, ny, nz : the mesh's structured grid resolution. ``nx*ny*nz`` must
        equal the mesh's real cell count (checked per-field via
        ``SplashCase.read_dense_volume``, which raises rather than
        silently misreshaping a mismatched mesh).

    Returns
    -------
    (frames, times)
        ``frames`` : ``(T, C, nz, ny, nx)`` float32 array, ``C =
        len(field_components)``.
        ``times`` : the real simulation times, ascending, one per frame.
    """
    case = open_case(splash_path, mesh_cache_dir=mesh_cache_dir)
    times = sorted(float(t) for t in case.time_dirs)
    frames = []
    for t in times:
        t_str = next(s for s in case.time_dirs if abs(float(s) - t) < 1e-9)
        chans = [case.read_dense_volume(t_str, field, comp, nx, ny, nz) for field, comp in field_components]
        frames.append(np.stack(chans, axis=0))
    return np.stack(frames, axis=0).astype(np.float32), times
