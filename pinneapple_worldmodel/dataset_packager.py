# -*- coding: utf-8 -*-
"""Stage 8 — Dataset Packager.

Assembles all pipeline outputs into the canonical Physical AI dataset
structure.

Final layout
------------
::

    <output_dir>/
      sample_000001/
        video_rgb.mp4
        video_thermal.mp4
        video_depth.mp4
        velocity.zarr/          (T, Ny, Nx) or (T, 2, Ny, Nx)
        pressure.zarr/          (T, Ny, Nx)
        temperature.zarr/       (T, Ny, Nx)
        concentration.zarr/     (T, Ny, Nx)   [if present]
        mesh.vtk                (structured grid VTK)
        metadata.json           physical + simulation metadata
        pde.json                governing PDE specification
        boundary_conditions.json
        camera.json
      sample_000002/
        ...
      dataset_manifest.json    (index of all samples)

Physical fields are stored in **Zarr v2** format (cloud-native, chunked,
compressed).  Falls back to NumPy .npy files when zarr is not installed.

Public API
----------
  PackagerConfig   — output format settings
  DatasetPackager  — writes one sample or a batch to disk
  DatasetManifest  — aggregated index over all samples
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional zarr
try:
    import zarr
    _ZARR = True
except ImportError:
    _ZARR = False

from .simulator import TrajectoryData
from .scenario_generator import ScenarioSpec


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PackagerConfig:
    """Controls how the packager writes the dataset.

    Parameters
    ----------
    output_dir : Path
        Root directory for all samples.
    format : str
        Physical field storage format: ``"zarr"`` (default) or ``"npy"``.
    zarr_chunks : tuple
        Zarr chunk shape for time-first storage.  ``None`` = auto.
    zarr_compressor : str
        Zarr compression codec: ``"blosc"`` or ``"gzip"``.
    save_vtk : bool
        Write a structured-grid VTK file for ParaView compatibility.
    save_videos : bool
        Copy video files from the render result.
    sample_id_digits : int
        Zero-padding width for sample folder names (default 6).
    overwrite : bool
        If False, skip samples whose directory already exists.
    """
    output_dir:        Path   = Path("./physics_dataset")
    format:            str    = "zarr"      # "zarr" | "npy"
    zarr_chunks:       Optional[Tuple] = None
    zarr_compressor:   str    = "blosc"
    save_vtk:          bool   = True
    save_videos:       bool   = True
    sample_id_digits:  int    = 6
    overwrite:         bool   = False


# ---------------------------------------------------------------------------
# Sample result container
# ---------------------------------------------------------------------------

@dataclass
class PackagedSample:
    """Metadata for a single packaged sample.

    Attributes
    ----------
    sample_id : str
        Zero-padded integer identifier.
    sample_dir : Path
        Absolute path to the sample directory.
    field_paths : dict[field_name -> Path]
        Paths to stored physical field files.
    video_paths : dict[sensor -> Path]
        Paths to video files.
    metadata_path : Path
    """
    sample_id:    str
    sample_dir:   Path
    field_paths:  Dict[str, Path] = field(default_factory=dict)
    video_paths:  Dict[str, Path] = field(default_factory=dict)
    metadata_path: Optional[Path] = None

    def to_manifest_entry(self) -> Dict[str, Any]:
        return {
            "sample_id":   self.sample_id,
            "sample_dir":  str(self.sample_dir),
            "fields":      {k: str(v) for k, v in self.field_paths.items()},
            "videos":      {k: str(v) for k, v in self.video_paths.items()},
            "metadata":    str(self.metadata_path) if self.metadata_path else None,
        }


# ---------------------------------------------------------------------------
# DatasetManifest
# ---------------------------------------------------------------------------

class DatasetManifest:
    """Index of all samples in a Physical AI dataset."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.entries:   List[Dict[str, Any]] = []
        self._path      = self.output_dir / "dataset_manifest.json"
        if self._path.exists():
            self.entries = json.loads(self._path.read_text())

    def add(self, sample: PackagedSample) -> None:
        self.entries.append(sample.to_manifest_entry())

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self.entries, indent=2))

    def __len__(self) -> int:
        return len(self.entries)


# ---------------------------------------------------------------------------
# DatasetPackager
# ---------------------------------------------------------------------------

class DatasetPackager:
    """Write one physics sample to the canonical dataset structure.

    Parameters
    ----------
    config : PackagerConfig

    Examples
    --------
    ::

        packager = DatasetPackager(PackagerConfig(output_dir=Path("./out")))
        result   = packager.package(
            trajectory    = traj,
            spec          = scenario_spec,
            render_result = render_result,
            sample_idx    = 0,
        )
    """

    def __init__(self, config: Optional[PackagerConfig] = None) -> None:
        self.cfg      = config or PackagerConfig()
        self.manifest = DatasetManifest(self.cfg.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def package(
        self,
        trajectory:     TrajectoryData,
        spec:           ScenarioSpec,
        render_result,                     # RenderResult from physics_renderer
        sample_idx:     int = 0,
        extra_fields:   Optional[Dict[str, np.ndarray]] = None,
        camera_config:  Optional[Dict[str, Any]] = None,
    ) -> PackagedSample:
        """Package one sample into the canonical directory structure.

        Parameters
        ----------
        trajectory : TrajectoryData
            Output of PhysicsSimulator.generate_trajectory().
        spec : ScenarioSpec
            Enriched scenario specification from ScenarioGenerator.
        render_result : RenderResult
            Output of PhysicsRenderer.render().
        sample_idx : int
            Sample index (determines folder name).
        extra_fields : dict, optional
            Additional field arrays to store alongside the trajectory.
        camera_config : dict, optional
            Camera metadata from CameraSystem.to_dict().

        Returns
        -------
        PackagedSample
        """
        sid      = str(sample_idx).zfill(self.cfg.sample_id_digits)
        sdir     = Path(self.cfg.output_dir) / f"sample_{sid}"

        if sdir.exists() and not self.cfg.overwrite:
            # Return existing manifest entry
            return PackagedSample(sample_id=sid, sample_dir=sdir)

        sdir.mkdir(parents=True, exist_ok=True)
        result = PackagedSample(sample_id=sid, sample_dir=sdir)

        # 1. Physical fields → zarr / npy
        states_np = trajectory.states.cpu().numpy()   # (T, C, Ny, Nx)
        T, C, Ny, Nx = states_np.shape
        field_names  = spec.scenario.field_names

        for i, fname in enumerate(field_names):
            if i >= C:
                break
            fdata = states_np[:, i]     # (T, Ny, Nx)
            fpath = self._write_field(fdata, sdir, fname)
            result.field_paths[fname] = fpath

        # Extra fields (e.g. concentration, species)
        if extra_fields:
            for fname, fdata in extra_fields.items():
                fpath = self._write_field(np.asarray(fdata), sdir, fname)
                result.field_paths[fname] = fpath

        # 2. Videos
        if self.cfg.save_videos and render_result is not None:
            for sensor, vid_path in render_result.video_paths.items():
                dst = sdir / f"video_{sensor}.mp4"
                try:
                    import shutil
                    shutil.copy2(vid_path, dst)
                    result.video_paths[sensor] = dst
                except Exception:
                    pass
            # Also copy PNG dirs if no video
            for sensor, png_dir in render_result.png_dirs.items():
                if sensor not in result.video_paths:
                    dst_dir = sdir / f"frames_{sensor}"
                    try:
                        import shutil
                        shutil.copytree(png_dir, dst_dir, dirs_exist_ok=True)
                        result.video_paths[sensor] = dst_dir
                    except Exception:
                        pass

        # 3. Structured VTK mesh
        if self.cfg.save_vtk:
            vtk_path = sdir / "mesh.vtk"
            _write_vtk(states_np, spec, vtk_path)

        # 4. metadata.json
        meta_path = sdir / "metadata.json"
        _write_json(meta_path, _build_metadata(trajectory, spec, T, C, Ny, Nx))
        result.metadata_path = meta_path

        # 5. pde.json
        _write_json(sdir / "pde.json", _build_pde_json(spec))

        # 6. boundary_conditions.json
        _write_json(sdir / "boundary_conditions.json", spec.boundary_conditions)

        # 7. camera.json
        if camera_config:
            _write_json(sdir / "camera.json", camera_config)
        else:
            _write_json(sdir / "camera.json", spec.sensor_config)

        # Update manifest
        self.manifest.add(result)
        self.manifest.save()

        return result

    def package_batch(
        self,
        trajectories:  List[TrajectoryData],
        spec:          ScenarioSpec,
        render_results,                    # List[RenderResult] or None
        start_idx:     int = 0,
    ) -> List[PackagedSample]:
        """Package multiple samples from the same scenario."""
        rr_list = render_results or [None] * len(trajectories)
        return [
            self.package(traj, spec, rr, start_idx + i)
            for i, (traj, rr) in enumerate(zip(trajectories, rr_list))
        ]

    # ------------------------------------------------------------------
    # Field storage
    # ------------------------------------------------------------------

    def _write_field(self, data: np.ndarray, sdir: Path, name: str) -> Path:
        """Write a single field array (T, Ny, Nx) to zarr or npy."""
        if self.cfg.format == "zarr" and _ZARR:
            return self._write_zarr(data, sdir, name)
        return self._write_npy(data, sdir, name)

    def _write_zarr(self, data: np.ndarray, sdir: Path, name: str) -> Path:
        path = sdir / f"{name}.zarr"
        # Chunks: (1, Ny, Nx) — one timestep per chunk
        chunks = self.cfg.zarr_chunks or (1, data.shape[1], data.shape[2])
        compressor = None
        if self.cfg.zarr_compressor == "blosc":
            try:
                from numcodecs import Blosc
                compressor = Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)
            except ImportError:
                pass
        zarr.save_array(str(path), data, chunks=chunks, compressor=compressor)
        return path

    def _write_npy(self, data: np.ndarray, sdir: Path, name: str) -> Path:
        path = sdir / f"{name}.npy"
        np.save(str(path), data)
        return path


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def _json_default(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    return str(obj)


def _build_metadata(
    traj: TrajectoryData,
    spec: ScenarioSpec,
    T: int, C: int, Ny: int, Nx: int,
) -> Dict[str, Any]:
    return {
        "sample_id_generation_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scenario_name":  spec.name,
        "pde_kind":       spec.pde_kind,
        "grid_shape":     [Ny, Nx],
        "n_timesteps":    T,
        "n_fields":       C,
        "field_names":    spec.field_names,
        "t_span":         list(spec.scenario.t_span),
        "dt":             spec.scenario.dt,
        "domain_bounds":  [list(b) for b in spec.scenario.domain_bounds],
        "parameters":     traj.params,
        "solver":         spec.scenario.solver,
        "domain":         spec.domain,
        "geometry":       spec.geometry,
        "fluid":          spec.fluid,
        "solver_meta":    traj.metadata,
    }


def _build_pde_json(spec: ScenarioSpec) -> Dict[str, Any]:
    _equation_map = {
        "heat":       "∂T/∂t = α·∇²T",
        "burgers":    "∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²",
        "wave":       "∂²u/∂t² = c²·∇²u",
        "advection":  "∂φ/∂t + v·∇φ = 0",
        "ns2d":       "∂u/∂t + (u·∇)u = -∇p/ρ + ν·∇²u,  ∇·u = 0",
        "elasticity": "∇·σ = 0,  σ = C:ε",
    }
    return {
        "equation":    _equation_map.get(spec.pde_kind, spec.pde_kind),
        "pde_kind":    spec.pde_kind,
        "field_names": spec.field_names,
        "parameters":  {k: list(v) for k, v in spec.scenario.param_ranges.items()},
        "ic_type":     spec.initial_conditions.get("type", "unknown"),
        "bc_type":     spec.scenario.bc_type,
        "tags":        spec.scenario.tags,
    }


# ---------------------------------------------------------------------------
# VTK writer (structured grid, ASCII)
# ---------------------------------------------------------------------------

def _write_vtk(states: np.ndarray, spec: ScenarioSpec, path: Path) -> None:
    """Write a minimal VTK structured-grid file for ParaView compatibility."""
    try:
        T, C, Ny, Nx = states.shape
        bounds = spec.scenario.domain_bounds
        x0, x1 = bounds[0] if len(bounds) > 0 else (0, 1)
        y0, y1 = bounds[1] if len(bounds) > 1 else (0, 1)
        xs = np.linspace(x0, x1, Nx)
        ys = np.linspace(y0, y1, Ny)

        snap = states[-1]   # last timestep
        lines = [
            "# vtk DataFile Version 3.0",
            f"PINNeAPPle dataset — {spec.name}",
            "ASCII",
            "DATASET RECTILINEAR_GRID",
            f"DIMENSIONS {Nx} {Ny} 1",
            f"X_COORDINATES {Nx} float",
            " ".join(f"{v:.6f}" for v in xs),
            f"Y_COORDINATES {Ny} float",
            " ".join(f"{v:.6f}" for v in ys),
            "Z_COORDINATES 1 float",
            "0.0",
            f"POINT_DATA {Nx * Ny}",
        ]
        for i, fname in enumerate(spec.field_names[:C]):
            ch = snap[i].ravel(order="C")
            lines.append(f"SCALARS {fname} float 1")
            lines.append("LOOKUP_TABLE default")
            lines.append(" ".join(f"{v:.6f}" for v in ch))

        path.write_text("\n".join(lines))
    except Exception:
        pass   # VTK is optional — don't fail the pipeline
