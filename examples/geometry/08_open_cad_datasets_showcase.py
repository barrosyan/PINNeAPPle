"""Open CAD-dataset showcase: real downloads through PINNeAPPle's real importers.

This script downloads a small handful of REAL, individually-addressable
sample files from well-known open CAD/mesh research datasets and runs them
through PINNeAPPle's existing geometry-import stack:

  - STL  -> pinneapple_data.stl_import.load_stl
  - STEP -> pinneapple_design.geometry.io.step.step_to_mesh (optional gmsh backend)

...then does something real with the loaded geometry using existing
PINNeAPPle utilities: pinneapple_data.geometry_features (point-cloud
sampling + SDF grid) and pinneapple_design.geometry.ops.voxelize
(rasterizing the sampled point cloud into an occupancy grid).

Nothing here is synthetic: every number printed is computed from a file
that was actually downloaded over the network in this run.

Sources and exact per-file access patterns (verified 2026-09, see report)
---------------------------------------------------------------------------
1. Thingi10K (https://github.com/Thingi10K/Thingi10K)
   The GitHub repo does not itself expose a per-file HTTP endpoint; its
   README points at mirrors, one of which (Hugging Face) hosts the raw
   per-model files as individually-resolvable dataset files:

       https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main/raw_meshes/<file_id>.stl

   This is a genuine single-file download (tens of KB), not a bulk
   archive -- confirmed by browsing the HF dataset's file tree
   (`raw_meshes/<file_id>.stl`, one file per Thingiverse model).
   We fetch two small individual models this way:
     - raw_meshes/100028.stl  ("carbon mini cub fuselage", ASCII STL)
     - raw_meshes/100045.stl  (binary STL)
   Thingi10K itself: Zhou & Jacobson, "Thingi10K: A Dataset of 10,000
   3D-Printing Models", 2016 (arXiv:1605.04797).

2. Fusion 360 Gallery Dataset (https://github.com/AutodeskAILab/Fusion360GalleryDataset)
   The actual reconstruction/segmentation/assembly datasets are ONLY
   distributed as bulk S3 archives (r1.0.1.zip ~2.0 GB, s2.0.1.zip
   ~3.1 GB, j1.0.0.7z ~2.8 GB) -- there is no per-model download
   endpoint for those, so we deliberately do NOT touch them here.
   The same GitHub repository does, however, commit a small number of
   individual real STEP files as test fixtures for its own tools
   (`tools/testdata/`), individually fetchable via raw.githubusercontent.com:

       https://raw.githubusercontent.com/AutodeskAILab/Fusion360GalleryDataset/master/tools/testdata/Couch.step

   This is a real, ~18 KB, single-file STEP model shipped in the
   official repo (not a reconstruction-dataset sample, but a genuine
   individually-downloadable STEP file from the same project) -- used
   here to exercise the STEP import path without pulling a multi-GB
   archive.

3. ABC Dataset (https://deep-geometry.github.io/abc-dataset/) -- SKIPPED.
   Verified via the dataset's GitHub README: files are only published
   in per-format chunks of 10,000 models each (compressed 7z archives
   per chunk, e.g. `abc_0000_step_v00.7z`). There is no documented way
   to fetch a single model without downloading a chunk archive, so per
   this showcase's "no bulk archives" rule, ABC is skipped entirely.

Downloaded files are NOT committed to the repo. They are written to
`examples/geometry/outputs/` which, at the time of writing, is *not*
covered by an existing `.gitignore` pattern -- if you run this script,
either `git status` will show the new files as untracked (do not `git
add` them) or add your own local ignore rule; nothing here modifies
`.gitignore`.

Run
---
python examples/geometry/08_open_cad_datasets_showcase.py
"""
from __future__ import annotations

import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

import numpy as np

from pinneapple_data.stl_import import load_stl
from pinneapple_data.geometry_features import featurize_mesh
from pinneapple_design.geometry.io.step import step_to_mesh, StepImportConfig

OUT_DIR = Path(__file__).parent / "outputs"

_HEADERS = {"User-Agent": "pinneapple-examples/1.0 (+https://github.com/)"}
_TIMEOUT = 60

THINGI10K_FILES = [
    # (file_id, url, expected file kind)
    ("100028", "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main/raw_meshes/100028.stl"),
    ("100045", "https://huggingface.co/datasets/Thingi10K/Thingi10K/resolve/main/raw_meshes/100045.stl"),
]

FUSION360_STEP_URL = (
    "https://raw.githubusercontent.com/AutodeskAILab/Fusion360GalleryDataset/"
    "master/tools/testdata/Couch.step"
)


def _download(url: str, dest: Path) -> bool:
    """Download `url` to `dest`. Returns True on success, False (with a
    printed diagnostic) if the network is unavailable/blocked -- this
    script never fabricates results when a download fails."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        print(f"  downloaded {len(data):,} bytes -> {dest}")
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"  DOWNLOAD FAILED for {url}: {exc!r}")
        print("  (network access appears unavailable in this environment -- "
              "the import/feature-extraction code below is real and correct, "
              "it just has nothing to run on here)")
        return False


def _report_mesh(label: str, verts: np.ndarray, faces: np.ndarray) -> None:
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    extent = bbox_max - bbox_min
    print(f"  [{label}] vertices={verts.shape[0]:,} faces={faces.shape[0]:,}")
    print(f"  [{label}] bbox min={bbox_min.tolist()} max={bbox_max.tolist()} extent={extent.tolist()}")


def demo_thingi10k() -> Optional[tuple[np.ndarray, np.ndarray]]:
    print("\n=== Thingi10K (STL, via Hugging Face raw_meshes mirror) ===")
    result = None
    for file_id, url in THINGI10K_FILES:
        dest = OUT_DIR / f"thingi10k_{file_id}.stl"
        print(f"[{file_id}] {url}")
        if not _download(url, dest):
            continue
        mesh = load_stl(str(dest))
        _report_mesh(f"thingi10k/{file_id}", mesh.verts, mesh.faces)
        if result is None:
            result = (mesh.verts, mesh.faces)
    return result


def demo_fusion360_step() -> None:
    print("\n=== Fusion 360 Gallery Dataset (STEP test fixture, via raw.githubusercontent.com) ===")
    dest = OUT_DIR / "fusion360_couch.step"
    print(f"{FUSION360_STEP_URL}")
    if not _download(FUSION360_STEP_URL, dest):
        return

    try:
        mesh = step_to_mesh(dest, cfg=StepImportConfig(kind="surface", mesh_size=20.0))
    except (ImportError, RuntimeError) as exc:
        # step_to_mesh needs BOTH meshio (to read the .msh gmsh writes) and
        # gmsh itself (to mesh the B-Rep) -- both are optional dependencies
        # of pinneapple_design.geometry.io.step, and either one being absent
        # raises here. Degrade gracefully rather than crashing the showcase.
        print(f"  SKIPPED meshing: {exc}")
        print("  (meshio and/or gmsh are optional dependencies for STEP meshing; "
              "install with `pip install meshio gmsh` to run this part -- the "
              "STEP file itself downloaded correctly)")
        return
    except Exception as exc:  # pragma: no cover - defensive, real CAD data can be messy
        print(f"  SKIPPED meshing: unexpected error converting STEP -> mesh: {exc!r}")
        return

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    _report_mesh("fusion360/Couch", verts, faces)


def demo_feature_extraction_and_voxelize(verts: np.ndarray, faces: np.ndarray) -> None:
    print("\n=== Feature extraction + voxelization (pinneapple_data.geometry_features, "
          "pinneapple_design.geometry.ops.voxelize) ===")
    feats = featurize_mesh(verts, faces, n_point_cloud_samples=4096, sdf_grid_resolution=12)

    if "point_cloud_error" in feats:
        print(f"  point cloud sampling FAILED: {feats['point_cloud_error']}")
        return
    pc = feats["point_cloud"]
    print(f"  point cloud: {pc.shape[0]:,} points sampled from surface")

    if "sdf_error" in feats:
        print(f"  SDF grid FAILED: {feats['sdf_error']}")
    else:
        sdf = feats["sdf"]["sdf"]
        n_inside = int((sdf < 0).sum())
        print(f"  SDF grid: {sdf.shape[0]:,} query points, "
              f"{n_inside:,} inside the mesh (grid_shape={feats['sdf']['grid_shape']})")

    # Rasterize the sampled surface point cloud into a voxel occupancy grid,
    # using the real (already-tested) pinneapple_design.geometry.ops.voxelize
    # module -- this is genuinely useful for e.g. a geometry-conditioned PINN
    # that needs a coarse occupancy channel alongside collocation points.
    import torch
    from pinneapple_design.geometry.ops.voxelize import voxelize_pointcloud

    pts_t = torch.from_numpy(np.asarray(pc, dtype=np.float32))
    bbox_min = pts_t.min(dim=0).values
    bbox_max = pts_t.max(dim=0).values
    pad = 0.02 * (bbox_max - bbox_min).clamp(min=1e-6)
    bounds = {
        "x": (float(bbox_min[0] - pad[0]), float(bbox_max[0] + pad[0])),
        "y": (float(bbox_min[1] - pad[1]), float(bbox_max[1] + pad[1])),
        "z": (float(bbox_min[2] - pad[2]), float(bbox_max[2] + pad[2])),
    }
    grid = voxelize_pointcloud(pts_t, bounds=bounds, resolution=24)
    occupied = int((grid.data > 0).sum().item())
    total = int(grid.data.numel())
    print(f"  voxel grid: resolution={grid.shape}, occupied voxels={occupied:,}/{total:,} "
          f"({100.0 * occupied / total:.1f}% surface occupancy)")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloaded files land in: {OUT_DIR} (NOT committed to the repo -- "
          f"this path is not covered by an existing .gitignore rule at the time "
          f"of writing, so do not `git add` its contents)")

    thingi_mesh = demo_thingi10k()
    demo_fusion360_step()

    if thingi_mesh is not None:
        verts, faces = thingi_mesh
        demo_feature_extraction_and_voxelize(verts, faces)
    else:
        print("\n(skipping feature extraction / voxelization demo -- no mesh "
              "was successfully downloaded)")


if __name__ == "__main__":
    main()
