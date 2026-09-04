"""Cell-center / cell-size reconstruction from a binary OpenFOAM polyMesh.

PINNeAPPle has no ``constant/polyMesh`` reader anywhere: cell-center
coordinates were only ever recoverable via an optional ``C`` field, which
only exists if the case was run with OpenFOAM's ``writeCellCentres``
function object -- i.e. never, for an arbitrary case someone hands you.
This module reconstructs per-cell positions and sizes directly from the
mesh itself (``points``, ``owner``, ``neighbour``, ``faces``), no solver
run required beyond ``blockMesh``/``snappyHexMesh`` having produced the
mesh files.

Method
------
For every face, sum the coordinates of its vertices and add that sum to
the owner cell's accumulator, and to the neighbour cell's accumulator for
internal faces. For an axis-aligned hexahedral cell (true for any
``blockMesh``-graded structured/rectilinear mesh -- channels, pipes,
ducts, and similar canonical cases) each of the 8 distinct vertices of a
cell is shared by exactly 3 of its 6 faces, so summing all face-vertices
with equal weight and dividing by the total count already equals the mean
of the 8 distinct vertices, i.e. the exact centroid of a rectangular box,
with no need to de-duplicate vertices per cell. The same running min/max
over face-vertex coordinates gives the cell's axis-aligned bounding box,
whose extents equal the true cell ``(dx, dy, dz)`` for these box-shaped
cells.

For a general (non-box) cell this centroid/bounding-box are the common,
cheap face-vertex-average approximation of the true (face-decomposition)
centroid/volume, not an exact finite-volume computation -- adequate for
placing training samples and sizing an LES filter width, not for
finite-volume accounting. A proper arbitrary-polyhedron centroid/volume
(tet-decomposition from face centers) is straightforward to add on top of
the same ``points``/``owner``/``neighbour``/``faces`` arrays this module
already parses, if a future caller needs it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import binary_reader as _bin


@dataclass
class MeshGeometry:
    """Minimal geometry container satisfying ``pinneapple_data.dataloaders
    .build_physical_sample_dataloader``'s mesh-domain contract (a
    ``.nodes`` attribute of shape ``(N, d)``), plus the extra per-cell
    sizing info a filter-width-dependent closure (e.g. LES) needs.
    """
    nodes: np.ndarray  # (n_cells, 3) cell centers -- the .nodes contract
    cell_size: np.ndarray  # (n_cells, 3) axis-aligned bounding-box extents
    cell_delta: np.ndarray  # (n_cells,) cubeRootVol = (dx*dy*dz)**(1/3)
    bounds_min: np.ndarray
    bounds_max: np.ndarray
    n_points: int
    n_faces: int
    n_internal_faces: int


def _read_member_bytes(zf_or_dir, name: str) -> bytes:
    if zf_or_dir is None:
        raise ValueError("no source given")
    if hasattr(zf_or_dir, "open"):  # zipfile.ZipFile
        with zf_or_dir.open(name) as fh:
            return fh.read()
    import os
    with open(os.path.join(zf_or_dir, name), "rb") as fh:
        return fh.read()


def load_mesh(source, prefix: str = "constant/polyMesh/") -> MeshGeometry:
    """Build cell centers/sizes from a binary polyMesh.

    ``source`` is either an open ``zipfile.ZipFile`` (e.g. for a zipped
    case export) or a plain case-directory path.
    """
    points = _bin.read_vector_field_points(_read_member_bytes(source, prefix + "points"), prefix + "points")
    owner = _bin.read_label_list(_read_member_bytes(source, prefix + "owner"), prefix + "owner")
    neighbour = _bin.read_label_list(_read_member_bytes(source, prefix + "neighbour"), prefix + "neighbour")
    offsets, indices = _bin.read_face_compact_list(_read_member_bytes(source, prefix + "faces"), prefix + "faces")

    n_faces = offsets.shape[0] - 1
    n_internal = neighbour.shape[0]
    n_cells = int(owner.max()) + 1

    face_points = points[indices]  # (total_indices, 3)
    starts = offsets[:-1].astype(np.int64)

    face_sum = np.add.reduceat(face_points, starts, axis=0)
    face_count = np.diff(offsets).astype(np.float64)[:, None]
    face_min = np.minimum.reduceat(face_points, starts, axis=0)
    face_max = np.maximum.reduceat(face_points, starts, axis=0)

    cell_sum = np.zeros((n_cells, 3), dtype=np.float64)
    cell_count = np.zeros((n_cells, 1), dtype=np.float64)
    cell_min = np.full((n_cells, 3), np.inf, dtype=np.float64)
    cell_max = np.full((n_cells, 3), -np.inf, dtype=np.float64)

    np.add.at(cell_sum, owner, face_sum)
    np.add.at(cell_count, owner, face_count)
    np.minimum.at(cell_min, owner, face_min)
    np.maximum.at(cell_max, owner, face_max)

    np.add.at(cell_sum, neighbour, face_sum[:n_internal])
    np.add.at(cell_count, neighbour, face_count[:n_internal])
    np.minimum.at(cell_min, neighbour, face_min[:n_internal])
    np.maximum.at(cell_max, neighbour, face_max[:n_internal])

    cell_centers = cell_sum / cell_count
    cell_size = cell_max - cell_min
    cell_delta = np.cbrt(np.prod(np.maximum(cell_size, 1e-30), axis=1))

    return MeshGeometry(
        nodes=cell_centers,
        cell_size=cell_size,
        cell_delta=cell_delta,
        bounds_min=points.min(axis=0),
        bounds_max=points.max(axis=0),
        n_points=points.shape[0],
        n_faces=n_faces,
        n_internal_faces=n_internal,
    )
