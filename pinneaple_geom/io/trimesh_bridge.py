"""Trimesh to MeshData adapter for loading, repair, and conversion."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from pinneaple_geom.core.mesh import MeshData


class TrimeshBridge:
    """
    Adapter between trimesh.Trimesh and Pinneaple MeshData.

    Responsibilities:
      - Load meshes from disk
      - Repair / clean meshes
      - Convert to MeshData (internal representation)
    """

    def load(self, path: Union[str, Path], *, repair: bool = True) -> MeshData:
        """
        Load geometry using trimesh and return MeshData.
        """
        tm = self._load_trimesh(path)
        if repair:
            tm = self._repair_trimesh(tm)
        return self.from_trimesh(tm)

    def from_trimesh(self, tm, *, compute_normals: bool = True) -> MeshData:
        """
        Convert trimesh.Trimesh -> MeshData.
        """
        import trimesh

        if not isinstance(tm, trimesh.Trimesh):
            raise TypeError("from_trimesh expects a trimesh.Trimesh")

        v = tm.vertices.view(np.ndarray)
        f = tm.faces.view(np.ndarray)

        n = None
        if compute_normals:
            try:
                n = tm.face_normals.view(np.ndarray)
            except Exception:
                n = None

        return MeshData(vertices=v, faces=f, normals=n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_trimesh(self, path: Union[str, Path]):
        import trimesh

        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        m = trimesh.load(str(path), force="mesh")
        if not isinstance(m, trimesh.Trimesh):
            raise TypeError(f"Loaded object from '{path.name}' is not a trimesh.Trimesh.")
        return m

    def _repair_trimesh(self, tm):
        import trimesh

        if not isinstance(tm, trimesh.Trimesh):
            raise TypeError("repair_trimesh expects a trimesh.Trimesh")

        # Basic cleanups (fast, safe)
        tm.remove_duplicate_faces()
        tm.remove_degenerate_faces()
        tm.remove_unreferenced_vertices()

        # Normals
        try:
            tm.fix_normals()
        except Exception:
            pass

        # Holes (best effort)
        try:
            if not tm.is_watertight:
                tm.fill_holes()
        except Exception:
            pass

        return tm
