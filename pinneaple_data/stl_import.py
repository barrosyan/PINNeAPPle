"""STL import utilities (ASCII + binary).

Lightweight, dependency-free STL reader for server ingestion.

Outputs:
  - verts: (V,3) float32
  - faces: (F,3) int64
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import struct


@dataclass
class STLMesh:
    verts: np.ndarray
    faces: np.ndarray


def _dedup_vertices(tris: np.ndarray, *, dedup_eps: float) -> Tuple[np.ndarray, np.ndarray]:
    assert tris.ndim == 3 and tris.shape[1:] == (3, 3)
    F = tris.shape[0]
    pts = tris.reshape(-1, 3)

    q = np.round(pts / max(dedup_eps, 1e-12)).astype(np.int64)
    keys = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)

    idx_map = {}
    verts = []
    faces = np.zeros((F, 3), dtype=np.int64)
    for i in range(pts.shape[0]):
        k = int(keys[i])
        k2 = (k, int(q[i, 0]), int(q[i, 1]), int(q[i, 2]))
        if k2 not in idx_map:
            idx_map[k2] = len(verts)
            verts.append(pts[i])
        faces[i // 3, i % 3] = idx_map[k2]
    return np.asarray(verts, dtype=np.float32), faces


def load_stl_bytes(data: bytes, *, dedup_eps: float = 1e-6) -> STLMesh:
    if len(data) < 84:
        raise ValueError("STL data too small")

    tri_count = struct.unpack("<I", data[80:84])[0]
    expected_len = 84 + tri_count * 50
    is_binary = expected_len == len(data)

    if is_binary:
        tris = np.zeros((tri_count, 3, 3), dtype=np.float32)
        off = 84
        for i in range(tri_count):
            off += 12  # normal
            v = struct.unpack_from("<9f", data, off)
            tris[i, 0] = (v[0], v[1], v[2])
            tris[i, 1] = (v[3], v[4], v[5])
            tris[i, 2] = (v[6], v[7], v[8])
            off += 36
            off += 2
        verts, faces = _dedup_vertices(tris, dedup_eps=dedup_eps)
        return STLMesh(verts=verts, faces=faces)

    # ASCII fallback
    text = data.decode("utf-8", errors="ignore")
    verts_raw = []
    for line in text.splitlines():
        s = line.strip().lower()
        if not s.startswith("vertex"):
            continue
        parts = s.split()
        if len(parts) != 4:
            continue
        try:
            verts_raw.append((float(parts[1]), float(parts[2]), float(parts[3])))
        except Exception:
            continue
    if len(verts_raw) < 3 or (len(verts_raw) % 3) != 0:
        raise ValueError("ASCII STL parse failed")

    tris = np.asarray(verts_raw, dtype=np.float32).reshape(-1, 3, 3)
    verts, faces = _dedup_vertices(tris, dedup_eps=dedup_eps)
    return STLMesh(verts=verts, faces=faces)


def load_stl(path: str, *, dedup_eps: float = 1e-6) -> STLMesh:
    with open(path, "rb") as f:
        return load_stl_bytes(f.read(), dedup_eps=dedup_eps)
