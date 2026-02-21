"""CadQuery solid-to-PhysicalSample conversion via tessellation."""
from __future__ import annotations

from typing import Any, Dict, Optional
import torch
from pinneaple_data.physical_sample import PhysicalSample


def cadquery_solid_to_upd(
    solid,
    *,
    tess_linear_deflection: float = 0.2,
    tess_angular_deflection: float = 0.3,
    units: str = "mm",
    meta: Optional[Dict[str, Any]] = None,
) -> PhysicalSample:
    """
    Convert a CadQuery solid/workplane to a triangle mesh and wrap as PhysicalSample.
    """
    import cadquery as cq  # optional

    shape = solid.val() if hasattr(solid, "val") else solid
    tess = shape.tessellate(tess_linear_deflection, tess_angular_deflection)
    vertices, triangles = tess[0], tess[1]

    V = torch.tensor(vertices, dtype=torch.float32)
    F = torch.tensor(triangles, dtype=torch.long)

    m = dict(meta or {})
    m.setdefault("upd", {"version": "0.1", "domain": "geometry", "source": "cadquery"})
    m.setdefault("units", {"vertices": units})

    return PhysicalSample(fields={"vertices": V, "faces": F}, coords={}, meta=m)
