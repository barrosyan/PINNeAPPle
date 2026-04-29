"""Physical scene representation for world-model integration.

Provides lightweight data classes for describing 3-D scenes that can be
passed to :class:`~pinneaple_worldmodel.adapter.CosmosAdapter` or used as
context for PINN training.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Scene object
# ---------------------------------------------------------------------------


@dataclass
class SceneObject:
    """A single rigid object in a physical scene.

    Attributes
    ----------
    name:
        Human-readable label.
    position:
        Centre-of-mass position ``(x, y, z)`` in world coordinates.
    orientation:
        Unit quaternion ``(w, x, y, z)`` (identity = no rotation).
    scale:
        Uniform scale factor.
    material:
        Material tag, e.g. ``"rigid"``, ``"elastic"``, ``"fluid"``.
    mesh_path:
        Optional path to an OBJ/STL mesh file.
    properties:
        Arbitrary physical properties (density, Young's modulus, …).
    """

    name: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: float = 1.0
    material: str = "rigid"
    mesh_path: Optional[str] = None
    properties: Dict[str, float] = field(default_factory=dict)

    def position_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Return position as a ``(3,)`` float32 tensor."""
        return torch.tensor(self.position, dtype=torch.float32, device=device)

    def orientation_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Return orientation quaternion as a ``(4,)`` float32 tensor."""
        return torch.tensor(self.orientation, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Physical scene
# ---------------------------------------------------------------------------


class PhysicalScene:
    """Container for all objects and fields in a simulated physical scene.

    Parameters
    ----------
    objects:
        List of :class:`SceneObject` instances.
    domain_bounds:
        ``((x_min, x_max), (y_min, y_max), (z_min, z_max))``
        bounding box of the simulation domain.
    description:
        Text description passed as a prompt to the world foundation model.
    """

    def __init__(
        self,
        objects: Optional[List[SceneObject]] = None,
        domain_bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
        description: str = "physical simulation",
    ) -> None:
        self.objects: List[SceneObject] = objects or []
        self.domain_bounds: Tuple[Tuple[float, float], ...] = domain_bounds or (
            (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)
        )
        self.description = description
        self._fields: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Object management
    # ------------------------------------------------------------------

    def add_object(self, obj: SceneObject) -> None:
        """Append a :class:`SceneObject` to the scene."""
        self.objects.append(obj)

    def get_object(self, name: str) -> Optional[SceneObject]:
        """Return the first object with *name*, or ``None``."""
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None

    # ------------------------------------------------------------------
    # Field storage
    # ------------------------------------------------------------------

    def set_field(self, name: str, data: torch.Tensor) -> None:
        """Store a named physical field (e.g. pressure, velocity)."""
        self._fields[name] = data

    def get_field(self, name: str) -> Optional[torch.Tensor]:
        """Retrieve a stored field by name."""
        return self._fields.get(name)

    # ------------------------------------------------------------------
    # Collocation point sampling
    # ------------------------------------------------------------------

    def sample_collocation_points(
        self, n: int, device: str = "cpu"
    ) -> torch.Tensor:
        """Sample *n* random collocation points inside the domain bounding box.

        Parameters
        ----------
        n : number of points
        device : target device

        Returns
        -------
        torch.Tensor  –  ``(n, 3)`` uniform samples in the domain.
        """
        lows = torch.tensor(
            [b[0] for b in self.domain_bounds], dtype=torch.float32, device=device
        )
        highs = torch.tensor(
            [b[1] for b in self.domain_bounds], dtype=torch.float32, device=device
        )
        return lows + (highs - lows) * torch.rand(n, len(self.domain_bounds),
                                                   device=device)

    def __repr__(self) -> str:
        return (
            f"PhysicalScene(n_objects={len(self.objects)}, "
            f"description='{self.description}')"
        )
