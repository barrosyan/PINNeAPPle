from __future__ import annotations
"""Hash Grid Positional Encoding — Instant-NGP style.

Reference: Müller et al. 2022
  "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
  https://arxiv.org/abs/2201.05989
"""

import math
from typing import List

import torch
import torch.nn as nn

from .base import BaseModel, ModelOutput


class HashGridEncoding(nn.Module):
    """Multi-resolution hash grid encoding (Instant-NGP style).

    Stores a set of learned embedding tables, one per resolution level.
    For each level the spatial coordinates are scaled to the level resolution,
    the corners of the surrounding voxel are identified, their embeddings are
    fetched via a spatial hash, and the result is interpolated (trilinearly for
    3-D, bilinearly for 2-D, etc.).  The concatenation across all levels forms
    the output feature vector.

    This representation is compact (hash tables rather than dense grids) and
    can resolve high-frequency details with few parameters.

    Args:
        in_dim: Spatial dimension (2 or 3 are most common).
        n_levels: Number of resolution levels ``L``.
        n_features_per_level: Feature dimension at each level ``F``.
        log2_hashmap_size: ``log2`` of the hash table size ``T = 2**log2_hashmap_size``.
        base_resolution: Coarsest grid resolution ``N_min``.
        finest_resolution: Finest grid resolution ``N_max``.
    """

    def __init__(
        self,
        in_dim: int = 3,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution
        self.out_dim = n_levels * n_features_per_level

        # Growth factor b such that N_min * b^(L-1) == N_max
        b = math.exp(
            (math.log(finest_resolution) - math.log(base_resolution)) / max(n_levels - 1, 1)
        )
        resolutions = [int(base_resolution * (b ** i)) for i in range(n_levels)]
        self.register_buffer("resolutions", torch.tensor(resolutions, dtype=torch.long))

        table_size = 2 ** log2_hashmap_size
        self.embeddings = nn.ModuleList(
            [nn.Embedding(table_size, n_features_per_level) for _ in range(n_levels)]
        )
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)

        # Spatial hash primes (one per dimension beyond the first)
        self._primes: List[int] = [1, 2654435761, 805459861, 3674653429]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hash(self, coords_int: torch.Tensor) -> torch.Tensor:
        """Spatial hash: XOR reduction across dimensions with large primes.

        Args:
            coords_int: ``(..., in_dim)`` integer grid coordinates.

        Returns:
            Hash indices of shape ``(...)`` in ``[0, 2**log2_hashmap_size)``.
        """
        h = coords_int[..., 0].clone()
        for d in range(1, self.in_dim):
            prime = self._primes[d % len(self._primes)]
            h = h ^ (coords_int[..., d] * prime)
        return h % (2 ** self.log2_hashmap_size)

    def _interpolate_level(self, x_norm: torch.Tensor, level_idx: int) -> torch.Tensor:
        """N-linear interpolation at a single resolution level.

        Args:
            x_norm: Normalised coordinates in ``[0, 1]``, shape ``(..., in_dim)``.
            level_idx: Which resolution level to query.

        Returns:
            Interpolated features, shape ``(..., n_features_per_level)``.
        """
        res = int(self.resolutions[level_idx].item())
        x_scaled = x_norm * (res - 1)
        x_floor = x_scaled.long().clamp(0, res - 2)
        x_frac = x_scaled - x_floor.float()

        n_corners = 2 ** self.in_dim
        feat = torch.zeros(
            *x_norm.shape[:-1],
            self.n_features_per_level,
            device=x_norm.device,
            dtype=x_norm.dtype,
        )

        for corner in range(n_corners):
            offset = torch.tensor(
                [(corner >> d) & 1 for d in range(self.in_dim)],
                device=x_norm.device,
                dtype=torch.long,
            )
            corner_coords = (x_floor + offset).clamp(0, res - 1)
            h = self._hash(corner_coords)
            corner_feat = self.embeddings[level_idx](h).to(dtype=x_norm.dtype)

            # Interpolation weight (product of per-dim barycentric weights)
            w = torch.ones(*x_norm.shape[:-1], 1, device=x_norm.device, dtype=x_norm.dtype)
            for d in range(self.in_dim):
                if (corner >> d) & 1:
                    w = w * x_frac[..., d : d + 1]
                else:
                    w = w * (1.0 - x_frac[..., d : d + 1])

            feat = feat + w * corner_feat

        return feat

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode coordinates.

        Args:
            x: Spatial coordinates in ``[0, 1]``, shape ``(..., in_dim)``.

        Returns:
            Multi-resolution features, shape ``(..., out_dim)``
            where ``out_dim = n_levels * n_features_per_level``.
        """
        features = [self._interpolate_level(x, l) for l in range(self.n_levels)]
        return torch.cat(features, dim=-1)


class HashGridMLP(BaseModel):
    """MLP with multi-resolution hash grid encoding for fast 3-D field representation.

    Combines the compact, high-capacity hash grid encoding with a small MLP
    decoder.  Achieves high accuracy with a low parameter count because the
    hash tables absorb most of the spatial complexity.

    Args:
        in_dim: Input coordinate dimension (typically 2 or 3).
        out_dim: Number of output fields.
        hidden_dim: Width of the decoder MLP hidden layers.
        n_hidden: Number of hidden layers in the decoder.
        n_levels: Number of hash grid resolution levels.
        n_features_per_level: Feature size per level.
        log2_hashmap_size: log2 of the per-level hash table size.
        base_resolution: Coarsest grid resolution.
        finest_resolution: Finest grid resolution.
    """

    family: str = "neural_operators"
    name: str = "hash_grid_mlp"

    def __init__(
        self,
        in_dim: int = 3,
        out_dim: int = 1,
        hidden_dim: int = 64,
        n_hidden: int = 2,
        n_levels: int = 8,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 16,
        base_resolution: int = 16,
        finest_resolution: int = 128,
    ):
        super().__init__()
        self.encoder = HashGridEncoding(
            in_dim=in_dim,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
        )
        enc_dim = self.encoder.out_dim

        layers: list[nn.Module] = [nn.Linear(enc_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> ModelOutput:  # type: ignore[override]
        """Forward pass.

        Args:
            x: Coordinates in ``[0, 1]``, shape ``(..., in_dim)``.

        Returns:
            :class:`~pinneaple_models.base.ModelOutput` with ``y`` of shape
            ``(..., out_dim)``.
        """
        return ModelOutput(y=self.mlp(self.encoder(x)))
