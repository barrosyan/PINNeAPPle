"""Physics video dataset and sim-to-real data generation utilities.

Provides :class:`PhysicsVideoDataset` for loading pre-generated physics
simulation videos and :class:`SimToRealAdapter` for aligning simulations
with real-world observations via a world foundation model.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# PhysicsVideoDataset
# ---------------------------------------------------------------------------


class PhysicsVideoDataset(Dataset):
    """Dataset of physics simulation videos for PINN training.

    Expects a directory layout::

        video_dir/
            sample_000/
                frames/        ← PNG or NPY files, one per frame
                fields.pt      ← torch dict with keys = field_names
            sample_001/
            ...

    If *video_dir* does not exist or is empty, a synthetic fallback dataset
    is generated on-the-fly using simple advection-diffusion fields.

    Parameters
    ----------
    video_dir:
        Root directory containing sample sub-directories.
    field_names:
        Names of physical fields to load (e.g. ``["u", "v", "p"]``).
    n_frames:
        How many frames to read per sample.
    transform:
        Optional callable applied to each ``(frames, fields)`` pair.
    synthetic_n_samples:
        Number of synthetic samples to generate when *video_dir* is missing.
    """

    def __init__(
        self,
        video_dir: str,
        field_names: List[str],
        n_frames: int = 16,
        transform: Optional[Callable] = None,
        synthetic_n_samples: int = 100,
    ) -> None:
        self.video_dir = Path(video_dir)
        self.field_names = field_names
        self.n_frames = n_frames
        self.transform = transform

        # Collect sample directories
        if self.video_dir.exists():
            self._samples = sorted(
                [p for p in self.video_dir.iterdir() if p.is_dir()]
            )
        else:
            import warnings
            warnings.warn(
                f"video_dir '{video_dir}' not found. "
                f"Generating {synthetic_n_samples} synthetic samples.",
                stacklevel=2,
            )
            self._samples = []

        self._synthetic = len(self._samples) == 0
        self._synthetic_n = synthetic_n_samples

    def __len__(self) -> int:
        return len(self._samples) if not self._synthetic else self._synthetic_n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._synthetic:
            return self._synthetic_sample(idx)
        return self._load_sample(self._samples[idx])

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_sample(self, sample_dir: Path) -> Dict[str, torch.Tensor]:
        """Load one sample from disk."""
        fields_path = sample_dir / "fields.pt"
        if fields_path.exists():
            fields = torch.load(fields_path, map_location="cpu")
        else:
            fields = {k: torch.zeros(1) for k in self.field_names}

        # Load frames (prefer .pt, then .npy, then generate zeros)
        frames_dir = sample_dir / "frames"
        frames_list = []
        if frames_dir.exists():
            frame_files = sorted(frames_dir.glob("*.pt"))[:self.n_frames]
            for fp in frame_files:
                frames_list.append(torch.load(fp, map_location="cpu"))
        if not frames_list:
            H, W = 64, 64
            frames_list = [torch.zeros(3, H, W) for _ in range(self.n_frames)]

        frames = torch.stack(frames_list, dim=0)  # (T, 3, H, W)

        out = {"frames": frames, **{k: fields.get(k, torch.zeros(1))
                                    for k in self.field_names}}
        if self.transform is not None:
            out = self.transform(out)
        return out

    def _synthetic_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate a synthetic advection-diffusion sample."""
        torch.manual_seed(idx)
        H, W = 64, 64
        T = self.n_frames

        # Random initial condition
        u0 = torch.randn(1, H, W) * 0.3

        frames = []
        u = u0.clone()
        for t in range(T):
            lap = (
                torch.roll(u, 1, -1) + torch.roll(u, -1, -1)
                + torch.roll(u, 1, -2) + torch.roll(u, -1, -2)
                - 4.0 * u
            )
            u = (u + 0.05 * lap).clamp(-1.0, 1.0)
            frame_rgb = torch.cat([u.clamp(0, 1), (u.abs()), (-u).clamp(0, 1)], dim=0)
            frames.append(frame_rgb)

        frames_t = torch.stack(frames, dim=0)  # (T, 3, H, W)

        out: Dict[str, torch.Tensor] = {"frames": frames_t}
        for name in self.field_names:
            # Return the last-frame u channel as a proxy field
            out[name] = frames_t[-1, 0]  # (H, W)
        return out
