# -*- coding: utf-8 -*-
"""Stage 9 — Model Training Layer.

Provides training-ready dataset loaders and model interfaces for the four
Physical AI learning objectives described in the SHIFT-Physics pipeline:

  1. Cosmos Encoder      video  → multimodal latent embedding
  2. Physics Decoder     embedding → physical fields (T, p, u, C)
  3. Inverse PINN        video  → physical parameters (viscosity, diffusivity, …)
  4. Neural Operator     state_t → state_{t+1}   (forward prediction)

Each hook wraps the packaged dataset (from DatasetPackager) into a
PyTorch Dataset / DataLoader that feeds the corresponding model architecture
available in ``pinneapple_neural``.

Public API
----------
  PhysicsAIDataset      — base PyTorch Dataset over packaged samples
  CosmosEncoderDataset  — video frames + (optional) field embeddings
  PhysicsDecoderDataset — latent embeddings → field snapshots
  InversePINNDataset    — video frames → PDE parameter targets
  NeuralOperatorDataset — (state_t, state_{t+1}) pairs
  TrainingHooks         — convenience wrapper: pick the right dataset
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------

def _load_manifest(dataset_dir: Path) -> List[Dict[str, Any]]:
    manifest_path = Path(dataset_dir) / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"dataset_manifest.json not found in {dataset_dir}")
    return json.loads(manifest_path.read_text())


_SEMANTIC_ALIASES: Dict[str, List[str]] = {
    # PDE name -> list of semantic field names that contain it
    "u":  ["velocity", "u", "ux"],
    "v":  ["velocity", "v", "uy"],
    "w":  ["velocity", "w", "uz"],
    "T":  ["temperature", "T", "temp"],
    "p":  ["pressure", "p"],
    "C":  ["concentration", "C", "phi"],
}
_VELOCITY_COMP = {"u": 0, "v": 1, "w": 2}


def _load_field(entry: Dict, field_name: str) -> Optional[np.ndarray]:
    """Load a physical field from a manifest entry.

    Handles both PDE-style names ('u', 'v', 'p', 'T') and semantic names
    ('velocity', 'pressure', 'temperature') to support both DatasetPackager
    and GroundTruthPackager manifest formats.
    """
    fields = entry.get("fields", {})

    def _read_path(path_str: str) -> Optional[np.ndarray]:
        path = Path(path_str)
        if path.suffix == ".zarr" or path.is_dir():
            try:
                import zarr
                return np.array(zarr.open(str(path), mode="r"))
            except Exception:
                pass
        npy_path = path.with_suffix(".npy")
        if npy_path.exists():
            return np.load(str(npy_path))
        if path.exists():
            return np.load(str(path))
        return None

    # Direct lookup
    if field_name in fields:
        return _read_path(fields[field_name])

    # Semantic alias lookup (e.g. "u" -> try "velocity", extract channel 0)
    for alias in _SEMANTIC_ALIASES.get(field_name, []):
        if alias in fields:
            arr = _read_path(fields[alias])
            if arr is None:
                continue
            # velocity.npy has shape (T, 2|3, Ny, Nx) -> extract component
            if field_name in _VELOCITY_COMP and alias == "velocity" and arr.ndim == 4:
                comp = _VELOCITY_COMP[field_name]
                if comp < arr.shape[1]:
                    return arr[:, comp]
            return arr

    return None


def _load_video_frames(entry: Dict, sensor: str) -> Optional[np.ndarray]:
    """Load video frames from a manifest entry (returns (T, H, W, 3) uint8)."""
    videos = entry.get("videos", {})
    key = f"cam_0_{sensor}" if f"cam_0_{sensor}" in videos else sensor
    if key not in videos:
        return None
    path = Path(videos[key])
    if path.is_file():
        try:
            import imageio
            reader = imageio.get_reader(str(path))
            frames = [frame for frame in reader]
            return np.stack(frames, axis=0)
        except Exception:
            pass
    # PNG sequence fallback
    if path.is_dir():
        pngs = sorted(path.glob("*.png"))
        if pngs:
            try:
                import imageio
                return np.stack([imageio.imread(str(p)) for p in pngs], axis=0)
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# 1. Base PhysicsAIDataset
# ---------------------------------------------------------------------------

class PhysicsAIDataset(Dataset):
    """Base dataset over a packaged Physical AI dataset directory.

    Loads physical fields and optionally video frames for each sample.

    Parameters
    ----------
    dataset_dir : Path or str
        Root directory containing the dataset_manifest.json.
    fields : list of str
        Physical field names to load (e.g. ``["u", "v", "p", "T"]``).
    sensors : list of str
        Sensor channels to load (e.g. ``["rgb", "thermal"]``).
    max_samples : int or None
        Limit the number of samples (for quick testing).
    device : str
        Device for tensors.
    """

    def __init__(
        self,
        dataset_dir:  Any,
        fields:       List[str]       = ("u", "v", "p", "T"),
        sensors:      List[str]       = (),
        max_samples:  Optional[int]   = None,
        device:       str             = "cpu",
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.fields      = list(fields)
        self.sensors     = list(sensors)
        self.device      = torch.device(device)
        self.manifest    = _load_manifest(self.dataset_dir)
        if max_samples:
            self.manifest = self.manifest[:max_samples]

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.manifest[idx]
        item: Dict[str, torch.Tensor] = {}

        for fname in self.fields:
            arr = _load_field(entry, fname)
            if arr is not None:
                item[fname] = torch.tensor(arr, dtype=torch.float32, device=self.device)

        for sensor in self.sensors:
            frames = _load_video_frames(entry, sensor)
            if frames is not None:
                # Normalise uint8 to [0, 1]
                item[f"video_{sensor}"] = torch.tensor(
                    frames.astype(np.float32) / 255.0,
                    device=self.device,
                )

        item["sample_id"] = torch.tensor(idx, device=self.device)
        return item

    def to_dataloader(self, batch_size: int = 8, shuffle: bool = True, **kw) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kw)


# ---------------------------------------------------------------------------
# 2. CosmosEncoderDataset  (video → embedding)
# ---------------------------------------------------------------------------

class CosmosEncoderDataset(PhysicsAIDataset):
    """Dataset for training a video → latent embedding encoder.

    Returns:
      - ``video_rgb``   : (T, H, W, 3) float32 in [0, 1]
      - ``video_thermal``: (T, H, W, 3) float32
      - ``fields``      : concatenated physical fields (T, C, Ny, Nx)
      - ``params``      : PDE parameter vector

    The embedding target can be learned via contrastive loss (video ↔ field)
    or via reconstruction loss (autoencoder).
    """

    def __init__(
        self,
        dataset_dir:   Any,
        video_sensors: List[str] = ("rgb", "thermal"),
        field_names:   List[str] = ("u", "v", "p", "T"),
        **kwargs,
    ) -> None:
        super().__init__(dataset_dir, fields=field_names, sensors=video_sensors, **kwargs)
        self.video_sensors = video_sensors
        self.field_names   = field_names

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base = super().__getitem__(idx)
        # Stack all fields along channel dimension
        flist = [base[f] for f in self.field_names if f in base]
        if flist:
            base["fields"] = torch.stack(flist, dim=1)   # (T, C, Ny, Nx)
        # Load PDE metadata
        entry    = self.manifest[idx]
        meta_path = Path(entry.get("metadata", ""))
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            params = list(meta.get("parameters", {}).values())
            base["params"] = torch.tensor(params, dtype=torch.float32, device=self.device)
        return base


# ---------------------------------------------------------------------------
# 3. PhysicsDecoderDataset  (embedding → physical fields)
# ---------------------------------------------------------------------------

class PhysicsDecoderDataset(Dataset):
    """Dataset for training a latent embedding → physical field decoder.

    Assumes embeddings have been pre-computed and saved alongside the samples.
    If no embedding file exists, returns a zero embedding as placeholder.

    Returns:
      - ``embedding``   : (D,) float32
      - ``fields``      : (T, C, Ny, Nx) float32
    """

    def __init__(
        self,
        dataset_dir:    Any,
        embedding_dim:  int       = 512,
        field_names:    List[str] = ("u", "v", "p", "T"),
        device:         str       = "cpu",
        max_samples:    Optional[int] = None,
    ) -> None:
        self.dataset_dir    = Path(dataset_dir)
        self.field_names    = field_names
        self.embedding_dim  = embedding_dim
        self.device         = torch.device(device)
        self.manifest       = _load_manifest(self.dataset_dir)
        if max_samples:
            self.manifest = self.manifest[:max_samples]

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.manifest[idx]
        sdir  = Path(entry["sample_dir"])

        # Embedding
        emb_path = sdir / "embedding.npy"
        if emb_path.exists():
            emb = torch.tensor(np.load(str(emb_path)), dtype=torch.float32, device=self.device)
        else:
            emb = torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device)

        # Fields
        flist = []
        for fname in self.field_names:
            arr = _load_field(entry, fname)
            if arr is not None:
                flist.append(torch.tensor(arr, dtype=torch.float32, device=self.device))
        fields = torch.stack(flist, dim=1) if flist else torch.zeros(1)

        return {"embedding": emb, "fields": fields}

    def to_dataloader(self, batch_size: int = 8, shuffle: bool = True, **kw) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kw)


# ---------------------------------------------------------------------------
# 4. InversePINNDataset  (video → PDE parameters)
# ---------------------------------------------------------------------------

class InversePINNDataset(PhysicsAIDataset):
    """Dataset for inverse PINN training: video frames → physical parameters.

    The model learns to estimate viscosity, diffusivity, reaction constants,
    or material properties from visual observations alone.

    Returns:
      - ``video_rgb``   : (T, H, W, 3) float32
      - ``params``      : (P,) float32 — ground-truth PDE parameters
      - ``param_names`` : list of str (stored as metadata attribute)
    """

    def __init__(
        self,
        dataset_dir:   Any,
        param_keys:    Optional[List[str]] = None,
        video_sensor:  str = "rgb",
        **kwargs,
    ) -> None:
        super().__init__(dataset_dir, sensors=[video_sensor], fields=[], **kwargs)
        self.param_keys   = param_keys
        self.video_sensor = video_sensor

        # Infer param keys from first sample if not provided
        if not self.param_keys and self.manifest:
            meta_path = Path(self.manifest[0].get("metadata", ""))
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                self.param_keys = list(meta.get("parameters", {}).keys())

    @property
    def param_names(self) -> List[str]:
        return self.param_keys or []

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        base  = super().__getitem__(idx)
        entry = self.manifest[idx]

        meta_path = Path(entry.get("metadata", ""))
        if meta_path.exists():
            meta   = json.loads(meta_path.read_text())
            params = meta.get("parameters", {})
            keys   = self.param_keys or list(params.keys())
            base["params"] = torch.tensor(
                [float(params.get(k, 0.0)) for k in keys],
                dtype=torch.float32, device=self.device,
            )
        return base


# ---------------------------------------------------------------------------
# 5. NeuralOperatorDataset  (state_t → state_{t+1})
# ---------------------------------------------------------------------------

class NeuralOperatorDataset(Dataset):
    """Dataset for training neural operators on (state_t, state_{t+1}) pairs.

    Compatible with FNO, DeepONet, and WNO architectures in
    ``pinneapple_neural.architectures``.

    Returns:
      - ``state_t``   : (C, Ny, Nx) float32 — input field snapshot
      - ``state_tp1`` : (C, Ny, Nx) float32 — target snapshot at t+horizon
      - ``params``    : (P,) float32 — PDE parameter conditioning vector
    """

    def __init__(
        self,
        dataset_dir:  Any,
        field_names:  List[str] = ("u", "v", "p"),
        horizon:      int       = 1,
        subsample_t:  int       = 1,
        device:       str       = "cpu",
        max_samples:  Optional[int] = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.field_names = field_names
        self.horizon     = horizon
        self.subsample_t = subsample_t
        self.device      = torch.device(device)
        self.manifest    = _load_manifest(self.dataset_dir)
        if max_samples:
            self.manifest = self.manifest[:max_samples]

        # Pre-build index: (sample_idx, t)
        self._index: List[Tuple[int, int]] = []
        for si, entry in enumerate(self.manifest):
            meta_path = Path(entry.get("metadata", ""))
            if meta_path.exists():
                n_steps = json.loads(meta_path.read_text()).get("n_timesteps", 32)
            else:
                n_steps = 32
            for t in range(0, n_steps - horizon, subsample_t):
                self._index.append((si, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        si, t = self._index[idx]
        entry = self.manifest[si]

        channels_t   = []
        channels_tp1 = []
        for fname in self.field_names:
            arr = _load_field(entry, fname)
            if arr is not None:
                channels_t.append(arr[t])
                channels_tp1.append(arr[min(t + self.horizon, arr.shape[0] - 1)])

        state_t   = torch.tensor(np.stack(channels_t,   axis=0), dtype=torch.float32, device=self.device)
        state_tp1 = torch.tensor(np.stack(channels_tp1, axis=0), dtype=torch.float32, device=self.device)

        meta_path = Path(entry.get("metadata", ""))
        params = torch.zeros(1, device=self.device)
        if meta_path.exists():
            p_dict = json.loads(meta_path.read_text()).get("parameters", {})
            if p_dict:
                params = torch.tensor(list(p_dict.values()), dtype=torch.float32, device=self.device)

        return {"state_t": state_t, "state_tp1": state_tp1, "params": params}

    def to_dataloader(self, batch_size: int = 8, shuffle: bool = True, **kw) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, **kw)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

class TrainingHooks:
    """Convenience factory for all training dataset types.

    Parameters
    ----------
    dataset_dir : Path
        Packaged dataset directory (contains ``dataset_manifest.json``).
    device : str
        PyTorch device.

    Examples
    --------
    ::

        hooks = TrainingHooks("./physics_dataset")

        # Neural operator
        ds_no = hooks.neural_operator(field_names=["u", "v", "p"], horizon=1)
        dl    = ds_no.to_dataloader(batch_size=16)

        # Inverse PINN
        ds_inv = hooks.inverse_pinn(video_sensor="rgb")
        dl_inv = ds_inv.to_dataloader(batch_size=8)
    """

    def __init__(self, dataset_dir: Any, device: str = "cpu") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.device      = device

    def base(self, **kw) -> PhysicsAIDataset:
        return PhysicsAIDataset(self.dataset_dir, device=self.device, **kw)

    def cosmos_encoder(self, **kw) -> CosmosEncoderDataset:
        return CosmosEncoderDataset(self.dataset_dir, device=self.device, **kw)

    def physics_decoder(self, **kw) -> PhysicsDecoderDataset:
        return PhysicsDecoderDataset(self.dataset_dir, device=self.device, **kw)

    def inverse_pinn(self, **kw) -> InversePINNDataset:
        return InversePINNDataset(self.dataset_dir, device=self.device, **kw)

    def neural_operator(self, **kw) -> NeuralOperatorDataset:
        return NeuralOperatorDataset(self.dataset_dir, device=self.device, **kw)
