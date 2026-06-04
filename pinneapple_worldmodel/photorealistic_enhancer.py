# -*- coding: utf-8 -*-
"""Stage 7 — Photorealistic Enhancer.

Improves the visual realism of rendered physics frames while strictly
preserving all physical field data.

CRITICAL CONSTRAINT
-------------------
The enhancer operates ONLY on image data (uint8 RGB frames).  It NEVER
reads or modifies:
  - velocity, pressure, temperature, concentration fields
  - zarr arrays stored in the physical dataset
  - metadata, boundary conditions, or PDE parameters

Backends
--------
  stub       Always available.  Returns input unchanged (identity pass).
             Use for development and when no GPU is available.

  local_diffusion
             Requires ``diffusers`` package (pip install diffusers).
             Applies a lightweight image-to-image Stable Diffusion pass
             with low denoising strength (0.15-0.30) to preserve structure.

  cosmos     Requires NVIDIA Cosmos API access.  Sends frames to the
             Cosmos video generation API and returns enhanced output.
             NOT YET AVAILABLE — placeholder for future integration.

  custom     User-supplied enhancement function registered via
             ``register_backend()``.

Public API
----------
  EnhancerConfig          — which backend + strength
  PhotorealisticEnhancer  — main class
  register_enhancer       — register a custom backend function
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Custom backend registry
# ---------------------------------------------------------------------------

_CUSTOM_BACKENDS: Dict[str, Callable] = {}


def register_enhancer(name: str, fn: Callable) -> None:
    """Register a custom enhancement backend.

    Parameters
    ----------
    name : str
        Backend identifier (used in ``EnhancerConfig.backend``).
    fn : callable
        Signature: ``fn(frames: np.ndarray, config: EnhancerConfig) -> np.ndarray``
        where ``frames`` is (T, H, W, 3) uint8.
    """
    _CUSTOM_BACKENDS[name] = fn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EnhancerConfig:
    """Configuration for the photorealistic enhancer.

    Parameters
    ----------
    backend : str
        Which enhancement backend to use:
        ``"stub"``, ``"local_diffusion"``, ``"cosmos"``, or a custom name.
    strength : float
        Denoising / enhancement strength (0 = no change, 1 = full re-generation).
        Keep below 0.35 to preserve physical structure in rendered frames.
    style_prompt : str
        Text prompt guiding the diffusion enhancer (ignored by stub).
    negative_prompt : str
        Negative prompt for diffusion (ignored by stub).
    model_id : str
        Hugging Face model ID for ``local_diffusion`` backend.
    device : str
        Compute device (``"cuda"``, ``"cpu"``).
    batch_size : int
        Frames processed per diffusion forward pass.
    apply_to_sensors : list of str
        Which sensor channels to enhance.  Empty list = enhance all.
    """
    backend:          str        = "stub"
    strength:         float      = 0.20
    style_prompt:     str        = (
        "photorealistic industrial fluid flow, high dynamic range, "
        "4K detail, physically accurate, lab quality"
    )
    negative_prompt:  str        = "cartoon, painting, sketch, blurry, artificial"
    model_id:         str        = "stabilityai/stable-diffusion-2-1"
    device:           str        = "cpu"
    batch_size:       int        = 4
    apply_to_sensors: List[str]  = field(default_factory=list)  # empty = all
    seed:             Optional[int] = 42


# ---------------------------------------------------------------------------
# PhotorealisticEnhancer
# ---------------------------------------------------------------------------

class PhotorealisticEnhancer:
    """Enhance visual realism of rendered frames WITHOUT touching physics data.

    Parameters
    ----------
    config : EnhancerConfig

    Examples
    --------
    Stub mode (default, always works)::

        enhancer = PhotorealisticEnhancer()
        enhanced = enhancer.enhance(frames)   # identity pass

    With local diffusion (requires diffusers + GPU)::

        cfg = EnhancerConfig(backend="local_diffusion", strength=0.20,
                             device="cuda")
        enhancer = PhotorealisticEnhancer(cfg)
        enhanced = enhancer.enhance(frames)
    """

    def __init__(self, config: Optional[EnhancerConfig] = None) -> None:
        self.cfg      = config or EnhancerConfig()
        self._backend = self._load_backend()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(
        self,
        frames: Dict[str, np.ndarray],   # {sensor -> (T, H, W, 3) uint8}
    ) -> Dict[str, np.ndarray]:
        """Apply enhancement to all configured sensor channels.

        Physical field data is never passed to this method.

        Returns
        -------
        dict[sensor -> (T, H, W, 3) uint8]
            Enhanced frame sequences.
        """
        out = {}
        sensors_to_process = (
            self.cfg.apply_to_sensors
            if self.cfg.apply_to_sensors
            else list(frames.keys())
        )
        for sensor, seq in frames.items():
            if sensor in sensors_to_process:
                out[sensor] = self._backend(seq)
            else:
                out[sensor] = seq
        return out

    def is_stub(self) -> bool:
        return self.cfg.backend == "stub"

    # ------------------------------------------------------------------
    # Backend loader
    # ------------------------------------------------------------------

    def _load_backend(self) -> Callable:
        name = self.cfg.backend

        if name == "stub":
            return _stub_backend

        if name in _CUSTOM_BACKENDS:
            return lambda seq: _CUSTOM_BACKENDS[name](seq, self.cfg)

        if name == "local_diffusion":
            return self._build_diffusion_backend()

        if name == "cosmos":
            return self._build_cosmos_backend()

        warnings.warn(
            f"Unknown enhancer backend '{name}'. Falling back to stub.",
            UserWarning, stacklevel=2,
        )
        return _stub_backend

    def _build_diffusion_backend(self) -> Callable:
        """Build a local Stable Diffusion img2img backend."""
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            import torch
            from PIL import Image

            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.cfg.model_id,
                torch_dtype=torch.float16 if "cuda" in self.cfg.device else torch.float32,
            ).to(self.cfg.device)

            if self.cfg.seed is not None:
                generator = torch.Generator(device=self.cfg.device).manual_seed(self.cfg.seed)
            else:
                generator = None

            cfg = self.cfg

            def _diffuse(seq: np.ndarray) -> np.ndarray:
                T, H, W, C = seq.shape
                out = np.empty_like(seq)
                for start in range(0, T, cfg.batch_size):
                    batch = seq[start:start + cfg.batch_size]
                    pil_images = [Image.fromarray(f) for f in batch]
                    results = pipe(
                        prompt          = cfg.style_prompt,
                        negative_prompt = cfg.negative_prompt,
                        image           = pil_images,
                        strength        = cfg.strength,
                        guidance_scale  = 7.5,
                        generator       = generator,
                    ).images
                    for j, img in enumerate(results):
                        out[start + j] = np.array(img.resize((W, H)))
                return out

            return _diffuse

        except ImportError:
            warnings.warn(
                "diffusers not installed (pip install diffusers). "
                "Falling back to stub enhancer.",
                UserWarning, stacklevel=3,
            )
            return _stub_backend

    def _build_cosmos_backend(self) -> Callable:
        """Placeholder for NVIDIA Cosmos API integration."""
        warnings.warn(
            "NVIDIA Cosmos backend is not yet available.  "
            "Falling back to stub enhancer.  "
            "Subscribe to the Cosmos API preview at developer.nvidia.com/cosmos",
            UserWarning, stacklevel=3,
        )
        return _stub_backend


# ---------------------------------------------------------------------------
# Built-in backends
# ---------------------------------------------------------------------------

def _stub_backend(seq: np.ndarray) -> np.ndarray:
    """Identity pass — returns input unchanged."""
    return seq


def _sharpening_backend(seq: np.ndarray) -> np.ndarray:
    """Simple unsharp-mask sharpening (CPU, no deep learning)."""
    try:
        import scipy.ndimage as ndi
        out = np.empty_like(seq)
        for t in range(seq.shape[0]):
            f     = seq[t].astype(np.float32)
            blur  = ndi.gaussian_filter(f, sigma=1.0)
            sharp = f + 0.8 * (f - blur)
            out[t] = np.clip(sharp, 0, 255).astype(np.uint8)
        return out
    except ImportError:
        return seq


# Register the sharpening backend as a built-in custom option
register_enhancer("sharpening", lambda seq, cfg: _sharpening_backend(seq))
