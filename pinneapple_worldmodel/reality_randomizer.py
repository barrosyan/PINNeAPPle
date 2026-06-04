# -*- coding: utf-8 -*-
"""Stage 6 — Reality Randomizer.

Reduces the sim-to-real gap by adding controlled variability to the rendered
visual observations while leaving physical field arrays untouched.

Randomizations applied
----------------------
  sensor_noise      Gaussian + Poisson photon noise
  jpeg_artifacts    JPEG compression + decompress at random quality 20-80
  lighting          Brightness, contrast, colour temperature perturbation
  shadows           Random soft-edge shadow overlays
  occlusions        Random rectangular occlusion patches
  blur              Lens / motion blur (Gaussian kernel)
  distortion        Barrel / pincushion lens distortion
  vignette          Radial light fall-off at image edges

Public API
----------
  RandomizerConfig  — what to apply and with what probability / strength
  RealityRandomizer — applies randomizations to image tensors
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RandomizerConfig:
    """Controls which augmentations are applied and their strength.

    All ``p_*`` parameters are per-sample application probabilities [0, 1].
    """
    seed:             Optional[int] = None

    # Sensor noise
    p_sensor_noise:   float = 0.90
    gaussian_std:     float = 0.01    # fraction of [0,1] dynamic range
    poisson_scale:    float = 0.005

    # JPEG compression
    p_jpeg:           float = 0.50
    jpeg_quality_lo:  int   = 25
    jpeg_quality_hi:  int   = 80

    # Lighting
    p_lighting:       float = 0.80
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range:   Tuple[float, float] = (0.8, 1.2)
    color_temp_std:   float = 0.05   # per-channel multiplicative noise

    # Shadow overlay
    p_shadow:         float = 0.40
    shadow_alpha:     float = 0.35   # max opacity of shadow

    # Occlusions
    p_occlusion:      float = 0.30
    n_occ_boxes:      int   = 3      # max occlusion rectangles
    occ_size_range:   Tuple[float, float] = (0.05, 0.20)  # fraction of image side

    # Gaussian blur
    p_blur:           float = 0.30
    blur_kernel_max:  int   = 5      # max kernel radius in pixels

    # Lens distortion
    p_distortion:     float = 0.20
    distortion_k:     float = 0.05   # barrel/pincushion coefficient

    # Vignette
    p_vignette:       float = 0.50
    vignette_strength: float = 0.4   # max darkening at corners


# ---------------------------------------------------------------------------
# RealityRandomizer
# ---------------------------------------------------------------------------

class RealityRandomizer:
    """Apply reality-gap augmentations to rendered image sequences.

    CRITICAL: The randomizer NEVER modifies physical field arrays (velocity,
    pressure, temperature, concentration).  It operates exclusively on uint8
    image data returned by PhysicsRenderer / CameraSystem.

    Parameters
    ----------
    config : RandomizerConfig

    Examples
    --------
    ::

        randomizer = RealityRandomizer(RandomizerConfig(seed=42))
        # frames: dict[sensor_name -> (T, H, W, 3) uint8]
        aug_frames = randomizer.augment(frames)
    """

    def __init__(self, config: Optional[RandomizerConfig] = None) -> None:
        self.cfg = config or RandomizerConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def augment(
        self,
        frames: Dict[str, np.ndarray],   # {sensor -> (T, H, W, 3) uint8}
    ) -> Dict[str, np.ndarray]:
        """Apply all configured augmentations to a dict of frame sequences.

        Returns a new dict with augmented frames.  Physical field data is
        never passed to this method — it operates only on image tensors.
        """
        out = {}
        for sensor, seq in frames.items():
            out[sensor] = self._augment_sequence(seq, sensor)
        return out

    def augment_single(self, frame: np.ndarray) -> np.ndarray:
        """Augment a single (H, W, 3) uint8 frame."""
        return self._augment_frame(frame.astype(np.float32) / 255.0)

    # ------------------------------------------------------------------
    # Sequence augmentation
    # ------------------------------------------------------------------

    def _augment_sequence(self, seq: np.ndarray, sensor: str) -> np.ndarray:
        """Augment (T, H, W, 3) — sample augmentation params once per clip."""
        T, H, W, C = seq.shape
        out = np.empty_like(seq)

        # Sample once per clip (temporal consistency)
        params = self._sample_params(H, W)

        for t in range(T):
            f = seq[t].astype(np.float32) / 255.0
            f = self._apply_params(f, params, sensor)
            out[t] = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        return out

    def _augment_frame(self, f: np.ndarray) -> np.ndarray:
        """Augment a (H, W, 3) float32 frame in [0, 1]."""
        H, W, _ = f.shape
        params   = self._sample_params(H, W)
        return np.clip(self._apply_params(f, params), 0, 1)

    # ------------------------------------------------------------------
    # Parameter sampling & application
    # ------------------------------------------------------------------

    def _sample_params(self, H: int, W: int) -> Dict:
        r = self.rng
        return {
            "do_noise":     r.random() < self.cfg.p_sensor_noise,
            "do_jpeg":      r.random() < self.cfg.p_jpeg,
            "do_light":     r.random() < self.cfg.p_lighting,
            "do_shadow":    r.random() < self.cfg.p_shadow,
            "do_occ":       r.random() < self.cfg.p_occlusion,
            "do_blur":      r.random() < self.cfg.p_blur,
            "do_distort":   r.random() < self.cfg.p_distortion,
            "do_vignette":  r.random() < self.cfg.p_vignette,
            # Lighting params
            "brightness":   r.uniform(*self.cfg.brightness_range),
            "contrast":     r.uniform(*self.cfg.contrast_range),
            "color_temp":   r.normal(1.0, self.cfg.color_temp_std, 3).clip(0.7, 1.3),
            # JPEG quality
            "jpeg_q":       int(r.integers(self.cfg.jpeg_quality_lo, self.cfg.jpeg_quality_hi + 1)),
            # Shadow
            "shadow_poly":  _random_shadow_polygon(H, W, r),
            "shadow_alpha": r.uniform(0, self.cfg.shadow_alpha),
            # Occlusions
            "occ_boxes":    _random_occ_boxes(H, W, r, self.cfg.n_occ_boxes, self.cfg.occ_size_range),
            # Blur
            "blur_k":       int(r.integers(1, self.cfg.blur_kernel_max + 1)),
            # Distortion
            "dist_k":       r.uniform(-self.cfg.distortion_k, self.cfg.distortion_k),
            # Vignette map
            "vignette":     _vignette_map(H, W, r.uniform(0, self.cfg.vignette_strength)),
        }

    def _apply_params(
        self,
        f:      np.ndarray,   # (H, W, 3) float in [0, 1]
        params: Dict,
        sensor: str = "rgb",
    ) -> np.ndarray:
        # 1. Lighting
        if params["do_light"]:
            f = f * params["brightness"]
            mean = f.mean()
            f    = (f - mean) * params["contrast"] + mean
            f   *= params["color_temp"].reshape(1, 1, 3)

        # 2. Sensor noise
        if params["do_noise"]:
            gauss  = np.random.normal(0, self.cfg.gaussian_std, f.shape)
            lam    = np.clip(f, 0, None) / max(self.cfg.poisson_scale, 1e-6)
            poiss  = np.random.poisson(lam)
            f      = f + gauss + poiss * self.cfg.poisson_scale

        # 3. Shadow
        if params["do_shadow"]:
            mask = params["shadow_poly"]   # (H, W) bool
            f[mask] *= (1.0 - params["shadow_alpha"])

        # 4. Occlusions
        if params["do_occ"]:
            for y0, x0, y1, x1 in params["occ_boxes"]:
                f[y0:y1, x0:x1] = 0.1 + 0.05 * np.random.random()

        # 5. Blur
        if params["do_blur"]:
            k = max(1, params["blur_k"])
            f = _gaussian_blur(f, k)

        # 6. Lens distortion
        if params["do_distort"]:
            f = _barrel_distort(f, params["dist_k"])

        # 7. Vignette
        if params["do_vignette"]:
            f = f * params["vignette"][:, :, None]

        # 8. JPEG artifacts (applied last, like a real capture pipeline)
        if params["do_jpeg"]:
            f = _jpeg_round_trip(f, params["jpeg_q"])

        return np.clip(f, 0, 1)


# ---------------------------------------------------------------------------
# Augmentation primitives
# ---------------------------------------------------------------------------

def _random_shadow_polygon(H: int, W: int, rng) -> np.ndarray:
    """Binary mask for a random half-plane shadow (like a building / pole shadow)."""
    mask = np.zeros((H, W), dtype=bool)
    x0 = rng.integers(0, W)
    y0 = rng.integers(0, H)
    x1 = rng.integers(0, W)
    y1 = rng.integers(0, H)
    for y in range(H):
        for x in range(W):
            # Point on the left side of directed line (x0,y0)→(x1,y1)
            cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
            if cross > 0:
                mask[y, x] = True
    return mask


def _random_occ_boxes(H, W, rng, n_boxes: int, size_range) -> List:
    boxes = []
    for _ in range(rng.integers(1, n_boxes + 1)):
        lo, hi = size_range
        bh = int(rng.uniform(lo, hi) * H)
        bw = int(rng.uniform(lo, hi) * W)
        y0 = int(rng.integers(0, H - bh + 1))
        x0 = int(rng.integers(0, W - bw + 1))
        boxes.append((y0, x0, y0 + bh, x0 + bw))
    return boxes


def _vignette_map(H: int, W: int, strength: float) -> np.ndarray:
    ys = np.linspace(-1, 1, H)
    xs = np.linspace(-1, 1, W)
    XX, YY = np.meshgrid(xs, ys)
    r = np.sqrt(XX**2 + YY**2)
    return 1.0 - strength * r**2


def _gaussian_blur(f: np.ndarray, radius: int) -> np.ndarray:
    """Apply separable Gaussian blur via convolution."""
    k = 2 * radius + 1
    x = np.linspace(-radius, radius, k)
    kernel_1d = np.exp(-0.5 * (x / (radius / 2 + 0.1))**2)
    kernel_1d /= kernel_1d.sum()

    out = np.empty_like(f)
    for c in range(f.shape[2]):
        ch = f[:, :, c]
        # Horizontal pass
        for row in range(ch.shape[0]):
            out[row, :, c] = np.convolve(ch[row], kernel_1d, mode="same")
        tmp = out[:, :, c].copy()
        # Vertical pass
        for col in range(ch.shape[1]):
            out[:, col, c] = np.convolve(tmp[:, col], kernel_1d, mode="same")
    return out


def _barrel_distort(f: np.ndarray, k: float) -> np.ndarray:
    """Apply barrel (k<0) or pincushion (k>0) lens distortion."""
    H, W = f.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    xs = (np.arange(W) - cx) / cx
    ys = (np.arange(H) - cy) / cy
    XX, YY = np.meshgrid(xs, ys)
    R2 = XX**2 + YY**2
    factor = 1.0 + k * R2
    Xd = (XX * factor * cx + cx).clip(0, W - 1).astype(int)
    Yd = (YY * factor * cy + cy).clip(0, H - 1).astype(int)
    return f[Yd, Xd]


def _jpeg_round_trip(f: np.ndarray, quality: int) -> np.ndarray:
    """Encode to JPEG and decode back (simulate compression artifacts)."""
    try:
        import imageio
        import io
        img_u8 = (f * 255).clip(0, 255).astype(np.uint8)
        buf = io.BytesIO()
        imageio.v2.imwrite(buf, img_u8, format="JPEG", quality=quality)
        buf.seek(0)
        decoded = imageio.v2.imread(buf, format="JPEG")
        return decoded.astype(np.float32) / 255.0
    except Exception:
        return f   # fallback: no artifact if imageio not available
