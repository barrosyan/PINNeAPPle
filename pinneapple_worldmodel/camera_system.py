# -*- coding: utf-8 -*-
"""Stage 5 — Camera System.

Simulates real-world sensor observations from physics field data.  Supports
multi-camera arrays, virtual camera pose, field-of-view, and per-sensor
response models (RGB, thermal, depth, infrared).

The camera system is designed to work on 2-D simulation grids (x-y slice)
with a virtual viewpoint perpendicular to the plane.  For 3-D simulations it
projects an iso-surface slice to the image plane.

Public API
----------
  CameraConfig        — single camera specification
  MultiCameraArray    — array of cameras with shared physics field access
  CameraSystem        — main class: project fields → observations
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Single camera specification
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """Single virtual camera / sensor configuration.

    Parameters
    ----------
    name : str
        Unique sensor name (e.g. ``"front_rgb"``, ``"left_thermal"``).
    sensor_type : str
        One of ``"rgb"``, ``"thermal"``, ``"depth"``, ``"ir"``.
    position : (x, y, z)
        Camera position in world coordinates.
    look_at : (x, y, z)
        Point the camera is aimed at.
    fov_deg : float
        Horizontal field-of-view in degrees.
    resolution : (H, W)
        Output image resolution.
    fps : float
        Sensor frame rate.
    near_clip : float
        Near clipping plane (for depth rendering).
    far_clip : float
        Far clipping plane.
    noise_std : float
        Additive Gaussian noise standard deviation (in normalised [0,1] space).
    bit_depth : int
        Quantisation bit depth (8 or 16).
    """
    name:        str                   = "camera_0"
    sensor_type: str                   = "rgb"
    position:    Tuple[float, ...]     = (0.5, 0.5, 2.0)
    look_at:     Tuple[float, ...]     = (0.5, 0.5, 0.0)
    fov_deg:     float                 = 60.0
    resolution:  Tuple[int, int]       = (256, 256)
    fps:         float                 = 24.0
    near_clip:   float                 = 0.01
    far_clip:    float                 = 100.0
    noise_std:   float                 = 0.005
    bit_depth:   int                   = 8

    @property
    def max_val(self) -> int:
        return 2**self.bit_depth - 1


# ---------------------------------------------------------------------------
# Multi-camera array
# ---------------------------------------------------------------------------

@dataclass
class MultiCameraArray:
    """A set of cameras sharing the same simulation field.

    Parameters
    ----------
    cameras : list of CameraConfig
    sync_fps : float or None
        If given, all cameras are assumed to record at this frame rate.
    """
    cameras:  List[CameraConfig] = field(default_factory=list)
    sync_fps: Optional[float]    = None

    @classmethod
    def default(cls) -> "MultiCameraArray":
        """Standard front-view array with RGB, thermal, and depth cameras."""
        return cls(cameras=[
            CameraConfig(name="front_rgb",     sensor_type="rgb",
                         position=(0.5, 0.5, 2.0), fov_deg=60.0),
            CameraConfig(name="top_thermal",   sensor_type="thermal",
                         position=(0.5, 2.0, 0.5), fov_deg=70.0),
            CameraConfig(name="side_depth",    sensor_type="depth",
                         position=(2.0, 0.5, 0.5), fov_deg=50.0),
        ])

    @classmethod
    def from_config(cls, sensor_config: Dict[str, Any]) -> "MultiCameraArray":
        """Build from a ScenarioSpec.sensor_config dict."""
        position = tuple(sensor_config.get("camera_position", [0.5, 0.5, 2.0]))
        fov      = float(sensor_config.get("fov", 60.0))
        fps      = float(sensor_config.get("fps", 24))
        res      = tuple(sensor_config.get("resolution", [256, 256]))
        sensors  = sensor_config.get("sensors", ["rgb", "thermal", "depth"])

        cameras = []
        for i, s in enumerate(sensors):
            cameras.append(CameraConfig(
                name        = f"cam_{i}_{s}",
                sensor_type = s,
                position    = position,
                fov_deg     = fov,
                fps         = fps,
                resolution  = res,
            ))
        return cls(cameras=cameras, sync_fps=fps)


# ---------------------------------------------------------------------------
# CameraSystem
# ---------------------------------------------------------------------------

class CameraSystem:
    """Project physics fields to camera-plane observations.

    For 2-D simulations the projection is an orthographic top-down view with
    optional perspective distortion based on ``fov_deg``.  For 3-D fields a
    z-midplane slice is taken before projection.

    Parameters
    ----------
    array : MultiCameraArray
    domain_bounds : ((x0, x1), (y0, y1))
        Physical extent of the simulation domain.

    Examples
    --------
    ::

        system = CameraSystem(MultiCameraArray.default())
        obs    = system.observe(states, field_names, t_idx=5)
        # obs["front_rgb"] → ndarray (H, W, 3) uint8
    """

    def __init__(
        self,
        array:         Optional[MultiCameraArray] = None,
        domain_bounds: Optional[Tuple] = None,
    ) -> None:
        self.array         = array or MultiCameraArray.default()
        self.domain_bounds = domain_bounds or ((0.0, 1.0), (0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        states:      np.ndarray,    # (T, C, Ny, Nx) or (C, Ny, Nx) for a single step
        field_names: List[str],
        t_idx:       Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Project physics fields through each camera.

        Returns
        -------
        dict[camera_name -> ndarray (H, W, 3)]
            uint8 observations in the camera's sensor modality.
        """
        if states.ndim == 3:
            snap = states                      # already single timestep
        elif t_idx is not None:
            snap = states[t_idx]
        else:
            snap = states[-1]                  # last timestep

        obs = {}
        for cam in self.array.cameras:
            field_2d = self._select_field(snap, field_names, cam.sensor_type)
            proj     = self._project(field_2d, cam)
            obs[cam.name] = proj

        return obs

    def observe_sequence(
        self,
        states:      np.ndarray,    # (T, C, Ny, Nx)
        field_names: List[str],
    ) -> Dict[str, np.ndarray]:
        """Observe all timesteps for all cameras.

        Returns
        -------
        dict[camera_name -> ndarray (T, H, W, 3)]
        """
        T = states.shape[0]
        first_obs = self.observe(states, field_names, t_idx=0)
        seq: Dict[str, List[np.ndarray]] = {k: [] for k in first_obs}

        for t in range(T):
            obs = self.observe(states, field_names, t_idx=t)
            for k, v in obs.items():
                seq[k].append(v)

        return {k: np.stack(v, axis=0) for k, v in seq.items()}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_field(
        self,
        snap:        np.ndarray,    # (C, Ny, Nx)
        field_names: List[str],
        sensor_type: str,
    ) -> np.ndarray:                 # (Ny, Nx)
        """Select / compute the scalar field for this sensor modality."""
        c_to_try = {
            "rgb":     [["u","v","velocity_x","velocity_y"], ["u"], ["p"]],
            "thermal": [["T", "temperature", "temp", "theta"], ["p"], ["u"]],
            "depth":   [["p", "pressure"], ["T"], ["u"]],
            "ir":      [["T", "temperature"], ["p"], ["u"]],
        }.get(sensor_type, [["u"]])

        for candidates in c_to_try:
            idx = _find_channel(field_names, candidates)
            if idx is not None:
                f = snap[idx]
                break
        else:
            f = snap[0]

        # Velocity magnitude for RGB / depth when two channels available
        if sensor_type in ("rgb", "depth") and snap.shape[0] >= 2:
            u_idx = _find_channel(field_names, ["u", "ux", "velocity_x"])
            v_idx = _find_channel(field_names, ["v", "uy", "velocity_y"])
            if u_idx is not None and v_idx is not None:
                f = np.sqrt(snap[u_idx]**2 + snap[v_idx]**2)

        return f

    def _project(self, field_2d: np.ndarray, cam: CameraConfig) -> np.ndarray:
        """Project a 2-D field slice into camera-plane image coordinates.

        Applies:
          1. Perspective warp (simple homography approximation)
          2. Colourmap application
          3. Sensor-specific post-processing
          4. Quantisation to the camera bit depth
        """
        H, W   = cam.resolution
        Ny, Nx = field_2d.shape

        # 1. Normalise field to [0, 1]
        f_n = _norm_01(field_2d)

        # 2. Perspective warp: compute view frustum crop
        f_warped = self._warp_to_camera(f_n, cam, Ny, Nx)

        # 3. Resize to camera resolution
        f_resized = _resize_nearest(f_warped, H, W)

        # 4. Apply sensor response model
        img = self._apply_sensor_model(f_resized, cam)

        return img

    def _warp_to_camera(
        self,
        field_n: np.ndarray,    # (Ny, Nx) float in [0,1]
        cam:     CameraConfig,
        Ny: int, Nx: int,
    ) -> np.ndarray:
        """Approximate perspective crop based on camera position and FoV."""
        # For a top-down (z-axis) camera, compute the visible footprint
        pos_z  = cam.position[2] if len(cam.position) > 2 else 2.0
        half_w = math.tan(math.radians(cam.fov_deg / 2)) * pos_z
        cx     = cam.look_at[0] if len(cam.look_at) > 0 else 0.5
        cy     = cam.look_at[1] if len(cam.look_at) > 1 else 0.5

        (x0, x1), (y0, y1) = self.domain_bounds
        Lx = x1 - x0
        Ly = y1 - y0

        # Clamp visible extent to domain
        vx0 = max(0.0, (cx - half_w - x0) / Lx)
        vx1 = min(1.0, (cx + half_w - x0) / Lx)
        vy0 = max(0.0, (cy - half_w - y0) / Ly)
        vy1 = min(1.0, (cy + half_w - y0) / Ly)

        ix0 = int(vx0 * Nx);  ix1 = max(ix0 + 1, int(vx1 * Nx))
        iy0 = int(vy0 * Ny);  iy1 = max(iy0 + 1, int(vy1 * Ny))
        return field_n[iy0:iy1, ix0:ix1]

    def _apply_sensor_model(
        self,
        field_n: np.ndarray,   # (H, W) float in [0, 1]
        cam:     CameraConfig,
    ) -> np.ndarray:            # (H, W, 3) uint8
        """Apply sensor-specific colour model and quantise."""
        H, W = field_n.shape

        if cam.sensor_type == "thermal":
            # Thermal: orange-red heat-signature palette
            r = np.clip(field_n * 2,       0, 1)
            g = np.clip(field_n * 2 - 0.5, 0, 1) * 0.6
            b = np.zeros_like(field_n)
            rgb = np.stack([r, g, b], axis=-1)
        elif cam.sensor_type == "depth":
            # Depth: cooler = farther
            inv = 1.0 - field_n
            rgb = np.stack([inv, inv, inv], axis=-1)
        elif cam.sensor_type == "ir":
            r = field_n
            g = field_n * 0.3
            b = np.zeros_like(field_n)
            rgb = np.stack([r, g, b], axis=-1)
        else:
            # RGB: viridis-like (blue → teal → yellow)
            r = np.clip(field_n * 1.5 - 0.5, 0, 1)
            g = np.clip(field_n * 1.2,        0, 1)
            b = np.clip(1.0 - field_n * 1.3,  0, 1)
            rgb = np.stack([r, g, b], axis=-1)

        # Add sensor noise
        if cam.noise_std > 0:
            rgb = np.clip(rgb + np.random.normal(0, cam.noise_std, rgb.shape), 0, 1)

        # Quantise
        return (rgb * cam.max_val).astype(np.uint8)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise camera configuration to a JSON-compatible dict."""
        return {
            "cameras": [
                {
                    "name":        c.name,
                    "sensor_type": c.sensor_type,
                    "position":    list(c.position),
                    "look_at":     list(c.look_at),
                    "fov_deg":     c.fov_deg,
                    "resolution":  list(c.resolution),
                    "fps":         c.fps,
                    "noise_std":   c.noise_std,
                    "bit_depth":   c.bit_depth,
                }
                for c in self.array.cameras
            ],
            "sync_fps": self.array.sync_fps,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _find_channel(names: List[str], candidates: List[str]) -> Optional[int]:
    for cand in candidates:
        for i, nm in enumerate(names):
            if nm.lower() == cand.lower():
                return i
    return None


def _norm_01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _resize_nearest(field: np.ndarray, H: int, W: int) -> np.ndarray:
    Ny, Nx = field.shape
    if Ny == H and Nx == W:
        return field
    iy = (np.arange(H) * Ny / H).astype(int).clip(0, Ny - 1)
    ix = (np.arange(W) * Nx / W).astype(int).clip(0, Nx - 1)
    return field[np.ix_(iy, ix)]
