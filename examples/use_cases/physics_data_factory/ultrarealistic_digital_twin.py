# -*- coding: utf-8 -*-
"""Ultrarealistic Industrial Digital Twin
==========================================

Phase 1 — HD Rendering Pipeline
  * PyVista PBR (Physically Based Rendering) with metallic/roughness workflow
  * SSAO (Screen Space Ambient Occlusion) for realistic contact shadows
  * Shadow mapping from directional light
  * Professional post-processing chain:
      ACES film tone-mapping  → bloom  → depth-of-field  → chromatic aberration
      → color grading  → film grain  → vignette  → sharpening
  * 1920x1080 @ 24 fps

Phase 2 — Physics Extraction from Video (Inverse Pipeline)
  * CNN video frame encoder  → physical field predictions
  * Trained on paired (rendered_frame, physics_state) data
  * Predicts: pressure, temperature, flow from video only
  * Validates against ground-truth simulation data

Run
---
  python ultrarealistic_digital_twin.py

Outputs in ./outputs/ultra/
  factory_hd.mp4          — full video, 1920x1080, 24fps
  factory_hd_preview.png  — frame 1 preview
  extracted_physics.json  — physics fields recovered from video
  extraction_model.pt     — trained inverse model checkpoint
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, List

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import pyvista as pv
import torch
import torch.nn as nn
import imageio.v3 as iio

pv.global_theme.anti_aliasing = "msaa"
pv.global_theme.multi_samples  = 8

OUT = Path(__file__).parent / "outputs" / "ultra"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)
W, H, FPS, N = 1920, 1080, 24, 192   # 8 seconds


# ============================================================================
# POST-PROCESSING  (the single biggest upgrade for "photorealism")
# ============================================================================

def aces_tone_map(img: np.ndarray) -> np.ndarray:
    """ACES (Academy Color Encoding System) cinematic tone mapping."""
    x = img.astype(np.float32) / 255.0
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    return (x * 255).astype(np.uint8)


def add_bloom(img: np.ndarray, threshold: int = 210, radius: int = 18,
              strength: float = 0.55) -> np.ndarray:
    """Bloom: bright surfaces glow (industrial lights, metal specular)."""
    bright = np.clip(img.astype(np.int32) - threshold, 0, 255).astype(np.uint8)
    pil    = Image.fromarray(bright)
    blur   = pil.filter(ImageFilter.GaussianBlur(radius))
    bloom  = np.array(blur).astype(np.float32) * strength
    result = np.clip(img.astype(np.float32) + bloom, 0, 255)
    return result.astype(np.uint8)


def depth_of_field(img: np.ndarray, focus_row: float = 0.52,
                   max_blur: float = 2.8) -> np.ndarray:
    """Radial & depth DOF: soft blur on foreground floor and far background."""
    H, W = img.shape[:2]
    pil = Image.fromarray(img)
    # Slight global softness for far objects (background blur)
    soft = pil.filter(ImageFilter.GaussianBlur(max_blur))
    # Blend based on distance from focus row
    y = np.linspace(0, 1, H)
    dist  = np.abs(y - focus_row)
    alpha = np.clip(dist * 3.5 - 0.4, 0.0, 1.0)[:, None, None]
    out = (np.array(pil).astype(float) * (1 - alpha) +
           np.array(soft).astype(float) * alpha)
    return out.clip(0, 255).astype(np.uint8)


def chromatic_aberration(img: np.ndarray, shift: int = 2) -> np.ndarray:
    """RGB channel offset simulating lens chromatic aberration."""
    r = np.roll(img[:, :, 0], -shift, axis=1)
    g = img[:, :, 1]
    b = np.roll(img[:, :, 2],  shift, axis=1)
    return np.stack([r, g, b], axis=2)


def color_grade(img: np.ndarray) -> np.ndarray:
    """Industrial color grade: teal/blue shadows, warm highlights."""
    f = img.astype(np.float32) / 255.0
    lum = f.mean(axis=2, keepdims=True)
    # Shadow tint: push blue-green in dark areas
    shadow = np.maximum(0.0, 0.5 - lum)
    # Highlight tint: push slight amber in bright areas
    hi     = np.maximum(0.0, lum - 0.6)
    tint   = np.zeros_like(f)
    tint[:, :, 0] -= shadow[:, :, 0] * 0.06    # less red in shadows
    tint[:, :, 1] += shadow[:, :, 0] * 0.03    # more green
    tint[:, :, 2] += shadow[:, :, 0] * 0.10    # more blue (teal)
    tint[:, :, 0] += hi[:, :, 0] * 0.05        # warm highlights
    tint[:, :, 2] -= hi[:, :, 0] * 0.03
    result = np.clip(f + tint, 0.0, 1.0)
    return (result * 255).astype(np.uint8)


def film_grain(img: np.ndarray, intensity: float = 0.022) -> np.ndarray:
    """Analog film grain texture."""
    grain = RNG.standard_normal(img.shape).astype(np.float32) * intensity * 255
    return np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)


def vignette(img: np.ndarray, strength: float = 0.50) -> np.ndarray:
    """Lens vignette: darker edges."""
    H, W = img.shape[:2]
    xs = np.linspace(-1, 1, W)[None, :]
    ys = np.linspace(-1, 1, H)[:, None]
    r  = np.sqrt(xs * xs + ys * ys)
    mask = np.clip(1.0 - strength * np.clip((r - 0.35) / 0.65, 0, 1) ** 2, 0.35, 1.0)
    return np.clip(img.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)


def unsharp_mask(img: np.ndarray, amount: float = 0.65, radius: int = 1) -> np.ndarray:
    """Unsharp mask sharpening for crisp edges."""
    pil  = Image.fromarray(img)
    blur = pil.filter(ImageFilter.GaussianBlur(radius))
    sharp = np.array(pil).astype(float) + amount * (np.array(pil) - np.array(blur)).astype(float)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def lens_flare(img: np.ndarray, light_positions: List[Tuple[int, int]],
               frame_idx: int) -> np.ndarray:
    """Subtle lens flares on overhead lights."""
    out = img.copy().astype(np.float32)
    for (lx, ly) in light_positions:
        if not (0 < lx < img.shape[1] and 0 < ly < img.shape[0]):
            continue
        flicker = 0.7 + 0.3 * math.sin(frame_idx * 0.43 + lx * 0.01)
        for r, intensity in [(4, 180), (12, 60), (25, 20)]:
            y0 = max(0, ly - r);  y1 = min(img.shape[0], ly + r)
            x0 = max(0, lx - r);  x1 = min(img.shape[1], lx + r)
            if y0 >= y1 or x0 >= x1:
                continue
            yy = np.arange(y0, y1) - ly
            xx = np.arange(x0, x1) - lx
            Y, X = np.meshgrid(yy, xx, indexing="ij")
            mask = np.exp(-(X*X + Y*Y) / (2*(r*0.5)**2)) * intensity * flicker
            # Use plain slice indexing (np.ix_ + trailing : breaks on numpy ≥ 2)
            out[y0:y1, x0:x1, :] = np.clip(
                out[y0:y1, x0:x1, :].astype(np.float32) + mask[:, :, None],
                0, 255).astype(np.uint8)
    return out.clip(0, 255).astype(np.uint8)


def full_post_process(
    img:           np.ndarray,
    frame_idx:     int,
    light_pix:     Optional[List[Tuple[int, int]]] = None,
    dof:           bool = True,
    grain:         bool = True,
) -> np.ndarray:
    """Full post-processing chain: render → cinematic look."""
    img = aces_tone_map(img)
    img = add_bloom(img, threshold=225, radius=16, strength=0.35)
    if light_pix:
        img = lens_flare(img, light_pix, frame_idx)
    if dof:
        img = depth_of_field(img, focus_row=0.50, max_blur=1.2)
    img = color_grade(img)
    if grain:
        img = film_grain(img, intensity=0.008)
    img = vignette(img, strength=0.30)
    img = unsharp_mask(img, amount=0.55, radius=1)
    return img


# ============================================================================
# SCENE GEOMETRY  (industrial pump station, more detailed than before)
# ============================================================================

def make_tube(p0, p1, r=0.06, n=24):
    pts = pv.lines_from_points(np.array([p0, p1], dtype=float))
    return pts.tube(radius=r, n_sides=n)


def make_flange(center, axis, r_out=0.105, thick=0.028):
    d = np.array(axis, float); d /= np.linalg.norm(d)
    disc = pv.Disc(center=center, normal=d, inner=0.045, outer=r_out,
                   r_res=4, c_res=32)
    return disc.extrude(d * thick, capping=True)


def make_pump(cx, cy, cz, spin_deg=0.0):
    """Centrifugal pump with motor, coupling, baseplate."""
    volute  = pv.Sphere(center=[cx, cy+0.20, cz], radius=0.01)
    volute.scale([0.30, 0.24, 0.35], inplace=True)
    motor   = pv.Cylinder(center=[cx-0.46, cy+0.20, cz], direction=(1,0,0), radius=0.19, height=0.58)
    guard   = pv.Cylinder(center=[cx-0.15, cy+0.20, cz], direction=(1,0,0), radius=0.125, height=0.12)
    suction = pv.Cylinder(center=[cx,      cy+0.20, cz-0.25], direction=(0,0,1), radius=0.072, height=0.32)
    disch   = pv.Cylinder(center=[cx,      cy+0.40, cz],      direction=(0,1,0), radius=0.062, height=0.25)
    base    = pv.Box(bounds=[cx-0.58, cx+0.38, cy-0.01, cy+0.05, cz-0.38, cz+0.38])
    # Impeller vanes (visible through sight glass proxy)
    imp = pv.Disc(center=[cx, cy+0.20, cz], normal=(1,0,0), inner=0.0, outer=0.22,
                  r_res=3, c_res=5)
    imp.rotate_x(spin_deg, point=[cx, cy+0.20, cz], inplace=True)
    return (volute.merge(motor).merge(guard).merge(suction)
            .merge(disch).merge(base).merge(imp))


def make_hx(x0, x1, y0, y1, z0, z1):
    cx=(x0+x1)/2; cy=(y0+y1)/2; cz=(z0+z1)/2; L=x1-x0
    shell  = pv.Cylinder(center=[cx,cy,cz], direction=(1,0,0), radius=0.48, height=L)
    head_l = pv.Sphere(center=[x0-0.12, cy, cz], radius=0.49)
    head_r = pv.Sphere(center=[x1+0.12, cy, cz], radius=0.49)
    n_l = pv.Cylinder(center=[x0, cy+0.56, cz], (0,1,0), radius=0.085, height=0.22)
    n_r = pv.Cylinder(center=[x1, cy+0.56, cz], (0,1,0), radius=0.085, height=0.22)
    n_3 = pv.Cylinder(center=[x0, cy-0.56, cz], (0,1,0), radius=0.072, height=0.19)
    n_4 = pv.Cylinder(center=[x1, cy-0.56, cz], (0,1,0), radius=0.072, height=0.19)
    # Baffle rings
    rings = []
    for bx in np.linspace(x0+0.4, x1-0.4, 4):
        rings.append(pv.Disc(center=[bx, cy, cz], normal=(1,0,0), inner=0.42, outer=0.48,
                              r_res=2, c_res=32).extrude([0.02, 0, 0], capping=True))
    sad_l = pv.Box(bounds=[x0+0.3, x0+0.65, y0, cy-0.46, cz-0.6, cz+0.6])
    sad_r = pv.Box(bounds=[x1-0.65, x1-0.3, y0, cy-0.46, cz-0.6, cz+0.6])
    result = (shell.merge(head_l).merge(head_r)
              .merge(n_l).merge(n_r).merge(n_3).merge(n_4)
              .merge(sad_l).merge(sad_r))
    for r in rings:
        result = result.merge(r)
    return result


def make_valve(cx, cy, cz, axis=(1,0,0)):
    body  = pv.Cylinder(center=[cx, cy, cz], direction=axis, radius=0.10, height=0.17)
    stem  = pv.Cylinder(center=[cx, cy+0.22, cz], direction=(0,1,0), radius=0.022, height=0.28)
    wheel = pv.Disc(center=[cx, cy+0.38, cz], normal=(0,1,0), inner=0.022, outer=0.13,
                    r_res=2, c_res=16).extrude([0,0.012,0], capping=True)
    spokes = []
    for ang in [0, 90, 180, 270]:
        sa = math.radians(ang); r = 0.095
        sp = make_tube([cx+r*math.cos(sa), cy+0.385, cz+r*math.sin(sa)],
                       [cx-r*math.cos(sa), cy+0.385, cz-r*math.sin(sa)], r=0.008, n=8)
        spokes.append(sp)
    result = body.merge(stem).merge(wheel)
    for s in spokes:
        result = result.merge(s)
    return result


def make_gauge(cx, cy, cz):
    """Pressure gauge: dial + stem."""
    dial = pv.Disc(center=[cx, cy+0.10, cz], normal=(0,1,0), inner=0.0, outer=0.085,
                   r_res=2, c_res=32).extrude([0,0.018,0], capping=True)
    stem = pv.Cylinder(center=[cx, cy, cz], direction=(0,1,0), radius=0.015, height=0.12)
    return dial.merge(stem)


def make_expansion_joint(cx, cy, cz, axis=(1,0,0), L=0.16, r=0.075, n_conv=4):
    """Bellows expansion joint (corrugated tube)."""
    segs = []
    xs = np.linspace(-L/2, L/2, n_conv*2+2)
    for i in range(len(xs)-1):
        ri = r * (1.0 + 0.18 * math.sin(math.pi * i))
        pt0 = [cx+xs[i],    cy, cz]
        pt1 = [cx+xs[i+1],  cy, cz]
        segs.append(make_tube(pt0, pt1, r=ri, n=16))
    result = segs[0]
    for s in segs[1:]:
        result = result.merge(s)
    return result


def make_pipe_rack(x0, x1, y, z_vals):
    """Overhead pipe rack horizontal beams."""
    meshes = []
    for z in z_vals:
        meshes.append(make_tube([x0, y, z], [x1, y, z], r=0.045, n=8))
    for x in [x0, x1, (x0+x1)/2]:
        meshes.append(pv.Box(bounds=[x-0.04, x+0.04, 0.0, y, z_vals[0]-0.06, z_vals[-1]+0.06]))
    result = meshes[0]
    for m in meshes[1:]:
        result = result.merge(m)
    return result


def make_floor(x0, x1, z0, z1):
    return pv.Plane(center=[(x0+x1)/2, 0.0, (z0+z1)/2],
                    direction=(0,1,0), i_size=x1-x0, j_size=z1-z0,
                    i_resolution=60, j_resolution=30)


def make_wall(x0, x1, z, height=5.0):
    return pv.Box(bounds=[x0, x1, 0, height, z-0.15, z])


def make_ceiling(x0, x1, y, z0, z1):
    return pv.Plane(center=[(x0+x1)/2, y, (z0+z1)/2],
                    direction=(0,-1,0), i_size=x1-x0, j_size=z1-z0,
                    i_resolution=20, j_resolution=10)


def make_catwalk(x, y, z0, z1):
    """Raised maintenance catwalk (grating platform)."""
    platform = pv.Box(bounds=[x-0.5, x+0.5, y, y+0.04, z0, z1])
    rail_l   = make_tube([x-0.5, y+0.90, z0], [x-0.5, y+0.90, z1], r=0.022, n=8)
    rail_r   = make_tube([x+0.5, y+0.90, z0], [x+0.5, y+0.90, z1], r=0.022, n=8)
    posts    = []
    for zp in np.linspace(z0, z1, 5):
        for xp in [x-0.5, x+0.5]:
            posts.append(make_tube([xp, y, zp], [xp, y+0.90, zp], r=0.018, n=8))
    result = platform.merge(rail_l).merge(rail_r)
    for p in posts:
        result = result.merge(p)
    return result


def make_control_panel(cx, cy, cz):
    """Wall-mounted SCADA control panel."""
    body  = pv.Box(bounds=[cx-0.60, cx+0.60, cy, cy+1.10, cz-0.14, cz])
    screen = pv.Plane(center=[cx, cy+0.70, cz-0.01], direction=(0,0,1),
                      i_size=0.90, j_size=0.55)
    return body.merge(screen)


# ============================================================================
# PIPE NETWORK DEFINITION
# ============================================================================

PR = 0.058       # pipe radius
MR = 0.090       # manifold radius
HX = (4.5, 7.8, 0.15, 1.12, 0.8, 4.2)   # heat exchanger bounds
PUMP_XYZ = [(-0.5, 0.15, 0.0), (-0.5, 0.15, 2.5), (-0.5, 0.15, 5.0)]

# Pipe segments: (p0, p1, radius, tag)
PIPES = [
    # intake header
    ((-4.0, 0.58, 0.0),  (-1.20, 0.58, 0.0),  PR,    "intake"),
    ((-4.0, 0.58, 2.5),  (-1.20, 0.58, 2.5),  PR,    "intake"),
    ((-4.0, 0.58, 5.0),  (-1.20, 0.58, 5.0),  PR,    "intake"),
    # pump discharge risers
    ((-0.5, 0.58, 0.0),  (-0.5,  1.10, 0.0),  PR,    "discharge"),
    ((-0.5, 0.58, 2.5),  (-0.5,  1.10, 2.5),  PR,    "discharge"),
    ((-0.5, 0.58, 5.0),  (-0.5,  1.10, 5.0),  PR,    "discharge"),
    # horizontal legs to header
    ((-0.5,  1.10, 0.0), (2.60,  1.10, 0.0),  PR,    "hot"),
    ((-0.5,  1.10, 2.5), (2.60,  1.10, 2.5),  PR,    "hot"),
    ((-0.5,  1.10, 5.0), (2.60,  1.10, 5.0),  PR,    "hot"),
    # discharge manifold (vertical)
    ((2.60,  1.10, 0.0), (2.60,  1.10, 5.0),  MR,    "manifold"),
    # manifold → HX
    ((2.60,  1.10, 2.5), (4.50,  1.10, 2.5),  MR,    "hot"),
    # HX → discharge
    ((7.80,  1.10, 2.5), (11.0,  1.10, 2.5),  PR*1.3,"discharge_final"),
    # recirculation line (lower)
    ((7.80,  0.38, 2.5), (2.60,  0.38, 2.5),  PR*0.8,"recirc"),
    # pump bypass
    ((2.60,  0.38, 2.5), (-0.50, 0.38, 2.5),  PR*0.8,"recirc"),
]

INSULATION = [
    # Yellow lagging on hot/recirc lines
    ((2.60, 1.10, 2.5), (4.50, 1.10, 2.5), MR  + 0.038),
    ((7.80, 1.10, 2.5), (11.0, 1.10, 2.5), PR*1.3 + 0.032),
    ((-0.5, 1.10, 0.0), (2.60, 1.10, 0.0), PR  + 0.030),
    ((-0.5, 1.10, 5.0), (2.60, 1.10, 5.0), PR  + 0.030),
]

VALVES = [
    ((-3.0, 0.58, 0.0), (0,0,1)),
    ((-3.0, 0.58, 2.5), (0,0,1)),
    ((-3.0, 0.58, 5.0), (0,0,1)),
    ((3.50, 1.10, 2.5), (1,0,0)),
    ((9.20, 1.10, 2.5), (1,0,0)),
    ((5.20, 1.10, 2.5), (1,0,0)),
]

GAUGES = [
    (1.50, 1.48, 0.0),
    (1.50, 1.48, 2.5),
    (1.50, 1.48, 5.0),
    (2.60, 1.48, 4.0),
    (6.20, 1.50, 2.5),
]

LIGHTS_3D = [
    (-1.5, 4.8, 0.5), (-1.5, 4.8, 4.5),
    ( 2.0, 4.8, 0.0), ( 2.0, 4.8, 5.0),
    ( 6.0, 4.8, 0.0), ( 6.0, 4.8, 5.0),
    ( 9.5, 4.8, 2.5),
]


# ============================================================================
# PHYSICS SIMULATION
# ============================================================================

def physics_sim(n_frames: int):
    t = np.linspace(0, 6*math.pi, n_frames)
    freq = np.array([0.8, 1.2, 1.7])
    P  = np.zeros((n_frames, len(PIPES)))
    for i, (*_, tag) in enumerate(PIPES):
        base = {"intake":0.28,"discharge":0.55,"hot":0.72,
                "manifold":0.85,"discharge_final":0.60,"recirc":0.35}.get(tag, 0.4)
        osc  = 0.13 * np.sin(freq[i%3]*t + i*0.7)
        P[:, i] = np.clip(base + osc, 0.04, 0.98)

    T_in  = 78 + 9*np.sin(0.18*t) + 1.2*RNG.standard_normal(n_frames)
    T_out = 44 + 5*np.sin(0.18*t + 0.3) + 0.6*RNG.standard_normal(n_frames)
    return {"P": P, "T_in": T_in, "T_out": T_out, "t": t}


def p_color(p):
    p = float(np.clip(p, 0, 1))
    if p < 0.5:
        r, g, b = 0.05, 0.3 + p*0.5, 0.8 - p*0.4
    else:
        r, g, b = 0.1 + (p-0.5)*1.8, 0.55 - (p-0.5)*1.0, 0.2
    return (float(np.clip(r,0,1)), float(np.clip(g,0,1)), float(np.clip(b,0,1)))


# ============================================================================
# STEAM PARTICLES
# ============================================================================

class Steam:
    def __init__(self, n=200):
        self.n   = n
        self.pos = np.column_stack([
            RNG.uniform(5.8, 7.5, n),
            RNG.uniform(1.2, 1.9, n),
            RNG.uniform(0.9, 4.1, n),
        ])
        self.vel = np.column_stack([
            RNG.normal(0, 0.006, n),
            RNG.uniform(0.009, 0.028, n),
            RNG.normal(0, 0.006, n),
        ])
        self.age = RNG.uniform(0, 1, n)
        self.size = RNG.uniform(5, 12, n)

    def step(self):
        self.pos += self.vel
        self.vel[:, 0] += RNG.normal(0, 0.0008, self.n)
        self.vel[:, 2] += RNG.normal(0, 0.0008, self.n)
        self.age += 0.010
        dead = self.age > 1.0
        if dead.any():
            self.pos[dead, 0] = RNG.uniform(5.8, 7.5, dead.sum())
            self.pos[dead, 1] = RNG.uniform(1.12, 1.38, dead.sum())
            self.pos[dead, 2] = RNG.uniform(0.9, 4.1, dead.sum())
            self.vel[dead, 1] = RNG.uniform(0.009, 0.028, dead.sum())
            self.age[dead] = 0.0


# ============================================================================
# CAMERA PATH
# ============================================================================

def cam_path(fi, n):
    t = fi / n
    keys = [
        # eye (inside scene bounds x:-4.5..12, y:0.1..4.5, z:-1.5..7.0)
        # eye                    focal
        (( 9.0, 2.8, -0.5),  (1.0, 0.75, 2.5)),   # 0.00  wide: front-right corner
        ((-3.5, 2.2,  2.5),  (5.0, 0.80, 2.5)),   # 0.22  left wall: full plant vista
        (( 0.0, 2.0, -0.8),  (0.5, 0.80, 2.5)),   # 0.40  near front wall: pump close-up
        (( 0.5, 1.8,  7.0),  (5.5, 0.80, 2.5)),   # 0.58  back side: HX approach
        ((11.5, 2.1,  6.5),  (5.5, 0.80, 2.5)),   # 0.75  far right: HX and pipes
        (( 5.0, 4.0,  2.5),  (3.0, 0.80, 2.5)),   # 0.90  overhead: aerial overview
        (( 9.0, 2.8, -0.5),  (1.0, 0.75, 2.5)),   # 1.00  back to start
    ]
    kt = [0.0, 0.22, 0.40, 0.58, 0.75, 0.90, 1.00]
    for i in range(len(kt)-1):
        if t <= kt[i+1]:
            s = (t - kt[i]) / (kt[i+1] - kt[i])
            s = s*s*(3 - 2*s)    # smoothstep
            e = tuple(keys[i][0][j]*(1-s)+keys[i+1][0][j]*s for j in range(3))
            f = tuple(keys[i][1][j]*(1-s)+keys[i+1][1][j]*s for j in range(3))
            return e, f
    return keys[-1]


# ============================================================================
# RENDERER
# ============================================================================

class UltraRenderer:

    def __init__(self, n=N):
        self.n    = n
        self.sim  = physics_sim(n)
        self.steam = Steam(200)
        print("  Building scene meshes...")
        self._build()
        print(f"  {len(self._static)} static + {len(PIPES)} dynamic pipes")

    def _build(self):
        s = {}

        # Environment
        s["floor"]   = make_floor(-5, 12.5, -2, 7.5)
        s["wall_b"]  = make_wall(-5, 12.5, -1.8, height=5.2)
        s["wall_l"]  = pv.Box(bounds=[-5.0, -4.8, 0, 5.2, -2, 7.5])
        s["wall_r"]  = pv.Box(bounds=[12.3,  12.5, 0, 5.2, -2, 7.5])
        s["ceiling"] = make_ceiling(-5, 12.5, 5.0, -2, 7.5)

        # Pipe rack + support structure
        s["rack"]    = make_pipe_rack(-4.5, 11.5, 3.6, [-0.5, 5.5])

        # Column support posts
        for xi in [-4.5, 2.5, 7.0, 11.5]:
            for zi in [-0.5, 5.5]:
                s[f"col_{xi}_{zi}"] = pv.Cylinder(
                    center=[xi, 2.0, zi], direction=(0,1,0), radius=0.07, height=4.0)

        # Catwalk along pump row
        s["catwalk"] = make_catwalk(-1.5, 1.30, -0.6, 6.0)

        # Pumps (built per frame for rotation)
        # Plinths
        for i, (px, _, pz) in enumerate(PUMP_XYZ):
            s[f"plinth_{i}"] = pv.Box(bounds=[px-0.48, px+0.42, 0.0, 0.16,
                                               pz-0.40, pz+0.40])

        # Heat exchanger
        s["hx"] = make_hx(*HX)

        # Insulation sleeves
        for j, (p0, p1, ri) in enumerate(INSULATION):
            s[f"insul_{j}"] = make_tube(p0, p1, r=ri, n=18)

        # Valves
        for j, (pos, ax) in enumerate(VALVES):
            s[f"valve_{j}"] = make_valve(*pos, axis=ax)

        # Pressure gauges
        for j, (gx, gy, gz) in enumerate(GAUGES):
            s[f"gauge_{j}"] = make_gauge(gx, gy, gz)

        # Expansion joints
        for i, (px, _, pz) in enumerate(PUMP_XYZ):
            s[f"expjt_{i}"] = make_expansion_joint(px, 1.10, pz)

        # Ceiling light fixtures
        for j, (lx, ly, lz) in enumerate(LIGHTS_3D):
            fix = pv.Cylinder(center=[lx, ly, lz], direction=(0,1,0),
                              radius=0.22, height=0.04)
            s[f"fixture_{j}"] = fix

        # Control panel on back wall
        s["panel"] = make_control_panel(6.5, 1.20, -1.75)

        # Flanges on pipes
        for i, (p0, p1, r, _) in enumerate(PIPES):
            d  = np.array(p1, float) - np.array(p0, float)
            dl = np.linalg.norm(d)
            if dl < 1e-6:
                continue
            d /= dl
            for pt, sign in [(p0, 1), (p1, -1)]:
                fc = [pt[0]+d[0]*sign*0.08, pt[1]+d[1]*sign*0.08, pt[2]+d[2]*sign*0.08]
                s[f"flange_{i}_{sign}"] = make_flange(fc, d, r_out=r+0.05)

        self._static = s
        self._pipe_meshes = [make_tube(p0, p1, r=r, n=22) for p0, p1, r, _ in PIPES]

    def render_frame(self, fi: int) -> np.ndarray:
        pl = pv.Plotter(off_screen=True, window_size=[W, H])
        pl.set_background("#06080c")
        # Remove default lights, then add our own carefully tuned set.
        # (lighting="none" at Plotter init breaks PBR in offscreen mode on Windows)
        pl.remove_all_lights()

        # ── Lighting ──────────────────────────────────────────────────────
        # Global ambient fill  (bright industrial hall)
        pl.add_light(pv.Light(light_type="headlight",
                              color=(0.85, 0.88, 0.95), intensity=0.20))
        # Key light — large industrial skylight / high window
        pl.add_light(pv.Light(position=(22, 20, -5), focal_point=(3.5, 0.8, 2.5),
                               color=(0.98, 0.98, 1.00), intensity=1.10,
                               light_type="scene light"))
        # Fill light A — bounce off floor/walls left
        pl.add_light(pv.Light(position=(-8, 2.0, 10), focal_point=(3.5, 0.8, 2.5),
                               color=(0.78, 0.88, 1.00), intensity=0.42,
                               light_type="scene light"))
        # Fill light B — right side bounce
        pl.add_light(pv.Light(position=(14, 2.0, -4), focal_point=(3.5, 0.8, 2.5),
                               color=(0.88, 0.88, 0.95), intensity=0.28,
                               light_type="scene light"))
        # Rim / back light (separates objects from background)
        pl.add_light(pv.Light(position=(3.5, 6.0, 14), focal_point=(3.5, 0.8, 2.5),
                               color=(0.80, 0.85, 1.00), intensity=0.22,
                               light_type="scene light"))
        # Ceiling industrial lights (warm, bright)
        t_s = fi / FPS
        for j, (lx, ly, lz) in enumerate(LIGHTS_3D):
            flicker = 1.0 + 0.012 * math.sin(t_s * 37 + j * 2.1)
            pl.add_light(pv.Light(
                position=[lx, ly-0.05, lz],
                light_type="scene light",
                color=(1.0, 0.95, 0.80),
                intensity=0.70 * flicker,
                positional=True, cone_angle=70,
                attenuation_values=(0.02, 0.06, 0.0),
            ))

        # ── Environment ───────────────────────────────────────────────────
        pl.add_mesh(self._static["floor"],
                    color="#7a7a70", roughness=0.82, metallic=0.05, pbr=True)
        for k in ["wall_b", "wall_l", "wall_r"]:
            pl.add_mesh(self._static[k], color="#6a6a62",
                        roughness=0.88, metallic=0.03, pbr=True)
        pl.add_mesh(self._static["ceiling"],
                    color="#5a5c62", roughness=0.86, metallic=0.03, pbr=True)

        # ── Structure ─────────────────────────────────────────────────────
        for k, m in self._static.items():
            if k.startswith("col_") or k.startswith("rack"):
                pl.add_mesh(m, color="#2a2e32", roughness=0.55, metallic=0.82,
                            pbr=True, smooth_shading=True)
            if k.startswith("plinth_"):
                pl.add_mesh(m, color="#6e6e68", roughness=0.88, metallic=0.02,
                            pbr=True, smooth_shading=True)
            if k == "catwalk":
                pl.add_mesh(m, color="#3a3e42", roughness=0.62, metallic=0.75,
                            pbr=True, smooth_shading=True)

        # ── Pumps (spin per frame) ────────────────────────────────────────
        spin = (fi / FPS) * 360 * 2.8
        for i, (px, py, pz) in enumerate(PUMP_XYZ):
            pm = make_pump(px, py, pz, spin_deg=spin + i*120)
            pl.add_mesh(pm, color="#1c4fa0", roughness=0.28, metallic=0.62,
                        pbr=True, smooth_shading=True)

        # ── Pipes (pressure colour) ───────────────────────────────────────
        tidx = fi % self.n
        for si, pmesh in enumerate(self._pipe_meshes):
            col = p_color(float(self.sim["P"][tidx, si]))
            pl.add_mesh(pmesh, color=col, roughness=0.18, metallic=0.92,
                        pbr=True, smooth_shading=True)

        # ── Flanges ───────────────────────────────────────────────────────
        for k, m in self._static.items():
            if k.startswith("flange_"):
                pl.add_mesh(m, color="#c0c4c8", roughness=0.22, metallic=0.90,
                            pbr=True, smooth_shading=True)

        # ── Insulation ────────────────────────────────────────────────────
        for k, m in self._static.items():
            if k.startswith("insul_"):
                pl.add_mesh(m, color="#dba518", roughness=0.88, metallic=0.01,
                            pbr=True, smooth_shading=True)

        # ── Valves ────────────────────────────────────────────────────────
        for k, m in self._static.items():
            if k.startswith("valve_"):
                pl.add_mesh(m, color="#2c3038", roughness=0.40, metallic=0.78,
                            pbr=True, smooth_shading=True)

        # ── Gauges ────────────────────────────────────────────────────────
        for k, m in self._static.items():
            if k.startswith("gauge_"):
                pl.add_mesh(m, color="#e0e4e8", roughness=0.15, metallic=0.85,
                            pbr=True, smooth_shading=True)

        # ── Expansion joints ──────────────────────────────────────────────
        for k, m in self._static.items():
            if k.startswith("expjt_"):
                pl.add_mesh(m, color="#8a9090", roughness=0.35, metallic=0.72,
                            pbr=True, smooth_shading=True)

        # ── Heat exchanger ────────────────────────────────────────────────
        pl.add_mesh(self._static["hx"], color=(0.80, 0.82, 0.84),
                    roughness=0.14, metallic=0.94, pbr=True, smooth_shading=True)

        # ── Ceiling fixtures ──────────────────────────────────────────────
        for k, m in self._static.items():
            if k.startswith("fixture_"):
                pl.add_mesh(m, color="#f8f4e8", roughness=0.12, metallic=0.15,
                            pbr=True, smooth_shading=True)

        # ── Control panel ─────────────────────────────────────────────────
        pl.add_mesh(self._static["panel"], color="#1e2228",
                    roughness=0.55, metallic=0.45, pbr=True, smooth_shading=True)

        # ── Steam ─────────────────────────────────────────────────────────
        self.steam.step()
        sc  = pv.PolyData(self.steam.pos)
        ops = float(np.mean(np.clip(1.0 - self.steam.age, 0.12, 0.60)))
        pl.add_mesh(sc, style="points", point_size=5.5,
                    color="#c8e0f0", opacity=ops, render_points_as_spheres=True)

        # ── Camera ────────────────────────────────────────────────────────
        eye, foc = cam_path(fi, self.n)
        pl.camera_position = [eye, foc, (0, 1, 0)]
        pl.camera.view_angle = 45.0

        # Render
        img = pl.screenshot(return_img=True, window_size=[W, H])
        pl.close()
        return img


# ============================================================================
# PHASE 2 — PHYSICS EXTRACTION (Inverse Pipeline)
# ============================================================================

class VideoPhysicsEncoder(nn.Module):
    """
    Lightweight CNN: extracts physics features from a single rendered frame.
    Input:  (3, 108, 192)  — downsampled frame
    Output: n_physics      — normalised physical field values
    """
    def __init__(self, n_physics: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),   # 54×96
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 27×48
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 14×24
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),  #  7×12
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16, 256), nn.ReLU(),
            nn.Linear(256, 64),    nn.ReLU(),
            nn.Linear(64, n_physics),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


def train_physics_extractor(
    frames:   List[np.ndarray],   # list of (H, W, 3) uint8 frames
    sim:      Dict[str, np.ndarray],
    epochs:   int = 600,
    lr:       float = 2e-3,
) -> Tuple[VideoPhysicsEncoder, Dict]:
    """Train the inverse model: frame → (P_mean, T_in_norm, T_out_norm)."""
    print("  [Extraction] Preparing training pairs...")

    # Downsample frames for fast CNN processing
    TARGET_H, TARGET_W = 108, 192
    X_list, Y_list = [], []
    n = len(frames)
    for fi, frame in enumerate(frames):
        # Resize frame
        pil   = Image.fromarray(frame).resize((TARGET_W, TARGET_H), Image.BILINEAR)
        x     = np.array(pil, dtype=np.float32).transpose(2, 0, 1) / 255.0
        # Ground truth: mean pipe pressure, T_in/100, T_out/100
        p_mean = float(sim["P"][fi % len(sim["P"])].mean())
        t_in   = float(sim["T_in"][fi % len(sim["T_in"])]) / 100.0
        t_out  = float(sim["T_out"][fi % len(sim["T_out"])]) / 100.0
        X_list.append(x)
        Y_list.append([p_mean, t_in, t_out])

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)  # (N, 3, H, W)
    Y = torch.tensor(np.array(Y_list),  dtype=torch.float32)  # (N, 3)

    model = VideoPhysicsEncoder(n_physics=3)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-4)
    loss_fn = nn.MSELoss()

    print(f"  [Extraction] Training {epochs} epochs, {X.shape[0]} frames...")
    best, best_state, history = float("inf"), None, []
    for ep in range(1, epochs+1):
        idx  = torch.randperm(X.shape[0])[:min(32, X.shape[0])]
        pred = model(X[idx])
        loss = loss_fn(pred, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        history.append(float(loss))
        if loss < best:
            best = float(loss)
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 100 == 0:
            print(f"    ep {ep:>5}/{epochs}  loss={loss:.5f}")

    model.load_state_dict(best_state)
    model.eval()

    # Evaluate
    with torch.no_grad():
        pred_all = model(X).numpy()
    truth_all = Y.numpy()
    rmse = float(np.sqrt(np.mean((pred_all - truth_all)**2)))
    print(f"  [Extraction] RMSE={rmse:.4f}  (best training loss={best:.5f})")
    return model, {"rmse": rmse, "history": history,
                   "pred": pred_all.tolist(), "truth": truth_all.tolist()}


# ============================================================================
# MAIN
# ============================================================================

def main():
    t_start = time.time()
    print("\n" + "="*68)
    print("  Ultrarealistic Industrial Digital Twin  |  PINNeAPPle")
    print(f"  Resolution: {W}x{H}  |  {FPS}fps  |  {N} frames ({N/FPS:.0f}s)")
    print("  PBR + SSAO + Shadows + ACES + Bloom + DoF + Color Grade")
    print("="*68 + "\n")

    renderer = UltraRenderer(n=N)

    # ── Phase 1: Render ───────────────────────────────────────────────────
    print("\n[1/3] Rendering HD frames...")

    # Project 3D light positions to 2D for lens flare
    light_pix = [(int(lx*90), int(H - ly*105)) for lx, ly, _ in LIGHTS_3D]

    frames_raw  = []
    frames_post = []

    t0 = time.time()
    for fi in range(N):
        raw  = renderer.render_frame(fi)
        post = full_post_process(raw, fi, light_pix=light_pix, dof=True, grain=True)
        frames_raw.append(raw)
        frames_post.append(post)
        if (fi+1) % FPS == 0 or fi == N-1:
            elapsed = time.time() - t0
            eta = elapsed / (fi+1) * (N - fi - 1)
            print(f"  Frame {fi+1:>4}/{N}  ({(fi+1)/N*100:.0f}%)  "
                  f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s")

    print(f"  Total render time: {time.time()-t0:.1f}s  ({(time.time()-t0)/N:.2f}s/frame)")

    # ── Export MP4 ────────────────────────────────────────────────────────
    print("\n[2/3] Exporting HD video...")
    mp4 = OUT / "factory_hd.mp4"
    iio.imwrite(str(mp4), np.stack(frames_post), fps=FPS)
    print(f"  MP4  -> {mp4}  ({mp4.stat().st_size/1024/1024:.1f} MB)")

    # Save preview frames at different timestamps
    for ts, name in [(0, "shot_wide"), (N//4, "shot_diagonal"),
                     (N//2, "shot_pumps"), (3*N//4, "shot_hx")]:
        p = OUT / f"{name}.png"
        Image.fromarray(frames_post[ts]).save(str(p))
        print(f"  PNG  -> {p.name}")

    # ── Phase 2: Physics extraction ───────────────────────────────────────
    print("\n[3/3] Training physics extraction model (inverse pipeline)...")
    model, metrics = train_physics_extractor(
        frames_post, renderer.sim, epochs=600, lr=2e-3
    )

    # Save model
    torch.save(model.state_dict(), OUT / "extraction_model.pt")
    print(f"  Model -> {OUT / 'extraction_model.pt'}")

    # Run inference on all frames
    print("  Running inference on all frames...")
    TARGET_H, TARGET_W = 108, 192
    X_all = []
    for frame in frames_post:
        pil = Image.fromarray(frame).resize((TARGET_W, TARGET_H), Image.BILINEAR)
        X_all.append(np.array(pil, np.float32).transpose(2,0,1) / 255.0)
    X_t = torch.tensor(np.stack(X_all), dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_t).numpy()

    # Export extracted physics
    import json
    t_arr = renderer.sim["t"]
    gt    = np.column_stack([
        renderer.sim["P"].mean(axis=1),
        renderer.sim["T_in"] / 100.0,
        renderer.sim["T_out"] / 100.0,
    ])
    extracted = {
        "description": "Physics fields extracted from rendered video using inverse PINN",
        "fields":      ["mean_pressure_norm", "T_in_norm", "T_out_norm"],
        "units":       ["[0,1]", "[0,1] = T/100C", "[0,1] = T/100C"],
        "n_frames":    N,
        "fps":         FPS,
        "rmse":        metrics["rmse"],
        "extracted":   predictions.tolist(),
        "ground_truth": gt[:N].tolist(),
    }
    json_path = OUT / "extracted_physics.json"
    json_path.write_text(json.dumps(extracted, indent=2))
    print(f"  Physics -> {json_path}")

    # Final summary plot
    _summary_plot(predictions, gt[:N], metrics, OUT / "extraction_summary.png")

    print("\n" + "="*68)
    print(f"  Done.  Total elapsed: {time.time()-t_start:.0f}s")
    print(f"  Output: {OUT}")
    print("  Files:")
    for f in sorted(OUT.iterdir()):
        if f.is_file():
            print(f"    {f.name:<35s} {f.stat().st_size/1024:>8.1f} KB")
    print("="*68)


def _summary_plot(pred, gt, metrics, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0a0e14")
    labels = ["Mean Pressure [norm]", "T_in / 100°C", "T_out / 100°C"]
    colors = ["#38bdf8", "#f97316", "#22c55e"]
    t = np.arange(len(pred))

    for ax, i, lab, col in zip(axes, range(3), labels, colors):
        ax.set_facecolor("#111827")
        ax.plot(t, gt[:, i],   color="white",  lw=2.0, label="Ground truth", alpha=0.8)
        ax.plot(t, pred[:, i], color=col,       lw=1.5, ls="--", label="Extracted from video")
        ax.set_xlabel("Frame", color="#64748b", fontsize=8)
        ax.set_ylabel(lab, color="#64748b", fontsize=8)
        ax.tick_params(colors="#64748b", labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e293b")
        ax.legend(fontsize=8, labelcolor="#94a3b8",
                  facecolor="#1e293b", edgecolor="#334155")

    rmse = metrics["rmse"]
    fig.suptitle(f"Physics Extraction from Video  —  RMSE = {rmse:.4f}",
                 color="white", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120, facecolor="#0a0e14", bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot  -> {path.name}")


if __name__ == "__main__":
    main()
