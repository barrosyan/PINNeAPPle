# -*- coding: utf-8 -*-
"""Industrial Factory 3D Render -- Photorealistic Digital Twin Video
====================================================================

Generates a 1920x1080 MP4 video of a photorealistic 3D industrial pump
station using PyVista / VTK Physically-Based Rendering (PBR).

What is in the scene
--------------------
  - 3 centrifugal pumps on concrete plinths (blue painted metal, PBR)
  - Parallel pipe manifold with flanges and valves (metallic steel)
  - Shell-and-tube heat exchanger (stainless steel)
  - Overhead pipe rack (structural steel I-beams)
  - Concrete floor with grid markings
  - 6 industrial ceiling lights (warm point lights)
  - Pipe insulation sleeves on hot lines (yellow lagging)
  - Steam particles rising from HX vents
  - Pressure-coded pipe colour overlay (physics-driven animation)
  - Slowly orbiting camera with depth-of-field

Run
---
  python factory_3d_render.py

Dependencies (already installed): pyvista, vtk, trimesh, numpy, imageio-ffmpeg
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pyvista as pv
import imageio.v3 as iio

pv.global_theme.anti_aliasing = "msaa"
pv.global_theme.multi_samples  = 8

OUT_DIR = Path(__file__).parent / "outputs" / "factory_3d"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(0)

# ── Resolution ──────────────────────────────────────────────────────────────
W, H  = 1280, 720
FPS   = 24
N_FRAMES = 180   # 7.5 seconds


# ============================================================================
# GEOMETRY BUILDERS
# ============================================================================

def tube(p0, p1, radius=0.06, n=20) -> pv.PolyData:
    """Straight tube between two 3-D points."""
    pts    = np.array([p0, p1], dtype=float)
    line   = pv.lines_from_points(pts)
    return line.tube(radius=radius, n_sides=n)


def elbow(center, r_pipe=0.06, r_elbow=0.14, n=16) -> pv.PolyData:
    """90-degree elbow approximated by a torus quarter."""
    theta = np.linspace(0, math.pi/2, n)
    pts   = np.column_stack([
        center[0] + r_elbow * np.cos(theta),
        np.full(n, center[1]),
        center[2] + r_elbow * np.sin(theta),
    ])
    path  = pv.Spline(pts, n_points=n)
    return path.tube(radius=r_pipe, n_sides=16)


def flange(center, direction, r_outer=0.10, thickness=0.025) -> pv.PolyData:
    """Pipe flange (flat disc)."""
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)
    return pv.Disc(center=center, normal=d, inner=0.04, outer=r_outer
                   ).extrude(d * thickness, capping=True)


def valve(center, axis=(1,0,0)) -> pv.PolyData:
    """Gate valve body + stem."""
    body  = pv.Cylinder(center=center, direction=axis, radius=0.09, height=0.15)
    c_s   = [center[0], center[1]+0.20, center[2]]
    stem  = pv.Cylinder(center=c_s, direction=(0,1,0), radius=0.02, height=0.25)
    wheel = pv.Disc(center=[center[0], center[1]+0.35, center[2]],
                    normal=(0,1,0), inner=0.02, outer=0.12)
    return body.merge(stem).merge(wheel)


def pump_body(center, angle_deg=0.0) -> pv.PolyData:
    """Centrifugal pump: volute casing + suction/discharge nozzles + motor."""
    cx, cy, cz = center

    # Volute (ellipsoid approximation via scaled sphere)
    volute = pv.Sphere(radius=1.0, center=[cx, cy+0.18, cz])
    volute.scale([0.28, 0.22, 0.32], inplace=True)

    # Motor housing
    motor = pv.Cylinder(center=[cx-0.45, cy+0.18, cz],
                        direction=(1,0,0), radius=0.18, height=0.55)

    # Coupling guard
    guard = pv.Cylinder(center=[cx-0.14, cy+0.18, cz],
                        direction=(1,0,0), radius=0.12, height=0.10)

    # Suction nozzle (Z-direction)
    suction = pv.Cylinder(center=[cx, cy+0.18, cz-0.22],
                           direction=(0,0,1), radius=0.07, height=0.28)

    # Discharge nozzle (Y-direction, upward)
    discharge = pv.Cylinder(center=[cx, cy+0.38, cz],
                             direction=(0,1,0), radius=0.06, height=0.22)

    # Baseplate
    base = pv.Box(bounds=[cx-0.55, cx+0.35, cy-0.01, cy+0.04, cz-0.35, cz+0.35])

    # Impeller peek through window (darker disc inside volute)
    impeller_angle = angle_deg
    imp = pv.Disc(center=[cx, cy+0.18, cz],
                  normal=(1,0,0), inner=0.0, outer=0.20)
    imp.rotate_x(impeller_angle, point=[cx, cy+0.18, cz], inplace=True)

    return volute.merge(motor).merge(guard).merge(suction).merge(discharge).merge(base)


def heat_exchanger(bounds) -> pv.PolyData:
    """Shell-and-tube HX: shell + channel heads + nozzles + support saddles."""
    x0, x1, y0, y1, z0, z1 = bounds
    cx = (x0+x1)/2; cy = (y0+y1)/2; cz = (z0+z1)/2
    L  = x1-x0

    shell   = pv.Cylinder(center=[cx, cy, cz], direction=(1,0,0),
                           radius=0.45, height=L)
    head_L  = pv.Sphere(center=[x0-0.10, cy, cz], radius=0.46)
    head_R  = pv.Sphere(center=[x1+0.10, cy, cz], radius=0.46)
    # Nozzles
    n1 = pv.Cylinder(center=[x0, cy+0.52, cz], direction=(0,1,0), radius=0.08, height=0.20)
    n2 = pv.Cylinder(center=[x1, cy+0.52, cz], direction=(0,1,0), radius=0.08, height=0.20)
    n3 = pv.Cylinder(center=[x0, cy-0.52, cz], direction=(0,1,0), radius=0.07, height=0.18)
    n4 = pv.Cylinder(center=[x1, cy-0.52, cz], direction=(0,1,0), radius=0.07, height=0.18)
    # Saddle supports
    sad_L = pv.Box(bounds=[x0+0.3, x0+0.6, y0, cy-0.44, cz-0.55, cz+0.55])
    sad_R = pv.Box(bounds=[x1-0.6, x1-0.3, y0, cy-0.44, cz-0.55, cz+0.55])

    return (shell.merge(head_L).merge(head_R)
            .merge(n1).merge(n2).merge(n3).merge(n4)
            .merge(sad_L).merge(sad_R))


def ibeam(p0, p1, h=0.20, w=0.12, t=0.012) -> pv.PolyData:
    """Structural I-beam between two points (always horizontal here)."""
    pts  = np.array([p0, p1], dtype=float)
    line = pv.lines_from_points(pts)
    # Use a rectangular cross-section tube as approximation
    return line.tube(radius=h/2 * 0.6, n_sides=4)


def plinth(cx, cz, hw=0.45, hd=0.40, hh=0.15) -> pv.PolyData:
    """Concrete pump plinth."""
    return pv.Box(bounds=[cx-hw, cx+hw, 0.0, hh, cz-hd, cz+hd])


def floor_mesh(bounds, grid_spacing=1.0) -> pv.PolyData:
    """Textured concrete floor with grid lines."""
    x0, x1, z0, z1 = bounds
    floor = pv.Plane(center=[(x0+x1)/2, 0.0, (z0+z1)/2],
                     direction=(0,1,0),
                     i_size=x1-x0, j_size=z1-z0,
                     i_resolution=int((x1-x0)/grid_spacing),
                     j_resolution=int((z1-z0)/grid_spacing))
    return floor


# ============================================================================
# SCENE ASSEMBLY
# ============================================================================

PUMP_CENTRES  = [(-0.5, 0.15, 0.0), (-0.5, 0.15, 2.5), (-0.5, 0.15, 5.0)]
HX_BOUNDS     = (4.5, 7.5, 0.15, 1.10, 1.0, 4.0)   # x0,x1, y0,y1, z0,z1
PIPE_R        = 0.055
HEADER_R      = 0.08

PIPE_DEFS: List[Tuple] = [
    # intake pipes (coming from -X side)
    ((-3.5,0.55,0.0), (-1.2,0.55,0.0), PIPE_R, "steel"),
    ((-3.5,0.55,2.5), (-1.2,0.55,2.5), PIPE_R, "steel"),
    ((-3.5,0.55,5.0), (-1.2,0.55,5.0), PIPE_R, "steel"),
    # pump discharge risers
    ((-0.5,0.55,0.0), (-0.5,1.05,0.0),  PIPE_R, "steel"),
    ((-0.5,0.55,2.5), (-0.5,1.05,2.5),  PIPE_R, "steel"),
    ((-0.5,0.55,5.0), (-0.5,1.05,5.0),  PIPE_R, "steel"),
    # horizontal legs to discharge header
    ((-0.5,1.05,0.0), (2.5,1.05,0.0), PIPE_R, "hot"),
    ((-0.5,1.05,2.5), (2.5,1.05,2.5), PIPE_R, "hot"),
    ((-0.5,1.05,5.0), (2.5,1.05,5.0), PIPE_R, "hot"),
    # discharge manifold header (Z direction)
    ((2.5,1.05,0.0), (2.5,1.05,5.0), HEADER_R, "manifold"),
    # manifold to HX
    ((2.5,1.05,2.5), (4.5,1.05,2.5), HEADER_R, "hot"),
    # HX to discharge
    ((7.5,1.05,2.5), (10.5,1.05,2.5), PIPE_R*1.2, "discharge"),
]

INSULATION_SEGS: List[Tuple] = [
    # hot lines get yellow insulation sleeves
    ((2.5,1.05,2.5), (4.5,1.05,2.5), HEADER_R+0.035),
    ((7.5,1.05,2.5), (10.5,1.05,2.5), PIPE_R*1.2+0.03),
]

VALVE_POS = [
    ((-2.5, 0.55, 0.0),  (0,0,1)),
    ((-2.5, 0.55, 2.5),  (0,0,1)),
    ((-2.5, 0.55, 5.0),  (0,0,1)),
    ((3.5,  1.05, 2.5),  (1,0,0)),
    ((9.0,  1.05, 2.5),  (1,0,0)),
]

OVERHEAD_BEAMS = [
    # overhead pipe rack (X direction at Z = -0.5)
    ((-4.0,3.5,-0.8), (11.0,3.5,-0.8)),
    ((-4.0,3.5, 5.8), (11.0,3.5, 5.8)),
    # cross members
    ((-4.0,3.5,-0.8), (-4.0,3.5, 5.8)),
    ((11.0,3.5,-0.8), (11.0,3.5, 5.8)),
    ((3.5, 3.5,-0.8), (3.5, 3.5, 5.8)),
]

SUPPORT_POSTS = [
    (-4.0,  3.5, -0.8), (-4.0,  3.5, 5.8),
    (11.0,  3.5, -0.8), (11.0,  3.5, 5.8),
    ( 3.5,  3.5, -0.8), ( 3.5,  3.5, 5.8),
]

CEILING_LIGHTS = [
    (-1.0, 4.2,  1.0),
    (-1.0, 4.2,  4.0),
    ( 3.0, 4.2,  0.5),
    ( 3.0, 4.2,  4.5),
    ( 7.0, 4.2,  1.5),
    ( 7.0, 4.2,  3.5),
]


# ============================================================================
# PHYSICS: pressure animation
# ============================================================================

def pressure_timeline(n_frames: int, n_pumps: int = 3):
    """Per-frame pipe pressure values, normalised [0, 1]."""
    t   = np.linspace(0, 4*math.pi, n_frames)
    freq = np.array([0.9, 1.3, 1.7])
    amps = np.array([0.15, 0.12, 0.18])
    P   = np.zeros((n_frames, len(PIPE_DEFS)))
    for seg_i, (_, _, _, label) in enumerate(PIPE_DEFS):
        base = {"steel": 0.3, "hot": 0.65, "manifold": 0.80, "discharge": 0.55}.get(label, 0.4)
        osc  = amps[seg_i % 3] * np.sin(freq[seg_i % 3] * t + seg_i)
        P[:, seg_i] = np.clip(base + osc, 0.05, 0.98)
    return P

def pressure_to_colour(p: float) -> Tuple[float, float, float]:
    """Map pressure [0,1] -> RGB tuple [0,1]."""
    p = float(np.clip(p, 0, 1))
    if p < 0.5:
        r, g, b = 0.0, p*2, 1.0-p
    else:
        r, g, b = (p-0.5)*2, 1.0-(p-0.5)*2, 0.0
    return (r, g, b)


# ============================================================================
# STEAM PARTICLES
# ============================================================================

class Steam:
    """Rising steam cloud from heat exchanger vent."""
    def __init__(self, n: int = 120):
        self.n    = n
        self.pos  = np.column_stack([
            RNG.uniform(6.0, 7.2, n),
            RNG.uniform(1.2, 1.8, n),
            RNG.uniform(1.2, 3.8, n),
        ])
        self.vel  = np.column_stack([
            RNG.normal(0, 0.008, n),
            RNG.uniform(0.012, 0.030, n),
            RNG.normal(0, 0.006, n),
        ])
        self.age  = RNG.uniform(0, 1, n)

    def step(self) -> None:
        self.pos += self.vel
        self.age += 0.012
        # Drift and spread
        self.vel[:, 0] += RNG.normal(0, 0.001, self.n)
        self.vel[:, 2] += RNG.normal(0, 0.001, self.n)
        # Reset old particles at base
        reset = self.age > 1.0
        if reset.any():
            self.pos[reset, 0] = RNG.uniform(6.0, 7.2, reset.sum())
            self.pos[reset, 1] = RNG.uniform(1.15, 1.4, reset.sum())
            self.pos[reset, 2] = RNG.uniform(1.2, 3.8, reset.sum())
            self.vel[reset, 1] = RNG.uniform(0.012, 0.030, reset.sum())
            self.age[reset]    = 0.0

    @property
    def opacity(self) -> np.ndarray:
        return (1.0 - self.age) * 0.55


# ============================================================================
# CAMERA PATH
# ============================================================================

def camera_path(frame: int, total: int):
    """
    Smooth camera orbit in three phases:
      0-25%  : wide establishing shot, slow right pan
      25-55% : move to pump cluster, tilt down-close
      55-80% : orbit HX from right side
      80-100%: pull back to master shot
    """
    t  = frame / total

    # Key camera positions: (eye, focal, up)
    # All cameras at human eye level (y=1.6-2.0) — industrial CCTV/walkthrough feel
    keys = [
        ((11,  2.0,  2.5), (2.0, 0.8, 2.5), (0,1,0)),  # t=0.00 wide master angle
        (( 7,  1.8, -1.5), (0.5, 0.9, 2.5), (0,1,0)),  # t=0.25 diagonal pump view
        ((-2,  1.7,  2.5), (3.0, 0.9, 2.5), (0,1,0)),  # t=0.45 facing pump array
        (( 2,  1.5,  8.5), (5.0, 0.9, 2.5), (0,1,0)),  # t=0.60 HX front approach
        (( 9,  1.8,  6.0), (5.5, 0.9, 2.5), (0,1,0)),  # t=0.78 HX side close
        ((11,  2.0,  2.5), (2.0, 0.8, 2.5), (0,1,0)),  # t=1.00 back to master
    ]
    key_t = [0.0, 0.25, 0.45, 0.60, 0.78, 1.00]

    # Find segment
    for i in range(len(key_t)-1):
        if t <= key_t[i+1]:
            s  = (t - key_t[i]) / (key_t[i+1] - key_t[i])
            s  = 3*s*s - 2*s*s*s   # smoothstep
            def _lerp(a, b): return tuple(a[j]*(1-s)+b[j]*s for j in range(3))
            eye   = _lerp(keys[i][0], keys[i+1][0])
            focal = _lerp(keys[i][1], keys[i+1][1])
            up    = keys[i][2]
            return eye, focal, up

    return keys[-1]


# ============================================================================
# RENDERER
# ============================================================================

class FactoryRenderer:

    def __init__(self, n_frames: int = N_FRAMES, res=(W, H)):
        self.n_frames = n_frames
        self.W, self.H = res
        self.pressure  = pressure_timeline(n_frames)
        self.steam     = Steam(n=150)
        self._build_static_meshes()

    # ── build static geometry ─────────────────────────────────────────────
    def _build_static_meshes(self):
        self.meshes = {}

        # Floor
        self.meshes["floor"] = floor_mesh((-5, 12, -1.5, 7.0))

        # Pump plinths + bodies
        for i, c in enumerate(PUMP_CENTRES):
            self.meshes[f"plinth_{i}"] = plinth(c[0], c[2])
            self.meshes[f"pump_{i}"]   = pump_body(c, angle_deg=0.0)

        # Pipes (geometry stored; colour set per frame)
        self.pipe_meshes = []
        for i, (p0, p1, r, _) in enumerate(PIPE_DEFS):
            t = tube(p0, p1, radius=r)
            self.pipe_meshes.append(t)
            self.meshes[f"pipe_{i}"] = t
            # Add flanges at both ends
            d = np.array(p1) - np.array(p0)
            d /= np.linalg.norm(d)
            for pt in (p0, p1):
                self.meshes[f"flange_{i}_{id(pt)}"] = flange(
                    [pt[0]+d[0]*0.06, pt[1]+d[1]*0.06, pt[2]+d[2]*0.06], d)

        # Insulation sleeves
        for j, (p0, p1, ri) in enumerate(INSULATION_SEGS):
            self.meshes[f"insul_{j}"] = tube(p0, p1, radius=ri, n=16)

        # Valves
        for j, (pos, ax) in enumerate(VALVE_POS):
            self.meshes[f"valve_{j}"] = valve(pos, ax)

        # Heat exchanger
        self.meshes["hx"] = heat_exchanger(HX_BOUNDS)

        # Overhead beams
        for j, (pa, pb) in enumerate(OVERHEAD_BEAMS):
            self.meshes[f"beam_{j}"] = ibeam(pa, pb)

        # Support posts
        for j, (px, py, pz) in enumerate(SUPPORT_POSTS):
            self.meshes[f"post_{j}"] = pv.Cylinder(
                center=[px, py/2, pz], direction=(0,1,0),
                radius=0.06, height=py)

        # Ceiling light fixtures (small discs)
        for j, lp in enumerate(CEILING_LIGHTS):
            fix = pv.Disc(center=lp, normal=(0,1,0), inner=0.0, outer=0.18)
            self.meshes[f"fixture_{j}"] = fix

    # ── render one frame ─────────────────────────────────────────────────
    def render_frame(self, frame_idx: int) -> np.ndarray:
        pl = pv.Plotter(off_screen=True, window_size=[self.W, self.H])
        pl.set_background("#0c0d10")   # near-black dark sky

        # ── Lights ────────────────────────────────────────────────────────
        pl.remove_all_lights()
        # Warm ambient
        amb = pv.Light(light_type="headlight", color=(0.85, 0.88, 1.0), intensity=0.12)
        pl.add_light(amb)
        # Key light (sun-like from upper right)
        key = pv.Light(position=(20, 18, -5), focal_point=(3, 0.5, 2.5),
                        color=(1.0, 0.97, 0.90), intensity=0.55,
                        light_type="scene light")
        pl.add_light(key)
        # Fill light
        fill = pv.Light(position=(-8, 8, 8), focal_point=(3, 0.5, 2.5),
                         color=(0.75, 0.85, 1.0), intensity=0.22,
                         light_type="scene light")
        pl.add_light(fill)
        # Ceiling point lights
        t_sec = frame_idx / FPS
        for lp in CEILING_LIGHTS:
            flicker = 1.0 + 0.015 * math.sin(t_sec * 47 + lp[2])
            pl.add_light(pv.Light(
                position=lp, light_type="scene light",
                color=(1.0, 0.97, 0.82),
                intensity=0.30 * flicker,
                positional=True, cone_angle=60,
                attenuation_values=(0.05, 0.15, 0.0),
            ))

        # ── Floor ─────────────────────────────────────────────────────────
        pl.add_mesh(self.meshes["floor"],
                    color="#5a5a52", roughness=0.90, metallic=0.02,
                    pbr=True, smooth_shading=False)

        # ── Plinths ───────────────────────────────────────────────────────
        for i in range(3):
            pl.add_mesh(self.meshes[f"plinth_{i}"],
                        color="#7a7a70", roughness=0.85, metallic=0.01,
                        pbr=True, smooth_shading=True)

        # ── Pumps ─────────────────────────────────────────────────────────
        pump_angle = (frame_idx / FPS) * 360 * 2.5   # ~2.5 rev/s
        for i, c in enumerate(PUMP_CENTRES):
            # Rebuild impeller rotation each frame (only for motor body)
            pm = pump_body(c, angle_deg=pump_angle + i * 120)
            pl.add_mesh(pm, color="#1a4f9e", roughness=0.30, metallic=0.55,
                        pbr=True, smooth_shading=True)

        # ── Pipes with pressure colour ─────────────────────────────────────
        tidx = frame_idx % self.n_frames
        for seg_i, pmesh in enumerate(self.pipe_meshes):
            p_val = float(self.pressure[tidx, seg_i])
            col   = pressure_to_colour(p_val)
            pl.add_mesh(pmesh, color=col, roughness=0.22, metallic=0.88,
                        pbr=True, smooth_shading=True)

        # ── Flanges (uniform steel) ────────────────────────────────────────
        for k, m in self.meshes.items():
            if k.startswith("flange_"):
                pl.add_mesh(m, color="#c8c8c8", roughness=0.25, metallic=0.90,
                            pbr=True, smooth_shading=True)

        # ── Insulation ────────────────────────────────────────────────────
        for k, m in self.meshes.items():
            if k.startswith("insul_"):
                pl.add_mesh(m, color="#e8b020", roughness=0.85, metallic=0.01,
                            pbr=True, smooth_shading=True)

        # ── Valves ────────────────────────────────────────────────────────
        for k, m in self.meshes.items():
            if k.startswith("valve_"):
                pl.add_mesh(m, color="#3a3a3a", roughness=0.45, metallic=0.72,
                            pbr=True, smooth_shading=True)

        # ── Heat Exchanger ─────────────────────────────────────────────────
        hx_col = (0.82, 0.84, 0.86)
        pl.add_mesh(self.meshes["hx"], color=hx_col,
                    roughness=0.18, metallic=0.92,
                    pbr=True, smooth_shading=True)

        # ── Structural steel ──────────────────────────────────────────────
        for k, m in self.meshes.items():
            if k.startswith(("beam_", "post_")):
                pl.add_mesh(m, color="#2c2c2c", roughness=0.55, metallic=0.80,
                            pbr=True, smooth_shading=True)

        # ── Ceiling light fixtures ────────────────────────────────────────
        for k, m in self.meshes.items():
            if k.startswith("fixture_"):
                pl.add_mesh(m, color="#fffbe8", roughness=0.2, metallic=0.1,
                            pbr=True, smooth_shading=True)

        # ── Steam particles ───────────────────────────────────────────────
        self.steam.step()
        steam_cloud = pv.PolyData(self.steam.pos)
        alpha       = float(np.mean(self.steam.opacity))
        pl.add_mesh(steam_cloud, style="points", point_size=4.5,
                    color="#d0e8f0", opacity=alpha, render_points_as_spheres=True)

        # ── Camera ────────────────────────────────────────────────────────
        eye, focal, up = camera_path(frame_idx, self.n_frames)
        pl.camera_position = [eye, focal, up]
        pl.camera.view_angle = 28.0   # narrow FOV = telephoto feel

        # ── Render ────────────────────────────────────────────────────────
        img = pl.screenshot(return_img=True, window_size=[self.W, self.H])
        pl.close()
        return img


# ============================================================================
# MAIN
# ============================================================================

def main():
    import time

    print("\n" + "="*65)
    print("  Industrial Factory 3D Render  |  PINNeAPPle Digital Twin")
    print("="*65)
    print(f"  Resolution : {W}x{H}  |  {FPS} fps  |  {N_FRAMES} frames ({N_FRAMES/FPS:.0f}s)")
    print(f"  Renderer   : PyVista {pv.__version__} + VTK (PBR mode)")
    print()

    renderer = FactoryRenderer(n_frames=N_FRAMES, res=(W, H))
    print(f"  Scene meshes : {len(renderer.meshes)} objects + {len(renderer.pipe_meshes)} dynamic pipes")

    frames = []
    t0     = time.time()
    for fi in range(N_FRAMES):
        frame = renderer.render_frame(fi)
        frames.append(frame)
        if (fi + 1) % 24 == 0 or fi == N_FRAMES - 1:
            elapsed = time.time() - t0
            eta = elapsed / (fi+1) * (N_FRAMES - fi - 1)
            print(f"  Frame {fi+1:>4}/{N_FRAMES}  ({(fi+1)/N_FRAMES*100:.0f}%)  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    print(f"\n  Total render time : {time.time()-t0:.1f}s")

    # Export MP4
    mp4_path = OUT_DIR / "factory_3d.mp4"
    print(f"\n  Exporting MP4 -> {mp4_path}")
    iio.imwrite(str(mp4_path), np.stack(frames), fps=FPS)
    print(f"  MP4 size : {mp4_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Export a preview frame
    preview = OUT_DIR / "preview_frame.png"
    from PIL import Image
    Image.fromarray(frames[N_FRAMES//4]).save(str(preview))
    print(f"  Preview  -> {preview}")

    print("\n" + "="*65)
    print("  Done.")
    print("="*65)


if __name__ == "__main__":
    main()
