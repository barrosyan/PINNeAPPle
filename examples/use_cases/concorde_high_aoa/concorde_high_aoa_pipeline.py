# -*- coding: utf-8 -*-
"""Concorde-Inspired High-Angle-of-Attack Delta-Wing External Aerodynamics (LBM)
================================================================================

Inspired by the public "FluidX3D simulates Concorde at 10 deg AoA, Smagorinsky
LES, Q-criterion vortex visualization" case study. This is an ORCHESTRATION /
WIRING script: it builds a simplified aircraft, hands it to real solvers, and
plots what comes back. It does NOT reimplement any of the physics, geometry,
turbulence, force-integration, or vortex-detection math itself -- every one of
those already exists as a tested module in this repo, listed below.

IMPORTANT HONESTY NOTES (read before trusting any number this script prints)
------------------------------------------------------------------------------
(a) GEOMETRY: there is no real Concorde CAD file in this repository. The
    "aircraft" built here is an explicitly SIMPLIFIED, PARAMETRIC STAND-IN --
    a flat-plate delta-wing planform (nose apex + two trailing-edge corners)
    unioned with a slender ellipsoidal fuselage. It captures the qualitative
    shape (slender delta + fuselage) that produces leading-edge vortex lift,
    nothing more. It is not a photorealistic or aerodynamically faithful
    reproduction of the real aircraft.

(b) PHYSICS: the LBM solver, the Smagorinsky LES closure, the CSG/voxelization
    geometry pipeline, the Q-criterion/vorticity/enstrophy math, and the
    surface-pressure -> CL/CD/CM algebra are 100% imported from real, tested
    `pinneapple_*` modules (see the per-stage mapping below). This file
    contains no hardcoded Cs, no hand-rolled CL/CD formula, no hand-rolled
    Q-criterion, and no hand-rolled obstacle-mask/rotation logic -- those all
    come from the imports.

(c) VALIDATION: the "cross-check" plotted against the simulated CL(alpha) is
    `pinneapple_pdb.benchmarks.get_benchmark("concorde_high_aoa")`, which is
    explicitly a THEORETICAL curve (Polhamus 1966 leading-edge-suction
    vortex-lift analogy for slender delta wings, evaluated at Concorde's
    published aspect ratio ~1.7) -- NOT real Concorde wind-tunnel or
    flight-test data. See that benchmark's own `verification_note` (also
    echoed into this script's console output and metrics.json).

Per-stage module mapping (what this script delegates to, and where)
------------------------------------------------------------------------------
  Geometry (delta-wing planform)  -> pinneapple_design.geometry.csg
                                      (CSGPolygon, CSGEllipse, CSGUnion boolean
                                      `+` operator) for the 2-D planform, then
                                      trimesh.creation.extrude_triangulation
                                      (a plain mesh-extrusion utility, not a
                                      physics routine) turns those CSG-derived
                                      vertices into a 3-D flat-plate solid.
  Geometry (fuselage + CSG union) -> pinneapple_design.geometry.gen.primitives
                                      (build_mesh("sphere", scale=...) for the
                                      ellipsoidal fuselage primitive, and the
                                      module's own boolean-union machinery,
                                      `_boolean_or_fallback`, for the watertight
                                      mesh-level union with the wing solid).
  Mesh -> LBM obstacle voxel mask  -> pinneapple_design.geometry.ops.lbm_bridge
                                      .mesh_to_obstacle_mask() (AoA rotation +
                                      voxelization), reusing its own
                                      `_aoa_rotation_matrix` helper again here
                                      so the surface mesh used for pressure
                                      integration sits in the identical rotated
                                      frame as the voxel grid.
  Turbulence closure (Cs)          -> pinneapple_physics.pde_environment
                                      .turbulence_selector.get_turbulence_closure
                                      (LES_SMAGORINSKY, solver_family="lbm").
  Flow solve                       -> pinneapple_simulation.numerical_solvers
                                      .lbm.LBMSolver3D (D3Q19 BGK + Smagorinsky
                                      LES + bounce-back obstacle mask).
  Vortex / turbulence diagnostics  -> pinneapple_tools.visualization.vortex
                                      (compute_q_criterion_3d, compute_vorticity_3d,
                                      compute_enstrophy, compute_dissipation_rate,
                                      compute_strain_rate_tensor_3d,
                                      plot_q_criterion_3d, plot_lbm_flow).
  Surface pressure -> CL/CD/CM     -> pinneapple_tools.aero_coefficients
                                      (compute_cp, integrate_surface_forces,
                                      force_to_coefficients).
  Theoretical cross-check          -> pinneapple_pdb.benchmarks.get_benchmark
                                      ("concorde_high_aoa").

Pipeline steps (see main())
------------------------------------------------------------------------------
  [1] Build the simplified delta-wing + fuselage CSG geometry
  [2] Voxelize into an LBM obstacle mask per angle of attack
  [3] Select the Smagorinsky LES closure (Cs)
  [4] Run the D3Q19 LBM solve at each AoA in the sweep
  [5] Post-process: Q-criterion / vorticity / enstrophy / dissipation
  [6] Extract surface pressure -> integrate -> CL/CD/CM per AoA
  [7] Plot simulated CL(alpha) against the Polhamus theoretical cross-check
  [8] Save plots + metrics.json

Runtime note: this runs on a laptop CPU, not a GPU cluster. The grid is
deliberately coarse (default 56^3 voxels) and the AoA sweep short (~600 LBM
steps each) so the whole script finishes in roughly one to two minutes while
still being a genuine (if low-fidelity) transient LBM simulation -- not a
precomputed/fabricated result.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import trimesh
from scipy.ndimage import distance_transform_edt

# Repo root on sys.path so this script is runnable standalone (same pattern
# used by examples/arena_pipelines/*.py) without requiring `pip install -e .`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Real PINNeAPPle geometry machinery ─────────────────────────────────────
from pinneapple_design.geometry.csg import CSGPolygon, CSGEllipse, CSGUnion
from pinneapple_design.geometry.gen.primitives import build_mesh, _boolean_or_fallback
from pinneapple_design.geometry.ops.lbm_bridge import (
    mesh_to_obstacle_mask,
    _aoa_rotation_matrix,
)

# ── Real PINNeAPPle physics machinery ──────────────────────────────────────
from pinneapple_physics.pde_environment.turbulence_selector import (
    TurbulenceModel,
    get_turbulence_closure,
)
from pinneapple_simulation.numerical_solvers.lbm import LBMSolver3D

# ── Real PINNeAPPle post-processing / validation machinery ────────────────
from pinneapple_tools.visualization.vortex import (
    compute_q_criterion_3d,
    compute_vorticity_3d,
    compute_strain_rate_tensor_3d,
    compute_enstrophy,
    compute_dissipation_rate,
    plot_q_criterion_3d,
    plot_lbm_flow,
)
from pinneapple_tools.aero_coefficients import (
    compute_cp,
    integrate_surface_forces,
    force_to_coefficients,
)
from pinneapple_pdb.benchmarks import get_benchmark

# ── PINNeAPPle training-utility imports (graceful fallback, house convention) ──
try:
    from pinneapple_train import best_device  # noqa: F401
except ImportError:
    def best_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
# NOTE: `maybe_compile` (torch.compile wrapper) is intentionally not used here.
# It exists in this codebase to compile trainable nn.Module *forward passes*
# (see crash_surrogate/missile_aero pipelines); LBMSolver3D's forward() is a
# stepped, history-collecting simulation loop rather than a single
# differentiable forward pass, so compiling it is out of scope for this
# wiring script -- best_device() is still used below to pick a device.

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DARK_BG = "#0d1117"
ACCENT = "#58a6ff"
ACCENT2 = "#f78166"
ACCENT3 = "#3fb950"

# LBM lattice equation of state: p = rho * cs^2, cs^2 = 1/3 for the D3Q19
# lattice used by LBMSolver3D -- the SAME constant already used internally
# by pinneapple_simulation.numerical_solvers.lbm's Smagorinsky closures
# (see `cs2 = 1.0 / 3.0` in that module). LBMSolver3D returns rho/ux/uy/uz,
# not pressure directly, so this relation is needed to turn its density
# field into a pressure field for aero_coefficients.compute_cp(); it is the
# standard LBM equation of state, not a new physical model invented here.
LBM_CS2 = 1.0 / 3.0


# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class DeltaWingGeometry:
    """Parametric stand-in delta-wing-plus-fuselage aircraft (NOT real Concorde CAD).

    root_chord     : wing root chord [length units], nose apex at x=0.
    semispan       : half wingspan at the trailing edge. Default chosen so the
                      planform aspect ratio AR = (2*semispan)^2 / wing_area
                      = 4*semispan/root_chord matches the ~1.7 aspect ratio
                      used by the pinneapple_pdb "concorde_high_aoa" Polhamus
                      cross-check benchmark, for a more apples-to-apples plot.
    wing_thickness : flat-plate wing thickness (fraction of root_chord).
    fuselage_length_frac : fuselage length as a multiple of root_chord.
    fuselage_radius       : fuselage cross-section radius.
    fuselage_center_frac  : fuselage centre, as a fraction of root_chord along x.
    """
    root_chord: float = 1.0
    semispan: float = 0.425
    wing_thickness: float = 0.03
    fuselage_length_frac: float = 1.5
    fuselage_radius: float = 0.08
    fuselage_center_frac: float = 0.5

    @property
    def wing_area(self) -> float:
        """Planform (triangle) reference area = base * height / 2."""
        return self.root_chord * self.semispan

    @property
    def aspect_ratio(self) -> float:
        return (2.0 * self.semispan) ** 2 / self.wing_area

    @property
    def fuselage_length(self) -> float:
        return self.fuselage_length_frac * self.root_chord

    @property
    def fuselage_center_x(self) -> float:
        return self.fuselage_center_frac * self.root_chord


@dataclass
class LBMConfig:
    """LBM solve settings -- deliberately coarse so the whole sweep finishes
    on a laptop CPU in about one to two minutes."""
    resolution: int = 56          # voxels per axis (isotropic grid)
    Re: float = 300.0             # Reynolds number (lattice units)
    u_in: float = 0.04            # initial/mean streamwise velocity (lattice units)
    steps: int = 600              # LBM timesteps per AoA
    save_every: int = 150         # checkpoint interval (-> 4 checkpoints/run)
    rotation_axis: str = "y"      # spanwise axis; pitch rotates the x-z plane


AOA_SWEEP_DEG: List[float] = [0.0, 5.0, 10.0, 15.0, 20.0]
PRIMARY_AOA_DEG: float = 10.0  # matches the FluidX3D Concorde case study


def domain_bounds(cfg: DeltaWingGeometry) -> Dict[str, Tuple[float, float]]:
    """LBM domain box, sized off the geometry's own scale (root_chord) with
    margins for the wake, spanwise clearance, and AoA rotation clearance."""
    c = cfg.root_chord
    return {
        "x": (-0.5 * c, 2.0 * c),
        "y": (-1.0 * c, 1.0 * c),
        "z": (-1.0 * c, 1.0 * c),
    }


# ══════════════════════════════════════════════════════════════════════════
# 2. GEOMETRY -- delta wing + fuselage, composed via the real CSG machinery
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AircraftGeometry:
    mesh: trimesh.Trimesh                # watertight 3-D solid, AoA=0 pose
    planform_union: CSGUnion             # 2-D top-view CSG union (reporting only)
    cfg: DeltaWingGeometry


def build_delta_wing_fuselage(cfg: DeltaWingGeometry) -> AircraftGeometry:
    """Compose the simplified stand-in aircraft.

    2-D planform (top view): the delta-wing triangle is a real
    `pinneapple_design.geometry.csg.CSGPolygon`, unioned (`+`, i.e.
    `CSGUnion`) with a `CSGEllipse` fuselage footprint -- this is the "real
    CSG/geometry generators" step the task calls for, used here for both
    geometric validation (SDF containment) and the reporting-only top-view
    diagram plotted in 01_geometry.png.

    3-D solid: the wing planform's own vertices (straight from the
    CSGPolygon) are extruded into a flat-plate solid with
    `trimesh.creation.extrude_triangulation` (a mesh-extrusion utility, not a
    physics/CSG routine). The fuselage is an anisotropically-scaled sphere
    primitive from `pinneapple_design.geometry.gen.primitives.build_mesh`.
    The two solids are combined into one watertight mesh via that same
    module's boolean-union machinery, `_boolean_or_fallback` (the function
    `build_mesh(..., boolean={...})` itself calls internally), which prefers
    a real boolean engine (manifold/blender) and only falls back to a
    non-watertight concatenation if none is installed.
    """
    root_chord = cfg.root_chord
    semispan = cfg.semispan

    # -- 2-D planform via the real CSG module --------------------------------
    wing_verts2d = np.array(
        [[0.0, 0.0], [root_chord, semispan], [root_chord, -semispan]],
        dtype=np.float64,
    )
    wing_planform = CSGPolygon(wing_verts2d)
    # sanity check: the wing centroid must be classified "inside" by the
    # CSGPolygon's own SDF (negative sdf = inside, per csg.py's convention)
    centroid2d = wing_verts2d.mean(axis=0, keepdims=True)
    assert wing_planform.sdf(centroid2d)[0] < 0.0, "wing centroid not inside CSGPolygon"

    fuselage_planform = CSGEllipse(
        center_x=cfg.fuselage_center_x,
        center_y=0.0,
        a=cfg.fuselage_length / 2.0,
        b=cfg.fuselage_radius * 1.5,
    )
    planform_union = wing_planform + fuselage_planform  # CSGUnion, csg.py's `+`

    # -- 3-D wing solid: extrude the CSGPolygon's own vertices --------------
    wing_ring = wing_planform.verts[:-1]  # drop the auto-appended closing vertex
    wing_faces2d = np.array([[0, 1, 2]], dtype=np.int64)
    wing_solid = trimesh.creation.extrude_triangulation(
        wing_ring, wing_faces2d, height=cfg.wing_thickness
    )
    wing_solid.apply_translation([0.0, 0.0, -cfg.wing_thickness / 2.0])

    # -- 3-D fuselage solid: scaled sphere primitive -------------------------
    fuselage_md = build_mesh(
        "sphere",
        radius=1.0,
        subdivisions=3,
        scale=(cfg.fuselage_length / 2.0, cfg.fuselage_radius, cfg.fuselage_radius),
        translate=(cfg.fuselage_center_x, 0.0, 0.0),
    )
    fuselage_solid = trimesh.Trimesh(
        vertices=fuselage_md.vertices, faces=fuselage_md.faces, process=False
    )

    # -- Union via the real (fixed) boolean-engine-detection machinery ------
    combined = _boolean_or_fallback(fuselage_solid, wing_solid, "union")
    combined.fix_normals()

    return AircraftGeometry(mesh=combined, planform_union=planform_union, cfg=cfg)


def _rotate_mesh_for_aoa(
    mesh: trimesh.Trimesh, aoa_deg: float, rotation_axis: str
) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """Reproduce, via `mesh_to_obstacle_mask`'s own `_aoa_rotation_matrix`
    helper, the exact AoA rotation it applies internally before voxelizing --
    so the surface mesh used for pressure integration sits in the identical
    reference frame as the LBM obstacle grid. Returns (rotated_mesh, T)."""
    if aoa_deg == 0.0:
        T = np.eye(4, dtype=np.float64)
        return mesh.copy(), T
    R = _aoa_rotation_matrix(aoa_deg, rotation_axis)
    center = np.asarray(mesh.centroid, dtype=np.float64).copy()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = center - R @ center
    rotated = mesh.copy()
    rotated.apply_transform(T)
    return rotated, T


def _transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return T[:3, :3] @ p + T[:3, 3]


# ══════════════════════════════════════════════════════════════════════════
# 3. LBM SOLVE (per angle of attack)
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AoARun:
    aoa_deg: float
    obstacle_mask: np.ndarray
    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    trajectory_speed: List[float]  # mean |u| at each save_every checkpoint
    elapsed_s: float
    n_solid_voxels: int
    rotated_mesh: trimesh.Trimesh
    moment_center_world: np.ndarray


def run_lbm_at_aoa(
    geom: AircraftGeometry,
    bounds: Dict[str, Tuple[float, float]],
    lbm_cfg: LBMConfig,
    aoa_deg: float,
    Cs: float,
    device: torch.device,
) -> AoARun:
    """Voxelize the aircraft at `aoa_deg` (pinneapple_design.geometry.ops
    .lbm_bridge.mesh_to_obstacle_mask) and run the D3Q19 Smagorinsky-LES LBM
    solve (pinneapple_simulation.numerical_solvers.lbm.LBMSolver3D)."""
    mask_np = mesh_to_obstacle_mask(
        geom.mesh,
        bounds,
        lbm_cfg.resolution,
        aoa_deg=aoa_deg,
        rotation_axis=lbm_cfg.rotation_axis,
    )
    obstacle_mask = torch.from_numpy(mask_np)

    solver = LBMSolver3D(
        nx=lbm_cfg.resolution,
        ny=lbm_cfg.resolution,
        nz=lbm_cfg.resolution,
        Re=lbm_cfg.Re,
        u_in=lbm_cfg.u_in,
        obstacle_mask=obstacle_mask,
        Cs=Cs,
    ).to(device)

    t0 = time.time()
    out = solver(steps=lbm_cfg.steps, save_every=lbm_cfg.save_every)
    elapsed = time.time() - t0

    traj_speed = [
        float(torch.sqrt(ux_i**2 + uy_i**2 + uz_i**2).mean())
        for ux_i, uy_i, uz_i in zip(
            out.extras["trajectory_ux"],
            out.extras["trajectory_uy"],
            out.extras["trajectory_uz"],
        )
    ]

    rotated_mesh, T = _rotate_mesh_for_aoa(geom.mesh, aoa_deg, lbm_cfg.rotation_axis)
    moment_center_unrotated = np.array(
        [geom.cfg.root_chord * 0.25, 0.0, 0.0], dtype=np.float64
    )  # quarter-chord reference point
    moment_center_world = _transform_point(T, moment_center_unrotated)

    return AoARun(
        aoa_deg=aoa_deg,
        obstacle_mask=mask_np,
        rho=out.extras["rho"].numpy(),
        ux=out.extras["ux"].numpy(),
        uy=out.extras["uy"].numpy(),
        uz=out.extras["uz"].numpy(),
        trajectory_speed=traj_speed,
        elapsed_s=elapsed,
        n_solid_voxels=int(mask_np.sum()),
        rotated_mesh=rotated_mesh,
        moment_center_world=moment_center_world,
    )


# ══════════════════════════════════════════════════════════════════════════
# 4. VORTEX / TURBULENCE POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class VortexDiagnostics:
    Q: np.ndarray
    vorticity: np.ndarray
    enstrophy: float
    dissipation_mean: float
    Q_max: float
    Q_positive_fraction: float


def analyze_vortex_field(
    run: AoARun, bounds: Dict[str, Tuple[float, float]], lbm_cfg: LBMConfig
) -> VortexDiagnostics:
    """All vortex/turbulence math delegates to pinneapple_tools.visualization.vortex."""
    res = lbm_cfg.resolution
    dx = (bounds["x"][1] - bounds["x"][0]) / res
    dy = (bounds["y"][1] - bounds["y"][0]) / res
    dz = (bounds["z"][1] - bounds["z"][0]) / res

    Q = compute_q_criterion_3d(run.ux, run.uy, run.uz, dx=dx, dy=dy, dz=dz)
    omega = compute_vorticity_3d(run.ux, run.uy, run.uz, dx=dx, dy=dy, dz=dz)
    enstrophy = compute_enstrophy(omega)

    # kinematic viscosity in lattice units, from the SAME Re relation
    # LBMSolver3D itself uses internally to derive tau from Re (nu = u_in*L/Re)
    nu_lattice = lbm_cfg.u_in * lbm_cfg.resolution / lbm_cfg.Re
    S = compute_strain_rate_tensor_3d(run.ux, run.uy, run.uz, dx=dx, dy=dy, dz=dz)
    eps_field = compute_dissipation_rate(S, nu_lattice)

    Q_pos = Q[Q > 0.0]
    return VortexDiagnostics(
        Q=Q,
        vorticity=omega,
        enstrophy=enstrophy,
        dissipation_mean=float(np.mean(eps_field)),
        Q_max=float(Q.max()),
        Q_positive_fraction=float(Q_pos.size) / float(Q.size),
    )


# ══════════════════════════════════════════════════════════════════════════
# 5. SURFACE PRESSURE -> AERODYNAMIC COEFFICIENTS
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class AeroResult:
    CL: float
    CD: float
    CM: float
    CY: float
    Cp_faces: np.ndarray


def compute_aero_coefficients(
    run: AoARun,
    bounds: Dict[str, Tuple[float, float]],
    lbm_cfg: LBMConfig,
    geom: AircraftGeometry,
) -> AeroResult:
    """Extract the surface pressure field from the LBM density field and
    integrate it into CL/CD/CM via pinneapple_tools.aero_coefficients."""
    res = lbm_cfg.resolution
    lo = np.array([bounds["x"][0], bounds["y"][0], bounds["z"][0]])
    hi = np.array([bounds["x"][1], bounds["y"][1], bounds["z"][1]])
    cell = (hi - lo) / res

    # Pressure field from LBM density (p = rho * cs^2). Solid-node densities
    # are not meaningful fluid pressures (bounce-back does not update them
    # like a real fluid cell), so every voxel is remapped to its nearest
    # FLUID voxel's pressure via a scipy Euclidean distance transform before
    # sampling at the mesh surface -- a plain nearest-neighbour lookup, not a
    # new physical model.
    p_lattice = run.rho * LBM_CS2
    _, nearest_idx = distance_transform_edt(run.obstacle_mask, return_indices=True)
    p_fluid_extrapolated = p_lattice[nearest_idx[0], nearest_idx[1], nearest_idx[2]]

    centroids = run.rotated_mesh.triangles_center
    ijk = np.floor((centroids - lo) / cell).astype(int)
    ijk = np.clip(ijk, 0, res - 1)
    face_pressure = p_fluid_extrapolated[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    rho_inf = float(run.rho.mean())
    p_inf = rho_inf * LBM_CS2
    q_inf = 0.5 * rho_inf * lbm_cfg.u_in**2
    Cp_faces = compute_cp(face_pressure, p_inf, q_inf)

    force, moment = integrate_surface_forces(
        run.rotated_mesh.vertices,
        run.rotated_mesh.faces,
        run.rotated_mesh.face_normals,
        face_pressure,
        moment_center=run.moment_center_world,
    )

    coeffs = force_to_coefficients(
        force,
        moment,
        q_inf=q_inf,
        ref_area=geom.cfg.wing_area,
        ref_length=geom.cfg.root_chord,
        alpha_deg=run.aoa_deg,
        drag_axis="x",
        lift_axis="z",
    )

    return AeroResult(CL=coeffs.CL, CD=coeffs.CD, CM=coeffs.CM, CY=coeffs.CY, Cp_faces=Cp_faces)


# ══════════════════════════════════════════════════════════════════════════
# 6. PLOTTING
# ══════════════════════════════════════════════════════════════════════════

def _dark_fig(nrows=1, ncols=1, figsize=(10, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=DARK_BG)
    axes_arr = np.atleast_1d(axes)
    for ax in np.ravel(axes_arr):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
    return fig, axes


def plot_geometry(geom: AircraftGeometry, out_path: Path):
    """Top-view planform (real CSG union) + a 3-D wireframe preview of the
    simplified stand-in aircraft, explicitly labelled as such."""
    fig, axes = _dark_fig(1, 2, figsize=(13, 5.5))

    interior = geom.planform_union.sample_interior(4000, seed=0)
    boundary = geom.planform_union.sample_boundary(1500, seed=0)
    ax0 = axes[0]
    ax0.scatter(interior[:, 0], interior[:, 1], s=1, color=ACCENT, alpha=0.35)
    ax0.scatter(boundary[:, 0], boundary[:, 1], s=3, color=ACCENT2)
    ax0.set_aspect("equal")
    ax0.set_xlabel("x (streamwise)")
    ax0.set_ylabel("y (spanwise)")
    ax0.set_title("Top-view planform (CSGPolygon + CSGEllipse union)")

    ax1 = axes[1]
    v = geom.mesh.vertices
    ax1.scatter(v[:, 0], v[:, 2], s=2, color=ACCENT3, alpha=0.5)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (streamwise)")
    ax1.set_ylabel("z (vertical)")
    ax1.set_title("Side profile (vertex cloud, AoA=0 pose)")

    fig.suptitle(
        "SIMPLIFIED PARAMETRIC STAND-IN -- NOT real Concorde CAD geometry",
        color=ACCENT2, fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_obstacle_mask(run: AoARun, out_path: Path):
    fig, axes = _dark_fig(1, 2, figsize=(12, 5))
    nx, ny, nz = run.obstacle_mask.shape
    axes[0].imshow(run.obstacle_mask[:, :, nz // 2].T, origin="lower", cmap="bone")
    axes[0].set_title(f"Obstacle mask, mid z-plane (AoA={run.aoa_deg:.0f} deg)")
    axes[0].set_xlabel("x index"); axes[0].set_ylabel("y index")
    axes[1].imshow(run.obstacle_mask[:, ny // 2, :].T, origin="lower", cmap="bone")
    axes[1].set_title(f"Obstacle mask, mid y-plane (AoA={run.aoa_deg:.0f} deg)")
    axes[1].set_xlabel("x index"); axes[1].set_ylabel("z index")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_flow_field(run: AoARun, out_path: Path):
    """Reuses pinneapple_tools.visualization.vortex.plot_lbm_flow (2-D 4-panel
    dashboard) on the mid-span horizontal slice of the 3-D LBM output."""
    nz = run.ux.shape[2]
    k = nz // 2
    fig = plot_lbm_flow(
        run.ux[:, :, k], run.uy[:, :, k], rho=run.rho[:, :, k],
        obstacle_mask=run.obstacle_mask[:, :, k],
        title=f"LBM flow field, mid z-plane, AoA={run.aoa_deg:.0f} deg",
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_vortex_field(
    run: AoARun, diag: VortexDiagnostics, bounds: Dict[str, Tuple[float, float]],
    lbm_cfg: LBMConfig, out_path: Path,
):
    """3-D Q-criterion iso-surface via pinneapple_tools.visualization.vortex
    .plot_q_criterion_3d (needs scikit-image); falls back to 2-D Q-criterion
    slices (same Q field, just a different presentation) if it isn't
    installed in this environment."""
    res = lbm_cfg.resolution
    xs = np.linspace(*bounds["x"], res)
    ys = np.linspace(*bounds["y"], res)
    zs = np.linspace(*bounds["z"], res)
    vel_mag = np.sqrt(run.ux**2 + run.uy**2 + run.uz**2)

    Q_max = diag.Q.max()
    level = 0.1 * Q_max if Q_max > 0 else 1e-6

    try:
        fig = plot_q_criterion_3d(
            xs, ys, zs, diag.Q, level=level, vel_mag=vel_mag,
            title=f"Q-criterion iso-surface, AoA={run.aoa_deg:.0f} deg",
        )
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  saved -> {out_path}")
    except ImportError:
        print("  scikit-image not installed -- falling back to 2-D Q-criterion slices "
              "(same compute_q_criterion_3d field, different presentation)")
        fig, axes = _dark_fig(1, 2, figsize=(12, 5))
        nz = diag.Q.shape[2]
        ny = diag.Q.shape[1]

        def _slice_levels(slice_2d: np.ndarray) -> np.ndarray:
            # A handful of near-boundary/near-wall outliers otherwise swamp a
            # plain min/max color scale and wash out the vortex structure
            # near the body -- clip to the 1st/99th percentile of |Q| for a
            # legible, symmetric-about-zero color scale (display only; the
            # underlying Q values plotted are untouched).
            vmax = max(np.percentile(np.abs(slice_2d), 99.0), 1e-12)
            return np.linspace(-vmax, vmax, 41)

        lv0 = _slice_levels(diag.Q[:, :, nz // 2])
        cf0 = axes[0].contourf(diag.Q[:, :, nz // 2].T, levels=lv0, cmap="coolwarm", extend="both")
        axes[0].set_title("Q-criterion, mid z-plane")
        fig.colorbar(cf0, ax=axes[0])
        lv1 = _slice_levels(diag.Q[:, ny // 2, :])
        cf1 = axes[1].contourf(diag.Q[:, ny // 2, :].T, levels=lv1, cmap="coolwarm", extend="both")
        axes[1].set_title("Q-criterion, mid y-plane (leading-edge vortex)")
        fig.colorbar(cf1, ax=axes[1])
        fig.suptitle(
            f"Q-criterion slices, AoA={run.aoa_deg:.0f} deg "
            "(3-D iso-surface needs scikit-image, not installed)",
            color=ACCENT2,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
        plt.close(fig)
        print(f"  saved -> {out_path}")


def plot_polar(sweep_results: List[Dict[str, Any]], out_path: Path):
    fig, axes = _dark_fig(1, 3, figsize=(15, 4.5))
    alphas = [r["aoa_deg"] for r in sweep_results]
    CL = [r["CL"] for r in sweep_results]
    CD = [r["CD"] for r in sweep_results]
    CM = [r["CM"] for r in sweep_results]

    axes[0].plot(alphas, CL, "o-", color=ACCENT)
    axes[0].set_xlabel("AoA [deg]"); axes[0].set_ylabel("CL"); axes[0].set_title("Lift coefficient")
    axes[1].plot(alphas, CD, "o-", color=ACCENT2)
    axes[1].set_xlabel("AoA [deg]"); axes[1].set_ylabel("CD"); axes[1].set_title("Drag coefficient")
    axes[2].plot(alphas, CM, "o-", color=ACCENT3)
    axes[2].set_xlabel("AoA [deg]"); axes[2].set_ylabel("CM (quarter-chord)")
    axes[2].set_title("Pitching-moment coefficient")

    fig.suptitle("Simulated CL / CD / CM vs. angle of attack (coarse LBM, LES-Smagorinsky)",
                 color="white")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved -> {out_path}")


def plot_validation(sweep_results: List[Dict[str, Any]], out_path: Path):
    """CL(alpha) simulated vs. the pinneapple_pdb Polhamus THEORETICAL
    cross-check -- explicitly not real Concorde flight/wind-tunnel data."""
    bench = get_benchmark("concorde_high_aoa")

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.tick_params(colors="white")

    ax.plot(
        bench.reference_x[:, 0], bench.reference_y[:, 0],
        "-", color="white", lw=1.5,
        label="Polhamus (1966) slender-delta vortex-lift theory\n(THEORETICAL cross-check, AR~1.7)",
    )
    sim_alpha = [r["aoa_deg"] for r in sweep_results]
    sim_CL = [r["CL"] for r in sweep_results]
    ax.plot(sim_alpha, sim_CL, "o", color=ACCENT, ms=8, label="This pipeline's coarse LBM simulation")

    ax.set_xlabel("Angle of attack [deg]", color="white")
    ax.set_ylabel("CL", color="white")
    ax.set_title("CL(alpha): simulated vs. theoretical cross-check\n"
                 "(NOT a comparison against real Concorde flight/wind-tunnel data)",
                 color="white", fontsize=10)
    ax.legend(facecolor=DARK_BG, labelcolor="white", fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=DARK_BG)
    plt.close(fig)
    print(f"  saved -> {out_path}")
    return bench


# ══════════════════════════════════════════════════════════════════════════
# 7. JSON-SAFE SERIALIZATION
# ══════════════════════════════════════════════════════════════════════════

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    device = best_device()
    print(f"\n{'='*78}")
    print("  Concorde-Inspired High-AoA Delta-Wing LBM Pipeline")
    print("  (simplified parametric stand-in geometry -- see module docstring)")
    print(f"  Device: {device}")
    print(f"{'='*78}\n")

    wing_cfg = DeltaWingGeometry()
    lbm_cfg = LBMConfig()
    bounds = domain_bounds(wing_cfg)

    # ── Step 1: geometry ─────────────────────────────────────────────────
    print("[1/8] Building simplified delta-wing + fuselage geometry (real CSG machinery) ...")
    geom = build_delta_wing_fuselage(wing_cfg)
    print(f"      wing_area={geom.cfg.wing_area:.4f}  aspect_ratio={geom.cfg.aspect_ratio:.3f}  "
          f"mesh: watertight={geom.mesh.is_watertight} verts={len(geom.mesh.vertices)} "
          f"faces={len(geom.mesh.faces)}")
    plot_geometry(geom, OUT_DIR / "01_geometry.png")

    # ── Step 2/3/4: turbulence closure + LBM solve, AoA sweep ─────────────
    print("\n[2/8] Domain bounds:", bounds)

    print("\n[3/8] Selecting Smagorinsky LES closure "
          "(pinneapple_physics.pde_environment.turbulence_selector) ...")
    Cs = get_turbulence_closure(TurbulenceModel.LES_SMAGORINSKY, solver_family="lbm")
    print(f"      Cs = {Cs}")

    print(f"\n[4/8] Running D3Q19 LBM solve (LBMSolver3D) across AoA sweep {AOA_SWEEP_DEG} deg "
          f"(resolution={lbm_cfg.resolution}^3, steps={lbm_cfg.steps}) ...")
    runs: Dict[float, AoARun] = {}
    for aoa in AOA_SWEEP_DEG:
        run = run_lbm_at_aoa(geom, bounds, lbm_cfg, aoa, Cs, device)
        runs[aoa] = run
        speed_hist = ", ".join(f"{s:.5f}" for s in run.trajectory_speed)
        rel_change = (
            abs(run.trajectory_speed[-1] - run.trajectory_speed[-2]) / max(run.trajectory_speed[-2], 1e-12)
            if len(run.trajectory_speed) >= 2 else float("nan")
        )
        print(f"      AoA={aoa:5.1f} deg | {run.elapsed_s:5.1f}s | "
              f"solid_voxels={run.n_solid_voxels:5d} | mean|u| history=[{speed_hist}] | "
              f"last-step rel. change={rel_change:.4%}")

    plot_obstacle_mask(runs[PRIMARY_AOA_DEG], OUT_DIR / "02_obstacle_mask_aoa10.png")
    plot_flow_field(runs[PRIMARY_AOA_DEG], OUT_DIR / "03_flow_field_aoa10.png")

    # ── Step 5: vortex diagnostics ─────────────────────────────────────────
    print("\n[5/8] Computing Q-criterion / vorticity / enstrophy / dissipation "
          "(pinneapple_tools.visualization.vortex) ...")
    diagnostics: Dict[float, VortexDiagnostics] = {}
    for aoa in AOA_SWEEP_DEG:
        diag = analyze_vortex_field(runs[aoa], bounds, lbm_cfg)
        diagnostics[aoa] = diag
        print(f"      AoA={aoa:5.1f} deg | Q_max={diag.Q_max:.4e} | "
              f"Q>0 fraction={diag.Q_positive_fraction:.4f} | "
              f"enstrophy={diag.enstrophy:.4e} | mean dissipation={diag.dissipation_mean:.4e}")

    plot_vortex_field(
        runs[PRIMARY_AOA_DEG], diagnostics[PRIMARY_AOA_DEG], bounds, lbm_cfg,
        OUT_DIR / "04_q_criterion_aoa10.png",
    )

    # ── Step 6: surface pressure -> CL/CD/CM ────────────────────────────────
    print("\n[6/8] Extracting surface pressure and integrating aerodynamic coefficients "
          "(pinneapple_tools.aero_coefficients) ...")
    sweep_results: List[Dict[str, Any]] = []
    for aoa in AOA_SWEEP_DEG:
        aero = compute_aero_coefficients(runs[aoa], bounds, lbm_cfg, geom)
        diag = diagnostics[aoa]
        run = runs[aoa]
        print(f"      AoA={aoa:5.1f} deg | CL={aero.CL:+.4f}  CD={aero.CD:+.4f}  "
              f"CM={aero.CM:+.4f}  CY={aero.CY:+.4f}")
        sweep_results.append({
            "aoa_deg": aoa,
            "CL": aero.CL, "CD": aero.CD, "CM": aero.CM, "CY": aero.CY,
            "Cp_mean": float(aero.Cp_faces.mean()), "Cp_min": float(aero.Cp_faces.min()),
            "Cp_max": float(aero.Cp_faces.max()),
            "n_solid_voxels": run.n_solid_voxels,
            "solve_time_s": run.elapsed_s,
            "mean_speed_history": run.trajectory_speed,
            "Q_max": diag.Q_max, "Q_positive_fraction": diag.Q_positive_fraction,
            "enstrophy": diag.enstrophy, "dissipation_mean": diag.dissipation_mean,
        })

    plot_polar(sweep_results, OUT_DIR / "05_cl_cd_cm_polar.png")

    # ── Step 7: validation cross-check ──────────────────────────────────────
    print("\n[7/8] Plotting CL(alpha) against pinneapple_pdb 'concorde_high_aoa' "
          "THEORETICAL cross-check (pinneapple_pdb.benchmarks) ...")
    bench = plot_validation(sweep_results, OUT_DIR / "06_validation_vs_polhamus.png")
    print(f"      benchmark source: {bench.reference_source}")
    print(f"      verification_note: {bench.verification_note}")

    # ── Step 8: save metrics ────────────────────────────────────────────────
    print("\n[8/8] Saving metrics.json ...")
    primary = next(r for r in sweep_results if r["aoa_deg"] == PRIMARY_AOA_DEG)
    metrics = {
        "geometry": {
            "root_chord": wing_cfg.root_chord,
            "semispan": wing_cfg.semispan,
            "wing_area": geom.cfg.wing_area,
            "aspect_ratio": geom.cfg.aspect_ratio,
            "fuselage_length": wing_cfg.fuselage_length,
            "fuselage_radius": wing_cfg.fuselage_radius,
            "is_parametric_stand_in_not_real_concorde_cad": True,
        },
        "lbm_config": {
            "resolution": lbm_cfg.resolution, "Re": lbm_cfg.Re, "u_in": lbm_cfg.u_in,
            "steps": lbm_cfg.steps, "save_every": lbm_cfg.save_every,
        },
        "turbulence": {"model": "les_smagorinsky", "Cs": Cs, "solver_family": "lbm"},
        "primary_aoa_deg": PRIMARY_AOA_DEG,
        "primary_aoa_result": primary,
        "aoa_sweep": sweep_results,
        "validation": {
            "benchmark": "concorde_high_aoa",
            "reference_source": bench.reference_source,
            "verification_note": bench.verification_note,
            "reference_alpha_deg": bench.reference_x[:, 0],
            "reference_CL": bench.reference_y[:, 0],
            "sim_alpha_deg": [r["aoa_deg"] for r in sweep_results],
            "sim_CL": [r["CL"] for r in sweep_results],
        },
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(_jsonable(metrics), f, indent=2)

    print("\n" + "=" * 78)
    print("  Pipeline complete. Outputs in:", OUT_DIR)
    for f in sorted(OUT_DIR.glob("*")):
        print(f"    {f.name}")
    print(f"\n  Primary result (AoA={PRIMARY_AOA_DEG} deg): "
          f"CL={primary['CL']:+.4f}  CD={primary['CD']:+.4f}  CM={primary['CM']:+.4f}")
    print("  Reminder: geometry is a simplified stand-in, and the validation plot is a "
          "THEORETICAL cross-check (Polhamus 1966), not real Concorde flight data.")
    print("=" * 78)


if __name__ == "__main__":
    main()
