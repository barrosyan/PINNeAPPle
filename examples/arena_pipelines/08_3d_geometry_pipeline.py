"""08_3d_geometry_pipeline.py — Arena pipeline with 3D geometries.

Shows how to run the full Arena benchmark pipeline on 3D physics problems
by combining:
  - ``pinneapple_design`` geometry domains (LidDrivenCavity, Channel, Pipe, Box, Sphere)
  - ``define_problem()`` for equation specification
  - ``ArenaProblem`` subclassing with geometry-aware data generation
  - Standard Arena training (VanillaPINN, SIREN, ModifiedMLP on 3D input)
  - 2D cross-section visualization of 3D fields

The ``geometry`` key in the config (or the constructor) selects:
  - Built-in domains: "lid_driven_cavity_3d", "channel_3d", "pipe_flow_3d"
  - Parametric primitives: "box", "sphere", "cylinder" (via pinneapple_design)
  - External STL: "stl" + a path

Cases
-----
  1. Laplace / Poisson on a unit box         (analytical: sin(πx)sin(πy)sin(πz))
  2. Laplace on a sphere                     (harmonic: u=xyz on ∂Ω)
  3. Steady heat equation in a 3D channel    (parabolic profile in cross-section)
  4. 3D lid-driven cavity Navier-Stokes      (geometry from domains3d registry)

Usage
-----
    python examples/arena_pipelines/08_3d_geometry_pipeline.py
    python examples/arena_pipelines/08_3d_geometry_pipeline.py --case box
    python examples/arena_pipelines/08_3d_geometry_pipeline.py --case sphere
    python examples/arena_pipelines/08_3d_geometry_pipeline.py --case channel
    python examples/arena_pipelines/08_3d_geometry_pipeline.py --case cavity
    python examples/arena_pipelines/08_3d_geometry_pipeline.py --epochs 2000
    python examples/arena_pipelines/08_3d_geometry_pipeline.py --case box --epochs 1000
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from pinneapple_arena import Arena, ArenaConfig, ArenaProblem, register_problem


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap(out):
    if torch.is_tensor(out):
        return out
    if hasattr(out, "y") and torch.is_tensor(out.y):
        return out.y
    if isinstance(out, dict):
        return torch.stack(list(out.values()), dim=-1)
    return out


def _grad(y, x, idx):
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y), create_graph=True
    )[0][:, idx:idx + 1]


def _laplacian_3d(u, xyz):
    """Compute Δu = u_xx + u_yy + u_zz via autograd."""
    u_x = _grad(u, xyz, 0)
    u_y = _grad(u, xyz, 1)
    u_z = _grad(u, xyz, 2)
    u_xx = _grad(u_x, xyz, 0)
    u_yy = _grad(u_y, xyz, 1)
    u_zz = _grad(u_z, xyz, 2)
    return u_xx + u_yy + u_zz


def _eval_slice_2d(xy_eval_3d: np.ndarray, n_grid: int) -> Tuple[np.ndarray, np.ndarray]:
    """Take an (N,3) 3D eval array and return a 2D z=0.5 cross-section grid for plotting.

    Returns (xy_slice, mask) where xy_slice is (M,3) with z=0.5.
    """
    z_fixed = 0.5
    gx = np.linspace(xy_eval_3d[:, 0].min(), xy_eval_3d[:, 0].max(), n_grid, dtype=np.float32)
    gy = np.linspace(xy_eval_3d[:, 1].min(), xy_eval_3d[:, 1].max(), n_grid, dtype=np.float32)
    GX, GY = np.meshgrid(gx, gy)
    z_col = np.full(GX.size, z_fixed, dtype=np.float32)
    return np.stack([GX.ravel(), GY.ravel(), z_col], axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# Case 1 — Laplace / Poisson on a unit box  [0,1]³
# ══════════════════════════════════════════════════════════════════════════════

class PoissonBox3D(ArenaProblem):
    """3D Poisson: Δu = f  on [0,1]³,  u=0 on ∂Ω.

    Manufactured solution: u(x,y,z) = sin(πx)·sin(πy)·sin(πz)
    Source term:           f = -3π²·sin(πx)·sin(πy)·sin(πz)
    """
    name = "poisson_box_3d"
    description = "3D Poisson on unit box — u=sin(πx)sin(πy)sin(πz)"
    domain = "Elliptic 3D"
    input_dim = 3
    output_dim = 1

    def analytical(self, x, y, z=None, **kw):
        if z is None:
            z = np.zeros_like(x)
        return {"u": np.sin(math.pi * x) * np.sin(math.pi * y) * np.sin(math.pi * z)}

    def _source(self, xyz: np.ndarray) -> np.ndarray:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        return -3.0 * math.pi**2 * np.sin(math.pi * x) * np.sin(math.pi * y) * np.sin(math.pi * z)

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, **kw):
        xyz = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xyz))
        lap = _laplacian_3d(u, xyz)
        f = torch.tensor(
            self._source(xy_int.detach().cpu().numpy()),
            dtype=torch.float32, device=xyz.device
        ).unsqueeze(1)
        pde_loss = ((lap - f) ** 2).mean()
        bc_loss = ((_unwrap(net(xy_bc)) - uv_bc) ** 2).mean()
        return pde_loss, bc_loss

    def supervised_data(self, n_train=500, n_bc=600, grid_n=20, **kw):
        rng = np.random.default_rng(42)
        # Interior
        xyz = rng.uniform(0, 1, (n_train, 3)).astype(np.float32)
        f = self.analytical(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        Y_int = f["u"].reshape(-1, 1).astype(np.float32)
        # Boundary: all six faces
        n_per_face = max(n_bc // 6, 1)
        bc_parts = []
        bc_y_parts = []
        for d in range(3):
            for val in [0.0, 1.0]:
                pts = rng.uniform(0, 1, (n_per_face, 3)).astype(np.float32)
                pts[:, d] = val
                bc_parts.append(pts)
                args = [pts[:, 0], pts[:, 1], pts[:, 2]]
                sol = self.analytical(*args)
                bc_y_parts.append(sol["u"].reshape(-1, 1).astype(np.float32))
        xy_bc = np.concatenate(bc_parts, axis=0)
        Y_bc = np.concatenate(bc_y_parts, axis=0)
        # Evaluation: 2D cross-section at z=0.5 for visualization
        g = np.linspace(0, 1, grid_n, dtype=np.float32)
        GX, GY = np.meshgrid(g, g)
        xy_eval = np.stack([GX.ravel(), GY.ravel(),
                             np.full(GX.size, 0.5, dtype=np.float32)], axis=1)
        sol_e = self.analytical(xy_eval[:, 0], xy_eval[:, 1], xy_eval[:, 2])
        Y_eval = sol_e["u"].reshape(-1, 1).astype(np.float32)
        return xyz, Y_int, xy_bc, Y_bc, xy_eval, Y_eval, ["u"]


register_problem(PoissonBox3D())


# ══════════════════════════════════════════════════════════════════════════════
# Case 2 — Laplace on a Sphere  (harmonic: u = x·y·z on ∂Ω)
# ══════════════════════════════════════════════════════════════════════════════

class LaplaceSpherePINN(ArenaProblem):
    """Laplace equation Δu=0 inside the unit sphere.

    Boundary condition: u = x·y·z (cubic harmonic).
    The PDE residual is computed in Cartesian coordinates.
    Training points are sampled from the sphere interior via rejection.
    Evaluation is on the equatorial cross-section (z=0 plane).
    """
    name = "laplace_sphere_3d"
    description = "Laplace on unit sphere: Δu=0, u=xyz on ∂Ω"
    domain = "Elliptic 3D"
    input_dim = 3
    output_dim = 1

    @staticmethod
    def _sample_sphere_interior(n, rng):
        collected = []
        while sum(len(c) for c in collected) < n:
            pts = rng.uniform(-1, 1, (n * 4, 3)).astype(np.float32)
            mask = (pts ** 2).sum(axis=1) <= 1.0
            collected.append(pts[mask])
        return np.concatenate(collected, axis=0)[:n]

    @staticmethod
    def _sample_sphere_surface(n, rng):
        th = rng.uniform(0, 2 * math.pi, n).astype(np.float32)
        ph = np.arccos(rng.uniform(-1, 1, n).astype(np.float32))
        x = np.sin(ph) * np.cos(th)
        y = np.sin(ph) * np.sin(th)
        z = np.cos(ph)
        return np.stack([x, y, z], axis=1)

    def analytical(self, x, y, z=None, **kw):
        # Only an approximation on interior; exact on boundary
        if z is None:
            z = np.zeros_like(x)
        return {"u": x * y * z}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, **kw):
        xyz = xy_int.clone().requires_grad_(True)
        u = _unwrap(net(xyz))
        lap = _laplacian_3d(u, xyz)
        pde_loss = (lap ** 2).mean()
        bc_out = _unwrap(net(xy_bc))
        bc_loss = ((bc_out - uv_bc) ** 2).mean()
        return pde_loss, bc_loss

    def supervised_data(self, n_train=600, n_bc=400, grid_n=20, **kw):
        rng = np.random.default_rng(42)
        # Interior (Laplace → no analytical, zeros as placeholder targets)
        xyz = self._sample_sphere_interior(n_train, rng)
        Y_int = np.zeros((n_train, 1), dtype=np.float32)
        # Boundary: u = x·y·z
        pts_bc = self._sample_sphere_surface(n_bc, rng)
        Y_bc = (pts_bc[:, 0] * pts_bc[:, 1] * pts_bc[:, 2]).reshape(-1, 1).astype(np.float32)
        # Eval: equatorial plane z≈0, r≤1
        g = np.linspace(-1, 1, grid_n, dtype=np.float32)
        GX, GY = np.meshgrid(g, g)
        mask = (GX.ravel() ** 2 + GY.ravel() ** 2) <= 1.0
        xe = GX.ravel()[mask]
        ye = GY.ravel()[mask]
        xy_eval = np.stack([xe, ye, np.zeros_like(xe)], axis=1)
        # Reference: harmonic xyz, z=0 → u=0 everywhere on equatorial plane
        Y_eval = np.zeros((len(xy_eval), 1), dtype=np.float32)
        return xyz, Y_int, pts_bc, Y_bc, xy_eval, Y_eval, ["u"]


register_problem(LaplaceSpherePINN())


# ══════════════════════════════════════════════════════════════════════════════
# Case 3 — Steady Heat in a 3D Rectangular Channel
# ══════════════════════════════════════════════════════════════════════════════

class HeatChannel3D(ArenaProblem):
    """Steady-state heat equation Δu=0 in a 3D channel [0,L]×[0,H]×[0,W].

    Boundary conditions (Dirichlet):
      - inlet  (x=0): u = parabolic profile T_in·(1 - (2y/H-1)²)·(1 - (2z/W-1)²)
      - outlet (x=L): u = T_out (cooled)
      - walls  (y,z faces): u = 0

    The manufactured solution is purely for illustration purposes.
    """
    name = "heat_channel_3d"
    description = "3D steady heat: Δu=0 in rectangular channel with inlet temperature"
    domain = "Diffusion 3D"
    input_dim = 3
    output_dim = 1

    def __init__(self, L=2.0, H=1.0, W=1.0, T_in=1.0, T_out=0.0):
        self.L, self.H, self.W = L, H, W
        self.T_in, self.T_out = T_in, T_out

    def _inlet_profile(self, y, z):
        yc = 2 * y / self.H - 1
        zc = 2 * z / self.W - 1
        return self.T_in * (1 - yc ** 2) * (1 - zc ** 2)

    def analytical(self, x, y, z=None, **kw):
        if z is None:
            z = np.zeros_like(x)
        # Simple linear blend x-direction × parabolic cross-section
        lin = 1.0 - x / self.L
        profile = self._inlet_profile(y, z)
        return {"T": lin * profile * self.T_in + (1 - lin) * self.T_out}

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, **kw):
        xyz = xy_int.clone().requires_grad_(True)
        T = _unwrap(net(xyz))
        lap = _laplacian_3d(T, xyz)
        pde_loss = (lap ** 2).mean()
        bc_loss = ((_unwrap(net(xy_bc)) - uv_bc) ** 2).mean()
        return pde_loss, bc_loss

    def supervised_data(self, n_train=600, n_bc=600, grid_n=20, **kw):
        rng = np.random.default_rng(42)
        L, H, W = self.L, self.H, self.W

        # Interior
        x = rng.uniform(0, L, n_train).astype(np.float32)
        y = rng.uniform(0, H, n_train).astype(np.float32)
        z = rng.uniform(0, W, n_train).astype(np.float32)
        xyz = np.stack([x, y, z], axis=1)
        sol = self.analytical(x, y, z)
        Y_int = sol["T"].reshape(-1, 1).astype(np.float32)

        # Boundary conditions
        n_face = max(n_bc // 6, 1)
        bc_parts, bc_y = [], []

        # Inlet x=0
        pts = np.zeros((n_face, 3), dtype=np.float32)
        pts[:, 1] = rng.uniform(0, H, n_face).astype(np.float32)
        pts[:, 2] = rng.uniform(0, W, n_face).astype(np.float32)
        bc_parts.append(pts)
        bc_y.append(self._inlet_profile(pts[:, 1], pts[:, 2]).reshape(-1, 1).astype(np.float32))

        # Outlet x=L
        pts = np.full((n_face, 3), [L, 0, 0], dtype=np.float32)
        pts[:, 1] = rng.uniform(0, H, n_face).astype(np.float32)
        pts[:, 2] = rng.uniform(0, W, n_face).astype(np.float32)
        bc_parts.append(pts)
        bc_y.append(np.full((n_face, 1), self.T_out, dtype=np.float32))

        # 4 wall faces (y=0, y=H, z=0, z=W)
        for d, vals in [(1, [0, H]), (2, [0, W])]:
            for v in vals:
                pts = rng.uniform([0, 0, 0], [L, H, W], (n_face, 3)).astype(np.float32)
                pts[:, d] = v
                bc_parts.append(pts)
                bc_y.append(np.zeros((n_face, 1), dtype=np.float32))

        xy_bc = np.concatenate(bc_parts, axis=0)
        Y_bc = np.concatenate(bc_y, axis=0)

        # Evaluation: cross-section at x = L/2
        gy = np.linspace(0, H, grid_n, dtype=np.float32)
        gz = np.linspace(0, W, grid_n, dtype=np.float32)
        GY, GZ = np.meshgrid(gy, gz)
        xy_eval = np.stack([
            np.full(GY.size, L / 2, dtype=np.float32),
            GY.ravel(), GZ.ravel()
        ], axis=1)
        sol_e = self.analytical(xy_eval[:, 0], xy_eval[:, 1], xy_eval[:, 2])
        Y_eval = sol_e["T"].reshape(-1, 1).astype(np.float32)

        return xyz, Y_int, xy_bc, Y_bc, xy_eval, Y_eval, ["T"]


register_problem(HeatChannel3D())


# ══════════════════════════════════════════════════════════════════════════════
# Case 4 — 3D Lid-Driven Cavity (Stokes) using pinneapple_design geometry
# ══════════════════════════════════════════════════════════════════════════════

class LidDrivenCavity3DPINN(ArenaProblem):
    """3D Stokes lid-driven cavity using geometry from pinneapple_design.

    The domain [0,1]³ is sampled via ``LidDrivenCavityDomain3D`` which provides
    structured interior/boundary sampling with named BC regions.

    Fields: u, v, w (velocities) and p (pressure) — 4 outputs.
    PDE: steady incompressible Stokes (Re→0 limit).
    """
    name = "lid_driven_cavity_3d_pinn"
    description = "3D Stokes lid-driven cavity — geometry from pinneapple_design"
    domain = "Fluid Mechanics 3D"
    input_dim = 3
    output_dim = 4   # u, v, w, p

    def __init__(self, size: float = 1.0, lid_velocity: float = 1.0, mu: float = 1.0):
        self.size = size
        self.lid_velocity = lid_velocity
        self.mu = mu

    def analytical(self, x, y, z=None, **kw):
        return None  # No closed-form for lid-driven cavity

    def pinn_residuals(self, net, xy_int, xy_bc, uv_bc, **kw):
        mu = self.mu
        xyz = xy_int.clone().requires_grad_(True)
        out = _unwrap(net(xyz))
        u, v, w, p = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]

        # Stokes: -μΔu + ∇p = 0,  div u = 0
        def _lap(f):
            return _laplacian_3d(f, xyz)

        p_x = _grad(p, xyz, 0)
        p_y = _grad(p, xyz, 1)
        p_z = _grad(p, xyz, 2)

        r1 = -mu * _lap(u) + p_x
        r2 = -mu * _lap(v) + p_y
        r3 = -mu * _lap(w) + p_z
        r4 = _grad(u, xyz, 0) + _grad(v, xyz, 1) + _grad(w, xyz, 2)

        pde_loss = (r1**2 + r2**2 + r3**2 + r4**2).mean()
        bc_loss = ((_unwrap(net(xy_bc)) - uv_bc) ** 2).mean()
        return pde_loss, bc_loss

    def supervised_data(self, n_train=800, n_bc=800, grid_n=15, **kw):
        from pinneapple_design.geometry.gen.domains3d import LidDrivenCavityDomain3D

        domain = LidDrivenCavityDomain3D(
            size=self.size, lid_velocity=self.lid_velocity
        )

        batch = domain.get_pinn_batch(n_col=n_train, n_bc_per_region=n_bc // 2)
        xyz_int = batch["x_col"].astype(np.float32)
        Y_int = np.zeros((len(xyz_int), 4), dtype=np.float32)  # no analytical

        xyz_bc = batch["x_bc"].astype(np.float32)
        Y_bc = np.zeros((len(xyz_bc), 4), dtype=np.float32)

        # Apply lid BC: u=lid_velocity, v=0, w=0 on the lid face
        regions = batch["bc_regions"]
        for i, reg in enumerate(regions):
            if reg == "lid":
                Y_bc[i, 0] = self.lid_velocity  # u
            # walls: all zeros (already set)

        # Evaluation: mid-plane cross-section at z=0.5
        xy_eval = domain.sample_structured_grid(grid_n, grid_n, 3)
        # Filter to z ≈ size/2
        zmid = self.size / 2
        close = np.abs(xy_eval[:, 2] - zmid) < self.size / (3 * 2)
        xy_eval = xy_eval[close]
        if len(xy_eval) == 0:
            xy_eval = domain.sample_structured_grid(grid_n, grid_n, 1)
        Y_eval = np.zeros((len(xy_eval), 4), dtype=np.float32)

        return xyz_int, Y_int, xyz_bc, Y_bc, xy_eval, Y_eval, ["u", "v", "w", "p"]


register_problem(LidDrivenCavity3DPINN())


# ─────────────────────────────────────────────────────────────────────────────
# Arena config builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(
    problem_name: str,
    epochs: int,
    output_prefix: str,
    models: Optional[List[Dict]] = None,
    grid_n: int = 20,
    n_col: int = 800,
    n_bc: int = 600,
    n_train: int = 600,
) -> Dict[str, Any]:
    if models is None:
        models = [
            {
                "name": "VanillaPINN",
                "type": "vanilla_pinn",
                "network": {"hidden": [128, 128, 128, 128], "activation": "tanh"},
                "training": {"epochs": epochs, "lr": 1e-3},
            },
            {
                "name": "SIREN",
                "type": "siren",
                "network": {"hidden": [128, 128, 128, 128], "omega_0": 30.0},
                "training": {"epochs": epochs, "lr": 5e-4},
            },
        ]
    return {
        "problem": {
            "name": problem_name,
            "params": {},
            "grid_n": grid_n,
            "n_col": n_col,
            "n_bc": n_bc,
            "n_train_supervised": n_train,
        },
        "models": models,
        "output": {
            "dir": f"outputs/geometry_3d/{output_prefix}/",
            "prefix": output_prefix,
            "save_figures": True,
            "dark_theme": True,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Case runners
# ─────────────────────────────────────────────────────────────────────────────

def run_poisson_box(epochs: int = 3000):
    print("\n" + "=" * 60)
    print("  Case 1: 3D Poisson on Unit Box")
    print("=" * 60)
    cfg = _make_config("poisson_box_3d", epochs, "poisson_box_3d",
                       grid_n=20, n_col=1000, n_bc=600, n_train=600)
    Arena(ArenaConfig.from_dict(cfg)).run()


def run_laplace_sphere(epochs: int = 3000):
    print("\n" + "=" * 60)
    print("  Case 2: Laplace on Unit Sphere")
    print("=" * 60)
    cfg = _make_config("laplace_sphere_3d", epochs, "laplace_sphere_3d",
                       grid_n=20, n_col=800, n_bc=400, n_train=600)
    Arena(ArenaConfig.from_dict(cfg)).run()


def run_heat_channel(epochs: int = 3000):
    print("\n" + "=" * 60)
    print("  Case 3: 3D Steady Heat in Channel")
    print("=" * 60)
    cfg = _make_config("heat_channel_3d", epochs, "heat_channel_3d",
                       grid_n=20, n_col=800, n_bc=600, n_train=600)
    Arena(ArenaConfig.from_dict(cfg)).run()


def run_lid_cavity(epochs: int = 3000):
    print("\n" + "=" * 60)
    print("  Case 4: 3D Lid-Driven Cavity (Stokes, geometry-aware)")
    print("=" * 60)
    models = [
        {
            "name": "VanillaPINN",
            "type": "vanilla_pinn",
            "network": {"hidden": [128, 128, 128, 128], "activation": "tanh"},
            "training": {"epochs": epochs, "lr": 5e-4},
        },
    ]
    cfg = _make_config("lid_driven_cavity_3d_pinn", epochs, "lid_cavity_3d",
                       models=models, grid_n=15, n_col=800, n_bc=800, n_train=800)
    Arena(ArenaConfig.from_dict(cfg)).run()


# ─────────────────────────────────────────────────────────────────────────────
# Quick demo: define_problem() for a 3D equation
# ─────────────────────────────────────────────────────────────────────────────

def run_define_problem_3d(epochs: int = 2000):
    """Show that define_problem() also works for 3D coordinates.

    Laplace equation Δu=0 on [0,1]³ with u=0 on all faces
    except the top z=1 face where u = sin(πx)·sin(πy).
    """
    from pinneapple_arena import define_problem

    print("\n" + "=" * 60)
    print("  Bonus: 3D Laplace via define_problem()")
    print("=" * 60)

    prob = define_problem(
        name="laplace_box_3d_easy",
        description="3D Laplace: Δu=0 on [0,1]³, heated top face",

        # Three spatial coordinates
        coords={"x": (0.0, 1.0), "y": (0.0, 1.0), "z": (0.0, 1.0)},
        fields=["u"],

        # PDE: Δu = u_xx + u_yy + u_zz = 0
        pde="u_xx + u_yy + u_zz",

        bcs=[
            # Cold walls
            {"type": "dirichlet", "at": "x_min", "field": "u", "value": 0.0},
            {"type": "dirichlet", "at": "x_max", "field": "u", "value": 0.0},
            {"type": "dirichlet", "at": "y_min", "field": "u", "value": 0.0},
            {"type": "dirichlet", "at": "y_max", "field": "u", "value": 0.0},
            {"type": "dirichlet", "at": "z_min", "field": "u", "value": 0.0},
            # Heated top face: u = sin(πx)·sin(πy)
            {
                "type": "dirichlet",
                "at": "z_max",
                "field": "u",
                "value": lambda x, y, z, **kw: np.sin(np.pi * x) * np.sin(np.pi * y),
            },
        ],
        params={},
        bc_weight=20.0,
    )

    prob.solve(
        models=["VanillaPINN"],
        epochs=epochs,
        hidden=[128, 128, 128, 128],
        output_dir="outputs/geometry_3d/laplace_box_easy/",
        grid_n=20,
        n_col=1000,
        n_bc=600,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Arena 3D geometry pipeline examples"
    )
    parser.add_argument(
        "--case",
        choices=["box", "sphere", "channel", "cavity", "easy3d", "all"],
        default="all",
        help="Which 3D case to run (default: all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epochs for all models",
    )
    args = parser.parse_args()

    defaults = {
        "box": 3000, "sphere": 3000, "channel": 3000,
        "cavity": 3000, "easy3d": 2000,
    }

    def _ep(case):
        return args.epochs if args.epochs is not None else defaults[case]

    runners = {
        "box":    lambda: run_poisson_box(_ep("box")),
        "sphere": lambda: run_laplace_sphere(_ep("sphere")),
        "channel": lambda: run_heat_channel(_ep("channel")),
        "cavity": lambda: run_lid_cavity(_ep("cavity")),
        "easy3d": lambda: run_define_problem_3d(_ep("easy3d")),
    }

    if args.case == "all":
        for fn in runners.values():
            fn()
    else:
        runners[args.case]()


if __name__ == "__main__":
    main()
