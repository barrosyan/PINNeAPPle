"""14_physics_simulation_surrogate_workflow.py -- Physics Simulation + Surrogate Workflow.

End-to-end pipeline:
  1. Create / load geometry (rectangle, plate-with-hole, cylinder domain)
  2. Generate structured mesh (Q4 for FEM, collocated grid for FDM)
  3. Choose physics problem and run numerical simulation
  4. Train surrogate models on simulation data
  5. Evaluate and visualize -- fields, error maps, metrics leaderboard

Structural problems
  hookes_2d  : 2D plane-stress linear elasticity (Hooke's Law)
  von_mises  : same + Von Mises yield analysis and critical-zone identification
  crash      : explicit FEM elastic wave under sudden impact load

Flow problems
  ns_internal: 2D incompressible NS, lid-driven cavity, vorticity-streamfunction FDM
  ns_external: 2D incompressible NS, flow past cylinder, projection-method FDM

  all        : run every problem in sequence

Surrogate models (--models, default = all three):
  pinn       : VanillaPINN  -- MLP + autograd PDE residuals
  fourier    : FourierPINN  -- Random-Fourier-Feature MLP + PDE residuals
  deeponet   : DeepONet     -- parametric branch/trunk operator, data-driven

Usage
-----
  python 14_physics_simulation_surrogate_workflow.py --problem von_mises --fast
  python 14_physics_simulation_surrogate_workflow.py --problem ns_internal --models pinn deeponet
  python 14_physics_simulation_surrogate_workflow.py --problem all --fast --save results/
  python 14_physics_simulation_surrogate_workflow.py --problem ns_external --save results/
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from scipy.sparse import lil_matrix
    from scipy.sparse.linalg import spsolve
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL = True
except ImportError:
    _MPL = False


# =============================================================================
# 0.  CLI + UTILITIES
# =============================================================================

ALL_PROBLEMS  = ["hookes_2d", "von_mises", "crash", "ns_internal", "ns_external"]
ALL_MODELS    = ["pinn", "fourier", "deeponet"]


def parse_args():
    p = argparse.ArgumentParser(description="Physics Simulation + Surrogate Workflow")
    p.add_argument("--problem", default="ns_internal",
                   choices=ALL_PROBLEMS + ["all"])
    p.add_argument("--models", nargs="+", default=ALL_MODELS,
                   choices=ALL_MODELS)
    p.add_argument("--fast",   action="store_true",
                   help="Coarse grid, few epochs (quick test)")
    p.add_argument("--device", default="auto")
    p.add_argument("--save",   default=None,
                   help="Directory to save plots and JSON results")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override training epochs")
    return p.parse_args()


def _dev(arg: str) -> str:
    if not _TORCH:
        return "cpu"
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def _hdr(title: str):
    bar = "=" * 68
    print(f"\n{bar}\n  {title}\n{bar}")


def _step(msg: str):
    print(f"\n  >> {msg}")


# =============================================================================
# 1.  GEOMETRY & MESH
# =============================================================================

@dataclass
class Geometry2D:
    """Parametric 2D geometry description."""
    kind: str            # "rectangle" | "plate_hole" | "channel_cylinder"
    Lx: float
    Ly: float
    params: dict = field(default_factory=dict)

    def describe(self):
        print(f"  Geometry   : {self.kind}")
        print(f"  Domain     : [0, {self.Lx}] x [0, {self.Ly}]")
        for k, v in self.params.items():
            print(f"  {k:<12}: {v}")


@dataclass
class RectMesh2D:
    """Structured Q4 mesh on [0,Lx] x [0,Ly] with nx*ny elements."""
    Lx: float
    Ly: float
    nx: int
    ny: int

    def __post_init__(self):
        xs = np.linspace(0.0, self.Lx, self.nx + 1)
        ys = np.linspace(0.0, self.Ly, self.ny + 1)
        XI, YI = np.meshgrid(xs, ys, indexing="ij")
        self.nodes = np.column_stack([XI.ravel(), YI.ravel()])  # (n_nodes, 2)

        elems = []
        for i in range(self.nx):
            for j in range(self.ny):
                n0 = i * (self.ny + 1) + j
                n1 = (i + 1) * (self.ny + 1) + j
                n2 = (i + 1) * (self.ny + 1) + j + 1
                n3 = i * (self.ny + 1) + j + 1
                elems.append([n0, n1, n2, n3])
        self.elements = np.array(elems, dtype=int)

        tol = 1e-12
        self.left   = np.where(self.nodes[:, 0] < tol)[0]
        self.right  = np.where(self.nodes[:, 0] > self.Lx - tol)[0]
        self.bottom = np.where(self.nodes[:, 1] < tol)[0]
        self.top    = np.where(self.nodes[:, 1] > self.Ly - tol)[0]

    @property
    def n_nodes(self): return len(self.nodes)
    @property
    def n_elem(self): return len(self.elements)
    @property
    def dx(self): return self.Lx / self.nx
    @property
    def dy(self): return self.Ly / self.ny


# =============================================================================
# 2.  FEM HELPERS -- Q4 PLANE-STRESS ASSEMBLY
# =============================================================================

def _q4_B(xi: float, eta: float, dx: float, dy: float) -> np.ndarray:
    """3x8 strain-displacement matrix for rectangular Q4 element."""
    xi_n  = np.array([-1.0,  1.0,  1.0, -1.0])
    eta_n = np.array([-1.0, -1.0,  1.0,  1.0])
    dNdxi  = xi_n  * (1.0 + eta_n * eta) / 4.0
    dNdeta = eta_n * (1.0 + xi_n  * xi)  / 4.0
    dNdx = dNdxi  * (2.0 / dx)
    dNdy = dNdeta * (2.0 / dy)
    B = np.zeros((3, 8))
    for i in range(4):
        B[0, 2*i]     = dNdx[i]   # eps_xx = du/dx
        B[1, 2*i + 1] = dNdy[i]   # eps_yy = dv/dy
        B[2, 2*i]     = dNdy[i]   # gamma_xy = du/dy + dv/dx
        B[2, 2*i + 1] = dNdx[i]
    return B


def _D_plane_stress(E: float, nu: float) -> np.ndarray:
    return E / (1.0 - nu**2) * np.array(
        [[1.0,  nu,           0.0],
         [nu,   1.0,          0.0],
         [0.0,  0.0, (1.0-nu)/2.0]]
    )


def _assemble_stiffness(mesh: RectMesh2D, E: float, nu: float):
    gp = 1.0 / np.sqrt(3.0) * np.array([-1.0, 1.0])
    D  = _D_plane_stress(E, nu)
    dx, dy = mesh.dx, mesh.dy
    n_dof = 2 * mesh.n_nodes

    if _SCIPY:
        K = lil_matrix((n_dof, n_dof))
    else:
        K = np.zeros((n_dof, n_dof))

    for elem in mesh.elements:
        dofs = np.array([[2*n, 2*n + 1] for n in elem]).ravel()
        Ke   = np.zeros((8, 8))
        for xi in gp:
            for eta in gp:
                B    = _q4_B(xi, eta, dx, dy)
                detJ = dx * dy / 4.0
                Ke  += B.T @ D @ B * detJ
        for i, di in enumerate(dofs):
            for j, dj in enumerate(dofs):
                K[di, dj] += Ke[i, j]

    return K.tocsr() if _SCIPY else K


def _apply_bcs_penalty(K, F, fix_dofs: np.ndarray, fix_vals=None):
    diag = K.diagonal() if _SCIPY else np.diag(K)
    pen  = float(np.max(np.abs(diag))) * 1e12
    if fix_vals is None:
        fix_vals = np.zeros(len(fix_dofs))
    if _SCIPY:
        K = K.tolil()
    for dof, val in zip(fix_dofs, fix_vals):
        K[dof, dof] += pen
        F[dof]      += pen * val
    return (K.tocsr() if _SCIPY else K), F


# =============================================================================
# 3.  STRUCTURAL MECHANICS -- STATIC LINEAR ELASTICITY (Hooke's / Von Mises)
# =============================================================================

def solve_elasticity(
    mesh: RectMesh2D,
    E: float,
    nu: float,
    traction_mag: float,
    traction_dir: str = "x",
) -> Dict[str, np.ndarray]:
    """Q4 plane-stress FEM.  Returns node-level fields."""
    n = mesh.n_nodes
    K = _assemble_stiffness(mesh, E, nu)
    F = np.zeros(2 * n)

    # Traction on right edge (x-direction pull or y-direction shear)
    load_dof   = 0 if traction_dir == "x" else 1
    edge_len   = mesh.Ly if traction_dir == "x" else mesh.Lx
    F_per_node = traction_mag * edge_len / max(len(mesh.right), 1)
    for nd in mesh.right:
        F[2*nd + load_dof] += F_per_node

    # Fix left edge (clamped)
    fix_x = 2 * mesh.left
    fix_y = 2 * mesh.left + 1
    fix_dofs = np.concatenate([fix_x, fix_y])
    K, F = _apply_bcs_penalty(K, F, fix_dofs)

    t0 = time.time()
    U  = spsolve(K, F) if _SCIPY else np.linalg.solve(K, F)
    print(f"    FEM solved in {time.time()-t0:.2f}s  "
          f"(n_nodes={n}, n_dof={2*n})")

    ux = U[0::2]
    uy = U[1::2]

    # Stress recovery at element centres (ξ=η=0), averaged to nodes
    D   = _D_plane_stress(E, nu)
    dx, dy = mesh.dx, mesh.dy
    s_xx = np.zeros(n); s_yy = np.zeros(n); s_xy = np.zeros(n)
    cnt  = np.zeros(n)
    for elem in mesh.elements:
        dofs = np.array([[2*nd, 2*nd + 1] for nd in elem]).ravel()
        B    = _q4_B(0.0, 0.0, dx, dy)
        eps  = B @ U[dofs]
        sig  = D @ eps
        for nd in elem:
            s_xx[nd] += sig[0]; s_yy[nd] += sig[1]; s_xy[nd] += sig[2]
            cnt[nd]  += 1
    cnt = np.maximum(cnt, 1)
    s_xx /= cnt; s_yy /= cnt; s_xy /= cnt

    vm = np.sqrt(s_xx**2 - s_xx*s_yy + s_yy**2 + 3.0*s_xy**2)

    print(f"    max |ux| = {np.abs(ux).max():.4e} m   "
          f"max von_Mises = {vm.max():.4e} Pa")

    return dict(
        x=mesh.nodes[:, 0], y=mesh.nodes[:, 1],
        ux=ux, uy=uy,
        sigma_xx=s_xx, sigma_yy=s_yy, sigma_xy=s_xy,
        von_mises=vm,
        E=E, nu=nu, traction=traction_mag,
    )


# =============================================================================
# 4.  STRUCTURAL MECHANICS -- CRASH / ELASTIC WAVE (Explicit Dynamics)
# =============================================================================

def solve_crash(
    mesh: RectMesh2D,
    E: float,
    nu: float,
    rho: float,
    F_impact: float,
    T: float,
    dt: float,
    store_every: int = 5,
) -> Dict[str, np.ndarray]:
    """
    Central-difference explicit FEM: elastic stress-wave under sudden impact.
    Impact load F_impact applied as step function at t=0 on the right edge.
    """
    n     = mesh.n_nodes
    n_dof = 2 * n
    nt    = int(T / dt)
    print(f"    Explicit FEM: nodes={n}, steps={nt}, dt={dt:.2e}")

    K = _assemble_stiffness(mesh, E, nu)

    # Lumped (diagonal) mass: m_node = rho * element_area / 4
    node_mass = np.zeros(n)
    for elem in mesh.elements:
        area = mesh.dx * mesh.dy
        for nd in elem:
            node_mass[nd] += rho * area / 4.0
    M = np.repeat(node_mass, 2)  # one per DOF

    # Impact: step force on right edge, x-direction
    F_ext = np.zeros(n_dof)
    for nd in mesh.right:
        F_ext[2*nd] = F_impact / max(len(mesh.right), 1)

    # Fixed DOFs (left edge, clamped)
    fix_dofs  = np.concatenate([2*mesh.left, 2*mesh.left + 1])
    free_mask = np.ones(n_dof, dtype=bool)
    free_mask[fix_dofs] = False

    U_prev = np.zeros(n_dof)
    U_curr = np.zeros(n_dof)

    D  = _D_plane_stress(E, nu)
    dx, dy = mesh.dx, mesh.dy

    times     = [0.0]
    snaps_ux  = [U_curr[0::2].copy()]
    snaps_vm  = [np.zeros(n)]

    def _nodal_vm(U_flat):
        s_xx = np.zeros(n); s_yy = np.zeros(n); s_xy = np.zeros(n); c = np.zeros(n)
        for el in mesh.elements:
            dfs = np.array([[2*nd, 2*nd+1] for nd in el]).ravel()
            B   = _q4_B(0.0, 0.0, dx, dy)
            sig = D @ (B @ U_flat[dfs])
            for nd in el:
                s_xx[nd] += sig[0]; s_yy[nd] += sig[1]; s_xy[nd] += sig[2]; c[nd] += 1
        c = np.maximum(c, 1)
        return np.sqrt((s_xx/c)**2 - (s_xx/c)*(s_yy/c) + (s_yy/c)**2 + 3*(s_xy/c)**2)

    for step in range(nt):
        KU = (K @ U_curr) if _SCIPY else (K @ U_curr)
        A  = (F_ext - KU) / np.maximum(M, 1e-30)
        A[~free_mask] = 0.0

        if step == 0:
            U_next = U_curr + 0.5 * dt * dt * A
        else:
            U_next = 2.0 * U_curr - U_prev + dt * dt * A
        U_next[fix_dofs] = 0.0
        U_prev, U_curr = U_curr, U_next

        if (step + 1) % store_every == 0:
            t_now = (step + 1) * dt
            times.append(t_now)
            snaps_ux.append(U_curr[0::2].copy())
            snaps_vm.append(_nodal_vm(U_curr))

    print(f"    Stored {len(times)} snapshots  "
          f"max |ux| final = {np.abs(U_curr[0::2]).max():.4e}")
    return dict(
        x=mesh.nodes[:, 0], y=mesh.nodes[:, 1],
        times=np.array(times),
        ux_snaps=np.array(snaps_ux),
        vm_snaps=np.array(snaps_vm),
        rho=rho, E=E, nu=nu, F_impact=F_impact,
    )


# =============================================================================
# 5.  FLOW MECHANICS -- LID-DRIVEN CAVITY (Vorticity-Streamfunction FDM)
# =============================================================================

def _build_sf_poisson(nx: int, ny: int, dx: float, dy: float):
    """Sparse Laplacian for interior streamfunction Poisson (all-Dirichlet)."""
    n_int = (nx - 1) * (ny - 1)
    A = lil_matrix((n_int, n_int))

    def k(i, j): return (i - 1) * (ny - 1) + (j - 1)

    for i in range(1, nx):
        for j in range(1, ny):
            ki = k(i, j)
            A[ki, ki] = -2.0 / dx**2 - 2.0 / dy**2
            if i > 1:      A[ki, k(i-1, j)]  += 1.0 / dx**2
            if i < nx - 1: A[ki, k(i+1, j)]  += 1.0 / dx**2
            if j > 1:      A[ki, k(i, j-1)]  += 1.0 / dy**2
            if j < ny - 1: A[ki, k(i, j+1)]  += 1.0 / dy**2
    return A.tocsr()


def solve_lid_driven_cavity(
    nx: int, ny: int, Re: float, U_lid: float, T: float, dt: float
) -> Dict[str, np.ndarray]:
    """
    Lid-driven cavity: vorticity-streamfunction FDM.
    Unit square [0,1]x[0,1], moving lid at top (y=1).
    """
    dx = 1.0 / nx; dy = 1.0 / ny
    nu = U_lid / Re
    nt = int(T / dt)
    print(f"    LDC FDM: {nx}x{ny}, Re={Re:.0f}, T={T:.1f}, "
          f"steps={nt}, nu={nu:.4e}")

    # Build Poisson matrix once (interior nodes)
    if not _SCIPY:
        raise RuntimeError("SciPy required for NS lid-driven cavity solver.")
    A_sf = _build_sf_poisson(nx, ny, dx, dy)

    n_int = (nx - 1) * (ny - 1)
    psi   = np.zeros((nx + 1, ny + 1))
    omega = np.zeros((nx + 1, ny + 1))

    def solve_streamfunction(omega_arr):
        rhs = np.zeros(n_int)
        for i in range(1, nx):
            for j in range(1, ny):
                rhs[(i-1)*(ny-1)+(j-1)] = -omega_arr[i, j]
        sol = spsolve(A_sf, rhs)
        psi_new = np.zeros((nx + 1, ny + 1))
        for i in range(1, nx):
            for j in range(1, ny):
                psi_new[i, j] = sol[(i-1)*(ny-1)+(j-1)]
        return psi_new

    psi = solve_streamfunction(omega)

    for step in range(nt):
        # Velocity from streamfunction
        u = np.zeros((nx+1, ny+1))
        v = np.zeros((nx+1, ny+1))
        u[1:-1, 1:-1] = (psi[1:-1, 2:]  - psi[1:-1, :-2]) / (2.0 * dy)
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0 * dx)
        u[:, -1] = U_lid  # lid

        # Vorticity transport: upwind convection + central diffusion
        u_c = u[1:-1, 1:-1]; v_c = v[1:-1, 1:-1]
        om  = omega

        adv_x = np.where(
            u_c >= 0,
            u_c * (om[1:-1, 1:-1] - om[:-2, 1:-1]) / dx,
            u_c * (om[2:, 1:-1]   - om[1:-1, 1:-1]) / dx,
        )
        adv_y = np.where(
            v_c >= 0,
            v_c * (om[1:-1, 1:-1] - om[1:-1, :-2]) / dy,
            v_c * (om[1:-1, 2:]   - om[1:-1, 1:-1]) / dy,
        )
        diff = nu * (
            (om[2:, 1:-1] - 2.0*om[1:-1, 1:-1] + om[:-2, 1:-1]) / dx**2 +
            (om[1:-1, 2:] - 2.0*om[1:-1, 1:-1] + om[1:-1, :-2]) / dy**2
        )
        omega_new = om.copy()
        omega_new[1:-1, 1:-1] = om[1:-1, 1:-1] + dt * (-adv_x - adv_y + diff)

        # Wall vorticity (Thom's formula)
        omega_new[1:-1, 0]  = -2.0 * psi[1:-1, 1]  / dy**2            # bottom
        omega_new[1:-1, -1] = -2.0 * (psi[1:-1, -2] - U_lid*dy) / dy**2  # lid
        omega_new[0, 1:-1]  = -2.0 * psi[1, 1:-1]  / dx**2            # left
        omega_new[-1, 1:-1] = -2.0 * psi[-2, 1:-1] / dx**2            # right
        omega = omega_new

        # Poisson solve every 10 steps (or first step)
        if (step + 1) % 10 == 0 or step == 0:
            psi = solve_streamfunction(omega)

        if (step + 1) % max(1, nt // 5) == 0:
            print(f"      step {step+1:5d}/{nt}  "
                  f"max_omega={np.abs(omega).max():.3f}")

    # Final velocity and derived fields
    psi = solve_streamfunction(omega)
    u = np.zeros((nx+1, ny+1)); v = np.zeros((nx+1, ny+1))
    u[1:-1, 1:-1] = (psi[1:-1, 2:]  - psi[1:-1, :-2]) / (2.0*dy)
    v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2.0*dx)
    u[:, -1] = U_lid

    # Pressure reconstruction from x-momentum (steady-state approx)
    xs = np.linspace(0.0, 1.0, nx+1)
    ys = np.linspace(0.0, 1.0, ny+1)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")

    # dp/dx = nu*(uxx+uyy) - u*ux - v*uy  (approximate, central differences)
    p = np.zeros((nx+1, ny+1))
    for i in range(1, nx):
        for j in range(1, ny):
            uxx = (u[i+1,j] - 2*u[i,j] + u[i-1,j]) / dx**2 if i>0 and i<nx else 0
            uyy = (u[i,j+1] - 2*u[i,j] + u[i,j-1]) / dy**2 if j>0 and j<ny else 0
            ux_  = (u[i+1,j] - u[i-1,j]) / (2*dx) if i>0 and i<nx else 0
            uy_  = (u[i,j+1] - u[i,j-1]) / (2*dy) if j>0 and j<ny else 0
            p[i,j] = nu*(uxx+uyy) - u[i,j]*ux_ - v[i,j]*uy_
    p -= p.mean()  # zero-mean pressure

    print(f"    Final: max|u|={np.abs(u).max():.4f}  "
          f"max|v|={np.abs(v).max():.4f}")
    return dict(
        x=XX.ravel(), y=YY.ravel(),
        u=u.ravel(), v=v.ravel(), p=p.ravel(), omega=omega.ravel(),
        u2d=u, v2d=v, p2d=p, omega2d=omega,
        xs=xs, ys=ys, nx=nx, ny=ny, Re=Re, U_lid=U_lid,
    )


# =============================================================================
# 6.  FLOW MECHANICS -- CYLINDER FLOW (Projection Method FDM)
# =============================================================================

def _build_pressure_poisson(nx: int, ny: int, dx: float, dy: float):
    """Pressure-Poisson sparse matrix (all-Neumann + one fixed point)."""
    n = nx * ny
    A = lil_matrix((n, n))

    def k(i, j): return i * ny + j

    for i in range(nx):
        for j in range(ny):
            ki = k(i, j)
            if i == 0 and j == 0:      # fix p=0 at one corner
                A[ki, ki] = 1.0
                continue
            A[ki, ki] = -2.0/dx**2 - 2.0/dy**2
            # x-neighbors
            if i > 0:       A[ki, k(i-1, j)] += 1.0/dx**2
            else:            A[ki, ki]        += 1.0/dx**2  # Neumann ghost
            if i < nx-1:    A[ki, k(i+1, j)] += 1.0/dx**2
            else:            A[ki, ki]        += 1.0/dx**2
            # y-neighbors
            if j > 0:       A[ki, k(i, j-1)] += 1.0/dy**2
            else:            A[ki, ki]        += 1.0/dy**2
            if j < ny-1:    A[ki, k(i, j+1)] += 1.0/dy**2
            else:            A[ki, ki]        += 1.0/dy**2
    return A.tocsr()


def solve_cylinder_flow(
    nx: int, ny: int, Re: float, U_inf: float, T: float, dt: float,
    r_cyl: float = 0.1, cx: float = 0.35, cy: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    2D incompressible NS past a cylinder, projection-method FDM.
    Uses upwind advection (stable at high cell-Peclet numbers).
    Domain: [0,2] x [0,1], cylinder centred at (cx*Lx, cy*Ly).
    """
    Lx, Ly = 2.0, 1.0
    dx = Lx / nx; dy = Ly / ny
    nu = U_inf * (2.0 * r_cyl) / Re

    # CFL-safe time step
    dt = min(dt, 0.4 * min(dx, dy) / (U_inf + 1e-10),
             0.25 * min(dx, dy)**2 / (nu + 1e-10))
    nt = int(T / dt) + 1
    print(f"    Cylinder FDM: {nx}x{ny}, Re={Re:.0f}, T={T:.1f}, "
          f"steps={nt}, dt={dt:.2e}, nu={nu:.4e}")

    if not _SCIPY:
        raise RuntimeError("SciPy required for cylinder-flow solver.")

    xs = np.linspace(0.0, Lx, nx); ys = np.linspace(0.0, Ly, ny)
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    dist = np.sqrt((XX - cx*Lx)**2 + (YY - cy*Ly)**2)
    cyl  = dist <= r_cyl

    u = U_inf * np.ones((nx, ny))
    v = np.zeros((nx, ny))
    p = np.zeros((nx, ny))
    u[cyl] = 0.0; v[cyl] = 0.0

    A_p = _build_pressure_poisson(nx, ny, dx, dy)

    def _adv_u(f, uf, vf):
        """Upwind first-order advection of f with velocity (uf, vf)."""
        dfdx = np.where(uf[1:-1, 1:-1] >= 0,
                        (f[1:-1, 1:-1] - f[:-2, 1:-1]) / dx,
                        (f[2:,   1:-1] - f[1:-1, 1:-1]) / dx)
        dfdy = np.where(vf[1:-1, 1:-1] >= 0,
                        (f[1:-1, 1:-1] - f[1:-1, :-2]) / dy,
                        (f[1:-1, 2:]   - f[1:-1, 1:-1]) / dy)
        return dfdx, dfdy

    def _diff(f):
        """Central-difference diffusion Laplacian (interior only)."""
        fxx = (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2
        fyy = (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2
        return fxx + fyy

    def _cdiv(fu, fv):
        """Central divergence of (fu, fv) for full grid via numpy."""
        return np.gradient(fu, dx, axis=0) + np.gradient(fv, dy, axis=1)

    def _cgrad(phi):
        return np.gradient(phi, dx, axis=0), np.gradient(phi, dy, axis=1)

    for step in range(nt):
        # Predict velocity (interior only, then BCs)
        u_adv_x, u_adv_y = _adv_u(u, u, v)
        v_adv_x, v_adv_y = _adv_u(v, u, v)

        u_star = u.copy()
        v_star = v.copy()
        u_star[1:-1, 1:-1] += dt * (-u[1:-1,1:-1]*u_adv_x - v[1:-1,1:-1]*u_adv_y
                                     + nu * _diff(u))
        v_star[1:-1, 1:-1] += dt * (-u[1:-1,1:-1]*v_adv_x - v[1:-1,1:-1]*v_adv_y
                                     + nu * _diff(v))

        # BCs
        u_star[0, :]  = U_inf; v_star[0, :]  = 0.0   # inlet
        u_star[:, 0]  = 0.0;   v_star[:, 0]  = 0.0   # bottom
        u_star[:, -1] = 0.0;   v_star[:, -1] = 0.0   # top
        u_star[-1, :] = u_star[-2, :]; v_star[-1, :] = v_star[-2, :]  # outlet
        u_star[cyl]   = 0.0;   v_star[cyl]   = 0.0   # cylinder

        # Pressure Poisson: nabla^2 phi = div(u_star)/dt
        div_u = _cdiv(u_star, v_star)
        rhs   = (div_u / dt).ravel()
        rhs[0] = 0.0

        phi       = spsolve(A_p, rhs).reshape(nx, ny)
        dphi_x, dphi_y = _cgrad(phi)
        u_new = u_star - dt * dphi_x
        v_new = v_star - dt * dphi_y
        p    += phi

        # Enforce BCs
        u_new[0, :]  = U_inf; v_new[0, :]  = 0.0
        u_new[:, 0]  = 0.0;   v_new[:, 0]  = 0.0
        u_new[:, -1] = 0.0;   v_new[:, -1] = 0.0
        u_new[-1, :] = u_new[-2, :]; v_new[-1, :] = v_new[-2, :]
        u_new[cyl]   = 0.0;   v_new[cyl]   = 0.0
        p[0, :]      = p[1, :]
        p[:, 0]      = p[:, 1]; p[:, -1] = p[:, -2]
        p[-1, :]     = 0.0

        # Safety clamp
        u_new = np.clip(u_new, -5*U_inf, 5*U_inf)
        v_new = np.clip(v_new, -5*U_inf, 5*U_inf)
        u, v = u_new, v_new

        if (step + 1) % max(1, nt // 5) == 0:
            print(f"      step {step+1:5d}/{nt}  "
                  f"max|u|={np.abs(u[~cyl]).max():.4f}")

    omega = np.gradient(v, dx, axis=0) - np.gradient(u, dy, axis=1)
    print(f"    Final: max|u|={np.abs(u[~cyl]).max():.4f}  "
          f"max|omega|={np.abs(omega[~cyl]).max():.4f}")
    return dict(
        x=XX.ravel(), y=YY.ravel(),
        u=u.ravel(), v=v.ravel(), p=p.ravel(), omega=omega.ravel(),
        u2d=u, v2d=v, p2d=p, omega2d=omega,
        xs=xs, ys=ys, nx=nx, ny=ny, Re=Re, U_inf=U_inf,
        cyl_mask=cyl.ravel(),
    )


# =============================================================================
# 7.  DATASET BUILDER
# =============================================================================

def build_dataset(
    sim_data: Dict[str, np.ndarray],
    field_names: List[str],
    n_train: int = 4096,
    n_test:  int = 1024,
    seed:    int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
    """
    Sample collocation points from simulation data.
    Returns X_train, Y_train, X_test, Y_test plus normalisation stats.
    """
    rng = np.random.default_rng(seed)

    x_all = sim_data["x"].ravel()
    y_all = sim_data["y"].ravel()
    n_all = len(x_all)

    # Build feature matrix
    has_t = "t" in sim_data
    if not has_t:
        X_all = np.stack([x_all, y_all], axis=1).astype(np.float32)
    else:
        t_all = sim_data["t"].ravel()
        X_all = np.stack([x_all, y_all, t_all], axis=1).astype(np.float32)

    Y_cols = []
    valid_names = []
    for name in field_names:
        if name in sim_data and sim_data[name].ravel().shape[0] == n_all:
            Y_cols.append(sim_data[name].ravel())
            valid_names.append(name)
    Y_all = np.stack(Y_cols, axis=1).astype(np.float32)

    # Optional: mask out cylinder interior
    if "cyl_mask" in sim_data:
        mask = ~sim_data["cyl_mask"].ravel().astype(bool)
        X_all, Y_all = X_all[mask], Y_all[mask]

    n_avail  = len(X_all)
    n_use    = min(n_train + n_test, n_avail)
    idx      = rng.permutation(n_avail)[:n_use]
    X_use, Y_use = X_all[idx], Y_all[idx]

    # Normalise inputs to [0, 1]
    X_min = X_use.min(axis=0, keepdims=True)
    X_max = X_use.max(axis=0, keepdims=True) + 1e-10
    X_norm = (X_use - X_min) / (X_max - X_min)

    # Normalise outputs by std
    Y_mean = Y_use.mean(axis=0, keepdims=True)
    Y_std  = Y_use.std(axis=0, keepdims=True) + 1e-10
    Y_norm = (Y_use - Y_mean) / Y_std

    n_tr = min(n_train, n_use * 3 // 4)
    X_tr, Y_tr = X_norm[:n_tr], Y_norm[:n_tr]
    X_te, Y_te = X_norm[n_tr:], Y_norm[n_tr:]

    stats = dict(X_min=X_min, X_max=X_max, Y_mean=Y_mean, Y_std=Y_std)
    return X_tr, Y_tr, X_te, Y_te, valid_names, stats


# =============================================================================
# 8.  SURROGATE MODEL ARCHITECTURES
# =============================================================================

if _TORCH:
    class _MLP(nn.Module):
        """Simple MLP with skip connections."""
        def __init__(self, in_dim: int, out_dim: int, hidden: List[int],
                     activation=nn.Tanh):
            super().__init__()
            dims   = [in_dim] + hidden + [out_dim]
            layers = []
            for i in range(len(dims) - 1):
                layers.append(nn.Linear(dims[i], dims[i+1]))
                if i < len(dims) - 2:
                    layers.append(activation())
            self.net    = nn.Sequential(*layers)
            self.in_dim = in_dim

        def forward(self, x):
            return self.net(x)

    class VanillaPINN(nn.Module):
        """Standard MLP-PINN."""
        def __init__(self, in_dim: int, out_dim: int, hidden: List[int] = None):
            super().__init__()
            hidden = hidden or [64, 128, 128, 64]
            self.net = _MLP(in_dim, out_dim, hidden)

        def forward(self, x):
            return self.net(x)

    class FourierPINN(nn.Module):
        """Random Fourier Feature network + MLP trunk."""
        def __init__(self, in_dim: int, out_dim: int,
                     n_fourier: int = 128, sigma: float = 1.0,
                     hidden: List[int] = None):
            super().__init__()
            hidden = hidden or [128, 128, 64]
            B = torch.randn(in_dim, n_fourier) * sigma
            self.register_buffer("B", B)
            feat_dim = 2 * n_fourier
            self.net = _MLP(feat_dim, out_dim, hidden)

        def forward(self, x):
            proj = x @ self.B  # (batch, n_fourier)
            feats = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
            return self.net(feats)

    class DeepONet(nn.Module):
        """
        Branch / trunk DeepONet.
        branch_in : dimension of the input function (parameter or sensor values)
        trunk_in  : spatial / temporal query dimension
        p         : latent dimension (number of basis functions)
        out_dim   : number of output fields
        """
        def __init__(self, branch_in: int, trunk_in: int,
                     out_dim: int, p: int = 64):
            super().__init__()
            self.p = p
            self.out_dim = out_dim
            self.branch = _MLP(branch_in, p * out_dim, [64, 64])
            self.trunk  = _MLP(trunk_in,  p * out_dim, [64, 128, 64])
            self.bias   = nn.Parameter(torch.zeros(out_dim))

        def forward(self, branch_in: torch.Tensor, trunk_in: torch.Tensor):
            b = self.branch(branch_in)  # (batch_b, p*out_dim) OR (1, p*out_dim)
            t = self.trunk(trunk_in)    # (batch_t, p*out_dim)
            # If branch has batch dim 1 (shared across all trunk points), broadcast
            b = b.expand(t.shape[0], -1)
            b = b.view(-1, self.out_dim, self.p)
            t = t.view(-1, self.out_dim, self.p)
            return (b * t).sum(dim=-1) + self.bias  # (batch_t, out_dim)


# =============================================================================
# 9.  PHYSICS LOSSES (autograd PDE residuals)
# =============================================================================

if _TORCH:
    def _grad1(f, x):
        return torch.autograd.grad(f.sum(), x, create_graph=True)[0]

    def pde_loss_elasticity(model, X_col: torch.Tensor,
                            E: float, nu: float, Lx: float, Ly: float
                            ) -> torch.Tensor:
        """Equilibrium residual: div(sigma) = 0 for plane stress."""
        xn = X_col[:, 0:1].requires_grad_(True)
        yn = X_col[:, 1:2].requires_grad_(True)
        xy = torch.cat([xn, yn], dim=1)
        U  = model(xy)
        ux = U[:, 0:1]; uy = U[:, 1:2]

        ux_xn = _grad1(ux, xn); ux_yn = _grad1(ux, yn)
        uy_xn = _grad1(uy, xn); uy_yn = _grad1(uy, yn)
        # Physical derivatives: d/dx = (1/Lx) * d/dxn
        ux_x = ux_xn / Lx; ux_y = ux_yn / Ly
        uy_x = uy_xn / Lx; uy_y = uy_yn / Ly

        C       = E / (1.0 - nu**2)
        sig_xx  = C * (ux_x + nu * uy_y)
        sig_yy  = C * (uy_y + nu * ux_x)
        sig_xy  = C * (1.0 - nu) / 2.0 * (ux_y + uy_x)

        sig_xx_x = _grad1(sig_xx, xn) / Lx
        sig_xy_y = _grad1(sig_xy, yn) / Ly
        sig_xy_x = _grad1(sig_xy, xn) / Lx
        sig_yy_y = _grad1(sig_yy, yn) / Ly

        R1 = sig_xx_x + sig_xy_y
        R2 = sig_xy_x + sig_yy_y
        return (R1**2 + R2**2).mean()

    def pde_loss_ns_steady(model, X_col: torch.Tensor,
                           nu_visc: float, Lx: float, Ly: float
                           ) -> torch.Tensor:
        """2D steady incompressible NS residual (continuity + momentum)."""
        xn = X_col[:, 0:1].requires_grad_(True)
        yn = X_col[:, 1:2].requires_grad_(True)
        xy = torch.cat([xn, yn], dim=1)
        out = model(xy)
        u = out[:, 0:1]; v = out[:, 1:2]; p = out[:, 2:3]

        u_xn  = _grad1(u, xn); u_yn  = _grad1(u, yn)
        v_xn  = _grad1(v, xn); v_yn  = _grad1(v, yn)
        p_xn  = _grad1(p, xn); p_yn  = _grad1(p, yn)
        u_xxn = _grad1(u_xn, xn); u_yyn = _grad1(u_yn, yn)
        v_xxn = _grad1(v_xn, xn); v_yyn = _grad1(v_yn, yn)

        u_x = u_xn/Lx; u_y = u_yn/Ly
        v_x = v_xn/Lx; v_y = v_yn/Ly
        p_x = p_xn/Lx; p_y = p_yn/Ly
        u_xx = u_xxn/Lx**2; u_yy = u_yyn/Ly**2
        v_xx = v_xxn/Lx**2; v_yy = v_yyn/Ly**2

        R_cont  = u_x + v_y
        R_mom_u = u*u_x + v*u_y + p_x - nu_visc*(u_xx + u_yy)
        R_mom_v = u*v_x + v*v_y + p_y - nu_visc*(v_xx + v_yy)
        return (R_cont**2 + R_mom_u**2 + R_mom_v**2).mean()

    def pde_loss_none(*args, **kwargs) -> torch.Tensor:
        return torch.tensor(0.0)


# =============================================================================
# 10.  TRAINING LOOP
# =============================================================================

def _to_tensor(arr, device):
    if _TORCH:
        return torch.tensor(arr, dtype=torch.float32, device=device)
    return arr


def train_surrogate(
    model_name: str,
    model,
    X_tr: np.ndarray, Y_tr: np.ndarray,
    X_te: np.ndarray, Y_te: np.ndarray,
    pde_loss_fn,
    pde_weight: float,
    n_col_pde: int,
    device: str,
    epochs: int,
    lr: float = 1e-3,
    batch: int = 512,
    is_deeponet: bool = False,
    branch_input: Optional[np.ndarray] = None,
) -> Dict:
    """
    Train a surrogate model.  Returns dict with training history and metrics.
    """
    if not _TORCH:
        print("  [skip] PyTorch not available")
        return {}

    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xtr = _to_tensor(X_tr, device)
    Ytr = _to_tensor(Y_tr, device)
    Xte = _to_tensor(X_te, device)
    Yte = _to_tensor(Y_te, device)
    if is_deeponet and branch_input is not None:
        B_shared = _to_tensor(branch_input.ravel(), device).unsqueeze(0)  # (1, b_dim)

    # Random collocation points in [0,1]^in_dim for PDE loss
    rng      = np.random.default_rng(42)
    in_dim   = X_tr.shape[1]

    hist_loss = []; hist_val = []
    n_tr = len(Xtr)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_tr, device=device)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, n_tr, batch):
            idx  = perm[start:start + batch]
            xb   = Xtr[idx]; yb = Ytr[idx]

            opt.zero_grad()
            if is_deeponet:
                yp = model(B_shared.expand(len(xb), -1), xb)
            else:
                yp = model(xb)
            data_loss = F.mse_loss(yp, yb)

            # PDE collocation
            if pde_weight > 0:
                X_col = _to_tensor(
                    rng.uniform(0, 1, (n_col_pde, in_dim)).astype(np.float32), device
                )
                if is_deeponet:
                    phys = pde_loss_none()
                else:
                    phys = pde_loss_fn(model, X_col)
            else:
                phys = torch.tensor(0.0, device=device)

            loss = data_loss + pde_weight * phys
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches  += 1

        sch.step()
        epoch_loss /= max(n_batches, 1)
        hist_loss.append(epoch_loss)

        if ep % max(1, epochs // 10) == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                if is_deeponet:
                    yp_te = model(B_shared.expand(len(Xte), -1), Xte)
                else:
                    yp_te = model(Xte)
                val = F.mse_loss(yp_te, Yte).item()
            hist_val.append((ep, val))
            print(f"    [{model_name}] ep {ep:5d}/{epochs}  "
                  f"loss={epoch_loss:.4e}  val_mse={val:.4e}  "
                  f"lr={sch.get_last_lr()[0]:.2e}")

    elapsed = time.time() - t0
    model.eval()
    with torch.no_grad():
        if is_deeponet:
            Yp = model(B_shared.expand(len(Xte), -1), Xte).cpu().numpy()
        else:
            Yp = model(Xte).cpu().numpy()
    Yg = Yte.cpu().numpy()

    return dict(
        model=model,
        model_name=model_name,
        hist_loss=hist_loss,
        hist_val=hist_val,
        Yp=Yp, Yg=Yg,
        Xte=X_te,
        elapsed=elapsed,
    )


# =============================================================================
# 11.  METRICS
# =============================================================================

def compute_metrics(Yg: np.ndarray, Yp: np.ndarray, field_names: List[str]) -> Dict:
    metrics = {}
    for i, name in enumerate(field_names):
        yg = Yg[:, i]; yp = Yp[:, i]
        mae    = np.mean(np.abs(yg - yp))
        rmse   = np.sqrt(np.mean((yg - yp)**2))
        rel_l2 = np.linalg.norm(yg - yp) / (np.linalg.norm(yg) + 1e-10)
        ss_res = np.sum((yg - yp)**2)
        ss_tot = np.sum((yg - yg.mean())**2) + 1e-10
        r2     = 1.0 - ss_res / ss_tot
        metrics[name] = dict(mae=mae, rmse=rmse, rel_l2=rel_l2, r2=r2)
    return metrics


def print_leaderboard(all_results: List[Dict], field_names: List[str]):
    _hdr("SURROGATE LEADERBOARD")
    header = f"  {'Model':<14}" + "".join(
        f"  {n[:8]:>10} rel-L2" for n in field_names
    ) + f"  {'Time(s)':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_results:
        if not r:
            continue
        m = compute_metrics(r["Yg"], r["Yp"], field_names)
        row = f"  {r['model_name']:<14}"
        for name in field_names:
            row += f"  {m[name]['rel_l2']:>14.3e}"
        row += f"  {r['elapsed']:>8.1f}"
        print(row)


# =============================================================================
# 12.  VISUALISATION
# =============================================================================

def _safe_imshow(ax, data2d, title, cmap="RdBu_r", symmetric=False):
    if symmetric:
        vmax = max(abs(data2d.min()), abs(data2d.max()))
        vmin = -vmax
    else:
        vmin, vmax = data2d.min(), data2d.max()
    im = ax.imshow(
        data2d.T, origin="lower", extent=[0, 1, 0, 1],
        cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
    )
    ax.set_title(title, fontsize=9)
    return im


def visualise_structural(sim: Dict, results: List[Dict],
                         field_names: List[str], problem: str,
                         save_dir: Optional[Path]):
    if not _MPL:
        return

    # Reconstruct grid dimensions from unique coordinates
    xs_u = np.unique(np.round(sim["x"], 8))
    ys_u = np.unique(np.round(sim["y"], 8))
    nx_grid = len(xs_u)
    ny_grid = len(ys_u)

    def to2d(arr):
        # Nodes are ordered as i*(ny+1)+j  (x-major)
        try:
            return arr.reshape(nx_grid, ny_grid)
        except ValueError:
            return arr.reshape(int(np.ceil(np.sqrt(len(arr)))), -1)

    n_models = len([r for r in results if r])
    n_fields = len(field_names)
    n_cols   = 1 + n_models  # simulation + one per model
    n_rows   = n_fields

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows),
                              squeeze=False)
    fig.suptitle(f"Structural: {problem.upper()}", fontsize=13, fontweight="bold")

    for fi, fname in enumerate(field_names):
        if fname not in sim:
            continue
        arr = sim[fname].ravel()
        try:
            arr2d = to2d(arr)
        except Exception:
            continue
        im = _safe_imshow(axes[fi][0], arr2d, f"FEM: {fname}",
                          cmap="RdYlBu_r", symmetric=fname in ("sigma_xx","sigma_yy","sigma_xy"))
        axes[fi][0].set_ylabel(fname, fontsize=9)
        plt.colorbar(im, ax=axes[fi][0], shrink=0.8)

        for mi, res in enumerate([r for r in results if r]):
            col = mi + 1
            # Reshape prediction to 2D
            Xte   = res["Xte"]
            Yp    = res["Yp"]
            Yg    = res["Yg"]
            fi_idx = field_names.index(fname) if fname in field_names else 0
            if fi_idx >= Yp.shape[1]:
                continue

            yp_arr = np.zeros(len(arr))
            yg_arr = np.zeros(len(arr))
            # predictions are normalised -- here just show normalised error
            err = np.abs(Yp[:, fi_idx] - Yg[:, fi_idx])
            ax  = axes[fi][col]
            ax.scatter(Xte[:, 0], Xte[:, 1], c=err, cmap="hot_r",
                       s=4, vmin=0, vmax=err.max())
            ax.set_title(f"{res['model_name']}: |error|", fontsize=9)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_dir:
        path = save_dir / f"{problem}_fields.png"
        fig.savefig(str(path), dpi=120)
        print(f"  Saved {path}")
    plt.close(fig)


def visualise_flow(sim: Dict, results: List[Dict],
                   field_names: List[str], problem: str,
                   save_dir: Optional[Path]):
    if not _MPL:
        return

    nx = sim.get("nx", 32)
    ny = sim.get("ny", 32)

    field_2d = {
        "u": sim.get("u2d"), "v": sim.get("v2d"),
        "p": sim.get("p2d"), "omega": sim.get("omega2d"),
    }
    present = {k: v for k, v in field_2d.items() if v is not None}
    n_show  = len(present)
    if n_show == 0:
        return

    n_models = len([r for r in results if r])
    n_cols   = n_show
    n_rows   = 1 + max(n_models, 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3.5*n_rows),
                              squeeze=False)
    fig.suptitle(f"Flow: {problem.upper()}", fontsize=13, fontweight="bold")

    cmaps = {"u": "RdYlBu_r", "v": "RdYlBu_r", "p": "plasma", "omega": "RdBu_r"}
    for ci, (fname, arr2d) in enumerate(present.items()):
        sym = fname in ("omega",)
        im  = _safe_imshow(axes[0][ci], arr2d, f"Simulation: {fname}",
                           cmap=cmaps.get(fname, "RdBu_r"), symmetric=sym)
        plt.colorbar(im, ax=axes[0][ci], shrink=0.8)

    for mi, res in enumerate([r for r in results if r]):
        row  = mi + 1
        Xte  = res["Xte"]
        Yp   = res["Yp"]
        Yg   = res["Yg"]

        for ci, fname in enumerate(present.keys()):
            fi_idx = field_names.index(fname) if fname in field_names else -1
            ax = axes[row][ci]
            if fi_idx < 0 or fi_idx >= Yp.shape[1]:
                ax.set_visible(False)
                continue
            err = np.abs(Yp[:, fi_idx] - Yg[:, fi_idx])
            sc  = ax.scatter(Xte[:, 0], Xte[:, 1], c=err, cmap="hot_r",
                             s=3, vmin=0, vmax=np.percentile(err, 95))
            plt.colorbar(sc, ax=ax, shrink=0.8)
            ax.set_title(f"{res['model_name']} |err|: {fname}", fontsize=9)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    if save_dir:
        path = save_dir / f"{problem}_fields.png"
        fig.savefig(str(path), dpi=120)
        print(f"  Saved {path}")
    plt.close(fig)


def visualise_training_curves(results: List[Dict], problem: str,
                              save_dir: Optional[Path]):
    if not _MPL:
        return
    valid = [r for r in results if r and r.get("hist_loss")]
    if not valid:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for r in valid:
        axes[0].plot(r["hist_loss"], label=r["model_name"])
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training loss")
    axes[0].set_yscale("log"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Training Curves")
    for r in valid:
        if r.get("hist_val"):
            eps, vals = zip(*r["hist_val"])
            axes[1].plot(eps, vals, "o-", label=r["model_name"])
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation MSE")
    axes[1].set_yscale("log"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Validation Curves")
    fig.suptitle(f"Learning Curves -- {problem.upper()}", fontsize=12)
    plt.tight_layout()
    if save_dir:
        path = save_dir / f"{problem}_curves.png"
        fig.savefig(str(path), dpi=120)
        print(f"  Saved {path}")
    plt.close(fig)


def visualise_crash_snapshots(sim: Dict, save_dir: Optional[Path]):
    if not _MPL:
        return
    times  = sim["times"]
    snaps  = sim["vm_snaps"]
    n_snap = min(6, len(times))
    idxs   = np.linspace(0, len(times)-1, n_snap, dtype=int)
    fig, axes = plt.subplots(1, n_snap, figsize=(3*n_snap, 3))
    if n_snap == 1:
        axes = [axes]
    vmax = snaps.max() + 1e-10
    for k, idx in enumerate(idxs):
        n_side = int(np.round(np.sqrt(len(sim["x"]))))
        try:
            arr2d = snaps[idx].reshape(n_side, n_side)
        except Exception:
            continue
        axes[k].imshow(arr2d.T, origin="lower", cmap="hot",
                       vmin=0, vmax=vmax, aspect="auto")
        axes[k].set_title(f"t={times[idx]:.3f}", fontsize=9)
        axes[k].axis("off")
    fig.suptitle("Von Mises stress wave propagation (Crash)", fontsize=12)
    plt.tight_layout()
    if save_dir:
        path = save_dir / "crash_snapshots.png"
        fig.savefig(str(path), dpi=120)
        print(f"  Saved {path}")
    plt.close(fig)


# =============================================================================
# 13.  WORKFLOW PIPELINE
# =============================================================================

class WorkflowPipeline:
    """Orchestrates all steps for a given physics problem."""

    def __init__(self, problem: str, model_names: List[str],
                 fast: bool, device: str, epochs: Optional[int],
                 save_dir: Optional[Path]):
        self.problem     = problem
        self.model_names = model_names
        self.fast        = fast
        self.device      = device
        self.save_dir    = save_dir

        # Sizing
        if fast:
            self.nx_struct = 32   # structural: fine enough for ~1000 data pts
            self.nx_flow   = 16   # flow: coarser grid, time integration is slow
            self.epochs    = epochs or 800
            self.n_train   = 4096
            self.n_test    = 1024
        else:
            self.nx_struct = 48
            self.nx_flow   = 32
            self.epochs    = epochs or 4000
            self.n_train   = 8000
            self.n_test    = 2000
        # unified property used by mesh builder
        self.nx = self.ny = self.nx_struct

    def run(self) -> Dict:
        _hdr(f"PROBLEM: {self.problem.upper()}")
        sim      = self._simulate()
        fields   = self._field_names()
        results  = self._train_all(sim, fields)

        if self.save_dir:
            self._save_json(sim, results, fields)

        if _MPL:
            if self.problem in ("hookes_2d", "von_mises"):
                visualise_structural(sim, results, fields, self.problem, self.save_dir)
            elif self.problem == "crash":
                visualise_crash_snapshots(sim, self.save_dir)
            else:
                visualise_flow(sim, results, fields, self.problem, self.save_dir)
            visualise_training_curves(results, self.problem, self.save_dir)
        else:
            print("  [skip] matplotlib not available -- no plots generated")

        print_leaderboard(results, fields)
        return dict(problem=self.problem, sim=sim, results=results, fields=fields)

    # ------------------------------------------------------------------
    # Simulation dispatch
    # ------------------------------------------------------------------

    def _simulate(self) -> Dict:
        _step("STEP 1 -- Geometry")
        geom = self._make_geometry()
        geom.describe()

        _step("STEP 2 -- Mesh generation")
        if self.problem in ("ns_internal", "ns_external"):
            nx_m = ny_m = self.nx_flow
        else:
            nx_m = ny_m = self.nx_struct
        mesh = RectMesh2D(geom.Lx, geom.Ly, nx_m, ny_m)
        print(f"  Mesh       : {mesh.nx} x {mesh.ny} Q4 elements")
        print(f"  Nodes      : {mesh.n_nodes}  DOFs: {2*mesh.n_nodes}")
        print(f"  dx={mesh.dx:.4f}  dy={mesh.dy:.4f}")

        _step("STEP 3 -- Physics simulation")
        return self._run_sim(geom, mesh)

    def _make_geometry(self) -> Geometry2D:
        prob = self.problem
        if prob in ("hookes_2d", "von_mises"):
            return Geometry2D("rectangle", 1.0, 0.5,
                              {"material": "steel", "E_GPa": 200, "nu": 0.3})
        if prob == "crash":
            return Geometry2D("rectangle", 0.5, 0.1,
                              {"material": "aluminium", "E_GPa": 70, "rho": 2700})
        if prob == "ns_internal":
            return Geometry2D("rectangle", 1.0, 1.0,
                              {"setup": "lid-driven cavity", "Re": 100})
        if prob == "ns_external":
            return Geometry2D("channel_cylinder", 2.0, 1.0,
                              {"setup": "flow past cylinder", "Re": 100, "D_cyl": 0.2})
        return Geometry2D("rectangle", 1.0, 1.0, {})

    def _run_sim(self, geom: Geometry2D, mesh: RectMesh2D) -> Dict:
        p = self.problem

        if p in ("hookes_2d", "von_mises"):
            return solve_elasticity(
                mesh, E=200e9, nu=0.3, traction_mag=1e7, traction_dir="x"
            )

        if p == "crash":
            T  = 2e-4 if self.fast else 5e-4
            dt = 2e-7 if self.fast else 5e-8
            # CFL check: c_wave = sqrt(E/rho), dt < dx/c
            c = np.sqrt(70e9 / 2700.0)
            dt_cfl = 0.5 * mesh.dx / c
            dt     = min(dt, dt_cfl)
            return solve_crash(
                mesh, E=70e9, nu=0.28, rho=2700.0,
                F_impact=1e8, T=T, dt=dt,
                store_every=max(1, int(T/(dt*30))),
            )

        if p == "ns_internal":
            nx = ny = self.nx_flow
            Re   = 100.0
            T    = 5.0  if self.fast else 25.0
            dt   = 5e-4 if self.fast else 2e-4
            return solve_lid_driven_cavity(nx, ny, Re, 1.0, T, dt)

        if p == "ns_external":
            nx_cyl = self.nx_flow * 2; ny_cyl = self.nx_flow
            Re     = 80.0
            T      = 3.0  if self.fast else 10.0
            dt     = 5e-4 if self.fast else 2e-4
            return solve_cylinder_flow(nx_cyl, ny_cyl, Re, 1.0, T, dt)

        raise ValueError(f"Unknown problem: {p}")

    # ------------------------------------------------------------------
    # Surrogate training dispatch
    # ------------------------------------------------------------------

    def _field_names(self) -> List[str]:
        return {
            "hookes_2d":   ["ux", "uy"],
            "von_mises":   ["ux", "uy", "von_mises"],
            "crash":       ["ux"],       # from final snapshot for simplicity
            "ns_internal": ["u", "v", "p"],
            "ns_external": ["u", "v", "p"],
        }.get(self.problem, ["u"])

    def _train_all(self, sim: Dict, fields: List[str]) -> List[Dict]:
        if not _TORCH:
            print("  [skip] PyTorch unavailable -- no surrogate training")
            return []

        # Flatten crash space-time snapshots into a (x,y,t) dataset
        if self.problem == "crash":
            sim = dict(sim)
            n_snaps = len(sim["times"])
            n_nodes = len(sim["x"])
            sim["x"]  = np.tile(sim["x"],   n_snaps)
            sim["y"]  = np.tile(sim["y"],   n_snaps)
            sim["t"]  = np.repeat(sim["times"], n_nodes)
            sim["ux"] = sim["ux_snaps"].ravel()
            for k in ("times", "ux_snaps", "vm_snaps"):
                sim.pop(k, None)

        _step("STEP 4 -- Dataset construction")
        X_tr, Y_tr, X_te, Y_te, valid_fields, stats = build_dataset(
            sim, fields, n_train=self.n_train, n_test=self.n_test
        )
        in_dim  = X_tr.shape[1]
        out_dim = Y_tr.shape[1]
        print(f"  Training points : {len(X_tr)}")
        print(f"  Test     points : {len(X_te)}")
        print(f"  Input dim       : {in_dim}  Output dim: {out_dim}")

        # Physics parameters for PDE loss
        Lx = float(sim.get("xs", [0, 1])[-1]) if "xs" in sim else 1.0
        Ly = float(sim.get("ys", [0, 1])[-1]) if "ys" in sim else 1.0

        # PDE loss functions are defined but not applied in this workflow
        # (model outputs are normalised; add pde_w > 0 + un-normalised outputs
        # to enable physics-informed training in a custom variant)
        pde_fn = pde_loss_none
        pde_w  = 0.0

        _step("STEP 5 -- Surrogate model training")
        # DeepONet branch input: scalar parameter (Re or load)
        param_val = float(sim.get("Re", sim.get("traction", 1.0)))
        branch_in = np.array([[param_val / 1000.0]], dtype=np.float32)  # normalised

        all_results = []
        for name in self.model_names:
            print(f"\n  Training {name.upper()} ...")
            hidden = [32, 64, 32] if self.fast else [64, 128, 128, 64]

            if name == "pinn":
                m = VanillaPINN(in_dim, out_dim, hidden)
            elif name == "fourier":
                m = FourierPINN(in_dim, out_dim,
                                n_fourier=64 if self.fast else 128,
                                hidden=[64, 64] if self.fast else [128, 128, 64])
            else:  # deeponet
                m = DeepONet(branch_in=1, trunk_in=in_dim, out_dim=out_dim,
                             p=32 if self.fast else 64)

            r = train_surrogate(
                model_name=name,
                model=m,
                X_tr=X_tr, Y_tr=Y_tr,
                X_te=X_te, Y_te=Y_te,
                pde_loss_fn=pde_fn,
                pde_weight=pde_w,
                n_col_pde=256 if self.fast else 1024,
                device=self.device,
                epochs=self.epochs,
                lr=1e-3,
                batch=256 if self.fast else 512,
                is_deeponet=(name == "deeponet"),
                branch_input=branch_in,
            )
            all_results.append(r)

        return all_results

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_json(self, sim: Dict, results: List[Dict], fields: List[str]):
        out = {"problem": self.problem, "fields": fields, "models": []}
        for r in results:
            if not r:
                continue
            m = compute_metrics(r["Yg"], r["Yp"], fields)
            out["models"].append({
                "name": r["model_name"],
                "elapsed_s": r["elapsed"],
                "metrics": {k: {mk: float(mv) for mk, mv in v.items()}
                            for k, v in m.items()},
            })
        path = self.save_dir / f"{self.problem}_results.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"  Saved {path}")


# =============================================================================
# 14.  MAIN
# =============================================================================

def main():
    args   = parse_args()
    device = _dev(args.device)

    if not _SCIPY:
        print("[ERROR] SciPy is required.  Install with:  pip install scipy")
        sys.exit(1)

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    problems = ALL_PROBLEMS if args.problem == "all" else [args.problem]

    all_problem_results = {}
    for prob in problems:
        pipe = WorkflowPipeline(
            problem=prob,
            model_names=args.models,
            fast=args.fast,
            device=device,
            epochs=args.epochs,
            save_dir=save_dir,
        )
        all_problem_results[prob] = pipe.run()

    _hdr("WORKFLOW COMPLETE")
    for prob, res in all_problem_results.items():
        fields  = res["fields"]
        results = res["results"]
        if results:
            print(f"\n  {prob.upper()}")
            m_all = [
                (r["model_name"],
                 np.mean([compute_metrics(r["Yg"], r["Yp"], fields)[f]["rel_l2"]
                          for f in fields]))
                for r in results if r
            ]
            m_all.sort(key=lambda x: x[1])
            for rank, (name, rl2) in enumerate(m_all, 1):
                print(f"    {rank}. {name:<14}  avg rel-L2 = {rl2:.4e}")
    print()


if __name__ == "__main__":
    main()
