"""
Vortex identification and flow visualization — Q-criterion, λ₂, vorticity.

Inspired by the Q-criterion iso-surface style used in production CFD post-processing
(e.g. the fastback/Ahmed-body visualisations from OpenFOAM / ParaView).

Functions
---------
Analysis (numpy / torch input):
  compute_vorticity_2d        — ω_z scalar field
  compute_q_criterion_2d      — Q = 0.5(|Ω|²-|S|²) on 2-D structured grid
  compute_q_criterion_3d      — Q on 3-D structured grid
  compute_lambda2_3d          — λ₂ vortex criterion (Jeong & Hussain 1995)

Visualization:
  plot_vorticity              — coloured vorticity map
  plot_q_criterion_2d         — filled Q contour with zero iso-line
  plot_q_criterion_3d         — 3-D iso-surface via marching cubes
  plot_vortex_identification  — one-shot: compute Q from (u, v) and plot
  plot_lbm_flow               — specialised 4-panel dashboard for LBM output
  plot_flow_panel             — pressure + |v| + Q + streamlines (4-panel)
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure

from .style import get_cmap, make_figure


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------

def _np(t):
    try:
        return t.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(t, dtype=float)


def _grad2d(field: np.ndarray, dx: float = 1.0, dy: float = 1.0):
    """Return (df/dx, df/dy) using second-order central differences."""
    dfdx = np.gradient(field, dx, axis=0)
    dfdy = np.gradient(field, dy, axis=1)
    return dfdx, dfdy


# ---------------------------------------------------------------------------
# Analysis — 2D
# ---------------------------------------------------------------------------

def compute_vorticity_2d(
    u: "ArrayLike",
    v: "ArrayLike",
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """
    z-component of vorticity: ω_z = ∂v/∂x − ∂u/∂y.

    u, v : 2-D arrays (nx, ny)
    Returns ω_z (nx, ny)
    """
    u_ = _np(u);  v_ = _np(v)
    dvdx = np.gradient(v_, dx, axis=0)
    dudy = np.gradient(u_, dy, axis=1)
    return dvdx - dudy


def compute_q_criterion_2d(
    u: "ArrayLike",
    v: "ArrayLike",
    dx: float = 1.0,
    dy: float = 1.0,
) -> np.ndarray:
    """
    Q-criterion for 2-D incompressible flow.

    Q = −(∂u/∂x)(∂v/∂y) − (∂v/∂x)(∂u/∂y)

    Q > 0 → rotation dominates strain → vortex core.
    u, v : 2-D arrays (nx, ny).  Returns Q (nx, ny).
    """
    u_ = _np(u);  v_ = _np(v)
    dudx = np.gradient(u_, dx, axis=0)
    dudy = np.gradient(u_, dy, axis=1)
    dvdx = np.gradient(v_, dx, axis=0)
    dvdy = np.gradient(v_, dy, axis=1)
    return -(dudx * dvdy) - (dvdx * dudy)


# ---------------------------------------------------------------------------
# Analysis — 3D
# ---------------------------------------------------------------------------

def compute_q_criterion_3d(
    u: "ArrayLike",
    v: "ArrayLike",
    w: "ArrayLike",
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> np.ndarray:
    """
    Q-criterion for 3-D flow:  Q = 0.5(‖Ω‖²_F − ‖S‖²_F)

    u, v, w : 3-D arrays (nx, ny, nz).  Returns Q (nx, ny, nz).
    """
    u_ = _np(u);  v_ = _np(v);  w_ = _np(w)

    dudx = np.gradient(u_, dx, axis=0)
    dudy = np.gradient(u_, dy, axis=1)
    dudz = np.gradient(u_, dz, axis=2)
    dvdx = np.gradient(v_, dx, axis=0)
    dvdy = np.gradient(v_, dy, axis=1)
    dvdz = np.gradient(v_, dz, axis=2)
    dwdx = np.gradient(w_, dx, axis=0)
    dwdy = np.gradient(w_, dy, axis=1)
    dwdz = np.gradient(w_, dz, axis=2)

    # Symmetric (strain) tensor components
    S11 = dudx;           S22 = dvdy;          S33 = dwdz
    S12 = 0.5*(dudy+dvdx); S13 = 0.5*(dudz+dwdx); S23 = 0.5*(dvdz+dwdy)
    norm_S2 = S11**2 + S22**2 + S33**2 + 2*(S12**2 + S13**2 + S23**2)

    # Antisymmetric (rotation) tensor
    W12 = 0.5*(dudy-dvdx); W13 = 0.5*(dudz-dwdx); W23 = 0.5*(dvdz-dwdy)
    norm_W2 = 2*(W12**2 + W13**2 + W23**2)

    return 0.5 * (norm_W2 - norm_S2)


def compute_lambda2_3d(
    u: "ArrayLike",
    v: "ArrayLike",
    w: "ArrayLike",
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> np.ndarray:
    """
    λ₂ vortex criterion (Jeong & Hussain, 1995).
    Vortex core: λ₂ < 0.

    Returns λ₂ field (nx, ny, nz).
    """
    u_ = _np(u);  v_ = _np(v);  w_ = _np(w)
    grads = {
        "dudx": np.gradient(u_, dx, axis=0), "dudy": np.gradient(u_, dy, axis=1),
        "dudz": np.gradient(u_, dz, axis=2), "dvdx": np.gradient(v_, dx, axis=0),
        "dvdy": np.gradient(v_, dy, axis=1), "dvdz": np.gradient(v_, dz, axis=2),
        "dwdx": np.gradient(w_, dx, axis=0), "dwdy": np.gradient(w_, dy, axis=1),
        "dwdz": np.gradient(w_, dz, axis=2),
    }
    g = grads
    nx, ny, nz = u_.shape

    # S² + Ω² tensor at each point (3×3 symmetric)
    S = np.array([
        [g["dudx"],                0.5*(g["dudy"]+g["dvdx"]),  0.5*(g["dudz"]+g["dwdx"])],
        [0.5*(g["dudy"]+g["dvdx"]), g["dvdy"],                 0.5*(g["dvdz"]+g["dwdy"])],
        [0.5*(g["dudz"]+g["dwdx"]), 0.5*(g["dvdz"]+g["dwdy"]), g["dwdz"]],
    ])  # (3, 3, nx, ny, nz)
    W = np.array([
        [np.zeros_like(u_),          0.5*(g["dudy"]-g["dvdx"]),  0.5*(g["dudz"]-g["dwdx"])],
        [0.5*(g["dvdx"]-g["dudy"]),  np.zeros_like(u_),           0.5*(g["dvdz"]-g["dwdy"])],
        [0.5*(g["dwdx"]-g["dudz"]),  0.5*(g["dwdy"]-g["dvdz"]),  np.zeros_like(u_)],
    ])

    # A = S²+Ω²: sum over middle index
    A = np.einsum("iknyz,jknyz->ijnyz", S, S) + np.einsum("iknyz,jknyz->ijnyz", W, W)

    # λ₂ is the middle eigenvalue of A(3×3) at each point — sort eigenvalues
    lambda2 = np.empty((nx, ny, nz))
    # Reshape for batch eigensolve
    A_flat = A.reshape(3, 3, -1).transpose(2, 0, 1)   # (N, 3, 3)
    eigs   = np.linalg.eigvalsh(A_flat)                # (N, 3) sorted ascending
    lambda2 = eigs[:, 1].reshape(nx, ny, nz)           # second-smallest = λ₂
    return lambda2


# ---------------------------------------------------------------------------
# Visualization — 2D
# ---------------------------------------------------------------------------

def plot_vorticity(
    x: "ArrayLike",
    y: "ArrayLike",
    omega: "ArrayLike",
    *,
    title: str = "Vorticity ω_z",
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    obstacle_mask: Optional["ArrayLike"] = None,
    figsize: Tuple = (10, 5),
    show: bool = False,
) -> Figure:
    """Coloured vorticity map, symmetric diverging colormap centred on zero."""
    x_ = _np(x).ravel();  y_ = _np(y).ravel();  omega_ = _np(omega)

    fig, ax = make_figure(figsize=figsize)
    if symmetric:
        vmax = np.abs(omega_).max()
        vmin = -vmax
    else:
        vmin, vmax = omega_.min(), omega_.max()

    cf = ax.contourf(
        _np(x).reshape(-1, _np(y).shape[-1]) if _np(x).ndim == 2 else
        np.meshgrid(np.unique(x_), np.unique(y_), indexing="ij")[0],
        np.meshgrid(np.unique(x_), np.unique(y_), indexing="ij")[1],
        omega_.reshape(len(np.unique(x_)), len(np.unique(y_))),
        levels=100, cmap=cmap, vmin=vmin, vmax=vmax,
    )
    if obstacle_mask is not None:
        mask_ = _np(obstacle_mask).astype(bool)
        ax.contourf(
            np.meshgrid(np.unique(x_), np.unique(y_), indexing="ij")[0],
            np.meshgrid(np.unique(x_), np.unique(y_), indexing="ij")[1],
            mask_.astype(float), levels=[0.5, 1.5], colors=["#333333"],
        )
    fig.colorbar(cf, ax=ax, label="ω_z")
    ax.set_aspect("equal"); ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if show: plt.show()
    return fig


def plot_q_criterion_2d(
    x: "ArrayLike",
    y: "ArrayLike",
    Q: "ArrayLike",
    vel_mag: Optional["ArrayLike"] = None,
    *,
    title: str = "Q-criterion",
    q_levels: int = 30,
    highlight_positive: bool = True,
    cmap_Q: str = "coolwarm",
    cmap_vel: str = "rainbow",
    obstacle_mask: Optional["ArrayLike"] = None,
    figsize: Tuple = (11, 5),
    show: bool = False,
) -> Figure:
    """
    2-D Q-criterion field.

    If vel_mag is provided, the background shows |v| and Q iso-lines are
    overlaid; otherwise Q itself is the filled contour.
    The zero iso-line (vortex boundary) is drawn as a white dashed curve.
    """
    x_ = _np(x);  y_ = _np(y);  Q_ = _np(Q)
    if x_.ndim == 1:
        X, Y = np.meshgrid(x_, y_, indexing="ij")
    else:
        X, Y = x_, y_
    Q_2d = Q_.reshape(X.shape)

    fig, ax = make_figure(figsize=figsize)

    if vel_mag is not None:
        vm = _np(vel_mag).reshape(X.shape)
        cf = ax.contourf(X, Y, vm, levels=80, cmap=get_cmap(cmap_vel))
        fig.colorbar(cf, ax=ax, label="|v|", pad=0.02)
        # Q iso-contours on top
        qmax = np.abs(Q_2d).max()
        ax.contourf(X, Y, Q_2d, levels=np.linspace(0, qmax, 15),
                    cmap=get_cmap("hot"), alpha=0.5)
    else:
        qmax = np.abs(Q_2d).max()
        cf = ax.contourf(X, Y, Q_2d, levels=q_levels,
                         cmap=get_cmap(cmap_Q), vmin=-qmax, vmax=qmax)
        fig.colorbar(cf, ax=ax, label="Q", pad=0.02)

    # Zero iso-line (vortex boundary)
    ax.contour(X, Y, Q_2d, levels=[0.0], colors=["white"], linewidths=0.8,
               linestyles="--")

    if obstacle_mask is not None:
        mask_2d = _np(obstacle_mask).reshape(X.shape).astype(float)
        ax.contourf(X, Y, mask_2d, levels=[0.5, 1.5], colors=["#1a1a2e"])

    ax.set_aspect("equal"); ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if show: plt.show()
    return fig


def plot_vortex_identification(
    x: "ArrayLike",
    y: "ArrayLike",
    u: "ArrayLike",
    v: "ArrayLike",
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    title: str = "Vortex Identification (Q-criterion)",
    obstacle_mask: Optional["ArrayLike"] = None,
    figsize: Tuple = (11, 5),
    show: bool = False,
) -> Figure:
    """
    One-shot wrapper: compute Q from velocity field and plot.
    Also overlays streamlines on the Q background.
    """
    u_ = _np(u);  v_ = _np(v)
    x_ = _np(x);  y_ = _np(y)
    Q = compute_q_criterion_2d(u_, v_, dx=dx, dy=dy)
    vel_mag = np.sqrt(u_**2 + v_**2)

    # Build 2D coordinate arrays
    if x_.ndim == 1 and y_.ndim == 1 and x_.shape != y_.shape:
        # x_ and y_ are axis vectors of different length → meshgrid
        xi, yi = x_, y_
        X, Y = np.meshgrid(xi, yi, indexing="ij")
    elif x_.ndim == 1:
        X, Y = np.meshgrid(x_, y_, indexing="ij")
        xi, yi = np.unique(x_), np.unique(y_)
    else:
        X, Y = x_, y_
        xi, yi = np.unique(x_.ravel()), np.unique(y_.ravel())
    Q_2d = Q.reshape(X.shape)

    fig, ax = make_figure(figsize=figsize)
    cf = ax.contourf(X, Y, Q_2d, levels=80, cmap="RdBu_r",
                     vmin=-np.abs(Q_2d).max(), vmax=np.abs(Q_2d).max())
    fig.colorbar(cf, ax=ax, label="Q", pad=0.02)
    ax.contour(X, Y, Q_2d, levels=[0.0], colors=["white"], linewidths=1.0,
               linestyles="--")

    # Streamlines on regular xi×yi grid — u_/v_ already on this grid
    Ui = u_.reshape(X.shape)
    Vi = v_.reshape(X.shape)
    mag = np.sqrt(Ui**2 + Vi**2)
    lw  = 1.0 * mag / (mag.max() + 1e-10) + 0.3
    ax.streamplot(xi, yi, Ui.T, Vi.T, color="white", linewidth=lw.T,
                  density=1.2, arrowsize=0.8)

    if obstacle_mask is not None:
        mask_2d = _np(obstacle_mask).reshape(X.shape).astype(float)
        ax.contourf(X, Y, mask_2d, levels=[0.5, 1.5], colors=["#1a1a2e"])

    ax.set_aspect("equal"); ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    if show: plt.show()
    return fig


# ---------------------------------------------------------------------------
# Visualization — 3D iso-surface
# ---------------------------------------------------------------------------

def plot_q_criterion_3d(
    x: "ArrayLike",
    y: "ArrayLike",
    z: "ArrayLike",
    Q: "ArrayLike",
    *,
    level: float = 0.01,
    vel_mag: Optional["ArrayLike"] = None,
    title: str = "Q-criterion iso-surface",
    cmap: str = "rainbow",
    alpha: float = 0.85,
    elev: float = 25.0,
    azim: float = -60.0,
    figsize: Tuple = (10, 7),
    show: bool = False,
) -> Figure:
    """
    3-D Q-criterion iso-surface coloured by velocity magnitude.

    Extracts the iso-surface Q = level using marching cubes and renders it
    with matplotlib's Poly3DCollection — the same style as OpenFOAM / ParaView
    Q-criterion plots of bluff-body wake flows.

    Requires scikit-image (pip install scikit-image).
    """
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise ImportError("pip install scikit-image  # required for 3-D iso-surfaces")

    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    Q_ = _np(Q)
    x_ = _np(x);  y_ = _np(y);  z_ = _np(z)

    if Q_.ndim != 3:
        raise ValueError(f"Q must be 3-D, got shape {Q_.shape}")

    # Marching cubes
    verts, faces, normals, _ = marching_cubes(Q_, level=level)

    # Map vertex indices to world coordinates
    shape = np.array(Q_.shape, dtype=float)
    vx = x_.ravel();  vy = y_.ravel();  vz = z_.ravel()
    # Assumes regular grid: map [0,nx-1] → [x.min, x.max]
    verts_world = verts.copy()
    verts_world[:, 0] = x_.min() + verts[:, 0] / (shape[0]-1) * (x_.max()-x_.min())
    verts_world[:, 1] = y_.min() + verts[:, 1] / (shape[1]-1) * (y_.max()-y_.min())
    verts_world[:, 2] = z_.min() + verts[:, 2] / (shape[2]-1) * (z_.max()-z_.min())

    # Color by velocity magnitude at vertices (trilinear interpolation)
    if vel_mag is not None:
        vm_ = _np(vel_mag)
        # Nearest-neighbour lookup on grid
        i_idx = np.clip(verts[:, 0].astype(int), 0, Q_.shape[0]-1)
        j_idx = np.clip(verts[:, 1].astype(int), 0, Q_.shape[1]-1)
        k_idx = np.clip(verts[:, 2].astype(int), 0, Q_.shape[2]-1)
        vert_colors = vm_[i_idx, j_idx, k_idx]
        vc_norm = (vert_colors - vert_colors.min()) / (vert_colors.ptp() + 1e-10)
        cm = plt.get_cmap(cmap)
        face_colors = cm(vc_norm[faces].mean(axis=1))
    else:
        face_colors = plt.get_cmap(cmap)(
            np.linspace(0, 1, len(faces))
        )

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(
        verts_world[faces], alpha=alpha,
        facecolor=face_colors, edgecolor="none",
    )
    ax.add_collection3d(poly)

    ax.set_xlim(x_.min(), x_.max())
    ax.set_ylim(y_.min(), y_.max())
    ax.set_zlim(z_.min(), z_.max())
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"{title}  (Q={level:.3g})")

    if vel_mag is not None:
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=mcolors.Normalize(vert_colors.min(), vert_colors.max()),
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.5, label="|v|")

    if show: plt.show()
    return fig


# ---------------------------------------------------------------------------
# LBM-specific dashboard
# ---------------------------------------------------------------------------

def plot_lbm_flow(
    ux: "ArrayLike",
    uy: "ArrayLike",
    rho: Optional["ArrayLike"] = None,
    obstacle_mask: Optional["ArrayLike"] = None,
    *,
    dx: float = 1.0,
    dy: float = 1.0,
    title: str = "LBM Flow Field",
    figsize: Tuple = (18, 9),
    show: bool = False,
) -> Figure:
    """
    4-panel LBM dashboard:
      [0] Velocity magnitude  [1] Vorticity ω_z
      [2] Q-criterion         [3] Pressure (density proxy)
    Solid obstacles are masked in grey.
    """
    ux_ = _np(ux);  uy_ = _np(uy)
    vel_mag = np.sqrt(ux_**2 + uy_**2)
    Q       = compute_q_criterion_2d(ux_, uy_, dx=dx, dy=dy)
    omega_z = compute_vorticity_2d(ux_, uy_, dx=dx, dy=dy)

    nx, ny = ux_.shape
    X, Y = np.meshgrid(np.arange(nx)*dx, np.arange(ny)*dy, indexing="ij")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13)

    panels = [
        (vel_mag,  "rainbow",  "|v| (velocity magnitude)", None, None),
        (omega_z,  "RdBu_r",   "ω_z (vorticity)",  -np.abs(omega_z).max(), np.abs(omega_z).max()),
        (Q,        "coolwarm", "Q-criterion",        -np.abs(Q).max(), np.abs(Q).max()),
        (_np(rho) if rho is not None else vel_mag,
                   "viridis",  "ρ (density / pressure)" if rho is not None else "|v|", None, None),
    ]

    for ax, (field, cmap, label, vmin, vmax) in zip(axes.ravel(), panels):
        cf = ax.contourf(X, Y, field, levels=80, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(cf, ax=ax, label=label, pad=0.02)
        if obstacle_mask is not None:
            mask_ = _np(obstacle_mask).astype(float)
            ax.contourf(X, Y, mask_, levels=[0.5, 1.5], colors=["#2d2d2d"])
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.tick_params(labelsize=7)

    # Overlay zero Q iso-line on Q panel
    ax_Q = axes[1, 0]
    ax_Q.contour(X, Y, Q, levels=[0.0], colors=["white"], linewidths=0.8,
                 linestyles="--")

    fig.tight_layout()
    if show: plt.show()
    return fig


def plot_flow_panel(
    x: "ArrayLike",
    y: "ArrayLike",
    ux: "ArrayLike",
    uy: "ArrayLike",
    pressure: Optional["ArrayLike"] = None,
    obstacle_mask: Optional["ArrayLike"] = None,
    *,
    title: str = "Flow Field Analysis",
    figsize: Tuple = (18, 10),
    show: bool = False,
) -> Figure:
    """
    4-panel flow analysis: pressure, |v| + streamlines, Q-criterion, vorticity.
    Mirrors the post-processing style of commercial CFD dashboards.
    """
    ux_ = _np(ux);  uy_ = _np(uy)
    x_  = _np(x);   y_  = _np(y)

    if x_.ndim == 1 and y_.ndim == 1 and x_.shape != y_.shape:
        xi, yi = x_, y_
        X, Y = np.meshgrid(xi, yi, indexing="ij")
    elif x_.ndim == 1:
        X, Y = np.meshgrid(x_, y_, indexing="ij")
        xi, yi = np.unique(x_), np.unique(y_)
    else:
        X, Y = x_, y_
        xi, yi = np.unique(x_.ravel()), np.unique(y_.ravel())

    vel_mag = np.sqrt(ux_**2 + uy_**2)
    Q       = compute_q_criterion_2d(ux_, uy_)
    omega_z = compute_vorticity_2d(ux_, uy_)
    press   = _np(pressure) if pressure is not None else -vel_mag   # Bernoulli proxy

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, y=1.01)

    def _panel(ax, field, cmap, label, vmin=None, vmax=None):
        cf = ax.contourf(X, Y, field.reshape(X.shape), levels=80,
                         cmap=cmap, vmin=vmin, vmax=vmax)
        fig.colorbar(cf, ax=ax, pad=0.01, shrink=0.9)
        if obstacle_mask is not None:
            mask_ = _np(obstacle_mask).reshape(X.shape).astype(float)
            ax.contourf(X, Y, mask_, levels=[0.5, 1.5], colors=["#1a1a2e"])
        ax.set_aspect("equal"); ax.set_title(label)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.tick_params(labelsize=7)
        return cf

    _panel(axes[0, 0], press,   "coolwarm", "Pressure / Bernoulli proxy")
    _panel(axes[0, 1], vel_mag, "rainbow",  "|v| velocity magnitude")
    _panel(axes[1, 0], Q, "RdBu_r", "Q-criterion",
           vmin=-np.abs(Q).max(), vmax=np.abs(Q).max())
    axes[1, 0].contour(X, Y, Q.reshape(X.shape), levels=[0.0],
                       colors=["white"], linewidths=1.0, linestyles="--")
    _panel(axes[1, 1], omega_z, "seismic", "Vorticity ω_z",
           vmin=-np.abs(omega_z).max(), vmax=np.abs(omega_z).max())

    # Add streamlines on |v| panel (u_/v_ already on structured grid)
    Ui = ux_.reshape(X.shape); Vi = uy_.reshape(X.shape)
    mag = np.sqrt(Ui**2 + Vi**2)
    lw  = 1.2 * mag / (mag.max() + 1e-10) + 0.3
    axes[0, 1].streamplot(xi, yi, Ui.T, Vi.T, color="white",
                          linewidth=lw.T, density=1.0, arrowsize=0.7)

    fig.tight_layout()
    if show: plt.show()
    return fig
