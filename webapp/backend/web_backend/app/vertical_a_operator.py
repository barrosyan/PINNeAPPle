from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .geometry_sdf import GeneticSDF2D, make_grid, marching_squares_zero_level

# Real STL import (mesh-based boundary sampling)
import numpy as np
from pinneaple_data.stl_import import load_stl_bytes


def stl_to_boundary_points(
    stl_bytes: bytes,
    *,
    n_points: int = 2000,
    normalize: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """Sample boundary points from an STL mesh.

    Output: (N,3) tensor.
    """
    mesh = load_stl_bytes(stl_bytes)
    v = mesh.verts.astype(np.float32)
    f = mesh.faces.astype(np.int64)
    tris = v[f]  # (F,3,3)

    a = tris[:, 1] - tris[:, 0]
    b = tris[:, 2] - tris[:, 0]
    area = np.linalg.norm(np.cross(a, b), axis=-1).clip(1e-12)
    prob = area / area.sum()

    rng = np.random.default_rng(int(seed))
    idx = rng.choice(tris.shape[0], size=int(n_points), replace=True, p=prob)
    t = tris[idx]

    r1 = rng.random((n_points, 1), dtype=np.float32)
    r2 = rng.random((n_points, 1), dtype=np.float32)
    s1 = np.sqrt(r1)
    pts = (1 - s1) * t[:, 0] + s1 * (1 - r2) * t[:, 1] + s1 * r2 * t[:, 2]

    if normalize:
        c = pts.mean(axis=0, keepdims=True)
        pts = pts - c
        scale = float(np.max(np.linalg.norm(pts, axis=-1)))
        pts = pts / max(scale, 1e-6)

    return torch.from_numpy(pts)


@dataclass
class OperatorDataset:
    kind: str
    u_in: torch.Tensor
    y_true: torch.Tensor
    coords: Optional[torch.Tensor] = None
    coords_points: Optional[torch.Tensor] = None
    u_points: Optional[torch.Tensor] = None
    boundary_points: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    solver_name: Optional[str] = None
    particles_pos0: Optional[torch.Tensor] = None
    particles_posT: Optional[torch.Tensor] = None


def _gaussian_bump_2d(B: int, H: int, W: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(0, 1, H, device=device, dtype=dtype)
    xs = torch.linspace(0, 1, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    xx = xx[None, None, ...].expand(B, 1, H, W)
    yy = yy[None, None, ...].expand(B, 1, H, W)

    cx = torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 0.6 + 0.2
    cy = torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 0.6 + 0.2
    sx = torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 0.10 + 0.05
    sy = torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 0.10 + 0.05
    amp = torch.rand(B, 1, 1, 1, device=device, dtype=dtype) * 1.0 + 0.2

    g = amp * torch.exp(-((xx - cx) ** 2) / (2 * sx * sx) - ((yy - cy) ** 2) / (2 * sy * sy))
    return g


def _apply_mask_dirichlet(u: torch.Tensor, mask: torch.Tensor, boundary_value: float = 0.0) -> torch.Tensor:
    # mask True inside geometry; outside set boundary_value
    b = torch.tensor(boundary_value, device=u.device, dtype=u.dtype)
    return torch.where(mask[None, None, ...], u, b)


def _heat2d_step(u: torch.Tensor, alpha: float = 0.01, dt: float = 1e-3) -> torch.Tensor:
    lap = (
        torch.roll(u, 1, dims=-1) + torch.roll(u, -1, dims=-1) +
        torch.roll(u, 1, dims=-2) + torch.roll(u, -1, dims=-2) - 4.0 * u
    )
    return u + alpha * dt * lap


def _solver_step_hook(u0: torch.Tensor, solver_cfg: Dict[str, Any], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Run a *real* solver from pinneaple_solvers when possible.

    This function adapts the Vertical-A field representation to different solver
    IO contracts:

    - FDM (Poisson 2D MVP): expects forcing f(H,W) and Dirichlet bc(H,W) on edges.
      We map u0 -> f, bc=0.
    - LBM (D2Q9 MVP): expects f0(9,H,W). We map u0 -> rho and create equilibrium f0.
      Output is rho(H,W) after steps.
    - SPH (WCSPH MVP): expects particle positions pos0(N,2) and optional vel0.
      We sample particles inside the SDF mask and output a coarse density grid.

    Fallback:
      If anything fails, uses a simple heat-step on u0.
    """
    name = str(solver_cfg.get("name", "fdm")).lower().strip()

    def fallback():
        return _heat2d_step(u0)

    try:
        from pinneaple_solvers.registry import SolverRegistry
        from pinneaple_solvers.registry import register_all as register_all_solvers
        register_all_solvers()

        dt = float(solver_cfg.get("dt", 1e-3))
        steps = int(solver_cfg.get("steps", 20))

        # normalize shape to (B,C,H,W)
        if u0.ndim == 2:
            u = u0[None, None, ...]
        elif u0.ndim == 3:
            u = u0[:, None, ...]
        else:
            u = u0

        B, C, H, W = int(u.shape[0]), int(u.shape[1]), int(u.shape[2]), int(u.shape[3])
        device, dtype = u.device, u.dtype

        if name == "fdm":
            # FDM: solve Poisson with forcing f=u0 and bc=0 at edges
            solver = SolverRegistry.build("fdm", iters=int(solver_cfg.get("iters", 2000)), omega=float(solver_cfg.get("omega", 0.9)))
            dx = float(solver_cfg.get("dx", 1.0 / max(1, (W - 1))))
            dy = float(solver_cfg.get("dy", 1.0 / max(1, (H - 1))))
            y = torch.zeros_like(u)
            for b in range(B):
                for c in range(C):
                    f = u[b, c]
                    bc = torch.zeros_like(f)
                    out = solver(f, bc, dx=dx, dy=dy)
                    y[b, c] = out.result
            return y if u0.ndim == 4 else (y[:,0] if u0.ndim==3 else y[0,0])

        if name == "lbm":
            # LBM: build equilibrium distribution f0 from rho = 1 + eps*u0, u=0
            from pinneaple_solvers.lbm import _d2q9
            cvec, wts = _d2q9()
            cvec = cvec.to(device)
            wts = wts.to(device, dtype=dtype)

            omega = float(solver_cfg.get("omega", 1.0))
            solver = SolverRegistry.build("lbm", omega=omega)

            eps = float(solver_cfg.get("rho_eps", 0.1))
            y = torch.zeros_like(u)
            for b in range(B):
                # use first channel as scalar driver
                rho = (1.0 + eps * u[b, 0]).clamp_min(1e-6)
                ux = torch.zeros_like(rho)
                uy = torch.zeros_like(rho)
                u2 = ux**2 + uy**2

                f0 = torch.zeros((9, H, W), device=device, dtype=dtype)
                for i in range(9):
                    cu = 3.0 * (float(cvec[i,0]) * ux + float(cvec[i,1]) * uy)
                    f0[i] = wts[i] * rho * (1 + cu + 0.5*cu**2 - 1.5*u2)

                out = solver(f0, steps=int(steps), force=None)
                fT = out.result
                rhoT = torch.sum(fT, dim=0)  # (H,W)
                # write rhoT into channel 0 (replicate if C>1)
                for c in range(C):
                    y[b, c] = rhoT
            return y if u0.ndim == 4 else (y[:,0] if u0.ndim==3 else y[0,0])

        if name == "sph":
            # SPH: sample particles inside mask (if provided) and return density grid
            solver = SolverRegistry.build(
                "sph",
                h=float(solver_cfg.get("h", 0.04)),
                mass=float(solver_cfg.get("mass", 1.0)),
                rho0=float(solver_cfg.get("rho0", 1000.0)),
                c0=float(solver_cfg.get("c0", 20.0)),
                mu=float(solver_cfg.get("mu", 0.02)),
                gravity=solver_cfg.get("gravity", [0.0, -9.81]),
                boundary_lo=solver_cfg.get("boundary_lo", [0.0, 0.0]),
                boundary_hi=solver_cfg.get("boundary_hi", [1.0, 1.0]),
            )

            n_particles = int(solver_cfg.get("n_particles", 1200))

            # candidate sampling grid
            if mask is None:
                m = torch.ones((H, W), device=device, dtype=torch.bool)
            else:
                m = mask.to(device=device)
            ys = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
            xs = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            pts = torch.stack([xx[m], yy[m]], dim=1)  # (M,2)
            if pts.shape[0] < 10:
                return fallback()

            # random subset + small jitter
            idx = torch.randperm(pts.shape[0], device=device)[: min(n_particles, pts.shape[0])]
            pos0 = pts[idx]
            pos0 = (pos0 + 0.002 * torch.randn_like(pos0)).clamp(0.0, 1.0)

            out = solver(pos0, None, dt=float(dt), steps=int(steps))
            traj = out.result  # (steps+1, N, 2)
            posT = traj[-1]    # (N,2)

            # binning to grid density
            dens = torch.zeros((H, W), device=device, dtype=dtype)
            ix = (posT[:,0] * (W - 1)).round().long().clamp(0, W-1)
            iy = (posT[:,1] * (H - 1)).round().long().clamp(0, H-1)
            dens.index_put_((iy, ix), torch.ones_like(ix, dtype=dtype), accumulate=True)
            dens = dens / dens.max().clamp_min(1.0)

            y = dens[None, None, ...].repeat(B, C, 1, 1)
            return y if u0.ndim == 4 else (y[:,0] if u0.ndim==3 else y[0,0])

        return fallback()

    except Exception:
        return fallback()


def _sample_coords(N: int, device, dtype) -> torch.Tensor:
    return torch.rand(N, 2, device=device, dtype=dtype)


def _grid_to_points(u: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    B, C, H, W = u.shape
    xy = coords.clone()
    grid = xy[None, None, :, :] * 2.0 - 1.0
    grid = grid.to(device=u.device, dtype=u.dtype).expand(B, 1, coords.shape[0], 2)
    samp = F.grid_sample(u, grid, mode="bilinear", align_corners=False)  # (B,C,1,N)
    samp = samp.squeeze(2).transpose(1, 2)  # (B,N,C)
    return samp



def _sph_particles_density(mask: torch.Tensor, solver_cfg: Dict[str, Any], *, H: int, W: int, device, dtype):
    """Run SPH solver and return (dens_grid(H,W), pos0(N,2), posT(N,2))."""
    from pinneaple_solvers.registry import SolverRegistry
    from pinneaple_solvers.registry import register_all as register_all_solvers
    register_all_solvers()

    solver = SolverRegistry.build(
        "sph",
        h=float(solver_cfg.get("h", 0.04)),
        mass=float(solver_cfg.get("mass", 1.0)),
        rho0=float(solver_cfg.get("rho0", 1000.0)),
        c0=float(solver_cfg.get("c0", 20.0)),
        mu=float(solver_cfg.get("mu", 0.02)),
        gravity=solver_cfg.get("gravity", [0.0, -9.81]),
        boundary_lo=solver_cfg.get("boundary_lo", [0.0, 0.0]),
        boundary_hi=solver_cfg.get("boundary_hi", [1.0, 1.0]),
    )
    dt = float(solver_cfg.get("dt", 1e-3))
    steps = int(solver_cfg.get("steps", 20))
    n_particles = int(solver_cfg.get("n_particles", 1200))

    m = mask.to(device=device)
    ys = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([xx[m], yy[m]], dim=1)
    if pts.shape[0] < 10:
        raise ValueError("Not enough inside-domain points for SPH sampling.")
    idx = torch.randperm(pts.shape[0], device=device)[: min(n_particles, pts.shape[0])]
    pos0 = (pts[idx] + 0.002 * torch.randn((idx.shape[0], 2), device=device, dtype=dtype)).clamp(0.0, 1.0)

    out = solver(pos0, None, dt=dt, steps=steps)
    posT = out.result[-1]

    dens = torch.zeros((H, W), device=device, dtype=dtype)
    ix = (posT[:,0] * (W - 1)).round().long().clamp(0, W-1)
    iy = (posT[:,1] * (H - 1)).round().long().clamp(0, H-1)
    dens.index_put_((iy, ix), torch.ones_like(ix, dtype=dtype), accumulate=True)
    dens = dens / dens.max().clamp_min(1.0)
    return dens, pos0, posT


def build_dataset_for_operator(
    *,
    input_kind: str,
    batch_size: int = 8,
    H: int = 96,
    W: int = 96,
    L: int = 256,
    branch_dim: int = 128,
    n_coords: int = 1024,
    n_points: int = 2048,
    geometry_params: Optional[Dict[str, Any]] = None,
    solver_cfg: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> OperatorDataset:
    dev = torch.device(device)
    solver_cfg = solver_cfg or {"name": "fdm", "equation": "heat2d"}

    if input_kind == "grid_1d":
        u = torch.rand(batch_size, 1, L, device=dev, dtype=dtype)
        lap = torch.roll(u, 1, dims=-1) + torch.roll(u, -1, dims=-1) - 2.0 * u
        y = u + 0.01 * 1e-3 * lap
        return OperatorDataset(kind="grid_1d", u_in=u.detach().cpu(), y_true=y.detach().cpu())

    # Genetic geometry via SDF + boundary sampling via marching squares
    sdf_model = GeneticSDF2D(**(geometry_params or {}))
    xx, yy = make_grid(H, W, dev, dtype)
    sdf = sdf_model.sdf(xx, yy)  # (H,W)
    mask = (sdf <= 0)            # inside domain

    boundary = marching_squares_zero_level(sdf, xx, yy)  # (M,2)

    u0 = _gaussian_bump_2d(batch_size, H, W, dev, dtype)
    u0 = _apply_mask_dirichlet(u0, mask, boundary_value=0.0)

    # Use solver hook (fdm/lbm/sph) if exists; fallback else
    u1 = _solver_step_hook(u0, solver_cfg, mask=mask)
    u1 = _apply_mask_dirichlet(u1, mask, boundary_value=0.0)

    if input_kind == "operator_branch_trunk":
        coords = _sample_coords(n_coords, dev, dtype)
        y_pts = _grid_to_points(u1, coords)
        flat = u0.view(batch_size, -1)
        stride = max(1, flat.shape[1] // branch_dim)
        idx = torch.arange(0, stride * branch_dim, stride, device=dev).clamp_max(flat.shape[1] - 1)
        u_branch = flat[:, idx]
        return OperatorDataset(
            kind="branch_trunk",
            u_in=u_branch.detach().cpu(),
            y_true=y_pts.detach().cpu(),
            coords=coords.detach().cpu(),
            boundary_points=boundary.detach().cpu(),
            mask=mask.detach().cpu(),
            solver_name=str(solver_cfg.get('name','')).lower().strip(),
            particles_pos0=particles_pos0,
            particles_posT=particles_posT,
        )

    if input_kind == "points":
        coords_pts = _sample_coords(n_points, dev, dtype)
        u_pts = _grid_to_points(u0, coords_pts)
        y_pts = _grid_to_points(u1, coords_pts)
        return OperatorDataset(
            kind="points",
            u_in=u0.detach().cpu(),
            y_true=y_pts.detach().cpu(),
            coords_points=coords_pts.detach().cpu(),
            u_points=u_pts.detach().cpu(),
            boundary_points=boundary.detach().cpu(),
            mask=mask.detach().cpu(),
            solver_name=str(solver_cfg.get('name','')).lower().strip(),
            particles_pos0=particles_pos0,
            particles_posT=particles_posT,
        )

    # grid default (2D)
    return OperatorDataset(
        kind="grid_2d",
        u_in=u0.detach().cpu(),
        y_true=u1.detach().cpu(),
        boundary_points=boundary.detach().cpu(),
        mask=mask.detach().cpu(),
    )


def train_operator_model(
    model,
    adapter,
    dataset: OperatorDataset,
    *,
    epochs: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
    emit=None,
    job_id: str = "",
) -> Dict[str, Any]:
    dev = torch.device(device)
    model.to(dev)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    u_in = dataset.u_in.to(dev)
    y_true = dataset.y_true.to(dev)

    coords = dataset.coords.to(dev) if dataset.coords is not None else None
    coords_points = dataset.coords_points.to(dev) if dataset.coords_points is not None else None
    u_points = dataset.u_points.to(dev) if dataset.u_points is not None else None

    best = {"loss": float("inf")}
    for ep in range(int(epochs)):
        opt.zero_grad(set_to_none=True)

        batch = {"y_true": y_true}
        if dataset.kind == "grid_1d":
            batch["u_grid_1d"] = u_in
        elif dataset.kind == "grid_2d":
            batch["u_grid"] = u_in
        elif dataset.kind == "branch_trunk":
            batch["u_branch"] = u_in
            batch["coords"] = coords
        elif dataset.kind == "points":
            batch["u_points"] = u_points
            batch["coords_points"] = coords_points
        else:
            batch["u_grid"] = u_in

        out = adapter.forward_batch(model, batch)

        if hasattr(out, "losses") and isinstance(out.losses, dict) and ("total" in out.losses):
            loss = out.losses["total"]
        else:
            pred = out.y if hasattr(out, "y") else out
            loss = torch.mean((pred - y_true) ** 2)

        loss.backward()
        opt.step()

        l = float(loss.detach().cpu())
        if l < best["loss"]:
            best = {"loss": l, "epoch": ep}

        if emit and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            emit(job_id, f"Vertical A training: epoch {ep+1}/{epochs} loss={l:.6g}")

    return {"best": best}


def render_preview(dataset: OperatorDataset, pred: torch.Tensor, out_path: str) -> None:
    pred = pred.detach().cpu()
    y_true = dataset.y_true.detach().cpu()

    solver = (dataset.solver_name or "").lower().strip()

    if dataset.kind == "grid_2d" and solver == "sph" and dataset.particles_posT is not None:
        plt.figure(figsize=(16,4))
        a = y_true[0,0].numpy()
        b = pred[0,0].numpy()

        plt.subplot(1,4,1); plt.title("y_true (density)"); plt.imshow(a); plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1,4,2); plt.title("y_pred"); plt.imshow(b); plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1,4,3); plt.title("particles (final)")
        pts = dataset.particles_posT.numpy()
        plt.scatter(pts[:,0], pts[:,1], s=2, alpha=0.8)
        plt.xlim(0,1); plt.ylim(0,1)

        plt.subplot(1,4,4); plt.title("boundary (marching squares)")
        if dataset.boundary_points is not None and dataset.boundary_points.numel() > 0:
            bp = dataset.boundary_points.numpy()
            plt.scatter(bp[:,0], bp[:,1], s=2)
        plt.xlim(0,1); plt.ylim(0,1)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return

    plt.figure(figsize=(12,4))
    if dataset.kind == "grid_2d":
        a = y_true[0,0].numpy()
        b = pred[0,0].numpy()
        plt.subplot(1,3,1); plt.title("y_true"); plt.imshow(a); plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(1,3,2); plt.title("y_pred"); plt.imshow(b); plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(1,3,3); plt.title("boundary (marching squares)")
        if dataset.boundary_points is not None and dataset.boundary_points.numel() > 0:
            bp = dataset.boundary_points.numpy()
            plt.scatter(bp[:,0], bp[:,1], s=2)
        plt.xlim(0,1); plt.ylim(0,1)
    elif dataset.kind == "grid_1d":
        plt.subplot(1,2,1); plt.title("y_true"); plt.plot(y_true[0,0].numpy())
        plt.subplot(1,2,2); plt.title("y_pred"); plt.plot(pred[0,0].numpy())
    else:
        plt.subplot(1,2,1); plt.title("y_true (points)"); plt.plot(y_true[0,:,0].numpy()[:400])
        plt.subplot(1,2,2); plt.title("y_pred (points)"); plt.plot(pred[0,:,0].numpy()[:400])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

