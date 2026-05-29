import os
import json
import hashlib
import threading
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt
import trimesh

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from scipy.spatial import cKDTree

from pinneaple_models.pinns.vanilla import VanillaPINN
from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.metrics import default_metrics

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(BASE_DIR, "runs")
UPLOAD_DIR = os.path.join(RUNS_DIR, "uploads")
REF_DIR = os.path.join(RUNS_DIR, "reference")
OUT_DIR = os.path.join(RUNS_DIR, "outputs")
CKPT_DIR = os.path.join(RUNS_DIR, "checkpoints")
for _d in [UPLOAD_DIR, REF_DIR, OUT_DIR, CKPT_DIR]:
    os.makedirs(_d, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
jobs_status: Dict[str, Dict[str, Any]] = {}


# =========================================================
# Config
# =========================================================
@dataclass
class Cfg:
    # voxelized geometry proxy for domain / boundary sampling
    nx: int = 72
    ny: int = 48
    nz: int = 24

    # train samples
    n_col: int = 50000
    n_bc: int = 12000
    n_data: int = 0

    # model
    hidden: tuple = (256, 256, 256, 256)
    activation: str = "tanh"

    # training
    epochs: int = 300
    lr: float = 1e-3
    batch_train: int = 1024
    batch_val: int = 2048

    # material
    young_modulus: float = 210e9
    poisson_ratio: float = 0.30

    # geometric BCs for v1
    support_axis: str = "x"
    support_frac: float = 0.10
    load_axis: str = "x"
    load_frac: float = 0.10
    traction_dir: tuple = (0.0, 0.0, -1.0)
    traction_value: float = 1.0
    body_force: tuple = (0.0, 0.0, 0.0)

    # weights
    w_pde: float = 1.0
    w_bc_dirichlet: float = 20.0
    w_bc_neumann: float = 1.0
    w_data: float = 10.0


cfg_default = Cfg()


# =========================================================
# Model
# =========================================================
class Wrap(nn.Module):
    def __init__(self, pinn: nn.Module):
        super().__init__()
        self.pinn = pinn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pinn(x)
        return out.y if hasattr(out, "y") else out


def build_model(cfg: Cfg) -> nn.Module:
    core = VanillaPINN(
        in_dim=3,
        out_dim=3,
        hidden=cfg.hidden,
        activation=cfg.activation,
    )
    return Wrap(core).to(DEVICE)


# =========================================================
# Mesh helpers
# =========================================================
def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    scale = np.max(mesh.bounding_box.extents)
    if scale > 0:
        mesh.apply_scale(1.0 / scale)
    return mesh


def scalar_to_rgb(values: np.ndarray, cmap_name: str = "viridis") -> Tuple[np.ndarray, Tuple[float, float]]:
    values = values.astype(np.float32)
    vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if abs(vmax - vmin) < 1e-12:
        t = np.zeros_like(values)
    else:
        t = (values - vmin) / (vmax - vmin)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(np.clip(t, 0.0, 1.0))
    rgb = (rgba[:, :3] * 255).astype(np.uint8)
    return rgb, (vmin, vmax)


def export_ply_with_vertex_colors(mesh: trimesh.Trimesh, colors_rgb: np.ndarray, out_path: str) -> str:
    mesh = mesh.copy()
    rgba = np.concatenate([colors_rgb, 255 * np.ones((len(colors_rgb), 1), dtype=np.uint8)], axis=1)
    mesh.visual.vertex_colors = rgba
    mesh.export(out_path)
    return out_path


# =========================================================
# Domain masks
# =========================================================
def build_voxel_grid(mesh: trimesh.Trimesh, cfg: Cfg):
    bmin, bmax = mesh.bounds
    xs = np.linspace(bmin[0], bmax[0], cfg.nx, dtype=np.float32)
    ys = np.linspace(bmin[1], bmax[1], cfg.ny, dtype=np.float32)
    zs = np.linspace(bmin[2], bmax[2], cfg.nz, dtype=np.float32)

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")
    pts = np.stack([XX.reshape(-1), YY.reshape(-1), ZZ.reshape(-1)], axis=1).astype(np.float32)

    shape = (cfg.ny, cfg.nx, cfg.nz)
    dx = float(xs[1] - xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1] - ys[0]) if len(ys) > 1 else 1.0
    dz = float(zs[1] - zs[0]) if len(zs) > 1 else 1.0
    pitch = min(dx, dy, dz)

    return pts, (xs, ys, zs), shape, (bmin.astype(np.float32), bmax.astype(np.float32)), pitch


def inside_mask_voxelized(mesh: trimesh.Trimesh, pts: np.ndarray, shape, pitch: float) -> np.ndarray:
    vg = mesh.voxelized(pitch=pitch).fill()
    inside = vg.is_filled(pts)
    return inside.reshape(shape)


def boundary_mask(mask_solid: np.ndarray) -> np.ndarray:
    ny, nx, nz = mask_solid.shape
    b = np.zeros_like(mask_solid, dtype=bool)
    neigh = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for dy, dx, dz in neigh:
        shifted = np.zeros_like(mask_solid, dtype=bool)
        ys = slice(max(0, dy), ny + min(0, dy))
        xs = slice(max(0, dx), nx + min(0, dx))
        zs = slice(max(0, dz), nz + min(0, dz))
        ys2 = slice(max(0, -dy), ny + min(0, -dy))
        xs2 = slice(max(0, -dx), nx + min(0, -dx))
        zs2 = slice(max(0, -dz), nz + min(0, -dz))
        shifted[ys, xs, zs] = mask_solid[ys2, xs2, zs2]
        b |= mask_solid & (~shifted)
    return b


# =========================================================
# Structural problem
# =========================================================
def axis_index(axis: str) -> int:
    lut = {"x": 0, "y": 1, "z": 2}
    if axis not in lut:
        raise ValueError(f"Invalid axis '{axis}'. Use x, y or z.")
    return lut[axis]


def classify_boundary_regions(mesh: trimesh.Trimesh, pts: np.ndarray, cfg: Cfg):
    bmin, bmax = mesh.bounds
    sup_ax = axis_index(cfg.support_axis)
    load_ax = axis_index(cfg.load_axis)

    Lsup = bmax[sup_ax] - bmin[sup_ax]
    Lload = bmax[load_ax] - bmin[load_ax]

    support_mask = pts[:, sup_ax] <= (bmin[sup_ax] + cfg.support_frac * Lsup)
    load_mask = pts[:, load_ax] >= (bmax[load_ax] - cfg.load_frac * Lload)

    # avoid overlap
    load_mask &= ~support_mask
    return support_mask, load_mask


def load_reference_data(path: str) -> Dict[str, np.ndarray]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path)
        out = {k: data[k].astype(np.float32) for k in data.files}
    elif ext == ".csv":
        arr = np.genfromtxt(path, delimiter=",", names=True)
        cols = arr.dtype.names or ()
        need = ["x", "y", "z", "ux", "uy", "uz"]
        if not all(c in cols for c in need):
            raise ValueError("CSV reference must include columns x,y,z,ux,uy,uz")
        out = {
            "xyz": np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32),
            "disp": np.stack([arr["ux"], arr["uy"], arr["uz"]], axis=1).astype(np.float32),
        }
    else:
        raise ValueError("Reference file must be .npz or .csv")

    if "xyz" not in out or "disp" not in out:
        raise ValueError("Reference data must contain 'xyz' [N,3] and 'disp' [N,3]")
    if out["xyz"].ndim != 2 or out["xyz"].shape[1] != 3:
        raise ValueError("reference xyz must have shape [N,3]")
    if out["disp"].ndim != 2 or out["disp"].shape[1] != 3:
        raise ValueError("reference disp must have shape [N,3]")
    return out


# =========================================================
# Dataset
# =========================================================
class Structural3DDataset(Dataset):
    def __init__(self, cfg: Cfg, mesh: trimesh.Trimesh, mask_pack: Dict[str, Any], ref_pack: Optional[Dict[str, np.ndarray]] = None, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)

        mask = mask_pack["mask"]
        bmask = mask_pack["bmask"]
        xs, ys, zs = mask_pack["grid"]
        bmin, bmax = mask_pack["bounds"]

        boundary_idx = np.argwhere(bmask)
        if boundary_idx.size == 0:
            raise RuntimeError("No boundary voxels found. Increase voxel resolution.")

        def idx_to_xyz(idx: np.ndarray) -> np.ndarray:
            iy, ix, iz = idx[:, 0], idx[:, 1], idx[:, 2]
            x = xs[ix]
            y = ys[iy]
            z = zs[iz]
            return np.stack([x, y, z], axis=1).astype(np.float32)

        boundary_xyz = idx_to_xyz(boundary_idx)
        support_mask, load_mask = classify_boundary_regions(mesh, boundary_xyz, cfg)
        support_pts = boundary_xyz[support_mask]
        load_pts = boundary_xyz[load_mask]

        if len(support_pts) == 0:
            raise RuntimeError("No support points found. Increase support_frac or change support_axis.")
        if len(load_pts) == 0:
            raise RuntimeError("No load points found. Increase load_frac or change load_axis.")

        n_fix = max(1, cfg.n_bc // 2)
        n_load = max(1, cfg.n_bc - n_fix)
        sel_fix = support_pts[rng.integers(0, len(support_pts), size=n_fix)]
        sel_load = load_pts[rng.integers(0, len(load_pts), size=n_load)]

        def sample_collocation(n: int) -> np.ndarray:
            pts_list = []
            attempts = 0
            while sum(p.shape[0] for p in pts_list) < n:
                attempts += 1
                m = max(20000, n)
                cand = rng.random((m, 3)).astype(np.float32)
                cand = cand * (bmax - bmin) + bmin

                ix = np.clip(((cand[:, 0] - bmin[0]) / (bmax[0] - bmin[0] + 1e-12) * (cfg.nx - 1)).round().astype(int), 0, cfg.nx - 1)
                iy = np.clip(((cand[:, 1] - bmin[1]) / (bmax[1] - bmin[1] + 1e-12) * (cfg.ny - 1)).round().astype(int), 0, cfg.ny - 1)
                iz = np.clip(((cand[:, 2] - bmin[2]) / (bmax[2] - bmin[2] + 1e-12) * (cfg.nz - 1)).round().astype(int), 0, cfg.nz - 1)

                keep = mask[iy, ix, iz]
                pts_list.append(cand[keep])
                if attempts > 40:
                    break
            pts = np.concatenate(pts_list, axis=0)
            if len(pts) < n:
                raise RuntimeError("Unable to sample enough interior collocation points.")
            return pts[:n].astype(np.float32)

        x_col = sample_collocation(cfg.n_col)

        self.x_col = torch.from_numpy(x_col)
        self.x_bc_fix = torch.from_numpy(sel_fix.astype(np.float32))
        self.y_bc_fix = torch.zeros((len(sel_fix), 3), dtype=torch.float32)
        self.x_bc_load = torch.from_numpy(sel_load.astype(np.float32))

        self.has_data = ref_pack is not None and cfg.n_data > 0
        if self.has_data:
            xyz_ref = ref_pack["xyz"].astype(np.float32)
            disp_ref = ref_pack["disp"].astype(np.float32)
            take = min(cfg.n_data, len(xyz_ref))
            sel = rng.choice(len(xyz_ref), size=take, replace=len(xyz_ref) < take)
            self.x_data = torch.from_numpy(xyz_ref[sel])
            self.y_data = torch.from_numpy(disp_ref[sel])
        else:
            self.x_data = torch.zeros((1, 3), dtype=torch.float32)
            self.y_data = torch.zeros((1, 3), dtype=torch.float32)

        self.N = len(self.x_col)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        j_fix = i % len(self.x_bc_fix)
        j_load = i % len(self.x_bc_load)
        j_data = i % len(self.x_data)
        return {
            "x": self.x_col[i],
            "x_col": self.x_col[i],
            "x_bc_fix": self.x_bc_fix[j_fix],
            "y_bc_fix": self.y_bc_fix[j_fix],
            "x_bc_load": self.x_bc_load[j_load],
            "x_data": self.x_data[j_data],
            "y_data": self.y_data[j_data],
        }


def dict_collate(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


# =========================================================
# Mechanics operators
# =========================================================
mse = nn.MSELoss()


def grad_scalar(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


def jacobian_u(u: torch.Tensor, x: torch.Tensor):
    ux = u[:, 0:1]
    uy = u[:, 1:2]
    uz = u[:, 2:3]
    return grad_scalar(ux, x), grad_scalar(uy, x), grad_scalar(uz, x)


def strain_tensor(u: torch.Tensor, x: torch.Tensor):
    gux, guy, guz = jacobian_u(u, x)
    exx = gux[:, 0:1]
    eyy = guy[:, 1:2]
    ezz = guz[:, 2:3]
    exy = 0.5 * (gux[:, 1:2] + guy[:, 0:1])
    exz = 0.5 * (gux[:, 2:3] + guz[:, 0:1])
    eyz = 0.5 * (guy[:, 2:3] + guz[:, 1:2])
    return exx, eyy, ezz, exy, exz, eyz


def stress_tensor(u: torch.Tensor, x: torch.Tensor, E: float, nu: float):
    exx, eyy, ezz, exy, exz, eyz = strain_tensor(u, x)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    trace = exx + eyy + ezz
    sxx = lam * trace + 2 * mu * exx
    syy = lam * trace + 2 * mu * eyy
    szz = lam * trace + 2 * mu * ezz
    sxy = 2 * mu * exy
    sxz = 2 * mu * exz
    syz = 2 * mu * eyz
    return sxx, syy, szz, sxy, sxz, syz


def div_stress(u: torch.Tensor, x: torch.Tensor, E: float, nu: float, body_force=(0.0, 0.0, 0.0)):
    sxx, syy, szz, sxy, sxz, syz = stress_tensor(u, x, E, nu)
    dsxx = grad_scalar(sxx, x)
    dsyy = grad_scalar(syy, x)
    dszz = grad_scalar(szz, x)
    dsxy = grad_scalar(sxy, x)
    dsxz = grad_scalar(sxz, x)
    dsyz = grad_scalar(syz, x)
    rx = dsxx[:, 0:1] + dsxy[:, 1:2] + dsxz[:, 2:3] + body_force[0]
    ry = dsxy[:, 0:1] + dsyy[:, 1:2] + dsyz[:, 2:3] + body_force[1]
    rz = dsxz[:, 0:1] + dsyz[:, 1:2] + dszz[:, 2:3] + body_force[2]
    return rx, ry, rz


def make_loss_fn(cfg: Cfg):
    traction_dir = np.asarray(cfg.traction_dir, dtype=np.float32)
    norm = np.linalg.norm(traction_dir)
    if norm < 1e-12:
        traction_dir = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        traction_dir = traction_dir / norm

    def loss_fn(model, y_hat, batch):
        device = next(model.parameters()).device
        tdir = torch.tensor(traction_dir, device=device).view(1, 3)

        # PDE equilibrium
        xcol = batch["x_col"].to(device).clone().detach().requires_grad_(True)
        with torch.enable_grad():
            ucol = model(xcol)
            rx, ry, rz = div_stress(
                ucol,
                xcol,
                E=cfg.young_modulus,
                nu=cfg.poisson_ratio,
                body_force=cfg.body_force,
            )
            l_pde = torch.mean(rx ** 2 + ry ** 2 + rz ** 2)

        # Fixed support Dirichlet BC
        xfix = batch["x_bc_fix"].to(device)
        yfix = batch["y_bc_fix"].to(device)
        ufix = model(xfix)
        l_fix = mse(ufix, yfix)

        # Simple geometric load proxy for v1
        xload = batch["x_bc_load"].to(device)
        uload = model(xload)
        proj = torch.sum(uload * tdir, dim=1, keepdim=True)
        l_load = -cfg.traction_value * torch.mean(proj)

        # Optional FEM supervision for v2
        xdt = batch["x_data"].to(device)
        ydt = batch["y_data"].to(device)
        udt = model(xdt)
        l_data = mse(udt, ydt)

        total = (
            cfg.w_pde * l_pde
            + cfg.w_bc_dirichlet * l_fix
            + cfg.w_bc_neumann * l_load
            + (cfg.w_data * l_data if cfg.n_data > 0 else 0.0 * l_data)
        )
        return {
            "total": total,
            "pde": l_pde.detach(),
            "fix": l_fix.detach(),
            "load": l_load.detach(),
            "data": l_data.detach(),
        }

    return loss_fn


# =========================================================
# Paths and reference files
# =========================================================
def job_paths(job_id: str):
    mesh_path = os.path.join(UPLOAD_DIR, f"{job_id}.stl")
    ref_npz = os.path.join(REF_DIR, f"{job_id}_ref.npz")
    ref_csv = os.path.join(REF_DIR, f"{job_id}_ref.csv")
    ckpt_path = os.path.join(CKPT_DIR, f"{job_id}_best.pt")
    meta_path = os.path.join(CKPT_DIR, f"{job_id}_meta.json")
    ply_path = os.path.join(OUT_DIR, f"{job_id}_disp.ply")
    compare_path = os.path.join(OUT_DIR, f"{job_id}_compare.json")
    return mesh_path, ref_npz, ref_csv, ckpt_path, meta_path, ply_path, compare_path


def find_reference_path(job_id: str) -> Optional[str]:
    _, ref_npz, ref_csv, _, _, _, _ = job_paths(job_id)
    if os.path.exists(ref_npz):
        return ref_npz
    if os.path.exists(ref_csv):
        return ref_csv
    return None


# =========================================================
# Training
# =========================================================
def run_async_train(job_id, data, cfg_proto):
    try:
        jobs_status[job_id] = {"status": "processing", "message": "Preparing structural POC..."}
        mesh_path, _, _, ckpt_path, meta_path, _, compare_path = job_paths(job_id)

        cfg = Cfg(**asdict(cfg_proto))
        for k in asdict(cfg_proto).keys():
            if k in data:
                current = getattr(cfg, k)
                value = data[k]
                if isinstance(current, tuple):
                    setattr(cfg, k, tuple(value))
                else:
                    setattr(cfg, k, type(current)(value))

        jobs_status[job_id] = {"status": "processing", "message": "Loading mesh..."}
        mesh = trimesh.load_mesh(mesh_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh = normalize_mesh(mesh)

        jobs_status[job_id] = {"status": "processing", "message": "Voxelizing domain and extracting simple geometric BCs..."}
        pts, grid, shape, bounds, pitch = build_voxel_grid(mesh, cfg)
        mask = inside_mask_voxelized(mesh, pts, shape, pitch=pitch)
        bmask = boundary_mask(mask)
        mask_pack = {
            "mask": mask,
            "bmask": bmask,
            "grid": grid,
            "shape": shape,
            "bounds": bounds,
            "pitch": pitch,
        }

        ref_path = find_reference_path(job_id)
        ref_pack = None
        if ref_path is not None:
            jobs_status[job_id] = {"status": "processing", "message": "Loading FEM reference data..."}
            ref_pack = load_reference_data(ref_path)
            if cfg.n_data <= 0:
                cfg.n_data = min(10000, len(ref_pack["xyz"]))

        jobs_status[job_id] = {"status": "processing", "message": "Sampling collocation, support and load regions..."}
        ds = Structural3DDataset(cfg, mesh, mask_pack, ref_pack=ref_pack, seed=7)
        train_loader = DataLoader(ds, batch_size=cfg.batch_train, shuffle=True, collate_fn=dict_collate)
        val_loader = DataLoader(ds, batch_size=cfg.batch_val, shuffle=False, collate_fn=dict_collate)

        model = build_model(cfg).to(DEVICE)
        loss_fn = make_loss_fn(cfg)
        trainer = Trainer(model=model, loss_fn=loss_fn, metrics=default_metrics())

        train_cfg = TrainConfig(
            epochs=cfg.epochs,
            lr=cfg.lr,
            device=DEVICE,
            log_dir=os.path.join(CKPT_DIR, "logs"),
            run_name=f"job_{job_id}",
            seed=7,
            deterministic=False,
            amp=False,
            save_best=True,
        )

        jobs_status[job_id] = {"status": "processing", "message": f"Training structural PINN for {cfg.epochs} epochs on {DEVICE}..."}
        out = trainer.fit(train_loader, val_loader, train_cfg)
        best_path = out.get("best_path")

        if best_path and os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=DEVICE)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
            torch.save(state, ckpt_path)

        metrics = None
        if ref_pack is not None and os.path.exists(ckpt_path):
            jobs_status[job_id] = {"status": "processing", "message": "Evaluating against FEM reference..."}
            metrics = compare_with_reference(model=None, ckpt_path=ckpt_path, cfg=cfg, ref_pack=ref_pack)
            with open(compare_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

        meta = {
            "job_id": job_id,
            "cfg": asdict(cfg),
            "problem_type": "3D_linear_elasticity_structural_poc",
            "version": 2 if ref_pack is not None else 1,
            "device": DEVICE,
            "pitch": float(pitch),
            "best_val": float(out.get("best_val")) if out.get("best_val") is not None else None,
            "ckpt_path": ckpt_path if os.path.exists(ckpt_path) else None,
            "reference_used": ref_pack is not None,
            "comparison": metrics,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        jobs_status[job_id] = {"status": "completed", "result": meta}
    except Exception as e:
        jobs_status[job_id] = {"status": "failed", "error": str(e)}


# =========================================================
# Inference / comparison
# =========================================================
def load_trained_model(cfg: Cfg, ckpt_path: str) -> nn.Module:
    model = build_model(cfg).to(DEVICE).eval()
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    return model


def predict_points(model: nn.Module, xyz: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for i in range(0, len(xyz), batch_size):
            xb = torch.from_numpy(xyz[i : i + batch_size].astype(np.float32)).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


def compare_with_reference(model: Optional[nn.Module], ckpt_path: Optional[str], cfg: Cfg, ref_pack: Dict[str, np.ndarray]):
    if model is None:
        if ckpt_path is None:
            raise ValueError("Either model or ckpt_path must be provided")
        model = load_trained_model(cfg, ckpt_path)
    xyz = ref_pack["xyz"].astype(np.float32)
    disp_ref = ref_pack["disp"].astype(np.float32)
    disp_pred = predict_points(model, xyz)

    err = disp_pred - disp_ref
    rmse_vec = np.sqrt(np.mean(err ** 2, axis=0))
    rmse_mag = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))))

    mag_ref = np.linalg.norm(disp_ref, axis=1)
    mag_pred = np.linalg.norm(disp_pred, axis=1)
    denom = float(np.sqrt(np.mean(mag_ref ** 2)) + 1e-12)
    rel_l2_disp_mag = float(np.sqrt(np.mean((mag_pred - mag_ref) ** 2)) / denom)

    return {
        "n_points": int(len(xyz)),
        "rmse_ux": float(rmse_vec[0]),
        "rmse_uy": float(rmse_vec[1]),
        "rmse_uz": float(rmse_vec[2]),
        "rmse_disp_vector": rmse_mag,
        "rel_l2_disp_magnitude": rel_l2_disp_mag,
        "ref_disp_mag_range": {
            "min": float(np.min(mag_ref)),
            "max": float(np.max(mag_ref)),
        },
        "pred_disp_mag_range": {
            "min": float(np.min(mag_pred)),
            "max": float(np.max(mag_pred)),
        },
    }


# =========================================================
# API
# =========================================================
@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    ext = os.path.splitext(f.filename or "")[1].lower()
    if ext not in {".stl"}:
        return jsonify({"error": "upload expects STL mesh. Convert SLDPRT/STEP to STL first."}), 400

    content = f.read()
    job_id = hashlib.md5(content).hexdigest()
    f.seek(0)

    mesh_path, _, _, _, _, _, _ = job_paths(job_id)
    f.save(mesh_path)
    jobs_status[job_id] = {"status": "uploaded"}
    return jsonify({"job_id": job_id})


@app.post("/api/upload_reference")
def upload_reference():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    job_id = request.form.get("job_id")
    if not job_id:
        return jsonify({"error": "missing job_id"}), 400

    f = request.files["file"]
    ext = os.path.splitext(f.filename or "")[1].lower()
    if ext not in {".npz", ".csv"}:
        return jsonify({"error": "reference must be .npz or .csv"}), 400

    _, ref_npz, ref_csv, _, _, _, _ = job_paths(job_id)
    target = ref_npz if ext == ".npz" else ref_csv
    f.save(target)
    # validate immediately
    load_reference_data(target)
    return jsonify({"job_id": job_id, "reference_path": target, "status": "reference_uploaded"})


@app.post("/api/train")
def train():
    data = request.get_json(force=True)
    job_id = data.get("job_id")
    if not job_id:
        return jsonify({"error": "missing job_id"}), 400

    mesh_path, _, _, ckpt_path, meta_path, _, _ = job_paths(job_id)
    if not os.path.exists(mesh_path):
        return jsonify({"error": "unknown job_id, upload STL first"}), 404

    if os.path.exists(ckpt_path):
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                result_meta = json.load(f)
        else:
            result_meta = {"job_id": job_id, "best_val": 0.0}
        return jsonify({
            "job_id": job_id,
            "status": "completed",
            "message": "Model found in cache (hash match).",
            "result": result_meta,
        }), 200

    thread = threading.Thread(target=run_async_train, args=(job_id, data, cfg_default), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id, "status": "started"}), 202


@app.get("/api/status/<job_id>")
def get_status(job_id):
    mesh_path, _, _, _, meta_path, _, _ = job_paths(job_id)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return jsonify({"status": "completed", "result": json.load(f)})
    if job_id in jobs_status:
        return jsonify(jobs_status[job_id])
    if os.path.exists(mesh_path):
        return jsonify({"status": "idle", "message": "Server restarted. Train again."})
    return jsonify({"status": "unknown", "error": "Job ID not found."}), 404


@app.post("/api/infer")
def infer():
    data = request.get_json(force=True)
    job_id = data.get("job_id")
    if not job_id:
        return jsonify({"error": "missing job_id"}), 400

    mesh_path, _, _, ckpt_path, meta_path, ply_path, _ = job_paths(job_id)
    if not os.path.exists(mesh_path):
        return jsonify({"error": "unknown job_id"}), 404
    if not os.path.exists(ckpt_path):
        return jsonify({"error": "trained checkpoint not found"}), 404

    cfg = cfg_default
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = Cfg(**meta.get("cfg", asdict(cfg_default)))

    mesh = trimesh.load_mesh(mesh_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = normalize_mesh(mesh)

    model = load_trained_model(cfg, ckpt_path)
    V = mesh.vertices.astype(np.float32)
    U = predict_points(model, V)
    disp_mag = np.linalg.norm(U, axis=1)

    rgb, (vmin, vmax) = scalar_to_rgb(disp_mag, cmap_name="viridis")
    export_ply_with_vertex_colors(mesh, rgb, ply_path)

    return jsonify({
        "job_id": job_id,
        "ply_url": f"/api/result/{job_id}",
        "field": "displacement_magnitude",
        "range": {"min": vmin, "max": vmax},
    })


@app.post("/api/compare")
def compare():
    data = request.get_json(force=True)
    job_id = data.get("job_id")
    if not job_id:
        return jsonify({"error": "missing job_id"}), 400

    _, _, _, ckpt_path, meta_path, _, compare_path = job_paths(job_id)
    ref_path = find_reference_path(job_id)
    if ref_path is None:
        return jsonify({"error": "reference file not found for this job"}), 404
    if not os.path.exists(ckpt_path):
        return jsonify({"error": "trained checkpoint not found"}), 404

    cfg = cfg_default
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = Cfg(**meta.get("cfg", asdict(cfg_default)))

    ref_pack = load_reference_data(ref_path)
    metrics = compare_with_reference(model=None, ckpt_path=ckpt_path, cfg=cfg, ref_pack=ref_pack)
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return jsonify({"job_id": job_id, "comparison": metrics})


@app.get("/api/result/<job_id>")
def result(job_id):
    _, _, _, _, _, ply_path, _ = job_paths(job_id)
    if not os.path.exists(ply_path):
        return jsonify({"error": "not found"}), 404
    return send_file(ply_path, mimetype="application/octet-stream", as_attachment=False)


@app.get("/api/reference_format")
def reference_format():
    return jsonify({
        "npz": {
            "required_arrays": {
                "xyz": ["N", 3],
                "disp": ["N", 3],
            },
            "optional_arrays": {
                "stress": ["N", 6],
            },
        },
        "csv": {
            "required_columns": ["x", "y", "z", "ux", "uy", "uz"],
        },
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
