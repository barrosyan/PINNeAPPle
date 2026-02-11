import os
import uuid
import json
import numpy as np
from dataclasses import dataclass
import threading
import hashlib
import traceback

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

import trimesh
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg

from pinneaple_models.pinns.vanilla import VanillaPINN
from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.metrics import default_metrics

# ---- NEW: symbolic physics (PINNFactory) + combined loss hook
from pinneaple_pinn.factory.pinn_factory import PINNFactory, PINNProblemSpec
from pinneaple_train.losses import CombinedLoss, SupervisedLoss, PhysicsLossHook

# =========================================================
# Flask
# =========================================================
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "runs", "uploads")
OUT_DIR = os.path.join(BASE_DIR, "runs", "outputs")
CKPT_DIR = os.path.join(BASE_DIR, "runs", "checkpoints")
TOOLS_DIR = os.path.join(BASE_DIR, "runs", "tools")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(TOOLS_DIR, exist_ok=True)

DEVICE = "cpu"

# global status registry (in-memory)
jobs_status = {}


# =========================================================
# Config
# =========================================================
@dataclass
class Cfg:
    # baseline voxel grid
    nx: int = 60
    ny: int = 45
    nz: int = 20

    # train samples
    n_col: int = 60000
    n_bc: int = 24000
    n_data: int = 24000

    # model
    hidden: tuple = (128, 128, 128, 128)
    activation: str = "tanh"

    # training
    epochs: int = 120
    lr: float = 1e-3
    batch_train: int = 1024
    batch_val: int = 2048

    # weights
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_data: float = 1.0


cfg_default = Cfg()


# =========================================================
# Model wrapper
# =========================================================
class Wrap(nn.Module):
    def __init__(self, pinn):
        super().__init__()
        self.pinn = pinn

    def forward(self, *inputs):
        """
        Compatível com:
          - forward(xyz) onde xyz é (B,3)
          - forward(x, y, z) onde cada um é (B,1)
        """
        if len(inputs) == 1:
            x = inputs[0]
        else:
            x = torch.cat(inputs, dim=1)

        out = self.pinn(x)
        return out.y if hasattr(out, "y") else out


def parse_hidden(hidden):
    # accepts list/tuple[int] or string "128,128,128"
    if isinstance(hidden, (list, tuple)):
        return [int(x) for x in hidden]
    if isinstance(hidden, str):
        s = hidden.strip()
        if not s:
            return [128, 128, 128, 128]
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    return [128, 128, 128, 128]


def build_model(cfg: Cfg):
    hidden = parse_hidden(cfg.hidden)
    core = VanillaPINN(in_dim=3, out_dim=1, hidden=hidden, activation=cfg.activation)
    model = Wrap(core).to(DEVICE)
    return model


# =========================================================
# Helpers: mesh normalize + export colored PLY
# =========================================================
def normalize_mesh(mesh: trimesh.Trimesh):
    mesh = mesh.copy()
    center = mesh.bounding_box.centroid
    mesh.apply_translation(-center)
    scale = np.max(mesh.bounding_box.extents)
    if scale > 0:
        mesh.apply_scale(1.0 / scale)
    return mesh


def scalar_to_rgb(values, cmap_name="viridis"):
    v = values.astype(np.float32)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if vmax - vmin < 1e-12:
        t = np.zeros_like(v)
    else:
        t = (v - vmin) / (vmax - vmin)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(t)  # Nx4 in [0,1]
    rgb = (rgba[:, :3] * 255).astype(np.uint8)
    return rgb, (vmin, vmax)


def export_ply_with_vertex_colors(mesh: trimesh.Trimesh, colors_rgb: np.ndarray, out_path: str):
    mesh = mesh.copy()
    rgba = np.concatenate([colors_rgb, 255*np.ones((len(colors_rgb), 1), dtype=np.uint8)], axis=1)
    mesh.visual.vertex_colors = rgba
    mesh.export(out_path)
    return out_path


# =========================================================
# Heat problem (Dirichlet heater patch vs cold boundary)
# =========================================================
@dataclass
class HeatProblem:
    heater_bbox_min: np.ndarray
    heater_bbox_max: np.ndarray
    T_cold: float = 0.0
    T_hot: float = 1.0


def make_default_heat_problem(mesh: trimesh.Trimesh) -> HeatProblem:
    bmin, bmax = mesh.bounds
    size = (bmax - bmin)
    cx = 0.5 * (bmin[0] + bmax[0])
    cy = 0.5 * (bmin[1] + bmax[1])
    z0 = bmin[2]

    heater_size = np.array([0.25*size[0], 0.25*size[1], 0.05*size[2]], dtype=np.float32)
    hmin = np.array([cx, cy, z0], dtype=np.float32) - 0.5 * heater_size
    hmax = np.array([cx, cy, z0], dtype=np.float32) + 0.5 * heater_size
    hmin[2] = bmin[2]
    hmax[2] = bmin[2] + 0.05*size[2]
    return HeatProblem(hmin.astype(np.float32), hmax.astype(np.float32), 0.0, 1.0)


def in_bbox(xyz: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    return np.all((xyz >= bmin) & (xyz <= bmax), axis=1)


# =========================================================
# Baseline solver: voxel + CG Laplace
# =========================================================
def build_voxel_grid(mesh: trimesh.Trimesh, cfg: Cfg):
    bmin, bmax = mesh.bounds
    xs = np.linspace(bmin[0], bmax[0], cfg.nx, dtype=np.float32)
    ys = np.linspace(bmin[1], bmax[1], cfg.ny, dtype=np.float32)
    zs = np.linspace(bmin[2], bmax[2], cfg.nz, dtype=np.float32)

    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="xy")  # (ny,nx,nz)
    pts = np.stack([XX.reshape(-1), YY.reshape(-1), ZZ.reshape(-1)], axis=1).astype(np.float32)

    shape = (cfg.ny, cfg.nx, cfg.nz)
    dx = float(xs[1]-xs[0]) if len(xs) > 1 else 1.0
    dy = float(ys[1]-ys[0]) if len(ys) > 1 else 1.0
    dz = float(zs[1]-zs[0]) if len(zs) > 1 else 1.0
    pitch = min(dx, dy, dz)

    return pts, (xs, ys, zs), shape, (bmin.astype(np.float32), bmax.astype(np.float32)), pitch


def inside_mask_voxelized(mesh: trimesh.Trimesh, pts: np.ndarray, shape, pitch: float):
    vg = mesh.voxelized(pitch=pitch).fill()
    inside = vg.is_filled(pts)
    return inside.reshape(shape)


def boundary_mask(mask_solid: np.ndarray):
    ny, nx, nz = mask_solid.shape
    b = np.zeros_like(mask_solid, dtype=bool)
    neigh = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for dy, dx, dz in neigh:
        shifted = np.zeros_like(mask_solid, dtype=bool)
        ys = slice(max(0,dy), ny+min(0,dy))
        xs = slice(max(0,dx), nx+min(0,dx))
        zs = slice(max(0,dz), nz+min(0,dz))
        ys2 = slice(max(0,-dy), ny+min(0,-dy))
        xs2 = slice(max(0,-dx), nx+min(0,-dx))
        zs2 = slice(max(0,-dz), nz+min(0,-dz))
        shifted[ys, xs, zs] = mask_solid[ys2, xs2, zs2]
        b |= mask_solid & (~shifted)
    return b


def solve_laplace_dirichlet(mesh, cfg: Cfg, prob: HeatProblem):
    pts, (xs,ys,zs), shape, (bmin,bmax), pitch = build_voxel_grid(mesh, cfg)

    mask = inside_mask_voxelized(mesh, pts, shape, pitch=pitch)
    bmask = boundary_mask(mask)

    heater = in_bbox(pts, prob.heater_bbox_min, prob.heater_bbox_max).reshape(shape)
    heater = heater & bmask

    T = np.full(shape, np.nan, dtype=np.float32)
    T[bmask] = prob.T_cold
    T[heater] = prob.T_hot

    unk = mask & (~bmask)
    idx_map = -np.ones(shape, dtype=np.int32)
    unk_indices = np.argwhere(unk)
    for k, (iy, ix, iz) in enumerate(unk_indices):
        idx_map[iy, ix, iz] = k

    N = unk_indices.shape[0]
    if N == 0:
        raise RuntimeError("No interior voxels found. Increase grid resolution or check STL scaling.")

    A = lil_matrix((N, N), dtype=np.float32)
    b = np.zeros((N,), dtype=np.float32)

    neigh = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for row, (iy, ix, iz) in enumerate(unk_indices):
        A[row, row] = 6.0
        for dy, dx, dz in neigh:
            jy, jx, jz = iy+dy, ix+dx, iz+dz
            if (0 <= jy < shape[0]) and (0 <= jx < shape[1]) and (0 <= jz < shape[2]) and mask[jy,jx,jz]:
                if bmask[jy,jx,jz]:
                    b[row] += T[jy,jx,jz]
                else:
                    col = idx_map[jy,jx,jz]
                    A[row, col] = -1.0
            else:
                b[row] += prob.T_cold

    sol, info = cg(A.tocsr(), b, x0=np.zeros_like(b), maxiter=2000, rtol=1e-6)
    if info != 0:
        print("[warn] CG did not fully converge. info=", info)

    T[unk] = sol.astype(np.float32)
    T[~mask] = np.nan

    return {
        "T": T, "mask": mask, "bmask": bmask, "heater": heater,
        "grid": (xs,ys,zs), "shape": shape, "bounds": (bmin,bmax),
        "pitch": pitch,
    }


# =========================================================
# Dataset: collocation + bc + baseline-supervised
# =========================================================
class Heat3DDataset(Dataset):
    def __init__(self, cfg: Cfg, baseline_pack: dict, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)

        Tvol = baseline_pack["T"]
        mask = baseline_pack["mask"]
        bmask = baseline_pack["bmask"]
        heater = baseline_pack["heater"]
        xs, ys, zs = baseline_pack["grid"]
        bmin, bmax = baseline_pack["bounds"]

        solid_idx = np.argwhere(mask & (~np.isnan(Tvol)))
        bnd_idx = np.argwhere(bmask)
        hot_idx = np.argwhere(heater)

        def idx_to_xyz(idx):
            iy, ix, iz = idx[:,0], idx[:,1], idx[:,2]
            x = xs[ix]
            y = ys[iy]
            z = zs[iz]
            return np.stack([x,y,z], axis=1).astype(np.float32)

        # baseline supervised
        sel = solid_idx[rng.integers(0, solid_idx.shape[0], size=(cfg.n_data,))]
        x_data = idx_to_xyz(sel)
        y_data = Tvol[sel[:,0], sel[:,1], sel[:,2]].astype(np.float32).reshape(-1,1)

        # boundary (hot+cold)
        n_hot = max(1, cfg.n_bc // 4)
        n_cold = cfg.n_bc - n_hot
        sel_hot = hot_idx[rng.integers(0, hot_idx.shape[0], size=(n_hot,))] if hot_idx.shape[0] > 0 else bnd_idx[:n_hot]
        sel_cold = bnd_idx[rng.integers(0, bnd_idx.shape[0], size=(n_cold,))]

        x_bc = np.concatenate([idx_to_xyz(sel_hot), idx_to_xyz(sel_cold)], axis=0)
        y_bc = np.concatenate([
            np.ones((sel_hot.shape[0],1), dtype=np.float32),
            np.zeros((sel_cold.shape[0],1), dtype=np.float32),
        ], axis=0)

        # collocation via bbox + mask lookup
        def sample_collocation(n):
            pts_list = []
            while sum(p.shape[0] for p in pts_list) < n:
                m = max(20000, n)
                cand = rng.random((m,3)).astype(np.float32)
                cand = cand * (bmax - bmin) + bmin

                ix = np.clip(((cand[:,0]-bmin[0])/(bmax[0]-bmin[0]+1e-12) * (cfg.nx-1)).round().astype(int), 0, cfg.nx-1)
                iy = np.clip(((cand[:,1]-bmin[1])/(bmax[1]-bmin[1]+1e-12) * (cfg.ny-1)).round().astype(int), 0, cfg.ny-1)
                iz = np.clip(((cand[:,2]-bmin[2])/(bmax[2]-bmin[2]+1e-12) * (cfg.nz-1)).round().astype(int), 0, cfg.nz-1)

                keep = mask[iy, ix, iz]
                pts_list.append(cand[keep])
                if len(pts_list) > 30:
                    break
            pts = np.concatenate(pts_list, axis=0)[:n].astype(np.float32)
            return pts

        x_col = sample_collocation(cfg.n_col)

        self.x_col = torch.from_numpy(x_col)
        self.x_bc = torch.from_numpy(x_bc)
        self.y_bc = torch.from_numpy(y_bc)
        self.x_data = torch.from_numpy(x_data)
        self.y_data = torch.from_numpy(y_data)

        self.N = self.x_col.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        j_bc = i % self.x_bc.shape[0]
        j_dt = i % self.x_data.shape[0]
        return {
            "x": self.x_col[i],
            "x_col": self.x_col[i],
            "x_bc": self.x_bc[j_bc],
            "y_bc": self.y_bc[j_bc],
            "x_data": self.x_data[j_dt],
            "y_data": self.y_data[j_dt],
        }


def dict_collate(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


# =========================================================
# Symbolic physics builder (PINNFactory)
# =========================================================
def build_physics_loss_from_payload(physics_payload: dict):
    default_spec = PINNProblemSpec(
        pde_residuals=[
            "Derivative(T(x,y,z), x, 2) + Derivative(T(x,y,z), y, 2) + Derivative(T(x,y,z), z, 2)"
        ],
        conditions=[],
        independent_vars=["x", "y", "z"],
        dependent_vars=["T"],
        inverse_params=[],
        loss_weights={"pde": 1.0},
        verbose=False,
    )

    if not physics_payload:
        spec = default_spec
    elif "spec" in physics_payload and isinstance(physics_payload["spec"], dict):
        s = physics_payload["spec"]
        spec = PINNProblemSpec(
            pde_residuals=s.get("pde_residuals", default_spec.pde_residuals),
            conditions=s.get("conditions", []),
            independent_vars=s.get("independent_vars", default_spec.independent_vars),
            dependent_vars=s.get("dependent_vars", default_spec.dependent_vars),
            inverse_params=s.get("inverse_params", []),
            loss_weights=s.get("loss_weights", default_spec.loss_weights),
            verbose=bool(s.get("verbose", False)),
        )
    else:
        # if using a registry by id in the future, map it here. For now: default
        spec = default_spec

    pinn_factory = PINNFactory(spec)
    physics_loss_fn = pinn_factory.generate_loss_function()
    return physics_loss_fn, spec


# =========================================================
# Loss: physics (symbolic) + BC + data
# =========================================================
def make_loss_fn(cfg: Cfg, physics_loss_fn):
    mse = nn.MSELoss()
    combined = CombinedLoss(
        supervised=SupervisedLoss("mse"),
        physics=PhysicsLossHook(physics_loss_fn),
        w_supervised=1.0,
        w_physics=1.0,
    )

    def loss_fn(model, y_hat, batch):
        device = next(model.parameters()).device

        # --- PDE collocation for physics hook ---
        xcol = batch["x_col"].to(device).clone().detach().requires_grad_(True)
        collocation = (xcol[:, 0:1], xcol[:, 1:2], xcol[:, 2:3])

        # --- supervised (baseline) ---
        xdt = batch["x_data"].to(device)
        ydt = batch["y_data"].to(device)
        pred_dt = model(xdt)

        # CombinedLoss uses "x","y","collocation"
        b = {"x": xdt, "y": ydt, "collocation": collocation}
        out_comb = combined(model, pred_dt, b)

        l_pde = out_comb.get("physics", out_comb.get("pde", torch.tensor(0.0, device=device)))
        l_data = out_comb.get("supervised", out_comb.get("mse", torch.tensor(0.0, device=device)))

        # --- BC Dirichlet (separate) ---
        xbc = batch["x_bc"].to(device)
        ybc = batch["y_bc"].to(device)
        pred_bc = model(xbc)
        l_bc = mse(pred_bc, ybc)

        total = cfg.w_pde * l_pde + cfg.w_data * l_data + cfg.w_bc * l_bc
        return {
            "total": total,
            "pde": l_pde.detach(),
            "data": l_data.detach(),
            "bc": l_bc.detach(),
        }

    return loss_fn


# =========================================================
# Job paths
# =========================================================
def job_paths(job_id: str):
    stl_path = os.path.join(UPLOAD_DIR, f"{job_id}.stl")
    ckpt_path = os.path.join(CKPT_DIR, f"{job_id}_best.pt")
    meta_path = os.path.join(CKPT_DIR, f"{job_id}_meta.json")
    ply_path = os.path.join(OUT_DIR, f"{job_id}_pred.ply")
    return stl_path, ckpt_path, meta_path, ply_path


# =========================================================
# Background train
# =========================================================
def run_async_train(job_id, data, cfg_default):
    try:
        jobs_status[job_id] = {"status": "processing", "message": "Preparing..."}

        stl_path, ckpt_path, meta_path, _ = job_paths(job_id)

        # cfg override
        cfg = Cfg(**cfg_default.__dict__)

        # allow hidden and activation from front
        if "activation" in data:
            cfg.activation = str(data["activation"])
        if "hidden" in data:
            cfg.hidden = data["hidden"]

        for k in ["epochs", "lr", "nx", "ny", "nz", "n_col", "n_bc", "n_data", "batch_train", "batch_val", "w_pde", "w_bc", "w_data"]:
            if k in data:
                setattr(cfg, k, type(getattr(cfg, k))(data[k]))

        # 1) mesh
        jobs_status[job_id] = {"status": "processing", "message": "Loading + normalizing STL..."}
        mesh = trimesh.load_mesh(stl_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh = normalize_mesh(mesh)

        # 2) baseline
        jobs_status[job_id] = {"status": "processing", "message": "Baseline (sparse CG Laplace) ..."}
        prob = make_default_heat_problem(mesh)
        base = solve_laplace_dirichlet(mesh, cfg, prob)

        # 3) dataset
        jobs_status[job_id] = {"status": "processing", "message": "Sampling points..."}
        ds = Heat3DDataset(cfg, base, seed=7)
        train_loader = DataLoader(ds, batch_size=cfg.batch_train, shuffle=True, collate_fn=dict_collate)
        val_loader = DataLoader(ds, batch_size=cfg.batch_val, shuffle=False, collate_fn=dict_collate)

        # 4) model + physics + loss
        jobs_status[job_id] = {"status": "processing", "message": "Building model + symbolic physics..."}
        model = build_model(cfg).to(DEVICE)
        physics_payload = data.get("physics", None)
        physics_loss_fn, physics_spec = build_physics_loss_from_payload(physics_payload)
        loss_fn = make_loss_fn(cfg, physics_loss_fn)

        # 5) trainer
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

        jobs_status[job_id] = {"status": "processing", "message": f"Training for {cfg.epochs} epochs..."}
        out = trainer.fit(train_loader, val_loader, train_cfg)
        best_path = out.get("best_path")

        # 6) save best
        if best_path and os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=DEVICE)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
            torch.save(state, ckpt_path)

        meta = {
            "job_id": job_id,
            "cfg": {**cfg.__dict__, "hidden": parse_hidden(cfg.hidden)},
            "baseline": {
                "T_min": float(np.nanmin(base["T"])),
                "T_max": float(np.nanmax(base["T"])),
                "pitch": float(base["pitch"]),
            },
            "physics_spec": {
                "pde_residuals": physics_spec.pde_residuals,
                "conditions": physics_spec.conditions,
                "independent_vars": physics_spec.independent_vars,
                "dependent_vars": physics_spec.dependent_vars,
                "inverse_params": physics_spec.inverse_params,
                "loss_weights": physics_spec.loss_weights,
            },
            "best_val": float(out.get("best_val")) if out.get("best_val") is not None else None,
            "ckpt_path": ckpt_path if os.path.exists(ckpt_path) else None,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        jobs_status[job_id] = {"status": "completed", "result": meta}
        print(f"Job {job_id} completed.")

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        jobs_status[job_id] = {"status": "failed", "error": str(e), "traceback": tb}


# =========================================================
# API: model spec (for frontend preview)
# =========================================================
@app.post("/api/model_spec")
def model_spec():
    data = request.get_json(force=True) or {}
    try:
        activation = str(data.get("activation", "tanh"))
        hidden = parse_hidden(data.get("hidden", "128,128,128,128"))
        dims = [3, *hidden, 1]

        layers = []
        total_params = 0
        for i in range(len(dims) - 1):
            layers.append({"type": "Linear", "in": dims[i], "out": dims[i+1]})
            total_params += dims[i] * dims[i+1] + dims[i+1]
            if i < len(dims) - 2:
                layers.append({"type": "Activation", "name": activation})

        cfg = {k: data.get(k) for k in data.keys()}
        cfg["hidden"] = hidden
        cfg["activation"] = activation

        return jsonify({
            "ok": True,
            "cfg": cfg,
            "arch": {
                "in_dim": 3,
                "out_dim": 1,
                "hidden": hidden,
                "activation": activation,
                "layers": layers,
                "total_params": int(total_params),
            }
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# =========================================================
# API: physics problems registry (for frontend dropdown)
# =========================================================
@app.get("/api/physics_problems")
def physics_problems():
    problems = [
        {
            "id": "laplace_T_3d",
            "label": "Steady Heat (Laplace 3D)",
            "description": "Steady-state heat conduction without sources: ∇²T = 0",
            "spec": {
                "pde_residuals": [
                    "Derivative(T(x,y,z), x, 2) + Derivative(T(x,y,z), y, 2) + Derivative(T(x,y,z), z, 2)"
                ],
                "conditions": [],
                "independent_vars": ["x", "y", "z"],
                "dependent_vars": ["T"],
                "inverse_params": [],
                "loss_weights": {"pde": 1.0},
                "verbose": False,
            }
        }
    ]
    return jsonify({"problems": problems})


# =========================================================
# API: validate/preview physics spec (compile PINNFactory)
# =========================================================
@app.post("/api/physics_spec")
def physics_spec():
    data = request.get_json(force=True) or {}
    try:
        physics_loss_fn, spec = build_physics_loss_from_payload(data)
        _ = physics_loss_fn  # just to ensure creation

        return jsonify({
            "ok": True,
            "spec": {
                "pde_residuals": spec.pde_residuals,
                "conditions": spec.conditions,
                "independent_vars": spec.independent_vars,
                "dependent_vars": spec.dependent_vars,
                "inverse_params": spec.inverse_params,
                "loss_weights": spec.loss_weights,
            }
        })
    except Exception as e:
        tb = traceback.format_exc()
        return jsonify({"ok": False, "error": str(e), "traceback": tb}), 400


# =========================================================
# Routes: upload / train / status / infer / result
# =========================================================
@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]

    file_content = f.read()
    job_id = hashlib.md5(file_content).hexdigest()
    f.seek(0)

    stl_path, _, _, _ = job_paths(job_id)
    f.save(stl_path)

    jobs_status[job_id] = {"status": "uploaded"}
    return jsonify({"job_id": job_id})


@app.post("/api/train")
def train():
    data = request.get_json(force=True) or {}
    job_id = data.get("job_id")
    if not job_id:
        return jsonify({"error": "missing job_id"}), 400

    stl_path, ckpt_path, meta_path, _ = job_paths(job_id)

    # cache hit
    if os.path.exists(ckpt_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            result_meta = json.load(f)
        return jsonify({
            "job_id": job_id,
            "status": "completed",
            "message": "Model found in cache (Hash match).",
            "result": result_meta
        }), 200

    if not os.path.exists(stl_path):
        return jsonify({"error": "unknown job_id (upload first)"}), 404

    thread = threading.Thread(target=run_async_train, args=(job_id, data, cfg_default), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id, "status": "started"}), 202


@app.get("/api/status/<job_id>")
def get_status(job_id):
    stl_path, _, meta_path, _ = job_paths(job_id)

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return jsonify({"status": "completed", "result": json.load(f)})

    if job_id in jobs_status:
        return jsonify(jobs_status[job_id])

    if os.path.exists(stl_path):
        return jsonify({"status": "idle", "message": "Server restarted. Click Train again."})

    return jsonify({"status": "unknown", "error": "Job ID not found. Upload again."}), 404


@app.post("/api/infer")
def infer():
    data = request.get_json(force=True) or {}
    job_id = data.get("job_id")
    if not job_id:
        return jsonify({"error": "missing job_id"}), 400

    stl_path, ckpt_path, _, ply_path = job_paths(job_id)
    if not os.path.exists(stl_path):
        return jsonify({"error": "unknown job_id"}), 404

    # mesh
    mesh = trimesh.load_mesh(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = normalize_mesh(mesh)

    # model
    cfg = cfg_default
    model = build_model(cfg).to(DEVICE).eval()

    use_trained = bool(data.get("use_trained", True))
    if use_trained and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)

    V = mesh.vertices.astype(np.float32)
    x = torch.from_numpy(V).to(DEVICE)
    with torch.no_grad():
        y = model(x).detach().cpu().numpy().reshape(-1)

    rgb, (vmin, vmax) = scalar_to_rgb(y, cmap_name="viridis")
    export_ply_with_vertex_colors(mesh, rgb, ply_path)

    return jsonify({
        "job_id": job_id,
        "trained_used": bool(use_trained and os.path.exists(ckpt_path)),
        "ply_url": f"/api/result/{job_id}",
        "range": {"min": vmin, "max": vmax}
    })


@app.get("/api/result/<job_id>")
def result(job_id):
    _, _, _, ply_path = job_paths(job_id)
    if not os.path.exists(ply_path):
        return jsonify({"error": "not found"}), 404
    return send_file(ply_path, mimetype="application/octet-stream", as_attachment=False)

@app.get("/api/tools/models")
def tools_models():
    try:
        from pinneaple_models.register_all import register_all
        from pinneaple_models.registry import ModelRegistry
        register_all()
        names = ModelRegistry.list()
        pinns = ModelRegistry.list(family="pinns")
        return jsonify({"ok": True, "total": len(names), "models": names, "pinns_total": len(pinns), "pinns": pinns})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/api/tools/mesh_ops")
def tools_mesh_ops():
    data = request.get_json(force=True) or {}
    job_id = data.get("job_id")
    target_faces = int(data.get("target_faces", 5000))
    backend = str(data.get("backend", "trimesh"))

    if not job_id:
        return jsonify({"ok": False, "error": "missing job_id"}), 400

    stl_path, _, _, _ = job_paths(job_id)
    if not os.path.exists(stl_path):
        return jsonify({"ok": False, "error": "unknown job_id (upload first)"}), 404

    try:
        from pinneaple_geom.io.trimesh_bridge import TrimeshBridge
        from pinneaple_geom.ops.repair import repair_mesh
        from pinneaple_geom.ops.simplify import simplify_mesh

        tm = trimesh.load_mesh(stl_path, force="mesh")
        if not isinstance(tm, trimesh.Trimesh):
            tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
        tm = normalize_mesh(tm)

        bridge = TrimeshBridge()
        g = bridge.from_trimesh(tm)

        g2 = repair_mesh(g)
        g3 = simplify_mesh(g2, target_faces=target_faces, backend=backend)

        out_stl = os.path.join(TOOLS_DIR, f"{job_id}_repaired_simplified_{target_faces}.stl")

        # voltar pra trimesh e exportar STL (ponte inversa pode variar; aqui usamos o trimesh original como fallback)
        # se seu bridge tiver to_trimesh(g3), use isso:
        if hasattr(bridge, "to_trimesh"):
            tm2 = bridge.to_trimesh(g3)
        else:
            tm2 = tm  # fallback

        tm2.export(out_stl)

        return jsonify({
            "ok": True,
            "job_id": job_id,
            "target_faces": target_faces,
            "n_faces": int(getattr(g3, "n_faces", -1)),
            "n_vertices": int(getattr(g3, "n_vertices", -1)),
            "download_url": f"/api/tools/download?path={os.path.basename(out_stl)}",
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/api/tools/nasa_power_to_zarr")
def tools_nasa_power_to_zarr():
    data = request.get_json(force=True) or {}
    try:
        latitude = float(data.get("latitude", -8.05))
        longitude = float(data.get("longitude", -34.9))
        start = str(data.get("start", "20240101"))
        end = str(data.get("end", "20240131"))
        parameters = data.get("parameters", ["T2M", "PRECTOTCORR", "ALLSKY_SFC_SW_DWN"])
        if not isinstance(parameters, list) or not parameters:
            return jsonify({"ok": False, "error": "parameters must be a non-empty list"}), 400

        import requests
        from pinneaple_data.physical_sample import PhysicalSample
        from pinneaple_data.zarr_store import UPDZarrStore

        base = "https://power.larc.nasa.gov/api/temporal/daily/point"
        params = {
            "parameters": ",".join(parameters),
            "community": "SB",
            "longitude": str(longitude),
            "latitude": str(latitude),
            "start": start,
            "end": end,
            "format": "JSON",
            "time-standard": "UTC",
        }
        r = requests.get(base, params=params, timeout=60)
        r.raise_for_status()
        power = r.json()

        props = power.get("properties", {})
        param = props.get("parameter", {})
        if not param:
            return jsonify({"ok": False, "error": "POWER response missing properties.parameter"}), 502

        first_k = sorted(param.keys())[0]
        dates = sorted(param[first_k].keys())
        keys = sorted(param.keys())
        T = len(dates)
        D = len(keys)

        x = torch.empty((T, D), dtype=torch.float32)
        for j, k in enumerate(keys):
            series = param[k]
            for i, d in enumerate(dates):
                v = series.get(d, float("nan"))
                x[i, j] = float(v) if v is not None else float("nan")

        t = torch.arange(T, dtype=torch.float32).unsqueeze(1)

        header = power.get("header", {})
        meta = {
            "source": "nasa_power",
            "title": header.get("title"),
            "api": header.get("api"),
            "start": props.get("start"),
            "end": props.get("end"),
            "parameters": keys,
        }
        units = {k: "POWER_unit_unknown" for k in keys}

        samples = [PhysicalSample(
            state={"x": x, "t": t},
            domain={"type": "point", "crs": "WGS84"},
            provenance=meta,
            schema={"units": units, "columns": keys, "time": "daily"},
        )]

        zarr_name = data.get("name", f"nasa_power_{latitude}_{longitude}_{start}_{end}".replace(".", "_"))
        zarr_path = os.path.join(TOOLS_DIR, f"{zarr_name}.zarr")
        UPDZarrStore.write(zarr_path, samples, manifest={
            "name": zarr_name,
            "source": "NASA POWER Daily API",
            "latitude": latitude,
            "longitude": longitude,
            "start": start,
            "end": end,
            "parameters": keys,
        })

        # salva raw json pra debug
        raw_path = os.path.join(TOOLS_DIR, f"{zarr_name}_raw.json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(power, f, indent=2)

        return jsonify({
            "ok": True,
            "zarr": os.path.basename(zarr_path),
            "raw_json": os.path.basename(raw_path),
            "download_zarr_url": f"/api/tools/download?path={os.path.basename(zarr_path)}",
            "download_raw_url": f"/api/tools/download?path={os.path.basename(raw_path)}",
            "shape": {"T": T, "D": D},
            "columns": keys,
            "dates_preview": dates[:10] + (["..."] if len(dates) > 10 else []),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.get("/api/tools/download")
def tools_download():
    name = request.args.get("path", "")
    if not name or ("/" in name) or ("\\" in name) or (".." in name):
        return jsonify({"ok": False, "error": "invalid path"}), 400
    full = os.path.join(TOOLS_DIR, name)
    if not os.path.exists(full):
        return jsonify({"ok": False, "error": "not found"}), 404
    return send_file(full, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)
