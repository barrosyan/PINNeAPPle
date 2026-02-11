import os
import uuid
import json
import numpy as np
from dataclasses import dataclass

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

import threading
import hashlib

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "runs", "uploads")
OUT_DIR = os.path.join(BASE_DIR, "runs", "outputs")
CKPT_DIR = os.path.join(BASE_DIR, "runs", "checkpoints")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cpu"

# Dicionário global para rastrear o status de cada job em memória
jobs_status = {}

# =========================================================
# Config (treino/baseline)
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
# Model wrapper (Tensor-only forward)
# =========================================================
class Wrap(nn.Module):
    def __init__(self, pinn):
        super().__init__()
        self.pinn = pinn
    def forward(self, x):
        out = self.pinn(x)
        return out.y if hasattr(out, "y") else out

def build_model(cfg: Cfg):
    core = VanillaPINN(in_dim=3, out_dim=1, hidden=cfg.hidden, activation=cfg.activation)
    model = Wrap(core).to(DEVICE)
    return model

# =========================================================
# Helpers: mesh + export colored PLY
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
    rgba = np.concatenate([colors_rgb, 255*np.ones((len(colors_rgb),1), dtype=np.uint8)], axis=1)
    mesh.visual.vertex_colors = rgba
    mesh.export(out_path)
    return out_path

# =========================================================
# Physics problem (Dirichlet heater patch vs cold boundary)
# =========================================================
@dataclass
class HeatProblem:
    heater_bbox_min: np.ndarray  # (3,)
    heater_bbox_max: np.ndarray  # (3,)
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
# Baseline solver: voxel mask via trimesh.voxelized().fill().is_filled()
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
# Loss: PDE + BC + supervised
# =========================================================
mse = nn.MSELoss()

def grad_safe(u, x):
    return torch.autograd.grad(
        outputs=u, inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True,
        allow_unused=False,
    )[0]

def laplacian_safe(u, x):
    g = grad_safe(u, x)
    uxx = torch.autograd.grad(g[:,0:1], x, torch.ones_like(g[:,0:1]), create_graph=True, retain_graph=True)[0][:,0:1]
    uyy = torch.autograd.grad(g[:,1:2], x, torch.ones_like(g[:,1:2]), create_graph=True, retain_graph=True)[0][:,1:2]
    uzz = torch.autograd.grad(g[:,2:3], x, torch.ones_like(g[:,2:3]), create_graph=True, retain_graph=True)[0][:,2:3]
    return uxx + uyy + uzz

def make_loss_fn(cfg: Cfg):
    def loss_fn(model, y_hat, batch):
        device = next(model.parameters()).device

        # PDE: ∇²T=0
        xcol = batch["x_col"].to(device).clone().detach().requires_grad_(True)
        with torch.enable_grad():
            Tcol = model(xcol)
            lap = laplacian_safe(Tcol, xcol)
            l_pde = torch.mean(lap**2)

        # BC Dirichlet
        xbc = batch["x_bc"].to(device)
        ybc = batch["y_bc"].to(device)
        Tbc = model(xbc)
        l_bc = mse(Tbc, ybc)

        # supervised from baseline
        xdt = batch["x_data"].to(device)
        ydt = batch["y_data"].to(device)
        Tdt = model(xdt)
        l_data = mse(Tdt, ydt)

        total = cfg.w_pde*l_pde + cfg.w_bc*l_bc + cfg.w_data*l_data
        return {"total": total, "pde": l_pde.detach(), "bc": l_bc.detach(), "data": l_data.detach()}
    return loss_fn

# =========================================================
# JOB utils
# =========================================================
def job_paths(job_id: str):
    stl_path = os.path.join(UPLOAD_DIR, f"{job_id}.stl")
    ckpt_path = os.path.join(CKPT_DIR, f"{job_id}_best.pt")
    meta_path = os.path.join(CKPT_DIR, f"{job_id}_meta.json")
    ply_path = os.path.join(OUT_DIR, f"{job_id}_pred.ply")
    return stl_path, ckpt_path, meta_path, ply_path

# =========================================================
# Função de Treino Assíncrona
# =========================================================
def run_async_train(job_id, data, cfg_default):
    """
    Executa o pipeline completo de física e deep learning em segundo plano.
    """
    try:
        # 1. Preparação Inicial
        jobs_status[job_id] = {"status": "processing", "message": "Preparando arquivos e configuração..."}
        
        stl_path, ckpt_path, meta_path, _ = job_paths(job_id)
        
        # Override das configurações enviadas pelo front-end
        cfg = Cfg(**cfg_default.__dict__)
        for k in ["epochs", "lr", "nx", "ny", "nz", "n_col", "n_bc", "n_data"]:
            if k in data:
                setattr(cfg, k, type(getattr(cfg, k))(data[k]))

        # 2. Processamento da Malha (Mesh)
        jobs_status[job_id] = {"status": "processing", "message": "Carregando e normalizando STL..."}
        mesh = trimesh.load_mesh(stl_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh = normalize_mesh(mesh)

        # 3. Solver de Baseline (Física Clássica)
        # Esta parte costuma ser pesada dependendo da resolução (nx, ny, nz)
        jobs_status[job_id] = {"status": "processing", "message": "Resolvendo Baseline (Matriz Esparsa CG)..."}
        prob = make_default_heat_problem(mesh)
        base = solve_laplace_dirichlet(mesh, cfg, prob)

        # 4. Preparação do Dataset para a PINN
        jobs_status[job_id] = {"status": "processing", "message": "Gerando pontos de colocação (collocation points)..."}
        ds = Heat3DDataset(cfg, base, seed=7)
        train_loader = DataLoader(ds, batch_size=cfg.batch_train, shuffle=True, collate_fn=dict_collate)
        val_loader = DataLoader(ds, batch_size=cfg.batch_val, shuffle=False, collate_fn=dict_collate)

        # 5. Configuração do Modelo e Trainer
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

        # 6. Treinamento da Rede Neural (PINN)
        # Aqui o status ficará em 'processing' por mais tempo
        jobs_status[job_id] = {"status": "processing", "message": f"Treinando PINN por {cfg.epochs} épocas..."}
        
        out = trainer.fit(train_loader, val_loader, train_cfg)
        best_path = out.get("best_path")

        # 7. Salvamento e Finalização
        if best_path and os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=DEVICE)
            state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
            torch.save(state, ckpt_path)

        meta = {
            "job_id": job_id,
            "cfg": cfg.__dict__,
            "baseline": {
                "T_min": float(np.nanmin(base["T"])),
                "T_max": float(np.nanmax(base["T"])),
                "pitch": float(base["pitch"]),
            },
            "best_val": float(out.get("best_val")) if out.get("best_val") is not None else None,
            "ckpt_path": ckpt_path if os.path.exists(ckpt_path) else None,
        }
        
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Sucesso: o front-end verá o status 'completed' e habilitará a inferência
        jobs_status[job_id] = {"status": "completed", "result": meta}
        print(f" Job {job_id} finalizado com sucesso.")

    except Exception as e:
        # Captura qualquer erro (falha de memória, erro de física, etc) e avisa o front
        error_msg = str(e)
        print(f" Erro crítico no treino do Job {job_id}: {error_msg}")
        jobs_status[job_id] = {"status": "failed", "error": error_msg}

# =========================================================
# Routes
# =========================================================

@app.post("/api/upload")
def upload():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["file"]
    
    # 1. Gerar ID baseado no CONTEÚDO do arquivo (Hash)
    file_content = f.read()
    job_id = hashlib.md5(file_content).hexdigest()
    f.seek(0) # Resetar o ponteiro para salvar o arquivo
    
    stl_path, _, _, _ = job_paths(job_id)
    f.save(stl_path)
    
    jobs_status[job_id] = {"status": "uploaded"}
    return jsonify({"job_id": job_id})

@app.post("/api/train")
def train():
    data = request.get_json(force=True)
    job_id = data.get("job_id")
    
    stl_path, ckpt_path, meta_path, _ = job_paths(job_id)
    
    # VERIFICAÇÃO CRÍTICA: Se o arquivo .pt existe com esse Hash
    if os.path.exists(ckpt_path):
        # Se o meta.json não existir, criamos um básico para não dar erro
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                result_meta = json.load(f)
        else:
            result_meta = {"job_id": job_id, "best_val": 0.000000}

        return jsonify({
            "job_id": job_id, 
            "status": "completed", 
            "message": "Model found in cache (Hash match).",
            "result": result_meta
        }), 200

    # Se não existe, inicia o treino normal
    thread = threading.Thread(target=run_async_train, args=(job_id, data, cfg_default))
    thread.start()
    return jsonify({"job_id": job_id, "status": "started"}), 202

@app.get("/api/status/<job_id>")
def get_status(job_id):
    """Verifica status na memória E no disco para evitar erro 404 após reiniciar."""
    stl_path, _, meta_path, _ = job_paths(job_id)
    
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return jsonify({"status": "completed", "result": json.load(f)})
    
    if job_id in jobs_status:
        return jsonify(jobs_status[job_id])
    
    if os.path.exists(stl_path):
        return jsonify({"status": "idle", "message": "Servidor reiniciado. Por favor, clique em Train novamente."})

    return jsonify({"status": "unknown", "error": "Job ID nao encontrado. Faca o upload novamente."}), 404

@app.post("/api/infer")
def infer():
    data = request.get_json(force=True)
    job_id = data.get("job_id")
    if not job_id:
        return jsonify({"error": "missing job_id"}), 400

    stl_path, ckpt_path, _, ply_path = job_paths(job_id)
    if not os.path.exists(stl_path):
        return jsonify({"error": "unknown job_id"}), 404

    # load mesh
    mesh = trimesh.load_mesh(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh = normalize_mesh(mesh)

    # build model + optionally load trained
    cfg = cfg_default
    model = build_model(cfg).to(DEVICE).eval()

    use_trained = bool(data.get("use_trained", True))
    if use_trained and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state, strict=False)

    # inference on vertices
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)