import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import trimesh
from torch.utils.data import Dataset, DataLoader

from pinneaple_geom.io.trimesh_bridge import TrimeshBridge
from pinneaple_geom.sample.points import sample_surface_points
from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.metrics import default_metrics
from pinneaple_models.pinns.vanilla import VanillaPINN

TRAIN = True  # True: train | False: load + report

# =========================================================
# Config
# =========================================================
@dataclass
class Cfg:
    k: float = 200.0
    h: float = 35.0
    T_base: float = 0.0
    T_inf: float = 1.0
    q_over_k: float = 6.0

    Lx: float = 0.080
    Ly: float = 0.060
    base_th: float = 0.006
    fin_h: float = 0.019
    fin_t: float = 0.002
    fin_pitch: float = 0.006
    n_fins: int = 9

    chip_lx: float = 0.020
    chip_ly: float = 0.015
    chip_z0: float = 0.002
    chip_z1: float = 0.006

    n_col: int = 40000
    n_surf: int = 24000
    n_base: int = 12000

    seed: int = 123
    deterministic: bool = False
    device: str = "cpu"
    epochs: int = 300
    lr: float = 1e-3
    batch_train: int = 1024
    batch_val: int = 2048

    w_pde: float = 1.0
    w_robin: float = 3.0
    w_base: float = 15.0

    out_dir: str = "examples/_runs_godela_heatsink"
    run_name: str = "heatsink_3d_steady_conduction_robin"
    best_name: str = "heatsink_3d_steady_conduction_robin.best.pt"
    cad_dir: str = "examples/_runs_godela_heatsink/cad"
    stl_path: str = "examples/_runs_godela_heatsink/cad/heatsink.stl"
    step_path: str = "examples/_runs_godela_heatsink/cad/heatsink.step"


# =========================================================
# Utils
# =========================================================
def as_y(out):
    return out.y if hasattr(out, "y") else out

def set_seed(seed: int, deterministic: bool = False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def dict_collate(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out

mse = nn.MSELoss()


# =========================================================
# CAD: STEP + STL
# =========================================================
def build_heatsink_step_and_stl(cfg: Cfg):
    os.makedirs(cfg.cad_dir, exist_ok=True)
    try:
        import cadquery as cq
    except Exception as e:
        raise RuntimeError(
            "CadQuery não está instalado. Rode:\n"
            "  pip install cadquery\n"
            f"Erro: {e}"
        )

    base = cq.Workplane("XY").box(cfg.Lx, cfg.Ly, cfg.base_th)

    top_z = cfg.base_th / 2.0
    fin_z_center = top_z + cfg.fin_h / 2.0

    span = cfg.Lx - cfg.fin_pitch
    x0 = -span / 2.0
    xs = [x0 + i * cfg.fin_pitch for i in range(cfg.n_fins)]

    fins = cq.Workplane("XY")
    for x in xs:
        fin = (
            cq.Workplane("XY")
            .center(x, 0.0)
            .box(cfg.fin_t, cfg.Ly * 0.95, cfg.fin_h)
            .translate((0.0, 0.0, fin_z_center))
        )
        fins = fins.union(fin)

    heatsink = base.union(fins)
    cq.exporters.export(heatsink, cfg.step_path)
    cq.exporters.export(heatsink, cfg.stl_path, tolerance=1e-4, angularTolerance=0.1)

    print("[cad] STEP:", cfg.step_path)
    print("[cad] STL :", cfg.stl_path)


# =========================================================
# Mesh + geom
# =========================================================
def load_mesh_and_geom(cfg: Cfg):
    mesh = trimesh.load_mesh(cfg.stl_path, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)

    print("[mesh] watertight:", mesh.is_watertight, "| euler:", mesh.euler_number)
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
            mesh.remove_degenerate_faces()
            mesh.remove_unreferenced_vertices()
            mesh.process(validate=True)
            print("[mesh] watertight after repair:", mesh.is_watertight)
        except Exception as e:
            print("[mesh] repair attempt failed:", repr(e))

    bridge = TrimeshBridge()
    geom = bridge.from_trimesh(mesh)
    return mesh, geom


# =========================================================
# Robust volume sampling
# =========================================================
def sample_points_in_mesh_volume(mesh: trimesh.Trimesh, n: int, seed: int, max_tries_mult: int = 400):
    rng = np.random.default_rng(seed)

    # try trimesh.sample.volume_mesh if present
    try:
        import trimesh.sample as ts
        if hasattr(ts, "volume_mesh"):
            pts = ts.volume_mesh(mesh, count=n)
            pts = np.asarray(pts, dtype=np.float32)
            if pts.shape == (n, 3):
                return pts
    except Exception:
        pass

    # fallback: rejection sampling with mesh.contains
    lo, hi = mesh.bounds[0], mesh.bounds[1]
    out = []
    tries = 0
    max_tries = n * max_tries_mult
    batch = max(4096, n // 2)

    while sum(p.shape[0] for p in out) < n and tries < max_tries:
        need = n - sum(p.shape[0] for p in out)
        cand = rng.uniform(lo, hi, size=(max(need * 2, batch), 3)).astype(np.float32)
        inside = mesh.contains(cand)
        picked = cand[inside]
        if picked.shape[0] > 0:
            out.append(picked[:need])
        tries += cand.shape[0]

    pts = np.concatenate(out, axis=0) if out else np.zeros((0, 3), dtype=np.float32)
    if pts.shape[0] < n:
        raise RuntimeError(
            f"Falhou em amostrar {n} pontos no volume; consegui {pts.shape[0]}. "
            "Provável causa: malha não-watertight."
        )
    return pts[:n].astype(np.float32)


def sample_points_on_base_face(mesh: trimesh.Trimesh, n: int, seed: int):
    rng = np.random.default_rng(seed)
    zmin = float(mesh.bounds[0, 2])
    lo, hi = mesh.bounds[0], mesh.bounds[1]

    xs = rng.uniform(lo[0], hi[0], size=n * 3)
    ys = rng.uniform(lo[1], hi[1], size=n * 3)
    pts = np.stack([xs, ys, np.full_like(xs, zmin)], axis=1).astype(np.float32)

    eps = 1e-5
    inside = mesh.contains(pts + np.array([0.0, 0.0, eps], dtype=np.float32))
    pts = pts[inside][:n]

    if pts.shape[0] < n:
        more = sample_points_on_base_face(mesh, n - pts.shape[0], seed + 17)
        pts = np.concatenate([pts, more], axis=0)
    return pts.astype(np.float32)


def chip_source_mask(xyz: np.ndarray, cfg: Cfg):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    cx0, cx1 = -cfg.chip_lx / 2, cfg.chip_lx / 2
    cy0, cy1 = -cfg.chip_ly / 2, cfg.chip_ly / 2
    z0 = (-cfg.base_th / 2 + cfg.chip_z0)
    z1 = (-cfg.base_th / 2 + cfg.chip_z1)
    return (x >= cx0) & (x <= cx1) & (y >= cy0) & (y <= cy1) & (z >= z0) & (z <= z1)


# =========================================================
# Dataset
# =========================================================
class HeatSinkDataset(Dataset):
    def __init__(self, cfg: Cfg, mesh: trimesh.Trimesh, geom, seed: int = 0):
        super().__init__()

        col = sample_points_in_mesh_volume(mesh, cfg.n_col, seed=seed).astype(np.float32)

        pts_s, nrm_s, _ = sample_surface_points(geom, n=cfg.n_surf)
        surf = np.asarray(pts_s, dtype=np.float32)
        nrm = np.asarray(nrm_s, dtype=np.float32)

        base = sample_points_on_base_face(mesh, cfg.n_base, seed=seed + 11).astype(np.float32)

        T_base = np.full((base.shape[0], 1), cfg.T_base, dtype=np.float32)
        src = chip_source_mask(col, cfg).astype(np.float32)[:, None]
        q_over_k = (cfg.q_over_k * src).astype(np.float32)

        self.x_col = torch.tensor(col)
        self.qok = torch.tensor(q_over_k)

        self.x_surf = torch.tensor(surf)
        self.n_surf = torch.tensor(nrm)

        self.x_base = torch.tensor(base)
        self.T_base = torch.tensor(T_base)

        self.N = self.x_col.size(0)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return {
            "x": self.x_col[i],
            "x_col": self.x_col[i],
            "qok": self.qok[i],
            "x_surf": self.x_surf[i % self.x_surf.size(0)],
            "n_surf": self.n_surf[i % self.n_surf.size(0)],
            "x_base": self.x_base[i % self.x_base.size(0)],
            "T_base": self.T_base[i % self.T_base.size(0)],
        }


# =========================================================
# Model
# =========================================================
class TrainerFriendlyVanilla(nn.Module):
    def __init__(self, pinn: VanillaPINN):
        super().__init__()
        self.pinn = pinn

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        return as_y(self.pinn(x))


# =========================================================
# Loss (FIXED for val no_grad)
# =========================================================
class XYZAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, y, z):
        inp = torch.cat([x, y, z], dim=1)
        return as_y(self.model(inp))


def make_loss_fn(cfg: Cfg, adapter: nn.Module):
    Bi = float(cfg.h / cfg.k)

    def loss_fn(model, y_hat, batch):
        """
        IMPORTANT:
        - Trainer usually evaluates validation with torch.no_grad() / grad disabled.
        - PDE/Robin need autograd => will crash if grad is disabled.
        - So: if grad disabled, compute ONLY base BC loss (cheap & stable).
        """
        device = y_hat.device if torch.is_tensor(y_hat) else cfg.device

        # Base Dirichlet always works (no derivatives)
        xb = batch["x_base"].to(device)
        Tb = batch["T_base"].to(device)
        Tb_hat = as_y(model(xb))
        l_base = mse(Tb_hat, Tb)

        # If grad is disabled (common in validation), skip PDE/Robin
        if not torch.is_grad_enabled():
            total = cfg.w_base * l_base
            return {"total": total, "pde": torch.tensor(0.0), "robin": torch.tensor(0.0), "base": l_base.detach()}

        # -------------------------
        # PDE: Laplace(T) + q/k = 0
        # -------------------------
        xcol = batch["x_col"].to(device)
        qok = batch["qok"].to(device)

        x = xcol[:, 0:1].detach().clone().requires_grad_(True)
        y = xcol[:, 1:2].detach().clone().requires_grad_(True)
        z = xcol[:, 2:3].detach().clone().requires_grad_(True)

        Tcol = adapter(x, y, z)

        # If something breaks the graph, fail fast with a clear message
        if not Tcol.requires_grad:
            raise RuntimeError(
                "Tcol não requer grad. Algo desligou o grafo (no_grad/detach) durante o treino.\n"
                "Cheque: TRAIN loop não está em no_grad e o modelo não faz detach internamente."
            )

        dTx = torch.autograd.grad(Tcol, x, grad_outputs=torch.ones_like(Tcol), create_graph=True)[0]
        dTy = torch.autograd.grad(Tcol, y, grad_outputs=torch.ones_like(Tcol), create_graph=True)[0]
        dTz = torch.autograd.grad(Tcol, z, grad_outputs=torch.ones_like(Tcol), create_graph=True)[0]
        d2Tx = torch.autograd.grad(dTx, x, grad_outputs=torch.ones_like(dTx), create_graph=True)[0]
        d2Ty = torch.autograd.grad(dTy, y, grad_outputs=torch.ones_like(dTy), create_graph=True)[0]
        d2Tz = torch.autograd.grad(dTz, z, grad_outputs=torch.ones_like(dTz), create_graph=True)[0]
        lap = d2Tx + d2Ty + d2Tz

        res_pde = lap + qok
        l_pde = torch.mean(res_pde ** 2)

        # -------------------------
        # Robin: dT/dn + Bi*(T - T_inf)=0
        # -------------------------
        xs = batch["x_surf"].to(device)
        ns = batch["n_surf"].to(device)
        xs_req = xs.detach().clone().requires_grad_(True)

        Ts = as_y(model(xs_req))
        grads = torch.autograd.grad(Ts, xs_req, grad_outputs=torch.ones_like(Ts), create_graph=True)[0]
        dTdn = torch.sum(grads * ns, dim=1, keepdim=True)

        res_robin = dTdn + Bi * (Ts - cfg.T_inf)
        l_robin = torch.mean(res_robin ** 2)

        total = cfg.w_pde * l_pde + cfg.w_robin * l_robin + cfg.w_base * l_base
        return {"total": total, "pde": l_pde.detach(), "robin": l_robin.detach(), "base": l_base.detach()}

    return loss_fn


# =========================================================
# Checkpoint loader
# =========================================================
def load_checkpoint_into_model(model: nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        raise ValueError("Unexpected checkpoint format.")

    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[ckpt] loaded: {ckpt_path}")
    print(f"[ckpt] missing keys: {len(missing)} | unexpected: {len(unexpected)}")


# =========================================================
# Reporting
# =========================================================
def eval_slice(model, cfg: Cfg, mesh: trimesh.Trimesh, z0: float, grid_n: int = 180):
    lo, hi = mesh.bounds[0], mesh.bounds[1]
    xs = np.linspace(lo[0], hi[0], grid_n).astype(np.float32)
    ys = np.linspace(lo[1], hi[1], grid_n).astype(np.float32)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = np.full_like(XX, z0, dtype=np.float32)
    pts = np.stack([XX.reshape(-1), YY.reshape(-1), ZZ.reshape(-1)], axis=1).astype(np.float32)

    inside = mesh.contains(pts)
    pts_in = pts[inside]
    if pts_in.shape[0] == 0:
        return None

    with torch.no_grad():
        T_pred = model(torch.tensor(pts_in, device=cfg.device)).detach().cpu().numpy().reshape(-1)

    img = np.full((grid_n, grid_n), np.nan, dtype=np.float32)
    img.reshape(-1)[inside] = T_pred.astype(np.float32)
    return img


def plot_report(model, cfg: Cfg, mesh: trimesh.Trimesh, out_png: str):
    zmin, zmax = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
    zs = [zmin + 0.15*(zmax-zmin), zmin + 0.55*(zmax-zmin), zmin + 0.90*(zmax-zmin)]

    imgs = [eval_slice(model, cfg, mesh, z0) for z0 in zs]
    vals = []
    for im in imgs:
        if im is not None:
            v = im[np.isfinite(im)]
            if v.size:
                vals.append(v)
    if not vals:
        print("[report] no valid slice values (mesh.contains may be failing).")
        return

    valid_vals = np.concatenate(vals)
    vmin, vmax = float(np.min(valid_vals)), float(np.max(valid_vals))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, z0, im in zip(axes, zs, imgs):
        if im is None:
            ax.set_title(f"z={z0:.4f} (empty)")
            ax.axis("off")
            continue
        h = ax.imshow(im, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"T_pred | z={z0:.4f}")
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

    fig.savefig(out_png, dpi=180)
    plt.show()
    print("[report] saved:", out_png)


# =========================================================
# Main
# =========================================================
def main():
    cfg = Cfg()
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.cad_dir, exist_ok=True)
    set_seed(cfg.seed, cfg.deterministic)

    build_heatsink_step_and_stl(cfg)
    mesh, geom = load_mesh_and_geom(cfg)

    ds = HeatSinkDataset(cfg, mesh, geom, seed=42)
    train_loader = DataLoader(ds, batch_size=cfg.batch_train, shuffle=True, collate_fn=dict_collate)
    val_loader = DataLoader(ds, batch_size=cfg.batch_val, shuffle=False, collate_fn=dict_collate)

    pinn_core = VanillaPINN(in_dim=3, out_dim=1, hidden=(128, 128, 128, 128), activation="tanh")
    model = TrainerFriendlyVanilla(pinn_core).to(cfg.device)
    adapter = XYZAdapter(model)

    loss_fn = make_loss_fn(cfg, adapter)
    ckpt_path = os.path.join(cfg.out_dir, cfg.best_name)

    if TRAIN:
        trainer = Trainer(model=model, loss_fn=loss_fn, metrics=default_metrics())
        train_cfg = TrainConfig(
            epochs=cfg.epochs,
            lr=cfg.lr,
            device=cfg.device,
            log_dir=cfg.out_dir,
            run_name=cfg.run_name,
            seed=cfg.seed,
            deterministic=cfg.deterministic,
            amp=False,
            save_best=True,
        )
        out = trainer.fit(train_loader, val_loader, train_cfg)
        print("[train] best_val:", out.get("best_val"))
        print("[train] best_path:", out.get("best_path"))

    if os.path.exists(ckpt_path):
        load_checkpoint_into_model(model, ckpt_path, device=cfg.device)
    else:
        print(f"[warn] ckpt not found at {ckpt_path} — using current weights.")

    model.eval()
    out_png = os.path.join(cfg.out_dir, f"{cfg.run_name}_slices.png")
    plot_report(model, cfg, mesh, out_png)

    print("\n=== Artifacts ===")
    print("STEP:", cfg.step_path)
    print("STL :", cfg.stl_path)
    print("RUN :", cfg.out_dir)

if __name__ == "__main__":
    main()
