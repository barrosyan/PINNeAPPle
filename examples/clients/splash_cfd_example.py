import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------
# STEP generation (CAD)
# ----------------------------
def generate_step_cold_plate(step_path: str, Lx=0.20, Ly=0.12, Lz=0.012, r=0.004):
    """
    Cold plate: box with a through cylindrical channel along x-direction.
    Units: meters.
    """
    try:
        import cadquery as cq
    except Exception as e:
        raise ImportError(
            "CadQuery não encontrado. Instale com: pip install cadquery\n"
            f"Erro original: {e}"
        )

    wp = cq.Workplane("XY")
    block = wp.box(Lx, Ly, Lz, centered=True)
    channel = cq.Workplane("YZ").circle(r).extrude(Lx, both=True)
    solid = block.cut(channel)

    os.makedirs(os.path.dirname(step_path), exist_ok=True)
    cq.exporters.export(solid, step_path)
    return step_path


# ----------------------------
# PINNeAPPle imports
# ----------------------------
from pinneaple_pinn.factory.pinn_factory import PINNFactory, PINNProblemSpec
from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.metrics import default_metrics
from pinneaple_models.pinns.vanilla import VanillaPINN

# =========================================================
# Config
# =========================================================
@dataclass
class Cfg:
    # geometry
    Lx: float = 0.20
    Ly: float = 0.12
    Lz: float = 0.012
    r: float = 0.004

    # physics
    k: float = 205.0          # W/mK
    T_amb: float = 298.15     # K
    T_cool: float = 293.15    # K (coolant bulk)
    h_amb: float = 8.0        # W/m2K  (natural convection-ish)
    h_cool: float = 2500.0    # W/m2K  (forced convection in channel-ish)

    # power map / chips
    P_total: float = 120.0    # W total dissipated on top surface
    n_hotspots: int = 3
    hotspot_sigma: float = 0.012  # m (spread)

    # sampling
    n_col: int = 80000
    n_top: int = 25000       # top face for heat flux
    n_out: int = 25000       # other outer faces for ambient convection
    n_chan: int = 25000      # channel wall for coolant convection
    n_sensors: int = 24      # thermocouples
    sensor_noise_std: float = 0.25  # K

    # training
    seed: int = 123
    deterministic: bool = False
    device: str = "cpu"      # "cuda" if available
    epochs: int = 500
    lr: float = 1e-3
    batch_train: int = 2048
    batch_val: int = 4096

    # loss weights (agora ficam estáveis porque normalizamos)
    w_pde: float = 1.0
    w_top: float = 3.0
    w_out: float = 1.0
    w_chan: float = 2.0
    w_sensors: float = 5.0

    # theta scaling
    dT_ref: float = 30.0     # K (escala típica de delta-T)

    # outputs
    out_dir: str = "examples/_runs_coldplate_industrial"
    run_name: str = "cold_plate_industrial_theta"
    best_name: str = "cold_plate_industrial_theta.best.pt"
    step_name: str = "cold_plate_channel.step"


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int, deterministic: bool = False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

mse = nn.MSELoss()

def inside_solid(xyz: np.ndarray, cfg: Cfg):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    in_box = (
        (x >= -cfg.Lx/2) & (x <= cfg.Lx/2) &
        (y >= -cfg.Ly/2) & (y <= cfg.Ly/2) &
        (z >= -cfg.Lz/2) & (z <= cfg.Lz/2)
    )
    in_hole = (y*y + z*z) <= (cfg.r * cfg.r)
    return in_box & (~in_hole)

def T_from_theta(theta: torch.Tensor, cfg: Cfg):
    return cfg.T_cool + cfg.dT_ref * theta

def theta_from_T(T: np.ndarray, cfg: Cfg):
    return (T - cfg.T_cool) / cfg.dT_ref


# =========================================================
# Industrial-like power map (top heat flux q'')
# =========================================================
def build_hotspots(cfg: Cfg, rng):
    xs = rng.uniform(-0.35*cfg.Lx, 0.35*cfg.Lx, size=cfg.n_hotspots)
    ys = rng.uniform(-0.35*cfg.Ly, 0.35*cfg.Ly, size=cfg.n_hotspots)
    ws = rng.uniform(0.8, 1.3, size=cfg.n_hotspots)
    return np.stack([xs, ys, ws], axis=1).astype(np.float64)

def q_top_map(xy: np.ndarray, cfg: Cfg, hotspots: np.ndarray):
    """
    Returns heat flux q''(x,y) [W/m^2] on top surface z=+Lz/2.
    Normalization: integral over top face equals cfg.P_total.
    """
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    sig2 = cfg.hotspot_sigma**2

    q = np.zeros((xy.shape[0], 1), dtype=np.float64)
    for (x0, y0, w) in hotspots:
        q += w * np.exp(-((x - x0)**2 + (y - y0)**2) / (2.0*sig2))

    q += 0.15

    A = cfg.Lx * cfg.Ly
    avg = float(np.mean(q))
    scale = cfg.P_total / (avg * A + 1e-12)
    return (q * scale).astype(np.float32)


# =========================================================
# Sampling
# =========================================================
def sample_collocation(n: int, cfg: Cfg, rng):
    out = []
    need = n
    while need > 0:
        m = int(need * 1.3) + 1024
        x = rng.uniform(-cfg.Lx/2, cfg.Lx/2, size=m)
        y = rng.uniform(-cfg.Ly/2, cfg.Ly/2, size=m)
        z = rng.uniform(-cfg.Lz/2, cfg.Lz/2, size=m)
        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        mask = inside_solid(xyz, cfg)
        xyz = xyz[mask]
        take = min(xyz.shape[0], need)
        out.append(xyz[:take])
        need -= take
    return np.concatenate(out, axis=0)

def sample_top_face(n: int, cfg: Cfg, rng):
    x = rng.uniform(-cfg.Lx/2, cfg.Lx/2, size=n)
    y = rng.uniform(-cfg.Ly/2, cfg.Ly/2, size=n)
    z = np.full(n, cfg.Lz/2, dtype=np.float32)
    return np.stack([x, y, z], axis=1).astype(np.float32)

def sample_outer_faces_excluding_top(n: int, cfg: Cfg, rng):
    face = rng.integers(0, 5, size=n)
    x = rng.uniform(-cfg.Lx/2, cfg.Lx/2, size=n).astype(np.float32)
    y = rng.uniform(-cfg.Ly/2, cfg.Ly/2, size=n).astype(np.float32)
    z = rng.uniform(-cfg.Lz/2, cfg.Lz/2, size=n).astype(np.float32)

    x[face == 0] = -cfg.Lx/2
    x[face == 1] =  cfg.Lx/2
    y[face == 2] = -cfg.Ly/2
    y[face == 3] =  cfg.Ly/2
    z[face == 4] = -cfg.Lz/2

    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    return xyz, face

def sample_channel_wall(n: int, cfg: Cfg, rng):
    x = rng.uniform(-cfg.Lx/2, cfg.Lx/2, size=n)
    th = rng.uniform(0, 2*np.pi, size=n)
    y = cfg.r * np.cos(th)
    z = cfg.r * np.sin(th)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    return xyz

def sample_sensors(n: int, cfg: Cfg, rng):
    out = []
    need = n
    while need > 0:
        m = int(need * 2.0) + 128
        x = rng.uniform(-0.45*cfg.Lx, 0.45*cfg.Lx, size=m)
        y = rng.uniform(-0.45*cfg.Ly, 0.45*cfg.Ly, size=m)
        z = rng.uniform(-0.40*cfg.Lz, 0.40*cfg.Lz, size=m)
        xyz = np.stack([x, y, z], axis=1).astype(np.float32)
        mask = inside_solid(xyz, cfg)
        xyz = xyz[mask]
        take = min(xyz.shape[0], need)
        out.append(xyz[:take])
        need -= take
    return np.concatenate(out, axis=0)

# =========================================================
# Synthetic "reference" sensors (in Kelvin)
# =========================================================
def T_reference(xyz: np.ndarray, cfg: Cfg, hotspots: np.ndarray):
    x = xyz[:, 0:1].astype(np.float64)
    y = xyz[:, 1:2].astype(np.float64)
    z = xyz[:, 2:3].astype(np.float64)

    sig2 = (cfg.hotspot_sigma**2)
    hot = np.zeros_like(x, dtype=np.float64)
    for (x0, y0, w) in hotspots:
        hot += w * np.exp(-((x-x0)**2 + (y-y0)**2) / (2.0*sig2))
    top_factor = (z - (-cfg.Lz/2)) / (cfg.Lz + 1e-12)
    hot_term = 18.0 * hot * (0.25 + 0.75*top_factor)

    r_yz = np.sqrt((y**2 + z**2) + 1e-12)
    cool_term = -10.0 * np.exp(-(r_yz - cfg.r) / (0.006 + 1e-12))

    base = cfg.T_amb + 2.0
    T = base + hot_term + cool_term
    return T.astype(np.float32).reshape(-1)


# =========================================================
# Dataset (SENSORES em THETA)
# =========================================================
from torch.utils.data import Dataset, DataLoader

def dict_collate(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out

class ColdPlateIndustrialDataset(Dataset):
    def __init__(self, cfg: Cfg, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        rng = np.random.default_rng(seed)

        self.hotspots = build_hotspots(cfg, rng)

        col = sample_collocation(cfg.n_col, cfg, rng)
        top = sample_top_face(cfg.n_top, cfg, rng)
        out, out_face = sample_outer_faces_excluding_top(cfg.n_out, cfg, rng)
        chan = sample_channel_wall(cfg.n_chan, cfg, rng)

        sens = sample_sensors(cfg.n_sensors, cfg, rng)

        # measured temperatures (Kelvin)
        Tm = T_reference(sens, cfg, self.hotspots)
        Tm = Tm + rng.normal(0.0, cfg.sensor_noise_std, size=Tm.shape).astype(np.float32)

        # convert sensors to theta targets
        theta_m = theta_from_T(Tm, cfg).astype(np.float32)

        qtop = q_top_map(top[:, :2], cfg, self.hotspots)  # (n_top,1)

        self.x_col = torch.tensor(col, dtype=torch.float32)
        self.x_top = torch.tensor(top, dtype=torch.float32)
        self.q_top = torch.tensor(qtop, dtype=torch.float32)

        self.x_out = torch.tensor(out, dtype=torch.float32)
        self.out_face = torch.tensor(out_face.astype(np.int64))

        self.x_chan = torch.tensor(chan, dtype=torch.float32)

        self.x_sens = torch.tensor(sens, dtype=torch.float32)

        # store both for reporting
        self.T_sens_K = torch.tensor(Tm[:, None], dtype=torch.float32)
        self.theta_sens = torch.tensor(theta_m[:, None], dtype=torch.float32)

        self.N = self.x_col.size(0)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        j_top = i % self.x_top.size(0)
        j_out = i % self.x_out.size(0)
        j_ch  = i % self.x_chan.size(0)
        j_s   = i % self.x_sens.size(0)

        return {
            "x": self.x_col[i],
            "x_col": self.x_col[i],

            "x_top": self.x_top[j_top],
            "q_top": self.q_top[j_top],

            "x_out": self.x_out[j_out],
            "out_face": self.out_face[j_out],

            "x_chan": self.x_chan[j_ch],

            "x_sens": self.x_sens[j_s],
            "theta_sens": self.theta_sens[j_s],  # target in theta
        }


# =========================================================
# Model wrappers
# - core network outputs THETA
# - Trainer model returns THETA
# =========================================================
class TrainerFriendlyTheta(nn.Module):
    def __init__(self, pinn: VanillaPINN):
        super().__init__()
        self.pinn = pinn

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        out = self.pinn(x)
        return out.y if hasattr(out, "y") else out  # theta


# =========================================================
# Adapter used by PINNeAPPle PDE loss & boundary terms
# - IMPORTANT: adapter returns T (Kelvin) because PDE/BCs in physical units
# =========================================================
class XYZAdapter_T(nn.Module):
    def __init__(self, model_theta: nn.Module, cfg: Cfg):
        super().__init__()
        self.model_theta = model_theta
        self.cfg = cfg

    def forward(self, x, y, z):
        inp = torch.cat([x, y, z], dim=1)
        theta = self.model_theta(inp)
        return T_from_theta(theta, self.cfg)


# =========================================================
# PINNeAPPle PDE loss (Laplacian)
# =========================================================
def build_pde_loss():
    spec = PINNProblemSpec(
        pde_residuals=[
            "Derivative(T(x,y,z), x, 2) + Derivative(T(x,y,z), y, 2) + Derivative(T(x,y,z), z, 2)"
        ],
        conditions=[],
        independent_vars=["x", "y", "z"],
        dependent_vars=["T"],
        inverse_params=[],
        verbose=False,
    )
    return PINNFactory(spec).generate_loss_function()


# =========================================================
# Boundary residuals helpers
# =========================================================
def grad_T_wrt_xyz(T, x, y, z):
    ones = torch.ones_like(T)
    dTdx = torch.autograd.grad(T, x, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    dTdy = torch.autograd.grad(T, y, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    dTdz = torch.autograd.grad(T, z, grad_outputs=ones, retain_graph=True, create_graph=True)[0]
    return dTdx, dTdy, dTdz


# =========================================================
# Loss (NORMALIZADO) + sensores em theta
# =========================================================
def make_loss_fn(cfg: Cfg, pde_loss_fn, adapter_T: nn.Module, q_ref: float, L_ref: float):
    k = cfg.k
    h_amb = cfg.h_amb
    h_cool = cfg.h_cool
    Tamb = cfg.T_amb
    Tcool = cfg.T_cool

    dT_ref = cfg.dT_ref
    # denominators (avoid tiny)
    denom_top = max(q_ref, 1.0)                       # W/m2
    denom_out = max(h_amb * dT_ref, 1.0)              # W/m2
    denom_ch  = max(h_cool * dT_ref, 1.0)             # W/m2

    # PDE scaling: make Laplacian term ~O(1) when T varies by dT_ref over L_ref
    # Lap ~ dT_ref / L_ref^2  => normalized residual = Lap * (L_ref^2 / dT_ref)
    pde_scale = (L_ref * L_ref) / max(dT_ref, 1e-6)

    def loss_fn(model_theta, y_hat, batch):
        device = next(model_theta.parameters()).device

        with torch.enable_grad():
            # -------------------------
            # PDE: Laplacian(T)=0 (normalized)
            # -------------------------
            xcol = batch["x_col"].to(device)
            x = xcol[:, 0:1].detach().clone().requires_grad_(True)
            y = xcol[:, 1:2].detach().clone().requires_grad_(True)
            z = xcol[:, 2:3].detach().clone().requires_grad_(True)

            l_lap, aux = pde_loss_fn(adapter_T, {"collocation": (x, y, z)})

            lap = aux.get("pde_residuals", None) if isinstance(aux, dict) else None
            if isinstance(lap, (list, tuple)):
                lap = lap[0]
            if lap is None:
                # fallback
                l_pde = l_lap
            else:
                res_pde_n = (lap * pde_scale)
                l_pde = torch.mean(res_pde_n**2)

            # -------------------------
            # TOP Neumann: -k dT/dn = q''  (normalized by q_ref)
            # -------------------------
            xtop = batch["x_top"].to(device)
            qtop = batch["q_top"].to(device)

            xt = xtop[:, 0:1].detach().clone().requires_grad_(True)
            yt = xtop[:, 1:2].detach().clone().requires_grad_(True)
            zt = xtop[:, 2:3].detach().clone().requires_grad_(True)

            Tt = adapter_T(xt, yt, zt)
            _, _, dTdz = grad_T_wrt_xyz(Tt, xt, yt, zt)

            res_top = (-k * dTdz) - qtop
            res_top_n = res_top / denom_top
            l_top = torch.mean(res_top_n**2)

            # -------------------------
            # OUTER Robin: -k dT/dn = h (T - Tamb) (normalized by h*dT_ref)
            # -------------------------
            xout = batch["x_out"].to(device)
            face = batch["out_face"].to(device)

            xo = xout[:, 0:1].detach().clone().requires_grad_(True)
            yo = xout[:, 1:2].detach().clone().requires_grad_(True)
            zo = xout[:, 2:3].detach().clone().requires_grad_(True)

            To = adapter_T(xo, yo, zo)
            dTdx, dTdy, dTdz = grad_T_wrt_xyz(To, xo, yo, zo)

            nx = torch.zeros_like(dTdx)
            ny = torch.zeros_like(dTdy)
            nz = torch.zeros_like(dTdz)
            nx[face == 0] = -1.0
            nx[face == 1] = +1.0
            ny[face == 2] = -1.0
            ny[face == 3] = +1.0
            nz[face == 4] = -1.0

            dTdn_out = dTdx*nx + dTdy*ny + dTdz*nz
            res_out = (-k * dTdn_out) - (h_amb * (To - Tamb))
            res_out_n = res_out / denom_out
            l_out = torch.mean(res_out_n**2)

            # -------------------------
            # CHANNEL Robin: -k dT/dn = h_c (T - Tcool) (normalized by h_c*dT_ref)
            # -------------------------
            xch = batch["x_chan"].to(device)

            xc = xch[:, 0:1].detach().clone().requires_grad_(True)
            yc = xch[:, 1:2].detach().clone().requires_grad_(True)
            zc = xch[:, 2:3].detach().clone().requires_grad_(True)

            Tc = adapter_T(xc, yc, zc)
            dTdx, dTdy, dTdz = grad_T_wrt_xyz(Tc, xc, yc, zc)

            rr = torch.sqrt(yc**2 + zc**2 + 1e-12)
            ny_c = yc / rr
            nz_c = zc / rr
            dTdn_chan = dTdy*ny_c + dTdz*nz_c

            res_chan = (-k * dTdn_chan) - (h_cool * (Tc - Tcool))
            res_chan_n = res_chan / denom_ch
            l_chan = torch.mean(res_chan_n**2)

            # -------------------------
            # SENSOR data in THETA (dimensionless, stable)
            # -------------------------
            xs = batch["x_sens"].to(device)
            theta_target = batch["theta_sens"].to(device)
            theta_pred = model_theta(xs)
            l_sens = mse(theta_pred, theta_target)

            total = (
                cfg.w_pde*l_pde +
                cfg.w_top*l_top +
                cfg.w_out*l_out +
                cfg.w_chan*l_chan +
                cfg.w_sensors*l_sens
            )

        return {
            "total": total,
            "pde": l_pde.detach(),
            "top": l_top.detach(),
            "out": l_out.detach(),
            "chan": l_chan.detach(),
            "sens": l_sens.detach(),
        }

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
    print(f"[ckpt] missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")


# =========================================================
# Plot helpers (report in K)
# =========================================================
def safe_hist(ax, data, label, max_bins=60, alpha=0.6):
    d = np.asarray(data, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 5:
        ax.text(0.5, 0.5, f"sem dados: {label}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        return
    dmin = float(d.min()); dmax = float(d.max())
    if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
        eps = 1e-6 if np.isfinite(dmin) else 1.0
        ax.hist(d, bins=3, range=((dmin if np.isfinite(dmin) else 0.0)-eps,
                                 (dmax if np.isfinite(dmax) else 0.0)+eps),
                alpha=alpha, label=label)
        return
    q25, q75 = np.percentile(d, [25, 75])
    iqr = float(q75 - q25)
    if iqr <= 0:
        bins = min(10, max_bins)
        ax.hist(d, bins=bins, range=(dmin, dmax), alpha=alpha, label=label)
        return
    bw = 2.0 * iqr * (d.size ** (-1/3))
    bins = int(np.clip(np.ceil((dmax - dmin) / (bw + 1e-12)), 3, max_bins))
    ax.hist(d, bins=bins, range=(dmin, dmax), alpha=alpha, label=label)

def eval_top_surface(model_theta, cfg: Cfg, nx=260, ny=180):
    xs = np.linspace(-cfg.Lx/2, cfg.Lx/2, nx, dtype=np.float32)
    ys = np.linspace(-cfg.Ly/2, cfg.Ly/2, ny, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    ZZ = np.full_like(XX, cfg.Lz/2, dtype=np.float32)
    pts = np.stack([XX.reshape(-1), YY.reshape(-1), ZZ.reshape(-1)], axis=1)

    with torch.no_grad():
        th = model_theta(torch.tensor(pts, dtype=torch.float32, device=cfg.device)).detach().cpu().numpy().reshape(-1)
    T = cfg.T_cool + cfg.dT_ref * th
    Timg = T.reshape(ny, nx)
    extent = [-cfg.Lx/2, cfg.Lx/2, -cfg.Ly/2, cfg.Ly/2]
    return Timg, extent

def eval_linecut(model_theta, cfg: Cfg, y0=0.0, z0=None, n=400):
    if z0 is None:
        z0 = cfg.Lz/2
    xs = np.linspace(-cfg.Lx/2, cfg.Lx/2, n, dtype=np.float32)
    ys = np.full_like(xs, y0, dtype=np.float32)
    zs = np.full_like(xs, z0, dtype=np.float32)
    pts = np.stack([xs, ys, zs], axis=1)
    with torch.no_grad():
        th = model_theta(torch.tensor(pts, dtype=torch.float32, device=cfg.device)).detach().cpu().numpy().reshape(-1)
    T = cfg.T_cool + cfg.dT_ref * th
    return xs, T

def eval_sensors(model_theta, ds: ColdPlateIndustrialDataset, cfg: Cfg):
    x = ds.x_sens.numpy()
    Tm = ds.T_sens_K.numpy().reshape(-1)
    with torch.no_grad():
        th = model_theta(ds.x_sens.to(cfg.device)).detach().cpu().numpy().reshape(-1)
    Tp = cfg.T_cool + cfg.dT_ref * th
    err = Tp - Tm
    return x, Tm, Tp, err

def compute_kpis(model_theta, cfg: Cfg):
    Timg, _ = eval_top_surface(model_theta, cfg, nx=220, ny=160)
    Tmax = float(np.nanmax(Timg))
    Tcool = cfg.T_cool
    P = cfg.P_total
    Rth = (Tmax - Tcool) / (P + 1e-12)
    return {"Tmax": Tmax, "Tcool": float(Tcool), "P": float(P), "Rth_K_per_W": float(Rth)}

def plot_industrial_report(model_theta, ds: ColdPlateIndustrialDataset, cfg: Cfg, out_png: str):
    Timg, extent = eval_top_surface(model_theta, cfg)
    xs, Tline = eval_linecut(model_theta, cfg, y0=0.0, z0=cfg.Lz/2)
    x_s, Tm, Tp, err = eval_sensors(model_theta, ds, cfg)
    kpi = compute_kpis(model_theta, cfg)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    im = axes[0, 0].imshow(Timg, origin="lower", extent=extent)
    axes[0, 0].set_title("Top surface temperature [K] (IR-like)")
    axes[0, 0].set_xlabel("x [m]"); axes[0, 0].set_ylabel("y [m]")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    XY = np.stack(np.meshgrid(
        np.linspace(-cfg.Lx/2, cfg.Lx/2, Timg.shape[1], dtype=np.float32),
        np.linspace(-cfg.Ly/2, cfg.Ly/2, Timg.shape[0], dtype=np.float32),
        indexing="xy"
    ), axis=-1).reshape(-1, 2)
    q = q_top_map(XY, cfg, ds.hotspots).reshape(Timg.shape[0], Timg.shape[1])
    im2 = axes[0, 1].imshow(q, origin="lower", extent=extent)
    axes[0, 1].set_title("Top heat flux q'' [W/m²] (power map)")
    axes[0, 1].set_xlabel("x [m]"); axes[0, 1].set_ylabel("y [m]")
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].plot(xs, Tline)
    axes[0, 2].set_title("Line cut on top: y=0, z=+Lz/2")
    axes[0, 2].set_xlabel("x [m]"); axes[0, 2].set_ylabel("T [K]")
    axes[0, 2].grid(True)

    axes[1, 0].scatter(Tm, Tp, s=25)
    mn = float(min(Tm.min(), Tp.min()))
    mx = float(max(Tm.max(), Tp.max()))
    axes[1, 0].plot([mn, mx], [mn, mx])
    axes[1, 0].set_title("Sensors parity: Pred vs Measured")
    axes[1, 0].set_xlabel("Measured [K]"); axes[1, 0].set_ylabel("Predicted [K]")
    axes[1, 0].grid(True)

    safe_hist(axes[1, 1], err, "sensor error (K)", max_bins=25, alpha=0.75)
    axes[1, 1].set_title(f"Sensor error histogram | MAE={np.mean(np.abs(err)):.3f} K")
    axes[1, 1].set_xlabel("Pred - Meas [K]"); axes[1, 1].set_ylabel("count")
    axes[1, 1].grid(True)

    axes[1, 2].axis("off")
    txt = (
        "KPIs (quick):\n"
        f"- P_total: {kpi['P']:.1f} W\n"
        f"- T_cool: {kpi['Tcool']:.2f} K\n"
        f"- T_max(top): {kpi['Tmax']:.2f} K\n"
        f"- R_th: {kpi['Rth_K_per_W']:.4f} K/W\n\n"
        "Obs: R_th=(Tmax-Tcool)/P\n"
    )
    axes[1, 2].text(0.0, 1.0, txt, va="top", fontsize=12)

    fig.suptitle("Cold Plate PINN Report (industry-style) — theta training", fontsize=16)
    fig.savefig(out_png, dpi=180)
    plt.show()


# =========================================================
# Main
# =========================================================
def main(TRAIN=True):
    cfg = Cfg()
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    # STEP
    step_path = os.path.join(cfg.out_dir, cfg.step_name)
    try:
        generate_step_cold_plate(step_path, Lx=cfg.Lx, Ly=cfg.Ly, Lz=cfg.Lz, r=cfg.r)
        print("[cad] STEP saved:", step_path)
    except Exception as e:
        print("[cad] STEP skipped (CadQuery missing or error):", str(e))

    # dataset
    ds = ColdPlateIndustrialDataset(cfg, seed=42)
    train_loader = DataLoader(ds, batch_size=cfg.batch_train, shuffle=True, collate_fn=dict_collate)
    val_loader   = DataLoader(ds, batch_size=cfg.batch_val, shuffle=False, collate_fn=dict_collate)

    # model: outputs theta
    pinn_core = VanillaPINN(in_dim=3, out_dim=1, hidden=(128, 128, 128, 128), activation="tanh")
    model_theta = TrainerFriendlyTheta(pinn_core).to(cfg.device)

    ckpt_path = os.path.join(cfg.out_dir, cfg.best_name)

    if TRAIN:
        # scales for normalization
        A = cfg.Lx * cfg.Ly
        q_ref = max(cfg.P_total / (A + 1e-12), 1.0)    # W/m2
        L_ref = min(cfg.Lx, cfg.Ly, cfg.Lz)

        pde_loss_fn = build_pde_loss()
        adapter_T = XYZAdapter_T(model_theta, cfg)

        loss_fn = make_loss_fn(cfg, pde_loss_fn, adapter_T, q_ref=q_ref, L_ref=L_ref)

        trainer = Trainer(model=model_theta, loss_fn=loss_fn, metrics=default_metrics())
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
        load_checkpoint_into_model(model_theta, ckpt_path, device=cfg.device)
    else:
        print(f"[warn] ckpt not found at {ckpt_path} — using current weights in memory.")

    model_theta.eval()
    out_png = os.path.join(cfg.out_dir, f"{cfg.run_name}_industrial_report.png")
    plot_industrial_report(model_theta, ds, cfg, out_png)
    print("[report] saved:", out_png)


if __name__ == "__main__":
    main(TRAIN=True)
