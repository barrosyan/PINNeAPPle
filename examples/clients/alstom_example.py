import os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

from pinneaple_pinn.factory.pinn_factory import PINNFactory, PINNProblemSpec
from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.metrics import default_metrics
from pinneaple_models.pinns.vanilla import VanillaPINN


TRAIN = True  # True: train | False: load best.pt and only report


# =========================================================
# Config
# =========================================================
@dataclass
class Cfg:
    # -------------------------
    # Vehicle params (linear bicycle model)
    # -------------------------
    m: float = 1500.0       # kg
    Iz: float = 3000.0      # kg*m^2
    a: float = 1.2          # m (CG -> front axle)
    b: float = 1.6          # m (CG -> rear axle)
    Cf: float = 80000.0     # N/rad (front cornering stiffness)
    Cr: float = 80000.0     # N/rad (rear cornering stiffness)
    Vx: float = 20.0        # m/s (constant longitudinal speed)

    # time domain
    t0: float = 0.0
    t1: float = 5.0

    # sampling
    n_col: int = 60000        # collocation points
    n_ic: int = 4096          # initial condition points (t=0 repeated)
    n_data: int = 2048        # supervised "measurements" points

    # training
    seed: int = 123
    deterministic: bool = False
    device: str = "cpu"  # "cuda"
    epochs: int = 600
    lr: float = 1e-3
    batch_train: int = 1024
    batch_val: int = 2048

    # weights
    w_pde: float = 1.0
    w_ic: float = 50.0
    w_data: float = 5.0

    # synthetic measurement noise (optional)
    y_noise_std: float = 0.02   # meters
    psi_noise_std: float = 0.002 # rad

    # output
    out_dir: str = "examples/_runs_vehicle_bicycle"
    run_name: str = "vehicle_bicycle_pinn"
    best_name: str = "vehicle_bicycle_pinn.best.pt"


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
# Steering profile (control input) delta(t)
# lane change-like smooth pulse
# =========================================================
def delta_np(t: np.ndarray):
    # smooth pulse using tanh
    # feel free to tweak amplitude/duration
    A = 0.06   # rad (~3.4 deg)
    t1, t2 = 1.0, 3.0
    k = 6.0
    return A * 0.5 * (np.tanh(k*(t - t1)) - np.tanh(k*(t - t2)))

def delta_torch(t: torch.Tensor):
    A = 0.06
    t1, t2 = 1.0, 3.0
    k = 6.0
    return A * 0.5 * (torch.tanh(k*(t - t1)) - torch.tanh(k*(t - t2)))


# =========================================================
# Reference simulator (for synthetic measurements only)
# Linear bicycle model in state x=[y, psi, vy, r]
#
# Equations (small angles, constant Vx):
# y_dot   = vy + Vx*psi
# psi_dot = r
#
# vy_dot  = -(Cf+Cr)/(m*Vx) * vy - (a*Cf - b*Cr)/(m*Vx) * r + (Cf/m)*delta
# r_dot   = -(a*Cf - b*Cr)/(Iz*Vx)*vy - (a^2*Cf + b^2*Cr)/(Iz*Vx)*r + (a*Cf/Iz)*delta
# =========================================================
def simulate_reference(cfg: Cfg, t: np.ndarray):
    dt = float(t[1] - t[0])

    y, psi, vy, r = 0.0, 0.0, 0.0, 0.0  # initial state
    Y = np.zeros_like(t)
    PSI = np.zeros_like(t)

    for i in range(len(t)):
        Y[i] = y
        PSI[i] = psi

        d = float(delta_np(np.array([t[i]]))[0])

        # compute derivatives
        y_dot = vy + cfg.Vx * psi
        psi_dot = r

        vy_dot = -((cfg.Cf + cfg.Cr) / (cfg.m * cfg.Vx)) * vy \
                 - ((cfg.a*cfg.Cf - cfg.b*cfg.Cr) / (cfg.m * cfg.Vx)) * r \
                 + (cfg.Cf / cfg.m) * d

        r_dot = -((cfg.a*cfg.Cf - cfg.b*cfg.Cr) / (cfg.Iz * cfg.Vx)) * vy \
                - ((cfg.a**2 * cfg.Cf + cfg.b**2 * cfg.Cr) / (cfg.Iz * cfg.Vx)) * r \
                + (cfg.a * cfg.Cf / cfg.Iz) * d

        # simple RK2 (midpoint)
        y_mid = y + 0.5*dt*y_dot
        psi_mid = psi + 0.5*dt*psi_dot
        vy_mid = vy + 0.5*dt*vy_dot
        r_mid = r + 0.5*dt*r_dot

        d_mid = float(delta_np(np.array([t[i] + 0.5*dt]))[0])

        y_dot2 = vy_mid + cfg.Vx * psi_mid
        psi_dot2 = r_mid

        vy_dot2 = -((cfg.Cf + cfg.Cr) / (cfg.m * cfg.Vx)) * vy_mid \
                  - ((cfg.a*cfg.Cf - cfg.b*cfg.Cr) / (cfg.m * cfg.Vx)) * r_mid \
                  + (cfg.Cf / cfg.m) * d_mid

        r_dot2 = -((cfg.a*cfg.Cf - cfg.b*cfg.Cr) / (cfg.Iz * cfg.Vx)) * vy_mid \
                 - ((cfg.a*cfg.Cf - cfg.b*cfg.Cr) / (cfg.Iz * cfg.Vx)) * 0.0 \
                 - ((cfg.a**2 * cfg.Cf + cfg.b**2 * cfg.Cr) / (cfg.Iz * cfg.Vx)) * r_mid \
                 + (cfg.a * cfg.Cf / cfg.Iz) * d_mid

        y += dt * y_dot2
        psi += dt * psi_dot2
        vy += dt * vy_dot2
        r += dt * r_dot2

    return Y, PSI


# =========================================================
# Dataset
# x = [t], y_target = [y, psi] at sparse data points
# =========================================================
class VehicleBicycleDataset(Dataset):
    def __init__(self, cfg: Cfg, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)

        # collocation points over time
        t_col = rng.random(cfg.n_col) * (cfg.t1 - cfg.t0) + cfg.t0
        t_col = t_col.astype(np.float32)[:, None]

        # IC points (t=0 repeated)
        t_ic = np.full((cfg.n_ic, 1), cfg.t0, dtype=np.float32)

        # synthetic measurement points
        t_data = rng.random(cfg.n_data) * (cfg.t1 - cfg.t0) + cfg.t0
        t_data = t_data.astype(np.float32)
        t_grid = np.linspace(cfg.t0, cfg.t1, 2001, dtype=np.float64)
        Yref, PSIref = simulate_reference(cfg, t_grid)

        # interpolate reference to measurement times
        y_meas = np.interp(t_data, t_grid, Yref).astype(np.float32)
        psi_meas = np.interp(t_data, t_grid, PSIref).astype(np.float32)

        # add noise (optional)
        y_meas += rng.normal(0.0, cfg.y_noise_std, size=y_meas.shape).astype(np.float32)
        psi_meas += rng.normal(0.0, cfg.psi_noise_std, size=psi_meas.shape).astype(np.float32)

        self.t_col = torch.tensor(t_col)
        self.t_ic = torch.tensor(t_ic)

        self.t_data = torch.tensor(t_data[:, None])
        self.ypsi_data = torch.tensor(np.stack([y_meas, psi_meas], axis=1))

        self.N = self.t_col.size(0)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return {
            "x": self.t_col[i],          # trainer expects "x" usually
            "t_col": self.t_col[i],
            "t_ic": self.t_ic[i % self.t_ic.size(0)],
            "t_data": self.t_data[i % self.t_data.size(0)],
            "ypsi_data": self.ypsi_data[i % self.ypsi_data.size(0)],
        }


# =========================================================
# Model wrapper
# outputs: [y(t), psi(t), vy(t), r(t)]
# but we will also supervise y,psi on sparse data
# =========================================================
class TrainerFriendlyVanilla(nn.Module):
    def __init__(self, pinn: VanillaPINN):
        super().__init__()
        self.pinn = pinn

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        return as_y(self.pinn(x))


class TAdapter(nn.Module):
    """
    Adapter so PINNFactory can call model(state)(t) style.
    state(t) is a vector: [y, psi, vy, r]
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, t):
        return as_y(self.model(t))


# =========================================================
# PINNFactory residuals
# Define state variables: y(t), psi(t), vy(t), r(t)
# =========================================================
def build_pde_loss(cfg: Cfg):
    # IMPORTANT:
    # PINNFactory expressions are symbolic; we pass constants as numbers in the string.
    m = cfg.m
    Iz = cfg.Iz
    a = cfg.a
    b = cfg.b
    Cf = cfg.Cf
    Cr = cfg.Cr
    Vx = cfg.Vx

    # delta(t) is control input:
    # We'll encode it directly as tanh pulse (same as delta_torch but in sympy-like string).
    # (Most robust: keep it simple and deterministic.)
    A = 0.06
    t1 = 1.0
    t2 = 3.0
    k = 6.0
    delta_expr = f"({A}*0.5*(tanh({k}*(t-{t1})) - tanh({k}*(t-{t2}))))"

    # ODE residuals:
    # y'   - (vy + Vx*psi) = 0
    # psi' - r            = 0
    # vy'  - [ ... ]      = 0
    # r'   - [ ... ]      = 0
    spec = PINNProblemSpec(
        pde_residuals=[
            f"Derivative(y(t), t) - (vy(t) + ({Vx})*psi(t))",
            f"Derivative(psi(t), t) - r(t)",
            (
                f"Derivative(vy(t), t) - ("
                f"-(({Cf}+{Cr})/({m}*{Vx}))*vy(t)"
                f" - (({a}*{Cf}-{b}*{Cr})/({m}*{Vx}))*r(t)"
                f" + ({Cf}/{m})*({delta_expr})"
                f")"
            ),
            (
                f"Derivative(r(t), t) - ("
                f"-(({a}*{Cf}-{b}*{Cr})/({Iz}*{Vx}))*vy(t)"
                f" - (({a}**2*{Cf}+{b}**2*{Cr})/({Iz}*{Vx}))*r(t)"
                f" + ({a}*{Cf}/{Iz})*({delta_expr})"
                f")"
            ),
        ],
        conditions=[],
        independent_vars=["t"],
        dependent_vars=["y", "psi", "vy", "r"],
        inverse_params=[],
        verbose=True,
    )
    return PINNFactory(spec).generate_loss_function()


# =========================================================
# Loss function (PDE + IC + DATA)
# =========================================================
def make_loss_fn(cfg: Cfg, pde_loss_fn, adapter: nn.Module):
    def loss_fn(model, y_hat, batch):
        device = cfg.device

        with torch.enable_grad():
            # -------------------------
            # PDE residual at collocation
            # -------------------------
            tcol = batch["t_col"].to(device).detach().clone().requires_grad_(True)
            l_pde, _ = pde_loss_fn(adapter, {"collocation": (tcol,)})

            # -------------------------
            # Initial conditions at t=0:
            # y(0)=0, psi(0)=0, vy(0)=0, r(0)=0
            # -------------------------
            tic = batch["t_ic"].to(device).detach().clone().requires_grad_(True)
            s0 = model(tic)  # [y,psi,vy,r]
            if s0.ndim == 1:
                s0 = s0[:, None]
            target0 = torch.zeros_like(s0)
            l_ic = mse(s0, target0)

            # -------------------------
            # Supervised data (sparse): y(t), psi(t)
            # -------------------------
            tdata = batch["t_data"].to(device).detach().clone().requires_grad_(True)
            ypsi_meas = batch["ypsi_data"].to(device)

            sdata = model(tdata)
            ypsi_pred = sdata[:, 0:2]
            l_data = mse(ypsi_pred, ypsi_meas)

            total = cfg.w_pde*l_pde + cfg.w_ic*l_ic + cfg.w_data*l_data

        return {
            "total": total,
            "pde": l_pde.detach(),
            "ic": l_ic.detach(),
            "data": l_data.detach(),
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
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")


# =========================================================
# Report
# =========================================================
@torch.no_grad()
def evaluate(model, cfg: Cfg, n_pts: int = 2000):
    t = np.linspace(cfg.t0, cfg.t1, n_pts, dtype=np.float32)[:, None]
    tt = torch.tensor(t, dtype=torch.float32, device=cfg.device)
    s = model(tt).detach().cpu().numpy()
    y = s[:, 0]
    psi = s[:, 1]
    vy = s[:, 2]
    r = s[:, 3]

    # reference
    t_ref = np.linspace(cfg.t0, cfg.t1, n_pts, dtype=np.float64)
    y_ref, psi_ref = simulate_reference(cfg, t_ref)

    return t.reshape(-1), y, psi, vy, r, t_ref, y_ref, psi_ref

def plot_report(model, cfg: Cfg, out_png: str):
    t, y, psi, vy, r, t_ref, y_ref, psi_ref = evaluate(model, cfg)

    plt.figure(figsize=(10, 5))
    plt.plot(t, y, label="PINN y(t)")
    plt.plot(t_ref, y_ref, "--", label="Ref y(t)")
    plt.title("Vehicle lateral motion (bicycle model) | y(t)")
    plt.xlabel("t [s]"); plt.ylabel("y [m]")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_y.png"), dpi=180)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(t, psi, label="PINN psi(t)")
    plt.plot(t_ref, psi_ref, "--", label="Ref psi(t)")
    plt.title("Vehicle lateral motion (bicycle model) | psi(t)")
    plt.xlabel("t [s]"); plt.ylabel("psi [rad]")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_psi.png"), dpi=180)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(t, vy, label="PINN vy(t)")
    plt.plot(t, r, label="PINN r(t)")
    plt.title("Vehicle states inferred by physics | vy(t), r(t)")
    plt.xlabel("t [s]"); plt.ylabel("state")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_vy_r.png"), dpi=180)
    plt.show()

    err_y = y - np.interp(t, t_ref, y_ref)
    err_psi = psi - np.interp(t, t_ref, psi_ref)

    mae_y = float(np.mean(np.abs(err_y)))
    rmse_y = float(np.sqrt(np.mean(err_y**2)))
    mae_psi = float(np.mean(np.abs(err_psi)))
    rmse_psi = float(np.sqrt(np.mean(err_psi**2)))

    print(f"[metrics] y:   MAE={mae_y:.3e} RMSE={rmse_y:.3e}")
    print(f"[metrics] psi: MAE={mae_psi:.3e} RMSE={rmse_psi:.3e}")


# =========================================================
# Main
# =========================================================
def main():
    cfg = Cfg()
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    ds = VehicleBicycleDataset(cfg, seed=42)
    train_loader = DataLoader(ds, batch_size=cfg.batch_train, shuffle=True, collate_fn=dict_collate)
    val_loader   = DataLoader(ds, batch_size=cfg.batch_val, shuffle=False, collate_fn=dict_collate)

    # outputs: y, psi, vy, r
    pinn_core = VanillaPINN(in_dim=1, out_dim=4, hidden=(128, 128, 128, 128), activation="tanh")
    model = TrainerFriendlyVanilla(pinn_core).to(cfg.device)

    ckpt_path = os.path.join(cfg.out_dir, cfg.best_name)

    if TRAIN:
        pde_loss_fn = build_pde_loss(cfg)
        adapter = TAdapter(model)
        loss_fn = make_loss_fn(cfg, pde_loss_fn, adapter)

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
        print(f"[warn] ckpt not found at {ckpt_path} — using current weights in memory.")

    model.eval()
    out_png = os.path.join(cfg.out_dir, f"{cfg.run_name}_report.png")
    plot_report(model, cfg, out_png)
    print("[report] saved base name:", out_png)


if __name__ == "__main__":
    main()
