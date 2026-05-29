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
    # Lane–Emden index
    n: float = 1.0

    # domain
    xi_max: float = 10.0
    eps_colloc0: float = 1e-6  # avoid exact 0 inside collocation residual (not needed, but safe)

    # sampling
    n_col: int = 60000
    n_bc: int = 4096  # points near 0 for BC (we'll just use xi=0 repeated)

    # training
    seed: int = 123
    deterministic: bool = False
    device: str = "cpu"  # "cuda"
    epochs: int = 400
    lr: float = 1e-3
    batch_train: int = 1024
    batch_val: int = 2048

    # weights
    w_pde: float = 1.0
    w_bc0: float = 50.0   # theta(0)=1
    w_bc1: float = 10.0   # theta'(0)=0

    # output
    out_dir: str = "examples/_runs_lane_emden"
    run_name: str = "lane_emden_pinn"
    best_name: str = "lane_emden_pinn.best.pt"


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
# Optional analytic solutions for sanity (n=0,1,5)
# =========================================================
def theta_true(xi: np.ndarray, n: float):
    xi = np.asarray(xi, dtype=np.float64)
    if abs(n - 0.0) < 1e-12:
        return (1.0 - xi**2 / 6.0).astype(np.float64)
    if abs(n - 1.0) < 1e-12:
        out = np.ones_like(xi)
        m = xi > 1e-12
        out[m] = np.sin(xi[m]) / xi[m]
        return out.astype(np.float64)
    if abs(n - 5.0) < 1e-12:
        return (1.0 / np.sqrt(1.0 + xi**2 / 3.0)).astype(np.float64)
    return None  # no closed form here


# =========================================================
# Dataset
# =========================================================
class LaneEmdenDataset(Dataset):
    def __init__(self, cfg: Cfg, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)

        # collocation points: sample in (0, xi_max]
        col = rng.random(cfg.n_col) * (cfg.xi_max - cfg.eps_colloc0) + cfg.eps_colloc0
        col = col.astype(np.float32)[:, None]

        # BC points: xi = 0 repeated
        bc = np.zeros((cfg.n_bc, 1), dtype=np.float32)

        self.x_col = torch.tensor(col)
        self.x_bc = torch.tensor(bc)
        self.N = self.x_col.size(0)

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        return {
            "x": self.x_col[i],
            "x_col": self.x_col[i],
            "x_bc": self.x_bc[i % self.x_bc.size(0)],
        }


# =========================================================
# Model wrapper
# =========================================================
class TrainerFriendlyVanilla(nn.Module):
    def __init__(self, pinn: VanillaPINN):
        super().__init__()
        self.pinn = pinn

    def forward(self, x):
        if isinstance(x, dict):
            x = x["x"]
        return as_y(self.pinn(x))


class XAdapter(nn.Module):
    """
    Adapter so PINNFactory can call model(theta)(x) style.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, xi):
        # xi shape (N,1)
        return as_y(self.model(xi))


# =========================================================
# PINNFactory residual (regularized Lane–Emden)
# d/dxi( xi^2 * dtheta/dxi ) + xi^2 * theta^n = 0
# =========================================================
def build_pde_loss(cfg: Cfg):
    # Lane–Emden expanded form:
    # theta'' + (2/x)*theta' + theta^n = 0
    # Important: collocation must avoid x=0 (you already do via eps_colloc0)
    spec = PINNProblemSpec(
        pde_residuals=[
            f"Derivative(theta(x), (x, 2)) + (2/x)*Derivative(theta(x), x) + theta(x)**{cfg.n}"
        ],
        conditions=[],
        independent_vars=["x"],
        dependent_vars=["theta"],
        inverse_params=[],
        verbose=True,
    )
    return PINNFactory(spec).generate_loss_function()



# =========================================================
# Loss function (PDE + BCs)
# =========================================================
def make_loss_fn(cfg: Cfg, pde_loss_fn, adapter: nn.Module):
    def loss_fn(model, y_hat, batch):
        device = cfg.device

        # IMPORTANT: Trainer geralmente roda val com torch.no_grad().
        # Precisamos re-habilitar grad aqui porque usamos autograd (PDE + BC derivada).
        with torch.enable_grad():
            # -------------------------
            # PDE residual at collocation
            # -------------------------
            xcol = batch["x_col"].to(device).detach().clone().requires_grad_(True)
            l_pde, _ = pde_loss_fn(adapter, {"collocation": (xcol,)})

            # -------------------------
            # BCs at xi = 0
            # theta(0) = 1
            # theta'(0) = 0
            # -------------------------
            x0 = batch["x_bc"].to(device).detach().clone().requires_grad_(True)

            th0 = model(x0)  # NÃO use y_hat; recompute com grad habilitado
            # garante shape consistente
            if th0.ndim == 1:
                th0 = th0[:, None]

            l_bc0 = mse(th0, torch.ones_like(th0))

            dth_dx = torch.autograd.grad(
                outputs=th0,
                inputs=x0,
                grad_outputs=torch.ones_like(th0),
                create_graph=True,
                retain_graph=True,
            )[0]
            l_bc1 = mse(dth_dx, torch.zeros_like(dth_dx))

            total = cfg.w_pde*l_pde + cfg.w_bc0*l_bc0 + cfg.w_bc1*l_bc1

        # logs (detach pra não segurar grafo)
        return {
            "total": total,
            "pde": l_pde.detach(),
            "bc_theta0": l_bc0.detach(),
            "bc_dtheta0": l_bc1.detach(),
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
def evaluate_curve(model, cfg: Cfg, n_pts: int = 2000):
    xi = np.linspace(0.0, cfg.xi_max, n_pts, dtype=np.float32)[:, None]
    xt = torch.tensor(xi, dtype=torch.float32, device=cfg.device)
    th_pred = model(xt).detach().cpu().numpy().reshape(-1)

    th_true = theta_true(xi.reshape(-1), cfg.n)
    return xi.reshape(-1), th_pred, th_true

def plot_report(model, cfg: Cfg, out_png: str):
    xi, th_pred, th_true = evaluate_curve(model, cfg)

    plt.figure(figsize=(10, 5))
    plt.plot(xi, th_pred, label="PINN (pred)")
    if th_true is not None:
        plt.plot(xi, th_true, "--", label="Analytic (true)")
    plt.title(f"Lane–Emden solution | n={cfg.n} | xi_max={cfg.xi_max}")
    plt.xlabel("xi"); plt.ylabel("theta(xi)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.show()

    if th_true is not None:
        err = th_pred - th_true
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err**2)))
        rel = float(np.linalg.norm(err) / (np.linalg.norm(th_true) + 1e-12))

        plt.figure(figsize=(10, 4))
        plt.plot(xi, err)
        plt.title(f"Error vs analytic | MAE={mae:.3e} RMSE={rmse:.3e} relL2={rel:.3e}")
        plt.xlabel("xi"); plt.ylabel("pred-true")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        print(f"[metrics] MAE={mae:.6e} RMSE={rmse:.6e} relL2={rel:.6e}")
    else:
        print("[info] No analytic reference for this n. (Still valid PINN.)")


# =========================================================
# Main
# =========================================================
def main():
    cfg = Cfg()
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed, deterministic=cfg.deterministic)

    ds = LaneEmdenDataset(cfg, seed=42)
    train_loader = DataLoader(ds, batch_size=cfg.batch_train, shuffle=True, collate_fn=dict_collate)
    val_loader   = DataLoader(ds, batch_size=cfg.batch_val, shuffle=False, collate_fn=dict_collate)

    pinn_core = VanillaPINN(in_dim=1, out_dim=1, hidden=(128, 128, 128, 128), activation="tanh")
    model = TrainerFriendlyVanilla(pinn_core).to(cfg.device)

    ckpt_path = os.path.join(cfg.out_dir, cfg.best_name)

    if TRAIN:
        pde_loss_fn = build_pde_loss(cfg)
        adapter = XAdapter(model)
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
    out_png = os.path.join(cfg.out_dir, f"{cfg.run_name}_n{cfg.n}_curve.png")
    plot_report(model, cfg, out_png)
    print("[report] saved:", out_png)


if __name__ == "__main__":
    main()
