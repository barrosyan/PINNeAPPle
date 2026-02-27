import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from pinneaple_environment.spec import ProblemSpec
from pinneaple_environment.pdes import PDEspec
from pinneaple_environment.conditions import ConditionSpec
from pinneaple_pinn.compiler.compile import compile_problem
from pinneaple_pinn.compiler.loss import LossWeights
from pinneaple_models.pinns.registry import PINNCatalog
from pinneaple_train.trainer import Trainer, TrainConfig


PINN_MODELS = ["vanilla_pinn", "vpinn", "xpinn", "pinnsformer", "xtfc"]


def _mask_inlet(X: np.ndarray, ctx: dict) -> np.ndarray:
    return (X[:, 0] <= 1e-3)

def _mask_outlet(X: np.ndarray, ctx: dict) -> np.ndarray:
    return (X[:, 0] >= 4.0 - 1e-3)

def _mask_obstacle(X: np.ndarray, ctx: dict) -> np.ndarray:
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    r = np.sqrt((y - 0.5) ** 2 + (z - 0.5) ** 2)
    return (x >= 1.4 - 2e-2) & (x <= 2.6 + 2e-2) & (np.abs(r - 0.15) <= 2.5e-2)

def _mask_walls(X: np.ndarray, ctx: dict) -> np.ndarray:
    # boundary that is not inlet/outlet/obstacle
    inlet = _mask_inlet(X, ctx)
    outlet = _mask_outlet(X, ctx)
    obs = _mask_obstacle(X, ctx)
    return ~(inlet | outlet | obs)

def _val_inlet(X: np.ndarray, ctx: dict) -> np.ndarray:
    Tin = float(ctx["T_inlet"])
    return np.full((X.shape[0], 1), Tin, dtype=np.float32)

def _val_outlet(X: np.ndarray, ctx: dict) -> np.ndarray:
    Tout = float(ctx.get("T_outlet", 0.0))
    return np.full((X.shape[0], 1), Tout, dtype=np.float32)

def _val_zero(X: np.ndarray, ctx: dict) -> np.ndarray:
    return np.zeros((X.shape[0], 1), dtype=np.float32)


class PINNScenarioDataset(Dataset):
    """
    Each item is a full-batch dict for one scenario (multi-scenario training).
    """
    def __init__(self, x_col, x_bc, n_bc, scenarios, device="cpu"):
        self.x_col = x_col
        self.x_bc = x_bc
        self.n_bc = n_bc
        self.scenarios = scenarios
        self.device = device

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, i):
        sc = self.scenarios[i]
        ctx = {
            "T_inlet": float(sc["T_inlet"]),
            "T_outlet": float(sc.get("T_outlet", 0.0)),
            "k": float(sc.get("k", 1.0)),
        }

        # y_bc values filled only for Dirichlet masks; Neumann uses n_bc + mask
        Xb = self.x_bc.numpy()
        y_bc = np.full((Xb.shape[0], 1), np.nan, dtype=np.float32)

        inlet = _mask_inlet(Xb, ctx)
        outlet = _mask_outlet(Xb, ctx)

        y_bc[inlet, 0] = ctx["T_inlet"]
        y_bc[outlet, 0] = ctx["T_outlet"]

        batch = {
            "x_col": self.x_col.to(self.device),
            "x_bc": self.x_bc.to(self.device),
            "y_bc": torch.tensor(y_bc, device=self.device),
            "n_bc": self.n_bc.to(self.device),
            "ctx": ctx,
        }
        return batch


def collate_keep_dict(batch_list):
    # dataset returns dict already, and each item is a full scenario batch
    return batch_list[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    ap.add_argument("--ddp", action="store_true")
    args = ap.parse_args()

    proj = Path(args.proj).resolve()
    derived = proj / "derived"
    specs = proj / "specs"
    runs = proj / "runs"
    runs.mkdir(parents=True, exist_ok=True)

    # points
    x_col = torch.tensor(pd.read_parquet(derived / "points_collocation.parquet")[["x","y","z"]].to_numpy(), dtype=torch.float32)
    bc_df = pd.read_parquet(derived / "points_boundary.parquet")
    x_bc = torch.tensor(bc_df[["x","y","z"]].to_numpy(), dtype=torch.float32)
    n_bc = torch.tensor(bc_df[["nx","ny","nz"]].to_numpy(), dtype=torch.float32)

    # scenarios
    conditions = json.loads((specs / "conditions.json").read_text())
    scenarios = conditions["scenarios"]
    train_scen = [s for s in scenarios if int(s["scenario_id"]) < 14]
    val_scen = [s for s in scenarios if 14 <= int(s["scenario_id"]) < 17]

    # ProblemSpec (Laplace + BCs)
    spec = ProblemSpec(
        coords=("x","y","z"),
        fields=("T",),
        pde=PDEspec(kind="laplace", params={}),
        conditions=(
            ConditionSpec(name="inlet", kind="dirichlet", fields=("T",), mask=_mask_inlet, values=_val_inlet),
            ConditionSpec(name="outlet", kind="dirichlet", fields=("T",), mask=_mask_outlet, values=_val_outlet),
            ConditionSpec(name="walls", kind="neumann", fields=("T",), mask=_mask_walls, values=_val_zero),
            ConditionSpec(name="obstacle", kind="neumann", fields=("T",), mask=_mask_obstacle, values=_val_zero),
        ),
    )

    loss_fn_compiled = compile_problem(spec, weights=LossWeights(pde=1.0, bc=50.0, data=0.0, ic=0.0))

    def loss_fn(model, y_hat, batch):
        # compiled loss expects full batch dict
        return loss_fn_compiled(model, y_hat, batch)

    train_ds = PINNScenarioDataset(x_col, x_bc, n_bc, train_scen, device="cuda" if torch.cuda.is_available() else "cpu")
    val_ds = PINNScenarioDataset(x_col, x_bc, n_bc, val_scen, device="cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_keep_dict)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_keep_dict)

    for name in PINN_MODELS:
        model = PINNCatalog().build(name, in_dim=3, out_dim=1, hidden=128, depth=6)
        cfg = TrainConfig(
            epochs=200,
            lr=1e-3,
            ddp=bool(args.ddp),
            log_dir=str(runs),
            run_name=f"pinn_{name}",
            amp=False,
        )
        trainer = Trainer(model=model, loss_fn=loss_fn)
        trainer.fit(train_loader, val_loader, cfg)
        print(f"[DONE] {name}")

if __name__ == "__main__":
    main()