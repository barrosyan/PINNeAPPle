import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from pinneaple_models.neural_operators.fno import FourierNeuralOperator
from pinneaple_models.neural_operators.pino import pino_physics_fn, GridSpec, ResidualSpec
from pinneaple_train.trainer import Trainer, TrainConfig


class LineOperatorDataset(Dataset):
    """
    Builds (u, y) for operator learning:
      u: (C_in, L) where C_in=1 channel = Tinlet repeated over x-grid
      y: (C_out, L) where C_out=1 channel = T(x) sampled from OpenFOAM
    """
    def __init__(self, line_sensors: pd.DataFrame, scenarios: list[dict], L: int):
        self.line = line_sensors
        self.scenarios = scenarios
        self.L = int(L)

        # group per scenario
        self.grp = {int(s["scenario_id"]): self.line[self.line["scenario_id"] == int(s["scenario_id"])].sort_values("x") for s in scenarios}

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, i):
        sc = self.scenarios[i]
        sid = int(sc["scenario_id"])
        Tin = float(sc["T_inlet"])

        df = self.grp[sid]
        # ensure L points
        if df.shape[0] != self.L:
            raise ValueError(f"Scenario {sid} has {df.shape[0]} line points but expected L={self.L}")

        y = df["T"].to_numpy(dtype=np.float32)[None, :]      # (1,L)
        u = np.full((1, self.L), Tin, dtype=np.float32)      # (1,L)

        return {
            "x": torch.tensor(u),    # Trainer uses batch['x']
            "y": torch.tensor(y),
            "ctx": {"T_inlet": Tin, "scenario_id": sid},
        }


def collate_first(batch_list):
    return batch_list[0]


def residual_laplace_1d(u: torch.Tensor, deriv: dict, params: dict, **kw) -> torch.Tensor:
    # laplace in 1D => u_xx = 0
    u_xx = deriv.get("d2", None)
    if u_xx is None:
        raise KeyError("Expected second derivative key 'd2' in deriv dict")
    return u_xx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    args = ap.parse_args()

    proj = Path(args.proj).resolve()
    specs = json.loads((proj / "specs" / "conditions.json").read_text())
    scenarios = specs["scenarios"]

    line = pd.read_parquet(proj / "datasets" / "line_sensors.parquet")
    L = pd.read_parquet(proj / "derived" / "line_points.parquet").shape[0]

    # split same as before
    train_scen = [s for s in scenarios if int(s["scenario_id"]) < 14]
    val_scen = [s for s in scenarios if 14 <= int(s["scenario_id"]) < 17]

    train_ds = LineOperatorDataset(line, train_scen, L=L)
    val_ds = LineOperatorDataset(line, val_scen, L=L)

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=collate_first)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate_first)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- FNO (supervised)
    fno = FourierNeuralOperator(in_channels=1, out_channels=1, width=64, modes=16, layers=4, use_grid=True).to(device)

    def loss_fno(model, y_hat, batch):
        y = batch["y"].to(device)
        # y_hat from Trainer is tensor shape (B? or 1?) - model expects (B,C,L)
        # Trainer passes x of shape (1,L) => model expects (B,C,L), so dataset returns (1,L) but we want (B,C,L)
        # We'll make sure batch["x"] is (1,L) and add batch dim inside model via unsqueeze in a wrapper.
        # To keep minimal, dataset returns (1,L) and Trainer sets x with ndim=2 -> ok; but model expects 3D.
        # We'll handle it here by calling model directly with correct shape.
        raise RuntimeError("Use OperatorWrapper below")

    # Operator wrapper to adapt Trainer's x=(C,L) into (B,C,L)
    class OperatorWrapper(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op
        def forward(self, x):
            if x.ndim == 2:      # (C,L)
                x = x.unsqueeze(0)
            out = self.op(x, y_true=None, return_loss=False)
            return out.y  # (B,C,L)

    fno_model = OperatorWrapper(fno).to(device)

    def loss_supervised(model, y_hat, batch):
        y = batch["y"].to(device)
        if y.ndim == 2:
            y = y.unsqueeze(0)
        return {"total": torch.mean((y_hat - y) ** 2)}

    cfg = TrainConfig(epochs=800, lr=2e-3, log_dir=str(proj / "runs"), run_name="op_fno_1d", amp=False)
    Trainer(fno_model, loss_fn=loss_supervised).fit(train_loader, val_loader, cfg)

    # ---- PINO (supervised + physics residual u_xx=0)
    # We'll reuse the same FNO as the operator backbone.
    pino_backbone = FourierNeuralOperator(in_channels=1, out_channels=1, width=64, modes=16, layers=4, use_grid=True).to(device)
    pino_model = OperatorWrapper(pino_backbone).to(device)

    grid = GridSpec(L=(4.0,), dims=(-1,))  # 1D, physical length 4.0 on x
    spec_res = ResidualSpec(residual=residual_laplace_1d, orders=(2,), method="spectral")

    def loss_pino(model, y_hat, batch):
        y = batch["y"].to(device)
        if y.ndim == 2:
            y = y.unsqueeze(0)

        sup = torch.mean((y_hat - y) ** 2)

        # PINO physics on predicted u: compute residual u_xx and minimize it
        # pino_physics_fn expects u shaped (B,C,L) => ok
        phys = pino_physics_fn(
            y_hat,
            grid=grid,
            spec=spec_res,
            params={},
            bc_mask=None,
            bc_value=None,
        )
        pde = phys["pde"]
        total = sup + 0.1 * pde
        return {"total": total, "sup": sup, "pde": pde}

    cfg2 = TrainConfig(epochs=800, lr=2e-3, log_dir=str(proj / "runs"), run_name="op_pino_1d", amp=False)
    Trainer(pino_model, loss_fn=loss_pino).fit(train_loader, val_loader, cfg2)

    print("[DONE] Trained FNO/PINO (1D multi-scenario)")

if __name__ == "__main__":
    main()