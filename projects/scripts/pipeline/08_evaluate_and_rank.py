import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch

from pinneaple_models.pinns.registry import PINNCatalog
from pinneaple_models.neural_operators.fno import FourierNeuralOperator
from pinneaple_train.checkpoint import load_checkpoint


PINN_MODELS = ["vanilla_pinn", "vpinn", "xpinn", "pinnsformer", "xtfc"]


def rmse(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def split_filter(df, split="test"):
    return df[df["split"] == split].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj", required=True)
    args = ap.parse_args()

    proj = Path(args.proj).resolve()
    runs = proj / "runs"
    rep = proj / "reports"
    rep.mkdir(parents=True, exist_ok=True)

    cloud = pd.read_parquet(proj / "datasets" / "sensors.parquet")
    line = pd.read_parquet(proj / "datasets" / "line_sensors.parquet")
    cloud_test = split_filter(cloud, "test")
    line_test = split_filter(line, "test")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []

    # ---- PINNs evaluated on cloud sensors (3D)
    # Note: For PINNs, we don't have exact saved weights naming here.
    # Trainer writes checkpoints under runs; simplest is to use the latest checkpoint by convention.
    # If you want exact filenames, fix TrainConfig.run_name and collect in a table.
    for name in PINN_MODELS:
        # You will likely have a checkpoint file in runs/pinn_<name>.ckpt (depending on Trainer checkpoint config).
        # Here we simply record placeholder; to make it strict, set ModelCheckpoint in training script.
        # We'll still provide inference logic if you load weights explicitly.
        rows.append({"model": name, "family": "pinn_3d", "RMSE_cloud": np.nan, "RMSE_line": np.nan, "note": "set checkpoint path in training script"})

    # ---- FNO/PINO evaluated on line (1D)
    # Rebuild operator architectures consistent with training script
    class OperatorWrapper(torch.nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op
        def forward(self, x):
            if x.ndim == 2:
                x = x.unsqueeze(0)
            out = self.op(x, y_true=None, return_loss=False)
            return out.y

    # Load checkpoints if you saved them. If not, you can keep Trainer checkpoints.
    # Here we evaluate by reloading the latest model state dict if you add saving.
    # For now, compute metrics is shown; plug loading accordingly.

    # Construct dataset for test scenarios
    # group by scenario: ensure consistent L
    L = line_test[line_test["scenario_id"] == line_test["scenario_id"].iloc[0]].shape[0]

    def make_u(Tin: float):
        u = np.full((1, L), Tin, dtype=np.float32)
        return torch.tensor(u, device=device)

    def eval_operator(tag: str, model: torch.nn.Module):
        errs = []
        for sid in sorted(line_test["scenario_id"].unique()):
            df = line_test[line_test["scenario_id"] == sid].sort_values("x")
            Tin = float(df["T"].iloc[0])  # not reliable; better read from conditions.json; keep simple
            # safer: infer Tin from boundary: since inlet is x=0 point on line
            Tin = float(df[df["x"] <= 1e-6]["T"].iloc[0]) if (df["x"] <= 1e-6).any() else float(df["T"].iloc[0])
            u = make_u(Tin)
            with torch.no_grad():
                pred = model(u).detach().cpu().numpy().reshape(-1)
            errs.append(rmse(pred, df["T"].to_numpy()))
        return float(np.mean(errs))

    # Placeholder results until you add explicit saving/loading
    rows.append({"model": "fno_1d", "family": "operator_1d", "RMSE_cloud": np.nan, "RMSE_line": np.nan, "note": "add checkpoint save/load in train_operator script"})
    rows.append({"model": "pino_1d", "family": "operator_1d", "RMSE_cloud": np.nan, "RMSE_line": np.nan, "note": "add checkpoint save/load in train_operator script"})

    df = pd.DataFrame(rows)
    df.to_csv(rep / "leaderboard.csv", index=False)
    print(df)
    print(f"Wrote {rep/'leaderboard.csv'}")

if __name__ == "__main__":
    main()