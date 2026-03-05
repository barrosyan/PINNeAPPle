import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from pinneaple_geom.io.stl import load_stl
from pinneaple_geom.ops.repair import repair_mesh
from pinneaple_geom.ops.simplify import simplify_mesh
from pinneaple_geom.sample.points import sample_surface_points

from pinneaple_data.physical_sample import PhysicalSample
from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.losses import CombinedLoss, SupervisedLoss
from pinneaple_train.metrics import default_metrics

# ----------------------------
# 1) Load mesh (STL) -> repair -> simplify
# ----------------------------
stl_path = "examples/end_to_end/assets/Dragon 2.5_stl.stl"
if not os.path.exists(stl_path):
    raise FileNotFoundError("Place an STL at examples/end_to_end/assets/Dragon 2.5_stl.stl")

mesh = load_stl(stl_path, repair=True, compute_normals=True)
mesh = repair_mesh(mesh)
mesh = simplify_mesh(mesh, target_faces=5000)

# ----------------------------
# 2) Sample surface points
# ----------------------------
pts, normals, face_id = sample_surface_points(mesh, n=20000, return_normals=True, return_face_id=False)
pts = pts.astype(np.float32)
normals = (normals.astype(np.float32) if normals is not None else None)

# toy target: learn normals from xyz (proxy)
x = torch.from_numpy(pts)          # (N,3)
y = torch.from_numpy(normals)      # (N,3)

# ----------------------------
# 3) Wrap as PhysicalSample (domain=mesh)
# ----------------------------
sample = PhysicalSample(
    state={"x": x, "y": y},
    geometry={"type": "mesh", "path": stl_path},
    domain={"type": "mesh", "representation": "pointcloud"},
    schema={"task": "regress_normals"},
    provenance={"source": "stl", "n": int(x.shape[0])},
)

# Create a simple dataset of “many batches” from one sample (MVP)
class PointDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, chunk=512):
        self.x = x
        self.y = y
        self.chunk = int(chunk)
        self.n = x.shape[0]
    def __len__(self):
        return (self.n + self.chunk - 1) // self.chunk
    def __getitem__(self, idx):
        i0 = idx * self.chunk
        i1 = min(self.n, i0 + self.chunk)
        return {"x": self.x[i0:i1], "y": self.y[i0:i1], "provenance": sample.provenance}

dl = DataLoader(PointDataset(x, y), batch_size=None, shuffle=True)

# ----------------------------
# 4) Train a small MLP
# ----------------------------
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 3),
        )
    def forward(self, x):
        return self.net(x)

combined = CombinedLoss(supervised=SupervisedLoss("mse"), physics=None)
def loss_fn(model, y_hat, batch):
    return combined(model, y_hat, batch)

trainer = Trainer(model=M(), loss_fn=loss_fn, metrics=default_metrics())

out_dir = "examples/_out/end_to_end_02"
os.makedirs(out_dir, exist_ok=True)
cfg = TrainConfig(
    epochs=2,
    lr=1e-3,
    device="cpu",
    log_dir=os.path.join(out_dir, "_runs"),
    run_name="mesh_points",
    seed=0,
    deterministic=False,
)

out = trainer.fit(dl, dl, cfg)
print("best_val:", out["best_val"])
print("best_path:", out["best_path"])
