import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from pinneaple_geom.gen.cadquery_gen import build_mesh_from_cadquery_object
from pinneaple_geom.sample.points import sample_surface_points

from pinneaple_data.physical_sample import PhysicalSample
from pinneaple_data.zarr_store import UPDZarrStore
from pinneaple_data.zarr_iterable import ZarrUPDIterable
from pinneaple_data.collate import collate_upd_supervised

from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.losses import CombinedLoss, SupervisedLoss
from pinneaple_train.metrics import default_metrics


# ----------------------------
# 0) CADQuery availability
# ----------------------------
try:
    import cadquery as cq
except Exception as e:
    print("cadquery not available:", e)
    raise SystemExit(0)

out_dir = "examples/_out/end_to_end_04"
os.makedirs(out_dir, exist_ok=True)
zarr_path = os.path.join(out_dir, "cad_variants.zarr")

# ----------------------------
# 1) Generate CAD variants
# ----------------------------
def make_box(w: float, d: float, h: float):
    # centered box
    return cq.Workplane("XY").box(w, d, h)

variants = []
for w in [0.8, 1.0, 1.2]:
    for h in [0.2, 0.3]:
        variants.append((w, 1.0, h))

# ----------------------------
# 2) Convert to MeshData + sample points -> build UPD samples
#    Task: predict normals from xyz (simple supervised proxy)
# ----------------------------
samples = []
for (w, d, h) in variants:
    cq_obj = make_box(w, d, h)
    mesh = build_mesh_from_cadquery_object(cq_obj)  # MeshData

    pts, nrm, _ = sample_surface_points(
        mesh,
        n=8000,
        return_normals=True,
        return_face_id=False,
    )

    pts = pts.astype(np.float32)
    nrm = nrm.astype(np.float32)

    x = torch.from_numpy(pts)   # (N,3)
    y = torch.from_numpy(nrm)   # (N,3)

    samples.append(
        PhysicalSample(
            state={"x": x, "y": y},
            domain={"type": "mesh", "representation": "pointcloud"},
            provenance={"w": w, "d": d, "h": h, "source": "cadquery"},
            schema={"task": "xyz_to_normals"},
        )
    )


# ----------------------------
# 3) Write to Zarr
# ----------------------------
if not os.path.isdir(zarr_path):
    UPDZarrStore.write(zarr_path, samples, manifest={"name": "cad_variants"})


# ----------------------------
# 4) Stream back from Zarr + train
# ----------------------------
ds = ZarrUPDIterable(zarr_path, fields=["x", "y"], coords=[])
dl = DataLoader(
    ds,
    batch_size=8,
    num_workers=0,  # start simple; >0 after stable on Windows
    collate_fn=collate_upd_supervised,  # <-- FIX: {"x": (B,N,3), "y": (B,N,3)}
)

# Model expects x as (B,N,3)
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

combined = CombinedLoss(
    supervised=SupervisedLoss("mse"),
    physics=None,
    w_supervised=1.0,
    w_physics=0.0,
)

def loss_fn(model, y_hat, batch):
    return combined(model, y_hat, batch)

trainer = Trainer(model=M(), loss_fn=loss_fn, metrics=default_metrics())

cfg = TrainConfig(
    epochs=2,
    lr=1e-3,
    device="cpu",
    log_dir=os.path.join(out_dir, "_runs"),
    run_name="cad_xyz_normals",
    seed=42,
    deterministic=False,
    save_best=True,
)

out = trainer.fit(dl, dl, cfg)
print("best_val:", out["best_val"])
print("best_path:", out["best_path"])
