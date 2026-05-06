import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from pinneaple_physics.pinn_solver.factory.pinn_factory import PINNFactory, PINNProblemSpec
from pinneaple_neural.trainer.trainer import Trainer, TrainConfig
from pinneaple_neural.trainer.losses import CombinedLoss, SupervisedLoss, PhysicsLossHook
from pinneaple_neural.trainer.metrics import default_metrics

# ----------------------------
# 1) Define a toy PDE: u_t + u = 0  (solution ~ exp(-t))
# ----------------------------
spec = PINNProblemSpec(
    pde_residuals=["Derivative(u(t), t) + u(t)"],
    conditions=[],  # could add IC: u(0)=1 using condition batches
    independent_vars=["t"],
    dependent_vars=["u"],
    inverse_params=[],
    loss_weights={"pde": 1.0},
    verbose=False,
)
pinn_factory = PINNFactory(spec)
physics_loss_fn = pinn_factory.generate_loss_function()  # (model, batch)->(loss, comps)

# ----------------------------
# 2) Build supervised data
# ----------------------------
t = torch.linspace(0, 1, 512).unsqueeze(1)
u_true = torch.exp(-t)

train = DataLoader(TensorDataset(t[:400], u_true[:400]), batch_size=64, shuffle=True)
val = DataLoader(TensorDataset(t[400:], u_true[400:]), batch_size=128)

# Trainer expects batches as dict or (x,y). Here x=t, y=u
# Our model will accept x tensor, but PINNFactory expects collocation tuple (t,)
# We'll place both in batch for CombinedLoss+PhysicsLossHook.

class WrapDataset(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, i):
        x, y = self.base[i]
        return {"x": x, "y": y, "collocation": (x,)}

train2 = DataLoader(WrapDataset(TensorDataset(t[:400], u_true[:400])), batch_size=64, shuffle=True)
val2 = DataLoader(WrapDataset(TensorDataset(t[400:], u_true[400:])), batch_size=128)

# ----------------------------
# 3) Model
# ----------------------------
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

model = M()

# ----------------------------
# 4) Loss = supervised + physics hook
# ----------------------------
combined = CombinedLoss(
    supervised=SupervisedLoss("mse"),
    physics=PhysicsLossHook(physics_loss_fn),
    w_supervised=1.0,
    w_physics=1.0,
)

def loss_fn(model, y_hat, batch):
    return combined(model, y_hat, batch)

trainer = Trainer(model=model, loss_fn=loss_fn, metrics=default_metrics())

out_dir = "examples/_out/end_to_end_03"
os.makedirs(out_dir, exist_ok=True)

cfg = TrainConfig(
    epochs=3,
    lr=1e-3,
    device="cpu",
    log_dir=os.path.join(out_dir, "_runs"),
    run_name="pinn_ut_plus_u",
    seed=7,
)

out = trainer.fit(train2, val2, cfg)
print("best_val:", out["best_val"])
print("best_path:", out["best_path"])
