import inspect
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from pinneaple_train.trainer import Trainer, TrainConfig
from pinneaple_train.losses import CombinedLoss, SupervisedLoss
from pinneaple_train.metrics import default_metrics
from pinneaple_train.checkpoint import load_checkpoint

from pinneaple_models.register_all import register_all
from pinneaple_models.registry import ModelRegistry

# ----------------------------
# Helpers
# ----------------------------
def unwrap_pred(y_hat, *, batch=None):
    """
    Convert AEOutput / ModelOutput / etc -> torch.Tensor

    Preference:
      - AEOutput: choose z if it matches y, else x_hat if it matches y, else x_hat
      - otherwise: try attributes in common order
      - fallback: return y_hat if it's already a tensor
    """
    if isinstance(y_hat, torch.Tensor):
        return y_hat

    # AEOutput-like
    if hasattr(y_hat, "x_hat") and hasattr(y_hat, "z"):
        x_hat = y_hat.x_hat
        z = y_hat.z

        y = None
        if isinstance(batch, dict):
            y = batch.get("y")

        if isinstance(y, torch.Tensor):
            if z.shape == y.shape:
                return z
            if x_hat.shape == y.shape:
                return x_hat

            # softer match: last dim
            if z.ndim == y.ndim and z.shape[-1] == y.shape[-1]:
                return z
            if x_hat.ndim == y.ndim and x_hat.shape[-1] == y.shape[-1]:
                return x_hat

        return x_hat  # default

    # Generic model outputs
    for attr in ("y", "pred", "logits", "x_hat", "recon", "z"):
        if hasattr(y_hat, attr):
            v = getattr(y_hat, attr)
            if isinstance(v, torch.Tensor):
                return v

    raise TypeError(f"Cannot unwrap prediction from type: {type(y_hat)}")

def _with_default(d: dict, k: str, v):
    if k in d:
        return d
    out = dict(d)
    out[k] = v
    return out

def build_model(name: str, **kwargs):
    """
    Tries common aliases, but never overwrites user-provided kwargs.
    """
    attempts = [
        dict(kwargs),
        _with_default(kwargs, "output_dim", 2),
        _with_default(kwargs, "out_dim", 2),
        _with_default(kwargs, "target_dim", 2),
        _with_default(kwargs, "latent_dim", 2),
        _with_default(kwargs, "input_dim", 8),
        _with_default(kwargs, "in_dim", 8),
    ]

    last_err = None
    for attempt in attempts:
        try:
            return ModelRegistry.build(name, **attempt)
        except TypeError as e:
            last_err = e
            continue

    raise last_err

# ----------------------------
# 1) Register models
# ----------------------------
register_all()

names = ModelRegistry.list()
print("Total registered:", len(names))
print("Some:", names[:20])

model_name = names[0] if names else None
if model_name is None:
    raise RuntimeError("No models registered. Ensure registries import and decorators execute.")

spec = ModelRegistry.spec(model_name)
print(spec.cls.__name__, inspect.signature(spec.cls.__init__))

model = build_model(model_name, input_dim=8, latent_dim=8)

# ----------------------------
# 2) Data
# ----------------------------
x = torch.randn(1024, 8)
y = torch.randn(1024, 8)

train = DataLoader(TensorDataset(x[:800], y[:800]), batch_size=64, shuffle=True)
val = DataLoader(TensorDataset(x[800:], y[800:]), batch_size=128)

# ----------------------------
# 3) Decide training mode (AE vs Regression)
# ----------------------------
model.eval()
with torch.no_grad():
    out_raw = model(x[:4])
    # Try to infer:
    # - If model returns AEOutput -> treat as AE by default
    # - Else if output last dim matches y -> regression
    # - Else if output last dim matches x -> AE-like reconstruction
    is_ae_output = hasattr(out_raw, "x_hat") and hasattr(out_raw, "z")

    if is_ae_output:
        mode = "AE"
    else:
        out_t = unwrap_pred(out_raw, batch={"y": y[:4]})
        if out_t.ndim >= 2 and out_t.shape[-1] == y.shape[-1]:
            mode = "REG"
        elif out_t.ndim >= 2 and out_t.shape[-1] == x.shape[-1]:
            mode = "AE"
        else:
            raise RuntimeError(
                f"Model '{model_name}' output shape {tuple(out_t.shape)} does not match "
                f"x dim {x.shape[-1]} or y dim {y.shape[-1]}."
            )

print("Selected mode:", mode)

# ----------------------------
# 4) Loss + Trainer
# ----------------------------
combined = CombinedLoss(supervised=SupervisedLoss("mse"), physics=None)

def loss_fn(model_, y_hat_raw, batch):
    # batch comes as {"x":..., "y":...} from Trainer._xy_batch
    b = dict(batch)

    # If AE mode: supervise with y <- x (reconstruction)
    if mode == "AE":
        b["y"] = b["x"]

    # unwrap pred using current batch (important for AEOutput: choose z vs x_hat)
    pred = unwrap_pred(y_hat_raw, batch=b)

    return combined(model_, pred, b)

trainer = Trainer(model=model, loss_fn=loss_fn, metrics=default_metrics())

out_dir = "examples/_out/end_to_end_05"
os.makedirs(out_dir, exist_ok=True)

cfg = TrainConfig(
    epochs=2,
    lr=1e-3,
    device="cpu",
    log_dir=os.path.join(out_dir, "_runs"),
    run_name=f"registry_{model_name}",
    seed=123,
    deterministic=False,
    save_best=True,
)

out = trainer.fit(train, val, cfg)
best_path = out["best_path"]
print("best_val:", out["best_val"])
print("best_path:", best_path)


# ----------------------------
# 5) Inference (load best)
# ----------------------------
try:
    ckpt = load_checkpoint(best_path)
    model.load_state_dict(ckpt.model_state)
except Exception as e:
    print("load_checkpoint not available, fallback torch.load:", e)
    raw = torch.load(best_path, map_location="cpu")
    if isinstance(raw, dict) and "model_state" in raw:
        model.load_state_dict(raw["model_state"])
    elif isinstance(raw, dict):
        model.load_state_dict(raw)

model.eval()
with torch.no_grad():
    pred5 = unwrap_pred(model(x[:5]), batch={"x": x[:5], "y": y[:5]})
print("pred shape:", tuple(pred5.shape))
print("mode:", "AE (y=x)" if mode == "AE" else "Regression (y=given)")
