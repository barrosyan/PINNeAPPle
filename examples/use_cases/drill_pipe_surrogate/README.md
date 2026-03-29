# Drill Pipe Wear Surrogate Model with pinneaple + FEniCS

This use case shows how to build a surrogate model for drill pipe wear analysis
using pinneaple and FEniCS as the reference solver.

Two scenarios are covered:
1. **Both pin and box rotating** — torsional wear on both mating surfaces
2. **Pin rotating only, box fixed** — asymmetric wear (more realistic for make-up connections)

The output is a trained surrogate that predicts von Mises stress fields and
highlights high-wear regions, with visualizations showing wear intensity maps.

---

## Problem description

A drill pipe connection (pin + box) under combined:
- **Axial load** F_z (tension/compression)
- **Torque** T (torsion from rotation)
- **Bending moment** M (from wellbore curvature)
- **Contact pressure** at thread flanks (Hertzian contact)

The **von Mises stress** σ_vm = √(σ_xx² - σ_xx σ_yy + σ_yy² + 3 σ_xy²) is used
to identify wear-prone regions.  Wear rate is proportional to σ_vm × relative slip.

---

## Pipeline

```
Geometry (pinneaple_geom)
        ↓
FEniCS solver (pinneaple_solvers.FEnicsBridge)
  → stress fields (σ_ij) for multiple load cases
        ↓
Dataset (N load cases × M spatial points)
        ↓
PINN / DeepONet surrogate (pinneaple_models)
  → predicts σ_vm(r, z, F, T) at any point
        ↓
Wear visualization (pinneaple_inference)
  → von Mises map + wear zone highlighting
        ↓
Digital twin (pinneaple_digital_twin)
  → live monitoring of downhole loads
```

---

## Prerequisites

```bash
pip install pinneaple          # or: pip install -e .
pip install fenics              # or: conda install -c conda-forge fenics
# Optional for 3D geometry:
pip install gmsh pygmsh
pip install pyvista             # 3D visualization
```

---

## Step 1 — Define the problem preset

pinneaple ships a `drill_pipe_torsion` preset. Extend it for the two scenarios:

```python
from pinneaple_environment import get_preset

# Scenario A: both pin and box rotating (symmetric torsion)
spec_both = get_preset(
    "drill_pipe_torsion",
    E=210e9,
    nu=0.3,
    r_inner=0.038,          # m — drill pipe inner radius
    r_outer=0.044,          # m — outer radius
    torque=15000.0,         # N·m — applied torque
    axial_force=500e3,      # N — axial tension
    length=0.12,            # m — connection length
)

# Scenario B: pin rotates only (box fixed)
spec_pin = get_preset(
    "drill_pipe_torsion",
    E=210e9,
    nu=0.3,
    r_inner=0.038,
    r_outer=0.044,
    torque=15000.0,
    axial_force=500e3,
    length=0.12,
)
# Modify boundary condition: box face is fixed (no rotation)
from pinneaple_environment.conditions import DirichletBC
spec_pin.conditions["box_face"] = DirichletBC({"ux": 0.0, "uy": 0.0, "uz": 0.0})
```

---

## Step 2 — Run FEniCS solver for multiple load cases

```python
from pinneaple_solvers import FEnicsBridge, SolverOutput
import numpy as np

bridge = FEnicsBridge(
    mesh_nx=40,
    mesh_ny=40,
    element_degree=2,
    solver_backend="dolfinx",   # or "legacy"
)

# Parameter sweep: torque × axial load
torques      = np.linspace(5000, 25000, 8)   # N·m
axial_forces = np.linspace(100e3, 800e3, 8)  # N

dataset_both = []
dataset_pin  = []

for T in torques:
    for F in axial_forces:
        # Scenario A
        spec = get_preset("drill_pipe_torsion",
                          torque=T, axial_force=F, r_inner=0.038, r_outer=0.044)
        result_a: SolverOutput = bridge.forward(spec)
        dataset_both.append({
            "torque": T, "axial_force": F,
            "fields": result_a.extras.get("fields", {}),
            "von_mises": result_a.extras.get("von_mises"),
        })

        # Scenario B (pin only)
        spec_b = get_preset("drill_pipe_torsion",
                            torque=T, axial_force=F, r_inner=0.038, r_outer=0.044)
        spec_b.conditions["box_face"] = DirichletBC({"ux": 0.0, "uy": 0.0, "uz": 0.0})
        result_b: SolverOutput = bridge.forward(spec_b)
        dataset_pin.append({
            "torque": T, "axial_force": F,
            "fields": result_b.extras.get("fields", {}),
            "von_mises": result_b.extras.get("von_mises"),
        })

print(f"Generated {len(dataset_both)} FEniCS solutions per scenario")
```

---

## Step 3 — Build training dataset

```python
import pandas as pd
from pinneaple_data import CollocationSampler

# Flatten FEniCS solutions into (r, z, F, T) → σ_vm arrays
rows = []
for sample in dataset_both:
    vm = sample["von_mises"]    # (N_dof,) array from FEniCS
    coords = sample["fields"].get("coords")   # (N_dof, 3) → r, theta, z
    if vm is None or coords is None:
        continue
    for i in range(len(vm)):
        rows.append({
            "r":          coords[i, 0],
            "z":          coords[i, 2],
            "torque":     sample["torque"],
            "axial_force": sample["axial_force"],
            "von_mises":  vm[i],
        })

df = pd.DataFrame(rows)
print(f"Dataset: {len(df):,} samples  |  σ_vm range: [{df.von_mises.min():.2e}, {df.von_mises.max():.2e}]")

# Normalise
from sklearn.preprocessing import StandardScaler
X_cols = ["r", "z", "torque", "axial_force"]
y_col  = "von_mises"

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(df[X_cols].values).astype("float32")
y = scaler_y.fit_transform(df[[y_col]].values).astype("float32")

# Train/val split
N = len(X)
idx = np.random.default_rng(42).permutation(N)
n_train = int(0.85 * N)
X_train, X_val = X[idx[:n_train]], X[idx[n_train:]]
y_train, y_val = y[idx[:n_train]], y[idx[n_train:]]
```

---

## Step 4 — Train the surrogate (PINN or DeepONet)

### Option A — Vanilla PINN (data-driven, no physics loss)

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pinneaple_train import Trainer, TrainConfig, best_device, maybe_compile

DEVICE = best_device()

# Build model:  (r, z, T_load, F_axial) → σ_vm
model = nn.Sequential(
    nn.Linear(4, 128), nn.Tanh(),
    nn.Linear(128, 128), nn.Tanh(),
    nn.Linear(128, 64), nn.Tanh(),
    nn.Linear(64, 1),
).to(DEVICE)
model = maybe_compile(model, mode="default")

X_tr_t = torch.from_numpy(X_train).to(DEVICE)
y_tr_t = torch.from_numpy(y_train).to(DEVICE)
X_vl_t = torch.from_numpy(X_val).to(DEVICE)
y_vl_t = torch.from_numpy(y_val).to(DEVICE)

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=1024, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_vl_t, y_vl_t), batch_size=1024)

def mse_loss(m, y_hat, batch):
    x, y = batch["x"], batch["y"]
    return {"total": torch.mean((m(x) - y) ** 2)}

trainer = Trainer(model, loss_fn=mse_loss)
cfg     = TrainConfig(epochs=500, lr=1e-3, device=str(DEVICE), amp=True)
result  = trainer.fit(train_loader, val_loader, cfg)
print(f"Best val loss: {result['best_val']:.4e}")
```

### Option B — DeepONet (operator: load case → stress field)

```python
from pinneaple_models.neural_operators.deeponet import DeepONet

# Branch: encodes load parameters [T, F] → modes
# Trunk: encodes spatial coords (r, z) → basis functions
deeponet = DeepONet(
    branch_dim=2,      # [torque, axial_force]
    trunk_dim=2,       # [r, z]
    out_dim=1,         # σ_vm
    hidden=128,
    modes=64,
).to(DEVICE)
```

---

## Step 5 — Von Mises field prediction and wear visualization

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Predict σ_vm on a dense (r, z) grid for a specific load case
r_grid = np.linspace(0.038, 0.044, 80)
z_grid = np.linspace(0.0, 0.12, 100)
RR, ZZ = np.meshgrid(r_grid, z_grid)

# Query with specific torque + axial force
T_query = 18000.0    # N·m
F_query = 600e3      # N
X_query = np.column_stack([
    RR.ravel(),
    ZZ.ravel(),
    np.full(RR.size, T_query),
    np.full(RR.size, F_query),
]).astype("float32")

X_norm = scaler_X.transform(X_query)
with torch.no_grad():
    vm_pred_norm = model(torch.from_numpy(X_norm).to(DEVICE)).cpu().numpy()
vm_pred = scaler_y.inverse_transform(vm_pred_norm).reshape(RR.shape)

# Wear index ≈ σ_vm / σ_yield  (dimensionless; >0.8 = high risk)
sigma_yield = 550e6   # Pa — drill pipe grade S-135
wear_index  = vm_pred / sigma_yield

# ---- Plots ----
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Von Mises stress field
im1 = axes[0].pcolormesh(ZZ * 1000, RR * 1000, vm_pred / 1e6,
                          cmap="RdYlGn_r", shading="auto")
plt.colorbar(im1, ax=axes[0], label="σ_vm (MPa)")
axes[0].axhline(sigma_yield / 1e6, color="k", lw=0.8, linestyle="--")
axes[0].set_xlabel("z (mm)")
axes[0].set_ylabel("r (mm)")
axes[0].set_title(f"Von Mises Stress\nT={T_query/1000:.0f} kN·m  F={F_query/1e3:.0f} kN")

# 2. Wear index map
wear_cmap = mcolors.LinearSegmentedColormap.from_list(
    "wear", [(0,"green"), (0.5,"yellow"), (0.8,"orange"), (1.0,"red")]
)
im2 = axes[1].pcolormesh(ZZ * 1000, RR * 1000, wear_index,
                          cmap=wear_cmap, vmin=0, vmax=1.0, shading="auto")
plt.colorbar(im2, ax=axes[1], label="Wear index (σ_vm / σ_y)")
cs = axes[1].contour(ZZ * 1000, RR * 1000, wear_index,
                     levels=[0.6, 0.8, 1.0],
                     colors=["yellow", "orange", "red"], linewidths=1.5)
axes[1].clabel(cs, fmt="%.1f")
axes[1].set_xlabel("z (mm)")
axes[1].set_ylabel("r (mm)")
axes[1].set_title("Wear Index Map\n(contours at 0.6, 0.8, 1.0)")

# 3. High-wear zone highlight
high_wear = wear_index > 0.8
axes[2].pcolormesh(ZZ * 1000, RR * 1000,
                    np.where(high_wear, wear_index, np.nan),
                    cmap="Reds", vmin=0.8, vmax=1.2, shading="auto")
axes[2].pcolormesh(ZZ * 1000, RR * 1000,
                    np.where(~high_wear, 0.0, np.nan),
                    cmap="Greens_r", vmin=0, vmax=1, alpha=0.3, shading="auto")
axes[2].set_xlabel("z (mm)")
axes[2].set_ylabel("r (mm)")
axes[2].set_title("High-Risk Wear Zones (σ_vm > 0.8 σ_y)\nRed = critical")

plt.tight_layout()
plt.savefig("drill_pipe_wear_map.png", dpi=200)
print("Saved: drill_pipe_wear_map.png")
```

Expected output:

![Drill pipe wear map showing three panels: von Mises stress field, wear index heatmap with contours, and high-risk zone highlighting](drill_pipe_wear_example.png)

---

## Step 6 — Scenario comparison (both rotating vs pin only)

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
titles = ["Both pin + box rotating", "Pin rotating only (box fixed)"]

for ax, data, title in zip(axes, [dataset_both, dataset_pin], titles):
    # Find worst-case load (max σ_vm across all load cases)
    vm_max_per_case = [np.nanmax(d["von_mises"]) for d in data if d["von_mises"] is not None]
    torques_flat = [d["torque"] for d in data if d["von_mises"] is not None]
    forces_flat  = [d["axial_force"] for d in data if d["von_mises"] is not None]

    sc = ax.scatter(
        [t / 1000 for t in torques_flat],
        [f / 1e3  for f in forces_flat],
        c=vm_max_per_case,
        cmap="RdYlGn_r", s=80, edgecolors="k", linewidths=0.3
    )
    plt.colorbar(sc, ax=ax, label="Max σ_vm (Pa)")
    ax.set_xlabel("Torque (kN·m)")
    ax.set_ylabel("Axial force (kN)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("drill_pipe_scenario_comparison.png", dpi=200)
```

---

## Step 7 — Digital twin for real-time monitoring

```python
from pinneaple_digital_twin import build_digital_twin, MockStream, ThresholdDetector

# Wrap the trained surrogate as a digital twin
dt = build_digital_twin(
    model,
    field_names=["von_mises"],
    coord_names=["r", "z", "torque", "axial_force"],
    update_interval=1.0,
)

# Anomaly threshold: 80% of yield strength
dt.anomaly_monitor.add_detector(
    ThresholdDetector({"von_mises": 0.8 * sigma_yield})
)

# Real sensor streams (surface torque/WOB measurements)
def surface_torque_sensor(t: float) -> dict:
    """Simulates downhole torque from surface measurement (with tool-face correction)."""
    T_nominal = 15000.0
    T_noise   = 500.0 * np.sin(0.5 * t) + 200 * np.random.randn()
    return {"torque": T_nominal + T_noise}

dt.add_stream(MockStream(
    "surface_torque",
    ["torque"],
    surface_torque_sensor,
    tick_interval=0.5,
    coords={"r": 0.041, "z": 0.06},
))

dt.on_anomaly(lambda ev: print(f"[WEAR ALERT] σ_vm = {ev.observed:.2e} Pa > 80% yield!"))

# Run monitoring
with dt:
    time.sleep(10)

print(f"Monitoring complete. Alerts: {len(dt.anomaly_monitor.all_events)}")
```

---

## What needs to be implemented

The following components are **already in pinneaple** and work out-of-the-box:

| Component | Module | Status |
|-----------|--------|--------|
| `drill_pipe_torsion` preset | `pinneaple_environment.presets.structural` | ✅ Ready |
| FEniCS bridge | `pinneaple_solvers.FEnicsBridge` | ✅ Ready |
| DeepONet / VanillaPINN | `pinneaple_models` | ✅ Ready |
| Trainer + AMP + compile | `pinneaple_train` | ✅ Ready |
| Digital twin + anomaly | `pinneaple_digital_twin` | ✅ Ready |

The following steps require **your domain-specific implementation**:

| Step | What to do |
|------|-----------|
| Geometry | Define actual CAD geometry of pin + box threads → export as STEP or create parametrically with `gmsh` |
| FEniCS mesh | Convert geometry to FEniCS mesh (`dolfinx` supports `.msh`, `.xdmf`) |
| Contact mechanics | Implement Hertzian contact between thread flanks as a Robin BC in FEniCS |
| Rotation BCs | For "pin rotating only": add `DirichletBC` on pin face with `u = ω r e_θ`, box face fixed |
| Wear model | Post-process σ_vm × slip distance → wear depth using Archard's law |
| 3D visualization | Use `pyvista` to render the 3D wear map on the actual geometry |

---

## File structure for this use case

```
my_drill_pipe_study/
├── README.md                        ← this file
├── 01_run_fenics_sweep.py           ← parameter sweep with FEniCS
├── 02_train_surrogate.py            ← surrogate training
├── 03_wear_visualization.py         ← wear maps
├── 04_digital_twin_monitoring.py    ← real-time monitoring
├── configs/
│   └── drill_pipe_config.yaml       ← full pipeline YAML
└── data/
    ├── fenics_solutions/            ← FEniCS output .xdmf files
    ├── dataset.parquet              ← training dataset
    └── surrogate_model.pt           ← trained model checkpoint
```

---

## YAML config for automated pipeline

```yaml
# configs/drill_pipe_config.yaml
pipeline:
  name: drill_pipe_wear_surrogate
  out_dir: data/results

problem:
  id: drill_pipe_torsion
  params:
    E: 210.0e9
    nu: 0.3
    r_inner: 0.038
    r_outer: 0.044
    torque: 15000.0
    axial_force: 500.0e3

solver:
  backend: fenics
  params:
    mesh_nx: 40
    mesh_ny: 40
    element_degree: 2
    solver_backend: dolfinx

dataset:
  n_collocation: 50000
  strategy: lhs
  seed: 42
  param_sweep:
    torque: [5000, 10000, 15000, 20000, 25000]
    axial_force: [100000, 300000, 500000, 700000]

models:
  - id: pinn_drill_pipe
    type: VanillaPINN
    params:
      hidden: [128, 128, 128, 128]
      activation: tanh
    train:
      epochs: 3000
      lr: 0.001
      device: cuda
      amp: true
      grad_clip: 1.0

metrics: [mse, rmse, rel_l2, r2]

report:
  format: html
  save_model: true
  plots:
    - type: field_2d
      field: von_mises
    - type: error_map_2d
      field: von_mises
    - type: loss_curve
```

Run with:
```bash
python -m pinneaple_arena.runner.run_pipeline --config configs/drill_pipe_config.yaml
# or:
python examples/pinneaple_arena/09_full_pipeline_yaml.py --config configs/drill_pipe_config.yaml
```

---

## References

- Archard, J.F. (1953). "Contact and Rubbing of Flat Surfaces." Journal of Applied Physics.
- Bourgoyne et al. (1991). *Applied Drilling Engineering*. SPE Textbook Series.
- Leake & Mortari (2020). "Deep Theory of Functional Connections." arXiv:2005.01219
- pinneaple docs: `QUICKSTART.md`
