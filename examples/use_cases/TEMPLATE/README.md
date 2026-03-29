# pinneaple Use Case Template

> **This is the standard scaffold for new use cases.**
> Copy this directory, rename it, and work through each section top to bottom.
> Every `{{ PLACEHOLDER }}` must be replaced. Every commented-out code block
> shows exactly what to fill in for your domain.

---

## Pre-flight Checklist

Work through this before writing a single line of code.

```
[ ] I know what physical quantity is the output (temperature, stress, velocity, ...)
[ ] I know the governing PDE (or ODE) and its coefficients
[ ] I know the domain shape and dimension (1D interval, 2D rectangle, 3D volume, ...)
[ ] I know the boundary conditions (Dirichlet / Neumann / Robin / periodic)
[ ] I know whether I have any reference data (solver output, experiments, sensors)
[ ] I know my target accuracy and latency (offline analysis vs real-time twin)
[ ] I know which solver will generate reference data (builtin FDM | OpenFOAM | FEniCS)
[ ] I have decided on a model family (see "Model Selection" section below)
[ ] I have at least one GPU available, or have budgeted extra CPU training time
[ ] My dependencies are installed (see "Prerequisites" section)
```

---

## Complexity Ladder

Pick the row that matches your situation and follow only those steps.

| Level | Situation | What to build | Est. time |
|-------|-----------|---------------|-----------|
| **Minimal** | Exploratory, 1D ODE, no reference data | PINN or XtFC, collocation only | 1-2 h |
| **Standard** | 2D PDE, built-in solver reference, offline analysis | PINN + metrics + plots | 4-8 h |
| **Extended** | 2D/3D, external solver (OpenFOAM/FEniCS), param sweep | DeepONet/FNO + surrogate dataset | 1-3 days |
| **Production** | Parametric PDE, real sensor data, real-time monitoring | Full pipeline + digital twin | 1-2 weeks |

---

## Model Selection Guide

Choose based on your problem structure, not preference.

| Problem type | Recommended model | Registry key | Notes |
|-------------|-------------------|--------------|-------|
| 1D ODE, exact BC enforcement needed | **XtFC** | `xtfc` | One-shot ELM solve available |
| 1D/2D PDE, smooth solution, physics residual loss | **VanillaPINN** | `vanilla_pinn` | Start here for new problems |
| Variational form available (weak formulation) | **VPINN** | `vpinn` | Better for shock-like solutions |
| Multi-domain decomposition | **XPINN** | `xpinn` | Domain boundaries become interfaces |
| Inverse problem (identify PDE coefficients) | **InversePINN** | `inverse_pinn` | Unknown params become trainable |
| Spatiotemporal sequence, long time horizons | **PINN-LSTM** | `pinn_lstm` | Adds recurrence to PINN trunk |
| Parametric PDE — learn the solution operator | **DeepONet** | `deeponet` | Branch=params, Trunk=coords |
| Parametric PDE on regular grid, fast inference | **FNO** | `fno` | Needs grid-structured data |
| Parametric PDE with physical constraints enforced | **PINO** | `pino` | FNO + physics residual loss |
| Irregular mesh, geometry-varying domain | **GNO** | `gno` | Graph-based, mesh-agnostic |
| Multi-scale phenomena | **MultiScaleDeepONet** | `multiscale_deeponet` | Hierarchical branch/trunk |
| Time-series / dynamic state, data-driven | **NeuralODE** | (continuous family) | Latent ODE for temporal data |
| Reduced-order model from snapshots | **POD / DMD** | (rom family) | Classical ROM + DL hybrid |

**Rule of thumb:**
- Have a PDE and no data? -> VanillaPINN or XtFC
- Have a PDE and sparse data? -> VanillaPINN with `data` weight > 0
- Have a large dataset of (param, field) pairs? -> DeepONet or FNO
- Need sub-millisecond inference from a surrogate? -> FNO on a fixed grid
- Need to identify unknown PDE parameters from data? -> InversePINN

---

## Problem Statement

**Title:** `{{ DESCRIPTIVE_TITLE }}`

**Domain:** `{{ FIELD_OF_ENGINEERING_OR_SCIENCE }}`

**Physical quantity of interest:** `{{ FIELD_NAME_e.g._temperature_stress_velocity }}`

**Why a surrogate / PINN?**
- `{{ REASON_1 }}`  *(e.g., reference solver takes 2 h per run — need 10,000 evaluations)*
- `{{ REASON_2 }}`  *(e.g., real-time monitoring requires < 1 ms inference)*

**Inputs:** `{{ LIST_OF_INPUTS }}`  *(e.g., spatial coords (x,y), time t, load parameters)*

**Outputs:** `{{ LIST_OF_OUTPUTS }}`  *(e.g., u(x,y,t), stress tensor σ_ij(x,y))*

**Pipeline at a glance:**

```
{{ GEOMETRY_OR_DOMAIN }}
        |
{{ REFERENCE_SOLVER }}   <- generate dataset / reference solution
        |
{{ DATASET_DESCRIPTION }} (N samples x M spatial points)
        |
{{ MODEL_TYPE }} surrogate
  -> predicts {{ OUTPUT_FIELD }}({{ INPUT_COORDS }})
        |
{{ VISUALIZATION_TYPE }}
        |
{{ DIGITAL_TWIN_OR_OFFLINE_ANALYSIS }}
```

---

## Prerequisites

```bash
# Core (always required)
pip install pinneaple            # or: pip install -e .  from repo root

# Reference solver — uncomment the one you need
# pip install fenics              # FEniCS legacy
# conda install -c conda-forge fenics-dolfinx  # FEniCS dolfinx (preferred)
# # OpenFOAM: install via system package manager, then source $WM_PROJECT_DIR/etc/bashrc

# Geometry / mesh (for 3D or complex domains)
# pip install gmsh pygmsh

# 3D visualization
# pip install pyvista

# Additional analysis
# pip install scikit-learn        # for StandardScaler, PCA, etc.
# pip install pandas pyarrow      # for parquet dataset storage
```

---

## Step 1 — Define the Problem Spec

pinneaple's `pinneaple_environment` module holds problem presets. Either use an
existing preset and override parameters, or define a custom spec from scratch.

### Option A — Use a registered preset

```python
from pinneaple_environment import get_preset, list_presets

# List all available presets:
print(list_presets())

# Load with parameter overrides:
spec = get_preset(
    "{{ PRESET_NAME }}",       # e.g. "burgers_1d", "heat_2d", "ns_incompressible_2d",
                               #      "drill_pipe_torsion", "cpu_heatsink_thermal"
    # -- override defaults below --
    # nu=0.01,
    # Re=100.0,
    # E=210e9,
)
print(f"PDE kind   : {spec.pde.kind}")
print(f"Fields     : {spec.fields}")
print(f"Coord names: {spec.coord_names}")
print(f"Domain     : {spec.domain_bounds}")
```

### Option B — Build a custom spec

```python
from pinneaple_environment.spec import ProblemSpec, PDESpec
from pinneaple_environment.conditions import DirichletBC, NeumannBC, PeriodicBC

spec = ProblemSpec(
    problem_id="{{ YOUR_PROBLEM_ID }}",
    pde=PDESpec(
        kind="{{ PDE_KIND }}",          # e.g. "heat", "wave", "poisson", "navier_stokes",
                                        #      "elasticity", "advection_diffusion"
        params={
            "{{ PARAM_1 }}": {{ VALUE_1 }},   # e.g. "nu": 0.01
            "{{ PARAM_2 }}": {{ VALUE_2 }},   # e.g. "k": 1.0  (thermal conductivity)
        },
        order={{ ORDER }},              # 1 (first-order) or 2 (second-order PDE)
    ),
    fields=["{{ FIELD_1 }}", "{{ FIELD_2 }}"],   # e.g. ["u"] or ["u", "v", "p"]
    coord_names=["{{ COORD_1 }}", "{{ COORD_2 }}"],  # e.g. ["x", "t"] or ["x", "y", "t"]
    domain_bounds={
        "{{ COORD_1 }}": [{{ X_MIN }}, {{ X_MAX }}],   # e.g. "x": [-1.0, 1.0]
        "{{ COORD_2 }}": [{{ T_MIN }}, {{ T_MAX }}],   # e.g. "t": [0.0, 1.0]
    },
    conditions={
        "{{ BC_NAME_1 }}": DirichletBC({"{{ FIELD }}": {{ VALUE }}}),
            # e.g. "left": DirichletBC({"u": 0.0})
        "{{ BC_NAME_2 }}": NeumannBC({"{{ FIELD }}": {{ FLUX }}}),
            # e.g. "top": NeumannBC({"u": 0.0})  # zero-flux / insulated
        # "{{ BC_NAME_3 }}": PeriodicBC(),
    },
    solver_spec={"name": "{{ SOLVER_NAME }}", "solver": "{{ SOLVER_METHOD }}"},
    meta={"description": "{{ BRIEF_DESCRIPTION }}"},
)
```

### Dimensional scaling note

Always non-dimensionalize before training. Typical choices:

```python
# 1D ODE: normalize x to [0, 1]
# 2D PDE: normalize (x, y) to [0, 1]^2 and t to [0, 1]
# Physical fields: normalize to unit variance or known reference scale

# Example: scale temperature T -> (T - T_ref) / delta_T
T_ref   = {{ T_REFERENCE }}    # e.g. 300.0  (K, ambient)
delta_T = {{ T_SCALE }}        # e.g. 100.0  (K, typical temperature rise)
```

---

## Step 2 — Geometry & Mesh

### 1D interval

```python
import numpy as np
# No explicit mesh needed — collocation sampler handles it.
# Domain: x in [{{ X_MIN }}, {{ X_MAX }}]
x_min, x_max = {{ X_MIN }}, {{ X_MAX }}
```

### 2D rectangle (most common)

```python
import numpy as np
# Domain: x in [{{ X_MIN }}, {{ X_MAX }}], y in [{{ Y_MIN }}, {{ Y_MAX }}]
x_lim = [{{ X_MIN }}, {{ X_MAX }}]
y_lim = [{{ Y_MIN }}, {{ Y_MAX }}]

# For structured grid reference data:
nx, ny = {{ NX }}, {{ NY }}
x_grid = np.linspace(*x_lim, nx, dtype=np.float32)
y_grid = np.linspace(*y_lim, ny, dtype=np.float32)
XX, YY = np.meshgrid(x_grid, y_grid)
```

### 2D complex geometry with gmsh (obstacle, curved boundary, etc.)

```python
import gmsh
gmsh.initialize()
gmsh.model.add("{{ GEOMETRY_NAME }}")

# Example: rectangle with circular hole
lc = {{ MESH_SIZE }}           # e.g. 0.05
gmsh.model.occ.addRectangle({{ X_MIN }}, {{ Y_MIN }}, 0,
                              {{ X_WIDTH }}, {{ Y_HEIGHT }})
# gmsh.model.occ.addDisk({{ OBSTACLE_X }}, {{ OBSTACLE_Y }}, 0,
#                         {{ OBSTACLE_R }}, {{ OBSTACLE_R }})
# gmsh.model.occ.cut([(2, 1)], [(2, 2)])
gmsh.model.occ.synchronize()
gmsh.option.setNumber("Mesh.MeshSizeMax", lc)
gmsh.model.mesh.generate(2)
gmsh.write("{{ GEOMETRY_NAME }}.msh")
gmsh.finalize()
```

### 3D structural geometry

```python
import gmsh
# For structural problems: export CAD geometry as STEP, import into gmsh
# gmsh.merge("{{ GEOMETRY_FILE }}.step")
# gmsh.model.mesh.generate(3)
# gmsh.write("{{ GEOMETRY_NAME }}.msh")

# Alternatively use pygmsh for scripted geometry:
# import pygmsh
# with pygmsh.occ.Geometry() as geom:
#     geom.add_box([{{ X0 }}, {{ Y0 }}, {{ Z0 }}],
#                  [{{ DX }},  {{ DY }},  {{ DZ }}])
#     mesh = geom.generate_mesh(dim=3)
```

---

## Step 3 — Solver Setup (Reference Data Generation)

Choose one backend. The outputs are always `SolverOutput` objects with `.fields`
(dict of numpy arrays) and `.extras` (problem-specific metadata).

### Backend A — Built-in FDM (simplest, 1D/2D)

```python
from pinneaple_solvers import SolverRegistry

solver = SolverRegistry.build(
    "builtin",
    method="{{ FDM_METHOD }}",     # e.g. "ftcs", "btcs", "crank_nicolson"
    nx={{ NX }},
    nt={{ NT }},
    dt={{ DT }},
)
result = solver.forward(spec)
# result.fields: dict[str, np.ndarray]  — e.g. {"u": (nt, nx)}
# result.coords: dict[str, np.ndarray]  — e.g. {"x": (nx,), "t": (nt,)}
```

### Backend B — OpenFOAM (3D CFD, turbulence, multiphase)

```python
from pinneaple_solvers import OpenFOAMBridge
from pinneaple_solvers.openfoam_bridge import generate_case, run_openfoam, openfoam_to_dataset

# 1. Generate case directory from spec
case_dir = generate_case(
    spec,
    case_dir="{{ CASE_DIR_PATH }}",
    mesh_cfg={
        "type": "blockMesh",        # or "snappyHexMesh" for complex geometry
        "nx": {{ NX }},
        "ny": {{ NY }},
        "nz": 1,                    # set > 1 for 3D
    },
)

# 2. Run solver
run_openfoam(
    case_dir,
    solver="{{ OF_SOLVER }}",       # e.g. "simpleFoam", "pisoFoam", "rhoSimpleFoam"
    n_cores={{ N_CORES }},
    n_iter={{ N_ITER }},
)

# 3. Extract fields
data = openfoam_to_dataset(
    case_dir,
    field_names=["{{ FIELD_1 }}", "{{ FIELD_2 }}"],  # e.g. ["U", "p"]
    time="{{ TIME_VALUE }}",        # e.g. "latest"
)
# data: dict[str, np.ndarray]
```

### Backend C — FEniCS / dolfinx (structural, heat, coupled physics)

```python
from pinneaple_solvers import FEnicsBridge, SolverOutput

bridge = FEnicsBridge(
    mesh_nx={{ MESH_NX }},         # e.g. 40
    mesh_ny={{ MESH_NY }},         # e.g. 40  (omit for 1D/3D)
    element_degree={{ ELEM_DEG }}, # e.g. 2  (quadratic Lagrange elements)
    solver_backend="{{ FENICS_BACKEND }}",  # "dolfinx" (preferred) or "legacy"
)
result: SolverOutput = bridge.forward(spec)
# result.extras["fields"]: dict[str, np.ndarray]
# result.extras.get("von_mises"):  np.ndarray  (structural problems)
# result.coords: coordinates of DOF nodes
```

### Parameter sweep (all backends)

```python
import numpy as np
import itertools

# Define sweep grid
param_1_values = np.linspace({{ P1_MIN }}, {{ P1_MAX }}, {{ N_P1 }})
param_2_values = np.linspace({{ P2_MIN }}, {{ P2_MAX }}, {{ N_P2 }})

dataset = []
for p1, p2 in itertools.product(param_1_values, param_2_values):
    spec_i = get_preset("{{ PRESET_NAME }}", {{ PARAM_1 }}=p1, {{ PARAM_2 }}=p2)
    result_i = solver.forward(spec_i)      # or bridge.forward(spec_i)
    dataset.append({
        "{{ PARAM_1 }}": p1,
        "{{ PARAM_2 }}": p2,
        "fields": result_i.fields,
    })

print(f"Generated {len(dataset)} solver runs")
```

---

## Step 4 — Dataset Generation

### For PINN (physics residual loss — no large dataset needed)

```python
import numpy as np
from pinneaple_environment.sampling import CollocationSampler

rng = np.random.default_rng({{ SEED }})   # e.g. 42

# Collocation points (interior — where PDE residual is enforced)
N_COL = {{ N_COLLOCATION }}   # e.g. 8000 for 2D, 50000 for 3D
X_col = rng.uniform(
    [{{ X_MIN }}, {{ Y_OR_T_MIN }}],
    [{{ X_MAX }}, {{ Y_OR_T_MAX }}],
    (N_COL, {{ N_INPUT_DIMS }}),
).astype(np.float32)

# Boundary condition points
N_BC = {{ N_BOUNDARY }}       # e.g. 500-2000
# Example: four sides of a 2D rectangle
t_bc = rng.uniform(0, 1, N_BC).astype(np.float32)
x_left  = np.column_stack([np.full(N_BC // 4, {{ X_MIN }}, np.float32), t_bc[:N_BC//4]])
x_right = np.column_stack([np.full(N_BC // 4, {{ X_MAX }}, np.float32), t_bc[N_BC//4:N_BC//2]])
x_bot   = np.column_stack([rng.uniform({{ X_MIN }}, {{ X_MAX }}, N_BC//4).astype(np.float32), np.zeros(N_BC//4, np.float32)])
x_top   = np.column_stack([rng.uniform({{ X_MIN }}, {{ X_MAX }}, N_BC//4).astype(np.float32), np.ones(N_BC//4,  np.float32)])
X_bc = np.vstack([x_left, x_right, x_bot, x_top])

# Initial condition points (time-dependent problems only)
N_IC = {{ N_INITIAL_CONDITION }}   # e.g. 1000; set 0 for steady-state
# x_ic = np.column_stack([rng.uniform({{ X_MIN }}, {{ X_MAX }}, N_IC).astype(np.float32),
#                          np.zeros(N_IC, np.float32)])   # t=0

# Reference data points (if solver output available)
N_DATA = {{ N_DATA_POINTS }}   # e.g. 200; set 0 if pure PINN (no reference data)
```

### For DeepONet / FNO (operator learning — large dataset required)

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Flatten solver sweeps into (params, coords) -> field value rows
rows = []
for sample in dataset:
    fields = sample["fields"]
    coords = fields.get("coords")       # (N_dof, n_dims)
    values = fields.get("{{ FIELD_NAME }}")  # (N_dof,)
    if coords is None or values is None:
        continue
    for i in range(len(values)):
        rows.append({
            "{{ COORD_1 }}": coords[i, 0],
            "{{ COORD_2 }}": coords[i, 1],     # omit for 1D
            "{{ PARAM_1 }}": sample["{{ PARAM_1 }}"],
            "{{ PARAM_2 }}": sample["{{ PARAM_2 }}"],
            "{{ FIELD_NAME }}": values[i],
        })

df = pd.DataFrame(rows)
print(f"Dataset: {len(df):,} samples")
print(f"Field range: [{df['{{ FIELD_NAME }}'].min():.3e}, {df['{{ FIELD_NAME }}'].max():.3e}]")

# Normalize — always normalize inputs and outputs before training
X_cols = ["{{ COORD_1 }}", "{{ COORD_2 }}", "{{ PARAM_1 }}", "{{ PARAM_2 }}"]
y_col  = "{{ FIELD_NAME }}"

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(df[X_cols].values).astype("float32")
y = scaler_y.fit_transform(df[[y_col]].values).astype("float32")

# Train / validation split
N = len(X)
idx      = np.random.default_rng({{ SEED }}).permutation(N)
n_train  = int({{ TRAIN_FRAC }} * N)   # e.g. 0.85
X_train, X_val = X[idx[:n_train]], X[idx[n_train:]]
y_train, y_val = y[idx[:n_train]], y[idx[n_train:]]
print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}")

# Save dataset for reuse
df.to_parquet("data/{{ USE_CASE_NAME }}/dataset.parquet", index=False)
```

---

## Step 5 — Model Selection & Construction

### VanillaPINN (baseline — start here)

```python
import torch
from pinneaple_models.pinns.vanilla import VanillaPINN
from pinneaple_train import best_device, maybe_compile

DEVICE = best_device()

model = VanillaPINN(
    in_dim={{ N_INPUT_DIMS }},           # e.g. 2 for (x,t) or 3 for (x,y,t)
    out_dim={{ N_OUTPUT_FIELDS }},       # e.g. 1 for scalar, 3 for (u,v,p)
    hidden={{ HIDDEN_LAYERS }},          # e.g. [64, 64, 64] or [128, 128, 128, 128]
    activation="{{ ACTIVATION }}",       # "tanh" (most common), "silu", "gelu", "sin"
).to(DEVICE)

model = maybe_compile(model, mode="default")   # torch.compile() if PyTorch >= 2.0
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### XtFC (exact BC enforcement — 1D/simple-domain problems)

```python
from pinneaple_models.pinns.xtfc import build_xtfc, tfc_available

# g(x): particular solution satisfying BCs exactly
# B(x): multiplier that vanishes on all boundaries
def g_fn(x):
    # Example: zero BCs -> g = 0
    return torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)

def B_fn(x):
    # Example: x=0 and x=1 BCs -> B(x) = x*(1-x)
    xi = x[:, 0:1]
    return xi * (1.0 - xi)

model = build_xtfc(
    in_dim=1,
    out_dim=1,
    rf_dim={{ RF_DIM }},           # e.g. 512  (random feature dimension)
    activation="{{ ACTIVATION }}",  # "tanh" recommended
    freeze_random=True,             # frozen random features -> ELM-style solve
    g_fn=g_fn,
    B_fn=B_fn,
    use_tfc=tfc_available(),        # True if tfc library installed
    tfc_n={{ TFC_N }},             # e.g. 100
    tfc_deg={{ TFC_DEG }},         # e.g. 20
    tfc_nC={{ TFC_NC }},           # number of constraints (= number of BCs)
    tfc_x0=[{{ X_MIN }}],
    tfc_xf=[{{ X_MAX }}],
).to(DEVICE)
```

### DeepONet (parametric operator learning)

```python
from pinneaple_models.neural_operators.deeponet import DeepONet

model = DeepONet(
    branch_dim={{ N_SENSOR_POINTS }},    # number of sensor/parameter inputs
    trunk_dim={{ N_SPATIAL_DIMS }},      # spatial coordinate dimension
    out_dim={{ N_OUTPUT_FIELDS }},       # number of output fields
    hidden={{ HIDDEN_WIDTH }},           # e.g. 128
    modes={{ N_MODES }},                 # e.g. 64  (inner product dimension)
).to(DEVICE)
```

### FNO (Fourier Neural Operator — grid-structured data)

```python
from pinneaple_models.neural_operators.fno import FourierNeuralOperator

model = FourierNeuralOperator(
    in_channels={{ N_INPUT_CHANNELS }},
    out_channels={{ N_OUTPUT_CHANNELS }},
    modes1={{ N_FOURIER_MODES_X }},     # e.g. 16
    modes2={{ N_FOURIER_MODES_Y }},     # e.g. 16  (omit for 1D FNO)
    width={{ CHANNEL_WIDTH }},          # e.g. 64
    n_layers={{ N_FNO_LAYERS }},        # e.g. 4
).to(DEVICE)
```

### InversePINN (identify unknown PDE parameters)

```python
from pinneaple_models.pinns.inverse import InversePINN

# Unknown parameters become nn.Parameter — trained jointly with the network
model = InversePINN(
    in_dim={{ N_INPUT_DIMS }},
    out_dim={{ N_OUTPUT_FIELDS }},
    hidden={{ HIDDEN_LAYERS }},
    activation="{{ ACTIVATION }}",
    unknown_params={
        "{{ PARAM_NAME }}": {{ INITIAL_GUESS }},  # e.g. "nu": 0.001
    },
).to(DEVICE)
# Access identified value after training: model.get_param("nu")
```

---

## Step 6 — Physics Loss Definition

This is the most problem-specific step. Define the PDE residual as a function of
model outputs and their automatic-differentiation derivatives.

### 1D ODE template

```python
def pde_residual_1d(model, x_col):
    """
    Example: u'' + {{ COEFF }} * u = f(x)
    Inputs : x_col — (N, 1) tensor, requires_grad=True
    Returns: scalar residual loss
    """
    x = x_col.requires_grad_(True)
    u = model(x)                          # (N, 1)

    u_x  = torch.autograd.grad(u, x, torch.ones_like(u),
                                create_graph=True, retain_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x),
                                create_graph=True)[0]

    # {{ YOUR_PDE }}: e.g. u'' + u = f(x)
    f = {{ RHS_FUNCTION }}(x)
    residual = u_xx + {{ COEFF }} * u - f
    return (residual ** 2).mean()
```

### 2D PDE template (steady-state Poisson / heat / Laplace)

```python
def pde_residual_2d(model, X_col):
    """
    Example: -k (u_xx + u_yy) = f(x,y)
    Inputs : X_col — (N, 2) tensor [x, y], requires_grad=True
    """
    X = X_col.requires_grad_(True)
    u = model(X)                          # (N, 1)

    grads = torch.autograd.grad(u, X, torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]
    u_x, u_y = grads[:, 0:1], grads[:, 1:2]

    u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x),
                                create_graph=True, retain_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, X, torch.ones_like(u_y),
                                create_graph=True)[0][:, 1:2]

    k = {{ DIFFUSIVITY }}          # e.g. 1.0  (thermal conductivity, normalized)
    f = {{ SOURCE_TERM }}(X)       # heat source, body force, etc.
    residual = -k * (u_xx + u_yy) - f
    return (residual ** 2).mean()
```

### 2D time-dependent PDE (advection-diffusion, Burgers, Navier-Stokes)

```python
def pde_residual_transient(model, X_col):
    """
    Example: u_t + a*u_x = nu*u_xx   (1D advection-diffusion)
    Inputs : X_col — (N, 2) tensor [x, t], requires_grad=True
    """
    X = X_col.requires_grad_(True)
    u = model(X)

    grads = torch.autograd.grad(u, X, torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]

    u_xx = torch.autograd.grad(u_x, X, torch.ones_like(u_x),
                                create_graph=True)[0][:, 0:1]

    a  = {{ ADVECTION_SPEED }}     # e.g. 1.0
    nu = {{ DIFFUSIVITY }}         # e.g. 0.01
    residual = u_t + a * u_x - nu * u_xx
    return (residual ** 2).mean()
```

### 3D structural elasticity

```python
def pde_residual_elasticity(model, X_col):
    """
    Example: equilibrium equations for linear elasticity
    Inputs : X_col — (N, 3) tensor [x, y, z], requires_grad=True
    Outputs: model returns (N, 3) [ux, uy, uz]
    """
    X = X_col.requires_grad_(True)
    u = model(X)                           # (N, 3)

    E  = {{ YOUNGS_MODULUS }}              # e.g. 210e9 Pa — use non-dim value
    nu = {{ POISSONS_RATIO }}              # e.g. 0.3

    # Compute strain tensor epsilon_ij = 0.5*(u_i,j + u_j,i)
    # Use torch.autograd.grad for each displacement component...
    # (See pinneaple_models/pinns for built-in elasticity residual helpers)
    pass
```

---

## Step 7 — Training

### Standard training loop

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from pinneaple_train import Trainer, TrainConfig, best_device, AMPContext

DEVICE = best_device()

# Convert numpy arrays to tensors
X_col_t  = torch.from_numpy(X_col).to(DEVICE)
X_bc_t   = torch.from_numpy(X_bc).to(DEVICE)
u_bc_t   = torch.from_numpy(u_bc).to(DEVICE)   # known BC values
# X_ic_t = torch.from_numpy(x_ic).to(DEVICE)   # uncomment for time-dependent
# X_dat_t = torch.from_numpy(X_data).to(DEVICE) # uncomment if using reference data
# u_dat_t = torch.from_numpy(y_data).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr={{ LEARNING_RATE }})
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma={{ LR_DECAY }})
    # alternatives: CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
amp_ctx   = AMPContext(device=str(DEVICE), enabled=DEVICE.type == "cuda")

# Loss weights — tune these for your problem
W_PDE  = {{ WEIGHT_PDE }}     # e.g. 1.0   (PDE residual)
W_BC   = {{ WEIGHT_BC }}      # e.g. 10.0  (boundary conditions)
W_IC   = {{ WEIGHT_IC }}      # e.g. 10.0  (initial condition — set 0.0 for steady)
W_DATA = {{ WEIGHT_DATA }}    # e.g. 1.0   (reference data — set 0.0 for pure PINN)

history = {"pde": [], "bc": [], "ic": [], "data": [], "total": []}

EPOCHS = {{ N_EPOCHS }}       # e.g. 2000-10000
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    with amp_ctx.autocast():
        loss_pde  = pde_residual_2d(model, X_col_t)
        pred_bc   = model(X_bc_t)
        loss_bc   = ((pred_bc - u_bc_t) ** 2).mean()
        # loss_ic  = ...  # uncomment for time-dependent
        # loss_data = ... # uncomment for data-driven

        loss = W_PDE * loss_pde + W_BC * loss_bc  # + W_IC*loss_ic + W_DATA*loss_data

    if amp_ctx.enabled:
        amp_ctx.scaler.scale(loss).backward()
        amp_ctx.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), {{ GRAD_CLIP }})  # e.g. 1.0
        amp_ctx.scaler.step(optimizer)
        amp_ctx.scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), {{ GRAD_CLIP }})
        optimizer.step()

    scheduler.step()

    history["pde"].append(float(loss_pde))
    history["bc"].append(float(loss_bc))
    history["total"].append(float(loss))

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1:5d} | total={loss:.3e} | pde={loss_pde:.3e} | bc={loss_bc:.3e}")
```

### Using `Trainer` + `TrainConfig` (preferred for production)

```python
from pinneaple_train import Trainer, TrainConfig

cfg = TrainConfig(
    epochs={{ N_EPOCHS }},
    lr={{ LEARNING_RATE }},
    device=str(DEVICE),
    seed={{ SEED }},
    grad_clip={{ GRAD_CLIP }},
    amp=DEVICE.type == "cuda",
    compile=False,                  # set True for PyTorch >= 2.0 + GPU
    grad_accum_steps=1,             # increase for large batches
)

def loss_fn(model, y_hat, batch):
    x_col, x_bc, u_bc = batch["x_col"], batch["x_bc"], batch["u_bc"]
    l_pde = pde_residual_2d(model, x_col)
    l_bc  = ((model(x_bc) - u_bc) ** 2).mean()
    return {"total": W_PDE * l_pde + W_BC * l_bc, "pde": l_pde, "bc": l_bc}

trainer = Trainer(model, loss_fn=loss_fn)
result  = trainer.fit(train_loader, val_loader, cfg)
print(f"Best val loss: {result['best_val']:.4e}")
```

### Save checkpoint

```python
import torch
from pathlib import Path

ckpt_dir = Path("data/{{ USE_CASE_NAME }}/checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "epoch": EPOCHS,
    "history": history,
    "scaler_X": scaler_X if "scaler_X" in dir() else None,
    "scaler_y": scaler_y if "scaler_y" in dir() else None,
}, ckpt_dir / "model.pt")
print(f"Saved: {ckpt_dir / 'model.pt'}")

# Load:
# ckpt = torch.load(ckpt_dir / "model.pt", map_location=DEVICE)
# model.load_state_dict(ckpt["model_state"])
```

---

## Step 8 — Evaluation & Visualization

### Compute metrics

```python
from pinneaple_train import build_metrics_from_cfg
import numpy as np

metrics = build_metrics_from_cfg(["mse", "rmse", "rel_l2", "r2", "max_error"])

model.eval()
with torch.no_grad():
    y_pred = model(X_val_t).cpu().numpy()

y_true = y_val   # shape: (N_val, n_out)

for name, fn in metrics.items():
    print(f"  {name:<12}: {fn(y_true, y_pred):.4e}")
```

### Inference on a dense evaluation grid

```python
from pinneaple_inference import infer_on_grid_2d
import numpy as np

# 1D case
x_eval = np.linspace({{ X_MIN }}, {{ X_MAX }}, {{ N_EVAL_PTS }}, dtype=np.float32)[:, None]
with torch.no_grad():
    u_pred = model(torch.from_numpy(x_eval).to(DEVICE)).cpu().numpy()

# 2D case — use batched_inference for memory-safe large grids
from pinneaple_train import batched_inference

nx_eval, ny_eval = {{ NX_EVAL }}, {{ NY_EVAL }}    # e.g. 200, 200
x_eval = np.linspace({{ X_MIN }}, {{ X_MAX }}, nx_eval, dtype=np.float32)
y_eval = np.linspace({{ Y_MIN }}, {{ Y_MAX }}, ny_eval, dtype=np.float32)
XX, YY = np.meshgrid(x_eval, y_eval)
XY_flat = np.column_stack([XX.ravel(), YY.ravel()]).astype(np.float32)

u_flat = batched_inference(model, XY_flat, batch_size={{ BATCH_SIZE }}, device=str(DEVICE))
u_grid = u_flat.reshape(ny_eval, nx_eval)
```

### Visualization templates

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out_dir = Path("data/{{ USE_CASE_NAME }}/plots")
out_dir.mkdir(parents=True, exist_ok=True)

# --- 1. Training loss curve ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(history["total"], label="total", lw=1.5)
ax.semilogy(history["pde"],   label="pde",   lw=1.2, linestyle="--")
ax.semilogy(history["bc"],    label="bc",    lw=1.2, linestyle=":")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("{{ PROBLEM_NAME }} — Training Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(out_dir / "loss_curve.png", dpi=150)
plt.close()

# --- 2. Solution field (2D) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Predicted field
im0 = axes[0].pcolormesh(XX, YY, u_grid, cmap="{{ COLORMAP }}", shading="auto")
    # good colormaps: "RdYlBu_r" (diverging), "viridis" (sequential), "RdYlGn_r" (risk)
plt.colorbar(im0, ax=axes[0], label="{{ FIELD_LABEL }}")
axes[0].set_xlabel("{{ COORD_1 }}")
axes[0].set_ylabel("{{ COORD_2 }}")
axes[0].set_title("Predicted {{ FIELD_NAME }}")

# Reference field (if available)
# im1 = axes[1].pcolormesh(XX, YY, u_ref_grid, cmap="{{ COLORMAP }}", shading="auto")
# plt.colorbar(im1, ax=axes[1], label="{{ FIELD_LABEL }}")
# axes[1].set_title("Reference {{ FIELD_NAME }}")

# Error map
# err = np.abs(u_grid - u_ref_grid)
# im2 = axes[2].pcolormesh(XX, YY, err, cmap="Reds", shading="auto")
# plt.colorbar(im2, ax=axes[2], label="|error|")
# axes[2].set_title(f"Absolute Error (max={err.max():.2e})")

plt.tight_layout()
plt.savefig(out_dir / "solution_field.png", dpi=150)
plt.close()

# --- 3. Model comparison (if multiple models trained) ---
# fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5))
# for ax, (model_id, pred) in zip(axes, predictions.items()):
#     im = ax.pcolormesh(XX, YY, pred.reshape(ny_eval, nx_eval), ...)
#     ax.set_title(model_id)
# plt.savefig(out_dir / "model_comparison.png", dpi=150)

print(f"Plots saved to: {out_dir}")
```

---

## Step 9 — Digital Twin (Optional)

Add this section when you need real-time monitoring, data assimilation from
live sensors, or anomaly detection.

```python
from pinneaple_digital_twin import (
    build_digital_twin, DigitalTwinConfig,
    MockStream, Sensor, SensorRegistry,
    ThresholdDetector, ZScoreDetector,
    EnsembleKalmanFilter,
)
from pinneaple_digital_twin.monitoring import AnomalyMonitor

# --- 9.1 Wrap trained surrogate as a digital twin ---
dt = build_digital_twin(
    model,
    field_names=["{{ FIELD_1 }}", "{{ FIELD_2 }}"],   # outputs to monitor
    coord_names=["{{ COORD_1 }}", "{{ COORD_2 }}"],   # spatial coordinates
    update_interval={{ UPDATE_INTERVAL }},             # seconds, e.g. 1.0
)

# --- 9.2 Anomaly detectors ---
# Threshold: alert if field value exceeds a limit
dt.anomaly_monitor.add_detector(
    ThresholdDetector({"{{ FIELD_1 }}": {{ ALERT_THRESHOLD }}})
        # e.g. {"temperature": 1200.0}  or  {"von_mises": 0.8 * sigma_yield}
)
# Z-score: alert if value deviates by more than N standard deviations
# dt.anomaly_monitor.add_detector(
#     ZScoreDetector(window={{ WINDOW_SIZE }}, n_sigma={{ N_SIGMA }})
# )

# --- 9.3 Data assimilation (EnKF) ---
# enkf = EnsembleKalmanFilter(
#     n_ensemble={{ N_ENSEMBLE }},    # e.g. 50
#     obs_noise_std={{ OBS_NOISE }},  # e.g. 0.01
# )
# dt.set_filter(enkf)

# --- 9.4 Sensor streams ---
def {{ SENSOR_FUNCTION }}(t: float) -> dict:
    """Returns dict of sensor readings at time t."""
    # Replace with actual sensor API or simulation
    value_nominal = {{ NOMINAL_VALUE }}
    noise = {{ NOISE_AMPLITUDE }} * np.random.randn()
    return {"{{ FIELD_1 }}": value_nominal + noise}

dt.add_stream(MockStream(
    "{{ SENSOR_NAME }}",
    ["{{ FIELD_1 }}"],
    {{ SENSOR_FUNCTION }},
    tick_interval={{ TICK_INTERVAL }},          # seconds between readings
    coords={"{{ COORD_1 }}": {{ SENSOR_X }},   # sensor location in domain
             "{{ COORD_2 }}": {{ SENSOR_Y }}},
))

# --- 9.5 Anomaly callback ---
dt.on_anomaly(lambda ev: print(
    f"[ALERT] {ev.field}={ev.observed:.3e} > threshold at t={ev.time:.2f}s"
))

# --- 9.6 Run twin ---
import time
with dt:
    time.sleep({{ RUN_DURATION }})   # seconds to monitor

print(f"Monitoring done. Alerts: {len(dt.anomaly_monitor.all_events)}")

# --- 9.7 Export history ---
df_history = dt.state_history_to_dataframe()
df_history.to_parquet("data/{{ USE_CASE_NAME }}/twin_history.parquet")
print(df_history.tail())
```

---

## Step 10 — YAML Config (Reproducible Pipeline)

All steps above can be driven by a YAML config file and the pipeline runner.
See `template_config.yaml` in this directory for the full annotated version.

```yaml
# configs/{{ USE_CASE_NAME }}.yaml  — minimal example
pipeline:
  name: {{ USE_CASE_NAME }}
  out_dir: data/artifacts/{{ USE_CASE_NAME }}
  log_level: INFO

problem:
  id: {{ PRESET_NAME }}
  params:
    {{ PARAM_1 }}: {{ VALUE_1 }}
    {{ PARAM_2 }}: {{ VALUE_2 }}

geometry:
  type: {{ GEOMETRY_TYPE }}          # interval | rectangle | mesh_file
  params:
    xlim: [{{ X_MIN }}, {{ X_MAX }}]
    ylim: [{{ Y_MIN }}, {{ Y_MAX }}]  # omit for 1D
    n_points: {{ N_GRID_PTS }}

solver:
  backend: {{ SOLVER_BACKEND }}      # builtin | openfoam | fenics
  params:
    method: {{ FDM_METHOD }}         # ftcs | btcs | crank_nicolson (builtin only)
    nx: {{ NX }}
    nt: {{ NT }}
    dt: {{ DT }}

dataset:
  n_collocation: {{ N_COLLOCATION }}
  n_boundary: {{ N_BOUNDARY }}
  n_ic: {{ N_IC }}
  n_data: {{ N_DATA }}
  strategy: lhs                       # uniform | lhs | sobol
  train_frac: 0.85
  seed: 42

models:
  - id: {{ MODEL_ID }}
    type: VanillaPINN                 # see model selection guide above
    params:
      hidden: {{ HIDDEN_LAYERS }}
      activation: tanh
    train:
      epochs: {{ N_EPOCHS }}
      lr: {{ LEARNING_RATE }}
      device: {{ DEVICE }}
      amp: false
    physics_weights:
      pde: {{ WEIGHT_PDE }}
      bc: {{ WEIGHT_BC }}
      ic: {{ WEIGHT_IC }}
      data: {{ WEIGHT_DATA }}

metrics: [mse, rmse, rel_l2, r2, max_error]

report:
  format: html
  save_model: true
  plots:
    - type: solution_field
      field: {{ FIELD_NAME }}
    - type: loss_curve
    - type: error_map
      field: {{ FIELD_NAME }}
```

Run with:

```bash
python -m pinneaple_arena.runner.run_pipeline \
    --config examples/use_cases/{{ USE_CASE_NAME }}/configs/{{ USE_CASE_NAME }}.yaml
# or:
python examples/pinneaple_arena/09_full_pipeline_yaml.py \
    --config examples/use_cases/{{ USE_CASE_NAME }}/configs/{{ USE_CASE_NAME }}.yaml
```

---

## Recommended File Structure

```
examples/use_cases/{{ USE_CASE_NAME }}/
├── README.md                        <- this file, adapted for your use case
├── 01_solver_sweep.py               <- reference data generation (Step 3)
├── 02_build_dataset.py              <- dataset construction (Step 4)
├── 03_train_surrogate.py            <- model training (Steps 5-7)
├── 04_evaluate_visualize.py         <- metrics + plots (Step 8)
├── 05_digital_twin.py               <- monitoring (Step 9, optional)
├── configs/
│   └── {{ USE_CASE_NAME }}.yaml    <- full pipeline YAML (Step 10)
└── data/
    ├── solver_output/               <- raw solver results (.xdmf, .nc, ...)
    ├── dataset.parquet              <- flattened training dataset
    ├── checkpoints/
    │   └── model.pt                 <- trained surrogate
    └── plots/                       <- generated figures
```

---

## What Is Already in pinneaple vs. What You Implement

| Component | pinneaple module | Status |
|-----------|-----------------|--------|
| Problem presets | `pinneaple_environment.presets` | Ready — use `list_presets()` |
| Built-in FDM solver | `pinneaple_solvers` (`builtin`) | Ready |
| OpenFOAM bridge | `pinneaple_solvers.openfoam_bridge` | Ready — needs OF installed |
| FEniCS bridge | `pinneaple_solvers.fenics_bridge` | Ready — needs dolfinx/fenics |
| VanillaPINN, XtFC, VPINN, XPINN | `pinneaple_models.pinns` | Ready |
| InversePINN | `pinneaple_models.pinns.inverse` | Ready |
| DeepONet, FNO, GNO, PINO | `pinneaple_models.neural_operators` | Ready |
| Trainer, AMP, compile, grad accum | `pinneaple_train` | Ready |
| Digital twin, EnKF, anomaly | `pinneaple_digital_twin` | Ready |
| Metrics (MSE, RMSE, rel L2, R2) | `pinneaple_train` | Ready |
| Grid inference, batch inference | `pinneaple_inference` | Ready |

**You implement:**

| Step | What to do |
|------|-----------|
| Problem spec | Define `ProblemSpec` or choose a preset and override params |
| PDE residual | Write the physics loss function for your governing equation |
| Boundary conditions | Specify `DirichletBC` / `NeumannBC` for your domain |
| Geometry (complex) | Build mesh with gmsh/pygmsh for non-rectangular domains |
| Data pipeline | Flatten solver output into (coords, params, fields) dataframe |
| Normalization | Apply `StandardScaler` or custom scaling to all inputs and outputs |
| g(x) and B(x) | For XtFC: particular solution + BC-vanishing multiplier |
| Sensor functions | For digital twin: wrap real sensor APIs or simulation calls |

---

## References

- Raissi, M. et al. (2019). "Physics-informed neural networks." *Journal of Computational Physics.*
- Lu, L. et al. (2021). "Learning nonlinear operators via DeepONet." *Nature Machine Intelligence.*
- Li, Z. et al. (2021). "Fourier Neural Operator for Parametric PDEs." *ICLR 2021.*
- Leake, C. & Mortari, D. (2020). "Deep Theory of Functional Connections." arXiv:2005.01219.
- pinneaple docs: `QUICKSTART.md`
- pinneaple examples: `examples/pinneaple_arena/`
