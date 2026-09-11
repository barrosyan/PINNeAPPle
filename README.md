# PINNeAPPle 🍍
### Your Physics AI Laboratory — from first principles to real-world systems

> *Experiment. Learn. Build. Then scale — anywhere.*

PINNeAPPle is an open-source **Physics AI research and experimentation platform** designed to take you from your first physics-informed neural network all the way to **robust, production-ready solutions** — independent of any specific framework, vendor, or ecosystem.

<div align="center">

| | |
|:---:|:---:|
| ![Clamped Plate](./data/viz_06_structural.png) | ![2D Heat Equation](./data/viz_02_heat_2d.png) |
| *Clamped Plate — deflection, Von Mises stress & bending moment* | *2D Heat Equation — Exact vs PINN across time steps* |
| ![Lamb-Oseen Vortex](./data/viz_03_vortex_dynamics.png) | ![Allen-Cahn Phase](./data/viz_04_phase_field.png) |
| *Lamb-Oseen Vortex Pair — vorticity evolution* | *Allen-Cahn Phase Separation — interface dynamics* |

</div>

---

## Why PINNeAPPle?

Modern Physics AI ecosystems are powerful — but they assume you already understand:

- How to formulate physical problems correctly
- Which architectures to use (PINNs, operators, surrogates…)
- How to validate physics consistency
- How to benchmark and trust your results

**PINNeAPPle is where you build that foundation.**

```
Your physics problem
        ↓
  [ PINNeAPPle ]   ← experiment freely here
    Understand the physics
    Try architectures
    Compare approaches
    Validate results
    Build intuition
        ↓
[ Your Target Stack ]
  (custom infra, HPC, cloud, internal platform, etc.)
  Scale, deploy, integrate
```

---

## Package Structure

PINNeAPPle is organized into **8 mega-modules**, each grouping related sub-modules:

```
pinneapple_physics/
├── pde_environment/    # PDE problem specs, BCs, ICs, presets, RANS
├── pinn_solver/        # PINN compiler, DoMINO domain decomposition
└── symbolic_pde/       # SymPy → autograd residual compiler

pinneapple_neural/
├── architectures/      # SIREN, ModifiedMLP, AFNO, HashGridMLP, MeshGraphNet
├── trainer/            # Trainer, TwoPhase, DDP, Causal, HPC utilities
└── predictor/          # Batched inference, grid evaluation, FlowVisualizer

pinneapple_analysis/
├── uncertainty/        # MC-Dropout, Ensemble UQ, conformal, calibration
├── validation/         # Conservation, BC, symmetry checks vs. reference
└── inverse_problems/   # Noise models, regularizers, EKI, SINDy discovery

pinneapple_adaptation/
├── transfer_learning/  # Fine-tuning, layer freezing, progressive unfreezing
└── meta_learning/      # MAML, Reptile, PDETaskSampler, few-shot adaptation

pinneapple_simulation/
├── numerical_solvers/  # FEM, FDM, FVM, Spectral, SPH, LBM, OpenFOAM, FEniCS
├── particle_dynamics/  # MPM, SPH particles, rigid-body (pure PyTorch)
└── external_solvers/   # OpenFOAM, MATLAB, FMU/Modelica, FEniCS bridges

pinneapple_systems/
├── time_series/        # LSTM, GRU, NBeats, TFT, TCN, XGBoost, HHT, FFT
├── cosimulation/       # Graph co-sim engine: PINNNode, CoSimGraph, CoSimTrainer
└── digital_twin/       # Live twin, sensor streams, EKF/EnKF, anomaly detection

pinneapple_design/
├── geometry/           # SDF library, CSG, physics domains, mesh, NACA airfoil
└── design_optimizer/   # Adjoint, Pareto, Bayesian/evolutionary optimization

pinneapple_tools/
├── visualization/      # CFD-style plots, streamlines, Q-criterion, animations
├── model_export/       # TorchScript, ONNX, CSV, NPZ
├── hpo_experiments/    # Paper discovery, knowledge base, HPO
├── benchmark_suite/    # Arena, leaderboards, transfer/meta benchmark pipelines
└── compute_backends/   # PyTorch (default) + JAX backend abstraction
```

Additional packages:

- `pinneapple_data` — UPD dataset
- `pinneapple_pdb` — physics database
- `pinneapple_problemdesign` — NLP → PDE agent
- `pinneapple_app` — FastAPI + frontend web app for benchmarking PINN models on physics problems (Docker-composed backend/frontend)
- `pinneapple_arena` — YAML/JSON-driven multi-model physics benchmark runner (~80+ architectures, physics losses, UQ, inverse problems)
- `pinneapple_blender` — export a field/trajectory as a `.ply` sequence, and optionally build/render a Blender scene via a real local Blender install
- `pinneapple_hub` — model hub client (`push_to_hub`/`from_pretrained` + `ModelCard`) built on the Hugging Face Hub
- `pinneapple_llm` — LLM-assisted physics-AI pipeline drafting, gated by a physics-grounded `PhysicsGuardrail` verification layer
- `pinneapple_models` — compatibility shim re-exporting `pinneapple_neural.architectures` (not a separate package)
- `pinneapple_perception` — extracts physics observations (velocity fields, boundary geometry, modal frequencies) from images, video, and audio
- `pinneapple_registry` — local, self-hosted artifact registry: versioned model/dataset storage, experiment tracking, and problem-spec history
- `pinneapple_solvers` — compatibility shim re-exporting `pinneapple_simulation.numerical_solvers` (not a separate package)
- `pinneapple_train` — compatibility shim re-exporting `pinneapple_neural.trainer` (not a separate package)
- `pinneapple_worldmodel` — generalist Physics Foundation Model trained across many physics domains

---

## Installation

```bash
pip install pinneapple
```

With optional extras:

```bash
pip install "pinneapple[solvers]"      # numba-accelerated FDM/FEM/LBM
pip install "pinneapple[pinn]"         # SymPy symbolic PDE compiler
pip install "pinneapple[geom]"         # trimesh, meshio, gmsh
pip install "pinneapple[fenics]"       # FEniCS / DOLFINx bridge
pip install "pinneapple[export]"       # ONNX export
pip install "pinneapple[all]"          # everything
```

---

## Three Tiers of Physics AI Experience

### Tier 1 — Explorer
> *"I understand the physics. I want to see what AI can do with it."*

```python
from pinneapple_physics import get_preset, solve_pde
from pinneapple_neural import build_model

# Load a 2D Poisson problem preset
spec = get_preset("poisson_2d")

# Build a SIREN network and train it in one call
model = build_model("siren", in_dim=2, out_dim=1, hidden_dim=64, n_layers=4)
result = solve_pde(spec, model, epochs=3000)
result["history"]  # {"loss": [...]}
```

---

### Tier 2 — Experimenter
> *"I want to test ideas and compare approaches."*

```python
from pinneapple_tools.benchmark_suite import Arena

runner  = Arena.from_preset("burgers_1d")
results = runner.compare(["VanillaPINN", "siren"], epochs=2000)
print(results.leaderboard())
```

<div align="center">

![Potential Flow Past Cylinder](./data/viz_05_wave_2d.png)
*Potential Flow Past Circular Cylinder — exact solution vs PINN vs pointwise error*

</div>

---

### Tier 3 — Builder
> *"I want to turn this into a real system."*

```python
from pinneapple_neural.trainer import DDPPINNTrainer, DDPTrainerConfig
from pinneapple_tools.model_export import export_onnx
from pinneapple_systems.digital_twin import build_digital_twin, MQTTStream

# Distributed training: one DDPPINNTrainer.setup(rank, world_size) call per
# spawned process, then loss_fn(model, epoch) -> Tensor each step
cfg     = DDPTrainerConfig(backend="nccl", world_size=4)
trainer = DDPPINNTrainer(model, cfg)
trainer.setup(rank=0, world_size=4)
history = trainer.train(loss_fn, n_epochs=10_000)

# Export to ONNX
export_onnx(model, "surrogate.onnx", example_input=x_sample)

# Wrap as a live digital twin
twin = build_digital_twin(model, field_names=["u", "v", "p"])
twin.add_stream(MQTTStream(broker="sensors.local", topic="plant/telemetry", sensor_id="s1", field_names=["u", "v", "p"]))
twin.start()
```

<div align="center">

![Model Comparison](./outputs/07_forecast_comparison.png)
*Multi-model forecast comparison across test windows — Naive, FFT-only, LSTM, FFT+LSTM*

</div>

---

## Key Features

| Mega-module | Sub-modules | What it does |
|---|---|---|
| `pinneapple_physics` | `pde_environment` · `pinn_solver` · `symbolic_pde` | Define PDEs, compile PINN losses, SymPy → autograd |
| `pinneapple_neural` | `architectures` · `trainer` · `predictor` | SIREN/AFNO/MGN models, distributed training, inference |
| `pinneapple_analysis` | `uncertainty` · `validation` · `inverse_problems` | UQ, physics consistency checks, parameter inversion |
| `pinneapple_adaptation` | `transfer_learning` · `meta_learning` | Fine-tune across PDEs, MAML/Reptile few-shot |
| `pinneapple_simulation` | `numerical_solvers` · `particle_dynamics` · `external_solvers` | FEM/FDM/SPH/LBM, OpenFOAM/FEniCS bridges |
| `pinneapple_systems` | `time_series` · `cosimulation` · `digital_twin` | Forecasting, co-sim graphs, live sensor fusion |
| `pinneapple_design` | `geometry` · `design_optimizer` | SDF/CSG geometry, adjoint + Bayesian shape opt |
| `pinneapple_tools` | `visualization` · `model_export` · `benchmark_suite` · `compute_backends` | CFD plots, ONNX export, Arena benchmarks, JAX backend |

---

## Quick Examples

```python
import torch

# ── Physics problem definition ──────────────────────────────────────────────
from pinneapple_physics.pde_environment import get_preset
from pinneapple_physics.pinn_solver import compile_problem

spec   = get_preset("burgers_1d")
losses = compile_problem(spec)

# ── Neural network architectures ────────────────────────────────────────────
from pinneapple_neural.architectures import ModelRegistry, SIREN, AFNO
from pinneapple_neural.trainer import Trainer, TrainConfig
from pinneapple_simulation.numerical_solvers.problem_runner import generate_pinn_dataset

model = ModelRegistry.build("siren", in_dim=2, out_dim=1, hidden_dim=128, n_layers=6)

# One full physics batch (collocation + BC/IC points), re-used every epoch
batch  = generate_pinn_dataset(spec, n_col=4096, n_bc=512)
loader = [{k: (torch.as_tensor(v) if hasattr(v, "dtype") else v) for k, v in batch.items()}]

cfg     = TrainConfig(epochs=5000, device="cuda")
trainer = Trainer(model, losses)
result  = trainer.fit(loader, loader, cfg)

# ── Uncertainty quantification ──────────────────────────────────────────────
from pinneapple_analysis.uncertainty import uq_predict
from pinneapple_analysis.validation import validate_model

uq_result  = uq_predict(model, x_test, method="mc_dropout")
val_report = validate_model(model, spec)

# ── Design optimization ─────────────────────────────────────────────────────
from pinneapple_design.geometry import get_domain
from pinneapple_design.design_optimizer import DesignOptLoop, DesignOptConfig

domain = get_domain("lid_driven_cavity_2d")
x_int  = domain.sample_interior(4096)

# ── Simulation data generation ──────────────────────────────────────────────
from pinneapple_simulation.numerical_solvers import HeatConduction3D
from pinneapple_simulation.numerical_solvers.fdm3d import HeatConfig3D

solver = HeatConduction3D(HeatConfig3D(nx=32, ny=32, nz=32))
data   = solver.solve()

# ── Time series forecasting ─────────────────────────────────────────────────
from pinneapple_systems.time_series import NaiveForecaster

forecaster = NaiveForecaster()
forecaster.fit(train_series)
forecast = forecaster.predict(24)

# ── Benchmarking ────────────────────────────────────────────────────────────
from pinneapple_tools.benchmark_suite import Arena

runner = Arena.from_preset("poisson_2d")
result = runner.run("siren", epochs=5000)
print(result.summary())
```

---

## Examples

| Folder | What it covers |
|--------|---------------|
| `examples/pde_environment/` | PDE presets, BCs, problem specs |
| `examples/pinn_solver/` | PINN compiler, symbolic losses |
| `examples/architectures/` | Model registry, SIREN, AFNO, GNN, operators |
| `examples/trainer/` | Training loops, DDP, HPC, AMP |
| `examples/numerical_solvers/` | FEM, FDM, FVM, SPH, LBM, spectral |
| `examples/time_series/` | Forecasting, backtesting, uncertainty |
| `examples/geometry/` | SDF, CSG, mesh, airfoil generation |
| `examples/benchmark_suite/` | Arena, YAML configs, leaderboards |
| `examples/hpo_experiments/` | Paper discovery, knowledge base |
| `examples/data_pipeline/` | UPD datasets, Zarr, active learning |
| `examples/physics_db/` | Physics database, NASA/Earthdata |
| `examples/problem_designer/` | NLP → PDE agent |

---

## Philosophy

> *If you can't validate it, you shouldn't deploy it.*

Physics AI is about:

- Correct formulations
- Reliable validation
- Understanding failure modes
- Making informed decisions

---

## Positioning

|  | PINNeAPPle |
|--|------------|
| Vendor lock-in | ❌ Not tied to any vendor |
| Just a PINN library | ❌ Much more than that |
| Just experimentation | ❌ Bridges to production |
| ✅ What it is | A controlled environment to **design, test, and validate** Physics AI systems |

---

## Citation

If you use **PINNeAPPle** in academic research, technical reports, benchmarks, or industrial publications, please cite the framework.

### BibTeX

```bibtex
@software{pinneapple2026,
  title        = {PINNeAPPle: An Open-Source Physics AI Research and Experimentation Platform},
  author       = {Barros, Yan and Contributors},
  year         = {2026},
  url          = {https://github.com/PINNeAPPle-Labs/PINNeAPPle},
  version      = {0.5.0}
}
```

---

## Support the Project

If this project makes sense to you, **give it a star** ⭐

It helps grow the ecosystem, attract contributors, and build a real standard.

---

*Built for researchers and engineers who take physics seriously.*
