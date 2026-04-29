# PINNeAPPle 🍍
**Your Physics AI Laboratory — the path to NVIDIA PhysicsNeMo**

> *Experiment. Learn. Build. Then scale with the best.*

PINNeAPPle is an open-source **Physics AI research and experimentation platform** built to take you from your first physics-informed neural network all the way to production-ready solutions — and to prepare you to get the most out of **NVIDIA PhysicsNeMo**, the industry-leading platform for large-scale Physics AI.

---

## Why PINNeAPPle?

NVIDIA PhysicsNeMo is the gold standard for Physics AI in production: battle-tested, GPU-optimized, enterprise-grade, and built for scale. But reaching that level of development requires a solid foundation in Physics AI concepts, workflows, and tooling.

**PINNeAPPle is the laboratory where you build that foundation.**

Think of it as the gap-bridger:

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
  [ PhysicsNeMo ]  ← go here when you're ready for production
  Scale to GPUs/clusters
  Deploy to digital twins
  Integrate with NVIDIA stack
```

You don't need to be an ML expert to start. PINNeAPPle meets you where you are.

---

## Three Tiers of Physics AI Experience

PINNeAPPle is designed around three user profiles. Find yours.

---

### 🌱 Tier 1 — Explorer
*"I understand the physics. I want to see what AI can do with it."*

**You are:** A physicist, engineer, or researcher new to Physics AI.
**You need:** Pre-built problems, clean APIs, results you can trust.
**You get:** Instant experiments on classical physics problems with zero ML boilerplate.

```python
from pinneaple_environment import BurgersPreset

problem = BurgersPreset(nu=0.01)
model   = problem.build_model()
trainer = problem.build_trainer(n_epochs=3000)
result  = trainer.fit(model)
result.plot()
```

**Milestone reached →** You can run, visualize and interpret PINN solutions for standard PDEs.

**Where to go next:** Tier 2, or directly to [PhysicsNeMo Quickstart](https://docs.nvidia.com/deeplearning/physicsnemo/physicsnemo-core/index.html) for simple problems.

**Getting started examples:**
- [`examples/getting_started/01_harmonic_oscillator.py`](examples/getting_started/01_harmonic_oscillator.py) — spring-mass system
- [`examples/getting_started/02_damped_oscillator.py`](examples/getting_started/02_damped_oscillator.py) — three damping regimes
- [`examples/getting_started/03_heat_diffusion_1d.py`](examples/getting_started/03_heat_diffusion_1d.py) — 1D heat equation
- [`examples/getting_started/04_wave_equation_1d.py`](examples/getting_started/04_wave_equation_1d.py) — 1D wave propagation
- [`examples/getting_started/05_logistic_growth.py`](examples/getting_started/05_logistic_growth.py) — population dynamics
- [`examples/getting_started/06_lotka_volterra.py`](examples/getting_started/06_lotka_volterra.py) — predator-prey system
- [`examples/getting_started/07_nonlinear_pendulum.py`](examples/getting_started/07_nonlinear_pendulum.py) — nonlinear pendulum
- [`examples/getting_started/08_van_der_pol.py`](examples/getting_started/08_van_der_pol.py) — nonlinear oscillator
- [`examples/getting_started/09_lorenz_system.py`](examples/getting_started/09_lorenz_system.py) — chaos and attractors
- [`examples/getting_started/10_coupled_oscillators.py`](examples/getting_started/10_coupled_oscillators.py) — multi-DOF system

---

### 🔬 Tier 2 — Experimenter
*"I understand Physics AI basics. I want to test ideas and compare approaches."*

**You are:** A scientist or ML engineer who has run PINNs before and wants to go deeper.
**You need:** Model comparison tools, parametric studies, active learning, uncertainty, inverse problems.
**You get:** A full experimentation framework — swap architectures, run benchmarks, validate physics, export results.

```python
from pinneaple_arena import ArenaRunner
from pinneaple_models.registry import ModelRegistry

# Compare FNO vs DeepONet vs PINN on the same problem
runner = ArenaRunner.from_yaml("configs/arena/burgers_benchmark.yaml")
results = runner.run_all()
results.leaderboard()
```

**Milestones reached →**
- You know which architecture works best for your problem class.
- You can estimate uncertainty, identify failure modes, and validate conservation laws.
- You have a trained surrogate ready to export.

**Where to go next:** Tier 3, or directly to PhysicsNeMo if your problem fits a supported preset.

**Key templates:**
- [`templates/16_uncertainty_quantification.py`](templates/16_uncertainty_quantification.py) — MC Dropout + Ensemble + Conformal
- [`templates/18_inverse_problem.py`](templates/18_inverse_problem.py) — ensemble Kalman inversion
- [`templates/19_fno_neural_operator.py`](templates/19_fno_neural_operator.py) — Fourier Neural Operator
- [`templates/20_deeponet_surrogate.py`](templates/20_deeponet_surrogate.py) — DeepONet operator learning
- [`templates/21_active_learning.py`](templates/21_active_learning.py) — residual-based adaptive sampling
- [`templates/26_rom_pod_dmd.py`](templates/26_rom_pod_dmd.py) — POD / DMD reduced order models
- [`templates/28_arena_benchmark.py`](templates/28_arena_benchmark.py) — multi-model leaderboard
- [`templates/32_physics_validation.py`](templates/32_physics_validation.py) — physics consistency checks

---

### 🚀 Tier 3 — Builder
*"I have a validated Physics AI approach. I want to scale it and put it in production."*

**You are:** A senior engineer or team lead building a Physics AI product.
**You need:** Distributed training, model serving, digital twins, ONNX export — and a clear migration path to PhysicsNeMo.
**You get:** Production-grade tooling and a direct comparison with PhysicsNeMo to guide your migration.

```python
# Train at scale with DDP
from pinneaple_train.distributed import DDPPINNTrainer

# Export for deployment
from pinneaple_export.onnx_exporter import ONNXExporter

# Compare directly with PhysicsNeMo
# see examples/vs_physicsnemo/
```

**Milestones reached →**
- Your model trains on multiple GPUs with validated physics.
- You have an ONNX / TorchScript artifact ready to serve.
- You know exactly what PhysicsNeMo gives you on top.

**Key templates:**
- [`templates/07_ddp_distributed.py`](templates/07_ddp_distributed.py) — multi-GPU DDP training
- [`templates/22_model_serving.py`](templates/22_model_serving.py) — FastAPI REST inference
- [`templates/23_model_export.py`](templates/23_model_export.py) — ONNX + TorchScript export
- [`templates/24_digital_twin.py`](templates/24_digital_twin.py) — live digital twin with assimilation
- [`templates/33_rans_turbulence.py`](templates/33_rans_turbulence.py) — RANS k-ω SST channel flow
- [`templates/34_heat_conduction_3d.py`](templates/34_heat_conduction_3d.py) — 3D solver with FEM comparison
- [`templates/35_jax_backend.py`](templates/35_jax_backend.py) — JAX backend benchmarking

---

## The Bridge to PhysicsNeMo

The `examples/vs_physicsnemo/` directory contains **side-by-side comparisons** of PINNeAPPle and PhysicsNeMo solving the same problems. Use these to:

1. Validate that your PINNeAPPle results match PhysicsNeMo outputs.
2. Understand what PhysicsNeMo adds (GPU kernel fusion, multi-node, Modulus presets).
3. Migrate your pipeline with confidence.

| PINNeAPPle | PhysicsNeMo equivalent |
|---|---|
| `pinneaple_pinn` + SymPy PDE compiler | `physicsnemo.sym` PDE system |
| `pinneaple_models` FNO / DeepONet | `physicsnemo.models` FNO / DeepONet |
| `pinneaple_train` DDPPINNTrainer | PhysicsNeMo distributed training |
| `pinneaple_digital_twin` | Omniverse / NVIDIA Digital Twin platform |
| `pinneaple_export` ONNX | Triton Inference Server deployment |
| `pinneaple_uq` ensemble | PhysicsNeMo UQ extensions |

> **Recommendation:** Use PINNeAPPle until your model trains correctly, your physics residuals are below tolerance, and your architecture choice is justified by benchmarks. Then move to PhysicsNeMo for the production run.

---

## Key Features

### 📦 Unified Physical Data (UPD)
A standardized container for physical samples across the entire pipeline:
- Physical state (grids, meshes, graphs)
- Geometry (CAD / mesh)
- Governing equations, ICs, BCs, forcings
- Units, regimes, metadata and provenance

Consistent across data loading, training, validation, and inference.

### 🌍 Data Pipeline (`pinneaple_data`)
- Zarr-backed datasets with lazy loading, sharding, LRU cache and adaptive prefetch
- Residual-based and variance-based active learning samplers
- Physical validation and schema enforcement
- NASA / scientific-ready data adapters

### 📐 Geometry & Mesh (`pinneaple_geom`)
- CSG primitives (rectangle, sphere, cylinder, L-shape, annulus)
- CAD generation (CadQuery) and STL / mesh I/O (trimesh, meshio)
- Mesh repair, remeshing, importance sampling
- OpenFOAM / FEniCS adapters

### 🧠 Model Zoo (`pinneaple_models`)
130+ architectures organized by family:

| Family | Models |
|--------|--------|
| PINNs | Vanilla, XPINN, VPINN, XTFC, Inverse PINN, PIELM |
| Neural Operators | FNO, DeepONet, PINO, GNO, UNO |
| Graph Neural Nets | MeshGraphNet, GNN-ODE, equivariant GNNs |
| Transformers | Informer, FEDformer, Autoformer, TFT |
| Reduced Order | POD, DMD, HAVOK, Operator Inference |
| Autoencoders | VAE, Koopman AE, β-VAE |
| Classical | ARIMA, Exponential Smoothing, Kalman, ESN |

### 🧮 Physics Loss Factory (`pinneaple_pinn`)
- SymPy-based symbolic PDE definitions → auto-diff residuals
- DoMINO domain decomposition
- Causal training, time marching
- Hard and soft boundary condition enforcement

### ⚙️ Solvers (`pinneaple_solvers`)
FEM, FDM, FVM, Spectral, SPH, Lattice Boltzmann — for data generation and validation.

### 🏗️ Training (`pinneaple_train`)
- DDP and FSDP distributed training
- Gradient accumulation and AMP
- Deterministic, auditable runs with checkpointing
- Physics-aware loss integration

### 📊 Uncertainty & Validation (`pinneaple_uq`, `pinneaple_validate`)
- MC Dropout, Deep Ensemble, Conformal Prediction
- Conservation law verification, BC consistency checks, residual maps

### 🔁 Transfer & Meta-Learning (`pinneaple_transfer`, `pinneaple_meta`)
- Fine-tune pretrained PINNs to new physical regimes
- MAML and Reptile for fast adaptation across PDE families

### 🛰️ Digital Twins (`pinneaple_digital_twin`)
- Live state estimation with Ensemble Kalman Filter
- Anomaly detection from sensor streams
- MQTT / Kafka adapters

### 🚢 Deployment (`pinneaple_serve`, `pinneaple_export`)
- FastAPI REST inference server
- ONNX and TorchScript export with latency benchmarks
- JAX backend with JIT compilation

### 🤖 Problem Design (`pinneaple_problemdesign`)
- Natural language → PDE specification via LLM
- AutoPINNBuilder: spec → runnable model in one call

### 🏆 Benchmarking (`pinneaple_arena`)
- YAML-driven experiment suites
- Leaderboard and Pareto-front visualizations
- Side-by-side comparisons with PhysicsNeMo

---

## Installation

```bash
git clone https://github.com/barrosyan/PINNeAPPle.git
cd PINNeAPPle
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows:     .venv\Scripts\Activate.ps1
pip install -e .
```

Optional extras:
```bash
pip install -e ".[geom]"     # trimesh, meshio, gmsh
pip install -e ".[cad]"      # CadQuery
pip install -e ".[solvers]"  # numba, FEniCS
pip install -e ".[pinn]"     # sympy
pip install -e ".[serve]"    # FastAPI, uvicorn
pip install -e ".[export]"   # ONNX, onnxruntime
pip install -e ".[uq]"       # scipy
pip install -e ".[all]"      # everything
```

Verify:
```python
from pinneaple_models.register_all import register_all
from pinneaple_models.registry import ModelRegistry
register_all()
print("Models available:", len(ModelRegistry.list()))
```

---

## 5-Minute Quick Start

**Tier 1 — Run your first PINN (harmonic oscillator):**
```bash
python examples/getting_started/01_harmonic_oscillator.py
```

**Tier 2 — Run a benchmark comparison:**
```bash
python templates/28_arena_benchmark.py
```

**Tier 3 — Export a trained model:**
```bash
python templates/23_model_export.py
```

**See PhysicsNeMo parity:**
```bash
python examples/vs_physicsnemo/01_pinneaple_uq_digital_twin/example.py
```

---

## Repository Structure

```
PINNeAPPle/
├── pinneaple/                  # Main package (lazy loader)
├── pinneaple_environment/      # Problem presets (Burgers, NS, Heat, Elasticity, …)
├── pinneaple_models/           # 130+ model architectures
├── pinneaple_pinn/             # Physics loss factory (SymPy → autograd)
├── pinneaple_data/             # UPD, Zarr pipeline, active learning
├── pinneaple_geom/             # Geometry, mesh, CSG, CAD
├── pinneaple_solvers/          # FEM, FDM, FVM, Spectral, LBM, SPH
├── pinneaple_train/            # Trainer, DDP, AMP, metrics
├── pinneaple_uq/               # Uncertainty quantification
├── pinneaple_transfer/         # Transfer learning & fine-tuning
├── pinneaple_meta/             # MAML, Reptile meta-learning
├── pinneaple_validate/         # Physics consistency validation
├── pinneaple_inverse/          # Inverse problems, EKI, sensitivity
├── pinneaple_digital_twin/     # Runtime, assimilation, anomaly detection
├── pinneaple_arena/            # Benchmark runner, leaderboard
├── pinneaple_serve/            # FastAPI REST serving
├── pinneaple_export/           # ONNX, TorchScript export
├── pinneaple_backend/          # JAX backend
├── pinneaple_design_opt/       # Adjoint, shape optimization
├── pinneaple_dynamics/         # MPM, rigid body, particles
├── pinneaple_timeseries/       # Classical TS models
├── pinneaple_worldmodel/       # World model training
├── pinneaple_problemdesign/    # LLM-assisted problem design
├── pinneaple_researcher/       # Research agent, knowledge base
│
├── templates/                  # 35 ready-to-run template scripts (Tier 1→3)
├── examples/
│   ├── getting_started/        # 10 simple physics problems (Tier 1)
│   ├── pinneaple_arena/        # Benchmarks and end-to-end pipelines
│   ├── pinneaple_pinn/         # PINN variants and compiler demos
│   ├── pinneaple_models/       # Architecture showcase
│   ├── pinneaple_geom/         # Geometry and mesh examples
│   ├── pinneaple_data/         # Data pipeline examples
│   ├── pinneaple_solvers/      # Classical solver demos
│   ├── pinneaple_timeseries/   # Time series forecasting
│   ├── vs_physicsnemo/         # Direct PhysicsNeMo comparisons ← start here for migration
│   └── use_cases/              # Industry use cases (aerodynamics, mechanics, CFD)
│
├── configs/                    # YAML experiment configurations
├── docs/                       # API reference (MkDocs)
└── tests/                      # Test suite
```

---

## Who Is PINNeAPPle For?

| Profile | Use PINNeAPPle to… |
|---|---|
| Physics / engineering student | Learn Physics AI hands-on with real equations |
| Research scientist | Prototype and benchmark new PINN / operator approaches |
| CFD / FEA engineer | Build surrogates and inverse solvers for simulation acceleration |
| ML engineer | Explore physics-constrained learning before committing to a production stack |
| Senior engineer / architect | Validate approaches and plan PhysicsNeMo migration |

---

## Contributing

We welcome contributions in:
- New presets, datasets and adapters
- Model implementations and benchmarks
- Documentation and tutorials
- PhysicsNeMo comparison examples

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use PINNeAPPle in research, please cite via `CITATION.cff`.

---

## Philosophy

> PINNeAPPle is not a competitor to PhysicsNeMo. It is the path to it.

Physics AI has a steep learning curve. PINNeAPPle lowers it — so that when you arrive at PhysicsNeMo, you arrive knowing exactly what you need, why you need it, and how to use it.

**Status:** Active development · Feedback and collaboration welcome.
