# PINNeAPPle — Library Overview

> **P**hysics-**I**nformed **N**eural **Ne**tworks **A**pplication & **P**hysics **P**roblem **le**arning
>
> An end-to-end framework for physics-informed neural networks (PINNs), neural operators, digital twins, and physics-driven design optimization in engineering.

---

## Table of Contents

1. [General Architecture](#1-general-architecture)
2. [pinneapple\_physics — Physical Problem Definition](#2-pinneapple_physics--physical-problem-definition)
3. [pinneapple\_neural — Architectures and Training](#3-pinneapple_neural--architectures-and-training)
4. [pinneapple\_design — Geometry and Design Optimization](#4-pinneapple_design--geometry-and-design-optimization)
5. [pinneapple\_simulation — Numerical Simulation and Particles](#5-pinneapple_simulation--numerical-simulation-and-particles)
6. [pinneapple\_analysis — Validation, Inversion, and Uncertainty](#6-pinneapple_analysis--validation-inversion-and-uncertainty)
7. [pinneapple\_adaptation — Transfer Learning and Meta-Learning](#7-pinneapple_adaptation--transfer-learning-and-meta-learning)
8. [pinneapple\_systems — Coupled Systems and Digital Twins](#8-pinneapple_systems--coupled-systems-and-digital-twins)
9. [pinneapple\_tools — Visualization, Export, and Benchmarking](#9-pinneapple_tools--visualization-export-and-benchmarking)
10. [pinneapple\_data — Collocation and Active Learning](#10-pinneapple_data--collocation-and-active-learning)
11. [pinneapple\_pdb — Physics Database](#11-pinneapple_pdb--physics-database)
12. [pinneapple\_problemdesign — NLP → PDE Agent](#12-pinneapple_problemdesign--nlp--pde-agent)
13. [Typical Usage Flows](#13-typical-usage-flows)
14. [Supported Physics Domains](#14-supported-physics-domains)
15. [Available Examples](#15-available-examples)

---

## 1. General Architecture

PINNeAPPle is organized into **independent modules** that integrate through well-defined interfaces. Each module can be used in isolation or in combination.

```
pinneapple_physics      ← Defines WHAT to solve (PDEs, BCs, domain)
       ↓
pinneapple_data         ← Defines WHERE to sample (collocation, active learning)
       ↓
pinneapple_simulation   ← Generates reference DATA (numerical solvers, ext. tools)
       ↓
pinneapple_neural       ← Defines HOW to solve (architecture, training)
       ↓
pinneapple_analysis     ← Validates and analyzes (UQ, inversion, validation)
       ↓
pinneapple_design       ← Optimizes geometry and parameters
pinneapple_adaptation   ← Transfer learning and meta-learning
pinneapple_systems      ← Coupled systems, time series, digital twins
pinneapple_tools        ← Visualization, export, benchmarking
```

**4-line quickstart:**

```python
import pinneapple as pp

spec  = pp.get_preset("burgers_1d", nu=0.01)
model = pp.build_model("SIREN", in_dim=2, out_dim=1, hidden_dim=64, n_layers=4)
result = pp.train_model(model, spec.compile_losses(), epochs=5000)
```

---

## 2. `pinneapple_physics` — Physical Problem Definition

**Purpose:** Specify physical problems as structured Python objects — without writing equations by hand. It is the entry point for the entire pipeline.

### 2.1 Problem Specification (`pde_environment`)

#### Core Classes

| Class | Description |
|-------|-------------|
| `ProblemSpec` | Complete specification: dimension, coordinates, fields, PDE, boundary conditions, scales, domain bounds |
| `PDETermSpec` | Equation descriptor: `kind` (PDE type), fields involved, numerical parameters |
| `ScaleSpec` | Nondimensionalization via length scale `L`, velocity `U`, and diffusivity `alpha` |
| `ConditionSpec` | Generic constraint (Dirichlet, Neumann, Robin, IC, supervised data) |

#### Boundary Condition Constructors

```python
# Simple dict form (field → value)
bc = DirichletBC({"u": 0.0, "v": 0.0})

# Full form (with spatial selector)
bc = DirichletBC(
    "inlet",
    fields=("u", "v"),
    selector_type="callable",
    selector=lambda X, ctx: X[:, 0] < 1e-6,
    value_fn=lambda X, ctx: np.column_stack([U_inf * np.ones(X.shape[0]),
                                              np.zeros(X.shape[0])]),
    weight=10.0,
)
NeumannBC(...)         # prescribed normal flux
RobinBC(...)           # linear combination u + ∂u/∂n = g
InitialCondition(...)  # u(x, t=0) = g(x)
DataConstraint(...)    # supervised loss at measured points
```

#### Registered Presets (41+)

Presets encapsulate the full problem knowledge (PDEs, BCs, scales, domain bounds) in a single call:

```python
from pinneapple_physics import get_preset, list_presets

spec = get_preset("ns_incompressible_2d", Re=200.0)
spec = get_preset("axial_compressor_meanline", num_stages=5, pressure_ratio=3.0)
```

**Academic**

| Preset | Domain | Fields |
|--------|--------|--------|
| `burgers_1d` | (x, t) | u |
| `laplace_2d` | (x, y) | u |
| `poisson_2d` | (x, y) | u |

**CFD / Navier-Stokes**

| Preset | Domain | Fields |
|--------|--------|--------|
| `ns_incompressible_2d` | (x, y) | u, v, p |
| `ns_incompressible_3d` | (x, y, z) | u, v, w, p |
| `lid_driven_cavity_3d` | (x, y, z) | u, v, w, p |
| `channel_flow_3d` | (x, y, z) | u, v, w, p |
| `pipe_flow_3d` | (r, z) | u_r, u_z, p |

**Aerospace**

| Preset | Description |
|--------|-------------|
| `rocket_nozzle_cfd` | Axisymmetric compressible Euler in a convergent-divergent nozzle |
| `rocket_structural` | Rocket casing under internal pressure + thermal gradient |
| `aircraft_wing_aerodynamics` | Simplified 2D RANS around an airfoil |
| `aircraft_wing_structural` | Composite wing spar (plane stress) |

**Turbomachinery** *(TurboDesigner integration)*

| Preset | Dimension | Fields |
|--------|-----------|--------|
| `axial_compressor_meanline` | 1D (s) | T_t, p_t, rho, u, c_theta |
| `axial_compressor_cascade_2d` | 2D (x,y) | rho, u, v, p, T |
| `axial_compressor_stage_3d` | 3D (r,θ,z) | rho, u_r, u_θ, u_z, p, T |

**Automotive, Industrial, Datacenter, Structural, Multidisciplinary**
(automotive thermal/fatigue/aero, furnace, datacenter airflow, elasticity, thermoelasticity, terramechanics, finance, epidemiology, pharmacokinetics — see `list_presets()`)

#### Registering Custom Presets

```python
from pinneapple_physics.pde_environment.presets.registry import register_preset
from pinneapple_physics import ProblemSpec, PDETermSpec, DirichletBC

@register_preset("my_problem")
def my_problem(nu: float = 0.01) -> ProblemSpec:
    coords = ("x", "t")
    fields = ("u",)
    return ProblemSpec(
        name="my_problem",
        dim=2,
        coords=coords,
        fields=fields,
        pde=PDETermSpec(kind="burgers", fields=fields, coords=coords, params={"nu": nu}),
        conditions=(DirichletBC({"u": 0.0}),),
        domain_bounds={"x": (-1.0, 1.0), "t": (0.0, 1.0)},
    )
```

### 2.2 PINN Compiler (`pinn_solver`)

Converts a `ProblemSpec` into loss functions ready for training.

| Function/Class | Description |
|----------------|-------------|
| `compile_problem(spec)` | Compiles ProblemSpec → loss function |
| `LossWeights` | Weights for components: `w_pde`, `w_bc`, `w_ic`, `w_data` |
| `grad()`, `jacobian()`, `divergence()`, `laplacian()` | Autograd operators |
| `Subdomain`, `SubdomainPINN` | Domain decomposition |
| `DoMINO` | Domain Decomposition PINN (DAS + time marching) |

**PDE types supported by the compiler:** `laplace`, `poisson`, `burgers`, `navier_stokes_incompressible`, `heat`, `wave`, `elasticity`, `darcy`, `helmholtz`, `advection`, `reaction_diffusion` — plus custom types via `SymbolicPDE`.

### 2.3 Symbolic PDEs (`symbolic_pde`)

Define equations with SymPy and automatically compile them into PyTorch residuals.

```python
from pinneapple_physics.symbolic_pde import SymbolicPDE, pde_from_sympy
import sympy as sp

x, t, u = sp.symbols("x t u")
pde = pde_from_sympy(sp.diff(u, t) + u * sp.diff(u, x), fields=["u"], coords=["x", "t"])
```

| Class | Description |
|-------|-------------|
| `SymbolicPDE` | Compiles SymPy expression → autograd residual |
| `HardBC` | Satisfies BCs exactly via distance-function ansatz |
| `PeriodicBC` | Periodic boundary conditions |
| `auto_residual()` | Automatic residual derivation |

### 2.4 Turbulence (RANS)

| Class | Description |
|-------|-------------|
| `KOmegaSSTResiduals` | Full k-ω SST model |
| `SpalartAllmarasResiduals` | One-equation Spalart-Allmaras model |
| `get_rans_preset()` | Quick preset for RANS problems |

### 2.5 PDE Identification

```python
from pinneapple_physics import identify, define_problem

spec = identify("incompressible flow in a channel with Re=500")
spec = define_problem("heat conduction in a plate with a source term")
```

---

## 3. `pinneapple_neural` — Architectures and Training

**Purpose:** Instantiate, train, and run inference with 100+ neural network architectures for computational physics.

### 3.1 Model Registry (`architectures`)

All models are accessible by name via `ModelRegistry`:

```python
from pinneapple_neural import build_model

model = build_model("SIREN", in_dim=3, out_dim=5, hidden_dim=256, n_layers=6)
model = build_model("FNO", in_channels=1, out_channels=1, modes=16, width=64)
model = build_model("DeepONet", branch_dim=100, trunk_dim=2, hidden=128, layers=4)
```

#### PINN Family

| Model | Description |
|-------|-------------|
| `VanillaPINN` | MLP with standard Tanh/ReLU/GELU activations |
| `SIREN` | Sinusoidal representation networks (captures high frequencies) |
| `ModifiedMLP` | MLP with Fourier feature embedding for PINNs |
| `HashGridMLP` | MLP with hash grid encoding (accelerated training) |
| `InversePINN` | PINN with trainable PDE parameters (inverse problems) |
| `VPINN` | Variational PINN |
| `XPINN` | Extended PINN with domain decomposition |
| `PINNsFormer` | Transformer-based PINN |
| `PIELM` | Physics-Informed Extreme Learning Machine |

#### Neural Operators

| Model | Description |
|-------|-------------|
| `FourierNeuralOperator` (FNO) | Spectral learning via FFT |
| `DeepONet` | Universal branch-trunk operator |
| `PINO` | Physics-Informed Neural Operator |
| `AFNO` | Adaptive Fourier Neural Operator |
| `GraphNeuralOperator` (GNO) | Operator on mesh graphs |

#### Continuous Models

| Model | Description |
|-------|-------------|
| `NeuralODE` | ODE with Runge-Kutta 4 integration |
| `NeuralSDE` | Stochastic differential equations |
| `NeuralCDE` | Controlled differential equations |
| `HamiltonianNN` | Preserves Hamiltonian structure |
| `SymplecticODE` | Preserves symplectic structure |
| `LatentODE` | ODE in variational latent space |
| `BayesianRNN` | Bayesian recurrent network |

#### Autoencoders / ROM

| Model | Description |
|-------|-------------|
| `VAE` | Variational Autoencoder |
| `KoopmanAE` | Autoencoder with Koopman operator |
| `AE_ROM_Hybrid` | ROM + neural correction |
| `DeepUQROM` | ROM with uncertainty quantification |

#### Graph Neural Networks

| Model | Description |
|-------|-------------|
| `MeshGraphNet` | Message passing on unstructured meshes |
| `EquivariantGNN` | Equivariant to rotations and translations |
| `GNN_ODE` | GNN + ODE integration |

#### Time-Series Transformers

`Transformer`, `Informer`, `Autoformer`, `FedFormer`, `TimesNet`, `TFT`

### 3.2 Training (`trainer`)

#### Available Trainers

| Class | Description |
|-------|-------------|
| `Trainer` | Unified trainer: metrics, callbacks, logging, AMP |
| `TwoPhaseTrainer` | Phase 1: physics; Phase 2: supervised |
| `CausalPINNTrainer` | Causal PINN with temporal curriculum |
| `TimeMarchingTrainer` | Stage-by-stage time marching |
| `DDPPINNTrainer` | Distributed Data Parallel (multi-GPU) |
| `GradAccumTrainer` | Gradient accumulation for large batches |

#### Loss Balancing

| Class | Strategy |
|-------|----------|
| `SelfAdaptiveWeights` | Learns weights automatically |
| `GradNormBalancer` | Balances by gradient norm |
| `NTKWeightBalancer` | Based on Neural Tangent Kernel |
| `LossRatioBalancer` | Maintains ratio between components |
| `WeightScheduler` | Dynamic weight schedule |

#### Training Utilities

```python
from pinneapple_neural.trainer import Trainer, TrainConfig

cfg = TrainConfig(
    epochs=10_000,
    lr=1e-3,
    device="cuda",
    amp=True,        # mixed precision
    log_every=500,
)
trainer = Trainer(model, loss_fn, cfg)
trainer.fit(train_loader)
```

**HPC infrastructure:** supports `FSDP`, `DeepSpeed ZeRO`, `CUDA Graphs`, gradient compression (`PowerSGD`, `TopK`), SLURM scripts, and integrated profiling.

### 3.3 Inference (`predictor`)

```python
from pinneapple_neural import predict, build_model

result = predict(model, x_test, device="cuda", batch_size=10_000)

# Evaluation on a structured grid
from pinneapple_neural.predictor import infer_on_grid_2d
field = infer_on_grid_2d(model, x_range=(0,1), y_range=(0,1), nx=256, ny=256)
```

---

## 4. `pinneapple_design` — Geometry and Design Optimization

**Purpose:** Define geometric domains (via SDF or meshes), sample collocation points, and run physics-driven design optimization.

### 4.1 Geometry (`geometry`)

#### 2D SDF Primitives

```python
from pinneapple_design.geometry import circle, rectangle, ellipse, annulus

d = circle(center=(0.5, 0.5), radius=0.3)
d = rectangle(center=(0.5, 0.5), half_extents=(0.4, 0.2))
d = annulus(center=(0,0), inner_r=0.2, outer_r=0.5)
```

#### CSG (Boolean) Operations

```python
from pinneapple_design.geometry import sdf_union, sdf_difference, sdf_intersection

shape = sdf_difference(rectangle(...), circle(...))    # subtraction
shape = sdf_smooth_union(circle1, circle2, k=0.05)    # smooth union
```

#### 3D SDF Primitives

`sdf3d_sphere`, `sdf3d_box`, `sdf3d_cylinder`, `sdf3d_torus`, `sdf3d_capsule`

#### 2D Physics Domains

| Domain | Description |
|--------|-------------|
| `ChannelDomain2D` | Rectangular channel with inlet/outlet/walls |
| `ChannelWithObstacleDomain2D` | Channel with cylindrical obstacle |
| `LidDrivenCavityDomain2D` | Cavity with sliding lid |
| `LShapeDomain2D` | L-shaped domain (stress concentrator) |
| `AnnularDomain2D` | Annular domain |
| `TJunctionDomain2D` | T-junction |
| `SDFDomain2D` | Generic domain from any SDF |

```python
from pinneapple_design.geometry import get_domain

domain = get_domain("channel_2d", length=4.0, height=1.0, obstacle_radius=0.1)
pts_interior = domain.sample_interior(n=50_000)
pts_boundary = domain.sample_boundary(n=10_000)
```

#### 3D Physics Domains

`LidDrivenCavityDomain3D`, `ChannelDomain3D`, `PipeFlowDomain3D`

#### Mesh and 3D Collocation

| Class | Description |
|-------|-------------|
| `MeshCollocator` | Samples interior/boundary of a 3D mesh |
| `STLDomainBatchBuilder` | STL → collocation batch pipeline |
| `mesh_rectangle_structured()` | Structured rectangular mesh |
| `mesh_sdf_2d()` | SDF mesh via marching squares |
| `RBFInterpolator` | Interpolation on point clouds |
| `naca_parametric()` | NACA 4-digit airfoil generation |

### 4.2 Design Optimization (`design_optimizer`)

Complete pipeline for physics-driven shape and parameter optimization.

#### Objectives

| Class | Objective |
|-------|-----------|
| `DragObjective` | Minimize drag (CFD) |
| `ThermalEfficiencyObjective` | Maximize thermal efficiency |
| `StructuralObjective` | Minimize stress/strain |
| `WeightMinimizationObjective` | Minimize mass |
| `CompositeObjective` | Weighted combination of objectives |

#### Optimizers

| Class | Method |
|-------|--------|
| `GradientDesignOptimizer` | Gradient via continuous adjoint |
| `BayesianDesignOptimizer` | Gaussian Process + Expected Improvement |
| `EvolutionaryDesignOptimizer` | Genetic algorithm |

#### Pareto and Multi-Objective

```python
from pinneapple_design.design_optimizer import compute_pareto_front, ParetoFront

front = compute_pareto_front(objectives_matrix)  # shape (N, n_obj)
front.plot_2d(labels=["Drag", "Weight"])
```

#### Optimization Loop

```python
from pinneapple_design.design_optimizer import DesignOptLoop, DesignOptConfig

loop = DesignOptLoop(
    surrogate=trained_pinn,
    objective=DragObjective(),
    param_space=ParamSpace(bounds={"thickness": (0.05, 0.3)}),
    config=DesignOptConfig(method="bayesian", n_trials=200),
)
best = loop.run()
```

---

## 5. `pinneapple_simulation` — Numerical Simulation and Particles

**Purpose:** Generate reference data for PINN training — using built-in solvers (FDM/FEM) or bridges to external tools (OpenFOAM, FEniCS, MATLAB, etc.).

### 5.1 Numerical Solvers (`numerical_solvers`)

#### Built-in 3D Solvers (FDM)

| Class | Equation |
|-------|----------|
| `HeatConduction3D` | Thermal diffusion (steady/transient) |
| `NavierStokes3D` | 3D Navier-Stokes (SIMPLE/SIMPLER) |
| `ElasticWave3D` | Elastic wave equation |
| `LidDrivenCavitySolver3D` | Lid-driven cavity — 3D NS |
| `ChannelFlowSolver3D` | 3D channel flow |

```python
from pinneapple_simulation import simulate, generate_pinn_dataset

output = simulate("heat_3d", nx=64, ny=64, nz=64, kappa=0.1, t_end=1.0)
dataset = generate_pinn_dataset("ns_3d", n_samples=100)
```

#### PINN Dataset Generation

```python
dataset = generate_pinn_dataset(
    scenario="heat_3d",
    n_samples=200,
    param_ranges={"kappa": (0.01, 1.0)},
)
```

### 5.2 Particle Dynamics (`particle_dynamics`)

All simulators are implemented in pure PyTorch (differentiable, autograd-compatible).

| Class | Method | Application |
|-------|--------|-------------|
| `RigidBodySystem` | Symplectic Euler | Multi-body 2D/3D |
| `MPMSimulator` | MLS-MPM | Elastic solids, viscous fluids, snow, sand |
| `SPHParticles` | SPH | Free-surface flows |
| `ParticleSystem` | Generic | Customizable particle system |

**MPM plasticity:** Drucker-Prager (granular/snow).

### 5.3 External Solver Bridges (`external_solvers`)

All bridges are optional — the library imports cleanly even without the external tool installed.

#### OpenFOAM

```python
from pinneapple_simulation.external_solvers import (
    OpenFOAMCaseTemplate, run_openfoam_case, openfoam_case_to_upd
)
template = OpenFOAMCaseTemplate.from_scenario("channel_flow")
run_openfoam_case(case_dir, cfg)
sample = openfoam_case_to_upd(case_dir)
```

#### FEniCS

```python
from pinneapple_simulation.external_solvers import FEniCSConfig, FEniCSWorkflow

wf = FEniCSWorkflow(FEniCSConfig(pde="heat_equation_steady", domain={...}, bcs=[...]))
samples = wf.sweep({"k": [0.5, 1.0, 2.0, 5.0]})
```

#### TurboDesigner *(integrated bridge)*

```python
from pinneapple_simulation.external_solvers.turbodesigner import (
    TurboDesignerConfig, TurboDesignerWorkflow
)
cfg = TurboDesignerConfig(pressure_ratio=3.0, num_stages=5, rpm=10_000)
wf  = TurboDesignerWorkflow(cfg)
data = wf.solve()                                         # single operating point
samples = wf.sweep({"pressure_ratio": [2.0, 3.0, 4.0]}, as_upd=True)
```

> Without `turbodesigner` installed, the built-in analytical solver is used automatically.

#### Other Bridges

| Bridge | Required package |
|--------|------------------|
| MATLAB | `matlab.engine` |
| OpenModelica / FMU | `fmpy`, `OMPython` |
| MuJoCo | `mujoco >= 3.0` |
| Genesis AI | `genesis-world` |

---

## 6. `pinneapple_analysis` — Validation, Inversion, and Uncertainty

**Purpose:** Evaluate the reliability of trained models: uncertainty quantification, physical validation, and parameter extraction via inversion.

### 6.1 Uncertainty Quantification (`uncertainty`)

| Method | Class | Type |
|--------|-------|------|
| MC Dropout | `MCDropoutWrapper` | Epistemic |
| Ensemble | `EnsembleUQ` | Epistemic |
| Aleatoric variance | `AleatoricHead` | Aleatoric |
| Conformal prediction | `ConformalPredictor` | Guaranteed coverage |
| Quantile regression | `QuantileHead` | Prediction intervals |

```python
from pinneapple_analysis import analyze_model

report = analyze_model(
    model, spec, x_test,
    uq_method="ensemble",    # "mc_dropout" | "ensemble" | "conformal"
    run_validate=True,
    run_uq=True,
)
print(report.uncertainty.coverage_90)
```

### 6.2 Physics Validation (`validation`)

Automatically checks whether the model respects conservation laws, BCs, and symmetries:

```python
from pinneapple_analysis.validation import PhysicsValidator

validator = PhysicsValidator(model, spec)
report = validator.run_all()
# Checks: ConservationCheck, BoundaryCheck, SymmetryCheck
print(report.summary())
```

### 6.3 Inverse Problems (`inverse_problems`)

#### Parameter Identification (PDE Parameter Estimation)

```python
from pinneapple_analysis import invert

result = invert(
    model=pinn,
    y_obs=sensor_readings,
    sensor_locs=sensor_positions,
    noise_std=0.01,
    lambda_reg=1e-4,
    method="eki",        # "gradient" | "eki" | "sindy"
    n_iters=100,
)
print(result.params_estimated)
```

#### Noise Models

| Class | When to use |
|-------|-------------|
| `GaussianMisfit` | Standard Gaussian noise |
| `HuberMisfit` | Robust to outliers |
| `HeteroscedasticMisfit` | Non-homogeneous variance |

#### Regularization

| Class | Penalty |
|-------|---------|
| `TikhonovRegularizer` | L2 — smoothness |
| `SparsityRegularizer` | L1 — sparsity |
| `TotalVariationRegularizer` | TV — discontinuities |
| `LCurveSelector` | Automatic λ selection |

#### Ensemble Kalman Inversion (EKI)

```python
from pinneapple_analysis.inverse_problems import EnsembleKalmanInversion, EKIConfig

eki = EnsembleKalmanInversion(
    forward_model=pinn,
    obs_operator=PointObsOperator(sensor_locs),
    config=EKIConfig(n_ensemble=200, n_iters=50),
)
result = eki.run(y_obs=observations)
```

#### Equation Discovery (SINDy)

```python
from pinneapple_analysis.inverse_problems import SINDyIdentifier

sindy = SINDyIdentifier(library=CandidateLibrary(poly_order=3))
result = sindy.fit(X_data, dX_data)
print(result.equations)
```

---

## 7. `pinneapple_adaptation` — Transfer Learning and Meta-Learning

**Purpose:** Reuse trained models for new physical domains with minimal additional data.

### 7.1 Transfer Learning (`transfer_learning`)

```python
from pinneapple_adaptation.transfer_learning import TransferTrainer, TransferConfig

# Fine-tune only the last 2 layers
cfg = TransferConfig(
    strategy="last_layers",
    epochs=500,
    finetune_lr=1e-4,
    layer_freezing={"freeze_prefix": "encoder"},
)
new_model = TransferTrainer(pretrained, new_spec, cfg).finetune(new_data)
```

| Function | Description |
|----------|-------------|
| `freeze_layers()` | Freeze layers by name/prefix |
| `layer_lr_groups()` | Discriminative learning rates per layer |
| `PhysicsTransferAdapter` | Domain adaptation with MMD loss |
| `ParametricFamilyTransfer` | Interpolation between parametric variants |

### 7.2 Meta-Learning (`meta_learning`)

Trains a model with fast adaptation capability (few-shot) for PDE families.

```python
from pinneapple_adaptation.meta_learning import MAMLTrainer, MAMLConfig, PDETaskSampler

sampler = PDETaskSampler(family="navier_stokes", param_ranges={"Re": (50, 2000)})
meta_model = MAMLTrainer(model, sampler, MAMLConfig(inner_steps=5, meta_lr=1e-3)).train()

# Fast adaptation to a new Re in 5 gradient steps
adapted = meta_adapt(meta_model, new_task_data, n_steps=5)
```

| Algorithm | Class |
|-----------|-------|
| MAML | `MAMLTrainer` |
| Reptile | `ReptileTrainer` |

---

## 8. `pinneapple_systems` — Coupled Systems and Digital Twins

**Purpose:** Model multi-component systems: time series of physical signals, co-simulation of multiple models, and live digital twins.

### 8.1 Time Series (`time_series`)

#### Available Models

**Baseline:** `NaiveForecaster`, `SeasonalNaiveForecaster`, `DriftForecaster`

**Machine Learning:** `XGBoostForecaster`, `LightGBMForecaster`, `RandomForestForecaster`, `GPRForecaster`

**Deep Learning:**

| Model | Architecture |
|-------|-------------|
| `LSTMForecaster` | LSTM encoder-decoder |
| `GRUForecaster` | GRU |
| `NBeats` | N-BEATS (interpretable) |
| `TCNForecaster` | Temporal Convolutional Network |
| `TFTForecaster` | Temporal Fusion Transformer |
| `FFTForecaster` | FFT + NN |
| `HHTNNForecaster` | Hilbert-Huang + NN |

```python
from pinneapple_systems import forecast

predictions = forecast(
    data=sensor_df,
    horizon=24,
    model="TFT",
    input_len=168,
)
```

#### Analysis and Visualization

`plot_trend`, `power_spectrum`, `plot_acf_pacf`, `stationarity_report`, `animate_rolling_forecast`

#### Backtesting

```python
from pinneapple_systems.time_series import BacktestRunner, BacktestConfig

runner = BacktestRunner(
    model=forecaster,
    config=BacktestConfig(n_splits=5, strategy="expanding"),
)
result = runner.run(series)
```

### 8.2 Co-Simulation (`cosimulation`)

Graph-based co-simulation engine — connects PINNs, analytical models, solvers, and time-series forecasters.

#### Node Types

| Node | Description |
|------|-------------|
| `PINNNode` | Trained PINN node |
| `AnalyticalNode` | Python analytical function |
| `TimeSeriesCoSimNode` | Time-series forecaster |
| `SymbolicPDENode` | Symbolic PDE |
| `BlackBoxNode` | Generic black-box |

```python
from pinneapple_systems.cosimulation import CoSimGraph, CoSimEngine

graph = CoSimGraph()
graph.add_node("cfd", PINNNode(ns_pinn))
graph.add_node("thermal", AnalyticalNode(heat_fn))
graph.add_connection(Connection("cfd.velocity", "thermal.u"))

engine = CoSimEngine(graph)
trajectory = engine.run(t_span=(0, 10), dt=0.01)
```

### 8.3 Digital Twin (`digital_twin`)

Real-time sensor data fusion with a PINN model + Kalman filters.

```python
from pinneapple_systems.digital_twin import build_digital_twin, DigitalTwinConfig

twin = build_digital_twin(
    model=pinn,
    spec=problem_spec,
    config=DigitalTwinConfig(
        assimilation="enkf",        # "ekf" | "enkf"
        anomaly_detector="zscore",  # "threshold" | "zscore" | "mahalanobis"
        stream="mqtt",              # "mqtt" | "kafka" | "http" | "file"
    ),
)
twin.start()   # starts the assimilation loop
```

#### Data Streams

| Stream | Protocol |
|--------|----------|
| `MQTTStream` | IoT / MQTT broker |
| `KafkaStream` | Apache Kafka |
| `HTTPPollStream` | REST API polling |
| `FileWatchStream` | File on disk |
| `MockStream` | Synthetic data (development) |

---

## 9. `pinneapple_tools` — Visualization, Export, and Benchmarking

**Purpose:** Visualize physical fields, export models to production, and systematically compare models.

### 9.1 Visualization (`visualization`)

```python
from pinneapple_tools.visualization import plot_scalar, plot_streamlines, animate_scalar_field

plot_scalar(x, y, field, cmap="coolwarm", title="Pressure")
plot_streamlines(x, y, u, v, density=2.0)
animate_scalar_field(field_sequence, dt=0.1, save_as="evolution.gif")
```

#### CFD-Specific Visualizations

| Function | Output |
|----------|--------|
| `plot_vorticity()` | 2D vorticity field |
| `plot_q_criterion_2d/3d()` | Q-criterion for vortex identification |
| `plot_lambda2_3d()` | Lambda-2 criterion |
| `plot_pde_residual()` | PDE residual per point |
| `plot_collocation()` | Collocation points colored by loss |
| `plot_loss_history()` | Multi-component convergence curve |

### 9.2 Model Export (`model_export`)

```python
from pinneapple_tools import export_model

export_model(model, "solver.onnx",  fmt="onnx",        input_shape=(1, 3))
export_model(model, "solver.pt",    fmt="torchscript")
export_model(model, "outputs.csv",  fmt="csv",          x_test=x)
export_model(model, "outputs.npz",  fmt="npz",          x_test=x)
```

### 9.3 Benchmarking — Arena (`benchmark_suite`)

YAML-driven system for comparing multiple models across multiple tasks:

```yaml
# configs/arena/burgers_benchmark.yaml
tasks:
  - burgers_1d
  - ns_incompressible_2d
models:
  - SIREN
  - FNO
  - DeepONet
metrics: [l2_error, max_error, training_time]
```

```python
from pinneapple_tools.benchmark_suite import Arena

runner = Arena.from_yaml("configs/arena/burgers_benchmark.yaml")
results = runner.run_all()
results.leaderboard()
results.plot_comparison()
```

---

## 10. `pinneapple_data` — Collocation and Active Learning

**Purpose:** Manage collocation points, physical datasets, and adaptive sampling.

### 10.1 Collocation

```python
from pinneapple_data import CollocationSampler, CollocationConfig

sampler = CollocationSampler(
    domain=my_domain,
    config=CollocationConfig(
        n_interior=50_000,
        n_boundary=10_000,
        method="latin_hypercube",   # "random" | "sobol" | "latin_hypercube"
    )
)
batch = sampler.sample()
```

### 10.2 Active Learning

Concentrates collocation points where the network has the highest residual or uncertainty:

| Strategy | Class | Criterion |
|----------|-------|-----------|
| Residual-based | `ResidualBasedAL` | Points with highest PDE residual |
| Variance-based | `VarianceBasedAL` | Points with highest variance (MC Dropout) |
| Combined | `CombinedAL` | Weighted combination |

```python
from pinneapple_data import AdaptiveCollocationTrainer

trainer = AdaptiveCollocationTrainer(
    model=pinn,
    loss_fn=losses,
    al_strategy="residual",
    refine_every=500,   # epochs between refinements
    n_add=1000,         # points added per refinement
)
trainer.fit(epochs=10_000)
```

### 10.3 UPD Format (Universal Physical Data)

Internal format for physical samples with metadata:

```python
from pinneapple_data.physical_sample import PhysicalSample
import torch

sample = PhysicalSample(
    fields={"u": torch.tensor([...]), "p": torch.tensor([...])},
    coords={"x": x_array, "y": y_array},
    meta={"units": {"u": "m/s", "p": "Pa"}, "source": "openfoam"},
)
```

### 10.4 Dataset Registry

```python
from pinneapple_data import load_dataset, list_datasets

print(list_datasets())
ds = load_dataset("cylinder_wake_re100")
```

---

## 11. `pinneapple_pdb` — Physics Database

**Purpose:** Build and query physical datasets from external sources (NASA, ECMWF, satellites, etc.) with a standardized schema.

```python
from pinneapple_pdb import PhysicalDatasetBuilder, HubQuery

builder = PhysicalDatasetBuilder()
query = HubQuery(
    variable=VariableSelection(names=["temperature", "pressure"]),
    space_time=SpaceTime(lat=(-90, 90), lon=(-180, 180), time=("2020-01", "2020-12")),
)
dataset = builder.build(query)
```

---

## 12. `pinneapple_problemdesign` — NLP → PDE Agent

**Purpose:** Convert natural language descriptions of physical problems into complete `ProblemSpec` objects, with automatic PDE identification, a solution plan, and generated code.

```python
from pinneapple_problemdesign import DesignAgent

agent = DesignAgent(provider="gemini")   # or a custom provider

report = agent.design(
    "I want to simulate laminar flow around a cylinder at Re=100 "
    "and extract the drag coefficient."
)

print(report.problem_spec)     # generated ProblemSpec
print(report.plan)             # step-by-step action plan
print(report.gaps)             # identified knowledge gaps
print(report.pinneapple_code)  # ready-to-run Python code
```

---

## 13. Typical Usage Flows

### A. Simple PINN (solve a PDE)

```python
import pinneapple as pp

spec   = pp.get_preset("burgers_1d", nu=0.01)
model  = pp.build_model("SIREN", in_dim=2, out_dim=1, hidden_dim=128, n_layers=6)
result = pp.train_model(model, spec.compile_losses(), epochs=10_000)
pp.plot(model, x_test, field_name="u", dim=1)
```

### B. Data generation + supervised training

```python
from pinneapple_simulation import generate_pinn_dataset
from pinneapple_neural import build_model, train_model

dataset = generate_pinn_dataset("heat_3d", n_samples=200)
model   = build_model("FNO", in_channels=3, out_channels=1)
result  = train_model(model, dataset, epochs=5_000, supervised=True)
```

### C. Inverse problem (parameter identification)

```python
from pinneapple_neural import build_model
from pinneapple_analysis import invert

pinn   = build_model("InversePINN", in_dim=2, out_dim=1, n_params=1)
result = invert(pinn, y_obs=measurements, sensor_locs=coords,
                noise_std=0.01, method="eki", n_iters=100)
print(f"Estimated ν: {result.params_estimated['nu']:.4f}")
```

### D. Design optimization

```python
from pinneapple_design.design_optimizer import DesignOptLoop, BayesianDesignOptimizer
from pinneapple_analysis.uncertainty import EnsembleUQ

surrogate = train_surrogate_pinn(ns_spec, domain)
loop = DesignOptLoop(
    surrogate=EnsembleUQ(surrogate, n_models=5),
    objective=DragObjective(),
    optimizer=BayesianDesignOptimizer(n_trials=100),
)
best_design = loop.run()
```

### E. Live digital twin

```python
from pinneapple_systems.digital_twin import build_digital_twin

twin = build_digital_twin(model=pinn, spec=spec,
                           config=DigitalTwinConfig(stream="mqtt", assimilation="enkf"))
twin.start()
# assimilation loop runs in background — anomalies trigger callbacks
```

### F. Turbomachinery (TurboDesigner + PINN)

```python
import pinneapple as pp
from pinneapple_simulation.external_solvers.turbodesigner import (
    TurboDesignerConfig, TurboDesignerWorkflow
)

# 1. Generate analytical data from TurboDesigner
cfg  = TurboDesignerConfig(pressure_ratio=3.0, num_stages=5, rpm=10_000)
data = TurboDesignerWorkflow(cfg).sweep({"pressure_ratio": [2.0, 3.0, 4.0]}, as_upd=True)

# 2. Train PINN with physics constraints + analytical data as anchor
spec  = pp.get_preset("axial_compressor_meanline", pressure_ratio=3.0)
model = pp.build_model("SIREN", in_dim=1, out_dim=5, hidden_dim=128, n_layers=6)
result = pp.train_model(model, spec.compile_losses(), epochs=15_000,
                         data_samples=data)
```

---

## 14. Supported Physics Domains

| Domain | Types |
|--------|-------|
| **Fluid flow** | Incompressible NS 2D/3D, Stokes, Darcy, Burgers, advection, channel/pipe flow |
| **Compressible** | Axisymmetric Euler, convergent-divergent nozzle, 2D blade cascade |
| **Turbulence** | RANS k-ω SST, Spalart-Allmaras |
| **Thermal** | Steady/transient conduction 2D/3D, thermoelasticity, PCB cooling, heat sinks |
| **Structural** | Linear elasticity 2D/3D, plane stress/strain, Von Mises, torsion, fatigue |
| **Wave** | Wave equation 1D/2D, ultrasound, acoustic Helmholtz |
| **Turbomachinery** | Axial mean-line, 2D cascade, 3D rotating-frame stage |
| **Terramechanics** | Bekker-Wong wheel-soil, rover mobility |
| **Electromagnetic** | Eddy current, Maxwell, EM wave, TM waveguide |
| **Reaction-diffusion** | Generic systems, simplified combustion |
| **Particles** | MPM (solid/fluid/snow/sand), SPH, rigid body |
| **Multi-physics** | Fluid-structure, thermoelastic, magneto-elastic |
| **Finance** | Black-Scholes, Heston PDE |
| **Biology** | SIR epidemiology, drug diffusion, PK compartmental |

---

## 15. Available Examples

The project includes **176 examples** organized into categories:

| Category | Examples | Highlights |
|----------|----------|------------|
| `getting_started/` | 10 | Harmonic oscillator, Lotka-Volterra, van der Pol, Lorenz |
| `pde_environment/` | 4 | Laplace 2D, Burgers 1D, NS 2D channel, Heat 3D STL |
| `pinn_solver/` | 6 | Symbolic loss, DoMINO, parameter inversion |
| `architectures/` | 14 | Registry tour, FNO, DeepONet, MeshGraphNet |
| `data_pipeline/` | 8 | UPD dataloaders, synthetic PDE, sharded zarr |
| `numerical_solvers/` | 11 | FEM, FVM, SPH, LBM, spectral, CAD-CFD |
| `geometry/` | 7 | STL batch, meshes, CSG, curvature sampling |
| `benchmark_suite/` | 17 | Arena YAML, NACA aerodynamics, digital twin |
| `time_series/` | 8 | FNO temporal, backtest, FFT-LSTM, HHT-LSTM |
| `arena_pipelines/` | 9 | Kovasznay NS, multi-model benchmarks |
| `trainer/` | 5 | DDP torchrun, audited training, DataModule |
| `cosimulation/` | 1 | Coupled spring-mass PINN |
| `electrodynamics/` | 6 | Capacitor, dipole, magnetostatics, TM waveguide |
| `visualizations/` | 6 | Cylinder flow, heat 2D, vortices, structural |
| `hpo_experiments/` | 7 | Discover, build KB, reproduce |
| `problem_designer/` | 5 | NLP→PDE API, batch generate |
| `use_cases/` | 2 | Drill pipe torsion, terramechanics rover |

**Run an example:**

```bash
cd examples/getting_started
python harmonic_oscillator.py
```

**Run an Arena benchmark:**

```bash
cd examples/arena_pipelines
python run_arena_yaml.py --config ../../configs/arena/burgers_benchmark.yaml
```

---

*Document generated from the `main` branch source code — May 2026.*
