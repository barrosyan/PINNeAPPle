# pinneaple — Quickstart Guide

## Installation

```bash
pip install -e .
# Optional extras:
pip install paho-mqtt kafka-python tqdm tfc  # digital twins + XtFC
pip install gmsh triangle                    # advanced meshing
```

## 5-minute tour

```python
import pinneaple as pp

# 1. See what's available
pp.info()               # device, GPU count, presets, models
pp.list_presets()       # all pre-defined physics problems
pp.list_models()        # all registered model architectures

# 2. Load a problem
spec = pp.get_preset("burgers_1d", nu=0.01)
pp.quickstart("cpu_heatsink_thermal")   # prints summary + next steps

# 3. Build a model
model = pp.build_model("VanillaPINN", in_dim=2, out_dim=1, hidden=[64,64,64])

# 4. Train with GPU + AMP
from pinneaple_train import Trainer, TrainConfig, best_device
cfg = TrainConfig(epochs=1000, lr=1e-3, device=str(pp.best_device()), amp=True)
trainer = Trainer(model, loss_fn=my_pinn_loss)
result = trainer.fit(train_loader, val_loader, cfg)

# 5. Create a digital twin
dt = pp.build_digital_twin(model, field_names=["u", "v", "p"])
from pinneaple_digital_twin import MockStream
dt.add_stream(MockStream("inlet", ["u"], lambda t: {"u": 1.0 + 0.1*t}))
with dt:
    import time; time.sleep(5)
print(dt.state.fields["u"].mean())
```

## Problem presets by domain

| Domain | Preset name |
|--------|------------|
| Fluid mechanics | `ns_incompressible_2d`, `ns_incompressible_3d` |
| Heat / Poisson | `laplace_2d`, `poisson_2d`, `heat_equation_steady` |
| Wave / Burgers | `burgers_1d`, `wave_equation_1d` |
| Structural | `plane_stress_2d`, `von_mises_2d`, `drill_pipe_torsion` |
| Aerospace | `rocket_nozzle_cfd`, `aircraft_wing_aerodynamics`, `rocket_structural` |
| Automotive | `car_external_aero`, `car_brake_thermal`, `car_suspension_fatigue` |
| PC cooling | `cpu_heatsink_thermal`, `pcb_thermal`, `fan_cooler_cfd` |
| Datacenter | `datacenter_airflow_2d`, `datacenter_server_thermal`, `datacenter_cfd_3d` |
| Furnace | `industrial_furnace_thermal`, `refractory_lining`, `furnace_combustion_zone` |
| Climate | `climate_atmosphere_2d`, `climate_ocean_gyre` |
| Finance | `black_scholes_1d`, `heston_pde_2d` |
| Pharma | `pk_two_compartment`, `drug_diffusion_tissue` |
| Social | `sir_epidemic`, `opinion_dynamics_2d` |

## Model architectures

| Family | Models |
|--------|--------|
| PINNs | `VanillaPINN`, `VPINN`, `XPINN`, `XtFC`, `PINN-LSTM`, `PINNsFormer` |
| Neural Operators | `DeepONet`, `FNO`, `GNO` (Galerkin), `PINO`, `MultiScaleDeepONet` |
| GNNs | `GNN`, `EquivariantGNN`, `SpatioTemporalGNN`, `GraphCast` |
| Autoencoders | `DenseAE`, `VAE`, `KoopmanAE`, `ROM-Hybrid` |
| Continuous | `NeuralODE`, `NeuralSDE`, `LagrangianNN`, `HamiltonianNN` |

## Key workflows

### 1. Pure PINN (physics loss only)
```python
# See examples/03_pinn_burgers_full_pipeline.py
```

### 2. Surrogate with solver data (OpenFOAM / FEniCS)
```python
# See examples/pinneaple_arena/configs/experiment_burgers_1d.yaml
# pp.arena.run_full_pipeline("my_config.yaml")
```

### 3. Digital twin with real sensor data
```python
# See examples/04_digital_twin_flow.py
from pinneaple_digital_twin import MQTTStream
stream = MQTTStream("broker.local", "sensors/inlet", "inlet", ["u","v","p"])
dt.add_stream(stream)
```

### 4. Operator learning with DeepONet
```python
# See examples/05_surrogate_deeponet_multifield.py
```

### 5. Hyperparameter search
```python
from pinneaple_train import run_parallel_sweep, SweepConfig
results = run_parallel_sweep(
    trial_fn,
    SweepConfig({"lr": [1e-3, 1e-4], "hidden": [64, 128]}, n_jobs=4)
)
```

## Extending the library

### Add a new problem preset
```python
from pinneaple_environment.presets.registry import register_preset
from pinneaple_environment.spec import PDETermSpec, ProblemSpec
from pinneaple_environment.conditions import DirichletBC

@register_preset("my_problem")
def my_problem(nu=0.01) -> ProblemSpec:
    return ProblemSpec(
        problem_id="my_problem",
        pde=PDETermSpec(kind="my_pde", params={"nu": nu}),
        fields=("u",),
        coord_names=("x", "t"),
        conditions={"bc": DirichletBC({"u": 0.0})},
        domain_bounds={"x": (0, 1), "t": (0, 1)},
    )
```

### Register a new model
```python
from pinneaple_models.registry import ModelRegistry
@ModelRegistry.register(name="my_model", family="pinn")
class MyModel(nn.Module):
    ...
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
Issues: https://github.com/your-org/pinneaple/issues
