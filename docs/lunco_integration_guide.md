# Integrating PINNeAPPle Terramechanics into LunCoSim `lunco-mobility`

> **Audience:** Rust / Bevy developers working on the LunCoSim project who want to replace or
> augment the raycast-based wheel model in `lunco-mobility` with the physics-accurate
> Bekker-Wong PINN surrogate trained by PINNeAPPle.

---

## Overview

PINNeAPPle's terramechanics pipeline (`examples/use_cases/terramechanics/terramechanics_rover_pinn.py`)
trains a **Physics-Informed Neural Network** that maps wheel **slip ratio** and **sinkage** →
**drawbar pull (Fx)**, **normal load (Fz)** and **driving torque (My)** for GRC-1 lunar regolith.

The trained model is exported as a **TorchScript** `.pt` file that can be loaded and evaluated
from Rust using the [`tch-rs`](https://github.com/LaurentMazare/tch-rs) crate (the official
LibTorch binding for Rust). An **ONNX** export path is also described for projects that prefer
ONNX Runtime.

### Why replace the raycast model?

| Capability | `lunco-mobility` raycast | PINNeAPPle PINN |
|---|---|---|
| Physical accuracy | Coulomb friction (approximate) | Bekker-Wong (validated against GRC-1 data) |
| Slip-dependent traction | No | Yes |
| Sinkage dependency | No | Yes |
| Throughput (CPU) | ~1 M evals/s (trivial math) | ~500 K evals/s (fully batched) |
| Throughput (GPU) | N/A | >10 M evals/s |
| Parameter sensitivity | Manual tuning | Trained from data |

---

## Prerequisites

### Python side (PINNeAPPle)

```bash
pip install torch numpy scipy matplotlib
# inside the PINNeAPPle repo root
python examples/use_cases/terramechanics/terramechanics_rover_pinn.py
```

This writes `examples/use_cases/terramechanics/outputs/terramechanics_surrogate.pt` —
the TorchScript model — plus `surrogate_metadata.json` with normalisation constants.

### Rust side (LunCoSim)

Add to `crates/lunco-mobility/Cargo.toml`:

```toml
[dependencies]
tch = "0.17"         # tch-rs — LibTorch Rust bindings
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

Download the LibTorch C++ distribution matching your PyTorch version from
<https://pytorch.org/get-started/locally/> and set the `LIBTORCH` environment variable:

```bash
# Linux / macOS
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH/lib:$LD_LIBRARY_PATH

# Windows (PowerShell)
$env:LIBTORCH = "C:\libtorch"
$env:PATH = "$env:LIBTORCH\lib;$env:PATH"
```

---

## Step 1 — Export the TorchScript model

The training script already exports TorchScript at the end.  If you need ONNX instead:

```python
# run once after training, inside the PINNeAPPle Python environment
import torch, json
from pathlib import Path
from examples.use_cases.terramechanics.terramechanics_rover_pinn import (
    TerraMechanicsPINN, Normalizer, train,
)

model, norm_x, norm_y, *_ = train(epochs=4000)

# TorchScript (recommended)
scripted = torch.jit.script(model)
scripted.save("terramechanics_surrogate.pt")

# ONNX (alternative — requires onnx + onnxruntime)
dummy = torch.zeros(1, 2)
torch.onnx.export(model, dummy, "terramechanics_surrogate.onnx",
                  input_names=["slip_sinkage_norm"],
                  output_names=["forces_norm"],
                  dynamic_axes={"slip_sinkage_norm": {0: "batch"}, "forces_norm": {0: "batch"}})

# Normalisation metadata
meta = {
    "norm_x_lo":  norm_x.lo.tolist(),
    "norm_x_hi":  norm_x.hi.tolist(),
    "norm_y_lo":  norm_y.lo.tolist(),
    "norm_y_hi":  norm_y.hi.tolist(),
    "units": {
        "input":  ["slip_ratio [-]", "sinkage [m]"],
        "output": ["Fx [N]", "Fz [N]", "My [Nm]"]
    }
}
with open("surrogate_metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Exported: terramechanics_surrogate.pt  +  surrogate_metadata.json")
```

---

## Step 2 — Rust wrapper (`lunco-terrain-pinn`)

Create a new crate inside `crates/`:

```
crates/
  lunco-terrain-pinn/
    Cargo.toml
    src/
      lib.rs
      normalizer.rs
      wheel_forces.rs
```

### `Cargo.toml`

```toml
[package]
name    = "lunco-terrain-pinn"
version = "0.1.0"
edition = "2021"

[dependencies]
tch          = "0.17"
serde        = { version = "1", features = ["derive"] }
serde_json   = "1"
bevy         = { workspace = true, optional = true }

[features]
bevy_plugin = ["dep:bevy"]
```

---

### `src/normalizer.rs`

```rust
//! Min-max normaliser matching PINNeAPPle's Python Normalizer class.
use tch::{Kind, Tensor};

#[derive(Debug, Clone)]
pub struct Normalizer {
    pub lo: Vec<f32>,
    pub hi: Vec<f32>,
}

impl Normalizer {
    /// Transform physical values → [-1, +1]
    pub fn transform_f32(&self, values: &[f32]) -> Vec<f32> {
        values.iter().zip(self.lo.iter().zip(self.hi.iter()))
            .map(|(&v, (&lo, &hi))| {
                let rng = (hi - lo).max(1e-12);
                2.0 * (v - lo) / rng - 1.0
            })
            .collect()
    }

    /// Transform physical values → tensor in [-1, +1]
    pub fn transform_tensor(&self, values: Tensor) -> Tensor {
        let lo = Tensor::of_slice(&self.lo);
        let hi = Tensor::of_slice(&self.hi);
        let rng = (&hi - &lo).clamp_min(1e-12_f64);
        2.0 * (values - lo) / rng - 1.0
    }

    /// Inverse: [-1, +1] → physical values
    pub fn inverse_tensor(&self, normed: Tensor) -> Tensor {
        let lo = Tensor::of_slice(&self.lo);
        let hi = Tensor::of_slice(&self.hi);
        let rng = (&hi - &lo).clamp_min(1e-12_f64);
        (normed + 1.0) * rng / 2.0 + lo
    }
}
```

---

### `src/wheel_forces.rs`

```rust
//! PINN-based wheel-soil force computation for a single wheel.
use tch::{CModule, Kind, Tensor, Device};
use crate::normalizer::Normalizer;

/// Wheel-terrain force prediction result.
#[derive(Debug, Clone, Copy)]
pub struct WheelForces {
    /// Drawbar pull (longitudinal traction force) [N]
    pub fx: f32,
    /// Normal load (vertical reaction) [N]
    pub fz: f32,
    /// Driving torque about wheel axle [Nm]
    pub my: f32,
}

/// Loaded PINN surrogate for Bekker-Wong terramechanics.
pub struct TerraPINN {
    model:  CModule,
    norm_x: Normalizer,
    norm_y: Normalizer,
    device: Device,
}

impl TerraPINN {
    /// Load from TorchScript `.pt` and JSON metadata files.
    pub fn load(
        model_path:    &str,
        metadata_path: &str,
        use_gpu:       bool,
    ) -> anyhow::Result<Self> {
        let device = if use_gpu && tch::Cuda::is_available() {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };

        let model = CModule::load_on_device(model_path, device)?;

        let meta: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(metadata_path)?
        )?;

        let parse_vec = |key: &str| -> Vec<f32> {
            meta[key].as_array().unwrap()
                .iter().map(|v| v.as_f64().unwrap() as f32)
                .collect()
        };

        Ok(Self {
            model,
            norm_x: Normalizer { lo: parse_vec("norm_x_lo"), hi: parse_vec("norm_x_hi") },
            norm_y: Normalizer { lo: parse_vec("norm_y_lo"), hi: parse_vec("norm_y_hi") },
            device,
        })
    }

    /// Predict forces for a single wheel.
    ///
    /// # Arguments
    /// * `slip`    — slip ratio (0.0 = free-rolling, 1.0 = full spin) [-]
    /// * `sinkage` — wheel sinkage into regolith [m]
    pub fn predict(&self, slip: f32, sinkage: f32) -> anyhow::Result<WheelForces> {
        let input_raw = Tensor::of_slice(&[slip, sinkage])
            .to_device(self.device)
            .unsqueeze(0);                              // shape: [1, 2]

        let input_norm = self.norm_x.transform_tensor(input_raw);

        let output_norm = tch::no_grad(|| {
            self.model.forward_ts(&[input_norm])
        })?;

        let output = self.norm_y.inverse_tensor(output_norm.squeeze());
        let vals: Vec<f32> = output.into();

        Ok(WheelForces { fx: vals[0], fz: vals[1], my: vals[2] })
    }

    /// Batched prediction — evaluate N wheels in a single forward pass.
    ///
    /// `inputs` is a flat &[(slip, sinkage)] slice.
    pub fn predict_batch(&self, inputs: &[(f32, f32)]) -> anyhow::Result<Vec<WheelForces>> {
        let flat: Vec<f32> = inputs.iter()
            .flat_map(|(s, z)| [*s, *z])
            .collect();

        let input_raw = Tensor::of_slice(&flat)
            .reshape(&[inputs.len() as i64, 2])
            .to_device(self.device);

        let input_norm  = self.norm_x.transform_tensor(input_raw);
        let output_norm = tch::no_grad(|| self.model.forward_ts(&[input_norm]))?;
        let output      = self.norm_y.inverse_tensor(output_norm);     // shape [N, 3]

        let flat_out: Vec<f32> = output.into();
        Ok(flat_out.chunks_exact(3)
            .map(|c| WheelForces { fx: c[0], fz: c[1], my: c[2] })
            .collect())
    }
}
```

---

### `src/lib.rs`

```rust
pub mod normalizer;
pub mod wheel_forces;

pub use wheel_forces::{TerraPINN, WheelForces};

#[cfg(feature = "bevy_plugin")]
pub mod bevy_plugin;
```

---

## Step 3 — Bevy plugin (optional `bevy_plugin` feature)

Create `src/bevy_plugin.rs`:

```rust
//! Optional Bevy plugin: replaces `lunco-mobility` Coulomb friction with PINN forces.
use bevy::prelude::*;
use crate::{TerraPINN, WheelForces};

/// Resource holding the loaded PINN surrogate.
#[derive(Resource)]
pub struct TerraPINNResource(pub TerraPINN);

/// Component: wheel slip ratio computed upstream by the drive controller.
#[derive(Component, Default)]
pub struct WheelSlip(pub f32);

/// Component: wheel sinkage estimated from raycast depth delta.
#[derive(Component, Default)]
pub struct WheelSinkage(pub f32);

/// Component: output forces from PINN (written each fixed update).
#[derive(Component, Default)]
pub struct PINNForces(pub WheelForces);

impl Default for WheelForces {
    fn default() -> Self { Self { fx: 0.0, fz: 0.0, my: 0.0 } }
}

pub struct TerraPINNPlugin {
    pub model_path:    String,
    pub metadata_path: String,
    pub use_gpu:       bool,
}

impl Plugin for TerraPINNPlugin {
    fn build(&self, app: &mut App) {
        let pinn = TerraPINN::load(&self.model_path, &self.metadata_path, self.use_gpu)
            .expect("failed to load TerraPINN surrogate");
        app.insert_resource(TerraPINNResource(pinn))
           .add_systems(FixedUpdate, evaluate_pinn_forces);
    }
}

/// System: batch-evaluate PINN for all wheels each physics tick.
fn evaluate_pinn_forces(
    pinn:  Res<TerraPINNResource>,
    mut query: Query<(&WheelSlip, &WheelSinkage, &mut PINNForces)>,
) {
    let inputs: Vec<(f32, f32)> = query.iter()
        .map(|(s, z, _)| (s.0, z.0))
        .collect();

    if inputs.is_empty() { return; }

    if let Ok(forces) = pinn.0.predict_batch(&inputs) {
        for ((_, _, mut pf), f) in query.iter_mut().zip(forces) {
            pf.0 = f;
        }
    }
}
```

---

## Step 4 — Wire into `lunco-mobility`

In `crates/lunco-mobility/src/traction.rs`, add a branch that uses `PINNForces`
when the component is present:

```rust
use lunco_terrain_pinn::PINNForces;

/// System: apply traction forces to the wheel rigid body.
fn apply_traction_forces(
    mut wheels: Query<(
        &Transform,
        Option<&PINNForces>,   // PINN forces — present if TerraPINNPlugin is loaded
        &mut ExternalForce,
    )>,
) {
    for (transform, pinn_forces, mut ext_force) in wheels.iter_mut() {
        let (fx, fz, _my) = if let Some(pf) = pinn_forces {
            (pf.0.fx, pf.0.fz, pf.0.my)
        } else {
            // Fallback to original Coulomb friction model
            coulomb_forces(/* ... */)
        };

        // Apply forces in wheel-local frame
        let forward = transform.forward();
        let up      = transform.up();
        ext_force.force += forward * fx + up * fz;
    }
}
```

Add to your `MobilityPlugin`:

```rust
// In lunco-mobility/src/lib.rs
use lunco_terrain_pinn::{TerraPINNPlugin};

app.add_plugins(TerraPINNPlugin {
    model_path:    "assets/models/terramechanics_surrogate.pt".to_string(),
    metadata_path: "assets/models/surrogate_metadata.json".to_string(),
    use_gpu: false,
});
```

---

## Step 5 — Sinkage estimation

The PINN requires a sinkage estimate. A practical approach from the existing
raycast suspension in `lunco-mobility`:

```rust
/// System: estimate wheel sinkage from suspension compression.
fn estimate_sinkage(
    mut wheels: Query<(&WheelRaycast, &mut WheelSinkage)>,
    // WheelRaycast.rest_length  — design ride height [m]
    // WheelRaycast.hit_distance — actual hit distance from avian3d raycast [m]
) {
    for (raycast, mut sinkage) in wheels.iter_mut() {
        // Sinkage = rest length minus actual compressed length, clamped to [0, max_sink]
        let compression = (raycast.rest_length - raycast.hit_distance).max(0.0);
        // Wheel radius R = 0.125 m; max sinkage from Bekker-Wong = 0.058 m
        sinkage.0 = compression.clamp(0.002, 0.058);
    }
}
```

---

## Step 6 — Asset bundling

Copy both output files into your Bevy asset directory:

```
lunco-sim/
  assets/
    models/
      terramechanics_surrogate.pt      # TorchScript model
      surrogate_metadata.json          # normalisation constants
```

Or embed the metadata at compile time using `include_str!`:

```rust
const SURROGATE_META: &str = include_str!("../assets/models/surrogate_metadata.json");
```

---

## ONNX Alternative (no LibTorch dependency)

If you prefer to avoid the LibTorch dependency, use ONNX Runtime:

```toml
[dependencies]
ort = "2"          # ort-rs — ONNX Runtime Rust bindings
ndarray = "0.16"
```

```rust
use ort::{Environment, Session, SessionBuilder, Value};
use ndarray::Array2;

let env     = Environment::builder().with_name("terra").build().unwrap();
let session = SessionBuilder::new(&env)?.with_model_from_file("terramechanics_surrogate.onnx")?;

let input = Array2::<f32>::from_shape_vec((1, 2), vec![slip_norm, sinkage_norm]).unwrap();
let outputs = session.run(vec![Value::from_array(session.allocator(), &input)?])?;
let result: ndarray::ArrayViewD<f32> = outputs[0].try_extract()?;
// result shape: [1, 3] -> [Fx_norm, Fz_norm, My_norm]
```

---

## Benchmark Summary

From the PINNeAPPle benchmark (`benchmark_vs_omnilrs.py`):

| Solver | Throughput (CPU) | Fx Relative Error | Notes |
|---|---|---|---|
| PINNeAPPle (scipy.quad) | 350 evals/s | reference | High-accuracy |
| OmniLRS (numpy trapz) | 19 000 evals/s | ~124% | 200-point quadrature |
| **PINNeAPPle PINN** | **~500 000 evals/s** | **< 1%** | TorchScript, CPU |

The PINN surrogate is **~1 420x faster** than scipy.quad while achieving < 1% relative error
on all three force components.  On GPU, throughput exceeds 10 M evals/s, enabling real-time
evaluation for large swarms or Monte-Carlo simulations.

---

## Data flow diagram

```
+------------------+         +-----------------+          +--------------------+
|  lunco-mobility  |  slip   |                 |  Fx, Fz  |  avian3d           |
|  (drive system)  | ------> | TerraPINNPlugin | -------> |  (force integrator)|
|                  | sinkage |  (PINN forward) |  My      |                    |
+------------------+         +-----------------+          +--------------------+
         ^                          ^
         |                          |
  WheelSlip component       terramechanics_surrogate.pt
  WheelSinkage component    (TorchScript, trained by PINNeAPPle)
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `LIBTORCH not found` | Set `$LIBTORCH` and `LD_LIBRARY_PATH` before `cargo build` |
| `CModule::load failed` | Ensure PyTorch and tch-rs versions match exactly |
| Forces are NaN | Check that slip and sinkage are within `[0.0, 0.75]` and `[0.002, 0.058]` |
| Very low Fx at zero slip | Expected — Bekker-Wong predicts minimal drawbar pull at free-roll |
| GPU not detected | Rebuild tch-rs with `--features cuda`; LibTorch must be CUDA-enabled |

---

## References

- PINNeAPPle terramechanics pipeline: `examples/use_cases/terramechanics/terramechanics_rover_pinn.py`
- Benchmark: `examples/use_cases/terramechanics/benchmark_vs_omnilrs.py`
- tch-rs: <https://github.com/LaurentMazare/tch-rs>
- ONNX Runtime Rust: <https://github.com/pykeio/ort>
- LunCoSim `lunco-mobility`: <https://github.com/LunCoSim/lunco-sim/tree/main/crates/lunco-mobility>
- LunCoSim `lunco-terrain`: <https://github.com/LunCoSim/lunco-sim/tree/main/crates/lunco-terrain>
- Bekker (1969) Introduction to Terrain-Vehicle Systems
- Wong (1978) Theory of Ground Vehicles
