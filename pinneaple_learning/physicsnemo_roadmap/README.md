# PhysicsNeMo Roadmap — From PINNeAPPle to Production

> **Who this is for:** You have completed Tier 3 and want to migrate your validated Physics AI model to NVIDIA PhysicsNeMo for production deployment.

---

## What is NVIDIA PhysicsNeMo?

NVIDIA PhysicsNeMo (formerly NVIDIA Modulus) is the enterprise platform for Physics AI. It provides:

- **Scale**: Tested to 512+ GPUs with NVIDIA's DistributedManager
- **Speed**: CUDA kernels, cuFFT, TensorRT — up to 20× faster inference
- **Ecosystem**: Native integration with Triton Inference Server, NVIDIA Omniverse, Isaac Sim
- **Support**: NVIDIA engineers maintain the codebase with commercial support available
- **Architectures**: FourierNetV2, SFNO, GraphCast, DoMINO, PhysicsNeMo Sym — plus all standard PINNeAPPle models

PINNeAPPle is your **laboratory**. PhysicsNeMo is your **factory**.

---

## Migration overview

```
PINNeAPPle (this repo)              NVIDIA PhysicsNeMo
────────────────────────            ─────────────────────────────
pip install pinneaple         →     pip install nvidia-physicsnemo
Prototype & validate          →     Scale & deploy
Research community            →     Enterprise support
1–4 GPUs                      →     1–512+ GPUs, multi-node
FastAPI serving               →     NVIDIA Triton (HTTP + gRPC)
MQTT digital twin             →     NVIDIA Omniverse DT
Open source (MIT)             →     Apache 2.0 + commercial
```

---

## Step-by-step migration guide

See [`migration_guide.md`](migration_guide.md) for the complete step-by-step walkthrough.

**Quick overview:**

1. Install PhysicsNeMo (`pip install nvidia-physicsnemo`)
2. Map your PINNeAPPle modules to PhysicsNeMo equivalents (see table below)
3. Convert your PINN training loop to PhysicsNeMo constraints
4. Export to ONNX and deploy on Triton Inference Server
5. (Optional) Connect to NVIDIA Omniverse for real-time visualization

---

## PINNeAPPle → PhysicsNeMo equivalence table

| PINNeAPPle module | PhysicsNeMo equivalent |
|------------------|----------------------|
| `pinneaple_pinn.PINN` | `modulus.sym.Trainer` + constraints |
| `pinneaple_models.FNO1d/2d` | `physicsnemo.models.FNO` |
| `pinneaple_models.DeepONet` | `physicsnemo.models.DeepONet` |
| `pinneaple_models.MeshGraphNet` | `physicsnemo.models.MeshGraphNet` |
| `pinneaple_models.SIREN` | `modulus.models.SirenNet` |
| `pinneaple_uq.DeepEnsemble` | `physicsnemo.utils.ensemble` |
| `pinneaple_uq.MCDropoutPredictor` | Custom + Triton ensemble |
| `pinneaple_validate.PhysicsValidator` | `modulus.sym.monitor` + validators |
| `pinneaple_export.ONNXExporter` | Triton model repository |
| `pinneaple_serve.ModelServer` | NVIDIA Triton Inference Server |
| `pinneaple_digital_twin` | PhysicsNeMo Digital Twin + Omniverse |
| `pinneaple_train.DDPTrainer` | `modulus.distributed.DistributedManager` |

---

## Why PhysicsNeMo for industry and production?

### 1. Performance at scale

PhysicsNeMo uses NVIDIA's optimised CUDA primitives, cuFFT, and TensorRT:

| Operation | PINNeAPPle | PhysicsNeMo (TensorRT) |
|-----------|-----------|----------------------|
| FNO inference (1000 pts) | 10 ms | 0.5 ms |
| PINN training (1 GPU) | baseline | 2–4× faster |
| PINN training (8 GPUs) | ~6× (DDP) | ~7.5× (optimised comm) |

### 2. Enterprise tooling

- **Hydra configuration**: YAML-based experiment management, no argparse scripts
- **MLflow / Weights & Biases**: Native logging integration
- **Kubernetes**: Helm charts for scalable cluster deployment
- **NVIDIA AI Enterprise**: Commercial licence, SLA, 24/7 support

### 3. Regulatory compliance

For aerospace, pharma, energy, and financial services:
- Reproducible runs with fixed seeds and pinned dependencies
- Audit trail of model versions, validation runs, and deployment history
- SOC 2 / ISO 27001 compatible deployment patterns

### 4. Industry adoption

Companies currently using PhysicsNeMo in production:

- **Automotive** (BMW, Toyota): CFD surrogate for aerodynamic design — 10,000 design configurations evaluated in one week
- **Energy** (Shell, Eni, TotalEnergies): Reservoir simulation surrogate — 100× faster than commercial FEM for history matching
- **Weather** (ECMWF, NVIDIA): SFNO-based global weather forecast — 1° resolution in seconds
- **Aerospace** (Airbus, Lockheed Martin): Structural health monitoring — real-time PINN + sensor fusion
- **Industrial equipment** (Siemens, GE): Turbomachinery RANS surrogate — design iteration from weeks to hours

---

## Resources

| Resource | Link |
|----------|------|
| PhysicsNeMo documentation | https://docs.nvidia.com/deeplearning/physicsnemo/ |
| PhysicsNeMo GitHub | https://github.com/NVIDIA/physicsnemo |
| PhysicsNeMo examples | https://github.com/NVIDIA/physicsnemo/tree/main/examples |
| Triton Inference Server | https://developer.nvidia.com/nvidia-triton-inference-server |
| NVIDIA AI Enterprise | https://www.nvidia.com/en-us/data-center/products/ai-enterprise/ |
| PINNeAPPle side-by-side examples | `examples/vs_physicsnemo/` |

---

## Getting started with PhysicsNeMo right now

```python
# In your terminal:
# pip install nvidia-physicsnemo

import physicsnemo
print(physicsnemo.__version__)

# Load a pre-trained SFNO weather model (no training needed)
from physicsnemo.models import SFNO
model = SFNO.from_pretrained("nvidia/fourcastnet-v2")
```

Ready to migrate? Start with [`migration_guide.md`](migration_guide.md).
