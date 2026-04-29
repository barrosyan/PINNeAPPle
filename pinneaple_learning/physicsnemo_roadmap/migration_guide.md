# Migration Guide: PINNeAPPle → NVIDIA PhysicsNeMo

This guide walks you step by step through migrating a PINNeAPPle project to production with NVIDIA PhysicsNeMo.

**Estimated time:** 1–3 days for a typical PINN project.

---

## Prerequisites

Before starting, verify you have:

- [ ] A validated PINNeAPPle model (L2 error meets target, conservation laws pass)
- [ ] ONNX export working (`templates/23_model_export.py`)
- [ ] DDP training tested on 2+ GPUs (`templates/07_ddp_distributed.py`)
- [ ] Passing the production checklist in `pinneaple_learning/tier3_builder/README.md`

---

## Part 1: Environment setup

### 1.1 Install PhysicsNeMo

```bash
# Minimum install
pip install nvidia-physicsnemo

# Full install (all architectures + utilities)
pip install "nvidia-physicsnemo[all]"

# Verify
python -c "import physicsnemo; print(physicsnemo.__version__)"
```

Requirements:
- Python 3.10+
- PyTorch 2.2+
- CUDA 12.x (for GPU features)
- 24 GB GPU RAM recommended for large models

### 1.2 Optional: NVIDIA Modulus Sym (symbolic PDE constraints)

```bash
pip install nvidia-modulus.sym
```

Modulus Sym provides a symbolic PDE definition interface that replaces manual residual functions.

---

## Part 2: Model migration

### 2.1 Basic PINN → PhysicsNeMo Sym

**Before (PINNeAPPle):**

```python
# pinneaple style
import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=-1)
        return self.net(xt)

# Manual residual
def pde_residual(net, x, t):
    xt = torch.cat([x, t], dim=-1).requires_grad_(True)
    u = net(xt[:, :1], xt[:, 1:])
    u_t = torch.autograd.grad(u.sum(), xt, create_graph=True)[0][:, 1:]
    u_x = torch.autograd.grad(u.sum(), xt, create_graph=True)[0][:, :1]
    u_xx = torch.autograd.grad(u_x.sum(), xt, create_graph=True)[0][:, :1]
    return u_t - alpha * u_xx   # heat equation
```

**After (PhysicsNeMo Sym):**

```python
from modulus.sym.eq.pdes import HeatEquation
from modulus.sym.geometry import Rectangle
from modulus.sym import Trainer, Domain
from modulus.sym.models import FullyConnectedArch

# Define geometry
geo = Rectangle(point_1=(-1, 0), point_2=(1, 1))

# Use pre-built PDE
heat = HeatEquation(T="u", time=True, dim=1, diffusivity=alpha)

# Network architecture
net = FullyConnectedArch(
    input_keys=["x", "t"],
    output_keys=["u"],
    layer_size=64,
    nr_layers=4,
)

# Domain: constraints replace manual loss
domain = Domain()
domain.add_constraint(
    PointwiseBoundaryConstraint(nodes=[heat.make_nodes(), net],
                                 geometry=geo, outvar={"u": 0}),
    name="BC",
)
domain.add_constraint(
    PointwiseInteriorConstraint(nodes=[heat.make_nodes(), net],
                                 geometry=geo, outvar={"heat_equation": 0}),
    name="PDE",
)

trainer = Trainer(domain, cfg)
trainer.solve()
```

### 2.2 FNO migration

**PINNeAPPle:**
```python
from pinneaple_models import FNO1d
model = FNO1d(n_modes=16, width=64, input_channels=1, output_channels=1)
```

**PhysicsNeMo:**
```python
from physicsnemo.models import FNO
model = FNO(
    in_channels=1,
    out_channels=1,
    decoder_layers=1,
    decoder_layer_size=32,
    dimension=1,
    latent_channels=64,
    num_fno_layers=4,
    num_fno_modes=16,
    padding=8,
)
```

The PhysicsNeMo FNO supports multi-GPU out of the box with `DistributedManager`.

### 2.3 MeshGraphNet migration

**PINNeAPPle:**
```python
from pinneaple_models import MeshGraphNet
model = MeshGraphNet(node_features=6, edge_features=3, hidden_dim=128, output_dim=3)
```

**PhysicsNeMo:**
```python
from physicsnemo.models import MeshGraphNet
model = MeshGraphNet(
    input_dim_nodes=6,
    input_dim_edges=3,
    output_dim=3,
    processor_size=15,
    hidden_dim_node_encoder=128,
    hidden_dim_edge_encoder=128,
    hidden_dim_node_decoder=128,
)
```

---

## Part 3: Distributed training

### 3.1 Using DistributedManager

PhysicsNeMo's `DistributedManager` is more powerful than raw DDP:

```python
from physicsnemo.distributed import DistributedManager

DistributedManager.initialize()
dist = DistributedManager()

# Wrap model
model = dist.wrap_model(model)

# Gradient communication is automatic
# Mixed precision is automatic (BF16 on Hopper GPUs)
```

### 3.2 Launch command

```bash
# 4 GPUs on one node
torchrun --nproc_per_node=4 train.py

# Multi-node (2 nodes × 8 GPUs = 16 GPUs total)
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  train.py
```

---

## Part 4: Production serving with Triton

### 4.1 Convert your model

```python
# Already done in Tier 3
from pinneaple_export import ONNXExporter
ONNXExporter(model, input_shape=(1, 2)).export("model.onnx")
```

### 4.2 Create Triton model repository

```
model_repository/
└── heat_pinn/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

`config.pbtxt`:
```protobuf
name: "heat_pinn"
backend: "onnxruntime"
max_batch_size: 1024

input [
  { name: "x" data_type: TYPE_FP32 dims: [1] },
  { name: "t" data_type: TYPE_FP32 dims: [1] }
]
output [
  { name: "u" data_type: TYPE_FP32 dims: [1] }
]

dynamic_batching { preferred_batch_size: [64, 256, 1024] }
```

### 4.3 Start Triton server

```bash
docker run \
  --gpus=all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

### 4.4 Query from Python

```python
import tritonclient.http as triton
import numpy as np

client = triton.InferenceServerClient("localhost:8000")

x_in  = triton.InferInput("x", [1, 1], "FP32")
t_in  = triton.InferInput("t", [1, 1], "FP32")
x_in.set_data_from_numpy(np.array([[0.5]], dtype=np.float32))
t_in.set_data_from_numpy(np.array([[1.0]], dtype=np.float32))

result = client.infer("heat_pinn", [x_in, t_in])
u_pred = result.as_numpy("u")
print(f"u(0.5, 1.0) = {u_pred[0, 0]:.6f}")
```

---

## Part 5: Validation after migration

After migration, always re-run your validation suite on the PhysicsNeMo output:

```python
from pinneaple_validate import PhysicsValidator, ConservationLawChecker

# Use your existing PINNeAPPle validator on the new model
validator = PhysicsValidator(physicsnemo_model, pde_residual_fn)
report = validator.full_report(x_test)
assert report.l2_error < 1e-4, "Migration broke accuracy!"

checker = ConservationLawChecker(physicsnemo_model)
checker.check_energy_conservation(x_test, u_test)
```

---

## Part 6: Monitoring in production

PhysicsNeMo integrates with standard observability tools:

```python
# Weights & Biases
import wandb
wandb.init(project="my-physics-twin")

# MLflow
import mlflow
mlflow.log_param("alpha", alpha)
mlflow.log_metric("l2_error", l2_error)

# Prometheus (via Triton /metrics endpoint)
# curl http://localhost:8002/metrics
```

---

## Migration checklist

- [ ] PhysicsNeMo installed and version verified
- [ ] PINN training loop converted to Domain + constraints
- [ ] FNO / DeepONet / MeshGraphNet replaced with PhysicsNeMo equivalents
- [ ] DDP → DistributedManager migration done
- [ ] ONNX model in Triton model repository
- [ ] Triton server starts and responds to /health
- [ ] Accuracy validated against PINNeAPPle baseline (Δε < 1%)
- [ ] Triton latency meets SLA (p99 < target ms)
- [ ] Monitoring and alerting configured
- [ ] Model card updated with PhysicsNeMo version

---

## Where to get help

| Resource | Link |
|----------|------|
| PhysicsNeMo documentation | https://docs.nvidia.com/deeplearning/physicsnemo/ |
| PhysicsNeMo GitHub issues | https://github.com/NVIDIA/physicsnemo/issues |
| NVIDIA Developer Forums | https://forums.developer.nvidia.com/c/ai-data-science/deep-learning/physicsnemo |
| NVIDIA AI Enterprise support | https://www.nvidia.com/en-us/data-center/products/ai-enterprise/ |
