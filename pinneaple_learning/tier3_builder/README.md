# Tier 3 — Builder: From Validated Model to Production

> **Who this is for:** You have a validated Physics AI model and want to scale it, harden it for production, and eventually migrate to NVIDIA PhysicsNeMo.

---

## Table of contents

1. [The Builder mindset](#1-the-builder-mindset)
2. [Distributed training with DDP](#2-distributed-training-with-ddp)
3. [Model export and inference optimization](#3-model-export-and-inference-optimization)
4. [REST API serving](#4-rest-api-serving)
5. [Digital twins and live sensor assimilation](#5-digital-twins)
6. [Production validation checklist](#6-production-validation-checklist)
7. [Why migrate to PhysicsNeMo?](#7-why-migrate-to-physicsnemo)
8. [Milestones before migration](#8-milestones)

---

## 1. The Builder mindset

At Tier 2 you validated an approach on a research problem. At Tier 3 you ask:

- *How do I train this on 8 GPUs without rewriting everything?*
- *Can I serve predictions in under 50 ms via REST API?*
- *How do I integrate live sensor data into a running simulation?*
- *What is the migration path to PhysicsNeMo when we go to production?*

The Builder cares about **reliability, latency, scale, and maintainability**.

---

## 2. Distributed training with DDP

PyTorch's `DistributedDataParallel` scales training across GPUs.

```bash
# Launch on 4 GPUs
torchrun --nproc_per_node=4 templates/07_ddp_distributed.py
```

Key patterns in [`templates/07_ddp_distributed.py`](../../templates/07_ddp_distributed.py):

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
net = DDP(net.to(rank), device_ids=[rank])
```

**Scaling efficiency targets:**
- 2 GPUs: > 1.8× speedup  
- 4 GPUs: > 3.2× speedup
- 8 GPUs: > 5.5× speedup

If you see worse scaling, the bottleneck is usually gradient communication or data loading.

---

## 3. Model export and inference optimization

```python
from pinneaple_export import ONNXExporter, ExportValidator

exporter = ONNXExporter(net, input_shape=(1, 1))
exporter.export("model.onnx")

validator = ExportValidator("model.onnx")
validator.check_accuracy(x_test, net)
validator.benchmark_latency(x_test, n_warmup=50, n_runs=500)
```

**Typical latency improvements:**

| Format | Single-point latency | Notes |
|--------|---------------------|-------|
| PyTorch eager | 1.0× (baseline) | Easy to debug |
| TorchScript | 0.6–0.8× | No Python overhead |
| ONNX Runtime | 0.2–0.4× | Cross-platform |
| TensorRT (via Triton) | 0.05–0.1× | NVIDIA GPU only |

See [`templates/23_model_export.py`](../../templates/23_model_export.py)

---

## 4. REST API serving

```python
# Start the server
from pinneaple_serve import ModelServer, ServeConfig

server = ModelServer(net, config=ServeConfig(host="0.0.0.0", port=8080))
server.start()
```

```bash
# Query it
curl -X POST http://localhost:8080/predict \
     -H "Content-Type: application/json" \
     -d '{"x": [[0.5]], "t": [[1.0]]}'
```

The server exposes:
- `POST /predict` — inference endpoint
- `GET /health` — readiness probe
- `GET /metrics` — Prometheus-compatible metrics

See [`templates/22_model_serving.py`](../../templates/22_model_serving.py)

---

## 5. Digital twins

A digital twin continuously assimilates live sensor data to update a running simulation.

```python
from pinneaple_digital_twin import DigitalTwinRuntime, StateAssimilator

twin = DigitalTwinRuntime(net, state_dim=100)
assimilator = StateAssimilator(twin, method="enkf", n_ensemble=50)

# Continuously update as sensors arrive
for reading in sensor_stream:
    assimilator.update(reading)
    state = twin.current_state()
    anomaly = twin.detect_anomaly()
```

Key capabilities:
- **EnKF** (Ensemble Kalman Filter) — handles nonlinear state updates
- **Anomaly detection** — alerts when state diverges > 3σ from ensemble mean
- **Drift detection** — flags when model accuracy degrades over time

See [`templates/24_digital_twin.py`](../../templates/24_digital_twin.py)

---

## 6. Production validation checklist

```python
import pinneaple_learning.tier3_builder as t3
t3.production_checklist()
```

Key items:
- Relative L2 error meets target (typically $\varepsilon < 10^{-4}$)
- Conservation laws pass at $10\times$ more test points than training
- ONNX export produces bitwise-identical results to PyTorch
- REST server handles 100 concurrent requests without timeout
- DDP training achieves > 70% GPU scaling efficiency

---

## 7. Why migrate to PhysicsNeMo?

PINNeAPPle gives you the fastest path to a validated Physics AI model.
NVIDIA PhysicsNeMo gives you everything needed for production at scale.

| Dimension | PINNeAPPle | NVIDIA PhysicsNeMo |
|-----------|-----------|-------------------|
| **Purpose** | Research, education, prototyping | Production, enterprise, scale |
| **Scale** | 1–4 GPUs | 1–512+ GPUs, multi-node |
| **Inference** | PyTorch eager / ONNX | NVIDIA Triton (HTTP + gRPC) |
| **Architecture** | All standard Physics AI models | Same + SFNO, GraphCast, DoMINO |
| **Digital twin** | MQTT / Kafka integration | NVIDIA Omniverse + Isaac Sim |
| **Support** | Open source community | NVIDIA enterprise support |
| **Ecosystem** | Standalone | MLflow, W&B, Kubernetes native |
| **Speed** | Baseline | Up to 20× faster with TensorRT |

**When to migrate:**
- Training time > 24 h on a single GPU
- Inference SLA requirements (< 100 ms)
- Need > 4 GPUs
- Deploying in regulated industries (oil & gas, aerospace, pharma)
- Integrating with SCADA / IoT systems at scale

---

## 8. Milestones

Complete all of these before migrating to PhysicsNeMo:

- [ ] Train on 2+ GPUs with DDP and verify scaling efficiency > 70%
- [ ] Export model to ONNX and verify accuracy is preserved
- [ ] Benchmark inference latency: target < 10 ms for 1D models
- [ ] Serve predictions via REST API with < 50 ms p99 latency
- [ ] Pass all PhysicsValidator checks at 5× held-out test density
- [ ] Complete the migration checklist in `physicsnemo_roadmap/migration_guide.md`
- [ ] Run at least one example from `examples/vs_physicsnemo/`

**When you are ready:**

```python
import pinneaple_learning as pl
pl.explain("physicsnemo")
```

Then read: `pinneaple_learning/physicsnemo_roadmap/migration_guide.md`
