"""Tier 3 Builder — production checklist and PhysicsNeMo readiness guide."""

from __future__ import annotations


def production_checklist() -> None:
    """Print the production readiness checklist."""
    print(_PRODUCTION_CHECKLIST)


def physicsnemo_readiness() -> None:
    """Print the PhysicsNeMo readiness assessment."""
    print(_PHYSICSNEMO_READINESS)


# ---------------------------------------------------------------------------

_PRODUCTION_CHECKLIST = """
╔══════════════════════════════════════════════════════════════╗
║        Production Readiness Checklist — Tier 3: Builder     ║
╚══════════════════════════════════════════════════════════════╝

Work through this list before declaring your model production-ready.
Each item links to the relevant PINNeAPPle template.

─── ACCURACY ─────────────────────────────────────────────────

  [ ] Relative L2 error meets your target (e.g., ε < 1e-4)
      → templates/32_physics_validation.py

  [ ] PDE residual at 10× more points than training
      (out-of-distribution collocation check)

  [ ] Boundary condition residual < 1e-5 everywhere

  [ ] Conservation laws verified (energy, mass, momentum)
      → pinneaple_validate.ConservationLawChecker

  [ ] Uncertainty bounds are calibrated (coverage ≥ 90%)
      → templates/16_uncertainty_quantification.py

─── PERFORMANCE ──────────────────────────────────────────────

  [ ] Inference latency measured on target hardware
      → templates/23_model_export.py (latency benchmark)

  [ ] Model exported to ONNX for framework-agnostic serving
      → templates/23_model_export.py

  [ ] TorchScript export works (fallback if ONNX fails)
      → templates/23_model_export.py

  [ ] Memory footprint measured (peak GPU MB)

  [ ] Batch inference tested (single query vs batch of 1000)

─── SCALE ────────────────────────────────────────────────────

  [ ] DDP (Distributed Data Parallel) training verified
      on 2+ GPUs → templates/07_ddp_distributed.py

  [ ] Checkpoint/resume works (training interrupted + resumed)
      → pinneaple_train.CheckpointCallback

  [ ] Training time scales sub-linearly with GPU count
      (good scaling efficiency > 0.7)

─── SERVING ──────────────────────────────────────────────────

  [ ] REST API endpoint responds correctly
      → templates/22_model_serving.py

  [ ] Endpoint handles concurrent requests (load test)

  [ ] Input validation rejects out-of-range inputs gracefully

  [ ] /health and /metrics endpoints exist for monitoring

─── DIGITAL TWIN (if applicable) ────────────────────────────

  [ ] Sensor stream ingestion works (MQTT / Kafka)
      → templates/24_digital_twin.py

  [ ] State assimilation (EnKF) converges within 10 cycles

  [ ] Anomaly detection triggers at > 3σ deviations

  [ ] Drift detection alerts when model error increases

─── DOCUMENTATION ────────────────────────────────────────────

  [ ] Model card: architecture, training data, known limits

  [ ] Validation report saved (PhysicsValidator.full_report)

  [ ] Reproducibility: fixed random seeds, pinned dependencies

─── PHYSICSNEMO MIGRATION ────────────────────────────────────

  [ ] Read migration_guide.md in physicsnemo_roadmap/
  [ ] Identify PINNeAPPle ↔ PhysicsNeMo equivalents for your
      modules (see the table in README.md)
  [ ] Run side-by-side comparison:
      → examples/vs_physicsnemo/

  When all boxes are checked → you are ready for production
  deployment with NVIDIA PhysicsNeMo.
"""

_PHYSICSNEMO_READINESS = """
╔══════════════════════════════════════════════════════════════╗
║         Are you ready for NVIDIA PhysicsNeMo?               ║
╚══════════════════════════════════════════════════════════════╝

PhysicsNeMo (formerly NVIDIA Modulus) is the enterprise platform
for production Physics AI. This assessment tells you whether you
are ready to migrate.

─── Prerequisite check ───────────────────────────────────────

  Answer YES or NO to each question:

  1. Do you have a validated Physics AI model?
     (ε < 1e-4, conservation laws checked, UQ calibrated)

  2. Do you need any of the following?
     • Training on 8+ GPUs
     • Inference SLAs (< 50 ms p99 latency)
     • Integration with enterprise ML pipelines (MLflow,
       Weights & Biases, Triton Inference Server)
     • NVIDIA-supported architectures (FourierNetV2,
       SFNO, GraphCast, DoMINO, PhysicsNeMo Sym)
     • Compliance / audit trail for industrial deployment

  3. Are you solving any of these at scale?
     • Turbulent flow (RANS, LES)
     • Weather / climate simulation
     • Large-scale structural mechanics
     • Multiphysics (fluid-structure interaction, etc.)

  If you answered YES to any of (2) or (3) AND YES to (1):
  → You are ready. See physicsnemo_roadmap/migration_guide.md

─── What PhysicsNeMo adds on top of PINNeAPPle ──────────────

  PINNeAPPle feature          PhysicsNeMo equivalent
  ─────────────────────────── ────────────────────────────────
  PINN training loop          modulus.sym training + constraint
  FNO1d/2d                    modulus.models.FNO (multi-GPU)
  DeepONet                    modulus.models.DeepONet
  MeshGraphNet                modulus.models.MeshGraphNet
  KOmegaSSTResiduals          modulus.sym.eq.pdes.turbulence
  DDP training                built-in, battle-tested
  ONNX export                 Triton Inference Server
  REST serving                NVIDIA Triton (HTTP/gRPC)
  Digital twin                NVIDIA Omniverse + PhysicsNeMo DT
  Active learning             PhysicsNeMo adaptive sampling
  ArenaRunner                 PhysicsNeMo benchmarking suite

─── Migration path in one picture ───────────────────────────

  PINNeAPPle                     PhysicsNeMo
  ──────────────────────────     ────────────────────────────
  Prototype in hours             Scale to production in days
  Understand every line          Hardened, NVIDIA-tested code
  Open source, no vendor lock    Enterprise support available
  CPU / single GPU               Multi-GPU, multi-node, TRT
  Research / education           Industry / production
       │
       │  ←  You are here (Tier 3, ready to migrate)
       ▼
  pip install nvidia-physicsnemo
  (see migration_guide.md for full instructions)

─── Why PhysicsNeMo for production? ─────────────────────────

  1. PERFORMANCE
     PhysicsNeMo uses CUDA kernels, cuFFT, and TensorRT
     under the hood. A 1000-point inference that takes
     10 ms in PINNeAPPle can take 0.2 ms in Triton.

  2. SCALE
     DistributedManager handles multi-node training with
     gradient compression, mixed precision, and fault
     tolerance — tested to 512 GPUs.

  3. ECOSYSTEM
     Native integration with NVIDIA Omniverse, Isaac Sim,
     Modulus Sym (symbolic PDEs), and NVIDIA AI Enterprise.

  4. SUPPORT
     NVIDIA engineers maintain the codebase, publish
     benchmark results, and provide commercial support.

  5. INDUSTRY ADOPTION
     BMW, Siemens, Shell, Eni, and hundreds of other
     companies run Physics AI in production with PhysicsNeMo.

  PINNeAPPle taught you the craft. PhysicsNeMo gives you
  the factory.

─── Next action ─────────────────────────────────────────────

  Read: pinneaple_learning/physicsnemo_roadmap/migration_guide.md
  Docs: https://docs.nvidia.com/deeplearning/physicsnemo/
  Code: https://github.com/NVIDIA/physicsnemo
"""
