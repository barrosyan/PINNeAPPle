"""PhysicsNeMo roadmap — printed migration overview."""

from __future__ import annotations


def migration_overview() -> None:
    """Print the PhysicsNeMo migration overview."""
    print(_MIGRATION_OVERVIEW)


_MIGRATION_OVERVIEW = """
╔══════════════════════════════════════════════════════════════╗
║         NVIDIA PhysicsNeMo — Migration Overview             ║
╚══════════════════════════════════════════════════════════════╝

This guide walks you from a PINNeAPPle prototype to a production
deployment on NVIDIA PhysicsNeMo in four stages.

─── STAGE 1: Install PhysicsNeMo ────────────────────────────

  pip install nvidia-physicsnemo

  For GPU-accelerated FNO / SFNO / DoMINO:
  pip install nvidia-physicsnemo[all]

  Requires: CUDA 12.x, Python 3.10+, PyTorch 2.2+

─── STAGE 2: Map PINNeAPPle → PhysicsNeMo ───────────────────

  PINNeAPPle                        PhysicsNeMo
  ────────────────────────────────  ──────────────────────────
  pinneaple_pinn.PINN               modulus.sym.Trainer
  pinneaple_models.FNO1d            physicsnemo.models.FNO
  pinneaple_models.DeepONet         physicsnemo.models.DeepONet
  pinneaple_models.MeshGraphNet     physicsnemo.models.MeshGraphNet
  pinneaple_uq.DeepEnsemble         physicsnemo.utils.ensemble
  pinneaple_validate.PhysicsValidator  modulus.sym.monitor
  pinneaple_export.ONNXExporter     Triton model repository
  pinneaple_serve.ModelServer       Triton Inference Server
  pinneaple_digital_twin            PhysicsNeMo Digital Twin

─── STAGE 3: Migrate your PINN training loop ────────────────

  PINNeAPPle (what you have now):

      net = build_net()
      opt = torch.optim.Adam(net.parameters(), lr=1e-3)
      for epoch in range(N_EPOCHS):
          loss = pde_loss(net) + ic_loss(net)
          loss.backward(); opt.step()

  PhysicsNeMo (what you migrate to):

      from modulus.sym import Trainer, Domain
      from modulus.sym.geometry import Rectangle
      from modulus.sym.eq.pdes import HeatEquation

      domain = Domain()
      domain.add_constraint(...)   # BCs, ICs, PDE constraints
      trainer = Trainer(domain, cfg)
      trainer.solve()

  Key differences:
    • Constraints replace manual loss terms
    • DistributedManager handles multi-GPU automatically
    • HydraConfig provides YAML-based experiment management
    • Validators run automatically after each checkpoint

─── STAGE 4: Production serving with Triton ─────────────────

  1. Export model to ONNX (already done in Tier 3):
         from pinneaple_export import ONNXExporter
         ONNXExporter(net).export("model.onnx")

  2. Place in Triton model repository:
         models/
           my_pinn/
             1/
               model.onnx
             config.pbtxt

  3. Start Triton server:
         docker run --gpus=all nvcr.io/nvidia/tritonserver:24.01-py3 \\
           tritonserver --model-repository=/models

  4. Query via HTTP or gRPC:
         import tritonclient.http as triton
         client = triton.InferenceServerClient("localhost:8000")
         result = client.infer("my_pinn", inputs)

  Latency improvement vs PINNeAPPle serve:  5–20×

─── Why this migration is worth it ──────────────────────────

  Research prototype → production deployment is the hardest step
  in any ML project. PhysicsNeMo solves the hard parts for you:

  ✓ Multi-GPU orchestration (tested to 512 GPUs at NVIDIA)
  ✓ Mixed-precision training (FP16/BF16 with loss scaling)
  ✓ Checkpointing + fault tolerance (job can be interrupted)
  ✓ Hydra-based experiment management (no more argparse scripts)
  ✓ NVIDIA Triton integration (enterprise inference)
  ✓ NVIDIA Omniverse integration (real-time 3D visualization)
  ✓ Commercial support + SLA guarantees

─── Industry use cases PhysicsNeMo is known for ─────────────

  1. Turbomachinery (Siemens, GE): RANS surrogate, 100× faster
     than FEM on design iteration

  2. Reservoir simulation (Shell, Eni): Darcy flow surrogate,
     uncertainty quantification on field data

  3. Weather prediction (ECMWF, NVIDIA): FourierCastNet,
     SFNO — global forecast at 1° resolution in seconds

  4. Structural health monitoring (aerospace): PINN + sensor
     fusion, real-time damage detection

  5. CFD surrogate (BMW, Toyota): Navier-Stokes on design
     parameter sweep — 10,000 configurations in a week

─── Full documentation ───────────────────────────────────────

  NVIDIA PhysicsNeMo docs:
    https://docs.nvidia.com/deeplearning/physicsnemo/

  GitHub:
    https://github.com/NVIDIA/physicsnemo

  Migration guide (this repo):
    pinneaple_learning/physicsnemo_roadmap/migration_guide.md

  Side-by-side code examples:
    examples/vs_physicsnemo/
"""
