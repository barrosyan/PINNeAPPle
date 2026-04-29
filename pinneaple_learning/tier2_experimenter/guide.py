"""Tier 2 Experimenter — architecture selection and pipeline anatomy guides."""

from __future__ import annotations


def architecture_guide() -> None:
    """Print the Physics AI architecture selection guide."""
    print(_ARCHITECTURE_GUIDE)


def pipeline_anatomy() -> None:
    """Print the full Physics AI pipeline anatomy for Tier 2."""
    print(_PIPELINE_ANATOMY)


# ---------------------------------------------------------------------------

_ARCHITECTURE_GUIDE = """
╔══════════════════════════════════════════════════════════════╗
║       Architecture Selection Guide — Tier 2: Experimenter   ║
╚══════════════════════════════════════════════════════════════╝

─── The four main families ───────────────────────────────────

  1. PINN  (Physics-Informed Neural Network)
     • Single problem, single parameter set
     • Solves the PDE via residual loss + autograd
     • Best for: inverse problems, irregular geometries,
                 data-sparse regimes, low-dim PDEs
     • Key file: templates/01_basic_pinn.py

  2. FNO  (Fourier Neural Operator)
     • Learns the operator  a(x) → u(x)  across parameters
     • Operates in Fourier space → global receptive field
     • Best for: parametric PDEs on regular grids (Burgers,
                 Navier-Stokes, Darcy), fast repeated evaluations
     • Key file: templates/19_fno_neural_operator.py

  3. DeepONet  (Deep Operator Network)
     • Branch net encodes the input function
     • Trunk net encodes the query coordinates
     • Best for: control/forcing → response problems,
                 arbitrary output locations
     • Key file: templates/20_deeponet_surrogate.py

  4. MeshGraphNet  (Graph Neural Network on meshes)
     • Graph topology = mesh connectivity
     • Message-passing over edges learns local PDE stencil
     • Best for: unstructured CFD meshes, FEM surrogate,
                 CAD-native geometry
     • Key file: templates/25_gnn_mesh.py

─── Decision tree ────────────────────────────────────────────

  Do you have a single specific problem with known params?
    YES → PINN
    NO  → Is your domain a regular grid?
            YES → Does the operator act on a function input?
                    YES → FNO
                    NO  → PINN or FNO
            NO  → Is the geometry unstructured / CAD-native?
                    YES → MeshGraphNet
                    NO  → DeepONet

─── Combining architectures ─────────────────────────────────

  PINN + FNO:     Pre-train FNO for fast initialisation,
                  then fine-tune with PINN physics loss
  PINN + data:    Hybrid loss L = L_pde + L_data
  DeepONet + UQ:  DeepEnsemble of DeepONets for uncertainty
  GNN + ROM:      Graph snapshot → POD reduction → fast ROM

─── Benchmark them side by side ─────────────────────────────

  Use pinneaple_arena:

      from pinneaple_arena import ArenaRunner, NativeBenchmarkTask
      # see templates/28_arena_benchmark.py

  ArenaRunner runs each model on the same problem and generates
  a leaderboard (accuracy × latency × memory).

─── Uncertainty Quantification ──────────────────────────────

  Any architecture can be wrapped with UQ:

  MC Dropout:       cheapest, add Dropout layers + keep dropout
                    enabled at inference, run N forward passes
  Deep Ensemble:    train M independent networks, take mean/std
                    — most reliable, costs M× compute
  Conformal:        post-hoc calibration, coverage guarantees,
                    no architecture changes needed

  Key file: templates/16_uncertainty_quantification.py

─── Inverse problems ────────────────────────────────────────

  Two approaches:

  Gradient-based:  unknown parameter θ as a trainable torch.nn.Parameter,
                   include L_data in the loss, backprop updates θ
  EKI:             Ensemble Kalman Inversion — derivative-free,
                   handles non-smooth objectives

  Key file: templates/18_inverse_problem.py

─── Active learning ─────────────────────────────────────────

  Instead of uniform collocation, focus points where residual
  is largest (ResidualActiveSampler) or uncertainty is highest
  (VarianceActiveSampler). Often reduces N_COL by 50%.

  Key file: templates/21_active_learning.py
"""

_PIPELINE_ANATOMY = """
╔══════════════════════════════════════════════════════════════╗
║          Physics AI Pipeline Anatomy — Tier 2               ║
╚══════════════════════════════════════════════════════════════╝

A complete Physics AI experiment has six stages:

──────────────────────────────────────────────────────────────
STAGE 1 — Problem Definition
──────────────────────────────────────────────────────────────

  Specify the PDE, domain, BCs, ICs, and (optionally) the
  parameter space you want to generalise over.

  PINNeAPPle tools:
    • pinneaple_pdb         — PDE database (100+ named PDEs)
    • pinneaple_geom        — CSG domain builder
    • pinneaple_problemdesign — LLM-assisted problem spec

  Questions to answer at this stage:
    1. Is the domain regular (rectangle) or irregular (CAD)?
    2. Is the problem forward (known PDE, find u) or inverse?
    3. Do you have any measurement data?
    4. What accuracy do you need?  (ε < 1e-3 vs ε < 1e-5)

──────────────────────────────────────────────────────────────
STAGE 2 — Architecture Selection
──────────────────────────────────────────────────────────────

  (see architecture_guide() for the full decision tree)

  Rule of thumb:
    • Single instance, small domain → PINN
    • Many parameters, regular grid → FNO
    • Forcing/control → response    → DeepONet
    • Complex unstructured mesh     → MeshGraphNet

──────────────────────────────────────────────────────────────
STAGE 3 — Training
──────────────────────────────────────────────────────────────

  Key decisions:

  Collocation strategy:
    • Uniform random:   baseline, always works
    • Latin Hypercube:  better coverage in high dimensions
    • Active learning:  adaptive, most efficient
    → templates/21_active_learning.py

  Optimiser schedule:
    • Adam (lr=1e-3) for ~80% of epochs (fast convergence)
    • L-BFGS for final ~20% (high accuracy)
    → templates/01_basic_pinn.py

  Loss balancing:
    • Manual λ weights  (simple but needs tuning)
    • NTK-based auto-weighting (theoretical justification)
    → pinneaple_train.LossBalancer

  Transfer learning (if you have a related solved problem):
    → templates/17_transfer_learning.py

──────────────────────────────────────────────────────────────
STAGE 4 — Validation
──────────────────────────────────────────────────────────────

  Never skip this. A low training loss does NOT mean a
  physically correct solution.

  Checks:
    ✓ Relative L2 error vs reference solution
    ✓ PDE residual at held-out test points
    ✓ Boundary condition residual
    ✓ Conservation law check (energy, mass, momentum)

  PINNeAPPle tool:
    → templates/32_physics_validation.py
    → pinneaple_validate.PhysicsValidator

──────────────────────────────────────────────────────────────
STAGE 5 — Uncertainty Quantification
──────────────────────────────────────────────────────────────

  Before using predictions for decisions, quantify how
  confident the model is.

    • Where is the uncertainty highest?  (near boundaries,
      sparse data regions, out-of-distribution inputs)
    • Does the confidence interval contain the true solution?
      (calibration check with conformal prediction)

  → templates/16_uncertainty_quantification.py

──────────────────────────────────────────────────────────────
STAGE 6 — Benchmarking
──────────────────────────────────────────────────────────────

  Compare your approach against baselines on the SAME problem.

  Metrics:
    • Accuracy:  relative L2 error
    • Speed:     training time, inference latency
    • Memory:    peak GPU RAM
    • Sample efficiency: accuracy vs number of collocation pts

  → templates/28_arena_benchmark.py
  → pinneaple_arena.ArenaRunner

──────────────────────────────────────────────────────────────
WHEN YOU ARE READY TO SCALE
──────────────────────────────────────────────────────────────

  Once Stage 1–6 are done and you have a validated approach,
  you are ready for Tier 3 (Builder) and ultimately for
  NVIDIA PhysicsNeMo.

      import pinneaple_learning as pl
      pl.learning_path(tier=3)
"""
