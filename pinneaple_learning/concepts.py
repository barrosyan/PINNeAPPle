"""ConceptGuide — in-code explanations of core Physics AI concepts."""

from __future__ import annotations

_CONCEPTS: dict[str, dict] = {
    "pinn": {
        "title": "Physics-Informed Neural Network (PINN)",
        "one_liner": "A neural network trained to satisfy a PDE, not just fit data.",
        "explanation": """
  A PINN replaces (or augments) a numerical solver by encoding the governing
  equation directly into the loss function.

  Standard neural net training:
      loss = ||NN(x) − y_data||²

  PINN training:
      loss = λ_pde · ||residual(NN, x_col)||²   ← PDE satisfied inside domain
           + λ_bc  · ||NN(x_bc) − u_bc||²        ← boundary conditions
           + λ_ic  · ||NN(x_ic) − u_ic||²        ← initial conditions

  The key ingredient is automatic differentiation:
      NN computes u(x,t)
      autograd computes ∂u/∂t, ∂²u/∂x², etc. exactly

  Why use it?
  - Meshfree: no finite-element mesh required
  - Embeds physics: cannot violate conservation laws if residual → 0
  - Differentiable everywhere: useful for inverse problems and optimisation
  - Works with sparse / noisy data combined with physics prior

  Limitations vs classical solvers:
  - Slower to train than a single FEM solve
  - Hard to get high accuracy (||residual|| < 1e-6) without care
  - Stiff PDEs (large Re, fast waves) are challenging
  → PhysicsNeMo addresses many of these with optimised CUDA kernels
    """,
        "code_example": """
  # Minimal 1D Poisson PINN — u'' = f, u(0)=u(1)=0
  import torch, torch.nn as nn

  net = nn.Sequential(nn.Linear(1,64), nn.Tanh(), nn.Linear(64,64), nn.Tanh(), nn.Linear(64,1))
  opt = torch.optim.Adam(net.parameters(), lr=1e-3)

  for _ in range(5000):
      x = torch.rand(300, 1, requires_grad=True)
      u = net(x) * x * (1-x)               # hard Dirichlet BC
      u_x  = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
      u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
      loss = (u_xx - (-torch.pi**2 * torch.sin(torch.pi * x))).pow(2).mean()
      loss.backward(); opt.step(); opt.zero_grad()
    """,
        "see_also": [
            "examples/getting_started/01_harmonic_oscillator.py",
            "templates/01_basic_pinn.py",
        ],
        "tier": 1,
    },

    "neural_operator": {
        "title": "Neural Operator (FNO, DeepONet)",
        "one_liner": "A network that learns a mapping between function spaces, not just point values.",
        "explanation": """
  A PINN predicts u(x) for a fixed PDE instance.
  A neural operator learns the map  F: (PDE parameters / IC) → solution function.

  Once trained, a neural operator can predict the solution for any new input
  in milliseconds — replacing thousands of PINN training runs.

  Two main architectures:

  1. FNO (Fourier Neural Operator)
     - Input: discretised function on a grid
     - Key idea: learn convolutions in Fourier space → O(N log N) complexity
     - Best for: regular grids, 1D/2D/3D PDEs (heat, Burgers, NS)
     - PhysicsNeMo: has highly optimised FNO / SFNO / AFNO implementations

  2. DeepONet
     - Input: (branch) function evaluated at sensor points + (trunk) query point
     - Key idea: separate encoding of the input function and evaluation location
     - Best for: operator learning with sparse / irregular sensor data
     - PhysicsNeMo: has DeepONet with physics constraints

  When to use operator learning vs PINN?
  - PINN:    single instance, small domain, inverse problems, limited data
  - FNO:     many instances, parametric families, regular grids, large data
  - DeepONet: irregular sensors, parametric forcing / ICs
    """,
        "code_example": """
  # FNO operator: initial condition u0(x) → solution u(x,T)
  from pinneaple_models.neural_operators.fno import FNO1d

  fno = FNO1d(n_modes=16, hidden_channels=32, n_layers=4,
              in_channels=1, out_channels=1)
  # Input: (batch, n_x, 1) — the initial condition
  # Output: (batch, n_x, 1) — the solution at time T
    """,
        "see_also": [
            "templates/19_fno_neural_operator.py",
            "templates/20_deeponet_surrogate.py",
        ],
        "tier": 2,
    },

    "physics_ai_pipeline": {
        "title": "Physics AI Pipeline",
        "one_liner": "The full workflow from problem definition to validated, deployed model.",
        "explanation": """
  A complete Physics AI pipeline has six stages:

  1. PROBLEM DEFINITION
     - What PDE governs the physics?
     - What are the domain, BCs, ICs, and parameters?
     - What do you want to predict? (forward / inverse / surrogate)
     → Tools: pinneaple_environment presets, pinneaple_problemdesign

  2. DATA & GEOMETRY
     - Collocation points for the PDE residual
     - Boundary / initial condition data
     - Optionally: measurement data from sensors or simulations
     → Tools: pinneaple_geom (CSG, STL, mesh), pinneaple_data (UPD, Zarr)

  3. MODEL SELECTION
     - PINN for single instances, inverse problems
     - FNO / DeepONet for operator learning (many instances)
     - GNN / MeshGraphNet for unstructured meshes
     - ROM (POD/DMD) for linear / weakly nonlinear systems
     → Tools: pinneaple_models registry, pinneaple_arena benchmarks

  4. TRAINING
     - Physics loss + BC/IC losses
     - Adaptive learning rate, gradient clipping
     - Active learning to focus on high-residual regions
     - DDP for large problems
     → Tools: pinneaple_train, pinneaple_data active learning

  5. VALIDATION
     - PDE residual map: is ||residual|| < tolerance everywhere?
     - Conservation law check: does ∫u dx match theory?
     - BC consistency: does u = g on ∂Ω?
     - Comparison with FEM/FDM reference solution
     → Tools: pinneaple_validate, pinneaple_solvers

  6. DEPLOYMENT
     - Export to ONNX / TorchScript for production inference
     - REST serving via FastAPI
     - Digital twin with live sensor assimilation
     - MIGRATE to PhysicsNeMo for NVIDIA-stack production
     → Tools: pinneaple_export, pinneaple_serve, pinneaple_digital_twin
       Then: NVIDIA PhysicsNeMo + Triton + Omniverse
    """,
        "code_example": """
  # Minimal end-to-end pipeline
  from pinneaple_environment import BurgersPreset
  from pinneaple_validate import PhysicsValidator
  from pinneaple_export.onnx_exporter import ONNXExporter

  # 1. Problem
  preset = BurgersPreset(nu=0.01)

  # 2-4. Data + model + training (preset handles this)
  model, history = preset.train(n_epochs=5000)

  # 5. Validate
  validator = PhysicsValidator.from_preset(preset, model)
  report = validator.run()
  print(report.summary())

  # 6. Export
  ONNXExporter(model).export("burgers_surrogate.onnx")
    """,
        "see_also": [
            "pinneaple_learning/tier2_experimenter/01_pipeline_anatomy.py",
            "templates/32_physics_validation.py",
            "templates/23_model_export.py",
        ],
        "tier": 2,
    },

    "physicsnemo": {
        "title": "NVIDIA PhysicsNeMo — Production Physics AI",
        "one_liner": "The industry-leading platform for Physics AI at scale. Your destination after PINNeAPPle.",
        "explanation": """
  PhysicsNeMo (formerly Modulus) is NVIDIA's production-grade Physics AI platform.
  It is the right tool when you need:

  SCALE
  - Multi-GPU and multi-node training with NVLink / InfiniBand
  - FSDP and tensor parallelism for billion-parameter models
  - Optimised CUDA kernels for PINN residuals (10-100x faster than pure PyTorch)

  MODELS
  - FNO, SFNO, AFNO (optimised for NVIDIA GPUs)
  - MeshGraphNet for very large unstructured meshes (>1M nodes)
  - FourCastNet for global weather/climate forecasting
  - DoMINO for domain decomposition at industrial scale

  DEPLOYMENT
  - Native Triton Inference Server integration
  - TensorRT optimisation for maximum throughput
  - Omniverse digital twin platform integration
  - NVIDIA DOCA / BlueField DPU data-path acceleration

  ECOSYSTEM
  - Validated against commercial CFD codes (Fluent, STAR-CCM+, OpenFOAM)
  - Certified for automotive, aerospace, energy, and semiconductor use cases
  - Enterprise support and SLA from NVIDIA

  WHEN TO MIGRATE
  Use PINNeAPPle until:
    ✓ Your model converges and physics residuals are below tolerance
    ✓ You have validated against a reference solver
    ✓ You know which architecture works best for your problem
    ✓ You are ready to scale to full industrial domains

  Then move to PhysicsNeMo for the production run.

  What PhysicsNeMo does NOT have (where PINNeAPPle complements):
  - Built-in UQ (MC Dropout, Conformal Prediction)
  - Digital twin with arbitrary sensor stream assimilation (EnKF)
  - LLM-assisted problem design
  - Multi-domain presets (finance, pharmacokinetics, ecology)
  - Direct model/architecture benchmarking framework
    """,
        "code_example": """
  # PhysicsNeMo equivalent of a Burgers PINN
  # (requires: pip install physicsnemo)

  # from physicsnemo.sym.geometry.primitives_1d import Line1D
  # from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
  # from physicsnemo.sym.solver import Solver
  # from physicsnemo.sym.domain import Domain
  # ... (see PhysicsNeMo documentation)

  # Migration checklist:
  # 1. Replace pinneaple_models with physicsnemo.models
  # 2. Replace pinneaple_pinn residuals with physicsnemo.sym PDEs
  # 3. Replace pinneaple_train with physicsnemo Solver
  # 4. Use Triton for inference instead of pinneaple_serve
  # See: pinneaple_learning/physicsnemo_roadmap/migration_guide.md
    """,
        "see_also": [
            "pinneaple_learning/physicsnemo_roadmap/README.md",
            "pinneaple_learning/physicsnemo_roadmap/migration_guide.md",
            "examples/vs_physicsnemo/",
        ],
        "tier": 3,
    },

    "uq": {
        "title": "Uncertainty Quantification (UQ)",
        "one_liner": "How confident is the model in its prediction?",
        "explanation": """
  A deterministic PINN gives one answer. UQ gives a range of answers
  and tells you how trustworthy each prediction is.

  Three methods available in PINNeAPPle:

  1. MC Dropout (fast, approximate)
     - Keep dropout active at inference time
     - Run N forward passes → mean ± std
     - Cheap: no extra training needed

  2. Deep Ensemble (gold standard)
     - Train N independent models with different random seeds
     - Mean and variance across predictions
     - More expensive but more reliable

  3. Conformal Prediction (distribution-free guarantee)
     - Calibrate on a held-out set
     - Guarantees: P(true value inside interval) ≥ 1−α
     - Coverage guarantee holds regardless of model quality

  Why UQ matters in Physics AI:
  - Physical simulations may be wrong in unexplored regions
  - Sensors are noisy — prediction near sensor gaps is uncertain
  - Regulators and safety-critical applications require confidence bounds
    """,
        "code_example": """
  from pinneaple_uq.mc_dropout import MCDropoutEstimator
  from pinneaple_uq.ensemble import DeepEnsemble
  from pinneaple_uq.conformal import ConformalPredictor

  # MC Dropout
  mc = MCDropoutEstimator(model=my_model, n_samples=100)
  mean, std = mc.predict(x_test)

  # Conformal (90% coverage)
  cp = ConformalPredictor(model=my_model, alpha=0.10)
  cp.calibrate(x_cal, y_cal)
  lower, upper = cp.predict_interval(x_test)
    """,
        "see_also": ["templates/16_uncertainty_quantification.py"],
        "tier": 2,
    },

    "inverse_problem": {
        "title": "Inverse Problem",
        "one_liner": "Given observations of the solution, recover the unknown physical parameters.",
        "explanation": """
  Forward problem: given parameters θ, compute solution u(x; θ).
  Inverse problem: given noisy observations y ≈ u(x_obs; θ_true), recover θ_true.

  Examples:
  - Given temperature measurements, find the thermal conductivity k
  - Given velocity field, find the viscosity ν
  - Given stress measurements, find the elastic modulus E

  PINNeAPPle supports two inverse approaches:

  1. Gradient-based (differentiable physics):
     - Make θ a learnable parameter
     - Minimise data misfit + physics residual jointly
     - Works for smooth, well-posed problems

  2. Ensemble Kalman Inversion (EKI):
     - Derivative-free: works when the forward model is a black box
     - Ensemble of parameter candidates evolves toward posterior
     - Naturally provides uncertainty on the estimated θ

  Why it matters:
  Inverse problems are everywhere in engineering:
  - Non-destructive evaluation (find defect location)
  - Geophysics (find subsurface properties)
  - Biomechanics (characterise tissue properties from strain data)
    """,
        "code_example": """
  from pinneaple_inverse.eki import EKISolver, EKIConfig

  config = EKIConfig(n_ensemble=30, n_iterations=10,
                     obs_noise_std=0.02, prior_mean=[0.5], prior_std=[0.3])
  solver = EKISolver(config=config,
                     forward_fn=my_forward_model,
                     observations=y_obs)
  result = solver.run()
  print("Estimated θ:", result["ensemble_final"].mean(axis=0))
    """,
        "see_also": ["templates/18_inverse_problem.py"],
        "tier": 2,
    },

    "digital_twin": {
        "title": "Digital Twin",
        "one_liner": "A live, continuously updated virtual replica of a physical system.",
        "explanation": """
  A digital twin combines:
  - A PINN / surrogate model that encodes the physics
  - A real-time data assimilation algorithm that updates the model state
    as new sensor measurements arrive
  - An anomaly detector that flags when the system deviates from predictions

  Key components in PINNeAPPle:
  - DigitalTwinRuntime: orchestrates the update loop
  - StateAssimilator (EnKF): fuses sensor data with model predictions
  - AnomalyDetector: flags deviations in innovation (data − prediction)
  - SyntheticSensorStream: replay pre-recorded sensor data offline

  Industrial applications:
  - Turbine blade temperature monitoring
  - Structural health monitoring (bridges, aircraft frames)
  - Thermal management of electronics (CPU, server racks)
  - Wind farm power prediction and turbine control

  PhysicsNeMo + Omniverse:
  For production digital twins at industrial scale, PhysicsNeMo integrates
  with NVIDIA Omniverse (3D visualisation), Isaac Sim (robotics), and
  NVIDIA DOCA for edge data ingestion.  PINNeAPPle is where you prototype
  and validate the physics model before deploying on that stack.
    """,
        "code_example": """
  from pinneaple_digital_twin.runtime import DigitalTwinRuntime, DigitalTwinConfig

  runtime = DigitalTwinRuntime(surrogate=model,
                               assimilator=assimilator,
                               anomaly_detector=detector,
                               config=dt_config)
  for obs in sensor_stream:
      state, innovation = runtime.update(obs, t=obs["timestamp"])
    """,
        "see_also": ["templates/24_digital_twin.py"],
        "tier": 3,
    },

    "rom": {
        "title": "Reduced Order Model (ROM)",
        "one_liner": "Compress a high-dimensional simulation into a few dominant modes.",
        "explanation": """
  A ROM reduces a PDE with N degrees of freedom to a system with r ≪ N
  modes, where r captures the dominant physics.

  Three classical methods in PINNeAPPle:

  1. POD (Proper Orthogonal Decomposition)
     - SVD of snapshot matrix → orthogonal modes φ_i
     - Reconstruct: u(x,t) ≈ Σ_i a_i(t) φ_i(x)
     - Best for: linear / weakly nonlinear problems, many snapshots

  2. DMD (Dynamic Mode Decomposition)
     - Fit a linear operator A such that u_{t+1} ≈ A u_t
     - Eigenvalues of A give frequencies and growth rates
     - Best for: future state prediction, oscillatory flows

  3. HAVOK (Hankel-based Koopman)
     - Delay-embedding of a single sensor signal
     - Identifies forcing structure in deterministic chaos
     - Best for: single-sensor data, intermittently forced systems

  When to use ROM vs PINN vs operator learning:
  - ROM:  linear dynamics, many high-fidelity snapshots available
  - PINN: unknown parameters, limited data, nonlinear physics
  - FNO:  parametric families, regular grids, fast online queries
    """,
        "code_example": """
  from pinneaple_models.rom.pod import PODReducedModel
  from pinneaple_models.rom.dmd import DMDModel

  pod = PODReducedModel(n_components=10)
  pod.fit(snapshot_matrix)     # (n_space, n_time)
  X_reconstructed = pod.reconstruct(snapshot_matrix)

  dmd = DMDModel(n_modes=10, dt=0.05)
  dmd.fit(snapshot_matrix)
  X_future = dmd.predict(n_steps=50, x0=snapshot_matrix[:, -1])
    """,
        "see_also": ["templates/26_rom_pod_dmd.py"],
        "tier": 2,
    },
}


class ConceptGuide:
    """Interactive concept reference for Physics AI topics."""

    @staticmethod
    def get(concept: str) -> dict:
        key = concept.lower().replace(" ", "_").replace("-", "_")
        if key not in _CONCEPTS:
            available = ", ".join(sorted(_CONCEPTS.keys()))
            raise KeyError(
                f"Unknown concept '{concept}'. Available: {available}"
            )
        return _CONCEPTS[key]

    @staticmethod
    def print(concept: str) -> None:
        info = ConceptGuide.get(concept)
        print(f"\n{'='*62}")
        print(f"  {info['title']}")
        print(f"  Tier {info['tier']} concept")
        print(f"{'='*62}")
        print(f"\n  {info['one_liner']}\n")
        print(info["explanation"])
        if info.get("code_example"):
            print("\nCode example:")
            print(info["code_example"])
        if info.get("see_also"):
            print("\nSee also:")
            for ref in info["see_also"]:
                print(f"  • {ref}")
        print()


def explain(concept: str) -> None:
    """Print a structured explanation of a Physics AI concept.

    Parameters
    ----------
    concept : str
        One of: 'pinn', 'neural_operator', 'physics_ai_pipeline',
                'physicsnemo', 'uq', 'inverse_problem', 'digital_twin', 'rom'
    """
    ConceptGuide.print(concept)


def list_topics() -> None:
    """Print all available concept topics."""
    print("\nAvailable Physics AI concepts in pinneaple_learning:\n")
    by_tier: dict[int, list] = {1: [], 2: [], 3: []}
    for key, info in sorted(_CONCEPTS.items()):
        by_tier[info["tier"]].append((key, info["title"], info["one_liner"]))

    labels = {1: "Tier 1 — Explorer", 2: "Tier 2 — Experimenter", 3: "Tier 3 — Builder"}
    for tier, items in by_tier.items():
        print(f"  {labels[tier]}")
        for key, title, one_liner in items:
            print(f"    pl.explain('{key}')")
            print(f"      {title}")
            print(f"      → {one_liner}\n")
