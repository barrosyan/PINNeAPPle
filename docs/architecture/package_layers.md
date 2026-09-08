# Package Layers

This is the verified mapping from pipeline stage to package, replacing an
earlier stub whose "Solver" row was inaccurate (it pointed at
`pinneapple_simulation.numerical_solvers`, which is a classical-solver
package, not the training-policy layer described in
[Solver](../core_concepts/solver.md)).

## Core pipeline

| Layer | Package | Notes |
|---|---|---|
| Problem definition | `pinneapple_physics.pde_environment` | `ProblemSpec`, `ProblemBuilder`, presets, BC/IC types |
| Geometry + sampling | `pinneapple_design.geometry` | SDF/CSG, `PhysicsDomain2D`/`3D`, mesh, `MeshCollocator` |
| Data container | `pinneapple_data` | `PhysicalSample`, collation, collocation, active learning |
| Model architectures | `pinneapple_neural.architectures` | `BaseModel` subclasses + `ModelRegistry` |
| Physics / residuals + constraints | `pinneapple_physics.pinn_solver`, `pinneapple_physics.symbolic_pde` | `compile_problem`, autograd ops, `DoMINO`; SymPy-based residuals |
| **Solver — training policy** | `pinneapple_neural.trainer` | `Trainer`, `TwoPhaseTrainer`, `TimeMarchingTrainer`, `CausalPINNTrainer`, `DDPPINNTrainer`, loss-weight balancers |
| Execution backend | `pinneapple_tools.compute_backends` | PyTorch (default) / JAX abstraction |
| Benchmarking | `pinneapple_tools.benchmark_suite` | `Arena`, `PINNArenaBenchmark`, transfer/meta benchmark pipelines |
| Research pipeline | `pinneapple_tools.hpo_experiments` | literature discovery, knowledge base, reproduction |
| Structured storage | `pinneapple_pdb` | Earth-observation-backed dataset builder + named-benchmark catalog |

Classical numerical solvers (`pinneapple_simulation.numerical_solvers` —
FEM/FDM/FVM/spectral/SPH/LBM, plus OpenFOAM/FEniCS bridges) are a distinct
capability from the training-policy "Solver": they generate reference/
comparison data (or couple to external tools), rather than optimizing a
neural network's weights. A `pinneapple_neural.trainer.Trainer` can train
*against* data a `numerical_solvers` run produced, but the two are not the
same layer.

## Extension packages

These sit on top of the core pipeline, consuming its outputs rather than
being required by it:

| Package | Adds |
|---|---|
| `pinneapple_analysis` | Uncertainty quantification, physics validation, inverse problems |
| `pinneapple_adaptation` | Transfer learning and meta-learning across PDE families |
| `pinneapple_simulation` | Classical numerical solvers, particle dynamics, external-solver bridges |
| `pinneapple_systems` | Time series, co-simulation, digital twins |
| `pinneapple_design.design_optimizer` | Adjoint/Pareto/Bayesian/evolutionary shape optimization |
| `pinneapple_problemdesign` | Natural-language-to-PDE problem design agent |
| `pinneapple_arena` | YAML/JSON-driven multi-model benchmark runner and leaderboard |
| `pinneapple_quantum` | Quantum and quantum-inspired PINNs |
| `pinneapple_worldmodel` | Generalist physics foundation models |
| `pinneapple_hub` | Model hub client (`push_to_hub`/`from_pretrained`) |
| `pinneapple_registry` | Local, self-hosted artifact/experiment registry |
| `pinneapple_llm` | LLM-assisted pipeline drafting, gated by `PhysicsGuardrail` |
| `pinneapple_blender` | Field/trajectory export to `.ply` and Blender scene rendering |
| `pinneapple_perception` | Extracting physics observations from images/video/audio |
| `pinneapple_app` | FastAPI + frontend web app for browser-based benchmarking |

## Compatibility shims

`pinneapple_models`, `pinneapple_solvers`, and `pinneapple_train` are
re-export shims (for `pinneapple_neural.architectures`,
`pinneapple_simulation.numerical_solvers`, and `pinneapple_neural.trainer`
respectively) kept only so older import paths keep working.
