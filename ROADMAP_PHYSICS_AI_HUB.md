# PINNeAPPle → "the Hugging Face of Physics AI": full roadmap

Written 2026-09-04, during the `feat/openfoam-binary-wale-adaptive-stochastic`
branch (see its commits for everything already shipped). This is the
complete plan for the full scope requested — a from-scratch, evidence-based
audit of the whole library; automatic hyperparameter tuning; a Blender
bridge; an LLM-assisted pipeline-generation module; and a physics-grounded
anti-hallucination guarantee layer — broken into phases with what's
actually achievable in a single working session (P0, done or in progress
now) versus what genuinely needs more time, a team, or external
infrastructure this repository alone cannot provide (P1–P3).

**Honesty note, load-bearing for the whole plan:** "a perfect and complete
audit" of an ~90-architecture, 8-mega-module framework, and "0%
hallucination guaranteed," are not real deliverables — no audit of a
codebase this size is ever complete (only ever more or less thoroughly
sampled), and no software can guarantee an LLM never hallucinates. What
*is* real and buildable: broad, evidence-based coverage (test as much as
time allows, report exactly what was and wasn't checked), and a
**verification layer that catches and blocks ungrounded results** rather
than a metaphysical promise that none occur. Every item below is scoped to
what can actually be shipped and checked, not to what sounds complete.

---

## P0 — done or in progress this session (branch `feat/openfoam-binary-wale-adaptive-stochastic`)

| Item | Status | Where |
|---|---|---|
| Binary OpenFOAM FoamFile reader + polyMesh reconstruction | ✅ done, validated byte-for-byte against a real 244 MB LES case | `pinneapple_simulation/external_solvers/openfoam/{binary_reader,mesh_reader}.py` |
| `openfoam_case_to_upd` domain-mistag fix (`"grid"` → `"mesh"` + real `geometry.nodes`) | ✅ done | `openfoam/field_reader.py` |
| `MultiPeriodicBC` (fixes the 2+-axis chaining bug in `PeriodicBC`) | ✅ done | `pinneapple_physics/symbolic_pde/bc.py` |
| `WALEResiduals` — first 3D/LES turbulence closure in the repo | ✅ done | `pinneapple_physics/pde_environment/turbulence_presets.py` |
| `body_force_fn` hook on `navier_stokes_incompressible` | ✅ done | `pinneapple_physics/pinn_solver/compiler/compile.py` |
| `AdaptiveWeights` (relative-to-hardest-term loss balancing, with the trivial-solution pitfall documented) | ✅ done | `pinneapple_physics/pinn_solver/compiler/loss.py` |
| Stochastic/latent-conditioned PINN utilities (`LatentConditionedModel`, `ensemble_forward`, `mean_covariance_loss`) | ✅ done | `pinneapple_physics/pinn_solver/stochastic.py` |
| `solve_pde()` — was calling a nonexistent `TrainConfig(n_epochs=...)`/`Trainer(...).train()`; now a real, tested Adam loop with auto-sampled BC/IC/data batches for `"all"`/`"callable"` selector conditions | ✅ done, tested against a real preset (`burgers_1d`, loss verified decreasing) | `pinneapple_physics/__init__.py` |
| `UPDDataset.save(format="zarr")` — was importing a `ZarrWriteSpec` that doesn't exist | ✅ done | `pinneapple_data/serialization.py` |
| CGNS / Exodus II / Fluent-mesh(ASCII, geometry-only) / Abaqus `.inp`+`.odb`-bridge readers | ✅ code done; ⚠️ **not validated against a real file from a real writer** (none available this session) — self-consistency only | `pinneapple_simulation/external_solvers/cfd_formats/` |
| `templates/33_rans_turbulence.py` (`KOmegaSSTConfig` import doesn't exist, plus a second bug: `kwsst.channel_residuals(...)` didn't exist either — the template assumed an API that was never built) | ✅ done — fixed the import, and added a real `KOmegaSSTResiduals.channel_residuals()` (1D fully-developed-channel reduction) since that's what the template actually needed; ran the template live, confirmed it trains without error | `templates/33_rans_turbulence.py`, `pde_environment/turbulence_presets.py` |
| `Trainer.fit`'s validation loop breaks `create_graph=True` residuals (ran inside `torch.no_grad()`) | ✅ done — new `TrainConfig.physics_aware_validation` (default True) runs validation under `torch.enable_grad()`; tested end-to-end with a real `compile_problem` residual through `Trainer.fit` (no crash, previously did) | `pinneapple_neural/trainer/trainer.py` |
| Model hub (`push_to_hub`/`from_pretrained` + `ModelCard`) | ✅ done — full local round-trip tested (push, download, reconstruct, weights verified identical via a faked-local HF Hub) | new `pinneapple_hub/` |
| CI: Tier A breadth matrix (90 architectures × 47 presets) + Tier B manufactured-solution physics-correctness tests + GitHub Actions workflow | ✅ done and actually run — see `AUDIT_REPORT.md` for the full, real results (62/137 Tier-A failures found and categorised; both Tier-B tests pass) | `tests/test_full_library_matrix.py`, `tests/test_manufactured_solutions.py`, `.github/workflows/tests.yml`, `AUDIT_REPORT.md` |
| `pinneapple.pipeline()` task abstraction | ✅ done, tested (`pp.pipeline("burgers_1d", nu=0.01, epochs=15).predict(...)`) | `pinneapple_physics/__init__.py` |
| k-ω SST 1D channel-flow reduction (`channel_residuals`) | ✅ done (see the template-fix row above); full 3D generalisation of the general 2D `__call__` residual itself is still open, see P2.4 | `pde_environment/turbulence_presets.py` |
| Blender bridge (`.ply`-sequence export, no dependency beyond numpy/matplotlib; a real Blender `bpy` add-on script for the import/render side, subprocess-bridged like the Abaqus `.odb` bridge) | ✅ done and tested (export path; the `bpy`-side add-on could not be executed in this session — no local Blender install — so it's spec-correct/self-consistent but not run inside real Blender, same caveat as the CGNS/Exodus/Fluent readers) | new `pinneapple_blender/` |
| LLM pipeline-generation module + physics guardrail (`pinneapple_llm`) | ✅ done and tested, including the anti-hallucination checks actually rejecting a fabricated preset name and a fabricated kwarg (not just "should work in theory") | new `pinneapple_llm/` |
| LLM-assisted geometry drafting, digital-twin drafting (+ a real Blender-bridged "3D live twin" glue function), literature/repo search (real arXiv+GitHub API results, LLM only summarises what was actually returned) | ✅ done and tested (beyond the original ask — added once the user asked for CAD/research/digital-twin LLM integration specifically) | `pinneapple_llm/{geometry_draft,twin_draft,research}.py` |
| Local LLM provider (Ollama, HTTP API, no SDK) + SQLite conversation log + LoRA fine-tuning driver on logged conversations | ✅ done (Ollama dispatch + conversation store tested locally; `finetune_lora` uses real `transformers`+`peft`+`datasets` APIs but was not run end-to-end this session — no local base model downloaded — so it is code-reviewed/spec-correct, not execution-verified, unlike everything else in this table) | `pinneapple_llm/{local_llm,conversation_store,finetune}.py` |
| Adaptive (Bayesian/TPE via Optuna, random-search fallback) hyperparameter search — the existing `run_parallel_sweep`/`SweepConfig` turned out to already exist and work, but is grid-only, not adaptive; also fixed that shim's own re-export gap (`pinneapple_train` never actually exported `SweepConfig`/`run_parallel_sweep` despite `QUICKSTART.md`'s own example importing them from there) | ✅ done and tested, both the Optuna path and the dependency-free fallback (found and fixed a real bug in the process: `suggest_float`'s `log` parameter is keyword-only, an early version passed it positionally and crashed every trial) | `pinneapple_neural/trainer/adaptive_sweep.py`, `pinneapple_train/__init__.py` |
| **Second audit pass: the pre-existing `tests/` suite** (never run successfully before this session — 4 `ModuleNotFoundError`s aborted collection of the *entire* tree before a single test could run) — fixed all 4 collection-blockers (`pinneapple_pinn` bad import, `pinneapple_train.trainer/.losses/.metrics` submodules missing, `pinneapple_solvers.fft` missing + a silent `FFTProcessor`→`None` swallow bug in the same file, `pinneapple_models.registry` missing), then found and fixed 4 more real bugs the now-runnable suite surfaced: `real_gas_eos.py`'s `NameError` (should be a clean `ImportError` when CoolProp isn't installed), `Trainer.fit()` never returning `"history"`, and — highest severity — `pinneapple_app`'s experiment `progress_cb` doing `ev["epoch"]` unconditionally when the auto-fix advisor loop sends event shapes without that key, **crashing any experiment that used the advisor/retrain feature** (confirmed pre-existing via `git stash`, not introduced this session) | ✅ done — suite went from 0 tests able to run to **0 failures** across the whole thing (`pytest tests/ --ignore=tests/test_full_library_matrix.py --ignore=tests/test_manufactured_solutions.py`); `tests/test_app_backend.py` alone went from 6 failures to 34/34 passing | see `AUDIT_REPORT.md`'s "Zero" section for the full per-bug breakdown; files touched: `pinneapple_train/{trainer,losses,metrics}.py` (new), `pinneapple_solvers/{__init__,fft}.py`, `pinneapple_models/registry.py` (new), `pinneapple_systems/process_components/real_gas_eos.py`, `pinneapple_neural/trainer/trainer.py`, `pinneapple_app/backend/routers/experiments.py`, `pyproject.toml` (`process` extra), 3 test files (`test_sympy_backend.py`, `test_mesh_ops.py`, `test_process_components.py`) |

**Third audit pass** (done): import-smoke-tested all 295 modules across the
previously-untouched `pinneapple_design`, `pinneapple_systems`,
`pinneapple_analysis`, `pinneapple_adaptation`, `pinneapple_tools`, and
`pinneapple_simulation` packages. Found and fixed 2 real bugs: a broken
import in `pinneapple_tools/benchmark_suite/runner/run_models_from_cfg.py`
(referenced a nonexistent `compiler.collate` module; `dict_collate`
actually lives in `compiler.dataset`), and `numerical_solvers/registry.py`'s
`register_all()` being a flat list of imports where one missing optional
dependency (`pywt`, for the wavelet solver) silently aborted registration
of every solver listed after it — `SolverRegistry.list()` returned only 8
of ~26 real solvers as a result; rewritten to import each module
independently (26/26 now register, with only the 2 genuinely-missing-
optional-dep ones reported back instead of eating everything after them).

**Still open**: this pass only checked "does it import," not "does it run
correctly" — upgrading Tier A from architecture-alone/preset-alone
(always paired with `modified_mlp`, 3 epochs, a tiny batch) to a genuine
architecture×preset cartesian product at a more realistic epoch/batch
size, and extending Tier-A-style "build one instance, run one
forward/backward pass" breadth testing to these 6 packages' actual
classes (not just their imports), are both still not done.

---

## Domain specialization: astrophysics/space (chosen initial vertical)

**Decision**: rather than staying a flat catalog of ~50 unrelated
presets, PINNeAPPle's first release picks ONE domain to go deep on and
demonstrably get right end-to-end, before expanding to others. This
session's choice, made explicitly by the user: **astrophysics and space
systems**, covering both research astrophysics and industrial/applied
space engineering (the user specifically asked for both, and separately
asked for space-debris and "similar space areas" to be included).

**Shipped this session**: 7 new, real benchmark presets in
`pinneapple_physics/pde_environment/presets/astrophysics.py`, each with a
literature reference, a real/representative default parameter set (not
placeholder numbers), and — critically — an independently-verified
reference/analytic solution used to prove the compiled residual actually
reproduces the correct physics (see `AUDIT_REPORT.md`'s astrophysics
section for the full per-preset verification table: 16/16 Tier-B checks
pass, and the process caught one real bug — a sign/parity error in the
Lane-Emden n=0 case — before it shipped).

| Preset | Applicability | New compiler kind? |
|---|---|---|
| `kepler_two_body_orbit` | research + industrial (every mission design tool) | yes: `kepler_two_body_orbit` |
| `space_debris_cw_relative_motion` | industrial (conjunction assessment, proximity ops) | yes: `space_debris_cw_relative_motion` |
| `satellite_j2_perturbation` | industrial (LEO/SSO satellite station-keeping) | yes: `satellite_j2_perturbation` |
| `spacecraft_attitude_euler_rotation` | industrial (ADCS design/verification) | yes: `spacecraft_attitude_euler_rotation` |
| `lane_emden_polytrope` | research (stellar structure) | yes: `lane_emden_polytrope` |
| `nfw_dark_matter_potential` | research (galactic dynamics/cosmology) | no — reused existing `poisson` kind |
| `sod_shock_tube_astro` | research (astrophysical hydro-code validation) | yes: `euler_compressible_1d` |

**Byproduct fix**: building the generic ODE-residual pattern for these
presets also fixed 2 pre-existing, unrelated Category-1 "Unsupported PDE
kind" gaps from the Tier A audit above: `sir_ode` and
`pk_two_compartment_ode` now compile and run (Tier A failures: 62→60/137).

**Not done this session** (tracked here so it isn't lost):
- No presets were actually *trained* end-to-end and compared to their
  reference solution after training — Tier B here only proves the
  residual/physics implementation is correct, the same scope as the
  existing Laplace Tier B test. Training a real network on
  `kepler_two_body_orbit` (the fastest-converging candidate) and plotting
  predicted vs. exact orbit would be the natural next demonstration.
- `satellite_j2_perturbation`'s cited secular drift-rate formulas (nodal
  regression, apsidal precession) were not checked against a long,
  many-orbit integrated/trained trajectory — only the instantaneous
  residual and the J2→two-body reduction were verified.
- `lane_emden_polytrope` has no closed-form validation for the
  astrophysically standard n=1.5 (white dwarf) / n=3 (Eddington standard
  model) — only n∈{0,1,5} have exact solutions; those would need a
  numerical reference (e.g. via `pinneapple_simulation`'s IVP solvers)
  instead.
- No N-body (3+ body) gravitational dynamics, no general-relativistic
  content (light bending, gravitational-wave inspiral/post-Newtonian
  orbits), no radiative-transfer/stellar-atmosphere preset, no
  accretion-disk (Shakura-Sunyaev) preset, no cosmological
  perturbation-growth preset — all considered, all deferred to keep this
  session's set small enough to verify properly rather than large and
  unverified. A second batch covering these would be the natural way to
  deepen this specialization before considering it "demonstrably done."
- No UI/documentation/tutorial notebook showcasing this vertical to a
  user browsing presets — the presets exist and are correct, but nothing
  yet highlights "PINNeAPPle is good at astrophysics/space" to someone
  exploring the library for the first time.

---

## P1 — near-term (days, one engineer, no new infrastructure)

### 1.1 Full-library evidenced audit (the honest version of "audit everything")
- **Method has two tiers, because "does it crash" and "is the physics
  right" are different questions** (the user's own distinction: model
  correctness, training correctness, *and* whether the physics losses
  being generated are actually correct):
  - **Tier A — runs-without-crashing (breadth)**: for every registered
    model in `ModelRegistry.list()` (90 names), build it with a small
    `in_dim`/`out_dim`, run one forward + backward pass, assert no
    NaN/exception. For every `numerical_solvers` class, instantiate at
    minimum size and run one step. Cheap, catches API-mismatch bugs (the
    `solve_pde`/zarr/`Trainer.fit` class of bug found in P0), but proves
    nothing about physical correctness — a residual with a sign error
    still "runs."
  - **Tier B — method of manufactured solutions (depth, the actual
    correctness check)**: for every `pde_environment` preset with a
    known closed-form or tabulated solution (`laplace_2d`: pick any
    harmonic function; `poisson_2d`: pick any `u`, forcing term is then
    `-∇²u` by construction; `burgers_1d`: the Cole-Hopf solution; `heat
    _equation_steady`, `advection_diffusion`, ...), plug the *exact*
    solution into `compile_problem`'s own compiled residual directly (no
    training involved) and assert the residual is ~0 to floating-point
    tolerance; then plug in a *deliberately wrong* function and assert
    the residual is measurably nonzero (a residual that returns ~0 for
    everything is exactly as broken as one that's wrong for everything,
    and Tier A cannot tell the two apart). This is the only rigorous way
    to answer "is this physics loss actually the PDE it claims to be" —
    a trained network's loss going down is consistent with a correct
    residual driving it toward the right answer *or* a broken residual
    it can trivially satisfy (the WALE `pde_momentum` weight-collapse
    failure mode documented in `AdaptiveWeights`, P0, is exactly this
    class of trap). Presets without a known analytical solution
    (most real-world CFD/structural presets) can't get a Tier B check
    this way — for those, Tier B instead means comparing a trained
    result against a cited reference dataset via `PhysicsGuardrail`
    (P3.2), the same mechanism this project's own `problem_config.json`
    validation table used by hand.
  - **Training-loop correctness**: for a handful of representative presets
    (one per PDE family: elliptic, parabolic, hyperbolic, Navier-Stokes),
    run a real `Trainer.fit`/`solve_pde` for enough epochs to expect
    convergence on a *tiny* version of the problem, and assert the final
    loss is below a fixed threshold, not just "lower than the first
    epoch" (a loop that trains for 5 epochs and calls "finite and
    decreasing" a pass would not have caught how badly under-trained
    v1-v3 of this very project's own channel-flow surrogate were before
    v4's fixes — "loss decreased" and "loss is now actually small enough
    to trust" are different bars).
- **Deliverable**: `tests/test_full_library_matrix.py` (Tier A, parametrized,
  `pytest -k audit_breadth`), `tests/test_manufactured_solutions.py` (Tier
  B, `pytest -k audit_physics`), + a generated `AUDIT_REPORT.md` table
  (pass/fail per item for both tiers, with the exception message or
  residual-magnitude number for failures) committed alongside.
- **Known-likely findings** (pattern-matched from what P0 already found):
  more `TrainConfig`/`Trainer` API mismatches wherever another module
  wraps them the way `solve_pde` did; more presets whose `value_fn`/
  `selector` combinations don't actually match what their own residual
  branch in `compile.py` expects; broken imports in `templates/` (only one
  found so far, `33_rans_turbulence.py`, but that directory was not
  exhaustively checked).
- **Effort**: ~2–3 days for one engineer to build the harness, run it, and
  triage/fix the P1-severity findings (crashes, wrong-signature calls);
  P2-severity findings (numerically-off-but-not-crashing results,
  physics-questionable defaults) need domain review, not just test-passing.

### 1.2 Automatic hyperparameter tuning
- **What exists today**: `pinneapple_train.run_parallel_sweep`/
  `SweepConfig` (per `QUICKSTART.md`) — a **grid/parallel sweep runner**,
  not an adaptive search (no Bayesian optimization, no successive halving/
  ASHA, no early-stopping-aware budget allocation). Needs to be confirmed
  directly (`pinneapple_train/` audit is part of 1.1) rather than assumed
  from the README.
- **Gap**: no adaptive/automatic tuner. Add `pinneapple_train.tuning`:
  - An Optuna-backed `AdaptiveSweep` (optional dependency, matching the
    project's existing `optional-dependencies` pattern in `pyproject.toml`)
    with TPE sampling + a median/ASHA pruner keyed off the training loop's
    per-epoch loss (needs a pruning callback hook in `Trainer.fit` — a
    small, additive change).
  - A dependency-free fallback (random search + successive halving) for
    users without `optuna` installed, so the feature isn't extras-gated
    entirely.
  - Wire `AdaptiveWeights` (P0) as a tunable itself (`momentum`,
    `max_ratio`) since loss-balancing hyperparameters are exactly the kind
    of thing that benefits from search.
- **Effort**: ~2 days.

### 1.3 `Trainer.fit` no-grad validation fix
- Wrap the validation forward/loss-fn call in `torch.enable_grad()`
  instead of leaving it inside the outer `torch.no_grad()` block, gated by
  a new `TrainConfig.physics_aware_validation: bool = True` (default on,
  since it's strictly more capable — the only reason to turn it off is
  memory pressure from keeping a validation graph, which the flag lets a
  memory-constrained caller opt out of explicitly rather than discovering
  the crash the hard way).
- **Effort**: ~2 hours + regression-testing every existing `Trainer` call
  site in `examples/` (part of 1.1's matrix).

### 1.4 `pinneapple.pipeline()` task abstraction
- One call per common named task
  (`pinneapple.pipeline("channel_flow_les", Ubar=..., Re_tau=...)`,
  `pinneapple.pipeline("airfoil_rans", naca="0012", alpha=5)`, ...) that
  wires `get_preset`/a custom `ProblemSpec` + `build_model` + `solve_pde`
  (P0) into one call, returning a thin object with `.model`, `.history`,
  and `.predict(x)`. Mirrors HF's `pipeline()` UX directly.
- **Effort**: ~1–2 days for a first set of ~10 named tasks across the
  domains QUICKSTART.md already advertises (fluid, thermal, structural).

### 1.5 Governance-as-code for the model hub
- `ModelCard` schema (architecture + weights hash + training config +
  data lineage + validation metrics + citation — see P1.6) is already
  planned as part of the hub (1.6). Governance-as-code on top:
  - `scripts/validate_model_card.py`: schema + required-fields check
    (rejects a push with no `validation_metrics` block or no
    `reference_source` citation).
  - A CI job (part of 1.1's GitHub Actions workflow) that runs this
    validator on any PR touching a model card.
  - This is the actual code-shaped version of "governance" — the *human*
    review process (who approves a new public model) is a project/policy
    decision for the repo owner, not something a plan like this can
    implement.
- **Effort**: ~1 day.

---

## P2 — mid-term (1–3 weeks, still no hosted infrastructure required)

### 2.1 Model hub client (`pinneapple_hub`)
- `push_to_hub(model, repo_id, model_card, weights_path=None)` /
  `from_pretrained(repo_id, revision="main")`, built on the **Hugging Face
  Hub's own infrastructure** via the `huggingface_hub` client library
  (optional dependency) — not a new hosting service. This is the
  pragmatic, already-proven pattern (many domain libraries piggyback on
  HF Hub's actual storage/auth/versioning rather than reinventing it): a
  PINNeAPPle checkpoint + `model_card.json` uploaded as files in a normal
  HF Hub model repo, with a `pinneapple`-specific loader convention on top
  (a `pinneapple_config.json` sidecar file recording which `ModelRegistry`
  name + `in_dim`/`out_dim`/`hidden_dim`/... to reconstruct the
  architecture before loading the `state_dict`).
- Ship `ModelCard` (dataclass: `architecture`, `training_config`,
  `data_lineage`, `validation_metrics: Dict[str, float]`,
  `reference_source: str`, `citation: str`) serialized alongside every
  push — the reproducibility contract from P1.5.
- **Effort**: ~1 week (client code + `ModelCard` schema + docs + a couple
  of real example pushes to validate the round-trip).

### 2.2 CGNS/Exodus/Fluent/Abaqus readers: real-file validation
- P0 shipped these against the documented spec only (no reference file
  available). Close the loop: generate or obtain one small real file per
  format (CGNS: `cgns_utils` or any open CFD tutorial case; Exodus: any
  SEACAS example mesh; Fluent: a Gambit/Fluent tutorial `.msh`; Abaqus:
  a trivial `.inp` + (if a license is available in CI) an `.odb` from
  running it) and add them as fixtures in `tests/`.
- **Effort**: ~2–3 days, mostly sourcing/generating the fixture files
  (the Abaqus `.odb` fixture specifically needs a licensed Abaqus
  install somewhere in CI or this stays a manual/local-only check).

### 2.3 Blender bridge (`pinneapple_blender`)
- `bpy` is Blender's own bundled Python and cannot be `pip install`ed into
  a normal environment, so — same shape as the Abaqus `.odb` bridge (P0)
  — this is a **subprocess bridge**, not an in-process import:
  - `pinneapple_blender.export_scene(sample_or_trajectory, out_path,
    format="abc"|"obj_sequence")`: PINNeAPPle-side, no Blender needed —
    writes an Alembic (`.abc`, time-varying point-cloud/mesh cache,
    industry standard for sim-to-DCC handoff) or a numbered OBJ sequence
    from a `PhysicalSample`/trajectory (field values baked as vertex
    colors or an attribute layer).
  - `pinneapple_blender/blender_addon/`: a genuine Blender add-on
    (`bpy`-only code, runs *inside* Blender) that imports that Alembic/OBJ
    sequence, sets up a color ramp keyed to the field values, and can kick
    off a headless render (`blender --background --python
    render_script.py`) driven from the PINNeAPPle side via
    `subprocess.run(["blender", ...])` — matching the Abaqus bridge
    pattern exactly (shell out to the real tool's own Python, never guess
    its internals).
  - Reverse direction (Blender geometry → PINNeAPPle domain/SDF): export
    the modeled geometry as `.obj`/`.stl` from Blender (built-in exporter,
    no custom code needed) and feed it through the already-existing
    `pinneapple_design.geometry` SDF/mesh pipeline.
- **Effort**: ~1–1.5 weeks (export writer + add-on + a documented example
  end-to-end: LES surrogate → Alembic → Blender render).

### 2.4 3D `KOmegaSSTResiduals`
- Full 3D generalization of the existing 2D-only k-ω SST residual
  (production/dissipation/cross-diffusion terms extended with the third
  velocity component and z-derivatives, mirroring the pattern
  `WALEResiduals` (P0) already established for going from a 2D helper
  style to a genuine 3D one in this same file).
- **Effort**: ~3–4 days (the physics derivation and its numerical
  stability testing are the actual cost, not the code volume).

---

## P3 — the "no software-only guarantee" part: physics-grounded verification layer

This is the honest shape of "make hallucination impossible": not a
promise, a **mandatory verification gate** every LLM-assisted or
auto-generated pipeline output must pass before being labeled trustworthy.

### 3.1 `pinneapple_llm` — LLM-assisted pipeline generation, routed through PINNeAPPle's own typed API
- **The differentiation mechanism, stated plainly**: an LLM asked to "just
  write CFD code" free-form can silently fabricate boundary conditions,
  invent physically-impossible parameter combinations, or produce code
  that runs and looks plausible while violating conservation. This module
  never lets the LLM's output *be* the result — the LLM only ever
  proposes a `ProblemSpec` + model/training config **using PINNeAPPle's
  existing typed dataclasses** (`ProblemSpec`, `PDETermSpec`,
  `ConditionSpec`, `TrainConfig`, ...), which then goes through
  PINNeAPPle's own compiler (`compile_problem`) and, critically, through
  **3.2's verification gate** before any result is reported back to the
  user as trustworthy. The LLM drafts; PINNeAPPle's physics — not the
  LLM's confidence — decides what's real.
- API sketch:
  ```python
  from pinneapple_llm import draft_problem, PhysicsGuardrail

  spec, model_config = draft_problem(
      "Steady turbulent flow around a NACA 0012 airfoil at Re=1e6, alpha=5deg",
      provider="anthropic",  # or "openai" — provider-agnostic, bring your own API key
  )
  result = solve_pde(spec, build_model(**model_config))
  report = PhysicsGuardrail(spec).check(result)  # -> see 3.2
  if not report.trustworthy:
      raise report.as_error()  # residuals/conservation/reference-mismatch, itemised
  ```
- Provider-agnostic (`anthropic`/`openai` SDKs, optional deps), with the
  prompt template constrained to emit only fields the `ProblemSpec`/
  `PDETermSpec` dataclasses actually accept (schema-constrained
  generation, e.g. via the target SDK's structured-output/tool-calling
  mode) — the LLM cannot emit arbitrary code, only a structured spec
  PINNeAPPle itself will validate and compile.
- **Effort**: ~1 week for the drafting module; depends on 3.2 to be
  meaningful at all (drafting alone, with no verification gate, would be
  exactly the ungrounded-LLM-output problem this whole item exists to
  avoid).

### 3.2 `PhysicsGuardrail` — the actual anti-hallucination layer
- A structured, automatic check run on any `solve_pde`/training result,
  independent of whether an LLM was involved at all (useful for *every*
  PINNeAPPle user, not just the LLM path):
  - **Residual check**: re-evaluate `compile_problem`'s own PDE residual
    on a fresh, dense collocation sample post-training; fail if the
    residual norm is above a spec-appropriate threshold (this is the one
    that would have caught, e.g., a training run that "looked" converged
    on its own reported loss but was actually numerically unstable).
  - **Conservation checks**: mass/momentum/energy balance over the domain
    boundary (already partially scaffolded by `pinneapple_analysis
    .validation` per the package map — audit (1.1) whether it's wired to
    anything, extend if not).
  - **Reference-data check** (when available): if the `ProblemSpec` (or
    the LLM-drafted one) references a known benchmark (a named DNS/
    experimental dataset, e.g. via `pinneapple_pdb`), automatically pull
    it and compute the same RMSE-vs-Ubar-style metric this very project's
    `problem_config.json` records by hand — turning "matches DNS to X%"
    from a documentation claim into a computed, re-checkable number every
    time.
  - **Dimensional-analysis sanity check**: verify the compiled residual's
    units are self-consistent (catches an LLM- or human-authored spec
    with a physically nonsensical parameter, e.g. a negative viscosity or
    incompatible unit system) — a cheap, high-value check to run first.
  - Output: a `GuardrailReport` (per-check pass/fail + numeric detail),
    not a single boolean — "trustworthy" is a compound claim and should
    show its work, exactly so a user isn't asked to take "0%
    hallucination" on faith either.
- **Effort**: ~1.5–2 weeks (the reference-data auto-fetch and dimensional-
  analysis checks are the largest pieces; the residual/conservation
  checks reuse existing `compile_problem`/`pinneapple_analysis` machinery
  once 1.1 confirms what's actually wired up there today).

### 3.3 Why an LLM alone cannot replace this
Worth stating explicitly, since it's the actual competitive thesis: a raw
LLM asked to solve a physics problem either (a) writes code from scratch,
with no guarantee it implements the stated closure/BCs/discretization
correctly and no automatic check that it does, or (b) states an answer
from training-data pattern-matching, with no residual/conservation
computation behind it at all. **PINNeAPPle's differentiation is not "our
LLM is smarter" (it's the same commodity LLMs everyone has) — it's that
every claim is required to pass through a compiled, autograd-checked
residual and a numeric verification report before it's presented as
correct.** That property is what a user cannot get "just using a language
model" regardless of which one, and it's the property 3.1+3.2 together
are built to make load-bearing rather than aspirational.

---

## Suggested execution order (if resumed by a team)

1. P0 items still open today (`Trainer.fit` no-grad fix, broken template,
   hub client, CI matrix, `pipeline()`, 3D k-ω SST, Blender, LLM module) —
   whatever this session gets through, in the order listed in the P0
   table.
2. P1.1 (evidenced audit harness) **before** P1.2–P1.5 — it will surface
   real bugs that change the priority of everything after it.
3. P2 in any order (independent of each other).
4. P3.2 (`PhysicsGuardrail`) before P3.1 (LLM drafting) — the guardrail is
   useful standalone and is a hard prerequisite for the LLM module to be
   trustworthy rather than decorative.
