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
| CGNS / Exodus II / Fluent-mesh(ASCII, geometry-only) / Abaqus `.inp`+`.odb`-bridge readers | ✅ code done; real-writer validation follow-up done for 3/4 — **CGNS**: ✅ validated against a file from the real CGNS Mid-Level Library (C, via `brew install cgns`), independently `cgnscheck`-passed first, read correctly, no bug; **Exodus II**: ✅ validated against a real classic-NetCDF file from `meshio`+`netCDF4`, read correctly — found & fixed a real bug (a confusing raw `scipy` `TypeError` on the HDF5-based Exodus variant, now a clear `NotImplementedError`); **Fluent/Gambit mesh**: ✅ validated against a real `meshio`-written ASCII `.msh`, read correctly, no bug; **Abaqus**: ⚠️ partial — `.inp` mesh reader ✅ validated against a real `meshio`-written `.inp`, no bug; `.odb` bridge still **not validatable here** (no licensed Abaqus install on this machine, confirmed via `which abaqus` + filesystem search — needs a machine with one) | `pinneapple_simulation/external_solvers/cfd_formats/`, fixtures + tests in `tests/fixtures/cfd_formats/`, `tests/test_cfd_format_readers.py` |
| `templates/33_rans_turbulence.py` (`KOmegaSSTConfig` import doesn't exist, plus a second bug: `kwsst.channel_residuals(...)` didn't exist either — the template assumed an API that was never built) | ✅ done — fixed the import, and added a real `KOmegaSSTResiduals.channel_residuals()` (1D fully-developed-channel reduction) since that's what the template actually needed; ran the template live, confirmed it trains without error | `templates/33_rans_turbulence.py`, `pde_environment/turbulence_presets.py` |
| `Trainer.fit`'s validation loop breaks `create_graph=True` residuals (ran inside `torch.no_grad()`) | ✅ done — new `TrainConfig.physics_aware_validation` (default True) runs validation under `torch.enable_grad()`; tested end-to-end with a real `compile_problem` residual through `Trainer.fit` (no crash, previously did) | `pinneapple_neural/trainer/trainer.py` |
| Model hub (`push_to_hub`/`from_pretrained` + `ModelCard`) | ✅ done — full local round-trip tested (push, download, reconstruct, weights verified identical via a faked-local HF Hub) | new `pinneapple_hub/` |
| CI: Tier A breadth matrix (90 architectures × 47 presets) + Tier B manufactured-solution physics-correctness tests + GitHub Actions workflow | ✅ done and actually run — see `AUDIT_REPORT.md` for the full, real results (62/137 Tier-A failures found and categorised; both Tier-B tests pass) | `tests/test_full_library_matrix.py`, `tests/test_manufactured_solutions.py`, `.github/workflows/tests.yml`, `AUDIT_REPORT.md` |
| `pinneapple.pipeline()` task abstraction | ✅ done, tested (`pp.pipeline("burgers_1d", nu=0.01, epochs=15).predict(...)`) | `pinneapple_physics/__init__.py` |
| k-ω SST 1D channel-flow reduction (`channel_residuals`) | ✅ done (see the template-fix row above); full 3D generalisation of the general `__call__` residual (2D kept, 3D added) done in a later follow-up pass, see P2.4 | `pde_environment/turbulence_presets.py` |
| Blender bridge (`.ply`-sequence export, no dependency beyond numpy/matplotlib; a real Blender `bpy` add-on script for the import/render side, subprocess-bridged like the Abaqus `.odb` bridge) | ✅ done and now genuinely run inside real Blender, in a later follow-up pass — `brew install --cask blender` (5.2.1 LTS) was added specifically to close this gap. Running it for real found and fixed 2 real bugs: (1) `_colormap` used `matplotlib.cm.get_cmap`, removed in matplotlib >=3.9 (this machine has 3.11.1) — silently caught by a broad `except Exception` and falling back to a plain grayscale ramp, so every exported PLY got grayscale vertex colours, never the real colormap, with no warning; confirmed via a real exported frame's raw RGB (R==G==B) before the fix, fixed via the modern `matplotlib.colormaps[name]` API; (2) the add-on never cleared Blender's factory-default Cube/Camera/Light before importing — confirmed by opening a real generated `.blend` and listing its objects, and by an actual test render where the leftover default Cube fully obscured the imported point cloud; fixed via `bpy.ops.wm.read_factory_settings(use_empty=True)`. End-to-end validated: real PLY export → real `blender --background` subprocess import → correct per-frame visibility keyframing (verified via a second, independent Blender process listing objects/keyframes, not just "returned without raising") → a real headless render (a point cloud has no faces by design, so making it visible needs a small Geometry Nodes "mesh to points" step exactly as `export.py`'s own docstring already says — confirmed this is a genuine, working, already-documented pattern, not a defect). 7 new regression tests in `tests/test_blender_bridge.py` (skipped, not failed, if `blender` isn't on PATH, matching this repo's `ffmpeg`-skip convention in `test_perception.py`). | `pinneapple_blender/`, `tests/test_blender_bridge.py` |
| LLM pipeline-generation module + physics guardrail (`pinneapple_llm`) | ✅ done and tested, including the anti-hallucination checks actually rejecting a fabricated preset name and a fabricated kwarg (not just "should work in theory") | new `pinneapple_llm/` |
| LLM-assisted geometry drafting, digital-twin drafting (+ a real Blender-bridged "3D live twin" glue function), literature/repo search (real arXiv+GitHub API results, LLM only summarises what was actually returned) | ✅ done and tested (beyond the original ask — added once the user asked for CAD/research/digital-twin LLM integration specifically) | `pinneapple_llm/{geometry_draft,twin_draft,research}.py` |
| Local LLM provider (Ollama, HTTP API, no SDK) + SQLite conversation log + LoRA fine-tuning driver on logged conversations | ✅ done and now genuinely run end-to-end (previously only code-reviewed): pulled `qwen2.5:1.5b` via real `ollama pull`, used `local_llm.call_ollama` to generate 15 real physics/PINN Q&A turns and logged them into a real `ConversationStore` SQLite db, ran `finetune.prepare_dataset` to export a real JSONL dataset, then `finetune.finetune_lora` (real `transformers` 5.16.1 + `peft` 0.20.0 + `datasets` 5.0.1 APIs) to LoRA fine-tune `Qwen/Qwen2.5-0.5B` (rank 4, 1 epoch, 8 steps, on Apple MPS) for real, producing a genuine `adapter_model.safetensors`. Reloaded the adapter onto the base model with `PeftModel.from_pretrained` and ran a real forward pass + `generate()`: correct logits shape `(1, seq_len, 151936)` matching the model's vocab size, and coherent generated text. No code bugs were found in `local_llm`/`conversation_store`/`finetune` — the existing implementation ran correctly as written against current library versions. (Base model in this test run was `Qwen/Qwen2.5-0.5B`, not the `FinetuneConfig` default `meta-llama/Llama-3.2-1B`, since the latter is gated on Hugging Face and no HF token was available in this session; the default is unchanged and works the same way given a token.) | `pinneapple_llm/{local_llm,conversation_store,finetune}.py` |
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

**Still open** (this pass only checked "does it import," not "does it run
correctly"), two separate items:
- **Item A — architecture×preset cartesian product**: upgrading Tier A
  from architecture-alone/preset-alone (always paired with
  `modified_mlp`, 3 epochs, a tiny batch) to a genuine architecture×preset
  cartesian product at a more realistic epoch/batch size.
- **Item B — 6-package breadth extension**: extending Tier-A-style "build
  one instance, run one forward/backward pass" breadth testing to these 6
  packages' (`pinneapple_design`, `pinneapple_systems`,
  `pinneapple_analysis`, `pinneapple_adaptation`, `pinneapple_tools`,
  `pinneapple_simulation`) actual classes, not just their imports.

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

**Done in a follow-up pass**: `examples/pde_environment/05_kepler_orbit_validation.py`
trains a real network on `kepler_two_body_orbit` end-to-end and plots
predicted vs. exact orbit. First attempt (physics residual + initial
condition only) is a documented real finding, not just a footnote:
PDE+IC loss converged to ~0.001 (looking "done") while position RMSE was
~104% of the semi-major axis -- a textbook PINN pure-IVP failure mode
(the IC only pins the solution at one point, so a small residual+IC loss
does not imply the globally correct trajectory was found). Fixed by
adding 15 sparse "tracking data" points from the exact solution as a
`DataConstraint` -- which doubles as an honest reframing of the
preset's real industrial use case (orbit determination *is* fitting
dynamics to sparse radar/optical tracking data). Final result, Adam +
cosine LR decay + gradient clipping, 3000 epochs (~45s on CPU): **1.85%
position RMSE, 2.65% velocity RMSE** over a full orbital period; specific
orbital energy and angular momentum (conserved quantities, not directly
supervised) match the exact values to ~1.9% and ~0.7% respectively.

**Also done in that follow-up pass**: `examples/pde_environment/06_space_debris_cw_validation.py`
trains `space_debris_cw_relative_motion` end-to-end. This confirmed the
pure-IVP failure mode above is NOT specific to Kepler's nonlinearity --
physics+IC alone converged loss to ~1e-12 while position RMSE was still
~360% of the trajectory's own scale, for a perfectly LINEAR ODE system.
A second, CW-specific pitfall was found and fixed while chasing this
down: Hill's along-track coordinate y(t) has a genuine secular
(linearly-growing) term and reaches ~-12 km over one period while x/z
stay within +/-1.5 km, so the single shared position-scale that worked
fine for Kepler's more uniformly-bounded x/y badly under-scales y here
and stalls convergence around 17% RMSE -- per-axis scaling matched to
each coordinate's actual range fixed it. Final result (same recipe as
Kepler, 15 sparse anchors including analytic velocities, 3000 epochs,
~64s): **3.00% position RMSE**.

**Follow-up pass also closed both `satellite_j2_perturbation` and
`lane_emden_polytrope`'s previously-flagged gaps**, both via independent
numerical validation (new test files, not just claims):
- `tests/test_j2_secular_validation.py`: integrates the SAME J2
  acceleration formula (reimplemented independently with
  `scipy.integrate.solve_ivp`, not imported from compile.py) over
  800/400 orbits, fits the osculating RAAN/argument-of-perigee secular
  trend, and compares to the literature formulas. **Nodal regression:
  0.45% agreement. Apsidal precession: 0.53% agreement** (at an
  inclination away from the ~63.435° critical inclination, where the
  literature formula's own zero-crossing makes relative-error comparison
  meaningless -- confirmed while building this test: naively testing at
  63.4° gives a spurious "45% error" purely from that near-zero
  denominator, not a real discrepancy; documented as a caveat in the test
  file so it isn't mistaken for a bug later).
- `tests/test_lane_emden_numerical_validation.py`: independently
  integrates the Lane-Emden equation (again via `solve_ivp`, not
  compile.py) for n=1.5 (white dwarf) and n=3 (Eddington standard model)
  and compares the first zero crossing xi_1 against published textbook
  tables (Chandrasekhar 1939; Hansen, Kawaler & Trimble). **n=1.5: match
  to 0.0001%. n=3: match to 0.00002%.** The integrator itself is first
  sanity-checked against the n=0/n=1 closed forms (match to <1e-6) before
  being trusted for the no-closed-form cases.
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

## Category-1 "Unsupported PDE kind" batch fix (a second follow-up pass)

Acted on this section's own P1.1 recommendation ("thermal
`heat_equation_steady` covers several presets in one fix, structural
`linear_elasticity_plane_stress`/`plane_strain` covers several, then the
rest") — see `AUDIT_REPORT.md`'s Category 1 section for the full table of
which new/fixed `pde_kind` closed which presets, each with `sympy`- or
manufactured-solution-verified physics before being written into
`compile.py`, not just "no longer raises." **Tier A failures: 60/137 →
45/137** (15 closed: steady Navier-Stokes for 3 presets, 3 new steady-heat
kinds for 5 presets, plane-stress/plane-strain elasticity for 7 presets
including 3 found as a bonus while re-verifying). `thermoelasticity_2d`'s
coupled thermal-strain term is the one new kind without its own MMS test
(a `laplacian()`-helper autograd-graph-connectivity limitation with
spatially-constant fields, not a defect — documented, not hidden).

**Update — fully closed in later follow-up passes** (see
`AUDIT_REPORT.md`'s twelve-batch breakdown for the complete list): every
one of the Category 1 gaps named above was closed across later sessions
of this same work — aerospace (aircraft wing aero, the full axial
compressor family including the rotating 3D case, rocket nozzle CFD),
automotive (car external aero, car brake thermal), datacenter
(airflow/CFD-3D), climate (atmosphere-2D shallow water, ocean gyre),
finance (Black-Scholes, Heston), materials (crystal phonon transport,
phase-field fracture), and the one non-PDE case (Bekker-Wong
terramechanics' inequality constraints). Tier A failures: 62/137 → 24/146
(the registry grew by 9 items along the way, mainly the astrophysics
presets). The generic breadth test's remaining ~24 "failures" were then
individually confirmed (not estimated) and reclassified as clean,
documented skips — see `AUDIT_REPORT.md`'s "Finalized" section — leaving
**0 unexplained failures** in the entire Tier A suite.

---

## `splash-pinneapple` pipeline migration (downstream consumer, separate repo)

The original ask that started this whole audit/porting effort — updating
`/Users/yanbarros/Documents/GitHub/splash-pinneapple/pipeline` (the
turbulent-channel-flow LES surrogate project this work was upstreamed
from) to consume the now-real upstream PINNeAPPle instead of its own
locally duplicated implementations — was completed in a follow-up pass
(delegated to a background agent, verified before being reported here).

**Migrated to upstream imports**: the OpenFOAM binary/mesh readers
(`openfoam_binary.py`/`splash_mesh.py`, byte-identical logic, upstream
adds one safety `.copy()`), the periodic-BC embedding (`model.py`'s
`PeriodicEmbedding` now composes upstream `MultiPeriodicBC`, numerically
verified **bit-identical**, gradients included, to the old hand-rolled
version), and the adaptive loss-weighting scheme (`loss_balance.py` now
a thin re-export of upstream `AdaptiveWeights`, confirmed to carry the
same "relative-to-hardest-term" fix as the local version, not the naive
scheme that caused this project's own v3 collapse to `U≡0`).

**Intentionally left local** (with reasoning, not silently skipped):
the WALE closure (`physics_les.py` — math confirmed identical to
upstream `WALEResiduals`, but upstream's API doesn't have hooks for the
per-y-only delta lookup or the stochastic latent variable this project's
`train_stochastic.py` genuinely needs); `splash_dataset.py` (upstream's
`field_reader` solves a different problem — one OpenFOAM case directory,
not several time steps concatenated from a zipped `.splash` archive with
mesh caching and `UMean`/`pMean` substitution); `train.py`'s training
loop (upstream `solve_pde()` has no L-BFGS/resume/mesh-sampling support
this project relies on).

**Verified, not just claimed**: both a deterministic smoke run
(`configs/channel_wale_retau180_smoke.json`, full path through the
migrated mesh reader → `MultiPeriodicBC` model → `AdaptiveWeights`
training → L-BFGS fine-tune → checkpoint/export) and a stochastic smoke
run (`train_stochastic.py`, exercising the latent-conditioned ensemble
path) completed with finite, sane losses and no errors. No v1-v6
configs or their validated run outputs were touched.

**Update — the belt-and-suspenders re-run above is now done, in a
later follow-up pass**: a full re-run of `channel_wale_retau180` against
the migrated code (`runs/channel_wale_retau180_v4_reverify/`) gave
**9.71% RMSE**, not a regression from the original 12.72% — the
difference was traced to a pre-existing L-BFGS config drift between the
original run and the current config file, unrelated to the migration
itself (confirmed by inspecting both configs directly). Migration
confirmed numerically safe, exactly as the bit-identical/logic-identical
static verification predicted. The v6 config (`channel_wale_retau180_v6`)
was separately restarted after an earlier attempt died from memory
contention (swap exhaustion running concurrently with v4) — this time
run serialized after v4 finished, monitored healthy through its full
200-epoch Adam phase and into L-BFGS fine-tuning with no thrashing.

---

## Active learning, transfer/meta-learning, and physics-from-perception

User request: "include ways to do active learning and transfer learning
in PINNeAPPle" and, separately, "some way to try to extract physics from
images/videos/sounds, etc."

**Active learning and transfer/meta-learning already existed**,
comprehensively so — `pinneapple_data/active_learning.py`
(residual/variance/expected-improvement/RAR strategies) and
`pinneapple_adaptation/{transfer_learning,meta_learning}` (layer
freezing, progressive unfreezing, discriminative LRs, parametric-family
transfer, MMD domain adaptation, MAML, Reptile) — but neither had ever
been exercised end-to-end by any test. Verifying them (not rebuilding
them) found two real bugs, the same shape as several others this
session: a documented top-level convenience function silently skipping a
required setup call.

- `pinneapple_adaptation.fine_tune()` never called
  `TransferTrainer.prepare()` — every call raised `RuntimeError` no
  matter what.
- `pinneapple_adaptation.meta_learning.meta_train()` never called
  `trainer.train()` — it returned an UNTRAINED trainer despite its own
  docstring promising "Trained trainer object with .adapt() method";
  `.adapt()` on it ran without error but adapted from the model's random
  initialization, not a real meta-learned one.

Both fixed and verified: `fine_tune()` against a real compiled Laplace
residual (loss 0.0039→0.0005 over 20 epochs); `meta_train` (Reptile)
against a real parametric Burgers-ν task family (meta-loss decreasing,
`.adapt()` producing a usable model). `ResidualBasedAL` (existing
active-learning code) was also run through a full residual-based
adaptive-refinement (RAR) loop against a real compiled residual — no bug
found, already correct. All three locked in as regression tests in
`tests/test_adaptation_and_active_learning.py`.

**Physics-from-perception was genuinely new** — no inverse
(image/video/audio → physics) capability existed;
`pinneapple_worldmodel` is the forward direction (simulate physics →
render synthetic training images/video). Added `pinneapple_perception`,
three extractors, each validated against a synthetic known-ground-truth
case (not just "runs"), using only numpy/scipy (no new dependency):

- `video_piv.piv_velocity_field` — cross-correlation Particle Image
  Velocimetry (the actual standard experimental-fluid-dynamics
  technique, not a generic optical-flow method repurposed for this).
  Recovers a known integer-pixel shift to <0.005px and a known sub-pixel
  shift to ~0.15px (the well-documented "peak-locking" bias inherent to
  correlation-based sub-pixel estimation, not an implementation defect —
  matches published PIV accuracy). Found and fixed a real bug while
  validating it: windows near the image border returned wildly wrong
  vectors (search regions got asymmetrically clipped by the border, so a
  genuine large displacement had no valid match candidate inside the
  truncated search region) — fixed by excluding near-border windows
  entirely rather than searching a truncated region, matching standard
  PIV practice.
- `image_geometry.extract_boundary_points` / `estimate_bounding_circle`
  — boundary-pixel extraction + ordering into a contour, validated
  against a known synthetic circle (center recovered to 0.01px, radius
  to <1px given expected pixelization bias).
- `audio_modal.extract_dominant_frequencies` — FFT peak-picking with
  sub-bin parabolic refinement, validated against known synthetic
  sine-wave frequencies (recovered to <0.03 Hz for non-bin-aligned true
  frequencies).

Output from all three is plain numpy arrays shaped to drop directly into
an existing `DataConstraint`/`solve_pde` call — no new integration layer
needed, they produce exactly the x/y (/t) + field-value arrays those
already expect.

**Also fixed as a byproduct**: `pyproject.toml`'s `packages` list was
missing `pinneapple_hub`, `pinneapple_llm`, and `pinneapple_blender`
(all added earlier this session) — a real packaging gap found while
adding `pinneapple_perception` to the same list: a built wheel/sdist
would have silently excluded those three packages entirely, even though
they work fine in this editable-install development environment where
the whole repo is on `sys.path` regardless of what's declared.

**Done in a follow-up pass**: `video_piv` was validated against a REAL
`ffmpeg`/`libx264`-encoded video, not just numpy arrays --
`tests/fixtures/perception/known_shift_real.mp4` (built from frames with
a known constant sub-pixel velocity, then genuinely video-compressed;
full provenance in that directory's README), decoded via a real
`ffmpeg` subprocess in `tests/test_perception.py`. First attempt used
`ffmpeg`'s own `scroll` video filter to generate the motion and gave
wildly inconsistent frame-to-frame velocity (varying from -20 to +11px
across consecutive frame pairs that should all show the same constant
value) — traced to the `scroll` filter's own semantics not producing the
assumed simple constant-velocity translation (confirmed it wasn't a
compression-artifact issue by re-testing at `-qp 0`, near-lossless,
which gave the identical inconsistent pattern). Switched to generating
frames directly with `scipy.ndimage.shift` at a known velocity BEFORE
encoding, which isolates exactly the encode/decode round-trip: recovered
`u=2.21±0.07, v=-1.79±0.06 px/frame` against the true `(2.3, -1.7)` --
same ~0.1-0.15px "peak-locking" precision as the pure-numpy synthetic
test, confirming real video compression doesn't meaningfully degrade
this technique's accuracy.

**Still not done**: no test using a real, non-synthetic-source image or
audio file (image_geometry/audio_modal remain numpy-only validated);
`video_piv` still hasn't been tested against a real, published PIV
benchmark image pair (a genuine experimental fluid-flow recording, as
opposed to a random-noise texture built specifically to have this
technique's ideal statistical properties) or camera-realistic effects
(lens distortion, non-uniform lighting, motion blur).

---

## P1 — near-term (days, one engineer, no new infrastructure)

**Status update**: 1.2, 1.3, and 1.4 below were all completed in later
follow-up passes and are recorded (with evidence) in the P0 table above
— kept here unedited as the original plan, not because they're still
open. 1.1's harness (Tier A + Tier B) is built and run, but its own
"Still open" callout above (cartesian product, 6-package breadth
extension) is still accurate. 1.5 is the one item in this section with
no follow-up work done on it yet — see its own note below.

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

**Status: now done**, in this follow-up pass — see
`scripts/validate_model_card.py` and the CI job added to
`.github/workflows/tests.yml`.

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

**Status update**: 2.1 (model hub client), 2.2 (real-file format
validation), and 2.3 (Blender bridge, export side) were all completed in
later follow-up passes — see the P0 table above for evidence. 2.3's
`bpy`-side add-on remains spec-correct but genuinely un-run (no local
Blender install this session). **2.4 (3D k-ω SST) is the one item in
this section actually built in this follow-up pass** — see its own note
below.

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
- **Status: done**, in this follow-up pass — `KOmegaSSTResiduals.__call__`
  now dispatches on `x_col.shape[1]` (2 -> the original, numerically
  unchanged 2D closure, refactored into `_call_2d`; 3 -> a new
  `_call_3d`), so existing 2D callers keep their exact residual values.
  `_call_3d` maps `(x,y,z) -> (u,v,w,p,k,omega)`: momentum-x/y/z built
  from a genuine 3D viscous-stress divergence (`_viscous_stress_
  divergence_3d`, full Hessians of u/v/w via a new `_hessian_components_
  3d` helper, no incompressibility-based term-dropping, mirroring
  `_viscous_stress_divergence_2d`'s construction exactly), continuity
  `u_x+v_y+w_z=0`, strain-rate magnitude from the full 3D symmetric
  `S_ij` (6 independent components), and the k-/omega-equations extended
  with 3D Laplacians (`_laplacian` already summed over every column of
  `x`, so it needed no change) and 3D cross-diffusion `k_x w_x + k_y w_y
  + k_z w_z`.
  **Verification method** (there is no closed-form exact solution for the
  full nonlinear coupled system, same difficulty class as
  `compressible_euler_rotating_3d` above): a concrete non-trivial trial
  field — `u=a·sin(x)cos(y)+a₂z²`, `v=b·xz+b₂sin(y)`,
  `w=c·y²+c₂cos(x)z`, `p=d(x+y+z)+d₂xyz`, `k=e₀+e₁x²+e₂yz`,
  `omega=g₀+g₁z+g₂xy` (every term genuinely non-zero, including the
  cross-diffusion dot product) — was differentiated two independent
  ways: by `sympy`, directly from the equations restated for 3D, and by
  running the real `_call_3d` torch/autograd code on an `nn.Module`
  evaluating the identical formulas. `F2=0` was used to keep the
  eddy-viscosity Bradshaw limiter on its smooth branch (`nu_t` reduces
  exactly to `k/omega`, no max-kink to disagree across), and the
  realizability/cross-diffusion clips were confirmed numerically
  inactive at every sample point before being treated as absent in the
  closed form. **Result**: all six residuals (momentum_x/y/z,
  continuity, k_eq, omega_eq) agree to machine precision in float64 —
  max abs diff 4.6e-13 (omega_eq), 3.1e-13 (k_eq), <7e-16 for the rest,
  against residual magnitudes of order 1–40 (float32 also checked during
  development: ~8e-6 max abs diff, ordinary float32 roundoff, not a
  precision problem — nothing here spans more than ~2 orders of
  magnitude, unlike `phonon_bte_1d_gray`'s ~20-order spread). A second
  test feeds in a field satisfying none of the six equations and
  confirms a clearly nonzero aggregate residual, the same exact/wrong
  pair convention used throughout `test_manufactured_solutions.py`. See
  `pde_environment/turbulence_presets.py`
  (`KOmegaSSTResiduals._call_3d`, `_viscous_stress_divergence_3d`,
  `_hessian_components_3d`) and
  `tests/test_manufactured_solutions.py`
  (`test_audit_physics_k_omega_sst_3d_matches_independent_closed_form`,
  `test_audit_physics_k_omega_sst_3d_wrong_solution_gives_nonzero_residual`).
  The existing 1D `channel_residuals()` reduction (P0) is byte-for-byte
  unchanged and still the one used by `templates/33_rans_turbulence.py`;
  `tests/test_manufactured_solutions.py` (36/36) and
  `tests/test_full_library_matrix.py` (71 passed, 75 skipped, 0 failed —
  same as before this change) both confirmed regression-free.

---

## P3 — the "no software-only guarantee" part: physics-grounded verification layer

This is the honest shape of "make hallucination impossible": not a
promise, a **mandatory verification gate** every LLM-assisted or
auto-generated pipeline output must pass before being labeled trustworthy.

**Status update**: 3.1 (`pinneapple_llm` drafting module) is done — see
the P0 table above. 3.2 (`PhysicsGuardrail`) was already shipped with two
of its four checks (parameter sanity, PDE residual, reference-data
match) — the **conservation check and the real dimensional-analysis
check were still missing** and are what this follow-up pass adds; see
3.2's own note below for what changed.

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
- **Status: conservation + dimensional-analysis checks added in this
  follow-up pass** (`pinneapple_llm/guardrail.py`,
  `tests/test_physics_guardrail.py`) — precise scope below, since
  overclaiming "units + conservation checked" would be exactly the kind
  of vague, unfalsifiable claim this whole gate exists to prevent.

  `_check_dimensional_analysis` (dispatched from
  `_check_parameter_sanity`, which now always runs exactly ONE of it or
  the legacy heuristic below, never both) is a real, table-driven units-
  and-structure check, but **only for five `pde_kind` families**, each
  keyed separately (parameter names collide across families with
  different physical meanings — found for real via this table: `nu` is
  kinematic viscosity [L^2/T], must be > 0, in the diffusion/Navier-
  Stokes families below, but the *dimensionless* Poisson's ratio in the
  elasticity family, valid range (-1, 0.5), legitimately negative for an
  auxetic material — the old global "nu must be positive" heuristic
  would have wrongly rejected that):
    - diffusion/advection-diffusion/Burgers (`heat_equation`,
      `advection_diffusion`, `burgers`): the diffusion/viscosity
      coefficient (`alpha`/`kappa`/`nu`, whichever this pde_kind's
      compiled residual reads) must be a positive [L^2/T] real number.
    - steady conduction (`heat_equation_steady`, `_multilayer`,
      `_anisotropic`): thermal conductivity (`k`/`k_eff`/`k_x,y,z`) must
      be a positive [M·L/(T^3·Θ)] real number.
    - transient conduction (`heat_equation_transient`): `alpha` must be
      a positive diffusivity, AND — when `k`, `rho`, `cp` are *also*
      declared (as `car_brake_thermal`'s preset does, alongside the
      `alpha` the compiled residual actually reads) — the NUMERIC
      relation `alpha == k/(rho*cp)` is verified to hold (the dimensions
      force this exactly: `[k]/([rho]*[cp]) = [L^2/T]`). Confirmed
      `car_brake_thermal`'s own declared values satisfy this to <0.01%;
      a synthetic mismatch is caught with a clear >2% relative-error
      report.
    - incompressible Navier-Stokes (`navier_stokes_incompressible`,
      `incompressible_navier_stokes_2d`): `Re`/`inv_Re` must be positive
      and dimensionless, `Re*inv_Re == 1` when both given, `nu` (if
      given) must be a positive [L^2/T] viscosity.
    - linear elasticity (`linear_elasticity`, `_plane_strain`,
      `_plane_stress`): `E` and `mu` (shear modulus) must be positive
      stresses; `lambda`/`lam` (first Lame parameter) need only be
      finite — NOT positive, since an auxetic material makes it
      negative — with the real stability constraint checked on the
      implied bulk modulus `lambda + 2*mu/3 > 0` instead; `nu` must be
      in (-1, 0.5); and when `E`, `nu`, `lambda`, `mu` are all given, the
      standard isotropic relations `lambda == E*nu/((1+nu)(1-2nu))` and
      `mu == E/(2*(1+nu))` are verified numerically. **Real finding from
      building this**: `aircraft_wing_structural` and
      `car_suspension_fatigue` (both in `presets/engineering.py`) declare
      their Lame parameter under the key `"lam"`, but
      `compile.py`'s `linear_elasticity*` residual reads
      `params["lambda"]` only (defaulting silently to `1.0` if absent) —
      this check now catches that mismatch and reports it as a failing
      `dimensional_analysis` check; the two presets themselves are NOT
      fixed as part of this pass (out of scope for the guardrail work,
      left as a now-documented, concretely-reproducible open item for
      whoever next touches `presets/engineering.py`).
  Every other `pde_kind` (laplace, poisson, wave_equation, darcy, stokes,
  hyperelasticity_neo_hookean, maxwell_te, biot_poroelasticity, ... —
  everything not in the five families above) falls back to the
  ORIGINAL positivity-only heuristic (still runs, so nothing regresses,
  but not the same rigor) — the report's own check name
  (`dimensional_analysis` vs. `parameter_sanity`) tells a caller which
  one actually ran for a given spec, rather than leaving it implicit.

  `_check_conservation` re-evaluates a Monte-Carlo boundary-flux estimate
  for **exactly two families**, and is ABSENT from the report (never a
  false pass) for every other `pde_kind`:
    - incompressible continuity (`navier_stokes_incompressible`,
      `incompressible_navier_stokes_2d`): net outward volume flux of the
      model's velocity field through `domain_bounds`'s box boundary,
      which the divergence theorem says must be ~0 for a divergence-free
      field. For a time-dependent spec, evaluated at one fixed time
      slice (continuity holds instantaneously, not just aggregated over
      time).
    - steady heat conduction with **no source term** (`heat_equation_
      steady`, `_multilayer`, `_anisotropic`): net Fourier's-law flux
      (`-k*grad(T)`, or `-k_i*dT/dx_i` per-axis for the anisotropic
      kind) through the boundary, must be ~0. This explicitly ASSUMES
      `q=0` — a spec solved with a nonzero `ctx['source_fn']` (a
      training-time argument, not part of `ProblemSpec` itself, so this
      check has no way to see it) will report a nonzero imbalance even
      for an otherwise-correct solution. Documented as an explicit scope
      limit, not silently ignored.

  Both report a *normalised* imbalance ratio (`|net flux| / sum(|per-face
  flux|)`), not a raw flux value, specifically so one threshold works
  across arbitrarily different domain sizes/field magnitudes. The
  default threshold (0.15) and sample count (2048 points/face) were
  picked from an empirical noise-floor characterization done BEFORE
  choosing them, not guessed: on a numerically-EXACT divergence-free 3D
  velocity field (built from `curl` of a smooth vector potential, so its
  divergence is exactly zero by construction), the observed imbalance
  ratio at 2048 points/face had mean ~0.007–0.010 and max ~0.027–0.05
  over 50-trial resampling on domains matching this repo's own
  `channel_flow_3d`/`lid_driven_cavity_3d` presets; the steady-conduction
  analogue is far quieter (~1e-4 at the same sample count, on an exact
  harmonic `T`). A deliberately non-conservative field (nonzero-
  divergence velocity; a non-harmonic `T` implying a hidden source) gave
  a ratio of exactly 1.0 in both cases (no cross-face cancellation at
  all). 0.15 sits ~3–20x above the observed noise ceiling and ~6.7x
  below the "clearly broken" ratio of 1.0 — comfortable margin on both
  sides for a Monte-Carlo estimator, not a coin-flip threshold; see
  `tests/test_physics_guardrail.py` for the exact-good/exact-bad
  contrast tests this was validated against (24 tests, all passing) and
  `pinneapple_llm/guardrail.py`'s own class docstring for the full
  derivation.

  **Reference-data auto-fetch: audited and partially built in this
  follow-up pass.** Audit of `pinneapple_pdb` (`builder.py`,
  `templates.py`, `validate.py`, `shard.py`, `derived.py`, read in full)
  found **no named-benchmark-dataset catalog** — the thing the paragraph
  above envisioned does not exist: `PhysicalDatasetBuilder` only ever
  builds datasets by querying external Earth-data hubs (NASA CMR /
  earthaccess) and writing the result to disk; it has no reader of its
  own. Its `catalog_path` parquet is an *output* manifest of what a given
  build produced (keyed by a content-hash `uid`), not an *input* registry
  a caller could resolve a friendly name against without having built it
  first. `schema_templates()` is the only name→dict lookup in the
  package, and it returns physical-schema metadata (governing equations,
  units policy) — never x/y data arrays. Inventing a fake named catalog
  with one or two placeholder entries just to have something to ship
  would be exactly the kind of overclaiming this whole gate exists to
  prevent, so that was not done.

  What was built instead — real and useful, but explicitly *not* the
  named-catalog lookup: `PhysicsGuardrail.check()` now also accepts a
  `reference_dataset_path` argument, an ADDITIVE alternative to the
  existing manual `reference_x`/`reference_y` arrays (which keep working
  completely unchanged). It resolves a real **file path** to the on-disk
  UPD zarr format `PhysicalDatasetBuilder._write_upd` actually writes for
  every shard (a plain `xr.Dataset.to_zarr(...)` store — a real,
  already-existing artifact of this codebase's own dataset-building
  path), loads it via the new `_load_reference_from_upd_zarr` helper
  (optional `reference_x_vars`/`reference_y_vars` pick which data
  variables/coordinates to stack, defaulting to `self.spec.coords`/
  `self.spec.fields`), and feeds the result into the existing, unchanged
  `_check_reference` exactly like a manually-supplied array pair would.
  Proven end-to-end against a real UPD zarr store built and written to
  disk inside the test (no mocks) — see `tests/test_physics_guardrail
  .py`'s new reference-data-auto-fetch section (6 new tests, 30 total in
  the file, all passing), including a test asserting the auto-fetch path
  and the manual-array path produce the identical `CheckResult` (same
  RMSE, same pass/fail) when fed the same underlying data.

  **Still NOT built**: the actual named-benchmark-dataset registry itself
  — that would require `pinneapple_pdb` to grow a real name→dataset
  catalog (e.g. a curated table of known DNS/experimental benchmarks
  resolvable by a short string like `"channel_flow_re_180_dns"`), which is
  out of scope for a guardrail-side change alone and remains open for
  whoever next touches `pinneapple_pdb`. Dimensional analysis and
  conservation both remain scoped to the families listed above, not
  generalized to every `pde_kind` this project supports (see
  `pinneapple_llm/guardrail.py`'s docstring for why a fully general
  per-equation symbolic balance-checker was explicitly not attempted this
  pass).

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
