# Evidenced audit report

Generated 2026-09-04 by actually running the test suites named below (not
estimated or guessed — every number here is a real test result). See
`ROADMAP_PHYSICS_AI_HUB.md` section 1.1 for the two-tier audit design this
implements (Tier A = breadth/"does it run", Tier B =
`tests/test_manufactured_solutions.py`, physics correctness).

## Astrophysics/space specialization: 7 new PDE/ODE kinds, all verified

PINNeAPPle's chosen initial domain specialization (see
`ROADMAP_PHYSICS_AI_HUB.md` section "Domain specialization"): 7 new,
literature-grounded benchmark presets in
`pinneapple_physics/pde_environment/presets/astrophysics.py`, spanning
industrial space engineering (satellite orbit propagation, space-debris
conjunction assessment, spacecraft attitude control) and research
astrophysics (stellar structure, dark-matter halo potentials, compressible
hydrodynamics). Each needed a new PDE/ODE residual kind in the compiler
(`pinneapple_physics/pinn_solver/compiler/compile.py`), except
`nfw_dark_matter_potential`, which reuses the existing "poisson" kind.

**Every closed-form solution used for validation was derived or verified
with `sympy` this session before being written into code** (substituted
into its governing ODE/PDE, confirmed exact zero residual) — not
recalled from memory and trusted. This caught one real error: the first
version of the J2 perturbation acceleration (from memory) had the wrong
overall sign; the version actually shipped was derived as -grad(V) of the
cited geopotential and is correct by construction.

`tests/test_astrophysics_validation.py` (Tier B, same method as
`test_manufactured_solutions.py`): plug the exact solution into the
compiled residual with no training, assert ~0; plug a deliberately wrong
solution in, assert clearly nonzero. **16/16 pass** (8 presets/kinds x
exact+wrong, except `satellite_j2_perturbation`, which has no closed-form
trajectory — see below):

| Preset | Kind | Exact solution used | Result |
|---|---|---|---|
| `kepler_two_body_orbit` | `kepler_two_body_orbit` | Kepler's equation (Newton-Raphson, differentiable) | ✅ residual ~1e-19 |
| `space_debris_cw_relative_motion` | `space_debris_cw_relative_motion` | Clohessy-Wiltshire closed form | ✅ residual ~0 |
| `satellite_j2_perturbation` | `satellite_j2_perturbation` | no closed form; checked instead: J2=0 reduces exactly to two-body (res ~1e-19), J2=Earth's value measurably perturbs it (res ~3e-11) | ✅ both directions confirmed |
| `spacecraft_attitude_euler_rotation` | `spacecraft_attitude_euler_rotation` | axisymmetric torque-free precession | ✅ residual ~0 (<1e-8) |
| `lane_emden_polytrope` | `lane_emden_polytrope` | n=0 and n=1 closed forms | ✅ both ~0 (n=0 caught a real bug, see below) |
| `nfw_dark_matter_potential` | `poisson` (existing kind) | NFW closed-form potential | ✅ residual ~0 |
| `sod_shock_tube_astro` | `euler_compressible_1d` | smooth advected-pulse MMS (the real Sod solution is discontinuous, not usable for pointwise autograd MMS) | ✅ residual ~0 (<1e-8) |

**Real bug caught by this process**: the Lane-Emden branch's
`theta_pow_n = sign(theta) * abs(theta)**n` "safe negative-base power"
trick is mathematically wrong for even integer n (it returns -1 instead
of the correct +1 for `theta**0` when theta<0) — the n=0 exact-solution
test failed with residual 0.234 instead of ~0 before the fix. Corrected
to use `theta ** int(n)` directly whenever n is an integer (0, 1, 5 all
are), falling back to the sign-preserving regularization only for
genuinely non-integer n. This is exactly the kind of defect Tier B is
for: the code ran without error before the fix (Tier A would have missed
it entirely) but computed the wrong physics.

**Also fixed as a byproduct**: two pre-existing Category-1 "Unsupported
PDE kind" gaps from the Tier A audit below, `sir_ode` and
`pk_two_compartment_ode`, were fixed using the same
compiler-branch pattern built for the new astrophysics ODE kinds (both
now compile and run; Tier A failure count went from 62/137 to 60/137 as a
direct result, confirmed by re-running the full suite).

**Known gaps from the first pass, closed in a follow-up pass** (below):
`satellite_j2_perturbation`'s secular drift-rate formulas and
`lane_emden_polytrope`'s astrophysically-standard n=1.5/n=3 both lacked
any check beyond the instantaneous-residual level. Both are now
independently validated (see "Follow-up pass" below). The CGNS/Exodus/
Fluent/Abaqus readers' "not validated against a real writer's file"
caveat remains open and unrelated to this specialization.

### Follow-up pass: end-to-end training + independent numerical validation

Two things the first pass explicitly flagged as missing, both addressed:

**1. Actually training networks end-to-end** (the first pass only proved
the residual/physics implementation is correct, never trained a network
and checked its own output against the truth):

- `examples/pde_environment/05_kepler_orbit_validation.py`: first attempt
  (physics residual + IC only) converged PDE+IC loss to ~0.001 — looking
  done — while position RMSE was **~104% of the semi-major axis**
  (completely wrong trajectory shape). This is a real, reproduced PINN
  pure-IVP failure mode, not a footnote: an IC pins the solution at one
  point only, so small residual+IC loss does not imply the network found
  the correct global trajectory among the many that locally satisfy it.
  Fixed by adding 15 sparse "tracking-data" points from the exact
  solution as a `DataConstraint` (which is also an honest reframing of
  the preset's real industrial use case: orbit determination genuinely
  is fitting dynamics to sparse tracking observations). Final result
  (Adam + cosine LR decay + grad clipping, 3000 epochs, ~45s CPU):
  **1.85% position RMSE, 2.65% velocity RMSE**; conserved quantities not
  directly supervised (specific energy, angular momentum) matched exact
  values to ~1.9%/~0.7%.
- `examples/pde_environment/06_space_debris_cw_validation.py`: repeated
  the same experiment for `space_debris_cw_relative_motion` (a *linear*
  ODE, unlike Kepler's nonlinear one) — the same pure-IVP collapse
  reproduced again (loss ~1e-12, RMSE ~360%), confirming the failure mode
  is about IVP structure, not nonlinearity. A second, independent
  pitfall was found while fixing it: Hill's along-track coordinate y(t)
  has a genuine secular (linearly-growing) term reaching ~-12 km over one
  period while x/z stay within ±1.5 km — the single shared
  position-scale that worked for Kepler badly under-scales y here and
  stalls convergence at ~17% RMSE; per-axis scaling matched to each
  coordinate's actual range fixed it. Final result: **3.00% position
  RMSE**.

**2. Independent numerical validation of the two previously-open gaps**
(both via `scipy.integrate.solve_ivp`, reimplemented independently of
`compile.py` — checking the physics, not the compiler code against
itself):

- `tests/test_j2_secular_validation.py`: integrates the J2-perturbed
  equations of motion over 800/400 orbits, fits the osculating
  RAAN/argument-of-perigee secular trend, compares to Vallado's
  literature formulas. **Nodal regression: 0.45% agreement. Apsidal
  precession: 0.53% agreement.** (Caveat documented in the test itself:
  naively checking apsidal precession near the ~63.435° critical
  inclination gives a spurious "45% error" from the literature formula's
  own near-zero denominator there, not a real discrepancy — the test
  deliberately uses 30° instead.)
- `tests/test_lane_emden_numerical_validation.py`: integrates the
  Lane-Emden equation for n=1.5 (white dwarf) and n=3 (Eddington standard
  model), finds the first zero crossing ξ₁, compares to published
  textbook tables (Chandrasekhar 1939). **n=1.5: 0.0001% agreement. n=3:
  0.00002% agreement.** (The integrator is first sanity-checked against
  the n=0/n=1 closed forms, matching to <1e-6, before being trusted for
  the no-closed-form cases.)

Both test files pass (2/2 and 4/4 respectively), run in a few seconds
each, and are part of the regular suite going forward.

---

## Zero — the pre-existing test suite (`tests/`, excluding the two new
Tier A/B files below): 4 collection-blocking bugs found and fixed, then
**0 failures across the entire suite**

Running `pytest tests/` at the start of this pass didn't run a single
test — it aborted immediately with 4 `ModuleNotFoundError`s at collection
time (pytest aborts collecting the *whole* directory tree on any
module-level import error, so these 4 broken files were silently blocking
every other pre-existing test in the repository, tested or not, from ever
running). All four turned out to be the same underlying pattern —
compatibility shim packages (`pinneapple_train`, `pinneapple_solvers`,
`pinneapple_models`) whose `__init__.py` only re-exports symbols at the
flat top level, while a large number of call sites across the repo
(examples, the library's own internal NLP-to-PDE agent knowledge base,
`pinneapple_tools.benchmark_suite`, and these tests) import from them as
genuine *submodules* (`pinneapple_train.trainer`, `.losses`, `.metrics`;
`pinneapple_solvers.fft`; `pinneapple_models.registry`) that never
existed as real files:

| Broken import | Real location | Blast radius found |
|---|---|---|
| `pinneapple_pinn.factory.pinn_factory` (package doesn't exist at all) | `pinneapple_physics.pinn_solver.factory.pinn_factory` | 1 test file (fixed the import directly — no established shim convention to extend for this one) |
| `pinneapple_train.trainer` / `.losses` / `.metrics` | `pinneapple_neural.trainer.{trainer,losses,metrics}` | 7 example scripts, `pinneapple_tools/benchmark_suite/{timeseries_pipeline,physics_pipeline}.py`, and **the library's own `pinneapple_problemdesign` NLP-to-PDE agent knowledge base** (`knowledge/{pinneapple_capabilities,mapping}.py`) documented this exact broken API — added real `pinneapple_train/{trainer,losses,metrics}.py` submodules |
| `pinneapple_solvers.fft.FFTSolver` | `pinneapple_simulation.numerical_solvers.fft.FFTSolver` | plus a second, independent bug in the same file: `pinneapple_solvers/__init__.py`'s own top-level re-export imported a nonexistent `FFTProcessor` name inside a bare `try/except`, silently leaving it `None` forever with no error ever surfacing — fixed both |
| `pinneapple_models.registry.ModelRegistry` | `pinneapple_neural.architectures.registry.ModelRegistry` | added `pinneapple_models/registry.py` |

Once collection was unblocked, running the full suite surfaced these
further real, independent bugs (all fixed, not just documented):

- **`pinneapple_systems/process_components/real_gas_eos.py`**: `CP.PT_INPUTS`/`CP.HmassP_INPUTS`/`CP.PSmass_INPUTS`
  are evaluated as plain function *arguments* at each `state_from_*` call
  site, i.e. before the function's own internal `_COOLPROP_AVAILABLE`
  check ever runs — so on any machine without CoolProp installed, all 12
  `test_process_components.py` tests failed with a bare, confusing
  `NameError: name 'CP' is not defined` instead of the clear
  `ImportError` the code was already set up to raise internally. Added a
  `_require_coolprop()` guard at the top of all three entry points, added
  `pytest.importorskip("CoolProp")` to the test module (so it skips
  cleanly rather than fails on an environment without it, the same fix
  applied to `tests/pinneapple_geom/test_mesh_ops.py`'s bare `import
  trimesh`), and added `CoolProp` to `pyproject.toml`'s new `process`
  extra (it wasn't listed anywhere before).
- **`pinneapple_neural/trainer/trainer.py`**: `Trainer.fit()` never
  returned per-epoch history at all (only the final `best_val`/
  `best_path`), despite `RunLogger` already computing exactly that data
  every epoch and simply not passing it back to the caller. Added a
  `"history"` key (list of per-epoch `{"epoch", "train_total",
  "val_total", ...metrics}` dicts) to the returned dict.
- **`pinneapple_app/backend/routers/experiments.py`**: the experiment
  progress callback unconditionally read `ev["epoch"]`/`ev["total_epochs"]`/
  `ev["model"]`, but `ExperimentRunner` sends two other, structurally
  different event shapes through the *same* callback for its auto-fix
  advisor loop (`type="advisor"`/`type="retrain"` events, which carry no
  `epoch`/`total_epochs` key at all) — so **any experiment run whose
  auto-fix advisor loop fired at all crashed the entire experiment** with
  `KeyError('epoch')`, reported to the user only as a bare `"failed"`
  status with message `"'epoch'"`. This is arguably the most
  user-visible bug found this session (a core, advertised feature of the
  app silently ate every experiment that used it) and was **not caught by
  any existing test** — `tests/test_app_backend.py`'s
  `TestExperiments` class existed and exercises exactly this path, but
  had never been run successfully before (it couldn't get past the
  collection-blocking bugs above to even execute). Fixed by branching on
  `ev.get("type")` before assuming the training-event shape.

**Net result**: `pytest tests/ --ignore=tests/test_full_library_matrix.py
--ignore=tests/test_manufactured_solutions.py` went from *4 collection
errors, 0 tests run* to **0 failures**, running the entire pre-existing
suite (all of it that doesn't need an unavailable optional dependency,
which now skip cleanly instead of erroring).

**Later re-run note**: in a later re-run this session (while validating
the astrophysics additions above), `tests/test_app_backend.py`'s 34
tests could not be re-verified in this particular environment because
`httpx` (required by `starlette.testclient.TestClient`) was not
installed and this session's sandbox blocked a global `pip install`.
This is an environment gap, not a code regression -- `httpx` was missing
from `pyproject.toml`'s `dev` extra entirely (now added) despite being a
hard requirement for this test file, so `pip install -e .[dev]` would
not have caught it either. The rest of the suite (everything except this
one file) was re-run in full and showed zero regressions from this
session's compile.py changes (60/137 Tier-A failures, down from 62, the
2-failure improvement being the `sir_ode`/`pk_two_compartment_ode` fixes
documented in the astrophysics section above -- not a new problem).

---

## Tier A / Tier B (this session's own new tests)

**Totals**: 90 registered architectures (`ModelRegistry.list()`) + 47
registered PDE presets (`list_presets()`) = 137 Tier-A checks. **62
failed** (45%), the rest passed or were skipped as not applicable to this
generic smoke test's assumptions (see categories below).

This is not a claim that 45% of the library is "broken" -- most of the 62
failures fall into two large, distinct categories, only one of which is a
real defect. Read the categories, not just the count.

## Category 1 — real defect: presets registered but not compilable at all

**~35 of the 62 failures.** These presets are discoverable via
`list_presets()` and `get_preset(name)` succeeds, but
`compile_problem(spec)`'s `pde_kind` dispatch (`pinneapple_physics
.pinn_solver.compiler.compile.py`) has no branch for their `spec.pde.kind`
at all — `solve_pde` (and therefore `pipeline()`, and therefore any normal
usage) raises `ValueError: Unsupported PDE kind: <kind>` immediately.
**This is the single largest real gap this audit found.** Preset names
observed with this exact failure (`pde.kind` values in parentheses; the
underlying kind, not always identical to the preset name):

```
aircraft_wing_aerodynamics, aircraft_wing_structural,
axial_compressor_cascade_2d, axial_compressor_meanline,
axial_compressor_stage_3d, bekker_wong_surrogate_2d, black_scholes_1d,
car_brake_thermal, car_external_aero, car_suspension_fatigue,
climate_atmosphere_2d, climate_ocean_gyre (stommel_gyre_2d),
cpu_heatsink_thermal (heat_equation_steady), crystal_phonon
(phonon_bte_1d_gray), datacenter_airflow_2d, datacenter_cfd_3d,
datacenter_server_thermal, drug_diffusion_tissue, fan_cooler_cfd,
furnace_combustion_zone, heston_pde_2d, industrial_furnace_thermal
(heat_equation_steady), material_fracture_2d (phase_field_fracture_2d),
opinion_dynamics_2d, pcb_thermal (heat_equation_steady), pk_two_compartment
(pk_two_compartment_ode), plane_strain_2d, plane_stress_2d
(linear_elasticity_plane_stress), refractory_lining
(heat_equation_steady), rocket_nozzle_cfd, rocket_structural, sir_epidemic
(sir_ode), thermoelasticity_2d, threaded_coupling_tc50_rotating,
von_mises_2d (linear_elasticity_plane_stress)
```

A second, smaller variant of the same defect: `channel_flow_3d`,
`lid_driven_cavity_3d`, `pipe_flow_3d` are **steady** presets (no `t`
coordinate by design — see the architecture report this project's
`splash-pinneapple` pipeline was built from), but
`navier_stokes_incompressible`'s branch in `compile.py` unconditionally
raises `ValueError: Navier–Stokes expects time coord 't'.` for any spec
without one. These three are registered, physically sensible, steady
Navier-Stokes presets that `solve_pde` cannot run at all.

**Recommendation**: this is the top-priority follow-up work, larger than
what this session had budget for (~20+ distinct `pde_kind` residual
implementations, each its own physics-derivation task, not a mechanical
fix). Tackle in the order real users would hit them: thermal
(`heat_equation_steady` covers 4 presets in one fix), structural
(`linear_elasticity_plane_stress`/`plane_strain` covers 3), then the rest.

## Category 2 — not a defect: generic smoke test doesn't fit the architecture's real input shape

**~20 of the 62 failures.** `ModelRegistry.list()` includes time-series
models (`arima`, `esn`, `esn_rc`, `koopman`, `dmd`, `pod`, `havok`,
`ode_rnn`, `neural_cde`, `hybrid_rbf`, ...) and sequence/image
architectures (`transformer`, `afno`, `conv2d`, `conv3d`, `gno`, ...) that
are not meant to take a flat `(N, coord_dim)` PINN collocation batch —
they expect `(B, T, features)` sequences or `(B, C, H, W, ...)` grids.
Feeding them this test's generic `(8, 4)` point-cloud tensor raises a
shape error, which is the test's input assumption not fitting them, not
evidence they are broken. The test explicitly `skip`s (not fails) on a
shape-flavoured error to make this distinction, and it still shows a
handful as `FAILED` rather than `SKIPPED` where the error message didn't
match the skip heuristic's keyword list — those specific names are
listed in the raw pytest output but were not further hand-verified this
session; treat them as "probably Category 2, not independently confirmed"
rather than assume they're Category 1.

## Category 3 — not a defect: missing optional dependency

The `noether_*` architecture family (`noether_abupt`,
`noether_aero_abupt`, `noether_aero_transformer`,
`noether_aero_transolver`, `noether_aero_upt`, `noether_transformer`,
`noether_transolver`, `noether_upt`) requires the `emmiai-noether`
package, not installed in this session's test environment. Genuinely not
a defect — expected behaviour without that optional dependency.

## Tier B (physics correctness) — passed

`tests/test_manufactured_solutions.py`'s two tests both pass: the
compiled `"laplace"` residual is ~0 (< 1e-8) for the exact harmonic
solution `u = x²-y²`, and clearly nonzero (> 1, in fact ≈16 as the exact
Laplacian predicts) for the non-harmonic `u = x²+y²`. This confirms the
`"laplace"` `pde_kind`'s residual implementation is genuinely correct, not
just "runs" — the one PDE kind this session had time to verify at Tier B
depth. Extending Tier B coverage to more `pde_kind`s (Poisson with a
manufactured source term, Burgers via Cole-Hopf, ...) is listed in
`ROADMAP_PHYSICS_AI_HUB.md` section 1.1 as follow-up work.

## What this session fixed vs. what it found but did not fix

Fixed this session (see `ROADMAP_PHYSICS_AI_HUB.md`'s P0 table for the
full list with file locations): `solve_pde()`, `UPDDataset.save(zarr)`,
`Trainer.fit`'s no-grad validation, the `33_rans_turbulence.py` template,
`PeriodicBC` multi-axis chaining, plus new capability additions (WALE LES,
NS body-force hook, `AdaptiveWeights`, stochastic/latent PINN utilities,
binary OpenFOAM + CGNS/Exodus/Fluent/Abaqus readers, model hub,
`pipeline()`, adaptive hyperparameter search, Blender bridge, `pinneapple
_llm`).

**Found but not fixed this session** (Category 1 above): ~20 distinct
`pde_kind`s referenced by registered presets with no `compile_problem`
branch, and the steady-Navier-Stokes gap. This is real, load-bearing
backlog, not a footnote — a preset that cannot be compiled is a preset a
user of `list_presets()`/`pipeline()` will hit and be confused by, and it
is the single biggest reliability gap this audit surfaced.
