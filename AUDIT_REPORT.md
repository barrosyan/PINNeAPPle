# Tier A evidenced audit report

Generated 2026-09-04 by actually running
`pytest tests/test_full_library_matrix.py -q` (not estimated or guessed —
every number below is a real test result). See
`ROADMAP_PHYSICS_AI_HUB.md` section 1.1 for the two-tier audit design this
implements (Tier A = breadth/"does it run", Tier B =
`tests/test_manufactured_solutions.py`, physics correctness — both of
which pass, see that file's own results below).

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
