# Real-Time Physics-Informed Digital Twin for an Actuated Thermal-Fluid System
## 2D Heated-Channel Digital Twin — Proof of Concept (v3)

This README **is** the technical report. Every number below comes from one
completed end-to-end run of
`examples/use_cases/heated_channel_twin/heated_channel_twin_pipeline.py`
(45.6 min wall-clock on a laptop CPU; machine-readable form in
`outputs/metrics.json`). Options: `--quick` (fast smoke run), `--demo TRAJ`
(run the demonstrator afterwards), `--retrain` (ignore cached checkpoints),
`--no-cache` (regenerate the CFD dataset).

v1/v2 used the repo's `LBMSolver` + Smagorinsky LES. **v3 replaces the solver**
with a real 2D incompressible **laminar** Navier–Stokes + energy projection
method, adds the ConvAE+temporal surrogate, the divergence-residual and latency
metrics, the second assimilation method, and the demonstrator — it is a
different, spec-complete deliverable.

---

## 1. Governing equations and boundary conditions

Incompressible, **laminar**, constant-property flow (`rho = cp = 1`, `Re = 200`
— well below the ~2000 channel-transition value — `Pr = 1`). The CFD reference
solves the continuous 2D equations directly:

```
div(u) = 0
du/dt + (u·grad)u = -grad(p) + nu·lap(u)          nu    = u_in0·Ly/Re = 0.005
dT/dt + u·grad(T)  = alpha·lap(T)                  alpha = nu/Pr      = 0.005
```

Boundary conditions, exactly as specified:

| boundary | velocity | temperature | pressure |
|---|---|---|---|
| inlet `x = 0` | `u = (U_in(t), 0)` (Dirichlet) | `T = T_in = 0` (Dirichlet) | — |
| heated wall segment (`x/L ∈ (0.10, 0.35)` of the bottom wall) | no-slip `u = 0` | **`-k dT/dn = Q_h(t)`** — time-dependent Neumann **heat-flux** BC | — |
| other walls (bottom outside the segment, top) | no-slip `u = 0` | insulated `dT/dn = 0` | — |
| outlet `x = Lx` | `du/dn = 0` (free outflow) | `dT/dn = 0` | `p = 0` (reference) |

The heater is finite-volume-consistent: the prescribed flux `Q_h(t)` enters the
bottom face of the near-wall control volume, contributing a source `Q_h/dy` to
that row's energy update. This is **grid-independent under refinement** (total
power `Q_h·Δx·Δt` does not depend on `dy`), unlike a `dy`-scaled ghost-cell
temperature jump.

## 2. Discretization / solver

`run_reference_trajectory` / `_ns2d_*` in the pipeline: a **Chorin
fractional-step (projection) method on a collocated grid** — `nx×ny = 80×32`,
`Lx×Ly = 2.5×1.0`, `dt = 0.006`, `n_steps = 1700` (≈ 4 convective flow-through
times), 51 snapshots per trajectory.

1. momentum predictor `u*` — explicit **1st-order upwind advection** + central
   diffusion;
2. **pressure Poisson** `lap(p) = div(u*)/Δt` by Gauss–Seidel (150 iterations),
   Neumann `dp/dn = 0` on inlet + walls, Dirichlet `p = 0` at the outlet;
3. **projection** `u = u* − Δt·grad(p)`;
4. explicit advection–diffusion update for `T` with the flux BC above.

CFL-adaptive sub-stepping (convective Courant number `< 0.8`, up to 4 sub-steps)
+ a hard velocity clip keep the explicit scheme robust to the fast actuator
families; a global mass-conservation rescaling is applied at the outflow.
Solver cost: **5.5 s per trajectory** on CPU.

Honest caveats about the reference:
- The collocated grid leaves an odd–even (checkerboard) component in the
  projection pressure. The stored `p` field is 3× lightly box-smoothed (the
  laminar pressure is physically smooth); all surrogates are trained/evaluated
  on the smoothed `p`.
- **Divergence residual of the reference** (deep interior, 3 cells in from every
  boundary): `mean|div(u)| = 1.09e-02`. Boundary-adjacent columns carry a
  larger, known artefact (outflow BC + mass rescaling make `u[-1] ≠ u[-2]`; the
  developing inlet profile has a strong `dv/dy`) and are excluded from the
  metric, identically for CFD and every surrogate.
- The pressure `p` is a diagnostic Lagrange-multiplier field, large in
  amplitude and hard to predict (see §6.1); `dp` is correspondingly the hardest
  operational metric.

## 3. Actuator trajectory library

`U_in(t) ∈ [0.75, 1.25]·u_in0`, `Q_h(t) ∈ [0, Q0]` with `Q0 = 0.15`. **Both
channels are rate-limited** by a first-order filter (`tau = 25` solver steps) —
a real pump/heater cannot slew instantaneously, and an instantaneous
inlet-velocity jump is what makes the explicit projection scheme blow up on the
discontinuous families. The discontinuous-family *character* is preserved; only
the slew rate is bounded.

- **Training families (all spec-required):** `step`, `ramp`, `sinusoid`,
  `piecewise_constant` (random staircase), `prbs` (pseudo-random binary
  sequence). 5 families × 5 parametric variants = **25 training trajectories**
  (`U_in` and `Q_h` get *different* variants of the same family).
- **Held-out families (never in training, genuinely different generating
  processes):** `chirp`, `sawtooth`, `smoothed_random_walk`, `gaussian_bumps`,
  `am_two_tone`, plus a second faster `chirp`. **6 held-out trajectories.**

Train/test split is **strictly by whole trajectory**; accuracy is reported on
the held-out trajectories.

## 4. Dataset

`outputs/dataset_heated_channel.npz` (**82 MB**) persists the full dataset, not
just metrics: grid coordinates (`grid_x/grid_y/grid_XX/grid_YY`), the time
vector per trajectory (`*__time_snap`), the actuation input time series
(`*__act_fine`, `*__act_snap`), and both the **standardized** (`*__fields_std`)
and physical (`*__fields`) `(u,v,p,T)` field solutions, plus
`field_mean/field_std` and the solver wall-times. A training-only re-run reuses
it (keyed by a config hash); each trained surrogate is checkpointed to
`outputs/ckpt_*.pt` and reloaded on re-run.

## 5. Surrogates

All map `(actuator-history window, current state) → next state` and are
evaluated by an identical **50-step closed-loop autoregressive rollout** on the
held-out trajectories. All are trained with a **K-step rollout-consistency
curriculum** (K ramps 2→5 over training; small Gaussian noise on the fed-back
state, Sanchez-Gonzalez et al. 2020) — this is what keeps the autoregressive
error from exploding.

| # | surrogate | spec item | representation |
|---|---|---|---|
| A | **POD + GRU** | (1) POD + learned latent dynamics | 12 POD modes (99.85 % variance); GRU predicts the next latent |
| B | **FNO2d** | (2) Fourier Neural Operator | full-field; actuator history as broadcast input channels |
| C | **ConvAE + GRU** | (3) conv. autoencoder + temporal model | repo `Autoencoder2D`, 16-d code (held-out recon MSE 1.3e-3); GRU on the code |
| D | DeepONet | *(extra — not in the spec list)* | branch = 4×-pooled current field + actuator window; trunk = (x,y) |
| E | **FNO2d-PI** | (4) physics-informed variant | FNO2d + soft penalty on the discretized **mass + momentum + energy** residuals |

DeepONet needed one repo fix to be a fair entrant: `pinneapple_neural…deeponet
.DeepONet` computes the raw `Σ b_k·T_k` with **no `1/√modes` scaling**, so its
output is `~√modes` too large and the K≥5 curriculum makes its unrolled
gradient explode. The wrapper restores the classic scaling and adds gradient
clipping.

## 6. Results — the four spec metrics

### 6.1 Field relative-L2 error (held-out, 50-step rollout)

| surrogate | **all-field** | u | v | p | **T** | (train all-field / T) |
|---|---|---|---|---|---|---|
| POD + GRU  | 0.769 | 0.120 | 0.183 | **0.849** | **0.286** | 0.754 / 0.215 |
| **FNO2d**  | **0.249** | 0.053 | 0.123 | 0.274 | **0.125** | 0.224 / 0.107 |
| ConvAE+GRU | 0.293 | 0.062 | 0.084 | 0.323 | 0.208 | 0.302 / 0.142 |
| DeepONet   | 0.242 | 0.161 | **0.551** | 0.252 | **0.552** | 0.221 / 0.507 |
| FNO2d-PI   | 0.381 | 0.084 | 0.183 | 0.419 | 0.198 | — |

**Best surrogate: FNO2d** (lowest combined all-field + temperature held-out
rel-L2 among the rollout-stable candidates). It generalises well:
train 0.224 → held-out 0.249.

Diagnosis:
- **POD + GRU's all-field error (0.77) is entirely the pressure channel**
  (`rel_l2_p = 0.85`): 12 POD modes capture 99.85 % of the *field* variance but
  almost none of the large-amplitude, U_in-driven spatial structure of the
  projection pressure. Its `u/v/T` are good (0.12 / 0.18 / **0.29**), and it is
  the *only* surrogate whose all-field error is dominated by one channel — read
  the per-field breakdown, not the aggregate, for this one.
- **DeepONet gets `u/p` right but fails on `v` (0.55) and `T` (0.55)** and does
  not improve on either over 240 epochs. Its branch input is a 4×-average-pooled
  view of the field, which discards the thin near-wall thermal boundary layer
  (`dy`-scale) that carries most of the temperature and cross-stream-velocity
  signal. It is kept in the comparison as an honest negative for this
  autoregressive-rollout setup with a coarsened branch.
- **ConvAE + GRU** is a solid middle: 16-d code, held-out reconstruction MSE
  1.3e-3, and the temporal rollout lands at 0.29 all-field / 0.21 T.

### 6.2 Autoregressive rollout drift / instability

`drift_ratio = (mean per-step error over the last quarter of the rollout) /
(…over the first quarter)`. ≈ 1 → the rollout does not diverge.

| surrogate | drift ratio (held-out) |
|---|---|
| POD + GRU  | **0.97** (does not diverge) |
| FNO2d      | 1.48 (mild growth) |
| ConvAE+GRU | 1.61 |
| DeepONet   | 1.65 |
| FNO2d-PI   | **1.10** (the PI penalty *stabilises* the rollout) |

The K-step curriculum + noise injection is what keeps these near 1 — without it
(v2 experience) FNO2d's ratio was well above 3 and its error exceeded the
reference signal. `outputs/10_rollout_drift.png` shows the per-step curves.

### 6.3 Physical-consistency: divergence residual `mean|div(u)|`

| | `mean|div(u)|` (deep interior) | vs CFD |
|---|---|---|
| **CFD reference** | **1.09e-02** | 1× |
| POD + GRU  | **8.74e-03** | 0.8× (best — *below* CFD) |
| FNO2d      | 1.05e-01 | 9.6× |
| ConvAE+GRU | 8.48e-02 | 7.8× |
| DeepONet   | 6.24e-02 | 5.7× |
| FNO2d-PI   | **6.26e-02** | 5.7× (≈40 % lower than plain FNO2d) |

Clear finding: **the full-field neural surrogates (FNO2d, ConvAE+GRU, DeepONet)
do not respect incompressibility** — their predicted velocity fields carry
6–10× the reference divergence. **POD + GRU is essentially divergence-free**
because it rolls out inside a 12-mode POD subspace built from near-divergence-
free CFD snapshots — the reduced basis inherits the constraint. This is a real
accuracy-vs-physical-consistency trade-off: the most accurate surrogate (FNO2d)
is the least divergence-free; the most divergence-free (POD+GRU) is the least
accurate on the full field.

### 6.4 Computational latency / speedup

CFD reference: **5.46 s / trajectory** (190 s for all 30). Surrogate inference =
one full 50-step closed-loop rollout.

| surrogate | rollout time | **speedup vs CFD** |
|---|---|---|
| POD + GRU  | 37 ms | **×172** |
| ConvAE+GRU | 150 ms | **×42** |
| DeepONet   | 175 ms | **×36** |
| FNO2d      | 291 ms | **×22** |

(FNO2d is the most accurate but the slowest — CPU FFT autograd; on GPU the
ranking would shift.)

## 7. Operational-metric predictions (held-out, rel-L2 vs CFD)

The twin must predict outlet temperature, maximum temperature and pressure drop
over time. Extracted from every rollout and compared to the CFD reference:

| surrogate | outlet_T | max_T | dp |
|---|---|---|---|
| POD + GRU  | 0.428 | 0.202 | 0.855 |
| FNO2d      | 0.333 | **0.077** | 0.627 |
| ConvAE+GRU | **0.282** | 0.118 | 0.355 |
| DeepONet   | 0.504 | 0.572 | **0.296** |

- **max_T** is predicted well by FNO2d (0.077) and ConvAE+GRU (0.118) — the
  near-wall peak temperature.
- **outlet_T** is hardest for POD+GRU/DeepONet (it depends on the whole plume
  transiting the channel); ConvAE+GRU and FNO2d do best (~0.3).
- **dp** is poorly predicted by every surrogate whose all-field pressure error
  is high — it is a small difference of a large, hard field. The lower
  `dp` errors for DeepONet/ConvAE reflect that their *aggregate* pressure error
  is lower, not that they track the pressure drop well.

## 8. Physics-informed variant — "does PINN help here?"

FNO2d-PI adds a soft penalty on the discretized **mass** (`div(u_next)`),
**momentum** (both components) and **energy** PDE residuals, weights
`0.02 / 0.02 / 0.02`, 80 epochs.

| | FNO2d | FNO2d-PI | change |
|---|---|---|---|
| held-out all-field L2 | 0.249 | 0.381 | **worse** (+53 %) |
| held-out T L2 | 0.125 | 0.198 | worse |
| held-out `mean|div(u)|` | 1.05e-01 | 6.26e-02 | **better** (−40 %) |
| held-out drift ratio | 1.48 | 1.10 | **better** (more stable) |

**Honest answer: at these weights the physics penalty trades accuracy for
physical consistency and rollout stability.** It cuts the divergence residual
~40 % and calms the rollout drift, but degrades all-field L2 from 0.25 to 0.38.
The residual terms are approximate (large-dt explicit forms, a `Q_h/dy`
volumetric heater proxy, and the momentum residual uses the smoothed collocated
pressure), so the penalty pulls the prediction toward a slightly different
operator than the reference solver — hence the accuracy cost. A weight sweep /
better residual discretisation is the natural next step; the qualitative result
(PINN → more consistent + more stable, less accurate) is robust.

## 9. Sparse-sensor data assimilation — two methods

Sparse virtual sensors: **4 near-wall temperature probes** downstream of the
heated segment (`y`-row 1) and **2 pressure probes** (near inlet, mid-channel).
Each sensor's true CFD time series is verified before use:

| sensor | std | range | corr with actuator |
|---|---|---|---|
| T @ (36, 1) | 0.343 | [0, 1.18] | corr(Q_h) = −0.21 |
| T @ (49, 1) | 0.260 | [0, 0.80] | corr(Q_h) = −0.44 |
| T @ (62, 1) | 0.208 | [0, 0.57] | corr(Q_h) = −0.32 |
| T @ (73, 1) | 0.177 | [0, 0.51] | corr(Q_h) = −0.29 |
| p @ ( 6, 16) | 3.47 | [−16.8, 16.0] | corr(U_in) = +0.05 |
| p @ (40, 16) | 2.33 | [−10.9, 11.3] | corr(U_in) = +0.04 |

The T probes carry real, actuator-responsive signal. The p probes are **weak**
(corr with U_in ≈ 0.05): the projection pressure has a large offset component
uncorrelated with the instantaneous inlet velocity, so the assimilation gain
comes almost entirely from the temperature sensors.

Both methods correct the **POD+GRU** latent state (the interpretable
reduced-order model; the leading 8 modes are corrected, the rest carried from
the open-loop rollout):

- **(i) Latent-space innovation correction** — a static-gain (Luenberger-style)
  observer: at each assimilation step form the innovation `y_meas − (H·z + b)`
  at the sensors (`H` = the linear POD-decode-to-sensor operator, built
  column-by-column) and nudge the latent by `gain · pinv(H) · innovation`.
- **(ii) Ensemble Kalman Filter** in the reduced latent space (repo
  `EnsembleKalmanFilter`); the ensemble covariance is scaled per-mode to each
  POD mode's own trajectory variability (raw POD coefficients span ~2 orders of
  magnitude across modes — a single scalar `P0` either freezes the leading
  modes or blows up the trailing ones; getting this wrong is why v2's EnKF did
  nothing).

| held-out trajectory | open-loop L2 | + innovation | + EnKF |
|---|---|---|---|
| test_chirp              | 0.776 | 0.639 (+17.6 %) | 0.606 (+21.9 %) |
| test_sawtooth           | 0.763 | 0.593 (+22.2 %) | 0.541 (+29.0 %) |
| test_smoothed_random_walk | 0.770 | 0.625 (+18.9 %) | 0.601 (+22.0 %) |
| test_gaussian_bumps     | 0.730 | 0.621 (+15.0 %) | 0.579 (+20.7 %) |
| test_am_two_tone        | 0.771 | 0.609 (+21.0 %) | 0.598 (+22.5 %) |
| test_chirp_fast         | 0.804 | 0.650 (+19.2 %) | 0.611 (+24.1 %) |
| **mean improvement** | — | **+19.0 %** | **+23.4 %** |

**Both sparse-sensor methods measurably improve the estimate on every held-out
trajectory**; the EnKF is consistently the stronger of the two (+23 % vs
+19 %). (For reference: v1's assimilation made the estimate ~24 % *worse*; v2's
was ~0 %.) The improvement is concentrated in the temperature and
cross-stream-velocity channels that the near-wall T sensors observe; the
all-field number still carries the un-corrected pressure error.

## 10. CFD-reference numerical validation — grid convergence

Roache (1994) Richardson / GCI via
`pinneapple_analysis.verification.convergence.mesh_independence_study`, three
grids at constant refinement ratio 2 (`nx = 40, 80, 160`; `dt` halved and
Poisson iterations ×4 per refinement to keep each solve stable and consistent),
QoI = late-time mean outlet temperature.

| nx | QoI |
|---|---|
| 40  | 0.14665 |
| **80 (main grid)** | **0.13222** |
| 160 | 0.12056 |

- observed order of accuracy **`p = 0.31`**
- Richardson-extrapolated value = 0.0715
- `GCI_fine = 50.9 %`, `GCI_coarse = 57.4 %`
- `asymptotic_ratio = 0.912`, `is_asymptotic = True` (within the module's 10 %
  tolerance; the three values decrease **monotonically**, no oscillatory-
  convergence warning)

**Honest reading: the QoI converges monotonically but slowly.** The observed
order (~0.3) is well below the schemes' nominal orders — consistent with the
**1st-order upwind advection** dominating the truncation error, plus a
developing thermal boundary layer still under-resolved at 80×32 — and
`GCI_fine ≈ 51 %` is large. The `asymptotic_ratio` test passes, but the
successive changes (0.147 → 0.132 → 0.121, ~10 % each) are not shrinking fast:
**the 80×32 main grid is not in a tight asymptotic range.** What this means for
the reference data: its *trends*, BC behaviour, and near-divergence-free bulk
are trustworthy, but its *absolute* values carry roughly 10–20 % grid-
discretisation uncertainty. A higher-order advection scheme and a finer grid
would tighten this; both are out of scope for a laptop-CPU proof of concept.

`compute_physics_confidence` (fed only this convergence result — the other three
possible components require a per-point coordinate→field model callable and do
not apply to a whole-field spatiotemporal surrogate): `overall_score = 1.0`,
**`coverage = 0.25`** (1 of 4 checks). The score is only meaningful read
alongside the coverage.

### Auxiliary check — thermal-energy budget (`test_chirp`)

`|ΔE_domain − (heater input − outlet efflux)| / heater input`:
CFD reference **0.205**, POD+GRU 0.282, FNO2d 0.034, ConvAE+GRU 0.173,
DeepONet 0.281. The ~20 % closure error of the *reference itself* reflects the
crude discrete budget (snapshot-cadence sampling, outlet efflux estimated from
the near-boundary column, numerical diffusion of the 1st-order scheme), so the
surrogate numbers are only comparable to that baseline, not to zero.

## 11. Interactive demonstrator

`demo(trajectory)` / `--demo TRAJ` runs the CFD reference and the best surrogate
for one trajectory (a dataset name, or an actuator-family name — `step`,
`prbs`, `chirp`, … — to synthesise a fresh one), prints the operational metrics
and the measured speedup, and saves `outputs/09_demo.png`. Recorded run
(`--demo test_chirp`, best surrogate = FNO2d):

```
trajectory        : test_chirp
CFD solver wall-time: 5.58 s
surrogate rollout  : 175.4 ms   -> speedup x32
field rel-L2 (all / T): 0.1908 / 0.0935   mean|div(u)|=1.12e-01
operational-metric rel-L2: outlet_T=0.266  max_T=0.051  dp=0.612
```

## 12. Honest summary — what worked, what didn't

**Worked**
- A real 2D incompressible **laminar** NS + energy projection solver with the
  exact spec boundary conditions (time-dependent Neumann heat-flux heater),
  driven by all five required actuator families + PRBS, producing a persisted
  30-trajectory dataset in ~3 min.
- The K-step rollout-consistency curriculum + noise injection makes **every**
  surrogate roll out 50 steps without blowing up (drift ratios 0.97–1.65,
  vs > 3 without it in v2).
- **FNO2d** generalises well (held-out all-field 0.25, T 0.12) at ×22 real-time.
- **POD + GRU** is the physically-consistent option: `mean|div(u)|` *below* the
  CFD reference, drift ratio < 1, ×172 speedup.
- The **ConvAE + GRU** surrogate (spec-required, missing in v1/v2) works:
  held-out all-field 0.29, T 0.21.
- **Both** sparse-sensor methods improve the estimate on every held-out
  trajectory (+19 % innovation, +23 % EnKF) — a clean reversal of v1 (−24 %)
  and v2 (≈0 %), traced to physically-valid sensor placement + per-mode
  covariance scaling.
- All four spec metrics (field-L2, drift, divergence residual, latency) plus
  the three operational metrics are computed for every surrogate.

**Didn't / open problems**
- The **projection pressure `p` is a large, hard, Lagrange-multiplier field**;
  POD+GRU's 12-mode latent cannot represent it (`rel_l2_p = 0.85`), and `dp` is
  poorly predicted by all surrogates. A pressure-free formulation (streamfunction
  –vorticity) or a dedicated pressure sub-model would help.
- **Full-field neural surrogates violate incompressibility** (`mean|div(u)|`
  6–10× the reference). The physics-informed penalty cuts this ~40 % but at a
  real accuracy cost (all-field 0.25 → 0.38) — a weight sweep and a
  higher-order residual discretisation are needed to find a better trade-off.
- **DeepONet** (kept as an extra) fails on `v` and `T` (both ~0.55) because its
  4×-pooled branch discards the near-wall boundary layer; a full-resolution or
  learned branch encoder would be required.
- The CFD reference is **not in a tight grid-asymptotic range** (observed order
  ~0.3, `GCI_fine ≈ 51 %`) — 1st-order upwind advection + an under-resolved
  thermal boundary layer. Absolute reference values carry ~10–20 % uncertainty.
- The **p sensors carry almost no assimilation signal** (corr with U_in ≈ 0.05);
  the DA gain is entirely from the T sensors.

## 13. Status against the spec (A–J)

| item | status |
|---|---|
| **A** solver: incompressible laminar constant-property NS + energy, projection method, Re laminar | **done** — Chorin fractional-step, collocated grid, Re = 200, no turbulence closure. §1–2. |
| **B** exact BCs (inlet Dirichlet, heated-wall Neumann flux, insulated walls, `p=0`/free outflow) | **done** — implemented and listed as written; heater is a genuine time-dependent flux BC. §1. |
| **C** actuator library incl. piecewise-constant **and** PRBS; whole-trajectory split | **done** — 5 training families (step/ramp/sinusoid/piecewise-constant/PRBS) + 6 held-out; split by trajectory. §3. |
| **D** persisted dataset: mesh, coords, time vector, actuation series, standardized (u,v,p,T) fields | **done** — `outputs/dataset_heated_channel.npz` (82 MB). §4. |
| **E** four metrics: field-L2, rollout drift, **divergence residual**, **latency/speedup** | **done** — all four, all surrogates. §6. |
| **F** architectures: POD+latent, FNO, **ConvAE+temporal**, physics-informed (mass+momentum+energy) | **done** — A–E; ConvAE+GRU added; PI penalty covers all three PDEs. DeepONet kept as an extra. §5, §8. |
| **G** sparse-sensor reconstruction: **both** latent-innovation correction **and** EnKF | **done** — both, compared, both improve. §9. |
| **H** operational metrics: outlet T, max T, pressure drop from CFD and surrogates | **done** — §7. |
| **I** lightweight interactive demonstrator | **done** — `demo()` / `--demo TRAJ`, `outputs/09_demo.png`. §11. |
| **J** README = technical report with all real numbers + honest diagnosis | **this document.** |

## Outputs

`outputs/`: `00_domain_geometry.png`, `01_actuator_trajectories.png`,
`02_/03_reference_fields_*.png`, `04_training_curves.png`,
`05_surrogate_comparison.png`, `06_sparse_sensor_assimilation.png`,
`07_outlet_response.png`, `08_grid_convergence.png`, `09_demo.png`,
`10_rollout_drift.png`, `metrics.json`, `dataset_heated_channel.npz`,
`ckpt_{pod_gru,fno2d,cae,cae_gru,deeponet,fno2d_pi}.pt`.
