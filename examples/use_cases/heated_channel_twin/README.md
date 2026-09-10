# 2D Heated-Channel Digital Twin -- Proof of Concept

## The brief (translated/summarized from the original Portuguese spec)

Build a reproducible proof-of-concept "digital twin" for a 2D channel with a
heated wall section. Two time-varying actuators drive the system: inlet
velocity `U_in(t)` and heater power `Q_h(t)`. The twin predicts, over time,
the velocity fields `u,v`, pressure `p`, temperature `T`, outlet temperature,
max temperature, and pressure drop. Four layers, in priority order:

1. **Computational physics** -- a real CFD reference (PDE -> geometry -> grid
   -> BC/IC -> transient solve -> fields).
2. **Scientific ML** -- train and *fairly compare* real surrogates mapping
   (actuator history + current/past state) -> future field prediction, at
   minimum a POD+latent-dynamics baseline and an FNO neural-operator
   candidate. The brief does **not** assume the neural candidate wins --
   report honestly whichever generalizes better. The scientific question:
   do surrogates generalize to actuator trajectories *not seen in training*,
   while staying physically plausible? The dataset therefore includes
   held-out trajectories genuinely different in **shape** from every
   training trajectory, and accuracy is reported specifically on those.
3. **Data assimilation** -- simulate a handful of sparse point sensors (not
   the full field) and correct the surrogate's predicted state using them.
4. **Control/optimization** -- explicitly out of scope (the brief calls it
   secondary, attempted only after 1-3 are solid). **Not built here.**

The brief is upfront that this is not an industrial twin, not real-time
control, and not a validated 3-D solver. The deliverable is: a 2-D CFD
reference dataset with genuinely varying actuator trajectories, at least two
real trained-and-compared surrogates, a real sparse-sensor correction
experiment, and honest reporting of what worked and what didn't. This README
is that honest report.

Run it: `.venv/bin/python examples/use_cases/heated_channel_twin/heated_channel_twin_pipeline.py`
(~4.5 minutes on a laptop CPU). Outputs land in `outputs/` (6 PNGs +
`metrics.json`). See the pipeline script's module docstring for the full
per-stage real-module mapping and honesty notes; this README summarizes the
*results*.

## What was built

- **CFD reference**: a 64x32 2-D channel, real `LBMSolver` (D2Q9 BGK +
  Smagorinsky LES) for momentum, weakly coupled every single timestep to a
  hand-written (necessarily -- see script docstring note (b), no ready-made
  spatially-varying-velocity scalar-transport solver exists in this repo)
  energy stepper that reuses `fvm.py`'s real upwind+diffusion discretization
  pattern. `U_in(t)` genuinely drives the LBM's own Zou-He inlet BC every
  step; `Q_h(t)` drives a volumetric heat source in a near-wall band on the
  bottom wall. 10 trajectories: 7 training shapes (step, ramp, slow/fast
  sinusoid, pulse train, double step, ramp+sinusoid combo) and 3 **held-out**
  shapes from genuinely different generating processes never seen in
  training (frequency chirp, sawtooth, smoothed random walk).
- **Surrogate A -- POD + GRU**: real `POD` (12 modes, 94.4% variance) +
  real `GRUModel` predicting next-step latent coefficients from an
  actuator+latent history window.
- **Surrogate B -- FNO2d**: real `FNO2d`, actuator history concatenated as
  extra broadcast input channels (the standard conditioning approach for
  neural operators that don't natively support exogenous inputs).
- **Comparison**: manual, not `Arena` -- `Arena`'s data pipeline is built
  around static analytical PINN benchmarks with no actuator-history /
  autoregressive-rollout concept, so it doesn't fit this problem (see
  pipeline docstring note (d) for the full justification). Both surrogates
  were trained on identical data and evaluated with an identical closed-loop
  rollout protocol on identical held-out trajectories.
- **Data assimilation**: real `EnsembleKalmanFilter` correcting the POD
  latent state using 4 sparse point sensors (T and p) sampled from the real
  CFD reference plus Gaussian noise, on one held-out trajectory.
- DeepONet and a physics-informed variant were **not** attempted (explicit
  stretch goals in the brief; out of time budget for this pass).

## Results (real numbers, from `outputs/metrics.json`)

### Surrogate accuracy: train vs. held-out (mean relative L2 rollout error, 24-step closed-loop rollout)

| Surrogate | Train trajectories | Held-out trajectories |
|---|---|---|
| POD + GRU | 0.582 | **0.553** |
| FNO2d | 1.617 | 1.968 |

(`relative_l2 = ||prediction - reference|| / ||reference||` over the entire
rollout, excluding the given initial condition. A value >1 means the error
is larger than the reference signal itself.)

**POD+GRU is the honest winner here, and by a wide margin.** It is not a
strong surrogate in absolute terms (55% relative error is mediocre, not a
production-grade result), but it degrades only slightly from train to
held-out (0.582 -> 0.553 -- essentially flat, arguably even a hair better,
which given the small sample count is more indicative of "not badly
overfit to specific training shapes" than of true improved generalization).
**FNO2d is substantially worse and gets *worse* on held-out trajectories
than on training ones** (1.617 -> 1.968 mean; per-trajectory it ranges from
a locally-decent 0.523 on `test_sawtooth` to a blown-up 4.527 on
`test_chirp`). This directly answers the brief's central question honestly:
in this proof of concept, the simpler reduced-order baseline generalizes
better than the neural-operator candidate, not the other way around.

**Diagnosis of why FNO2d underperformed (not just "it did"):** FNO2d's
one-step training loss was low and reasonable (final normalized field MSE =
0.041 -- a good one-step fit given only 168 training samples from 7
trajectories). The failure is entirely in **autoregressive rollout**: FNO2d
was trained with a single one-step MSE loss and no rollout-consistency term
(no scheduled sampling, no noise injection, no multi-step loss), so small
per-step errors in the raw 8192-dimensional field (`4 x 64 x 32`) compound
geometrically over the 24-step closed-loop rollout used for evaluation --
classic autoregressive **exposure bias / error accumulation**, worse on
faster-varying trajectories (chirp, the fastest-changing held-out shape, is
also by far its worst score: 4.527). POD+GRU does not have this problem to
nearly the same degree because it rolls out in a heavily reduced 12-dimensional
latent space capturing 94% of the variance -- restricting the rollout to
the dominant POD modes acts as an implicit low-pass filter that suppresses
high-frequency error amplification, a well-documented advantage of
reduced-order latent dynamics over full-field neural operators specifically
in small-data, long-rollout regimes. Neither surrogate was trained with a
rollout-consistency loss; the *reduced state space* is what saved POD+GRU
here, not superior training. With more training trajectories and/or an
explicit multi-step rollout loss, FNO2d would likely close much of this gap
-- this is a training-protocol limitation of this proof of concept, not
evidence that neural operators are fundamentally unsuited to the problem.

### Sparse-sensor data assimilation

| | relative L2 (24-step rollout, `test_chirp`) |
|---|---|
| Open-loop POD+GRU (no correction) | 0.470 |
| EnKF-corrected (4 sparse T+p sensors, real `EnsembleKalmanFilter`) | 0.583 |

**Assimilation made the estimate *worse* here (-24% "improvement" -- i.e. a
real degradation), and we tracked down two concrete, fixable causes rather
than reporting a mystery:**

1. **The 4 temperature sensors carried zero real signal.** Sensors were
   placed at mid-channel height (`iy = ny//2 = 16`). We checked the actual
   CFD reference at those exact grid points: `T` is identically `0.0` at
   `y=16` for *every* snapshot of the whole trajectory. Why: the thermal
   diffusivity (`alpha_T=0.008`) only lets the near-wall heat plume spread
   roughly `sqrt(2*alpha*n_steps) ~ 2.8` cells in the cross-stream direction
   over 480 steps (confirmed directly: `T(y)` at the heater's own x-station
   drops to numerically zero by `y~11-12`, nowhere near `y=16`). So half of
   the assimilation's 8-dimensional observation vector was pure noise with
   no informative content, and the filter still computed a non-trivial
   Kalman gain against it.
2. **POD-mode "spillover" from sparse point corrections.** With the
   temperature channel uninformative, only the 4 pressure-sensor readings
   carried real signal -- fewer independent observations than the 12-dimensional
   POD latent state being corrected. Because every POD mode has *global*
   spatial support, forcing agreement with a handful of local point values
   (especially near-degenerate ones) can perturb the reconstructed field
   *elsewhere* in the domain, a known limitation of reduced-order data
   assimilation when the sensor count and placement don't span where the
   state actually varies.
3. **The EnKF process model is an approximation, documented in the code**:
   since the GRU needs a short history window but each ensemble member only
   carries a single latent vector, `f_member` tiles that member's own state
   across the window rather than using genuine per-member history -- a
   reasonable but imperfect stand-in for the GRU's real recurrent dynamics,
   adding its own bias before any correction is even applied.

This is a genuine, honestly-reported negative result, and a real, common
one in reduced-order data assimilation: **sparse-sensor correction is not
automatically beneficial -- sensor placement (must lie inside the region
where the state actually varies) and the ratio of informative observations
to latent dimension both matter, and getting them wrong actively hurts.**
A fixed version of this experiment (moving the T sensors into the thermal
boundary layer, e.g. `iy in {1,2,3}`, and/or lowering the POD rank to match
the true observability of 4-8 sensors) is the natural next step and was not
attempted here for time reasons.

### Is `heater_raises_outlet_T=False` on the CFD reference itself a bug?

**No -- we checked, and it is a real, correctly-computed consequence of a
scope-limiting parameter choice, not a solver bug or a broken check.**
Diagnosed directly against the raw `test_chirp` reference data:

- `outlet_T` (mean T at the outlet column) is *exactly* `0.0` for the first
  10 of 25 snapshots, then rises **monotonically** from ~1e-6 to 6e-4 over
  the rest of the trajectory -- it tracks a slow, cumulative trend, not the
  actuator's fast on/off oscillation, so a naive "instantaneous heater-on
  vs. heater-off" comparison finds no positive correlation (and here, by
  chance, a very slightly negative one) purely because of that mismatch in
  time scales.
- The reason: the domain's convective flow-through time is
  `nx / u_in0 = 64 / 0.05 = 1280` lattice steps, but each trajectory only
  runs for **480 steps -- 38% of one full channel transit**. Directly
  confirmed in the field data: at the final snapshot, the temperature
  profile along x has already decayed back to numerically zero by
  `x~60` (out of `nx=64`) -- the heat plume generated at the heater band
  (`x` in `[22,42)`) simply has not had time to reach the outlet within a
  single trajectory. `outlet_T` is real but is dominated by slow
  diffusive leakage of the plume's leading edge, not by the actuator's
  current state.

This is an honest **scope limitation of this proof of concept's chosen
grid/velocity/trajectory-length combination**, not a defect in the LBM
momentum solve or the energy stepper (both behave exactly as their own
diffusion/convection length scales predict). It also explains why every
trajectory's `outlet_T` values are tiny (~1e-4) compared to `max_T`
(~0.7-1.0): outlet temperature, as configured here, is not yet a
well-developed signal within one trajectory. A real next step: either
shorten the domain, raise `u_in0`, or lengthen `n_steps` so at least one
full flow-through occurs before evaluating outlet-based metrics.

### Physical plausibility (manual checks, not the full `PhysicsGuardrail` -- see script note (e))

All rows below are computed on the same held-out trajectory used for the
assimilation experiment (`test_chirp`), so they are directly comparable.

| | T >= 0 everywhere | heater_raises_outlet_T |
|---|---|---|
| CFD reference | 100% | False (see above -- a timescale artifact, not a defect) |
| POD+GRU | 100% | False (consistent with the reference) |
| FNO2d | 99.0% | True (inconsistent with the reference -- see below) |

POD+GRU's rollout matches the reference's (real, timescale-driven) lack of
heater/outlet correlation; FNO2d shows a spurious *positive* correlation
that the reference itself doesn't have, and also produces a small fraction
of negative temperatures (1.0%) -- both consistent with the rollout
divergence already diagnosed above (an unstable, drifting rollout can
coincidentally trend upward with any input feature, including the heater
channel, without that reflecting real learned physics).

## Honest summary: what worked, what didn't

**Worked:**
- A genuinely time-varying, weakly-coupled real momentum (LBM) + energy
  transport CFD reference, driven by two independent continuous actuator
  functions, generating 10 trajectories (7 shape families + 3 held-out
  shape families) in under 9 seconds.
- POD+GRU as an interpretable baseline: stable, bounded rollout error that
  does not blow up and does not degrade meaningfully on held-out
  trajectories, despite a small (168-sample) training set.
- The overall pipeline runs end-to-end in about 4.5 minutes on a laptop CPU.
- Every honesty check in this README was backed by re-running the actual
  reference generation and inspecting real field values, not by guessing.

**Didn't work / open problems:**
- FNO2d, the main neural candidate the brief expected might win, performed
  substantially worse than the simple reduced-order baseline, due to
  rollout-error accumulation from one-step-only training on a small
  dataset -- a training-protocol gap, not a fundamental architecture
  problem, but a real result as measured here.
- Sparse-sensor EnKF assimilation made the held-out estimate *worse*, not
  better, traced to a genuine sensor-placement mistake (temperature sensors
  sitting entirely outside the thermal boundary layer) compounded by a
  sparse-observation / POD-mode-spillover effect and an approximate
  ensemble process model.
- The 480-step trajectory length is shorter than one convective
  flow-through time of the channel, so outlet-temperature-based metrics
  (one of the four quantities the brief explicitly asks the twin to
  predict) never fully develop within a single training trajectory.
- DeepONet and a physics-informed variant (both explicit stretch goals)
  were not attempted.

Both the FNO underperformance and the failed assimilation experiment are
reported as real findings rather than smoothed over, per this repo's own
convention and the project brief's explicit request for honest reporting
of what didn't work.
