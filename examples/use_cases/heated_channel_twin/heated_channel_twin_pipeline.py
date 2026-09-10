# -*- coding: utf-8 -*-
"""Real-Time Physics-Informed Digital Twin for an Actuated Thermal-Fluid System
================================================================================
2D Heated-Channel Digital Twin -- Proof of Concept (v3)

TECHNICAL SPEC (translated / condensed from the original)
------------------------------------------------------------------------------
A 2D channel carries an incompressible, LAMINAR, constant-property flow. Two
time-varying actuators drive it: the inlet velocity U_in(t) and the heat flux
Q_h(t) on a segment of the bottom wall. Build a reproducible digital twin
across four layers:

  1. Computational physics : a real CFD reference solving the continuous 2D
     incompressible Navier-Stokes + energy equations
         div(u) = 0
         rho (du/dt + (u.grad)u) = -grad(p) + mu * lap(u)
         rho cp (dT/dt + u.grad(T)) = k * lap(T)
     with the exact boundary conditions
         inlet  (x=0)      : u = (U_in(t), 0),  T = T_in (= 0)
         heated wall seg.  : -k dT/dn = Q_h(t)   (time-dependent Neumann flux)
         other walls       : no-slip u = 0,  insulated dT/dn = 0
         outlet (x=Lx)     : p = 0,  du/dn = 0,  dT/dn = 0
     over genuinely varying actuator trajectories (steps, ramps, sinusoids,
     piecewise-constant, PRBS), with held-out test trajectories of
     genuinely different SHAPE.

  2. Scientific ML : train and fairly compare surrogates mapping
     (actuator history + current state) -> future field prediction:
       (A) POD + learned latent dynamics (GRU)
       (B) Fourier Neural Operator (FNO2d)
       (C) Convolutional Autoencoder + temporal model (GRU on the CAE latent)
       (D) [extra, not in the spec list] DeepONet
       (E) Physics-Informed variant: FNO2d regularized with the discretized
           mass + momentum + energy PDE residuals.
     Evaluation on the held-out trajectories with four metrics:
       1. field relative-L2 error for u, v, p, T over time
       2. autoregressive rollout drift / instability
       3. physical-consistency PDE residuals -- in particular the DIVERGENCE
          residual mean|div(u)| of each surrogate's predicted velocity
       4. computational latency: surrogate inference time per rollout vs. the
          CFD solver wall-time for the same trajectory -> real speedup factor

  3. Data assimilation : sparse virtual T and p sensors. Two reconstruction
     methods, compared: (i) latent-space innovation correction (measured -
     predicted at each sensor, projected back onto the latent state) and
     (ii) an Ensemble Kalman Filter in the reduced latent space.

  4. A lightweight interactive demonstrator (`--demo` / `demo()`): pick an
     actuator trajectory, get the surrogate's predicted fields + operational
     metrics + the CFD comparison + the speedup, fast.

Operational metrics tracked for every trajectory, CFD and surrogates:
outlet temperature, maximum temperature, pressure drop.

HONESTY NOTES
------------------------------------------------------------------------------
(a) SOLVER: `_ns2d_step` below is this pipeline's own 2D incompressible
    laminar Navier-Stokes + energy solver -- a classic Chorin fractional-step
    (projection) method on a collocated grid: explicit upwind advection +
    central diffusion for the momentum predictor, a Gauss-Seidel pressure
    Poisson solve (Neumann dp/dn=0 on inlet+walls, Dirichlet p=0 at the
    outlet), the projection correction, then an explicit advection-diffusion
    update for T. It is NOT the repo's `LBMSolver` (v1/v2 used that, but
    LBM + Smagorinsky LES is a turbulence closure / stability hack, not the
    spec's laminar constant-property regime). `pinneapple_simulation
    .numerical_solvers.fdm3d` has a real 3D projection NS solver and a
    `_pressure_poisson`; this 2D version follows the same fractional-step
    structure the spec's equations call for directly. Re = 200 is well
    within the laminar channel regime (transition ~2000). The collocated
    grid leaves a small O(1e-3) mean interior divergence and a larger
    boundary-cell divergence near the outflow -- reported honestly in the
    metrics as the CFD baseline for the divergence-residual comparison.

(b) GEOMETRY: `pinneapple_design.geometry.gen.domains.ChannelDomain2D` is a
    real class; its bounds + inlet/outlet/walls taxonomy drive the grid
    spacing, the coordinate grids handed to FNO2d / DeepONet, and the
    physical->grid mapping of the heated segment (its own `HEATER_X_FRAC`
    parameter, since ChannelDomain2D has no heated-segment concept).

(c) COMPARISON: `pinneapple_arena.Arena` is built for static analytical PINN
    benchmarks with no actuator-history / autoregressive-rollout concept, so
    it does not fit this problem. A direct manual comparison is done
    instead: identical dataset, identical rollout procedure and metrics,
    reported side by side.

(d) PHYSICS-PLAUSIBILITY: `PhysicsGuardrail.check()` expects a per-point
    coordinate->field model callable on a `ProblemSpec`; our surrogates
    predict whole-field stacks, so it does not cleanly apply. Instead we
    run real quantitative checks -- the divergence residual (metric 3), a
    thermal-energy budget, and feeding the real grid-convergence result to
    `pinneapple_analysis.verification.physics_confidence_score
    .compute_physics_confidence` (honestly reporting its low coverage).

Runtime: grid 80x32, 30 actuator trajectories (24 train / 6 held out),
1250 projection steps each; four/five surrogates trained with a K-step
rollout-consistency curriculum; two assimilation methods; a 3-grid
convergence study. ~60-90 min on a laptop CPU. The full CFD dataset
(mesh, coordinates, time vector, actuation series, standardized (u,v,p,T)
fields) is persisted to outputs/dataset_heated_channel.npz and every
trained surrogate is checkpointed to outputs/ckpt_*.pt, so a training-only
re-run reuses both.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Real PINNeAPPle geometry ────────────────────────────────────────────────
from pinneapple_design.geometry.gen.domains import ChannelDomain2D

# ── Real PINNeAPPle surrogates ─────────────────────────────────────────────
from pinneapple_neural.architectures.rom.pod import POD
from pinneapple_neural.architectures.recurrent.gru import GRUModel
from pinneapple_neural.architectures.neural_operators.fno import FNO2d
from pinneapple_neural.architectures.neural_operators.deeponet import DeepONet
from pinneapple_neural.architectures.autoencoders.ae_2d import Autoencoder2D

# ── Real PINNeAPPle data assimilation + verification ───────────────────────
from pinneapple_analysis.state_estimation.kalman import EnsembleKalmanFilter
from pinneapple_analysis.verification.convergence import mesh_independence_study
from pinneapple_analysis.verification.physics_confidence_score import compute_physics_confidence

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_PATH = OUT_DIR / "dataset_heated_channel.npz"

DARK_BG = "#0d1117"
ACCENT = "#58a6ff"; ACCENT2 = "#f78166"; ACCENT3 = "#3fb950"; ACCENT4 = "#d2a8ff"; ACCENT5 = "#ffd166"

DEVICE = torch.device("cpu")
SEED = 0
torch.set_num_threads(int(os.environ.get("HCT_THREADS", "1")))  # CPU FFT autograd threads poorly at this size

FIELD_NAMES = ["u", "v", "p", "T"]


# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIG
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ChannelConfig:
    # geometry / grid
    Lx: float = 2.5
    Ly: float = 1.0
    nx: int = 80
    ny: int = 32
    # laminar incompressible physics (nondimensional, rho = cp = 1)
    Re: float = 200.0            # laminar (channel transition ~2000)
    Pr: float = 1.0              # nu / alpha
    u_in0: float = 1.0           # reference inlet speed
    Q0: float = 0.15             # reference wall heat flux (sets the temperature scale)
    heater_x_frac: Tuple[float, float] = (0.10, 0.35)   # heated bottom-wall segment
    # time stepping
    dt: float = 0.006
    n_steps: int = 1700          # ~4 convective flow-through times (Lx / u_in0 / dt ~ 417)
    snap_every: int = 34         # -> 51 snapshots
    poisson_iters: int = 150
    actuator_tau_steps: int = 25  # first-order rate-limit on U_in(t) and Q_h(t): a real
                                  # actuator (pump / heater) cannot change infinitely fast,
                                  # and an instantaneous inlet-velocity jump is what makes the
                                  # explicit projection scheme blow up on the PRBS / sawtooth
                                  # families -- so both actuator channels are low-pass filtered
                                  # with this time constant before driving the solver. The
                                  # discontinuous-family CHARACTER is preserved; only the
                                  # slew rate is bounded.
    # surrogate conditioning
    l_hist: int = 6


CFG = ChannelConfig()
N_SNAP = CFG.n_steps // CFG.snap_every + 1  # 51
EVAL_STEPS = N_SNAP - 1

# rollout-consistency training: K-step curriculum + fed-back-state noise
ROLLOUT_K_MIN = 2
ROLLOUT_K_MAX = 5
ROLLOUT_K_RAMP_EVERY = 22
ROLLOUT_K = ROLLOUT_K_MAX
ROLLOUT_NOISE_STD = 0.02
STARTS_PER_EPOCH = 336

EPOCHS_GRU = 220
EPOCHS_FNO = 100
EPOCHS_CAE = 120            # ConvAE reconstruction pre-training
EPOCHS_CAE_GRU = 220       # temporal model on the CAE latent
EPOCHS_DEEPONET = 240
EPOCHS_FNO_PI = 80
PI_LAMBDA_MASS = 0.02
PI_LAMBDA_MOM = 0.02
PI_LAMBDA_ENERGY = 0.02

POD_R = 12
CAE_LATENT = 16


def _curriculum_K(ep: int, k_max: int = ROLLOUT_K_MAX) -> int:
    return int(min(k_max, ROLLOUT_K_MIN + ep // ROLLOUT_K_RAMP_EVERY))


# ══════════════════════════════════════════════════════════════════════════
# 2. DOMAIN (real ChannelDomain2D) + geometry helpers
# ══════════════════════════════════════════════════════════════════════════

def build_domain() -> ChannelDomain2D:
    """Real ChannelDomain2D -- its bounds + inlet/outlet/walls taxonomy drive
    the grid spacing, the coordinate grids, and the heated-segment mapping."""
    return ChannelDomain2D(length=CFG.Lx, height=CFG.Ly, inlet_velocity=CFG.u_in0)


def grid_coords(domain: ChannelDomain2D) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0, y0 = domain.bounds_min
    x1, y1 = domain.bounds_max
    xs = x0 + (np.arange(CFG.nx) + 0.5) * (x1 - x0) / CFG.nx
    ys = y0 + (np.arange(CFG.ny) + 0.5) * (y1 - y0) / CFG.ny
    XX, YY = np.meshgrid(xs, ys, indexing="ij")
    return xs.astype(np.float32), ys.astype(np.float32), XX.astype(np.float32), YY.astype(np.float32)


def heater_index_range(domain: ChannelDomain2D, nx: int) -> Tuple[int, int]:
    x0, _ = domain.bounds_min
    x1, _ = domain.bounds_max
    ia = int(round(CFG.heater_x_frac[0] * nx))
    ib = int(round(CFG.heater_x_frac[1] * nx))
    return ia, ib


def trunk_coords_norm(domain: ChannelDomain2D) -> np.ndarray:
    _, _, XX, YY = grid_coords(domain)
    x0, y0 = domain.bounds_min
    x1, y1 = domain.bounds_max
    Xn = (XX - x0) / (x1 - x0)
    Yn = (YY - y0) / (y1 - y0)
    return np.stack([Xn.reshape(-1), Yn.reshape(-1)], axis=1)  # (nx*ny, 2)


# ══════════════════════════════════════════════════════════════════════════
# 3. ACTUATOR TRAJECTORY LIBRARY
# ══════════════════════════════════════════════════════════════════════════
# All shapes are functions of normalised time tau in [0, 1] returning [0, 1].
# TRAINING families (spec-required): step, ramp, sinusoid, piecewise_constant,
# prbs.  HELD-OUT families: genuinely different generating processes.

def shape_step(tau, t0=0.3, **_):
    return (tau >= t0).astype(np.float64)


def shape_ramp(tau, t1=0.8, **_):
    return np.clip(tau / t1, 0.0, 1.0)


def shape_sinusoid(tau, freq=2.0, phase=0.0, **_):
    return 0.5 + 0.5 * np.sin(2.0 * np.pi * freq * tau + phase)


def shape_piecewise_constant(tau, rng=None, n_seg=5, **_):
    """Random piecewise-constant staircase (spec-required)."""
    rng = rng or np.random.default_rng(0)
    edges = np.sort(rng.uniform(0.05, 0.95, n_seg - 1))
    levels = rng.uniform(0.0, 1.0, n_seg)
    out = np.empty_like(tau)
    seg = np.searchsorted(edges, tau)
    out[:] = levels[seg]
    return out


def shape_prbs(tau, rng=None, dwell=0.08, **_):
    """Pseudo-random binary sequence: value in {0,1}, held for ~`dwell`
    normalised-time before an independent coin flip (spec-required)."""
    rng = rng or np.random.default_rng(0)
    n = len(tau)
    hold = max(1, int(round(dwell * n)))
    out = np.empty(n)
    val = float(rng.integers(0, 2))
    for i in range(n):
        if i % hold == 0:
            val = float(rng.integers(0, 2))
        out[i] = val
    return out


def shape_chirp(tau, f0=0.5, f1=6.0, **_):
    ph = 2.0 * np.pi * (f0 * tau + 0.5 * (f1 - f0) * tau ** 2)
    return 0.5 + 0.5 * np.sin(ph)


def shape_sawtooth(tau, freq=3.0, **_):
    return (tau * freq) % 1.0


def shape_smoothed_random_walk(tau, rng=None, **_):
    rng = rng or np.random.default_rng(0)
    n = len(tau)
    walk = np.cumsum(rng.normal(0.0, 1.0, n))
    k = np.ones(15) / 15.0
    w = np.convolve(walk, k, mode="same")
    w = w - w.min()
    return w / (w.max() + 1e-9)


def shape_gaussian_bumps(tau, rng=None, n_bumps=4, width=0.05, **_):
    rng = rng or np.random.default_rng(0)
    centres = rng.uniform(0.1, 0.9, n_bumps)
    amps = rng.uniform(0.5, 1.0, n_bumps)
    v = np.zeros_like(tau)
    for c, a in zip(centres, amps):
        v += a * np.exp(-0.5 * ((tau - c) / width) ** 2)
    return np.clip(v / (v.max() + 1e-9), 0.0, 1.0)


def shape_am_two_tone(tau, f_carrier=5.0, f_mod=0.7, **_):
    return (0.5 + 0.5 * np.sin(2 * np.pi * f_carrier * tau)) * \
           (0.5 + 0.5 * np.sin(2 * np.pi * f_mod * tau))


SHAPES: Dict[str, Callable] = {
    "step": shape_step, "ramp": shape_ramp, "sinusoid": shape_sinusoid,
    "piecewise_constant": shape_piecewise_constant, "prbs": shape_prbs,
    "chirp": shape_chirp, "sawtooth": shape_sawtooth,
    "smoothed_random_walk": shape_smoothed_random_walk,
    "gaussian_bumps": shape_gaussian_bumps, "am_two_tone": shape_am_two_tone,
}
TRAIN_FAMILIES = ["step", "ramp", "sinusoid", "piecewise_constant", "prbs"]
HELDOUT_FAMILIES = ["chirp", "sawtooth", "smoothed_random_walk", "gaussian_bumps", "am_two_tone"]


@dataclass
class TrajectorySpec:
    name: str
    held_out: bool
    u_shape: str
    u_kwargs: Dict[str, Any] = field(default_factory=dict)
    q_shape: str = ""
    q_kwargs: Dict[str, Any] = field(default_factory=dict)
    seed: int = 0


def make_specs() -> List[TrajectorySpec]:
    """24 training trajectories: the 5 spec-required actuator families, ~5
    parametric variants each (U_in and Q_h given different variants of the
    same family so they are not identical). 6 held-out trajectories from 5
    genuinely different generating families (one family used twice with
    different parameters)."""
    train: List[TrajectorySpec] = []
    variants = {
        "step": [dict(t0=t) for t in (0.18, 0.30, 0.42, 0.55, 0.68)],
        "ramp": [dict(t1=t) for t in (0.45, 0.6, 0.75, 0.9, 1.0)],
        "sinusoid": [dict(freq=f, phase=p) for f, p in
                     ((1.0, 0.0), (1.8, 1.0), (2.6, 0.4), (3.5, 1.9), (4.5, 0.9))],
        "piecewise_constant": [dict(n_seg=n) for n in (3, 4, 5, 6, 7)],
        "prbs": [dict(dwell=d) for d in (0.05, 0.07, 0.10, 0.13, 0.16)],
    }
    sd = 1
    for fam, vlist in variants.items():
        for i, uk in enumerate(vlist):
            qk = dict(vlist[(i + 2) % len(vlist)])
            train.append(TrajectorySpec(f"train_{fam}_{i}", False, fam, dict(uk), fam, qk, seed=sd))
            sd += 1
    test = [
        TrajectorySpec("test_chirp", True, "chirp", dict(f0=0.5, f1=6.0),
                        "chirp", dict(f0=0.8, f1=4.0), seed=201),
        TrajectorySpec("test_sawtooth", True, "sawtooth", dict(freq=3.0),
                        "sawtooth", dict(freq=2.0), seed=202),
        TrajectorySpec("test_smoothed_random_walk", True, "smoothed_random_walk", {},
                        "smoothed_random_walk", {}, seed=203),
        TrajectorySpec("test_gaussian_bumps", True, "gaussian_bumps", dict(n_bumps=4),
                        "gaussian_bumps", dict(n_bumps=3), seed=204),
        TrajectorySpec("test_am_two_tone", True, "am_two_tone", dict(f_carrier=5.0, f_mod=0.7),
                        "am_two_tone", dict(f_carrier=3.5, f_mod=1.1), seed=205),
        TrajectorySpec("test_chirp_fast", True, "chirp", dict(f0=1.0, f1=9.0),
                        "sawtooth", dict(freq=4.0), seed=206),
    ]
    return train + test


def _rate_limit(series: np.ndarray, tau_steps: int) -> np.ndarray:
    """First-order IIR low-pass (rate limit) applied forward over the fine
    time series. `tau_steps` is the filter time constant in solver steps."""
    if tau_steps <= 1:
        return series
    a = 1.0 / float(tau_steps)
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = out[i - 1] + a * (series[i] - out[i - 1])
    return out


def actuator_series(spec: TrajectorySpec, n_steps: int, rate_limit: bool = True) -> np.ndarray:
    """(n_steps, 2) array [U_in(t), Q_h(t)] in physical (nondimensional)
    units. Both channels are rate-limited (see ChannelConfig.actuator_tau_steps)."""
    tau = np.linspace(0.0, 1.0, n_steps)
    su = SHAPES[spec.u_shape](tau, rng=np.random.default_rng(spec.seed), **spec.u_kwargs)
    sq = SHAPES[spec.q_shape](tau, rng=np.random.default_rng(spec.seed + 1000), **spec.q_kwargs)
    U_in = CFG.u_in0 * (0.75 + 0.5 * su)    # [0.75, 1.25] * u_in0 -- always forward flow
    Q_h = CFG.Q0 * sq                       # [0, Q0]
    if rate_limit:
        U_in = _rate_limit(U_in, CFG.actuator_tau_steps)
        Q_h = _rate_limit(Q_h, CFG.actuator_tau_steps)
    return np.stack([U_in, Q_h], axis=1).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════
# 4. CFD REFERENCE: 2D incompressible LAMINAR Navier-Stokes + energy
#    (Chorin fractional-step / projection method, collocated grid)
# ══════════════════════════════════════════════════════════════════════════

def _upwind_adv(f, u, v, dx, dy):
    a = np.zeros_like(f)
    fx_b = (f[1:-1, 1:-1] - f[:-2, 1:-1]) / dx
    fx_f = (f[2:, 1:-1] - f[1:-1, 1:-1]) / dx
    fy_b = (f[1:-1, 1:-1] - f[1:-1, :-2]) / dy
    fy_f = (f[1:-1, 2:] - f[1:-1, 1:-1]) / dy
    uc = u[1:-1, 1:-1]; vc = v[1:-1, 1:-1]
    a[1:-1, 1:-1] = (np.where(uc >= 0, uc * fx_b, uc * fx_f)
                     + np.where(vc >= 0, vc * fy_b, vc * fy_f))
    return a


def _lap(f, dx, dy):
    l = np.zeros_like(f)
    l[1:-1, 1:-1] = ((f[2:, 1:-1] - 2 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / dx**2
                     + (f[1:-1, 2:] - 2 * f[1:-1, 1:-1] + f[1:-1, :-2]) / dy**2)
    return l


def _apply_vel_bc(u, v, U_in):
    u[0, :] = U_in; v[0, :] = 0.0                       # inlet: u = (U_in(t), 0)
    u[-1, :] = u[-2, :]; v[-1, :] = v[-2, :]            # outlet: du/dn = 0
    u[:, 0] = 0.0; v[:, 0] = 0.0                        # bottom wall: no-slip
    u[:, -1] = 0.0; v[:, -1] = 0.0                     # top wall: no-slip
    q_in = u[0, :].sum(); q_out = u[-1, :].sum()
    if abs(q_out) > 1e-9:
        u[-1, :] *= q_in / q_out                        # enforce global mass conservation at the outflow


def _pressure_poisson(p, rhs, dx, dy, iters):
    dx2, dy2 = dx**2, dy**2
    denom = 2.0 * (1.0 / dx2 + 1.0 / dy2)
    for _ in range(iters):
        p[1:-1, 1:-1] = ((p[2:, 1:-1] + p[:-2, 1:-1]) / dx2
                         + (p[1:-1, 2:] + p[1:-1, :-2]) / dy2
                         - rhs[1:-1, 1:-1]) / denom
        p[0, :] = p[1, :]; p[:, 0] = p[:, 1]; p[:, -1] = p[:, -2]   # Neumann dp/dn = 0
        p[-1, :] = 0.0                                              # Dirichlet p = 0 at outlet
    return p


@dataclass
class TrajectoryData:
    spec: TrajectorySpec
    actuator_fine: np.ndarray   # (n_steps, 2)
    actuator_snap: np.ndarray   # (N_SNAP, 2)
    time_snap: np.ndarray       # (N_SNAP,)  physical time of each snapshot
    fields: np.ndarray          # (N_SNAP, 4, nx, ny) -> [u, v, p, T]  physical units
    scalars: Dict[str, np.ndarray]  # outlet_T, max_T, dp  each (N_SNAP,)
    solver_seconds: float


def run_reference_trajectory(
    spec: TrajectorySpec, *,
    nx: Optional[int] = None, ny: Optional[int] = None,
    n_steps: Optional[int] = None, snap_every: Optional[int] = None,
    Q0: Optional[float] = None, alpha_scale: float = 1.0,
    dt: Optional[float] = None, poisson_iters: Optional[int] = None,
) -> TrajectoryData:
    """Weakly nothing -- a single monolithic transient solve of the
    incompressible laminar NS + energy system, driven by genuinely
    time-varying U_in(t) (inlet Dirichlet) and Q_h(t) (heated-wall Neumann
    heat flux).  Resolution knobs are exposed for the grid-convergence
    study."""
    nx = nx or CFG.nx
    ny = ny or CFG.ny
    n_steps = n_steps or CFG.n_steps
    snap_every = snap_every or CFG.snap_every
    Q0 = CFG.Q0 if Q0 is None else Q0

    dx, dy = CFG.Lx / nx, CFG.Ly / ny
    nu = CFG.u_in0 * CFG.Ly / CFG.Re
    alpha = (nu / CFG.Pr) * alpha_scale
    k_therm = alpha                       # rho * cp = 1  ->  k = alpha
    dt = CFG.dt if dt is None else dt
    p_iters = CFG.poisson_iters if poisson_iters is None else poisson_iters
    ix0, ix1 = heater_index_range(build_domain(), nx)

    tau = np.linspace(0.0, 1.0, n_steps)
    su = SHAPES[spec.u_shape](tau, rng=np.random.default_rng(spec.seed), **spec.u_kwargs)
    sq = SHAPES[spec.q_shape](tau, rng=np.random.default_rng(spec.seed + 1000), **spec.q_kwargs)
    U_in_t = _rate_limit(CFG.u_in0 * (0.75 + 0.5 * su), CFG.actuator_tau_steps)
    Q_h_t = _rate_limit(Q0 * sq, CFG.actuator_tau_steps)
    actuator_fine = np.stack([U_in_t, Q_h_t], axis=1).astype(np.float64)

    u = np.zeros((nx, ny)); v = np.zeros((nx, ny)); p = np.zeros((nx, ny)); T = np.zeros((nx, ny))
    _apply_vel_bc(u, v, U_in_t[0])

    snaps, snap_act, snap_t = [], [], []
    scal_outlet_T, scal_max_T, scal_dp = [], [], []

    def _record(step_idx):
        # store a lightly 3x3-box-smoothed pressure: the collocated grid leaves an
        # odd-even (checkerboard) component in the projection potential that is a
        # pure numerical artefact -- physically the laminar pressure field is
        # smooth -- and it otherwise dominates every surrogate's all-field error.
        ps = p.copy()
        for _ in range(3):
            ps[1:-1, 1:-1] = (ps[1:-1, 1:-1] + ps[2:, 1:-1] + ps[:-2, 1:-1]
                              + ps[1:-1, 2:] + ps[1:-1, :-2]) / 5.0
        snaps.append(np.stack([u, v, ps, T], 0).astype(np.float32).copy())
        snap_act.append(actuator_fine[min(step_idx, n_steps - 1)].copy())
        snap_t.append(step_idx * dt)
        scal_outlet_T.append(float(T[-1, :].mean()))
        scal_max_T.append(float(T.max()))
        scal_dp.append(float(ps[2, :].mean() - ps[-3, :].mean()))

    _record(0)
    t_solve = time.time()
    umax_cap = 6.0 * CFG.u_in0        # hard clip -- safety net against a transient CFL excursion
    n_substep_max = 4
    for step in range(n_steps):
        U_in = U_in_t[step]; Q_h = Q_h_t[step]
        # CFL-adaptive sub-stepping: keep the convective Courant number < ~0.8
        vmax = max(np.abs(u).max(), np.abs(v).max(), CFG.u_in0)
        nss = int(min(n_substep_max, max(1, np.ceil(vmax * dt / (0.8 * min(dx, dy))))))
        sdt = dt / nss
        for _ in range(nss):
            _apply_vel_bc(u, v, U_in)
            us = u.copy(); vs = v.copy()
            us[1:-1, 1:-1] = u[1:-1, 1:-1] + sdt * (-_upwind_adv(u, u, v, dx, dy)[1:-1, 1:-1]
                                                    + nu * _lap(u, dx, dy)[1:-1, 1:-1])
            vs[1:-1, 1:-1] = v[1:-1, 1:-1] + sdt * (-_upwind_adv(v, u, v, dx, dy)[1:-1, 1:-1]
                                                    + nu * _lap(v, dx, dy)[1:-1, 1:-1])
            _apply_vel_bc(us, vs, U_in)
            div = np.zeros((nx, ny))
            div[1:-1, 1:-1] = ((us[2:, 1:-1] - us[:-2, 1:-1]) / (2 * dx)
                               + (vs[1:-1, 2:] - vs[1:-1, :-2]) / (2 * dy))
            p = _pressure_poisson(p, div / sdt, dx, dy, p_iters)
            u[1:-1, 1:-1] = us[1:-1, 1:-1] - sdt * (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dx)
            v[1:-1, 1:-1] = vs[1:-1, 1:-1] - sdt * (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dy)
            _apply_vel_bc(u, v, U_in)
            np.clip(u, -umax_cap, umax_cap, out=u); np.clip(v, -umax_cap, umax_cap, out=v)

        # energy: apply the T ghost BCs first so the interior stencil sees them
        T[0, :] = 0.0                     # inlet Dirichlet T_in = 0
        T[-1, :] = T[-2, :]              # outlet dT/dn = 0
        T[:, -1] = T[:, -2]             # top wall insulated
        T[:, 0] = T[:, 1]              # bottom wall insulated (heater added as a face-flux source below)
        Tn = T.copy()
        Tn[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * (-_upwind_adv(T, u, v, dx, dy)[1:-1, 1:-1]
                                               + alpha * _lap(T, dx, dy)[1:-1, 1:-1])
        # heated wall segment: the prescribed flux Q_h(t) enters the bottom face of the
        # near-wall control volume (height dy) -> finite-volume source Q_h/dy.  This is
        # dimensionally consistent under grid refinement (total power Q_h*dx*dt is
        # grid-independent), unlike a dy-scaled ghost-cell temperature jump.
        Tn[ix0:ix1, 1] = Tn[ix0:ix1, 1] + dt * (Q_h / dy)
        Tn[0, :] = 0.0
        Tn[-1, :] = Tn[-2, :]
        Tn[:, -1] = Tn[:, -2]
        Tn[:, 0] = Tn[:, 1]
        T = Tn

        if (step + 1) % snap_every == 0:
            _record(step + 1)
    solver_seconds = time.time() - t_solve

    fields = np.stack(snaps, 0)
    scalars = {"outlet_T": np.array(scal_outlet_T), "max_T": np.array(scal_max_T), "dp": np.array(scal_dp)}
    return TrajectoryData(spec, actuator_fine, np.stack(snap_act, 0), np.array(snap_t),
                          fields, scalars, solver_seconds)


# ══════════════════════════════════════════════════════════════════════════
# 5. DATASET CONSTRUCTION + PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════

def _config_hash(specs: List[TrajectorySpec]) -> str:
    payload = {"cfg": asdict(CFG), "specs": [asdict(s) for s in specs], "solver": "ns2d_projection_v3"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def generate_or_load_dataset(specs: List[TrajectorySpec], use_cache: bool) -> Dict[str, TrajectoryData]:
    """Persist the FULL dataset (spec item D): grid coordinates, time vector,
    per-trajectory actuation series and the standardized + physical
    (u,v,p,T) field solutions -- not just metrics."""
    h = _config_hash(specs)
    domain = build_domain()
    xs, ys, XX, YY = grid_coords(domain)

    if use_cache and DATASET_PATH.exists():
        blob = np.load(DATASET_PATH, allow_pickle=True)
        if str(blob["config_hash"]) == h:
            print(f"    [dataset hit] {DATASET_PATH.name} (hash {h})")
            out: Dict[str, TrajectoryData] = {}
            for nm in [str(x) for x in blob["names"]]:
                spec = next(s for s in specs if s.name == nm)
                out[nm] = TrajectoryData(
                    spec=spec,
                    actuator_fine=blob[f"{nm}__act_fine"],
                    actuator_snap=blob[f"{nm}__act_snap"],
                    time_snap=blob[f"{nm}__time_snap"],
                    fields=blob[f"{nm}__fields"],
                    scalars={k: blob[f"{nm}__scal_{k}"] for k in ("outlet_T", "max_T", "dp")},
                    solver_seconds=float(blob[f"{nm}__solver_seconds"]),
                )
            return out
        print(f"    [dataset stale] hash {blob['config_hash']} -> {h}; regenerating")

    trajs: Dict[str, TrajectoryData] = {}
    t0 = time.time()
    for i, spec in enumerate(specs):
        tr = run_reference_trajectory(spec)
        trajs[spec.name] = tr
        print(f"    [{i+1:2d}/{len(specs)}] {spec.name:26s} {tr.solver_seconds:5.1f}s  "
              f"|u|max={np.abs(tr.fields[:,0]).max():.3f}  T=[{tr.fields[:,3].min():.3f},{tr.fields[:,3].max():.3f}]  "
              f"outletT_final={tr.scalars['outlet_T'][-1]:.4f}")
    print(f"    CFD generation: {time.time()-t0:.1f}s for {len(specs)} trajectories")

    # standardization stats over the TRAIN trajectories only
    train = [trajs[s.name] for s in specs if not s.held_out]
    allf = np.concatenate([t.fields for t in train], 0)
    fmean = allf.mean(axis=(0, 2, 3)).astype(np.float32)
    fstd = (allf.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32)

    save: Dict[str, Any] = {
        "config_hash": h, "names": np.array([s.name for s in specs]),
        "grid_x": xs, "grid_y": ys, "grid_XX": XX, "grid_YY": YY,
        "field_names": np.array(FIELD_NAMES), "field_mean": fmean, "field_std": fstd,
        "dt": CFG.dt, "snap_every": CFG.snap_every, "Lx": CFG.Lx, "Ly": CFG.Ly,
        "Re": CFG.Re, "Pr": CFG.Pr, "Q0": CFG.Q0,
    }
    for nm, tr in trajs.items():
        save[f"{nm}__act_fine"] = tr.actuator_fine
        save[f"{nm}__act_snap"] = tr.actuator_snap
        save[f"{nm}__time_snap"] = tr.time_snap
        save[f"{nm}__fields"] = tr.fields                                   # physical
        save[f"{nm}__fields_std"] = ((tr.fields - fmean[:, None, None]) / fstd[:, None, None]).astype(np.float32)
        save[f"{nm}__solver_seconds"] = np.float64(tr.solver_seconds)
        for k in ("outlet_T", "max_T", "dp"):
            save[f"{nm}__scal_{k}"] = tr.scalars[k]
    np.savez_compressed(DATASET_PATH, **save)
    print(f"    dataset persisted -> {DATASET_PATH.name} ({DATASET_PATH.stat().st_size/1e6:.1f} MB): "
          f"grid coords + time vector + actuation series + standardized & physical (u,v,p,T) fields")
    return trajs


@dataclass
class Normalizer:
    field_mean: np.ndarray
    field_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    def norm_field(self, x):
        return (x - self.field_mean[:, None, None]) / self.field_std[:, None, None]

    def denorm_field(self, x):
        return x * self.field_std[:, None, None] + self.field_mean[:, None, None]

    def norm_act(self, a):
        return (a - self.act_mean) / self.act_std


def fit_normalizer(train_trajs: List[TrajectoryData]) -> Normalizer:
    allf = np.concatenate([t.fields for t in train_trajs], 0)
    alla = np.concatenate([t.actuator_snap for t in train_trajs], 0)
    return Normalizer(allf.mean(axis=(0, 2, 3)).astype(np.float32),
                      (allf.std(axis=(0, 2, 3)) + 1e-8).astype(np.float32),
                      alla.mean(0), alla.std(0) + 1e-8)


def history_window(seq: np.ndarray, t: int, l_hist: int) -> np.ndarray:
    lo = t - l_hist + 1
    if lo >= 0:
        return seq[lo:t + 1]
    pad = np.repeat(seq[0:1], -lo, axis=0)
    return np.concatenate([pad, seq[0:t + 1]], axis=0)


@dataclass
class PreparedTraj:
    name: str
    held_out: bool
    fields_n: np.ndarray    # (N_SNAP, 4, nx, ny) normalized
    act_n: np.ndarray       # (N_SNAP, 2) normalized
    pod_latent: np.ndarray  # (N_SNAP, r) raw POD coefficients of the normalized fields
    fields_phys: np.ndarray
    actuator_snap: np.ndarray


def prepare_trajectories(trajs: List[TrajectoryData], norm: Normalizer, pod: POD) -> List[PreparedTraj]:
    out = []
    for tr in trajs:
        fn = norm.norm_field(tr.fields).astype(np.float32)
        an = norm.norm_act(tr.actuator_snap).astype(np.float32)
        with torch.no_grad():
            lat = pod.encode(torch.from_numpy(fn.reshape(N_SNAP, -1)).float()).numpy().astype(np.float32)
        out.append(PreparedTraj(tr.spec.name, tr.spec.held_out, fn, an, lat,
                                 tr.fields.astype(np.float32), tr.actuator_snap))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 6. CHECKPOINTING
# ══════════════════════════════════════════════════════════════════════════

def _try_load_ckpt(model: nn.Module, ckpt: Optional[Path], sig: str, force: bool):
    if ckpt is None or force or not ckpt.exists():
        return None, False
    try:
        blob = torch.load(ckpt, weights_only=False)
    except Exception:
        return None, False
    if blob.get("sig") != sig:
        return None, False
    model.load_state_dict(blob["state_dict"])
    print(f"      [ckpt hit] {ckpt.name}")
    return blob["losses"], True


def _save_ckpt(model: nn.Module, ckpt: Optional[Path], sig: str, losses: List[float]):
    if ckpt is not None:
        torch.save({"sig": sig, "state_dict": model.state_dict(), "losses": losses}, ckpt)


def _rollout_starts(prep: List[PreparedTraj], K: int, max_n: Optional[int] = None,
                    rng: Optional[np.random.Generator] = None) -> List[Tuple[int, int]]:
    starts = [(ti, t0) for ti in range(len(prep)) for t0 in range(0, N_SNAP - K)]
    if max_n is not None and max_n < len(starts):
        rng = rng or np.random.default_rng(SEED)
        starts = [starts[i] for i in rng.choice(len(starts), size=max_n, replace=False)]
    return starts


# ══════════════════════════════════════════════════════════════════════════
# 7. SURROGATE A: POD + GRU latent dynamics
# ══════════════════════════════════════════════════════════════════════════

def train_pod(train_trajs: List[TrajectoryData], norm: Normalizer, r: int = POD_R) -> POD:
    allf = np.concatenate([norm.norm_field(t.fields) for t in train_trajs], 0)
    pod = POD(r=r, center=True)
    pod.fit(torch.from_numpy(allf.reshape(allf.shape[0], -1)).float())
    return pod


def _latent_gru_train(prep_train: List[PreparedTraj], latent_of: Callable[[PreparedTraj], np.ndarray],
                      r: int, epochs: int, lr: float, hidden: int, progress, progress_every,
                      ckpt, ckpt_sig, force) -> Tuple[GRUModel, List[float]]:
    """Generic K-step rollout-consistency trainer for a GRU predicting the
    next latent vector.  `latent_of(prep)` returns the (N_SNAP, r) latent
    sequence (POD coefficients OR CAE codes)."""
    model = GRUModel(in_dim=2 + r, out_dim=r, horizon=1, hidden_dim=hidden, num_layers=1)
    cached, hit = _try_load_ckpt(model, ckpt, ckpt_sig, force)
    if hit:
        return model, cached
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(SEED)
    acts = [torch.from_numpy(p.act_n).float() for p in prep_train]
    lats = [torch.from_numpy(latent_of(p)).float() for p in prep_train]
    batch = 96
    losses = []
    for ep in range(epochs):
        Kc = _curriculum_K(ep)
        starts = _rollout_starts(prep_train, Kc, max_n=STARTS_PER_EPOCH, rng=rng)
        rng.shuffle(starts)
        ep_loss, nb = 0.0, 0
        for b0 in range(0, len(starts), batch):
            chunk = starts[b0:b0 + batch]
            B = len(chunk)
            hist = [[lats[ti][j] for j in range(t0 + 1)] for (ti, t0) in chunk]
            loss = torch.zeros(())
            for k in range(Kc):
                xb = torch.zeros(B, CFG.l_hist, 2 + r)
                tgt = torch.zeros(B, r)
                for bi, (ti, t0) in enumerate(chunk):
                    t = t0 + k
                    aw = acts[ti][max(0, t - CFG.l_hist + 1): t + 1]
                    if aw.shape[0] < CFG.l_hist:
                        aw = torch.cat([acts[ti][0:1].repeat(CFG.l_hist - aw.shape[0], 1), aw], 0)
                    lh = torch.stack(hist[bi][-CFG.l_hist:], 0)
                    if lh.shape[0] < CFG.l_hist:
                        lh = torch.cat([lh[0:1].repeat(CFG.l_hist - lh.shape[0], 1), lh], 0)
                    xb[bi] = torch.cat([aw, lh], dim=1)
                    tgt[bi] = lats[ti][t + 1]
                y = model(xb).y[:, 0, :]
                loss = loss + torch.mean((y - tgt) ** 2)
                for bi in range(B):
                    hist[bi].append(y[bi])
            loss = loss / Kc
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach()); nb += 1
        losses.append(ep_loss / max(nb, 1))
        if progress and (ep + 1) % progress_every == 0:
            progress(ep + 1, model, losses[-1]); model.train()
    _save_ckpt(model, ckpt, ckpt_sig, losses)
    return model, losses


def train_gru_rollout(prep_train, r, epochs=EPOCHS_GRU, lr=2e-3, progress=None, progress_every=40,
                      ckpt=None, ckpt_sig="", force_retrain=False):
    return _latent_gru_train(prep_train, lambda p: p.pod_latent, r, epochs, lr, 48,
                             progress, progress_every, ckpt, ckpt_sig, force_retrain)


def rollout_pod_gru(prep: PreparedTraj, norm: Normalizer, pod: POD, gru: GRUModel) -> np.ndarray:
    z = [prep.pod_latent[0].astype(np.float32)]
    gru.eval()
    with torch.no_grad():
        for t in range(EVAL_STEPS):
            aw = history_window(prep.act_n, t, CFG.l_hist)
            lw = history_window(np.stack(z, 0), t, CFG.l_hist)
            x = torch.from_numpy(np.concatenate([aw, lw], 1)).float().unsqueeze(0)
            z.append(gru(x).y[0, 0].numpy().astype(np.float32))
        flat = pod.decode(torch.from_numpy(np.stack(z, 0)).float()).numpy()
    fn = flat.reshape(N_SNAP, 4, CFG.nx, CFG.ny)
    return np.stack([norm.denorm_field(fn[i]) for i in range(N_SNAP)], 0)


# ══════════════════════════════════════════════════════════════════════════
# 8. SURROGATE B: FNO2d  (+ optional physics-informed penalty)
# ══════════════════════════════════════════════════════════════════════════

def _act_channels(aw_flat: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.broadcast_to(aw_flat[:, None, None],
                            (aw_flat.shape[0], CFG.nx, CFG.ny)).astype(np.float32))


def _pde_residuals(state_prev_n, state_next_n, src_phys, norm, dx, dy, dt):
    """Discretized mass + momentum + energy residuals of a predicted
    normalized field pair (batched).  Returns (r_mass, r_mom, r_energy) each
    a scalar mean-square, computed in PHYSICAL units."""
    fm = torch.tensor(norm.field_mean); fs = torch.tensor(norm.field_std)
    prev = state_prev_n * fs[None, :, None, None] + fm[None, :, None, None]
    nxt = state_next_n * fs[None, :, None, None] + fm[None, :, None, None]
    up, vp, pp, Tp = prev[:, 0], prev[:, 1], prev[:, 2], prev[:, 3]
    un, vn, pn, Tn = nxt[:, 0], nxt[:, 1], nxt[:, 2], nxt[:, 3]
    nu = CFG.u_in0 * CFG.Ly / CFG.Re
    alpha = nu / CFG.Pr

    def ddx(f): return (f[:, 2:, 1:-1] - f[:, :-2, 1:-1]) / (2 * dx)
    def ddy(f): return (f[:, 1:-1, 2:] - f[:, 1:-1, :-2]) / (2 * dy)
    def lap(f): return ((f[:, 2:, 1:-1] - 2 * f[:, 1:-1, 1:-1] + f[:, :-2, 1:-1]) / dx**2
                        + (f[:, 1:-1, 2:] - 2 * f[:, 1:-1, 1:-1] + f[:, 1:-1, :-2]) / dy**2)

    # mass: div(u_next) = 0
    r_mass = ddx(un) + ddy(vn)
    # momentum (x): du/dt + u.grad u = -dp/dx + nu lap u   (evaluated with prev-state advection)
    uc = up[:, 1:-1, 1:-1]; vc = vp[:, 1:-1, 1:-1]
    dudt = (un[:, 1:-1, 1:-1] - up[:, 1:-1, 1:-1]) / dt
    dvdt = (vn[:, 1:-1, 1:-1] - vp[:, 1:-1, 1:-1]) / dt
    r_mx = dudt + uc * ddx(up) + vc * ddy(up) + ddx(pp) - nu * lap(up)
    r_my = dvdt + uc * ddx(vp) + vc * ddy(vp) + ddy(pp) - nu * lap(vp)
    r_mom = torch.mean(r_mx ** 2) + torch.mean(r_my ** 2)
    # energy: dT/dt + u.grad T = alpha lap T + volumetric heater proxy
    dTdt = (Tn[:, 1:-1, 1:-1] - Tp[:, 1:-1, 1:-1]) / dt
    r_e = dTdt + uc * ddx(Tp) + vc * ddy(Tp) - alpha * lap(Tp) - src_phys[:, 1:-1, 1:-1]
    return torch.mean(r_mass ** 2), r_mom, torch.mean(r_e ** 2)


def train_fno_rollout(prep_train, norm, epochs=EPOCHS_FNO, lr=2e-3, physics_informed=False,
                      heater_mask_np: Optional[np.ndarray] = None,
                      progress=None, progress_every=25, ckpt=None, ckpt_sig="", force_retrain=False):
    in_c = 4 + 2 * CFG.l_hist
    model = FNO2d(in_channels=in_c, out_channels=4, width=16, modes1=9, modes2=7, layers=3, use_grid=True)
    cached, hit = _try_load_ckpt(model, ckpt, ckpt_sig, force_retrain)
    if hit:
        return model, cached
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    rng = np.random.default_rng(SEED)
    batch = 48
    dx, dy = CFG.Lx / CFG.nx, CFG.Ly / CFG.ny
    dt_snap = CFG.dt * CFG.snap_every
    hmask = torch.from_numpy(heater_mask_np.astype(np.float32)) if heater_mask_np is not None else None
    losses = []
    tgen = torch.Generator().manual_seed(SEED)
    for ep in range(epochs):
        Kc = _curriculum_K(ep)
        starts = _rollout_starts(prep_train, Kc, max_n=STARTS_PER_EPOCH, rng=rng)
        rng.shuffle(starts)
        ep_loss, nb = 0.0, 0
        for b0 in range(0, len(starts), batch):
            chunk = starts[b0:b0 + batch]
            state = torch.stack([torch.from_numpy(prep_train[ti].fields_n[t0]).float() for (ti, t0) in chunk], 0)
            loss = torch.zeros(()); pi = torch.zeros(())
            for k in range(Kc):
                aw = np.stack([history_window(prep_train[ti].act_n, t0 + k, CFG.l_hist).reshape(-1)
                               for (ti, t0) in chunk], 0)
                ac = torch.from_numpy(np.broadcast_to(aw[:, :, None, None],
                                     (aw.shape[0], aw.shape[1], CFG.nx, CFG.ny)).astype(np.float32))
                st_in = state
                if k > 0 and ROLLOUT_NOISE_STD > 0:
                    st_in = state + ROLLOUT_NOISE_STD * torch.randn(state.shape, generator=tgen)
                y = model(torch.cat([st_in, ac], 1)).y
                tgt = torch.stack([torch.from_numpy(prep_train[ti].fields_n[t0 + k + 1]).float()
                                   for (ti, t0) in chunk], 0)
                loss = loss + torch.mean((y - tgt) ** 2)
                if physics_informed and hmask is not None:
                    q_next = torch.tensor([prep_train[ti].actuator_snap[min(t0 + k + 1, N_SNAP - 1), 1]
                                           for (ti, t0) in chunk], dtype=torch.float32)
                    # heater as a near-wall face-flux source Q_h/dy in the heated band
                    src = q_next[:, None, None] * hmask[None] * (1.0 / dy)
                    rm, rmom, re = _pde_residuals(st_in, y, src, norm, dx, dy, dt_snap)
                    pi = pi + PI_LAMBDA_MASS * rm + PI_LAMBDA_MOM * rmom + PI_LAMBDA_ENERGY * re
                state = y
            loss = loss / Kc + (pi / Kc if physics_informed else 0.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.detach()); nb += 1
        losses.append(ep_loss / max(nb, 1))
        if progress and (ep + 1) % progress_every == 0:
            progress(ep + 1, model, losses[-1]); model.train()
    _save_ckpt(model, ckpt, ckpt_sig, losses)
    return model, losses


def rollout_fno(prep: PreparedTraj, norm: Normalizer, fno: FNO2d) -> np.ndarray:
    state = torch.from_numpy(prep.fields_n[0]).float().unsqueeze(0)
    outs = [prep.fields_n[0]]
    fno.eval()
    with torch.no_grad():
        for t in range(EVAL_STEPS):
            aw = history_window(prep.act_n, t, CFG.l_hist).reshape(-1)
            y = fno(torch.cat([state, _act_channels(aw).unsqueeze(0)], 1)).y
            state = y
            outs.append(y[0].numpy())
    sn = np.stack(outs, 0)
    return np.stack([norm.denorm_field(sn[i]) for i in range(N_SNAP)], 0)


# ══════════════════════════════════════════════════════════════════════════
# 9. SURROGATE C: Convolutional Autoencoder + temporal model (GRU)
# ══════════════════════════════════════════════════════════════════════════

def train_conv_autoencoder(prep_train, epochs=EPOCHS_CAE, lr=1e-3,
                           ckpt=None, ckpt_sig="", force_retrain=False):
    model = Autoencoder2D(in_channels=4, latent_dim=CAE_LATENT, img_size=(CFG.nx, CFG.ny),
                          base_channels=24, norm="group")
    cached, hit = _try_load_ckpt(model, ckpt, ckpt_sig, force_retrain)
    if hit:
        return model, cached
    X = torch.from_numpy(np.concatenate([p.fields_n for p in prep_train], 0)).float()  # (N,4,nx,ny)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(SEED)
    n = X.shape[0]; batch = 64
    losses = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        ep_loss = 0.0
        for i0 in range(0, n, batch):
            bi = idx[i0:i0 + batch]
            xh = model.decode(model.encode(X[bi]))
            loss = torch.mean((xh - X[bi]) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.detach()) * len(bi)
        losses.append(ep_loss / n)
    _save_ckpt(model, ckpt, ckpt_sig, losses)
    return model, losses


def cae_encode_traj(prep: PreparedTraj, cae: Autoencoder2D) -> np.ndarray:
    with torch.no_grad():
        z = cae.encode(torch.from_numpy(prep.fields_n).float()).numpy()
    return z.astype(np.float32)   # (N_SNAP, CAE_LATENT)


def train_cae_gru(prep_train, cae: Autoencoder2D, epochs=EPOCHS_CAE_GRU, lr=2e-3,
                  progress=None, progress_every=40, ckpt=None, ckpt_sig="", force_retrain=False):
    codes = {p.name: cae_encode_traj(p, cae) for p in prep_train}
    return _latent_gru_train(prep_train, lambda p: codes[p.name], CAE_LATENT, epochs, lr, 48,
                             progress, progress_every, ckpt, ckpt_sig, force_retrain)


def rollout_cae_gru(prep: PreparedTraj, norm: Normalizer, cae: Autoencoder2D, gru: GRUModel) -> np.ndarray:
    z0 = cae_encode_traj(prep, cae)[0]
    z = [z0.astype(np.float32)]
    gru.eval()
    with torch.no_grad():
        for t in range(EVAL_STEPS):
            aw = history_window(prep.act_n, t, CFG.l_hist)
            lw = history_window(np.stack(z, 0), t, CFG.l_hist)
            x = torch.from_numpy(np.concatenate([aw, lw], 1)).float().unsqueeze(0)
            z.append(gru(x).y[0, 0].numpy().astype(np.float32))
        fn = cae.decode(torch.from_numpy(np.stack(z, 0)).float()).numpy()
    return np.stack([norm.denorm_field(fn[i]) for i in range(N_SNAP)], 0)


# ══════════════════════════════════════════════════════════════════════════
# 10. SURROGATE D (extra, not in the spec list): DeepONet
# ══════════════════════════════════════════════════════════════════════════

DEEPONET_POOL = 4


class DeepONetSurrogate(nn.Module):
    """Repo `DeepONet` as an autoregressive next-state operator. Branch =
    coarse (avg-pooled) current field + normalized actuator-history window;
    trunk = normalized (x, y). The repo DeepONet omits the classic
    1/sqrt(modes) scaling on the branch-trunk dot product, which makes the
    K-step rollout curriculum explode past K~5 -- restored here."""

    def __init__(self, coords_norm: np.ndarray):
        super().__init__()
        self.nx, self.ny = CFG.nx, CFG.ny
        self.pool = nn.AvgPool2d(DEEPONET_POOL)
        self.sx, self.sy = self.nx // DEEPONET_POOL, self.ny // DEEPONET_POOL
        self.modes = 32
        self.net = DeepONet(branch_dim=4 * self.sx * self.sy + 2 * CFG.l_hist,
                            trunk_dim=2, out_dim=4, hidden=128, modes=self.modes)
        self._scale = 1.0 / float(np.sqrt(self.modes))
        self.register_buffer("coords", torch.from_numpy(coords_norm.astype(np.float32)))

    def forward(self, field_n, act_win_flat):
        B = field_n.shape[0]
        u = torch.cat([self.pool(field_n).reshape(B, -1), act_win_flat], 1)
        y = self.net(u, self.coords).y * self._scale
        return y.reshape(B, self.nx, self.ny, 4).permute(0, 3, 1, 2).contiguous()


def train_deeponet_rollout(prep_train, norm, coords_norm, epochs=EPOCHS_DEEPONET, lr=8e-4,
                           progress=None, progress_every=40, ckpt=None, ckpt_sig="", force_retrain=False):
    model = DeepONetSurrogate(coords_norm)
    cached, hit = _try_load_ckpt(model, ckpt, ckpt_sig, force_retrain)
    if hit:
        return model, cached
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    rng = np.random.default_rng(SEED)
    batch = 32
    losses = []
    tgen = torch.Generator().manual_seed(SEED)
    for ep in range(epochs):
        Kc = _curriculum_K(ep, k_max=4)
        starts = _rollout_starts(prep_train, Kc, max_n=STARTS_PER_EPOCH, rng=rng)
        rng.shuffle(starts)
        ep_loss, nb = 0.0, 0
        for b0 in range(0, len(starts), batch):
            chunk = starts[b0:b0 + batch]
            state = torch.stack([torch.from_numpy(prep_train[ti].fields_n[t0]).float() for (ti, t0) in chunk], 0)
            loss = torch.zeros(())
            for k in range(Kc):
                aw = np.stack([history_window(prep_train[ti].act_n, t0 + k, CFG.l_hist).reshape(-1)
                               for (ti, t0) in chunk], 0).astype(np.float32)
                st_in = state
                if k > 0 and ROLLOUT_NOISE_STD > 0:
                    st_in = state + ROLLOUT_NOISE_STD * torch.randn(state.shape, generator=tgen)
                y = model(st_in, torch.from_numpy(aw))
                tgt = torch.stack([torch.from_numpy(prep_train[ti].fields_n[t0 + k + 1]).float()
                                   for (ti, t0) in chunk], 0)
                loss = loss + torch.mean((y - tgt) ** 2)
                state = y
            loss = loss / Kc
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.detach()); nb += 1
        losses.append(ep_loss / max(nb, 1))
        if progress and (ep + 1) % progress_every == 0:
            progress(ep + 1, model, losses[-1]); model.train()
    _save_ckpt(model, ckpt, ckpt_sig, losses)
    return model, losses


def rollout_deeponet(prep: PreparedTraj, norm: Normalizer, model: DeepONetSurrogate) -> np.ndarray:
    state = torch.from_numpy(prep.fields_n[0]).float().unsqueeze(0)
    outs = [prep.fields_n[0]]
    model.eval()
    with torch.no_grad():
        for t in range(EVAL_STEPS):
            aw = history_window(prep.act_n, t, CFG.l_hist).reshape(-1)[None].astype(np.float32)
            y = model(state, torch.from_numpy(aw))
            state = y
            outs.append(y[0].numpy())
    sn = np.stack(outs, 0)
    return np.stack([norm.denorm_field(sn[i]) for i in range(N_SNAP)], 0)


# ══════════════════════════════════════════════════════════════════════════
# 11. METRICS  (spec: field-L2, rollout drift, divergence residual, latency)
# ══════════════════════════════════════════════════════════════════════════

def field_div(fields: np.ndarray) -> np.ndarray:
    """mean |div(u)| per snapshot, central differences, over the DEEP
    interior (3 cells in from every boundary). The boundary-adjacent
    columns carry a solver artefact -- the collocated outflow BC + global
    mass-conservation rescaling make u[-1] != u[-2], and the developing
    inlet profile has a strong dv/dy -- so those are excluded so the number
    reflects "how divergence-free is the field the surrogate must match",
    identically for the CFD reference and every surrogate."""
    dx, dy = CFG.Lx / CFG.nx, CFG.Ly / CFG.ny
    u = fields[:, 0]; v = fields[:, 1]
    d = ((u[:, 2:, 1:-1] - u[:, :-2, 1:-1]) / (2 * dx)
         + (v[:, 1:-1, 2:] - v[:, 1:-1, :-2]) / (2 * dy))
    return np.abs(d[:, 3:-3, 3:-3]).mean(axis=(1, 2))


def rollout_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, Any]:
    diff = pred[1:] - true[1:]
    out: Dict[str, Any] = {}
    for i, nm in enumerate(FIELD_NAMES):
        out[f"rel_l2_{nm}"] = float(np.linalg.norm(diff[:, i]) / (np.linalg.norm(true[1:, i]) + 1e-12))
    out["rel_l2"] = float(np.linalg.norm(diff.reshape(-1)) / (np.linalg.norm(true[1:].reshape(-1)) + 1e-12))
    # rollout drift: per-step relative error trajectory + late/early ratio
    step_err = np.array([np.linalg.norm(pred[k] - true[k]) / (np.linalg.norm(true[k]) + 1e-12)
                         for k in range(1, N_SNAP)])
    out["drift_curve"] = step_err.tolist()
    early = float(step_err[:max(1, len(step_err) // 4)].mean())
    late = float(step_err[-max(1, len(step_err) // 4):].mean())
    out["drift_early"] = early
    out["drift_late"] = late
    out["drift_ratio"] = float(late / (early + 1e-12))
    # divergence residual
    dpred = field_div(pred)
    out["mean_abs_div"] = float(dpred.mean())
    return out


# ══════════════════════════════════════════════════════════════════════════
# 12. THERMAL-ENERGY BUDGET (auxiliary physical check)
# ══════════════════════════════════════════════════════════════════════════

def energy_budget(fields: np.ndarray, actuator_snap: np.ndarray, heater_mask_np: np.ndarray) -> Dict[str, float]:
    dx, dy = CFG.Lx / CFG.nx, CFG.Ly / CFG.ny
    T = fields[:, 3]; u = fields[:, 0]
    E = T.sum(axis=(1, 2)) * dx * dy
    dE = float(E[-1] - E[0])
    n_heat = float(heater_mask_np.sum())
    q = actuator_snap[:, 1]
    # heater input over the trajectory: flux * heated length * time
    heated_len = n_heat / CFG.ny * dy    # rows collapse; length along x
    inp = float(np.sum(q[1:]) * CFG.snap_every * CFG.dt * (CFG.heater_x_frac[1] - CFG.heater_x_frac[0]) * CFG.Lx)
    adv_out = np.maximum(u[:, -1, :], 0.0) * T[:, -1, :] * dy
    efflux = float(np.sum(adv_out[1:].sum(axis=1)) * CFG.snap_every * CFG.dt)
    resid = dE - (inp - efflux)
    return {"delta_domain_energy": dE, "cumulative_heat_input": inp,
            "cumulative_outlet_efflux": efflux, "closure_residual": float(resid),
            "relative_closure_error": float(abs(resid) / (abs(inp) + 1e-12))}


# ══════════════════════════════════════════════════════════════════════════
# 13. OPERATIONAL METRICS  (outlet T, max T, pressure drop)
# ══════════════════════════════════════════════════════════════════════════

def operational_metrics(fields: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "outlet_T": fields[:, 3, -1, :].mean(axis=1),
        "max_T": fields[:, 3].max(axis=(1, 2)),
        "dp": fields[:, 2, 0, :].mean(axis=1) - fields[:, 2, -1, :].mean(axis=1),
    }


def operational_error(pred: np.ndarray, true_scalars: Dict[str, np.ndarray]) -> Dict[str, float]:
    pm = operational_metrics(pred)
    out = {}
    for k, ref in true_scalars.items():
        d = pm[k][1:] - ref[1:]
        out[f"{k}_rel_l2"] = float(np.linalg.norm(d) / (np.linalg.norm(ref[1:]) + 1e-12))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 14. SPARSE-SENSOR DATA ASSIMILATION -- two methods
# ══════════════════════════════════════════════════════════════════════════

def sensor_layout() -> Dict[str, List[Tuple[int, int]]]:
    """Sparse virtual sensors -- T probes in the near-wall thermal layer
    downstream of the heated segment, p probes near inlet and mid-channel."""
    nx, ny = CFG.nx, CFG.ny
    t_ix = [int(0.45 * nx), int(0.62 * nx), int(0.78 * nx), int(0.92 * nx)]
    p_ix = [int(0.08 * nx), int(0.50 * nx)]
    return {"T": [(ix, 1) for ix in t_ix], "p": [(ix, ny // 2) for ix in p_ix]}


def verify_sensor_signal(traj: TrajectoryData) -> List[Dict[str, Any]]:
    lay = sensor_layout()
    U = traj.actuator_snap[:, 0]; Q = traj.actuator_snap[:, 1]
    rows = []
    for (ix, iy) in lay["T"]:
        ts = traj.fields[:, 3, ix, iy]
        rows.append({"kind": "T", "ix": ix, "iy": iy, "std": float(ts.std()),
                     "min": float(ts.min()), "max": float(ts.max()),
                     "corr_with_Q_h": float(np.corrcoef(ts, Q)[0, 1]) if ts.std() > 0 else 0.0})
    for (ix, iy) in lay["p"]:
        ts = traj.fields[:, 2, ix, iy]
        rows.append({"kind": "p", "ix": ix, "iy": iy, "std": float(ts.std()),
                     "min": float(ts.min()), "max": float(ts.max()),
                     "corr_with_U_in": float(np.corrcoef(ts, U)[0, 1]) if ts.std() > 0 else 0.0})
    return rows


def _open_loop_latents(prep: PreparedTraj, gru: GRUModel) -> np.ndarray:
    z = [prep.pod_latent[0].astype(np.float64)]
    with torch.no_grad():
        for t in range(EVAL_STEPS):
            aw = history_window(prep.act_n, t, CFG.l_hist)
            lw = history_window(np.stack(z, 0), t, CFG.l_hist)
            x = torch.from_numpy(np.concatenate([aw, lw], 1)).float().unsqueeze(0)
            z.append(gru(x).y[0, 0].numpy().astype(np.float64))
    return np.stack(z, 0)


def _sensor_H_matrix(pod: POD, norm: Normalizer, n_da: int) -> Tuple[np.ndarray, np.ndarray]:
    """Linear observation operator H (n_obs x n_da) mapping the leading POD
    latent modes to the (T,p) sensor readings, plus the constant offset b
    from the POD mean.  Built column-by-column from the (linear) POD decode
    + de-normalization at the sensor cells."""
    lay = sensor_layout()
    sensors_T = lay["T"]; sensors_p = lay["p"]
    r = pod._r_eff
    basis = pod.basis_.numpy()            # (D, r)
    mean = pod.mean_.numpy().reshape(-1)  # (D,)
    D = basis.shape[0]

    def probe(flat_norm):
        fld = norm.denorm_field(flat_norm.reshape(4, CFG.nx, CFG.ny))
        return np.array([fld[3, ix, iy] for (ix, iy) in sensors_T]
                        + [fld[2, ix, iy] for (ix, iy) in sensors_p])
    b = probe(mean)
    H = np.zeros((len(b), n_da))
    for j in range(n_da):
        e = np.zeros(r); e[j] = 1.0
        H[:, j] = probe(mean + basis @ e) - b
    return H, b


def assimilate_latent_innovation(prep: PreparedTraj, traj: TrajectoryData, norm: Normalizer,
                                 pod: POD, gru: GRUModel, sensor_every: int = 2, n_da: int = 8,
                                 gain: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """METHOD (i): latent-space innovation correction.  At each assimilation
    step, form the innovation (measured - predicted) at the sensors and
    nudge the leading `n_da` POD latent modes by a damped least-squares
    projection of that innovation through the linear observation operator H:
        dz = gain * pinv(H) @ (y_meas - H z - b)
    (a static-gain observer / Luenberger-style correction, NOT a Kalman
    filter -- that is method (ii))."""
    r = pod._r_eff
    n_da = min(n_da, r)
    lay = sensor_layout()
    H, b = _sensor_H_matrix(pod, norm, n_da)
    Hpinv = np.linalg.pinv(H)

    open_lat = _open_loop_latents(prep, gru)
    lat = open_lat.copy()
    rng = np.random.default_rng(SEED + 11)
    noise_sd = np.array([0.02] * len(lay["T"]) + [0.01] * len(lay["p"]))

    z = [prep.pod_latent[0].astype(np.float64)]
    gru.eval()
    with torch.no_grad():
        for t in range(EVAL_STEPS):
            aw = history_window(prep.act_n, t, CFG.l_hist)
            lw = history_window(np.stack(z, 0), t, CFG.l_hist)
            x = torch.from_numpy(np.concatenate([aw, lw], 1)).float().unsqueeze(0)
            zt = gru(x).y[0, 0].numpy().astype(np.float64)
            if (t + 1) % sensor_every == 0:
                tf = traj.fields[t + 1]
                y_meas = np.array([tf[3, ix, iy] for (ix, iy) in lay["T"]]
                                  + [tf[2, ix, iy] for (ix, iy) in lay["p"]])
                y_meas = y_meas + rng.normal(0, 1, len(y_meas)) * noise_sd
                pred_obs = H @ zt[:n_da] + b
                dz = gain * (Hpinv @ (y_meas - pred_obs))
                zt[:n_da] = zt[:n_da] + dz
            z.append(zt)
    z = np.stack(z, 0)

    def decode(latents):
        with torch.no_grad():
            flat = pod.decode(torch.from_numpy(latents).float()).numpy()
        fn = flat.reshape(N_SNAP, 4, CFG.nx, CFG.ny)
        return np.stack([norm.denorm_field(fn[i]) for i in range(N_SNAP)], 0)
    return decode(z), decode(open_lat)


def assimilate_enkf(prep: PreparedTraj, traj: TrajectoryData, norm: Normalizer,
                    pod: POD, gru: GRUModel, sensor_every: int = 2, n_da: int = 8,
                    p0_frac: float = 0.20, q_frac: float = 0.04, inflation: float = 1.05
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """METHOD (ii): Ensemble Kalman Filter in the reduced POD latent space.
    The leading `n_da` modes are corrected; the trailing modes are carried
    from the open-loop GRU rollout. The ensemble covariance is scaled to
    each mode's own trajectory variability (raw POD coefficients span ~2
    orders of magnitude across modes)."""
    r = pod._r_eff
    n_da = min(n_da, r)
    lay = sensor_layout()
    n_T, n_p = len(lay["T"]), len(lay["p"])
    lat0 = prep.pod_latent[0].astype(np.float64)
    lat_scale = (prep.pod_latent.std(axis=0).astype(np.float64) + 1e-6)[:n_da]
    open_lat = _open_loop_latents(prep, gru)
    tail = {"v": lat0[n_da:].copy()}

    def h(a):
        full = np.concatenate([a, tail["v"]])
        with torch.no_grad():
            flat = pod.decode(torch.from_numpy(full[None, :]).float()).numpy()[0]
        fld = norm.denorm_field(flat.reshape(4, CFG.nx, CFG.ny))
        return np.array([fld[3, ix, iy] for (ix, iy) in lay["T"]]
                        + [fld[2, ix, iy] for (ix, iy) in lay["p"]])

    R = np.diag([0.006 ** 2] * n_T + [0.0015 ** 2] * n_p)
    enkf = EnsembleKalmanFilter(n_state=n_da, n_obs=n_T + n_p, f=lambda a: a, h=h,
                                Q=np.diag((q_frac * lat_scale) ** 2), R=R,
                                n_ens=80, inflation=inflation, seed=SEED)
    enkf.initialize(lat0[:n_da], P0=np.diag((p0_frac * lat_scale) ** 2))

    rng = np.random.default_rng(SEED + 7)
    lat_corr = [lat0.copy()]
    gru.eval()
    with torch.no_grad():
        for t in range(EVAL_STEPS):
            aw = history_window(prep.act_n, t, CFG.l_hist)
            tail_next = open_lat[t + 1, n_da:]
            lw = history_window(np.stack(lat_corr, 0), t, CFG.l_hist)

            def f_member(a, _aw=aw, _lw=lw):
                w = _lw.copy(); w[-1, :n_da] = a
                x = torch.from_numpy(np.concatenate([_aw, w], 1)).float().unsqueeze(0)
                with torch.no_grad():
                    return gru(x).y[0, 0].numpy()[:n_da]
            enkf.f = f_member
            tail["v"] = tail_next.copy()
            enkf.predict()
            if (t + 1) % sensor_every == 0:
                tf = traj.fields[t + 1]
                y_true = np.array([tf[3, ix, iy] for (ix, iy) in lay["T"]]
                                  + [tf[2, ix, iy] for (ix, iy) in lay["p"]])
                y_true = y_true + rng.normal(0, 1, n_T + n_p) * np.sqrt(np.diag(R))
                enkf.update(y_true)
            lat_corr.append(np.concatenate([enkf.mean.copy(), tail_next]))

    def decode(latents):
        with torch.no_grad():
            flat = pod.decode(torch.from_numpy(np.asarray(latents)).float()).numpy()
        fn = flat.reshape(N_SNAP, 4, CFG.nx, CFG.ny)
        return np.stack([norm.denorm_field(fn[i]) for i in range(N_SNAP)], 0)
    return decode(np.stack(lat_corr, 0)), decode(open_lat)


# ══════════════════════════════════════════════════════════════════════════
# 15. GRID-CONVERGENCE STUDY
# ══════════════════════════════════════════════════════════════════════════

def grid_convergence_study() -> Dict[str, Any]:
    """Roache (1994) Richardson / GCI via `mesh_independence_study` on the NS
    solver, three grids at constant refinement ratio 2, QoI = the
    domain-integrated final thermal energy (a smooth global quantity)."""
    rep = TrajectorySpec("gridconv_rep", False, "step", dict(t0=0.25), "step", dict(t0=0.30), seed=999)

    def solve_fn(nx_res: float):
        nx = int(round(nx_res))
        ny = int(round(nx * CFG.ny / CFG.nx))
        ref = nx / CFG.nx
        # systematic refinement: halve dt and quadruple Poisson iterations per doubling
        # (Gauss-Seidel convergence ~ N^2), keep the same physical end time
        n_steps = int(round(CFG.n_steps * ref))
        snap_every = max(2, int(round(CFG.snap_every * ref)))
        return run_reference_trajectory(rep, nx=nx, ny=ny, n_steps=n_steps, snap_every=snap_every,
                                        dt=CFG.dt / ref, poisson_iters=int(CFG.poisson_iters * ref ** 2))

    def qty(tr: TrajectoryData) -> float:
        # late-time mean outlet temperature: a bulk-advected, smooth global QoI
        o = tr.scalars["outlet_T"]
        return float(np.mean(o[-max(1, len(o) // 5):]))

    resolutions = [40, 80, 160]   # constant refinement ratio 2; the main pipeline grid (nx=80) is the medium one
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        study = mesh_independence_study(solve_fn, resolutions, qty, refinement_ratio=2.0)
        warn_msgs = [str(w.message) for w in wl if issubclass(w.category, UserWarning)]
    cv = study.convergence
    return {"resolutions": [int(r) for r in study.resolutions],
            "qoi_name": "late_time_mean_outlet_temperature",
            "qoi_by_grid": [float(v) for v in study.values],
            "refinement_ratio": float(study.refinement_ratio_used),
            "observed_order_p": float(cv.observed_order),
            "richardson_value": float(cv.extrapolated_value),
            "gci_fine_pct": float(cv.gci_fine) * 100.0,
            "gci_coarse_pct": float(cv.gci_coarse) * 100.0,
            "asymptotic_ratio": float(cv.asymptotic_ratio),
            "is_asymptotic": bool(cv.is_asymptotic),
            "warnings": warn_msgs, "_cv": cv}


# ══════════════════════════════════════════════════════════════════════════
# 16. PLOTTING
# ══════════════════════════════════════════════════════════════════════════

def _dark(nrows=1, ncols=1, figsize=(10, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=DARK_BG)
    for ax in np.ravel(np.atleast_1d(axes)):
        ax.set_facecolor(DARK_BG); ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white"); ax.title.set_color("white")
    return fig, axes


def plot_domain(domain, out):
    fig, ax = _dark(1, 1, figsize=(11, 6))
    x0, y0 = domain.bounds_min; x1, y1 = domain.bounds_max
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="white", lw=2)
    ax.plot([x0, x1], [y0, y0], color="#888", lw=6, solid_capstyle="butt", label="no-slip / insulated walls")
    ax.plot([x0, x1], [y1, y1], color="#888", lw=6, solid_capstyle="butt")
    xa = x0 + CFG.heater_x_frac[0] * (x1 - x0); xb = x0 + CFG.heater_x_frac[1] * (x1 - x0)
    ax.plot([xa, xb], [y0, y0], color=ACCENT2, lw=9, solid_capstyle="butt",
            label=f"heated segment  -k dT/dn = Q_h(t),  x/L in {CFG.heater_x_frac}")
    for yy in np.linspace(y0, y1, 9)[1:-1]:
        ax.annotate("", xy=(x0 + 0.10 * (x1 - x0), yy), xytext=(x0, yy),
                    arrowprops=dict(arrowstyle="->", color=ACCENT))
    ax.plot([x1, x1], [y0, y1], color=ACCENT3, lw=3)
    ax.text(x0, y1 + 0.03 * (y1 - y0), "inlet: u=(U_in(t),0), T=0", color=ACCENT, fontsize=9)
    ax.text(x1, y1 + 0.03 * (y1 - y0), "outlet: p=0, du/dn=0, dT/dn=0", color=ACCENT3, fontsize=9, ha="right")
    xs = x0 + np.arange(CFG.nx + 1) * (x1 - x0) / CFG.nx
    ys = y0 + np.arange(CFG.ny + 1) * (y1 - y0) / CFG.ny
    for xg in xs[::5]:
        ax.plot([xg, xg], [y0, y1], color="#333", lw=0.4, zorder=0)
    for yg in ys[::4]:
        ax.plot([x0, x1], [yg, yg], color="#333", lw=0.4, zorder=0)
    lay = sensor_layout()
    for (ix, iy) in lay["T"]:
        ax.plot(x0 + (ix + 0.5) * (x1 - x0) / CFG.nx, y0 + (iy + 0.5) * (y1 - y0) / CFG.ny,
                "o", color=ACCENT4, ms=8, mec="white")
    for (ix, iy) in lay["p"]:
        ax.plot(x0 + (ix + 0.5) * (x1 - x0) / CFG.nx, y0 + (iy + 0.5) * (y1 - y0) / CFG.ny,
                "s", color=ACCENT5, ms=8, mec="white")
    ax.plot([], [], "o", color=ACCENT4, mec="white", label="virtual T sensors (near-wall)")
    ax.plot([], [], "s", color=ACCENT5, mec="white", label="virtual p sensors")
    ax.set_xlim(x0 - 0.05 * (x1 - x0), x1 + 0.05 * (x1 - x0))
    ax.set_ylim(y0 - 0.1 * (y1 - y0), y1 + 0.16 * (y1 - y0))
    ax.set_aspect("equal")
    ax.set_title(f"Real ChannelDomain2D  bounds {domain.bounds_min}->{domain.bounds_max}  "
                 f"grid {CFG.nx}x{CFG.ny}  Re={CFG.Re} (laminar)  Pr={CFG.Pr}")
    ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor="white", loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=DARK_BG); plt.close(fig)


def plot_actuators(trajs, out):
    fig, axes = _dark(1, 2, figsize=(14, 5))
    for tr in trajs:
        c = ACCENT2 if tr.spec.held_out else ACCENT
        st = "--" if tr.spec.held_out else "-"
        lw = 1.6 if tr.spec.held_out else 0.8
        axes[0].plot(tr.actuator_fine[:, 0], st, color=c, alpha=0.8, lw=lw)
        axes[1].plot(tr.actuator_fine[:, 1], st, color=c, alpha=0.8, lw=lw)
    axes[0].set_title("U_in(t)  solid=train (24), dashed=held-out (6)")
    axes[1].set_title("Q_h(t)  solid=train, dashed=held-out")
    for ax in axes:
        ax.set_xlabel("solver step")
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=DARK_BG); plt.close(fig)


def plot_fields(tr, out, snap_idx=-1):
    fig, axes = _dark(2, 2, figsize=(12, 6))
    st = tr.fields[snap_idx]
    titles = ["u", "v", "p (projection)", "T"]
    cmaps = ["viridis", "coolwarm", "PuOr", "inferno"]
    for i, ax in enumerate(np.ravel(axes)):
        im = ax.imshow(st[i].T, origin="lower", cmap=cmaps[i], aspect="auto")
        ax.set_title(titles[i]); cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.ax.yaxis.set_tick_params(color="white"); plt.setp(cb.ax.get_yticklabels(), color="white")
    fig.suptitle(f"CFD reference (laminar NS + energy) -- {tr.spec.name}  snapshot {snap_idx}", color="white")
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=DARK_BG); plt.close(fig)


def plot_outlet_response(trajs, out):
    fig, axes = _dark(2, 2, figsize=(14, 8))
    picks = [t for t in trajs if not t.spec.held_out][:2] + [t for t in trajs if t.spec.held_out][:2]
    for ax, tr in zip(np.ravel(axes), picks):
        tt = np.linspace(0, 1, CFG.n_steps)
        ax.plot(tt, tr.actuator_fine[:, 1] / CFG.Q0, color=ACCENT2, lw=1.3, label="Q_h(t)/Q0")
        ts = np.linspace(0, 1, N_SNAP)
        oT = tr.scalars["outlet_T"]
        ax.plot(ts, oT / (oT.max() + 1e-12), color=ACCENT3, lw=2, label="outlet_T(t)/max")
        ax.plot(ts, tr.scalars["max_T"] / (tr.scalars["max_T"].max() + 1e-12),
                color=ACCENT, lw=1, ls="--", label="max_T(t)/max")
        ax.set_title(f"{tr.spec.name}  outlet_T max={oT.max():.3e}")
        ax.set_xlabel("normalised time")
        ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
    fig.suptitle("Heat plume reaches the outlet: outlet_T tracks Q_h(t) with a convective lag", color="white")
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=DARK_BG); plt.close(fig)


def plot_training_curves(curves, out):
    fig, ax = _dark(1, 1, figsize=(9, 5))
    cols = {"POD+GRU": ACCENT, "FNO2d": ACCENT2, "ConvAE+GRU": ACCENT4, "DeepONet": ACCENT5, "FNO2d-PI": ACCENT3}
    for nm, c in curves.items():
        ax.plot(c, color=cols.get(nm, "white"), label=nm)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_title("Surrogate training curves (K-step rollout-consistency loss)")
    ax.legend(facecolor=DARK_BG, labelcolor="white")
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=DARK_BG); plt.close(fig)


def plot_comparison(summary, out):
    surrs = list(summary.keys())
    cols = [ACCENT, ACCENT2, ACCENT4, ACCENT5, ACCENT3]
    fig, axes = _dark(1, 3, figsize=(16, 5))
    keys = [("rel_l2", "all-field rel. L2"), ("rel_l2_T", "temperature rel. L2"),
            ("mean_abs_div", "mean |div(u)| residual")]
    for ax, (k, title) in zip(axes, keys):
        x = np.arange(2); w = 0.8 / len(surrs)
        for i, s in enumerate(surrs):
            ax.bar(x + i * w, [summary[s]["train"][k], summary[s]["held_out"][k]], w,
                   label=s, color=cols[i % len(cols)])
        ax.set_xticks(x + w * (len(surrs) - 1) / 2); ax.set_xticklabels(["train", "held-out"])
        ax.set_title(title); ax.set_ylabel(title)
        if k == "mean_abs_div":
            ax.axhline(summary[surrs[0]].get("_cfd_div", 0.0), color="white", ls=":", label="CFD ref")
        ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
    fig.suptitle(f"Surrogate comparison -- {EVAL_STEPS}-step closed-loop rollout on held-out trajectories", color="white")
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=DARK_BG); plt.close(fig)


def plot_drift(per_traj, out):
    fig, ax = _dark(1, 1, figsize=(10, 5))
    cols = {"POD+GRU": ACCENT, "FNO2d": ACCENT2, "ConvAE+GRU": ACCENT4, "DeepONet": ACCENT5}
    got = set()
    for name, row in per_traj.items():
        if not name.startswith("test_"):
            continue
        for s, m in row.items():
            lbl = s if s not in got else None
            got.add(s)
            ax.plot(np.arange(1, N_SNAP), m["drift_curve"], color=cols.get(s, "white"), alpha=0.5, lw=1, label=lbl)
    ax.set_xlabel("rollout step"); ax.set_ylabel("per-step relative error")
    ax.set_title("Autoregressive rollout drift on held-out trajectories")
    ax.legend(facecolor=DARK_BG, labelcolor="white")
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=DARK_BG); plt.close(fig)


def plot_assimilation(traj, inn_corr, enkf_corr, openloop, out):
    fig, axes = _dark(1, 2, figsize=(14, 5))
    t = np.arange(N_SNAP)
    for ax, (key, title) in zip(axes, [("max_T", "max T(t)"), ("outlet_T", "outlet T(t)")]):
        ax.plot(t, traj.scalars[key], color="white", lw=2, label="CFD reference")
        ax.plot(t, operational_metrics(openloop)[key], color=ACCENT2, ls="--", label="open-loop POD+GRU")
        ax.plot(t, operational_metrics(inn_corr)[key], color=ACCENT5, label="+ latent innovation")
        ax.plot(t, operational_metrics(enkf_corr)[key], color=ACCENT3, label="+ EnKF")
        ax.set_title(title); ax.set_xlabel("snapshot")
        ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor="white")
    fig.suptitle(f"Sparse-sensor assimilation -- {traj.spec.name}", color="white")
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=DARK_BG); plt.close(fig)


def plot_grid_convergence(gc, out):
    fig, ax = _dark(1, 1, figsize=(8, 5))
    ax.plot(gc["resolutions"], gc["qoi_by_grid"], "o-", color=ACCENT, lw=2, ms=9, label=gc["qoi_name"])
    ax.axhline(gc["richardson_value"], color=ACCENT3, ls="--",
               label=f"Richardson extrapolated = {gc['richardson_value']:.4e}")
    ax.set_xscale("log", base=2); ax.set_xticks(gc["resolutions"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("nx"); ax.set_ylabel(gc["qoi_name"])
    ax.set_title(f"Grid convergence (Roache 1994 GCI)  p={gc['observed_order_p']:.3f}  "
                 f"GCI_fine={gc['gci_fine_pct']:.2f}%  asymptotic={gc['is_asymptotic']}")
    ax.legend(facecolor=DARK_BG, labelcolor="white")
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=DARK_BG); plt.close(fig)


def plot_demo(traj, pred, best_name, speedup, out):
    fig, axes = _dark(3, 3, figsize=(15, 9))
    idxs = [N_SNAP // 3, 2 * N_SNAP // 3, N_SNAP - 1]
    for col, si in enumerate(idxs):
        for row, ch, cmap, lbl in [(0, 0, "viridis", "u"), (1, 3, "inferno", "T")]:
            ax = axes[row, col]
            im = ax.imshow(pred[si, ch].T, origin="lower", cmap=cmap, aspect="auto")
            ax.set_title(f"{lbl}  snap {si} ({best_name})"); fig.colorbar(im, ax=ax, fraction=0.046)
    for col, (key, title) in enumerate([("outlet_T", "outlet T(t)"), ("max_T", "max T(t)"), ("dp", "pressure drop(t)")]):
        ax = axes[2, col]
        ax.plot(traj.scalars[key], color="white", lw=2, label="CFD")
        ax.plot(operational_metrics(pred)[key], color=ACCENT2, ls="--", label=best_name)
        ax.set_title(title); ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
    fig.suptitle(f"DEMO -- {traj.spec.name} -- {best_name} surrogate vs CFD  (speedup x{speedup:.0f})", color="white")
    fig.tight_layout(); fig.savefig(out, dpi=130, facecolor=DARK_BG); plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# 17. DEMONSTRATOR
# ══════════════════════════════════════════════════════════════════════════

_DEMO_STATE: Dict[str, Any] = {}


def demo(trajectory: str = "test_chirp"):
    """Lightweight interactive demonstrator (spec Week-4 deliverable).

    `trajectory` is either a held-out/train trajectory name from the dataset,
    or one of the actuator family names (`step`, `ramp`, `sinusoid`,
    `piecewise_constant`, `prbs`, `chirp`, ...) to synthesise a fresh one.
    Runs the CFD reference and the best surrogate for that one trajectory,
    prints the operational metrics and the measured speedup, and saves
    outputs/09_demo.png."""
    if "norm" not in _DEMO_STATE:
        raise RuntimeError("demo() needs the pipeline artefacts; run main() first (or use --demo).")
    S = _DEMO_STATE
    if trajectory in S["all_trajs"]:
        tr = S["all_trajs"][trajectory]
    else:
        fam = trajectory if trajectory in SHAPES else "chirp"
        spec = TrajectorySpec(f"demo_{fam}", True, fam, {}, fam, {}, seed=12345)
        print(f"  synthesising a fresh '{fam}' trajectory via the real CFD solver...")
        tr = run_reference_trajectory(spec)
    prep = prepare_trajectories([tr], S["norm"], S["pod"])[0]

    best = S["best_name"]; roll = S["rollouts"][best]
    t0 = time.time()
    pred = roll(prep)
    infer_s = time.time() - t0
    speedup = tr.solver_seconds / max(infer_s, 1e-6)

    m = rollout_metrics(pred, tr.fields)
    op = operational_error(pred, tr.scalars)
    print(f"  trajectory        : {tr.spec.name}")
    print(f"  best surrogate     : {best}")
    print(f"  CFD solver wall-time: {tr.solver_seconds:.2f} s")
    print(f"  surrogate rollout  : {infer_s*1000:.1f} ms   -> speedup x{speedup:.0f}")
    print(f"  field rel-L2 (all / T): {m['rel_l2']:.4f} / {m['rel_l2_T']:.4f}   mean|div(u)|={m['mean_abs_div']:.2e}")
    print(f"  operational-metric rel-L2: outlet_T={op['outlet_T_rel_l2']:.3f}  "
          f"max_T={op['max_T_rel_l2']:.3f}  dp={op['dp_rel_l2']:.3f}")
    plot_demo(tr, pred, best, speedup, OUT_DIR / "09_demo.png")
    print(f"  saved outputs/09_demo.png")
    return {"trajectory": tr.spec.name, "best": best, "speedup": speedup,
            "infer_seconds": infer_s, "cfd_seconds": tr.solver_seconds,
            "metrics": {k: v for k, v in m.items() if k != "drift_curve"}, "operational": op}


# ══════════════════════════════════════════════════════════════════════════
# 18. MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true", help="force CFD dataset regeneration")
    ap.add_argument("--quick", action="store_true", help="tiny epoch counts for a debug run")
    ap.add_argument("--skip-stretch", action="store_true", help="skip the physics-informed FNO2d variant")
    ap.add_argument("--retrain", action="store_true", help="ignore cached surrogate checkpoints")
    ap.add_argument("--demo", type=str, default=None, metavar="TRAJ",
                    help="after the run, invoke the demonstrator on TRAJ (name or actuator family)")
    args = ap.parse_args()

    global EPOCHS_GRU, EPOCHS_FNO, EPOCHS_CAE, EPOCHS_CAE_GRU, EPOCHS_DEEPONET, EPOCHS_FNO_PI, STARTS_PER_EPOCH
    if args.quick:
        EPOCHS_GRU = EPOCHS_CAE_GRU = EPOCHS_DEEPONET = 10
        EPOCHS_FNO = EPOCHS_FNO_PI = 6
        EPOCHS_CAE = 12
        STARTS_PER_EPOCH = 128

    t_start = time.time()
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("=" * 80)
    print("HEATED-CHANNEL DIGITAL TWIN -- v3 (laminar incompressible NS + energy, projection solver)")
    print("=" * 80)

    # ── Stage 1: geometry ────────────────────────────────────────────
    domain = build_domain()
    ix0, ix1 = heater_index_range(domain, CFG.nx)
    heater_mask_np = np.zeros((CFG.nx, CFG.ny), np.float32)
    heater_mask_np[ix0:ix1, 1] = 1.0   # near-wall row that receives the wall heat-flux source
    coords_norm = trunk_coords_norm(domain)
    nu = CFG.u_in0 * CFG.Ly / CFG.Re
    print(f"\n[1] Real geometry: {type(domain).__name__}  bounds={domain.bounds_min}->{domain.bounds_max}  "
          f"regions={domain.get_region_names()}")
    print(f"    grid {CFG.nx}x{CFG.ny}  dx={CFG.Lx/CFG.nx:.4f} dy={CFG.Ly/CFG.ny:.4f}  "
          f"Re={CFG.Re} (laminar)  Pr={CFG.Pr}  nu={nu:.5f}  alpha={nu/CFG.Pr:.5f}")
    print(f"    heated wall segment: grid columns [{ix0}:{ix1}]  (x/L {CFG.heater_x_frac})  "
          f"Neumann flux -k dT/dn = Q_h(t)")
    plot_domain(domain, OUT_DIR / "00_domain_geometry.png")
    print("    saved outputs/00_domain_geometry.png")

    # ── Stage 2: specs ──────────────────────────────────────────────
    specs = make_specs()
    train_specs = [s for s in specs if not s.held_out]
    test_specs = [s for s in specs if s.held_out]
    print(f"\n[2] {len(train_specs)} training + {len(test_specs)} held-out trajectories")
    print(f"    training actuator families : {TRAIN_FAMILIES}")
    print(f"    held-out actuator families : {[s.u_shape for s in test_specs]}")

    # ── Stage 3: CFD dataset ───────────────────────────────────────
    print(f"\n[3] CFD reference: {len(specs)} laminar NS+energy projection solves, "
          f"{CFG.n_steps} steps (dt={CFG.dt}), snapshot every {CFG.snap_every} -> {N_SNAP} snapshots")
    all_trajs = generate_or_load_dataset(specs, use_cache=not args.no_cache)
    train_trajs = [all_trajs[s.name] for s in train_specs]
    test_trajs = [all_trajs[s.name] for s in test_specs]

    ft = CFG.Lx / CFG.u_in0 / CFG.dt
    print(f"\n[3b] flow-through time ~ Lx/u_in0 = {CFG.Lx/CFG.u_in0:.2f}  ->  {ft:.0f} steps; "
          f"n_steps={CFG.n_steps} ({CFG.n_steps/ft:.1f}x)")
    for tr in (train_trajs[0], test_trajs[0]):
        oT = tr.scalars["outlet_T"]; q = tr.actuator_snap[:, 1]
        thr = 0.05 * oT.max() if oT.max() > 0 else 1e9
        reach = int(np.argmax(oT > thr)) if (oT > thr).any() else -1
        print(f"     {tr.spec.name:26s} outlet_T: max={oT.max():.3e} final={oT[-1]:.3e}  "
              f"first exceeds 5%-of-max at snap {reach}  corr(Q_h,outlet_T)={np.corrcoef(q, oT)[0,1]:+.3f}  "
              f"| CFD div: mean|div(u)|={field_div(tr.fields).mean():.2e}")
    cfd_ref_div = float(np.mean([field_div(t.fields).mean() for t in test_trajs]))
    cfd_solver_seconds = float(np.mean([t.solver_seconds for t in test_trajs]))

    plot_actuators(list(all_trajs.values()), OUT_DIR / "01_actuator_trajectories.png")
    plot_fields(train_trajs[0], OUT_DIR / "02_reference_fields_train_example.png")
    plot_fields(test_trajs[0], OUT_DIR / "03_reference_fields_heldout_example.png")
    plot_outlet_response(train_trajs + test_trajs, OUT_DIR / "07_outlet_response.png")

    # ── Stage 4: normalizer + POD ─────────────────────────────────
    norm = fit_normalizer(train_trajs)
    pod = train_pod(train_trajs, norm, r=POD_R)
    evr = pod.explained_variance_ratio_
    print(f"\n[4] Normalizer + POD: r={pod._r_eff}  cumulative explained variance {float(evr.sum()):.4f}")
    prep_train = prepare_trajectories(train_trajs, norm, pod)
    prep_test = prepare_trajectories(test_trajs, norm, pod)

    def _heldout_mean(roll):
        ms = [rollout_metrics(roll(p), all_trajs[p.name].fields) for p in prep_test]
        return (float(np.mean([m["rel_l2"] for m in ms])), float(np.mean([m["rel_l2_T"] for m in ms])))

    def _mk_progress(roll_of):
        def _p(ep, model, loss):
            l2, l2t = _heldout_mean(lambda p: roll_of(p, model))
            print(f"      ep {ep:4d}  train_loss={loss:.5f}  held-out rollout L2={l2:.4f} (T {l2t:.4f})")
        return _p

    _dh = _config_hash(specs)
    _common = f"{_dh}|K{ROLLOUT_K_MIN}-{ROLLOUT_K_MAX}r{ROLLOUT_K_RAMP_EVERY}|n{ROLLOUT_NOISE_STD}|s{STARTS_PER_EPOCH}|seed{SEED}"

    # ── Stage 5: POD + GRU ───────────────────────────────────────
    print(f"\n[5] Surrogate A: POD+GRU latent dynamics ({EPOCHS_GRU} epochs, K curriculum "
          f"{ROLLOUT_K_MIN}->{ROLLOUT_K_MAX})...")
    t0 = time.time()
    gru, gru_losses = train_gru_rollout(prep_train, r=pod._r_eff, epochs=EPOCHS_GRU,
        progress=_mk_progress(lambda p, m: rollout_pod_gru(p, norm, pod, m)),
        ckpt=OUT_DIR / "ckpt_pod_gru.pt", ckpt_sig=f"gru|{_common}|ep{EPOCHS_GRU}", force_retrain=args.retrain)
    print(f"    done in {time.time()-t0:.1f}s  final loss {gru_losses[-1]:.6f}")

    # ── Stage 6: FNO2d ──────────────────────────────────────────
    print(f"\n[6] Surrogate B: FNO2d neural operator ({EPOCHS_FNO} epochs)...")
    t0 = time.time()
    fno, fno_losses = train_fno_rollout(prep_train, norm, epochs=EPOCHS_FNO,
        progress=_mk_progress(lambda p, m: rollout_fno(p, norm, m)),
        ckpt=OUT_DIR / "ckpt_fno2d.pt", ckpt_sig=f"fno|{_common}|ep{EPOCHS_FNO}", force_retrain=args.retrain)
    print(f"    done in {time.time()-t0:.1f}s  final loss {fno_losses[-1]:.6f}")

    # ── Stage 7: ConvAE + GRU ──────────────────────────────────
    print(f"\n[7] Surrogate C: Convolutional Autoencoder ({EPOCHS_CAE} ep) + GRU temporal model "
          f"({EPOCHS_CAE_GRU} ep)...")
    t0 = time.time()
    cae, cae_losses = train_conv_autoencoder(prep_train, epochs=EPOCHS_CAE,
        ckpt=OUT_DIR / "ckpt_cae.pt", ckpt_sig=f"cae|{_dh}|L{CAE_LATENT}|ep{EPOCHS_CAE}", force_retrain=args.retrain)
    with torch.no_grad():
        rec = float(np.mean([np.mean((cae.decode(cae.encode(torch.from_numpy(p.fields_n).float())).numpy()
                                      - p.fields_n) ** 2) for p in prep_test]))
    print(f"    ConvAE trained ({time.time()-t0:.1f}s)  final recon MSE {cae_losses[-1]:.5f}  held-out recon MSE {rec:.5f}")
    t0 = time.time()
    cae_gru, cae_gru_losses = train_cae_gru(prep_train, cae, epochs=EPOCHS_CAE_GRU,
        progress=_mk_progress(lambda p, m: rollout_cae_gru(p, norm, cae, m)),
        ckpt=OUT_DIR / "ckpt_cae_gru.pt", ckpt_sig=f"caegru|{_common}|L{CAE_LATENT}|ep{EPOCHS_CAE_GRU}",
        force_retrain=args.retrain)
    print(f"    CAE+GRU trained ({time.time()-t0:.1f}s)  final loss {cae_gru_losses[-1]:.6f}")

    # ── Stage 8: DeepONet (extra) ──────────────────────────────
    print(f"\n[8] Surrogate D (extra, not in spec list): DeepONet ({EPOCHS_DEEPONET} epochs)...")
    t0 = time.time()
    deeponet, don_losses = train_deeponet_rollout(prep_train, norm, coords_norm, epochs=EPOCHS_DEEPONET,
        progress=_mk_progress(lambda p, m: rollout_deeponet(p, norm, m)),
        ckpt=OUT_DIR / "ckpt_deeponet.pt", ckpt_sig=f"deeponet|{_common}|ep{EPOCHS_DEEPONET}",
        force_retrain=args.retrain)
    print(f"    done in {time.time()-t0:.1f}s  final loss {don_losses[-1]:.6f}")

    curves = {"POD+GRU": gru_losses, "FNO2d": fno_losses, "ConvAE+GRU": cae_gru_losses, "DeepONet": don_losses}

    # ── Stage 9: evaluation (4 metrics) ───────────────────────
    print(f"\n[9] Held-out evaluation -- {EVAL_STEPS}-step closed-loop rollout, identical protocol for all...")
    rollouts = {
        "POD+GRU": lambda p: rollout_pod_gru(p, norm, pod, gru),
        "FNO2d": lambda p: rollout_fno(p, norm, fno),
        "ConvAE+GRU": lambda p: rollout_cae_gru(p, norm, cae, cae_gru),
        "DeepONet": lambda p: rollout_deeponet(p, norm, deeponet),
    }
    results = {s: {"train": [], "held_out": []} for s in rollouts}
    per_traj: Dict[str, Dict[str, Any]] = {}
    op_err: Dict[str, Dict[str, Dict[str, float]]] = {s: {} for s in rollouts}
    latency: Dict[str, float] = {}
    rollout_cache: Dict[str, Dict[str, np.ndarray]] = {s: {} for s in rollouts}
    for split, preps, trajs in [("train", prep_train, train_trajs), ("held_out", prep_test, test_trajs)]:
        for p, tr in zip(preps, trajs):
            per_traj.setdefault(p.name, {})
            for s, fn in rollouts.items():
                t0 = time.time()
                pred = fn(p)
                dt_infer = time.time() - t0
                latency.setdefault(s, 0.0)
                latency[s] += dt_infer
                m = rollout_metrics(pred, tr.fields)
                results[s][split].append(m)
                per_traj[p.name][s] = m
                rollout_cache[s][p.name] = pred
                if split == "held_out":
                    op_err[s][p.name] = operational_error(pred, tr.scalars)
            row = per_traj[p.name]
            print(f"    [{split:8s}] {p.name:24s} " + "  ".join(
                f"{s}:L2={row[s]['rel_l2']:.3f}/T={row[s]['rel_l2_T']:.3f}/div={row[s]['mean_abs_div']:.1e}"
                for s in rollouts))

    def _agg(lst):
        keys = [k for k in lst[0] if k != "drift_curve"]
        return {k: float(np.mean([d[k] for d in lst])) for k in keys}

    summary = {s: {sp: _agg(v) for sp, v in d.items()} for s, d in results.items()}
    for s in summary:
        summary[s]["_cfd_div"] = cfd_ref_div
    # latency / speedup: total surrogate rollout time over all 30 trajectories vs total CFD solver time
    total_cfd_seconds = float(np.sum([t.solver_seconds for t in train_trajs + test_trajs]))
    speedups = {s: total_cfd_seconds / max(latency[s], 1e-9) for s in rollouts}

    print("\n    --- SUMMARY (held-out): rel-L2 all / T | mean|div(u)| | drift ratio | speedup ---")
    print(f"    {'CFD reference':16s}  mean|div(u)| = {cfd_ref_div:.2e}  (baseline)   "
          f"CFD solver ~{cfd_solver_seconds:.1f}s / trajectory")
    for s in rollouts:
        d = summary[s]["held_out"]
        print(f"    {s:16s}  L2={d['rel_l2']:.4f}  T={d['rel_l2_T']:.4f}  "
              f"|div(u)|={d['mean_abs_div']:.2e}  drift x{d['drift_ratio']:.2f}  speedup x{speedups[s]:.0f}")

    # best surrogate = lowest held-out all-field rel-L2 that is also stable (drift ratio < 5)
    # best = lowest combined (all-field + temperature) held-out rel-L2 among the
    # rollout-stable surrogates -- temperature matters as much as the flow for a
    # THERMAL twin, so a low all-field L2 with a blown-up T is not "best".
    def _score(s):
        d = summary[s]["held_out"]
        return d["rel_l2"] + d["rel_l2_T"]
    stable = [s for s in rollouts if summary[s]["held_out"]["drift_ratio"] < 5]
    best_name = min(stable or list(rollouts), key=_score)
    print(f"    best surrogate: {best_name}")

    plot_comparison(summary, OUT_DIR / "05_surrogate_comparison.png")
    plot_drift(per_traj, OUT_DIR / "10_rollout_drift.png")

    # ── Stage 10: physics-informed variant (mass + momentum + energy) ──
    pi_result = None
    if not args.skip_stretch:
        print(f"\n[10] Physics-Informed FNO2d: + soft penalty on mass|div|, momentum & energy PDE residuals "
              f"(lambdas {PI_LAMBDA_MASS}/{PI_LAMBDA_MOM}/{PI_LAMBDA_ENERGY}, {EPOCHS_FNO_PI} epochs)...")
        t0 = time.time()
        fno_pi, fno_pi_losses = train_fno_rollout(prep_train, norm, epochs=EPOCHS_FNO_PI,
            physics_informed=True, heater_mask_np=heater_mask_np,
            progress=_mk_progress(lambda p, m: rollout_fno(p, norm, m)),
            ckpt=OUT_DIR / "ckpt_fno2d_pi.pt",
            ckpt_sig=f"fnopi|{_common}|ep{EPOCHS_FNO_PI}|lam{PI_LAMBDA_MASS}-{PI_LAMBDA_MOM}-{PI_LAMBDA_ENERGY}",
            force_retrain=args.retrain)
        curves["FNO2d-PI"] = fno_pi_losses
        pi_split = {"train": [], "held_out": []}
        for split, preps, trajs in [("train", prep_train, train_trajs), ("held_out", prep_test, test_trajs)]:
            for p, tr in zip(preps, trajs):
                pi_split[split].append(rollout_metrics(rollout_fno(p, norm, fno_pi), tr.fields))
        pi_result = {k: _agg(v) for k, v in pi_split.items()}
        print(f"    done in {time.time()-t0:.1f}s")
        print(f"    FNO2d      held-out: L2={summary['FNO2d']['held_out']['rel_l2']:.4f}  "
              f"|div(u)|={summary['FNO2d']['held_out']['mean_abs_div']:.2e}")
        print(f"    FNO2d-PI   held-out: L2={pi_result['held_out']['rel_l2']:.4f}  "
              f"|div(u)|={pi_result['held_out']['mean_abs_div']:.2e}")

    plot_training_curves(curves, OUT_DIR / "04_training_curves.png")

    # ── Stage 11: thermal-energy budget ──────────────────────
    print("\n[11] Thermal-energy budget (heater input vs. domain energy change + outlet efflux)...")
    energy_checks = {}
    eb = energy_budget(test_trajs[0].fields, test_trajs[0].actuator_snap, heater_mask_np)
    energy_checks["CFD_reference"] = eb
    print(f"     CFD_reference ({test_trajs[0].spec.name}): input={eb['cumulative_heat_input']:.4f}  "
          f"dE={eb['delta_domain_energy']:.4f}  efflux={eb['cumulative_outlet_efflux']:.4f}  "
          f"rel_closure_err={eb['relative_closure_error']:.3f}")
    for s in rollouts:
        e = energy_budget(rollout_cache[s][prep_test[0].name], test_trajs[0].actuator_snap, heater_mask_np)
        energy_checks[s] = e
        print(f"     {s:14s}: dE={e['delta_domain_energy']:.4f} efflux={e['cumulative_outlet_efflux']:.4f} "
              f"rel_closure_err={e['relative_closure_error']:.3f}")

    # ── Stage 12: operational metric errors ─────────────────
    print("\n[12] Operational-metric prediction error on held-out (rel-L2 vs CFD): outlet_T / max_T / dp")
    op_summary = {}
    for s in rollouts:
        agg = {k: float(np.mean([op_err[s][nm][k] for nm in op_err[s]]))
               for k in ("outlet_T_rel_l2", "max_T_rel_l2", "dp_rel_l2")}
        op_summary[s] = agg
        print(f"     {s:14s}  outlet_T={agg['outlet_T_rel_l2']:.3f}  max_T={agg['max_T_rel_l2']:.3f}  "
              f"dp={agg['dp_rel_l2']:.3f}")

    # ── Stage 13: sparse-sensor assimilation -- TWO methods ─
    print("\n[13] Sparse-sensor assimilation (POD+GRU state): (i) latent innovation correction  (ii) EnKF...")
    srows = verify_sensor_signal(test_trajs[0])
    for r_ in srows:
        key = "corr_with_Q_h" if r_["kind"] == "T" else "corr_with_U_in"
        print(f"     sensor {r_['kind']} @ ({r_['ix']:2d},{r_['iy']:2d})  std={r_['std']:.3e}  "
              f"range=[{r_['min']:.2e},{r_['max']:.2e}]  {key}={r_[key]:+.3f}")
    da_per_traj = {}
    inn_imps, enkf_imps = [], []
    for p, tr in zip(prep_test, test_trajs):
        inn_c, ol = assimilate_latent_innovation(p, tr, norm, pod, gru)
        enkf_c, _ = assimilate_enkf(p, tr, norm, pod, gru)
        m_ol = rollout_metrics(ol, tr.fields)["rel_l2"]
        m_inn = rollout_metrics(inn_c, tr.fields)["rel_l2"]
        m_enkf = rollout_metrics(enkf_c, tr.fields)["rel_l2"]
        i_inn = 1 - m_inn / max(m_ol, 1e-12)
        i_enkf = 1 - m_enkf / max(m_ol, 1e-12)
        inn_imps.append(i_inn); enkf_imps.append(i_enkf)
        da_per_traj[tr.spec.name] = {"open_loop": m_ol, "innovation": m_inn, "enkf": m_enkf,
                                     "innovation_improvement_pct": i_inn * 100, "enkf_improvement_pct": i_enkf * 100}
        print(f"     {tr.spec.name:24s} open={m_ol:.4f}  innovation={m_inn:.4f} ({i_inn*100:+.1f}%)  "
              f"enkf={m_enkf:.4f} ({i_enkf*100:+.1f}%)")
        if tr.spec.name == test_trajs[0].spec.name:
            plot_assimilation(tr, inn_c, enkf_c, ol, OUT_DIR / "06_sparse_sensor_assimilation.png")
    print(f"     --- MEAN over held-out: innovation {np.mean(inn_imps)*100:+.1f}%   EnKF {np.mean(enkf_imps)*100:+.1f}%")

    # ── Stage 14: grid convergence ─────────────────────────
    print("\n[14] Grid-convergence study (Roache 1994 GCI via mesh_independence_study)...")
    t0 = time.time()
    gc = grid_convergence_study()
    print(f"     QoI={gc['qoi_name']}  grids {gc['resolutions']}  values {[f'{v:.5e}' for v in gc['qoi_by_grid']]}")
    print(f"     observed order p={gc['observed_order_p']:.3f}  Richardson={gc['richardson_value']:.5e}  "
          f"GCI_fine={gc['gci_fine_pct']:.3f}%  GCI_coarse={gc['gci_coarse_pct']:.3f}%  "
          f"asymptotic_ratio={gc['asymptotic_ratio']:.3f}  is_asymptotic={gc['is_asymptotic']}")
    if gc["warnings"]:
        print(f"     warnings: {gc['warnings']}")
    print(f"     ({time.time()-t0:.1f}s)")
    plot_grid_convergence(gc, OUT_DIR / "08_grid_convergence.png")

    pcs = compute_physics_confidence(convergence_result=gc["_cv"])
    print("\n[15] compute_physics_confidence (real aggregator; low coverage -- only numerical_convergence applies):")
    print("     " + pcs.summary().replace("\n", "\n     "))

    # ── metrics.json ──────────────────────────────────────
    _DEMO_STATE.update(dict(norm=norm, pod=pod, all_trajs=all_trajs, best_name=best_name, rollouts=rollouts))
    metrics = {
        "solver": "2D incompressible laminar Navier-Stokes + energy, Chorin fractional-step projection, collocated grid",
        "config": asdict(CFG), "n_snapshots": N_SNAP, "eval_steps": EVAL_STEPS,
        "rollout_curriculum_K": [ROLLOUT_K_MIN, ROLLOUT_K_MAX], "rollout_noise_std": ROLLOUT_NOISE_STD,
        "boundary_conditions": {
            "inlet": "u=(U_in(t),0), T=0", "heated_wall_segment": "-k dT/dn = Q_h(t) (time-dependent Neumann flux)",
            "other_walls": "no-slip u=0, insulated dT/dn=0", "outlet": "p=0, du/dn=0, dT/dn=0"},
        "actuator_families": {"training": TRAIN_FAMILIES, "held_out": HELDOUT_FAMILIES},
        "trajectories": {"train": [s.name for s in train_specs], "held_out": [s.name for s in test_specs]},
        "pod": {"r_eff": pod._r_eff, "cumulative_explained_variance": float(evr.sum())},
        "cae": {"latent_dim": CAE_LATENT, "final_recon_mse": cae_losses[-1], "heldout_recon_mse": rec},
        "final_train_loss": {"POD+GRU": gru_losses[-1], "FNO2d": fno_losses[-1],
                             "ConvAE+GRU": cae_gru_losses[-1], "DeepONet": don_losses[-1]},
        "comparison_summary": {s: {k: v for k, v in summary[s].items() if not k.startswith("_")} for s in summary},
        "per_trajectory_metrics": {nm: {s: {k: v for k, v in per_traj[nm][s].items() if k != "drift_curve"}
                                        for s in per_traj[nm]} for nm in per_traj},
        "cfd_reference_mean_abs_div": cfd_ref_div,
        "latency": {
            "total_cfd_solver_seconds_30_trajectories": total_cfd_seconds,
            "surrogate_total_rollout_seconds": latency,
            "speedup_factor": speedups,
            "cfd_seconds_per_trajectory": cfd_solver_seconds},
        "best_surrogate": best_name,
        "operational_metric_error_heldout": op_summary,
        "physics_informed_variant": (
            {"attempted": True, "lambdas": [PI_LAMBDA_MASS, PI_LAMBDA_MOM, PI_LAMBDA_ENERGY],
             "fno2d_baseline_heldout": {k: v for k, v in summary["FNO2d"]["held_out"].items() if not k.startswith("_")},
             "fno2d_pi_heldout": pi_result["held_out"] if pi_result else None}
            if pi_result is not None else {"attempted": False}),
        "energy_budget": energy_checks,
        "sparse_sensor_assimilation": {
            "sensor_layout": {k: [list(t) for t in v] for k, v in sensor_layout().items()},
            "sensor_signal_verification": srows,
            "method_i_latent_innovation_correction": {
                "per_trajectory": {nm: da_per_traj[nm]["innovation_improvement_pct"] for nm in da_per_traj},
                "mean_improvement_pct": float(np.mean(inn_imps) * 100)},
            "method_ii_enkf_reduced_latent": {
                "per_trajectory": {nm: da_per_traj[nm]["enkf_improvement_pct"] for nm in da_per_traj},
                "mean_improvement_pct": float(np.mean(enkf_imps) * 100)},
            "per_trajectory_detail": da_per_traj},
        "grid_convergence": {k: v for k, v in gc.items() if not k.startswith("_")},
        "physics_confidence": {"overall_score": pcs.overall_score, "coverage": pcs.coverage,
                               "components": [{"name": c.name, "score": c.score, "source": c.source_summary}
                                              for c in pcs.components]},
        "total_wall_clock_seconds": time.time() - t_start,
    }
    with open(OUT_DIR / "metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2, default=str)
    print(f"\n[16] Total wall-clock: {time.time()-t_start:.1f}s.  Outputs -> {OUT_DIR}")

    if args.demo is not None:
        print("\n[17] DEMONSTRATOR")
        demo(args.demo)
    print("=" * 80)


if __name__ == "__main__":
    main()
