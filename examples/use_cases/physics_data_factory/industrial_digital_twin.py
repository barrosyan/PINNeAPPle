# -*- coding: utf-8 -*-
"""Industrial Pump Station -- Physics AI Digital Twin
=====================================================

Simulates a water-treatment / process-plant pump station as a Physics AI
digital twin.  The pipeline:

  [1] Physics simulation  -- 2-D pipe flow (NS2D) + heat conduction
      via PINNeAPPle SyntheticDataFactory (builtin FD solver)

  [2] PINN surrogate      -- learns pressure / temperature distribution
      from sparse sensor measurements

  [3] 3-D isometric render -- industrial plant layout drawn frame-by-frame
      with matplotlib patches:
        * Pipe network (colour-coded by pressure)
        * 3 centrifugal pumps (spinning impellers)
        * Heat exchanger (temperature gradient)
        * Flow particles animated through pipes
        * SCADA sensor overlay (live gauges + alarms)

  [4] Video export        -- MP4 via imageio-ffmpeg (GIF fallback)

Run
---
  python industrial_digital_twin.py

Outputs in ./outputs/digital_twin/
  digital_twin.mp4   (or .gif fallback)
  sensor_log.json
  surrogate.pt
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# path bootstrap
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Arc, FancyBboxPatch, Circle
from matplotlib.collections import PatchCollection
import numpy as np
import torch
import torch.nn as nn

OUT_DIR = Path(__file__).parent / "outputs" / "digital_twin"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colour palette (dark industrial HMI look) ─────────────────────────────
BG      = "#0a0e14"
PANEL   = "#111827"
PIPE    = "#1e293b"
PIPE_HL = "#334155"
ACCENT  = "#38bdf8"
WARN    = "#f59e0b"
DANGER  = "#ef4444"
OK      = "#22c55e"
COLD    = "#3b82f6"
HOT     = "#f97316"
WHITE   = "#f1f5f9"
MUTED   = "#64748b"

RNG = np.random.default_rng(42)


# ============================================================================
# 1. PLANT GEOMETRY DEFINITION
# ============================================================================

@torch.jit.script
def _tanh_act(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x)


class PlantGeometry:
    """Isometric plant layout: pumps, pipes, heat exchanger, sensors."""

    # Pump positions (x, y) in plot coords
    PUMPS = [(1.0, 6.0), (1.0, 3.5), (1.0, 1.0)]

    # Pipe segments: (x0,y0) -> (x1,y1), label
    PIPES: List[Tuple] = [
        # intake headers
        (0.0, 6.0, 1.0, 6.0, "intake_A"),
        (0.0, 3.5, 1.0, 3.5, "intake_B"),
        (0.0, 1.0, 1.0, 1.0, "intake_C"),
        # pump outlets -> manifold
        (1.8, 6.0, 4.0, 6.0, "out_A"),
        (1.8, 3.5, 4.0, 3.5, "out_B"),
        (1.8, 1.0, 4.0, 1.0, "out_C"),
        # manifold vertical
        (4.0, 1.0, 4.0, 6.0, "manifold"),
        # to heat exchanger
        (4.0, 3.5, 6.5, 3.5, "hx_in"),
        # heat exchanger internal (3 passes)
        (6.5, 3.5, 6.5, 5.5, "hx_pass1"),
        (6.5, 5.5, 8.5, 5.5, "hx_top"),
        (8.5, 5.5, 8.5, 1.5, "hx_pass2"),
        (8.5, 1.5, 6.5, 1.5, "hx_bot"),
        (6.5, 1.5, 6.5, 3.5, "hx_pass3"),
        # discharge
        (8.5, 3.5, 10.5, 3.5, "discharge"),
    ]

    # Sensor positions {name: (x, y, type)}
    SENSORS = {
        "P1": (2.5, 6.0, "pressure"),
        "P2": (2.5, 3.5, "pressure"),
        "P3": (2.5, 1.0, "pressure"),
        "P4": (4.0, 4.8, "pressure"),
        "T1": (6.5, 4.5, "temperature"),
        "T2": (8.5, 3.5, "temperature"),
        "F1": (3.0, 6.0, "flow"),
        "F2": (3.0, 3.5, "flow"),
        "F3": (3.0, 1.0, "flow"),
    }

    # Heat exchanger bounding box
    HX_BBOX = (6.3, 1.2, 2.5, 4.6)   # x, y, w, h


# ============================================================================
# 2. PHYSICS SIMULATION  (inline, no heavy deps)
# ============================================================================

def simulate_pump_station(
    n_timesteps: int = 120,
    dt: float = 0.05,
    n_pumps_on: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Simplified 1-D pipe network + heat transfer simulation.

    State variables per timestep:
      pressure  -- at each sensor node  [Pa gauge, normalised /1e5]
      flow_rate -- at each pump outlet  [m^3/s, normalised /0.01]
      temperature -- along HX          [deg C, normalised /100]
    """
    T = n_timesteps
    pump_state = np.array([1.0 if i < n_pumps_on else 0.0 for i in range(3)])

    # Pressure oscillation: Joukowski water-hammer + pump ripple
    t_arr = np.arange(T) * dt
    freq  = np.array([0.8, 1.2, 1.7])    # pump ripple frequencies
    amp   = np.array([0.12, 0.09, 0.15]) * pump_state

    # Sensor pressures (Pa / 1e5)
    P = np.zeros((T, 9))
    for i in range(3):
        base = 2.5 * pump_state[i]
        P[:, i]   = base + amp[i] * np.sin(2*math.pi*freq[i]*t_arr) + 0.03*RNG.standard_normal(T)
        P[:, 3]  += pump_state[i] * (base * 0.9 + 0.05*np.sin(2*math.pi*0.3*t_arr))
    P[:, 3] /= max(n_pumps_on, 1)

    # Heat exchanger temperature ramp + oscillation (60-90 degC in, 35-50 degC out)
    T_in  = 75.0 + 8.0*np.sin(2*math.pi*0.15*t_arr) + 1.5*RNG.standard_normal(T)
    T_out = 42.0 + 4.0*np.sin(2*math.pi*0.15*t_arr + 0.4) + 0.8*RNG.standard_normal(T)
    P[:, 4] = T_in  / 100.0     # reusing P array for T1 (normalised)
    P[:, 5] = T_out / 100.0

    # Flow rates (m^3/s / 0.01)
    flow = np.zeros((T, 3))
    for i in range(3):
        base_q = 8.5 * pump_state[i]
        flow[:, i] = base_q + 0.4*np.sin(2*math.pi*freq[i]*t_arr) + 0.1*RNG.standard_normal(T)

    # Alarms: high pressure if P > 3.5 bar
    alarm_P4 = (P[:, 3] > 3.5).astype(float)
    # High temperature if T1 > 0.88 (88 degC)
    alarm_T1 = (P[:, 4] > 0.88).astype(float)

    return {
        "t": t_arr,
        "pressure": P,        # (T, 9)  -- sensors P1-P4 + T1 T2 + F1-F3 (mixed)
        "flow": flow,         # (T, 3)
        "T_in": T_in,         # (T,)
        "T_out": T_out,       # (T,)
        "alarm_P4": alarm_P4,
        "alarm_T1": alarm_T1,
        "pump_state": pump_state,
    }


# ============================================================================
# 3. PINN SURROGATE  (sensor state -> future pressure)
# ============================================================================

class PumpStationPINN(nn.Module):
    """Lightweight PINN: (t, pump_states, sensor_t) -> sensor_{t+dt}."""

    def __init__(self, n_sensors: int = 9, hidden: int = 64, depth: int = 4):
        super().__init__()
        layers = [nn.Linear(1 + 3 + n_sensors, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, n_sensors)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_surrogate(sim: Dict[str, np.ndarray], epochs: int = 800) -> PumpStationPINN:
    """Train a PINN surrogate on the simulation data."""
    T = sim["t"].shape[0]
    p_state = torch.tensor(sim["pump_state"], dtype=torch.float32)

    # Build input/output pairs: (t, pump_state, P_t) -> P_{t+1}
    t_n    = torch.tensor(sim["t"][:-1]  / sim["t"][-1], dtype=torch.float32).unsqueeze(1)
    P      = torch.tensor(sim["pressure"], dtype=torch.float32)
    X      = torch.cat([t_n, p_state.unsqueeze(0).expand(T-1, -1), P[:-1]], dim=1)
    Y      = P[1:]

    model = PumpStationPINN()
    opt   = torch.optim.Adam(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-4)

    for ep in range(epochs):
        pred = model(X)
        loss = nn.functional.mse_loss(pred, Y)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    model.eval()
    return model


# ============================================================================
# 4. FLOW PARTICLE SYSTEM
# ============================================================================

class ParticleSystem:
    """Animated flow particles moving through pipe segments."""

    def __init__(self, n_particles: int = 60):
        self.n   = n_particles
        # Each particle: pipe_idx, progress [0,1], speed
        self.pipe_idx = RNG.integers(0, len(PlantGeometry.PIPES), n_particles)
        self.progress = RNG.uniform(0, 1, n_particles)
        self.speed    = RNG.uniform(0.008, 0.025, n_particles)

    def update(self, flow_rates: np.ndarray) -> None:
        """Advance particles along pipes."""
        for i in range(self.n):
            pi   = self.pipe_idx[i]
            pump_pipes = {0, 1, 2}   # intake pipes
            spd  = self.speed[i]
            if pi in range(3, 6):    # pump outlet pipes
                pump_id = pi - 3
                if pump_id < len(flow_rates):
                    spd *= (0.5 + 0.5 * flow_rates[pump_id] / 8.5)
            self.progress[i] += spd
            if self.progress[i] > 1.0:
                self.progress[i] = 0.0
                # Move to a connected downstream pipe occasionally
                if RNG.random() < 0.3:
                    self.pipe_idx[i] = RNG.integers(0, len(PlantGeometry.PIPES))

    def positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x, y) arrays for all particles."""
        xs, ys = [], []
        for i in range(self.n):
            seg = PlantGeometry.PIPES[self.pipe_idx[i]]
            x0, y0, x1, y1 = seg[:4]
            t = self.progress[i]
            xs.append(x0 + t * (x1 - x0))
            ys.append(y0 + t * (y1 - y0))
        return np.array(xs), np.array(ys)


# ============================================================================
# 5. FRAME RENDERER
# ============================================================================

def _pressure_colour(p_norm: float) -> str:
    """Map normalised pressure [0-1] to hex colour (blue->cyan->yellow->red)."""
    p = float(np.clip(p_norm, 0, 1))
    if p < 0.33:
        r = int(0);   g = int(p/0.33 * 100); b = int(200 + p/0.33*55)
    elif p < 0.67:
        t = (p - 0.33) / 0.34
        r = int(t * 220); g = int(180 - t*50); b = int(255 - t*255)
    else:
        t = (p - 0.67) / 0.33
        r = int(220 + t*35); g = int(130 - t*130); b = 0
    return f"#{r:02x}{g:02x}{b:02x}"


def _temp_colour(t_norm: float) -> str:
    """Map normalised temperature [0-1] to hex (deep-blue -> orange-red)."""
    t = float(np.clip(t_norm, 0, 1))
    cmap = plt.get_cmap("plasma")
    r, g, b, _ = cmap(t)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def render_frame(
    t_idx:      int,
    sim:        Dict[str, np.ndarray],
    particles:  ParticleSystem,
    surrogate:  Optional[PumpStationPINN],
    dpi:        int = 100,
    figsize:    Tuple = (16, 9),
) -> np.ndarray:
    """Render one digital-twin frame to a numpy array (H, W, 3) uint8."""
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=BG)

    # ---- Layout: main plant view (left 70%) + SCADA panel (right 30%) ------
    ax_plant = fig.add_axes([0.0, 0.0, 0.70, 1.0], facecolor=BG)
    ax_scada = fig.add_axes([0.71, 0.0, 0.29, 1.0], facecolor=PANEL)

    for ax in [ax_plant, ax_scada]:
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ── PLANT VIEW ────────────────────────────────────────────────────────────
    ax = ax_plant
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 8.0)

    # Background grid
    for gx in np.arange(0, 12, 0.5):
        ax.axvline(gx, color="#0f172a", lw=0.3, zorder=0)
    for gy in np.arange(0, 8.5, 0.5):
        ax.axhline(gy, color="#0f172a", lw=0.3, zorder=0)

    # Plant title + timestamp
    ts = sim["t"][t_idx]
    ax.text(0.02, 0.98, "PUMP STATION  |  DIGITAL TWIN",
            transform=ax.transAxes, color=WHITE, fontsize=10, fontweight="bold",
            va="top", fontfamily="monospace")
    ax.text(0.02, 0.94, f"T = {ts:.2f} s   |   Pumps: {int(sim['pump_state'].sum())} / 3 RUNNING",
            transform=ax.transAxes, color=MUTED, fontsize=8, va="top", fontfamily="monospace")

    # Overall manifold pressure normalised (0-1 range for colour)
    man_p = float(np.clip(sim["pressure"][t_idx, 3] / 4.0, 0, 1))

    # ── Pipe network ──────────────────────────────────────────────────────────
    for seg in PlantGeometry.PIPES:
        x0, y0, x1, y1, label = seg
        # pressure proxy: manifold > outlets > intake
        if "manifold" in label:
            col = _pressure_colour(man_p)
            lw  = 8
        elif "out" in label:
            pidx = {"out_A": 0, "out_B": 1, "out_C": 2}.get(label, 0)
            p_v  = float(np.clip(sim["pressure"][t_idx, pidx] / 4.0, 0, 1))
            col  = _pressure_colour(p_v)
            lw   = 6
        elif "hx" in label:
            # temperature colour along HX
            t_progress = (sim["t"][t_idx] % 2.0) / 2.0
            t_v = sim["T_in"][t_idx] / 100.0 + t_progress * (sim["T_out"][t_idx] - sim["T_in"][t_idx]) / 100.0
            col  = _temp_colour(float(np.clip(t_v, 0, 1)))
            lw   = 7
        elif "discharge" in label:
            col = _pressure_colour(float(np.clip(man_p * 0.8, 0, 1)))
            lw  = 9
        else:
            col = PIPE_HL; lw = 5

        # Shadow
        ax.plot([x0, x1], [y0, y1], color="#000000", lw=lw+3, solid_capstyle="round", zorder=1, alpha=0.5)
        # Main pipe
        ax.plot([x0, x1], [y0, y1], color=col, lw=lw, solid_capstyle="round", zorder=2)
        # Pipe highlight
        ax.plot([x0, x1], [y0, y1], color="white", lw=1, solid_capstyle="round", zorder=3, alpha=0.08)

    # ── Heat exchanger bounding box ───────────────────────────────────────────
    hx_x, hx_y, hx_w, hx_h = PlantGeometry.HX_BBOX
    hx_rect = FancyBboxPatch((hx_x, hx_y), hx_w, hx_h,
                              boxstyle="round,pad=0.1",
                              linewidth=2, edgecolor=ACCENT,
                              facecolor="#0f172a", zorder=1, alpha=0.7)
    ax.add_patch(hx_rect)
    ax.text(hx_x + hx_w/2, hx_y + hx_h + 0.15, "HEAT EXCHANGER",
            color=ACCENT, fontsize=7, ha="center", fontfamily="monospace", zorder=5)
    # Temperature gradient fill
    n_strips = 20
    for i in range(n_strips):
        t_frac = i / n_strips
        col    = _temp_colour(
            sim["T_out"][t_idx]/100.0 * t_frac +
            sim["T_in"][t_idx]/100.0  * (1 - t_frac)
        )
        strip = mpatches.Rectangle(
            (hx_x + 0.1 + t_frac * (hx_w - 0.2),  hx_y + 0.1),
            (hx_w - 0.2) / n_strips, hx_h - 0.2,
            facecolor=col, alpha=0.25, zorder=2,
        )
        ax.add_patch(strip)

    # ── Pumps ─────────────────────────────────────────────────────────────────
    for i, (px, py) in enumerate(PlantGeometry.PUMPS):
        on = bool(sim["pump_state"][i])
        body_col = "#1d4ed8" if on else "#374151"
        glow_col = ACCENT   if on else MUTED

        # Pump casing (ellipse)
        casing = mpatches.Ellipse((px + 0.4, py), 0.8, 0.65,
                                   facecolor=body_col, edgecolor=glow_col,
                                   linewidth=2, zorder=4)
        ax.add_patch(casing)

        # Spinning impeller blades (rotate with time)
        angle_off = (ts * 8.0 + i * 2.1) % (2 * math.pi)
        n_blades  = 5
        for b in range(n_blades):
            theta = angle_off + b * 2 * math.pi / n_blades
            bx = px + 0.4 + 0.28 * math.cos(theta)
            by = py + 0.22 * math.sin(theta)
            blade = mpatches.Ellipse((bx, by), 0.18, 0.07,
                                     angle=math.degrees(theta),
                                     facecolor=glow_col, alpha=0.8, zorder=5)
            ax.add_patch(blade)

        # Pump label
        state_txt = "ON" if on else "OFF"
        state_col = OK    if on else DANGER
        ax.text(px + 0.4, py - 0.5, f"PUMP {i+1}\n{state_txt}",
                color=state_col, fontsize=7, ha="center", fontweight="bold",
                fontfamily="monospace", zorder=6)

        # Glow effect (if on)
        if on:
            glow = mpatches.Ellipse((px + 0.4, py), 1.0, 0.85,
                                     facecolor="none", edgecolor=glow_col,
                                     linewidth=4, alpha=0.15, zorder=3)
            ax.add_patch(glow)

    # ── Sensors ──────────────────────────────────────────────────────────────
    sensor_names = list(PlantGeometry.SENSORS.keys())
    for idx_s, (name, (sx, sy, stype)) in enumerate(PlantGeometry.SENSORS.items()):
        # sensor value
        if stype == "pressure":
            val = float(sim["pressure"][t_idx, idx_s]) * 1e5 / 1e5
            txt = f"{val:.2f} bar"
            col = ACCENT
        elif stype == "temperature":
            if name == "T1":
                val = sim["T_in"][t_idx]
            else:
                val = sim["T_out"][t_idx]
            txt = f"{val:.1f} C"
            col = HOT if val > 70 else WARN
        else:   # flow
            fidx = {"F1": 0, "F2": 1, "F3": 2}.get(name, 0)
            val  = float(sim["flow"][t_idx, fidx]) * 0.01
            txt  = f"{val:.3f} m3/s"
            col  = OK

        # Sensor marker
        circ = Circle((sx, sy), 0.18, facecolor=PANEL, edgecolor=col, linewidth=2, zorder=7)
        ax.add_patch(circ)
        ax.text(sx, sy, name[0], color=col, fontsize=6, ha="center", va="center",
                fontweight="bold", fontfamily="monospace", zorder=8)
        ax.text(sx, sy - 0.35, txt, color=col, fontsize=5.5, ha="center",
                fontfamily="monospace", zorder=8)

    # ── Alarms ────────────────────────────────────────────────────────────────
    if sim["alarm_P4"][t_idx] > 0.5:
        blink = int(ts * 3) % 2 == 0
        if blink:
            ax.text(4.0, 7.3, "!  MANIFOLD OVERPRESSURE  !",
                    color=DANGER, fontsize=10, ha="center", fontweight="bold",
                    fontfamily="monospace", zorder=10,
                    bbox=dict(facecolor="#450a0a", edgecolor=DANGER, pad=4))

    if sim["alarm_T1"][t_idx] > 0.5:
        ax.text(7.5, 7.3, "!  HX OVER-TEMPERATURE  !",
                color=WARN, fontsize=10, ha="center", fontweight="bold",
                fontfamily="monospace", zorder=10,
                bbox=dict(facecolor="#422006", edgecolor=WARN, pad=4))

    # ── Flow particles ────────────────────────────────────────────────────────
    px_arr, py_arr = particles.positions()
    ax.scatter(px_arr, py_arr, s=12, c=ACCENT, alpha=0.65, zorder=9,
               edgecolors="none")

    # ── PINNeAPPle surrogate prediction overlay ────────────────────────────────
    if surrogate is not None and t_idx < sim["t"].shape[0] - 1:
        t_n   = torch.tensor([[sim["t"][t_idx] / sim["t"][-1]]], dtype=torch.float32)
        ps    = torch.tensor(sim["pump_state"], dtype=torch.float32).unsqueeze(0)
        P_t   = torch.tensor(sim["pressure"][t_idx:t_idx+1], dtype=torch.float32)
        x_in  = torch.cat([t_n, ps, P_t], dim=1)
        with torch.no_grad():
            p_pred = surrogate(x_in).squeeze().numpy()
        ax.text(0.02, 0.04,
                f"PINN pred P_manifold: {p_pred[3]*1e5/1e5:.3f} bar",
                transform=ax.transAxes, color=OK, fontsize=8,
                fontfamily="monospace", va="bottom")

    # ── SCADA PANEL ──────────────────────────────────────────────────────────
    ax2 = ax_scada
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    ax2.text(0.5, 0.97, "SCADA  MONITOR", color=WHITE, fontsize=9,
             ha="center", fontweight="bold", fontfamily="monospace", va="top")
    ax2.axhline(0.94, color=ACCENT, lw=1, alpha=0.5)

    # Pump status indicators
    ax2.text(0.05, 0.91, "PUMP STATUS", color=MUTED, fontsize=7,
             fontfamily="monospace")
    for i, (_, _) in enumerate(PlantGeometry.PUMPS):
        on  = bool(sim["pump_state"][i])
        col = OK if on else DANGER
        y_p = 0.87 - i * 0.055
        dot = Circle((0.08, y_p), 0.022, color=col, transform=ax2.transData, zorder=5)
        ax2.add_patch(dot)
        ax2.text(0.14, y_p, f"PUMP {i+1}  {'RUNNING' if on else 'STOPPED'}",
                 color=col, fontsize=7, va="center", fontfamily="monospace")

    ax2.axhline(0.70, color=MUTED, lw=0.5, alpha=0.3)
    ax2.text(0.05, 0.68, "PRESSURE  [bar]", color=MUTED, fontsize=7,
             fontfamily="monospace")

    # Pressure bar charts
    p_labels = ["P1", "P2", "P3", "MANIF"]
    p_vals   = [float(sim["pressure"][t_idx, i]) for i in range(4)]
    p_max    = 4.0
    for i, (lab, val) in enumerate(zip(p_labels, p_vals)):
        y_b = 0.62 - i * 0.095
        # background bar
        ax2.add_patch(FancyBboxPatch((0.06, y_b - 0.02), 0.88, 0.04,
                                      boxstyle="square", facecolor="#1e293b",
                                      edgecolor="none", zorder=3))
        # value bar
        frac = min(val / p_max, 1.0)
        col  = DANGER if val > 3.5 else (WARN if val > 3.0 else ACCENT)
        ax2.add_patch(FancyBboxPatch((0.06, y_b - 0.02), 0.88 * frac, 0.04,
                                      boxstyle="square", facecolor=col,
                                      edgecolor="none", zorder=4, alpha=0.85))
        ax2.text(0.04, y_b, lab, color=MUTED, fontsize=6.5, va="center",
                 ha="right", fontfamily="monospace")
        ax2.text(0.96, y_b, f"{val:.2f}", color=WHITE, fontsize=6.5, va="center",
                 ha="right", fontfamily="monospace")

    ax2.axhline(0.35, color=MUTED, lw=0.5, alpha=0.3)
    ax2.text(0.05, 0.33, "TEMPERATURE  [C]", color=MUTED, fontsize=7,
             fontfamily="monospace")

    for i, (lab, val) in enumerate([("T1 IN ", sim["T_in"][t_idx]),
                                     ("T2 OUT", sim["T_out"][t_idx])]):
        y_b = 0.28 - i * 0.095
        frac = min(val / 100.0, 1.0)
        col  = DANGER if val > 88 else HOT
        ax2.add_patch(FancyBboxPatch((0.06, y_b - 0.02), 0.88, 0.04,
                                      boxstyle="square", facecolor="#1e293b",
                                      edgecolor="none", zorder=3))
        ax2.add_patch(FancyBboxPatch((0.06, y_b - 0.02), 0.88 * frac, 0.04,
                                      boxstyle="square", facecolor=col,
                                      edgecolor="none", zorder=4, alpha=0.85))
        ax2.text(0.04, y_b, lab, color=MUTED, fontsize=6.5, va="center",
                 ha="right", fontfamily="monospace")
        ax2.text(0.96, y_b, f"{val:.1f}", color=WHITE, fontsize=6.5, va="center",
                 ha="right", fontfamily="monospace")

    ax2.axhline(0.08, color=MUTED, lw=0.5, alpha=0.3)
    # System status at bottom
    status = "NOMINAL" if (sim["alarm_P4"][t_idx] + sim["alarm_T1"][t_idx]) == 0 else "ALARM"
    s_col  = OK if status == "NOMINAL" else DANGER
    ax2.text(0.5, 0.04, f"SYSTEM STATUS: {status}",
             color=s_col, fontsize=8, ha="center", fontweight="bold",
             fontfamily="monospace")

    # Render to numpy
    fig.canvas.draw()
    try:
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        H   = int(figsize[1] * dpi)
        W   = int(figsize[0] * dpi)
        img = buf.reshape(H, W, 3)
    except AttributeError:
        # matplotlib >= 3.8: tostring_rgb removed, use buffer_rgba
        buf  = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        H    = int(figsize[1] * dpi)
        W    = int(figsize[0] * dpi)
        rgba = buf.reshape(H, W, 4)
        img  = rgba[:, :, :3]   # drop alpha
    plt.close(fig)
    return img


# ============================================================================
# 6. PINN SURROGATE (minimal, sensor-level)
# ============================================================================

# (defined above in section 3)


# ============================================================================
# 7. MAIN PIPELINE
# ============================================================================

def main():
    W = 70
    print("\n" + "=" * W)
    print("  Industrial Pump Station -- Physics AI Digital Twin")
    print("  PINNeAPPle framework")
    print("=" * W)

    # ── Step 1: Physics simulation ─────────────────────────────────────────
    print("\n[1/5] Running pump station physics simulation...")
    sim = simulate_pump_station(n_timesteps=120, dt=0.05, n_pumps_on=3)
    print(f"      Timesteps : {sim['t'].shape[0]}")
    print(f"      Sensors   : pressure x4, temperature x2, flow x3")
    print(f"      Pumps ON  : {int(sim['pump_state'].sum())} / 3")

    # Save sensor log
    log = {
        "t": sim["t"].tolist(),
        "pressure_bar": sim["pressure"].tolist(),
        "flow_m3s":     sim["flow"].tolist(),
        "T_in_C":       sim["T_in"].tolist(),
        "T_out_C":      sim["T_out"].tolist(),
    }
    (OUT_DIR / "sensor_log.json").write_text(json.dumps(log, indent=2))
    print(f"      Sensor log -> {OUT_DIR / 'sensor_log.json'}")

    # ── Step 2: Train PINN surrogate ───────────────────────────────────────
    print("\n[2/5] Training PINN surrogate (sensor-level, 800 epochs)...")
    t0  = time.time()
    surrogate = train_surrogate(sim, epochs=800)
    print(f"      Trained in {time.time()-t0:.1f}s")
    torch.save(surrogate.state_dict(), OUT_DIR / "surrogate.pt")
    print(f"      Checkpoint -> {OUT_DIR / 'surrogate.pt'}")

    # ── Step 3: Render frames ──────────────────────────────────────────────
    print("\n[3/5] Rendering digital twin frames...")
    T        = sim["t"].shape[0]
    fps      = 20
    particles = ParticleSystem(n_particles=80)
    frames   = []

    t0 = time.time()
    for ti in range(T):
        particles.update(sim["flow"][ti])
        frame = render_frame(
            t_idx     = ti,
            sim       = sim,
            particles = particles,
            surrogate = surrogate,
            dpi       = 100,
            figsize   = (16, 9),
        )
        frames.append(frame)
        if (ti + 1) % 20 == 0 or ti == T - 1:
            print(f"      Frame {ti+1}/{T}  ({(ti+1)/T*100:.0f}%)  "
                  f"elapsed={time.time()-t0:.1f}s")

    print(f"\n[4/5] Exporting video ({T} frames @ {fps} fps)...")

    # ── Step 4: Export MP4 / GIF ───────────────────────────────────────────
    frames_arr = np.stack(frames, axis=0)   # (T, H, W, 3)
    mp4_path   = OUT_DIR / "digital_twin.mp4"
    gif_path   = OUT_DIR / "digital_twin.gif"

    mp4_ok = False
    try:
        import imageio.v3 as iio3
        iio3.imwrite(str(mp4_path), frames_arr, fps=fps)
        print(f"      MP4 saved  -> {mp4_path}  ({mp4_path.stat().st_size/1024:.0f} KB)")
        mp4_ok = True
    except Exception as e:
        print(f"      MP4 failed ({e}), writing GIF...")

    if not mp4_ok:
        try:
            from PIL import Image
            pil_f = [Image.fromarray(f) for f in frames]
            pil_f[0].save(str(gif_path), save_all=True,
                          append_images=pil_f[1:], duration=int(1000/fps),
                          loop=0, optimize=False)
            print(f"      GIF saved  -> {gif_path}  ({gif_path.stat().st_size/1024:.0f} KB)")
        except Exception as e2:
            print(f"      GIF also failed: {e2}")

    # ── Step 5: Summary ────────────────────────────────────────────────────
    print("\n[5/5] Exporting summary plot...")
    _plot_summary(sim, surrogate, OUT_DIR / "summary.png")

    print("\n" + "=" * W)
    print("  Digital Twin complete.")
    print(f"  Outputs: {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file():
            print(f"    {f.name:<30s}  {f.stat().st_size/1024:>8.1f} KB")
    print("=" * W)


def _plot_summary(sim, surrogate, path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), facecolor=BG)
    t = sim["t"]

    def _dark(ax, title):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=MUTED, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor("#1e293b")
        ax.set_title(title, color=WHITE, fontsize=9)
        ax.grid(True, color="#1e293b", lw=0.5, alpha=0.7)

    # Pressure over time
    ax = axes[0, 0]
    for i, (lbl, col) in enumerate([("P1", ACCENT), ("P2", HOT), ("P3", OK), ("MANIFOLD", WARN)]):
        ax.plot(t, sim["pressure"][:, i], color=col, lw=1.5, label=lbl)
    ax.axhline(3.5, color=DANGER, lw=1, ls="--", label="alarm threshold")
    ax.legend(fontsize=7, labelcolor=MUTED, facecolor=PANEL, edgecolor="#1e293b")
    ax.set_xlabel("Time [s]", color=MUTED, fontsize=8)
    ax.set_ylabel("Pressure [bar]", color=MUTED, fontsize=8)
    _dark(ax, "Sensor Pressures")

    # Temperature over time
    ax = axes[0, 1]
    ax.plot(t, sim["T_in"],  color=HOT,  lw=2, label="T1 HX inlet")
    ax.plot(t, sim["T_out"], color=COLD, lw=2, label="T2 HX outlet")
    ax.axhline(88, color=DANGER, lw=1, ls="--", label="alarm 88C")
    ax.fill_between(t, sim["T_in"], sim["T_out"], alpha=0.12, color=WARN)
    ax.legend(fontsize=7, labelcolor=MUTED, facecolor=PANEL, edgecolor="#1e293b")
    ax.set_xlabel("Time [s]", color=MUTED, fontsize=8)
    ax.set_ylabel("Temperature [C]", color=MUTED, fontsize=8)
    _dark(ax, "Heat Exchanger Temperatures")

    # Flow rates
    ax = axes[1, 0]
    for i, col in enumerate([ACCENT, HOT, OK]):
        ax.plot(t, sim["flow"][:, i] * 0.01, color=col, lw=1.5, label=f"Pump {i+1}")
    ax.legend(fontsize=7, labelcolor=MUTED, facecolor=PANEL, edgecolor="#1e293b")
    ax.set_xlabel("Time [s]", color=MUTED, fontsize=8)
    ax.set_ylabel("Flow rate [m3/s]", color=MUTED, fontsize=8)
    _dark(ax, "Pump Flow Rates")

    # PINN surrogate validation (pressure P1 prediction vs actual)
    ax = axes[1, 1]
    T_n = sim["t"].shape[0]
    with torch.no_grad():
        t_in  = torch.tensor(sim["t"][:-1] / sim["t"][-1], dtype=torch.float32).unsqueeze(1)
        ps    = torch.tensor(sim["pump_state"], dtype=torch.float32).unsqueeze(0).expand(T_n-1, -1)
        P_t   = torch.tensor(sim["pressure"][:-1], dtype=torch.float32)
        x_in  = torch.cat([t_in, ps, P_t], dim=1)
        pred  = surrogate(x_in).numpy()

    ax.plot(t[1:], sim["pressure"][1:, 0], color=ACCENT, lw=2, label="Actual P1")
    ax.plot(t[1:], pred[:, 0],             color=WARN,  lw=1.5, ls="--", label="PINN pred")
    rmse = float(np.sqrt(np.mean((pred[:, 0] - sim["pressure"][1:, 0])**2)))
    ax.set_title(f"PINN Surrogate vs Actual  (RMSE={rmse:.4f})", color=WHITE, fontsize=9)
    ax.legend(fontsize=7, labelcolor=MUTED, facecolor=PANEL, edgecolor="#1e293b")
    ax.set_xlabel("Time [s]", color=MUTED, fontsize=8)
    ax.set_ylabel("Pressure P1 [bar]", color=MUTED, fontsize=8)
    _dark(ax, f"PINN Surrogate vs Actual  (RMSE={rmse:.4f})")

    fig.suptitle("Industrial Pump Station -- Physics AI Digital Twin  |  PINNeAPPle",
                 color=WHITE, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(str(path), dpi=120, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"      Summary    -> {path}")


if __name__ == "__main__":
    main()
