"""End-to-end demonstration for the astrophysics/space specialization:
train a PINN on the `space_debris_cw_relative_motion` preset (Clohessy-
Wiltshire relative motion -- the standard tool for space-debris
conjunction assessment and proximity operations) and compare against the
exact closed-form solution.

Same pure-IVP pitfall as `05_kepler_orbit_validation.py`, confirmed here
independently for a LINEAR ODE system (Kepler's is nonlinear): physics
residual + initial condition alone converged loss to ~1e-12 while
position RMSE was still ~360% of the trajectory's own scale -- proving
this failure mode is not specific to Kepler's nonlinearity. A second,
CW-specific pitfall was found and fixed while building this script: the
along-track coordinate y(t) in Hill's equations has a genuine SECULAR
(linearly-growing-with-time) term (6*(sin(nt)-nt)*x0), so it reaches
~-12 km over one period while x and z stay within +/-1.5 km -- a single
shared position-normalization scale (as used successfully for Kepler's
more uniformly-bounded x/y) badly under-scales y and stalls convergence
around 17% RMSE; per-axis scaling matched to each coordinate's actual
range fixes it.

Run: python examples/pde_environment/06_space_debris_cw_validation.py
Output: examples/pde_environment/_runs_kepler/space_debris_cw_validation.png
"""
from __future__ import annotations

import copy
import dataclasses
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn

import pinneapple_neural.architectures  # noqa: F401
from pinneapple_neural.architectures.registry import ModelRegistry
from pinneapple_physics.pde_environment.conditions import DataConstraint
from pinneapple_physics.pinn_solver.compiler.compile import compile_problem
from pinneapple_physics.pde_environment.presets.astrophysics import (
    space_debris_cw_relative_motion,
    _cw_exact_state,
)


def _cw_exact_velocity(t, n, x0, y0, z0, vx0, vy0, vz0):
    """Analytic derivative of `_cw_exact_state`'s closed form (not
    re-exported from astrophysics.py -- reference/data-generation-only
    helper, matching the module's own convention of doing this inline in
    _kepler_exact_torch/_cw_exact_torch for the Tier B tests)."""
    nt = n * t
    s, c = np.sin(nt), np.cos(nt)
    vx = 3 * n * s * x0 + c * vx0 + 2 * s * vy0
    vy = 6 * n * (c - 1) * x0 + 2 * s * vx0 + (4 * c - 3) * vy0
    vz = -z0 * n * s + c * vz0
    return vx, vy, vz


class _Nondimensionalized(nn.Module):
    def __init__(self, net, period, scales):
        super().__init__()
        self.net = net
        self.period = period
        self.scales = scales  # (x,y,z,vx,vy,vz)

    def forward(self, t):
        y = self.net(t / self.period)
        if hasattr(y, "y"):
            y = y.y
        cols = [y[:, i:i + 1] * s for i, s in enumerate(self.scales)]
        return torch.cat(cols, dim=1)


def main():
    n_mm = 0.0011  # mean motion, ~ISS altitude
    x0, y0, z0, vx0, vy0, vz0 = 1.0, 0.0, 0.2, 0.0, -0.0015, 0.0008
    base_spec = space_debris_cw_relative_motion(n=n_mm, x0=x0, y0=y0, z0=z0, vx0=vx0, vy0=vy0, vz0=vz0)
    period = base_spec.domain_bounds["t"][1]

    # Per-axis scales matched to this trajectory's actual range (see
    # module docstring -- y's secular drift makes a single shared scale
    # badly wrong for it).
    x_scale, y_scale, z_scale = 2.0, 15.0, 1.0
    vx_scale, vy_scale, vz_scale = n_mm * x_scale, n_mm * y_scale, n_mm * z_scale
    scales = (x_scale, y_scale, z_scale, vx_scale, vy_scale, vz_scale)

    n_data = 12
    t_data_np = np.linspace(0.0, period, n_data, endpoint=False).astype(np.float32)[:, None]
    exact_data = _cw_exact_state(t_data_np[:, 0], n_mm, x0, y0, z0, vx0, vy0, vz0)
    vx_ex, vy_ex, vz_ex = _cw_exact_velocity(t_data_np[:, 0], n_mm, x0, y0, z0, vx0, vy0, vz0)
    y_data_np = np.stack(
        [exact_data["x"], exact_data["y"], exact_data["z"], vx_ex, vy_ex, vz_ex], axis=1
    ).astype(np.float32)
    t_data = torch.tensor(t_data_np, requires_grad=True)
    y_data = torch.tensor(y_data_np)

    data_cond = DataConstraint(name="tracking_data", fields=("x", "y", "z", "vx", "vy", "vz"),
                                selector_type="all", weight=15.0)
    spec = dataclasses.replace(base_spec, conditions=base_spec.conditions + (data_cond,))
    loss_fn = compile_problem(spec)

    torch.manual_seed(0)
    raw_net = ModelRegistry.build("modified_mlp", in_dim=1, out_dim=6, hidden_dim=64, n_layers=4)
    model = _Nondimensionalized(raw_net, period, scales)

    n_epochs = 3000
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-6)

    n_ic = 500
    ic_values = {"x": x0, "y": y0, "z": z0, "vx": vx0, "vy": vy0, "vz": vz0}
    y_ic_fixed = torch.tensor(np.tile(np.array(list(ic_values.values()), dtype=np.float32), (n_ic, 1)))
    ic_mask = torch.ones(n_ic, dtype=torch.bool)
    data_mask = torch.ones(n_data, dtype=torch.bool)

    print(f"Training space_debris_cw_relative_motion (n={n_mm} rad/s), period={period:.1f} s, "
          f"{n_data} sparse tracking-data anchors, {n_epochs} Adam epochs ...")
    best_loss = float("inf")
    best_state = None
    t0 = time.time()
    for epoch in range(n_epochs):
        opt.zero_grad()
        t_col = (torch.rand(1024, 1) * period).requires_grad_(True)
        t_ic = torch.zeros(n_ic, 1, requires_grad=True)
        batch = {
            "x_col": t_col, "ctx": {},
            "x_bc": torch.zeros((0, 1)), "y_bc": torch.zeros((0, 6)),
            "x_ic": t_ic, "y_ic": y_ic_fixed,
            "x_data": t_data, "y_data": y_data,
            **{f"mask_ic_{f}": ic_mask for f in ic_values},
            "mask_tracking_data": data_mask,
        }
        y_hat = model(t_col)
        out = loss_fn(model, y_hat, batch)
        loss = out["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        sched.step()

        lv = float(loss.detach())
        if lv < best_loss:
            best_loss = lv
            best_state = copy.deepcopy(model.state_dict())
        if epoch % 500 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:5d}  pde={float(out['pde']):.4g}  "
                  f"data={float(out.get('data_tracking_data', 0.0)):.4g}  total={lv:.4g}  "
                  f"({time.time() - t0:.0f}s)")

    print(f"Best training loss: {best_loss:.6g}")
    model.load_state_dict(best_state)

    n_eval = 400
    t_eval = np.linspace(0.0, period, n_eval).astype(np.float32)
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(t_eval[:, None])).numpy()
    exact = _cw_exact_state(t_eval, n_mm, x0, y0, z0, vx0, vy0, vz0)

    pos_err2 = (pred[:, 0] - exact["x"]) ** 2 + (pred[:, 1] - exact["y"]) ** 2 + (pred[:, 2] - exact["z"]) ** 2
    pos_rmse = float(np.sqrt(np.mean(pos_err2)))
    scale_ref = math.sqrt(x_scale ** 2 + y_scale ** 2 + z_scale ** 2)
    print(f"Position RMSE: {pos_rmse:.4f} km ({100 * pos_rmse / scale_ref:.3f}% of combined trajectory scale)")

    out_dir = Path(__file__).parent / "_runs_kepler"
    out_dir.mkdir(exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        ax = axes[0]
        ax.plot(exact["x"], exact["y"], "k-", lw=2, label="Exact (Clohessy-Wiltshire)")
        ax.plot(pred[:, 0], pred[:, 1], "r--", lw=1.5, label="PINN prediction")
        ax.plot(exact_data["x"], exact_data["y"], "b^", ms=6, label="Tracking-data anchors")
        ax.plot(0, 0, "gx", ms=10, mew=2, label="Chief satellite")
        ax.set_xlabel("x, radial (km)")
        ax.set_ylabel("y, along-track (km)")
        ax.set_title("Relative-motion trajectory (Hill frame)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        pos_err = np.sqrt(pos_err2)
        ax.plot(t_eval, pos_err, "b-")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Position error (km)")
        ax.set_title(f"RMSE={pos_rmse:.3f} km ({100 * pos_rmse / scale_ref:.2f}% of scale)")
        ax.grid(alpha=0.3)

        fig.suptitle("space_debris_cw_relative_motion: trained PINN vs. exact solution")
        fig.tight_layout()
        out_path = out_dir / "space_debris_cw_validation.png"
        fig.savefig(out_path, dpi=120)
        print(f"Saved plot to {out_path}")
    except ImportError:
        print("matplotlib not installed -- skipping plot, metrics above still apply.")


if __name__ == "__main__":
    main()
