"""End-to-end demonstration for the astrophysics specialization: train a
PINN on the `kepler_two_body_orbit` preset and compare the trained
network's predicted trajectory against the exact (Kepler's-equation)
solution.

This is the "did we actually reproduce it, not just implement a residual
that runs" check that AUDIT_REPORT.md's astrophysics section originally
flagged as not yet done when the preset/compiler kind were added:
tests/test_astrophysics_validation.py plugs the EXACT solution into the
compiled residual and confirms it's ~0 -- that proves the physics
implementation is correct, but never trains a network end-to-end and
checks ITS output against the truth. This script does that.

First attempt (physics residual + initial condition only, no interior
supervision) trained a PDE+IC loss down to ~0.001 -- looking converged --
while the actual trajectory RMSE was ~104% of the orbit's semi-major axis
(completely wrong shape). This is a well-known PINN failure mode for
pure-IVP problems: the IC only pins down the solution at a single point
(t=0), so a low residual+IC loss does not by itself imply the network
found the globally correct trajectory rather than some other, spurious
one consistent with a locally small residual. Fix used here: add a
handful of sparse "tracking data" points sampled from the exact solution
along the orbit (15 points over one period) as a `DataConstraint` --
which is not a workaround, it is literally what real orbit determination
already is (fitting dynamics to sparse radar/optical tracking
observations), so it is a physically honest way to frame this preset's
industrial use case, not just a numerical trick to make the demo work.

Run: python examples/pde_environment/05_kepler_orbit_validation.py
Output: examples/pde_environment/_runs_kepler/kepler_orbit_validation.png
        plus printed quantitative error metrics.
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

import pinneapple_neural.architectures  # noqa: F401  registers the model zoo
from pinneapple_neural.architectures.registry import ModelRegistry
from pinneapple_physics.pde_environment.conditions import DataConstraint
from pinneapple_physics.pinn_solver.compiler.compile import compile_problem
from pinneapple_physics.pde_environment.presets.astrophysics import (
    kepler_two_body_orbit,
    kepler_exact_state,
)


class _Nondimensionalized(nn.Module):
    """Wraps a raw network with input/output nondimensionalization using
    the preset's own physical scales (a, v_p): the raw network sees
    t/period (O(1)) and outputs O(1) values rescaled back to physical
    km / km-per-s. Neither `solve_pde` nor this training loop normalizes
    automatically -- for a problem at orbital-mechanics scale (thousands
    of km, several km/s) this is standard PINN practice, not a shortcut
    around the physics; the compiled residual itself still evaluates in
    real physical units on this wrapper's output."""

    def __init__(self, net: nn.Module, period: float, a: float, v_p: float):
        super().__init__()
        self.net = net
        self.period = period
        self.a = a
        self.v_p = v_p

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_n = t / self.period
        y = self.net(t_n)
        if hasattr(y, "y"):
            y = y.y
        x = y[:, 0:1] * self.a
        y_ = y[:, 1:2] * self.a
        vx = y[:, 2:3] * self.v_p
        vy = y[:, 3:4] * self.v_p
        return torch.cat([x, y_, vx, vy], dim=1)


def main():
    mu, a, e = 398600.4418, 8000.0, 0.15  # Earth GM, km^3/s^2; eccentric LEO-to-MEO-like orbit
    base_spec = kepler_two_body_orbit(mu=mu, a=a, e=e)
    period = base_spec.domain_bounds["t"][1]
    r_p = a * (1.0 - e)
    v_p = math.sqrt(mu * (2.0 / r_p - 1.0 / a))

    # Sparse "tracking data" anchors from the exact solution (see module
    # docstring): 15 points over one period, standing in for sparse
    # radar/optical orbit-determination observations.
    n_data = 15
    t_data_np = np.linspace(0.0, period, n_data, endpoint=False).astype(np.float32)[:, None]
    exact_data = kepler_exact_state(t_data_np[:, 0], mu=mu, a=a, e=e)
    y_data_np = np.stack(
        [exact_data["x"], exact_data["y"], exact_data["vx"], exact_data["vy"]], axis=1
    ).astype(np.float32)
    t_data = torch.tensor(t_data_np, requires_grad=True)
    y_data = torch.tensor(y_data_np)

    data_cond = DataConstraint(name="tracking_data", fields=("x", "y", "vx", "vy"),
                                selector_type="all", weight=15.0)
    spec = dataclasses.replace(base_spec, conditions=base_spec.conditions + (data_cond,))
    loss_fn = compile_problem(spec)

    torch.manual_seed(0)
    raw_net = ModelRegistry.build("modified_mlp", in_dim=1, out_dim=4, hidden_dim=64, n_layers=4)
    model = _Nondimensionalized(raw_net, period=period, a=a, v_p=v_p)

    n_epochs = 3000
    n_col = 1024
    n_ic = 500
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=1e-5)

    y_ic_fixed = torch.tensor(np.tile(np.array([r_p, 0.0, 0.0, v_p], dtype=np.float32), (n_ic, 1)))
    ic_mask = torch.ones(n_ic, dtype=torch.bool)
    data_mask = torch.ones(n_data, dtype=torch.bool)

    print(f"Training kepler_two_body_orbit (mu={mu}, a={a} km, e={e}), period={period:.1f} s, "
          f"{n_data} sparse tracking-data anchors, {n_epochs} Adam epochs ...")
    best_loss = float("inf")
    best_state = None
    t0 = time.time()
    for epoch in range(n_epochs):
        opt.zero_grad()
        t_col = (torch.rand(n_col, 1) * period).requires_grad_(True)
        t_ic = torch.zeros(n_ic, 1, requires_grad=True)
        batch = {
            "x_col": t_col, "ctx": {},
            "x_bc": torch.zeros((0, 1)), "y_bc": torch.zeros((0, 4)),
            "x_ic": t_ic, "y_ic": y_ic_fixed,
            "x_data": t_data, "y_data": y_data,
            "mask_ic_x": ic_mask, "mask_ic_y": ic_mask,
            "mask_ic_vx": ic_mask, "mask_ic_vy": ic_mask,
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
                  f"data={float(out.get('data_tracking_data', 0.0)):.4g}  "
                  f"total={lv:.4g}  lr={opt.param_groups[0]['lr']:.2e}  "
                  f"({time.time() - t0:.0f}s)")

    print(f"Best training loss: {best_loss:.6g}")
    model.load_state_dict(best_state)

    # Evaluate on a dense time grid and compare against the exact trajectory.
    n_eval = 400
    t_eval = np.linspace(0.0, period, n_eval).astype(np.float32)
    model.eval()
    with torch.no_grad():
        pred = model(torch.as_tensor(t_eval[:, None])).numpy()
    x_pred, y_pred, vx_pred, vy_pred = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]

    exact = kepler_exact_state(t_eval, mu=mu, a=a, e=e)
    x_ex, y_ex, vx_ex, vy_ex = exact["x"], exact["y"], exact["vx"], exact["vy"]

    pos_rmse = float(np.sqrt(np.mean((x_pred - x_ex) ** 2 + (y_pred - y_ex) ** 2)))
    vel_rmse = float(np.sqrt(np.mean((vx_pred - vx_ex) ** 2 + (vy_pred - vy_ex) ** 2)))
    pos_scale = a
    vel_scale = v_p
    print(f"Position RMSE: {pos_rmse:.3f} km ({100 * pos_rmse / pos_scale:.3f}% of semi-major axis)")
    print(f"Velocity RMSE: {vel_rmse:.5f} km/s ({100 * vel_rmse / vel_scale:.3f}% of perigee speed)")

    # Energy/angular-momentum conservation of the TRAINED network's own
    # output (a real physical-consistency check independent of the exact
    # reference trajectory): specific energy eps=0.5v^2-mu/r should be
    # constant at -mu/(2a); specific angular momentum h=x*vy-y*vx should
    # be constant.
    r_pred = np.sqrt(x_pred ** 2 + y_pred ** 2)
    v2_pred = vx_pred ** 2 + vy_pred ** 2
    eps_pred = 0.5 * v2_pred - mu / r_pred
    eps_exact = -mu / (2 * a)
    h_pred = x_pred * vy_pred - y_pred * vx_pred
    h_exact = math.sqrt(mu * a * (1 - e ** 2))
    print(f"Specific orbital energy: predicted mean={np.mean(eps_pred):.4f} std={np.std(eps_pred):.4f}, "
          f"exact={eps_exact:.4f}")
    print(f"Specific angular momentum: predicted mean={np.mean(h_pred):.4f} std={np.std(h_pred):.4f}, "
          f"exact={h_exact:.4f}")

    out_dir = Path(__file__).parent / "_runs_kepler"
    out_dir.mkdir(exist_ok=True)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

        ax = axes[0]
        ax.plot(x_ex, y_ex, "k-", lw=2, label="Exact (Kepler's equation)")
        ax.plot(x_pred, y_pred, "r--", lw=1.5, label="PINN prediction")
        ax.plot(exact_data["x"], exact_data["y"], "b^", ms=6, label="Tracking-data anchors")
        ax.plot(0, 0, "yo", ms=12, label="Focus (Earth)")
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")
        ax.set_title(f"Orbit trajectory (a={a} km, e={e})")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

        ax = axes[1]
        pos_err = np.sqrt((x_pred - x_ex) ** 2 + (y_pred - y_ex) ** 2)
        ax.plot(t_eval, pos_err, "b-")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("Position error (km)")
        ax.set_title(f"RMSE={pos_rmse:.2f} km ({100 * pos_rmse / pos_scale:.2f}% of a)")
        ax.grid(alpha=0.3)

        fig.suptitle("kepler_two_body_orbit: trained PINN vs. exact solution")
        fig.tight_layout()
        out_path = out_dir / "kepler_orbit_validation.png"
        fig.savefig(out_path, dpi=120)
        print(f"Saved plot to {out_path}")
    except ImportError:
        print("matplotlib not installed -- skipping plot, metrics above still apply.")


if __name__ == "__main__":
    main()
