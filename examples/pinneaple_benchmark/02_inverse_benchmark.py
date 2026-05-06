"""
02_inverse_benchmark.py — PINNeAPPle Inverse Problem Benchmark

"How many data points do I actually need to recover a physical parameter?"
This script answers that question before you commit to expensive training.

Two canonical inverse problems
-------------------------------
  (1) Heat 1D+t  — recover unknown diffusivity  k
      u_t = k · u_xx,  true k  = 0.40,  initial guess = 0.10
      Exact: u = sin(πx) · exp(−k π² t)

  (2) Burgers 1D+t — recover unknown viscosity  ν
      u_t + u·u_x = ν·u_xx,  true ν ≈ 0.003183,  initial guess = 0.020
      Reference: scipy Radau method-of-lines

Strategy
--------
  For each problem, sweep N_data ∈ {25, 50, 100, 200}.
  Train PINN jointly with a learnable log-parameterised scalar.
  Measure:
    • parameter recovery error  |p_ident − p_true| / p_true  (%)
    • field L2 relative error   ||u_pred − u_ref|| / ||u_ref||

Plots (saved to outputs/03_inverse_benchmark.png)
-------------------------------------------------
  Col 0: reference field + observation locations
  Col 1: parameter convergence curve during training (N_data=100)
  Col 2: data-efficiency — param error vs N_data (bar chart)
  Col 3: best prediction vs reference at end of training

  Two rows: row 0 = Heat,  row 1 = Burgers
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================
# Settings
# ============================================================

EPOCHS    = 3000
LR        = 2e-3
N_COL     = 2000   # collocation points per epoch
N_BC      = 300
N_IC      = 300
W_BC      = 10.0
W_IC      = 10.0
W_DATA    = 20.0   # strong weight on observations for parameter recovery
NOISE     = 0.01   # relative noise level on observations
N_DATA_SWEEP = [25, 50, 100, 200]

HIDDEN = [64, 64, 64, 64]


# ============================================================
# Utilities (shared with 01_pde_comparison)
# ============================================================

def _pred(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, torch.Tensor):
        return out
    return out.y


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y.sum(), x, create_graph=True, retain_graph=True
    )[0]


# ============================================================
# Inverse PINN model
#   VanillaPINN (field predictor) + one learnable scalar (unknown param)
#   Log-parameterisation keeps the scalar strictly positive.
# ============================================================

class InversePINNModel(nn.Module):
    """
    Wraps VanillaPINN and adds one learnable physics parameter.

    The parameter is stored as log(p) so it stays positive during
    optimisation.  The true_val is stored only for progress logging.
    """

    def __init__(
        self,
        in_dim:     int,
        out_dim:    int,
        hidden:     List[int],
        param_name: str,
        init_guess: float,
        true_val:   float,
    ) -> None:
        super().__init__()
        from pinneaple_neural.architectures.pinns.vanilla import VanillaPINN
        self.net        = VanillaPINN(in_dim=in_dim, out_dim=out_dim, hidden=hidden)
        self._log_p     = nn.Parameter(torch.tensor(math.log(float(init_guess))))
        self.param_name = param_name
        self.true_val   = true_val

    @property
    def param(self) -> torch.Tensor:
        return torch.exp(self._log_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _pred(self.net, x)


# ============================================================
# Problem 1 — Inverse Heat (recover k)
# ============================================================

class InverseHeatProblem:
    name       = "Heat: recover k"
    in_dim     = 2       # (x, t)
    out_dim    = 1
    true_k     = 0.40
    init_guess = 0.10
    x_end      = 1.0
    t_end      = 0.50

    @staticmethod
    def exact(x: np.ndarray, t: np.ndarray, k: float = 0.40) -> np.ndarray:
        return np.sin(np.pi * x) * np.exp(-k * np.pi**2 * t)

    def sample_interior(self, n: int) -> torch.Tensor:
        x = torch.FloatTensor(n, 1).uniform_(0.0, self.x_end)
        t = torch.FloatTensor(n, 1).uniform_(0.0, self.t_end)
        return torch.cat([x, t], dim=1)

    def sample_bc(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.FloatTensor(n, 1).uniform_(0.0, self.t_end)
        pts = torch.cat([
            torch.cat([torch.zeros(n, 1), t], dim=1),
            torch.cat([torch.ones(n, 1),  t], dim=1),
        ], dim=0)
        return pts, torch.zeros(2 * n, 1)

    def sample_ic(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x   = torch.FloatTensor(n, 1).uniform_(0.0, self.x_end)
        pts = torch.cat([x, torch.zeros(n, 1)], dim=1)
        return pts, torch.sin(math.pi * x)

    def generate_observations(
        self, n: int, noise: float = NOISE
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x   = torch.FloatTensor(n, 1).uniform_(0.0, self.x_end)
        t   = torch.FloatTensor(n, 1).uniform_(0.0, self.t_end)
        pts = torch.cat([x, t], dim=1)
        u   = torch.tensor(
            self.exact(x.numpy(), t.numpy(), k=self.true_k), dtype=torch.float32
        )
        u   = u + noise * u.abs().max() * torch.randn_like(u)
        return pts, u

    def physics_residual(
        self, model: InversePINNModel, pts: torch.Tensor
    ) -> torch.Tensor:
        k  = model.param
        u  = model(pts)
        d1 = _grad(u, pts)
        u_t = d1[:, 1:2]
        u_x = d1[:, 0:1]
        u_xx = _grad(u_x, pts)[:, 0:1]
        return u_t - k * u_xx

    def evaluate(
        self, model: InversePINNModel
    ) -> Tuple[float, float]:
        """Return (L2 field error, param recovery %)."""
        Ng   = 60
        x    = np.linspace(0, self.x_end, Ng)
        t    = np.linspace(0, self.t_end, Ng)
        X, T = np.meshgrid(x, t)
        pts  = torch.tensor(
            np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32
        )
        with torch.no_grad():
            u_pred = model(pts).cpu().numpy().flatten()
        u_true = self.exact(X.flatten(), T.flatten())
        l2_rel = float(
            np.sqrt(np.mean((u_pred - u_true)**2))
            / (np.sqrt(np.mean(u_true**2)) + 1e-12)
        )
        k_err_pct = abs(
            float(model.param.item()) - self.true_val
        ) / self.true_val * 100.0
        return l2_rel, k_err_pct

    def ref_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(x, t, u) for pcolormesh."""
        Ng   = 80
        x    = np.linspace(0, self.x_end, Ng)
        t    = np.linspace(0, self.t_end, Ng)
        X, T = np.meshgrid(x, t)
        return x, t, self.exact(X, T)          # (Ng,) (Ng,) (Ng, Ng)


# ============================================================
# Problem 2 — Inverse Burgers (recover ν)
# ============================================================

class InverseBurgersProblem:
    name       = "Burgers: recover ν"
    in_dim     = 2
    out_dim    = 1
    true_nu    = 0.01 / math.pi
    init_guess = 0.020

    def __init__(self) -> None:
        self._build_reference()

    def _build_reference(self) -> None:
        try:
            from scipy.integrate import solve_ivp
            Nx  = 128
            x   = np.linspace(-1.0, 1.0, Nx)
            dx  = x[1] - x[0]
            nu  = self.true_nu

            def rhs(t, u):
                u = u.copy(); u[0] = u[-1] = 0.0
                u_fwd = (np.roll(u, -1) - u) / dx
                u_bwd = (u - np.roll(u, 1)) / dx
                u_x   = np.where(u >= 0, u_bwd, u_fwd)
                u_x[0] = u_x[-1] = 0.0
                u_xx  = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
                u_xx[0] = u_xx[-1] = 0.0
                return -u * u_x + nu * u_xx

            sol = solve_ivp(rhs, [0.0, 1.0], -np.sin(np.pi * x),
                            t_eval=np.linspace(0, 1, 101),
                            method="Radau", rtol=1e-5, atol=1e-7)
            self._x  = x
            self._t  = sol.t
            self._u  = sol.y.T       # (n_t, Nx)
        except Exception:
            Nx, Nt   = 128, 101
            x        = np.linspace(-1.0, 1.0, Nx)
            t        = np.linspace( 0.0, 1.0, Nt)
            X, T     = np.meshgrid(x, t)
            self._x  = x
            self._t  = t
            self._u  = -np.sin(np.pi * X) * np.exp(-self.true_nu * np.pi**2 * T)

    def sample_interior(self, n: int) -> torch.Tensor:
        x = torch.FloatTensor(n, 1).uniform_(-1.0, 1.0)
        t = torch.FloatTensor(n, 1).uniform_( 0.0, 1.0)
        return torch.cat([x, t], dim=1)

    def sample_bc(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        t   = torch.FloatTensor(n, 1).uniform_(0.0, 1.0)
        pts = torch.cat([
            torch.cat([torch.full((n, 1), -1.0), t], dim=1),
            torch.cat([torch.full((n, 1),  1.0), t], dim=1),
        ], dim=0)
        return pts, torch.zeros(2 * n, 1)

    def sample_ic(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x   = torch.FloatTensor(n, 1).uniform_(-1.0, 1.0)
        pts = torch.cat([x, torch.zeros(n, 1)], dim=1)
        return pts, -torch.sin(math.pi * x)

    def generate_observations(
        self, n: int, noise: float = NOISE
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample random (x,t) from the reference grid and add noise."""
        Nt, Nx  = self._u.shape
        idx_t   = np.random.randint(0, Nt, n)
        idx_x   = np.random.randint(0, Nx, n)
        t_obs   = self._t[idx_t].reshape(n, 1)
        x_obs   = self._x[idx_x].reshape(n, 1)
        u_obs   = self._u[idx_t, idx_x].reshape(n, 1)
        pts     = torch.tensor(np.hstack([x_obs, t_obs]), dtype=torch.float32)
        u_t     = torch.tensor(u_obs, dtype=torch.float32)
        u_t     = u_t + noise * u_t.abs().max() * torch.randn_like(u_t)
        return pts, u_t

    def physics_residual(
        self, model: InversePINNModel, pts: torch.Tensor
    ) -> torch.Tensor:
        nu  = model.param
        u   = model(pts)
        du  = _grad(u, pts)
        u_x = du[:, 0:1];  u_t = du[:, 1:2]
        u_xx = _grad(u_x, pts)[:, 0:1]
        return u_t + u * u_x - nu * u_xx

    def evaluate(self, model: InversePINNModel) -> Tuple[float, float]:
        Nt, Nx   = self._u.shape
        X, T     = np.meshgrid(self._x, self._t)
        pts      = torch.tensor(
            np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32
        )
        with torch.no_grad():
            u_pred = model(pts).cpu().numpy().flatten()
        u_true = self._u.flatten()
        l2_rel = float(
            np.sqrt(np.mean((u_pred - u_true)**2))
            / (np.sqrt(np.mean(u_true**2)) + 1e-12)
        )
        nu_err_pct = abs(
            float(model.param.item()) - self.true_nu
        ) / self.true_nu * 100.0
        return l2_rel, nu_err_pct

    def ref_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._x, self._t, self._u    # (Nx,) (Nt,) (Nt, Nx)


# ============================================================
# Training
# ============================================================

def run_inverse(
    problem,
    n_data:    int,
    *,
    epochs:    int   = EPOCHS,
    log_every: int   = 500,
) -> Dict:
    """Train one (problem, n_data) pair. Return results dict."""
    torch.manual_seed(0)
    np.random.seed(0)

    model = InversePINNModel(
        in_dim     = problem.in_dim,
        out_dim    = problem.out_dim,
        hidden     = HIDDEN,
        param_name = getattr(problem, "param_name", "p"),
        init_guess = problem.init_guess,
        true_val   = getattr(problem, "true_nu", None)
                     or getattr(problem, "true_k", None),
    )

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=5e-6
    )

    pts_bc, u_bc   = problem.sample_bc(N_BC)
    ic_data        = problem.sample_ic(N_IC)
    pts_obs, u_obs = problem.generate_observations(n_data)

    param_hist: List[float] = []
    loss_hist:  List[float] = []
    model.train()
    t0 = time.time()

    for epoch in range(epochs):
        optimizer.zero_grad()

        pts_col = problem.sample_interior(N_COL)
        pts_col.requires_grad_(True)

        # Physics residual
        res       = problem.physics_residual(model, pts_col)
        loss      = res.pow(2).mean()

        # BC
        loss      = loss + W_BC * F.mse_loss(_pred(model, pts_bc), u_bc)

        # IC
        if ic_data is not None:
            pts_ic, u_ic = ic_data
            loss  = loss + W_IC * F.mse_loss(_pred(model, pts_ic), u_ic)

        # Observation data (drives parameter identification)
        loss      = loss + W_DATA * F.mse_loss(_pred(model, pts_obs), u_obs)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        param_hist.append(float(model.param.item()))
        loss_hist.append(float(loss.detach()))

        if (epoch + 1) % log_every == 0:
            print(
                f"      [{epoch+1:5d}/{epochs}]"
                f"  loss={loss_hist[-1]:.4e}"
                f"  p_ident={param_hist[-1]:.6f}"
            )

    elapsed          = time.time() - t0
    model.eval()
    l2_rel, p_err    = problem.evaluate(model)

    return {
        "model":      model,
        "l2_rel":     l2_rel,
        "param_err":  p_err,
        "param_hist": param_hist,
        "loss_hist":  loss_hist,
        "time_s":     elapsed,
        "n_data":     n_data,
        "p_final":    param_hist[-1],
    }


# ============================================================
# Benchmark runner
# ============================================================

PROBLEMS = [InverseHeatProblem(), InverseBurgersProblem()]
all_results: Dict[Tuple[str, int], Dict] = {}

print("=" * 68)
print("  PINNeAPPle — Inverse Problem Benchmark")
print(f"  N_data sweep: {N_DATA_SWEEP}  |  epochs: {EPOCHS}")
print("=" * 68)

for prob in PROBLEMS:
    true_p = getattr(prob, "true_k", None) or getattr(prob, "true_nu", None)
    print(f"\n── {prob.name}  (true param = {true_p:.6f}) ──")

    for n_d in N_DATA_SWEEP:
        print(f"  N_data = {n_d:4d}")
        res = run_inverse(prob, n_d, log_every=max(1, EPOCHS // 4))
        all_results[(prob.name, n_d)] = res
        print(
            f"    → p_ident = {res['p_final']:.6f}"
            f"  param_err = {res['param_err']:.2f}%"
            f"  field_L2 = {res['l2_rel']:.4f}"
            f"  ({res['time_s']:.0f}s)"
        )


# ============================================================
# Summary table
# ============================================================

SEP = "─" * 72
print(f"\n{SEP}")
print("  Inverse Benchmark — Summary")
print(SEP)

for prob in PROBLEMS:
    true_p = getattr(prob, "true_k", None) or getattr(prob, "true_nu", None)
    print(f"\n  {prob.name}  (true = {true_p:.6f})")
    print(f"  {'N_data':>8}  {'p_ident':>12}  {'param_err%':>12}  {'field_L2':>10}")
    print("  " + "─" * 50)
    for n_d in N_DATA_SWEEP:
        key = (prob.name, n_d)
        if key not in all_results:
            continue
        r = all_results[key]
        print(
            f"  {n_d:8d}  {r['p_final']:12.6f}"
            f"  {r['param_err']:>11.2f}%  {r['l2_rel']:10.4f}"
        )

print(f"\n{SEP}")


# ============================================================
# Plots — 2×4 grid
#   Col 0: reference field + observation markers
#   Col 1: parameter convergence (N_data=100)
#   Col 2: data-efficiency bar chart (param_err vs N_data)
#   Col 3: best prediction colormap
# ============================================================

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mticker

    _DATA_COLORS = {
        25:  "#d62728",
        50:  "#ff7f0e",
        100: "#2ca02c",
        200: "#1f77b4",
    }

    fig = plt.figure(figsize=(18, 9))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.40)

    for row, prob in enumerate(PROBLEMS):
        true_p = getattr(prob, "true_k", None) or getattr(prob, "true_nu", None)
        p_name = "k" if isinstance(prob, InverseHeatProblem) else "ν"

        # ── Col 0: reference field + observations ──────────────────────
        ax0 = fig.add_subplot(gs[row, 0])

        if isinstance(prob, InverseHeatProblem):
            x_r, t_r, u_r = prob.ref_data()
            X, T = np.meshgrid(x_r, t_r)
            im   = ax0.pcolormesh(T, X, u_r, cmap="viridis", shading="auto")
            ax0.set_xlabel("t"); ax0.set_ylabel("x")
            # Overlay observations for N_data=100
            pts_obs, _ = prob.generate_observations(100, noise=0.0)
            ax0.scatter(pts_obs[:, 1].numpy(), pts_obs[:, 0].numpy(),
                        c="red", s=6, alpha=0.7, label="obs (N=100)")
            ax0.legend(fontsize=7)
            plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04, label="u")

        else:   # Burgers
            x_r, t_r, u_r = prob.ref_data()
            im   = ax0.pcolormesh(t_r, x_r, u_r.T, cmap="RdBu_r", shading="auto")
            ax0.set_xlabel("t"); ax0.set_ylabel("x")
            pts_obs, _ = prob.generate_observations(100, noise=0.0)
            ax0.scatter(pts_obs[:, 1].numpy(), pts_obs[:, 0].numpy(),
                        c="red", s=6, alpha=0.7, label="obs (N=100)")
            ax0.legend(fontsize=7)
            plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04, label="u")

        ax0.set_title(f"{prob.name}\n(reference + observations)", fontsize=9)

        # ── Col 1: parameter convergence for each N_data ───────────────
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.axhline(true_p, color="k", lw=1.5, ls="--", label=f"true {p_name}={true_p:.4f}")

        for n_d in N_DATA_SWEEP:
            key = (prob.name, n_d)
            if key not in all_results:
                continue
            hist = all_results[key]["param_hist"]
            # Smooth
            win  = max(1, len(hist) // 80)
            hs   = np.convolve(hist, np.ones(win) / win, mode="valid")
            ax1.plot(hs, color=_DATA_COLORS[n_d], lw=1.3,
                     label=f"N={n_d}")

        ax1.set_xlabel("Epoch"); ax1.set_ylabel(p_name)
        ax1.set_title(f"{prob.name}\n({p_name} convergence)", fontsize=9)
        ax1.legend(fontsize=7)
        ax1.grid(True, alpha=0.25)

        # ── Col 2: data efficiency (param error vs N_data) ─────────────
        ax2 = fig.add_subplot(gs[row, 2])
        n_vals  = [n for n in N_DATA_SWEEP if (prob.name, n) in all_results]
        p_errs  = [all_results[(prob.name, n)]["param_err"]  for n in n_vals]
        l2_errs = [all_results[(prob.name, n)]["l2_rel"] * 100 for n in n_vals]

        x_pos = np.arange(len(n_vals))
        w     = 0.35
        b1    = ax2.bar(x_pos - w/2, p_errs,  w, label=f"|Δ{p_name}| %",  color="#2ca02c", alpha=0.8)
        b2    = ax2.bar(x_pos + w/2, l2_errs, w, label="field L2 %", color="#1f77b4", alpha=0.8)
        ax2.set_xticks(x_pos); ax2.set_xticklabels([str(n) for n in n_vals])
        ax2.set_xlabel("N observations"); ax2.set_ylabel("Error (%)")
        ax2.set_title(f"{prob.name}\n(data efficiency)", fontsize=9)
        ax2.legend(fontsize=7)
        ax2.grid(True, axis="y", alpha=0.25)

        # ── Col 3: best prediction colormap (N_data=200) ───────────────
        ax3   = fig.add_subplot(gs[row, 3])
        best_key = (prob.name, max(N_DATA_SWEEP))
        if best_key in all_results:
            best_m = all_results[best_key]["model"]
            best_m.eval()

            if isinstance(prob, InverseHeatProblem):
                x_r, t_r, u_ref = prob.ref_data()
                Ng   = len(x_r)
                X, T = np.meshgrid(x_r, t_r)
                pts  = torch.tensor(
                    np.stack([X.flatten(), T.flatten()], axis=1),
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    u_p = best_m(pts).cpu().numpy().reshape(Ng, Ng)
                err  = np.abs(u_p - u_ref)
                im   = ax3.pcolormesh(T, X, err, cmap="Oranges", shading="auto")
                ax3.set_xlabel("t"); ax3.set_ylabel("x")
                plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="|error|")
            else:
                x_r, t_r, u_ref = prob.ref_data()
                X, T  = np.meshgrid(x_r, t_r)
                pts   = torch.tensor(
                    np.stack([X.flatten(), T.flatten()], axis=1),
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    u_p = best_m(pts).cpu().numpy().reshape(len(t_r), len(x_r))
                err   = np.abs(u_p - u_ref)
                im    = ax3.pcolormesh(t_r, x_r, err.T, cmap="Oranges", shading="auto")
                ax3.set_xlabel("t"); ax3.set_ylabel("x")
                plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04, label="|error|")

            p_id = all_results[best_key]["p_final"]
            ax3.set_title(
                f"{prob.name}\n|pred−ref|  (N=200, {p_name}={p_id:.4f})",
                fontsize=9,
            )

    fig.suptitle(
        "PINNeAPPle — Inverse Problem Benchmark\n"
        "Recovering unknown PDE parameters from sparse noisy observations",
        fontsize=12, fontweight="bold",
    )

    out_path = Path("outputs") / "03_inverse_benchmark.png"
    out_path.parent.mkdir(exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n  Plot saved: {out_path.resolve()}")
    plt.close()

except ImportError:
    print("  (matplotlib not available — plots skipped)")

print("\nInverse benchmark complete.")
