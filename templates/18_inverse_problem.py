"""18_inverse_problem.py — Inverse problem with Ensemble Kalman Inversion.

Demonstrates:
- EKISolver (Ensemble Kalman Inversion) for parameter estimation
- SensitivityAnalyser for parameter importance ranking
- Recovery of unknown diffusivity α from noisy PDE observations
- Posterior uncertainty visualisation
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_inverse.eki import EKISolver, EKIConfig
from pinneaple_inverse.sensitivity import SensitivityAnalyser


# ---------------------------------------------------------------------------
# Forward problem: 1D steady diffusion  -(α u_x)_x = f  on [0,1]
# f = π² sin(πx),  u(0) = u(1) = 0
# Exact solution: u = sin(πx) / α  (for constant α)
# We observe u at 20 sensor locations and want to recover α.
# ---------------------------------------------------------------------------

import math

ALPHA_TRUE = 0.35
N_SENSORS  = 20
OBS_NOISE  = 0.02


def u_exact(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.sin(math.pi * x) / alpha


def forward_model_pinn(alpha: float, n_col: int = 300, n_epochs: int = 800) -> nn.Module:
    """Train a tiny PINN for given α and return the model."""
    device = torch.device("cpu")
    net = nn.Sequential(
        nn.Linear(1, 32), nn.Tanh(),
        nn.Linear(32, 32), nn.Tanh(),
        nn.Linear(32, 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)

    x_col = torch.linspace(0, 1, n_col, device=device).unsqueeze(1)
    x_bc  = torch.tensor([[0.0], [1.0]], device=device)
    u_bc  = torch.zeros(2, 1, device=device)
    f_val = (math.pi ** 2) * torch.sin(math.pi * x_col)

    for _ in range(n_epochs):
        opt.zero_grad()
        xc = x_col.clone().requires_grad_(True)
        u  = net(xc)
        u_x  = torch.autograd.grad(u.sum(), xc, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x.sum(), xc, create_graph=True)[0]
        res  = -alpha * u_xx - f_val
        loss = res.pow(2).mean() + 50 * (net(x_bc) - u_bc).pow(2).mean()
        loss.backward()
        opt.step()

    return net


def observations(alpha: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Generate noisy sensor readings."""
    rng = np.random.default_rng(seed)
    x_s = np.linspace(0.05, 0.95, N_SENSORS)
    y_s = u_exact(x_s, alpha) + rng.normal(0, OBS_NOISE, N_SENSORS)
    return x_s, y_s


def build_forward_callable(x_sensors: np.ndarray):
    """Return F(params) for EKI — params = [log_alpha]."""
    def F(params: np.ndarray) -> np.ndarray:
        log_alpha = float(params[0])
        alpha = math.exp(log_alpha)
        pinn = forward_model_pinn(alpha, n_epochs=600)
        x_t  = torch.tensor(x_sensors[:, None], dtype=torch.float32)
        with torch.no_grad():
            u_pred = pinn(x_t).numpy().ravel()
        return u_pred
    return F


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    # --- Observations --------------------------------------------------------
    x_sensors, y_obs = observations(ALPHA_TRUE)
    print(f"True α = {ALPHA_TRUE:.3f} | {N_SENSORS} sensors | noise σ={OBS_NOISE}")

    # --- Sensitivity analysis (first-order finite difference) ----------------
    print("\nRunning sensitivity analysis ...")
    analyser = SensitivityAnalyser(
        param_names=["log_alpha"],
        forward_fn=build_forward_callable(x_sensors),
        base_params=np.array([math.log(0.4)]),
        eps=0.05,
    )
    sensitivities = analyser.compute()
    print(f"  ∂F/∂(log_α) mean|abs| = {np.abs(sensitivities).mean():.4f}")

    # --- EKI solver ----------------------------------------------------------
    print("\nRunning EKI for α recovery ...")
    eki_config = EKIConfig(
        n_ensemble=20,
        n_iterations=8,
        obs_noise_std=OBS_NOISE,
        prior_mean=np.array([math.log(0.5)]),
        prior_std=np.array([0.8]),
        device="cpu",
    )
    eki_solver = EKISolver(
        config=eki_config,
        forward_fn=build_forward_callable(x_sensors),
        observations=y_obs,
    )
    eki_result = eki_solver.run()

    # Posterior ensemble of log_alpha
    log_alpha_posterior = eki_result["ensemble_final"][:, 0]   # (n_ensemble,)
    alpha_posterior = np.exp(log_alpha_posterior)
    alpha_mean = float(alpha_posterior.mean())
    alpha_std  = float(alpha_posterior.std())
    print(f"  Estimated α = {alpha_mean:.4f} ± {alpha_std:.4f}  (true = {ALPHA_TRUE})")

    # --- Visualisation -------------------------------------------------------
    x_fine = np.linspace(0, 1, 200)
    u_true_fine = u_exact(x_fine, ALPHA_TRUE)

    # Posterior predictive band
    u_samples = np.stack([
        u_exact(x_fine, a) for a in alpha_posterior
    ])  # (n_ensemble, 200)
    u_lo, u_hi = np.percentile(u_samples, [5, 95], axis=0)
    u_post_mean = u_samples.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: posterior predictive vs. truth
    axes[0].fill_between(x_fine, u_lo, u_hi, alpha=0.3, color="blue",
                         label="90% posterior band")
    axes[0].plot(x_fine, u_true_fine, "k--", label=f"True (α={ALPHA_TRUE})")
    axes[0].plot(x_fine, u_post_mean, "b-", label=f"Post. mean (α≈{alpha_mean:.3f})")
    axes[0].scatter(x_sensors, y_obs, s=25, c="red", zorder=5, label="Observations")
    axes[0].set_title("Posterior predictive (EKI)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: posterior histogram of α
    axes[1].hist(alpha_posterior, bins=12, color="steelblue", edgecolor="k", alpha=0.8)
    axes[1].axvline(ALPHA_TRUE, color="red", lw=2, ls="--", label=f"True α={ALPHA_TRUE}")
    axes[1].axvline(alpha_mean, color="blue", lw=2, label=f"EKI mean={alpha_mean:.3f}")
    axes[1].set_xlabel("α (diffusivity)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Posterior ensemble histogram")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("18_inverse_problem_result.png", dpi=120)
    print("Saved 18_inverse_problem_result.png")


if __name__ == "__main__":
    main()
