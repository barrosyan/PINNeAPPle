"""35_jax_backend.py — JAX backend for PINN training.

Demonstrates:
- JAXBackend: enable JAX-based computation via pinneaple_backend
- JaxPINN: Haiku/Flax-style PINN with jit-compiled training step
- PinnTrainerJAX: drop-in trainer compatible with JAX arrays
- Speed comparison: PyTorch CPU vs. JAX CPU vs. JAX JIT
- Automatic fallback to PyTorch if JAX is not installed
"""

import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_backend import JAXBackend, BackendConfig

# Detect JAX availability through pinneaple_backend
_backend = JAXBackend(config=BackendConfig(device="cpu"))
_JAX_OK  = _backend.is_available()

if _JAX_OK:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    from pinneaple_backend.jax_pinn import JaxPINN, JaxPINNConfig
    from pinneaple_backend.jax_trainer import PinnTrainerJAX, TrainerConfigJAX
    print("JAX backend available.")
else:
    print("JAX not installed — running PyTorch baseline only.")

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Problem: 1D Poisson  u'' = -π²sin(πx),  u(0)=u(1)=0
# Exact: u(x) = sin(πx)
# ---------------------------------------------------------------------------

def u_exact(x: np.ndarray) -> np.ndarray:
    return np.sin(math.pi * x)


# ============================================================================
# PyTorch baseline
# ============================================================================

def build_torch_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(1, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    )


def torch_loss(model: nn.Module, n: int = 300) -> torch.Tensor:
    x = torch.linspace(0, 1, n).unsqueeze(1).requires_grad_(True)
    u = model(x) * x * (1 - x)      # hard BC
    u_x  = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    f    = -math.pi**2 * torch.sin(math.pi * x)
    return (u_xx - f).pow(2).mean()


def train_torch(n_epochs: int = 3000) -> tuple[list, float]:
    model = build_torch_model()
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist  = []
    t0    = time.perf_counter()
    for _ in range(n_epochs):
        opt.zero_grad()
        loss = torch_loss(model)
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
    elapsed = time.perf_counter() - t0

    x_t = torch.linspace(0, 1, 200).unsqueeze(1)
    with torch.no_grad():
        u_pred = (model(x_t) * x_t * (1 - x_t)).numpy().ravel()
    u_ex = u_exact(np.linspace(0, 1, 200))
    l2 = float(np.sqrt(((u_pred - u_ex)**2).mean()) / np.sqrt((u_ex**2).mean()))
    return hist, elapsed, l2


# ============================================================================
# JAX backend training
# ============================================================================

def train_jax(n_epochs: int = 3000):
    if not _JAX_OK:
        return None, float("nan"), float("nan")

    pinn_config = JaxPINNConfig(
        input_dim=1,
        output_dim=1,
        hidden_dims=[64, 64],
        activation="tanh",
        apply_hard_bc=True,   # multiplies output by x*(1-x)
    )
    pinn = JaxPINN(config=pinn_config)

    trainer_config = TrainerConfigJAX(
        n_epochs=n_epochs,
        lr=1e-3,
        n_col=300,
        source_fn=lambda x: -math.pi**2 * jnp.sin(math.pi * x),
        use_jit=True,
    )
    trainer = PinnTrainerJAX(pinn=pinn, config=trainer_config)

    t0 = time.perf_counter()
    hist, params = trainer.train()
    elapsed = time.perf_counter() - t0

    # Evaluate
    x_eval = jnp.linspace(0, 1, 200)[:, None]
    u_pred = pinn.apply(params, x_eval).ravel()
    u_pred = np.array(u_pred * x_eval.ravel() * (1 - x_eval.ravel()))
    u_ex   = u_exact(np.linspace(0, 1, 200))
    l2 = float(np.sqrt(((u_pred - u_ex)**2).mean()) / np.sqrt((u_ex**2).mean()))
    return hist, elapsed, l2


def main():
    N_EPOCHS = 3000
    print(f"Training 1D Poisson PINN for {N_EPOCHS} epochs ...")

    print("\n[PyTorch CPU]")
    torch_hist, torch_time, torch_l2 = train_torch(N_EPOCHS)
    print(f"  time = {torch_time:.2f} s  L2 = {torch_l2:.4e}")

    print("\n[JAX backend]")
    jax_hist, jax_time, jax_l2 = train_jax(N_EPOCHS)
    if _JAX_OK:
        print(f"  time = {jax_time:.2f} s  L2 = {jax_l2:.4e}")
        speedup = torch_time / jax_time
        print(f"  JAX speedup: {speedup:.2f}x")
    else:
        print("  (skipped — JAX not available)")

    # --- Prediction ----------------------------------------------------------
    x_plot = np.linspace(0, 1, 200)

    # PyTorch prediction
    m = build_torch_model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(N_EPOCHS):
        opt.zero_grad()
        torch_loss(m).backward()
        opt.step()
    x_t = torch.tensor(x_plot[:, None], dtype=torch.float32)
    with torch.no_grad():
        u_torch = (m(x_t) * x_t * (1 - x_t)).numpy().ravel()

    # --- Visualisation -------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x_plot, u_exact(x_plot), "k--", label="Exact")
    axes[0].plot(x_plot, u_torch, "b-", label=f"PyTorch (L2={torch_l2:.2e})")
    if _JAX_OK and jax_hist is not None:
        # Re-evaluate JAX model for plotting
        pass   # already evaluated — use stored l2
        axes[0].set_title(
            f"Solution comparison\nPyTorch L2={torch_l2:.2e}  JAX L2={jax_l2:.2e}"
        )
    else:
        axes[0].set_title(f"Solution  (L2={torch_l2:.2e})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u(x)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(torch_hist, label=f"PyTorch ({torch_time:.1f}s)")
    if _JAX_OK and jax_hist is not None:
        axes[1].semilogy(jax_hist, label=f"JAX ({jax_time:.1f}s)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PDE loss")
    axes[1].set_title("Training convergence")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig("35_jax_backend_result.png", dpi=120)
    print("\nSaved 35_jax_backend_result.png")


if __name__ == "__main__":
    main()
