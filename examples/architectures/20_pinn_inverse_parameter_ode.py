"""Inverse parameter identification with VanillaPINN.

We solve a classic 1D eigen-ODE (strong form):
  u''(x) + a * u(x) = 0,  x in (0, 1)
  u(0) = 0, u(1) = 0

One known solution is u(x)=sin(pi x) with a = pi^2.

Demo goal:
- use `VanillaPINN` *and* its `inverse_params` mechanism to learn `a`.

Run:
  python examples/pinneaple_models_showcase/20_pinn_inverse_parameter_ode.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as a script: add repo root to PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import math

import torch

from pinneaple_models.pinns.vanilla import VanillaPINN


def physics_fn(model: VanillaPINN, data: dict) -> tuple[torch.Tensor, dict]:
    """Compute physics + boundary losses.

    Expected data keys:
      x_col: (N,1) collocation points in (0,1)
      x_bc:  (2,1) boundary points [0,1]
      u_bc:  (2,1) boundary target (zeros)
    """
    x_col = data["x_col"]
    x_bc = data["x_bc"]
    u_bc = data["u_bc"]

    # PDE residual at collocation points
    u = model(x_col).y  # (N,1)
    du = torch.autograd.grad(u, x_col, torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, x_col, torch.ones_like(du), create_graph=True)[0]

    a = model.inverse_params["a"]
    res = d2u + a * u
    loss_phys = torch.mean(res**2)

    # BC loss
    u_hat_bc = model(x_bc).y
    loss_bc = torch.mean((u_hat_bc - u_bc) ** 2)

    total = loss_phys + 10.0 * loss_bc
    return total, {"total": total, "physics": loss_phys, "bc": loss_bc}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # ground-truth
    a_true = math.pi**2

    model = VanillaPINN(
        in_dim=1,
        out_dim=1,
        hidden=[64, 64, 64, 64],
        activation="tanh",
        inverse_params_names=["a"],
        initial_guesses={"a": 1.0},
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    # data
    N_col = 256
    x_col = torch.rand(N_col, 1, device=device)
    x_col.requires_grad_(True)

    x_bc = torch.tensor([[0.0], [1.0]], device=device)
    u_bc = torch.zeros_like(x_bc)

    # (optional) add a few noisy observations to stabilize the phase
    x_obs = torch.linspace(0, 1, 25, device=device)[:, None]
    u_obs = torch.sin(math.pi * x_obs)

    print("Training inverse PINN for a (expect ~pi^2 = %.6f)" % a_true)
    for step in range(1, 2001):
        # refresh collocation points occasionally
        if step % 200 == 0:
            x_col = torch.rand(N_col, 1, device=device)
            x_col.requires_grad_(True)

        physics_data = {"x_col": x_col, "x_bc": x_bc, "u_bc": u_bc}
        out = model(x_col, physics_fn=physics_fn, physics_data=physics_data)

        # small supervised term (helps avoid trivial u=0)
        u_hat = model(x_obs).y
        loss_data = torch.mean((u_hat - u_obs) ** 2)

        loss = out.losses["total"] + 0.5 * loss_data

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 250 == 0:
            a_hat = float(model.inverse_params["a"].detach().cpu())
            print(
                f"step {step:4d} | loss {float(loss.detach()):.4e} | "
                f"a_hat {a_hat:.6f}"
            )

    a_hat = float(model.inverse_params["a"].detach().cpu())
    print("\nFinal a_hat:", a_hat)
    print("True  a:", a_true)


if __name__ == "__main__":
    main()
