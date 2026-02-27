"""Inverse parameter identification (1D heat equation) with PINNFactory.

This example demonstrates the *symbolic* pipeline:
  - write PDE residuals / conditions as strings
  - compile with SymPy -> torch
  - learn an unknown physical coefficient (inverse parameter)

Heat equation on (t,x) in [0,1]x[0,1]:
  u_t - alpha * u_xx = 0

We set the *true* alpha and generate sparse supervised measurements at a later time.
The PINN learns both the field u(t,x) and alpha.

Run:
  python examples/pinneaple_pinn/04_factory_inverse_parameter_heat_1d.py
"""

from __future__ import annotations

import math

import numpy as np
import torch

from pinneaple_pinn.factory.pinn_factory import NeuralNetwork, PINN, PINNFactory, PINNProblemSpec


def u_analytic(t: np.ndarray, x: np.ndarray, alpha: float) -> np.ndarray:
    """Analytic solution for IC u(0,x)=sin(pi x) and zero BC: u(t,x)=exp(-pi^2*alpha*t)*sin(pi x)."""
    return (np.exp(-(math.pi**2) * alpha * t) * np.sin(math.pi * x)).astype(np.float32)


def sample_uniform(n: int, low: float, high: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=(n, 1)).astype(np.float32)


def main() -> None:
    torch.manual_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    alpha_true = 0.15

    # PDE residual uses inverse parameter 'alpha'
    # NOTE: dependent var is 'u' and independent are 't','x'
    spec = PINNProblemSpec(
        pde_residuals=["Derivative(u(t,x), t) - alpha*Derivative(u(t,x), (x,2))"],
        conditions=[
            {"name": "ic", "equation": "u(t,x) - sin(pi*x)", "weight": 1.0},
            {"name": "bc_left", "equation": "u(t,x)", "weight": 1.0},
            {"name": "bc_right", "equation": "u(t,x)", "weight": 1.0},
        ],
        independent_vars=["t", "x"],
        dependent_vars=["u"],
        inverse_params=["alpha"],
        loss_weights={"pde": 1.0, "conditions": 50.0, "data": 5.0},
        verbose=True,
    )

    factory = PINNFactory(spec)
    loss_fn = factory.generate_loss_function()

    net = NeuralNetwork(num_inputs=2, num_outputs=1, num_layers=4, num_neurons=64, activation=torch.nn.Tanh())
    model = PINN(net, inverse_params_names=["alpha"], initial_guesses={"alpha": 0.5}, dtype=dtype).to(device=device, dtype=dtype)

    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    # --- Build training batches ---
    # Collocation points
    t_col = torch.from_numpy(sample_uniform(6000, 0.0, 1.0, seed=0)).to(device=device, dtype=dtype).requires_grad_(True)
    x_col = torch.from_numpy(sample_uniform(6000, 0.0, 1.0, seed=1)).to(device=device, dtype=dtype).requires_grad_(True)

    # Conditions:
    # IC: t=0, x in [0,1]
    t_ic = torch.zeros((1500, 1), device=device, dtype=dtype, requires_grad=True)
    x_ic = torch.from_numpy(sample_uniform(1500, 0.0, 1.0, seed=2)).to(device=device, dtype=dtype).requires_grad_(True)
    # BC left: x=0
    t_bl = torch.from_numpy(sample_uniform(1500, 0.0, 1.0, seed=3)).to(device=device, dtype=dtype).requires_grad_(True)
    x_bl = torch.zeros((1500, 1), device=device, dtype=dtype, requires_grad=True)
    # BC right: x=1
    t_br = torch.from_numpy(sample_uniform(1500, 0.0, 1.0, seed=4)).to(device=device, dtype=dtype).requires_grad_(True)
    x_br = torch.ones((1500, 1), device=device, dtype=dtype, requires_grad=True)

    # Sparse supervised data at t = 0.5
    n_data = 256
    t_d = np.full((n_data, 1), 0.5, dtype=np.float32)
    x_d = sample_uniform(n_data, 0.0, 1.0, seed=5)
    y_d = u_analytic(t_d, x_d, alpha_true)
    t_d_t = torch.from_numpy(t_d).to(device=device, dtype=dtype).requires_grad_(True)
    x_d_t = torch.from_numpy(x_d).to(device=device, dtype=dtype).requires_grad_(True)
    y_d_t = torch.from_numpy(y_d).to(device=device, dtype=dtype)

    batch = {
        "collocation": (t_col, x_col),
        "conditions": [(t_ic, x_ic), (t_bl, x_bl), (t_br, x_br)],
        "data": ((t_d_t, x_d_t), y_d_t),
    }

    # --- Train ---
    steps = 2500
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, comps = loss_fn(model, batch)
        loss.backward()
        opt.step()

        if step % 250 == 0 or step == 1:
            alpha_hat = float(model.inverse_params["alpha"].detach().cpu().item())
            print(f"step={step:04d} total={comps['total']:.4e} pde={comps.get('pde',0):.4e} cond={comps.get('conditions',0):.4e} data={comps.get('data',0):.4e} | alpha={alpha_hat:.5f}")

    alpha_hat = float(model.inverse_params["alpha"].detach().cpu().item())
    print(f"alpha_true={alpha_true:.5f} alpha_learned={alpha_hat:.5f}")

    # small evaluation: relative L2 at t=0.5 over a grid
    xg = np.linspace(0, 1, 200, dtype=np.float32)[:, None]
    tg = np.full_like(xg, 0.5)
    with torch.no_grad():
        pred = model(
            torch.from_numpy(tg).to(device=device, dtype=dtype),
            torch.from_numpy(xg).to(device=device, dtype=dtype),
        ).cpu().numpy()
    true = u_analytic(tg, xg, alpha_true)
    rel = float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))
    print(f"rel_L2(t=0.5) = {rel:.3e}")


if __name__ == "__main__":
    main()
