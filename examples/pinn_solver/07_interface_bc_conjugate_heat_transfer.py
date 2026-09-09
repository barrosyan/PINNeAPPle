"""InterfaceBC: a conjugate-heat-transfer-style, two-material coupling.

Showcases:
  - ``InterfaceBC`` (pinneapple_physics.pde_environment.conditions), a
    ``ConditionSpec`` that couples two fields evaluated at the SAME points
    with value continuity (field_a == field_b) and weighted-flux continuity
    (k_a * n.grad(field_a) == k_b * n.grad(field_b)) -- exactly the pair of
    matching conditions used at a conjugate-heat-transfer interface between
    two materials of different conductivity.
  - the real compiler dispatch for ``cond.kind == "interface"`` in
    ``pinneapple_physics/pinn_solver/compiler/compile.py``.
  - each side's own outer Dirichlet boundary condition, combined with the
    interface condition in one shared boundary batch via the compiler's
    ``mask_<condition_name>`` mechanism (each condition only "sees" the
    rows of ``x_bc`` selected for it).

Problem setup
-------------
There is currently only one PDE kind wired into ``compile_problem`` that
natively carries TWO independently-diffusing fields at the same points:
``"reaction_diffusion"`` (Gray-Scott, fields u, v, diffusivities Du, Dv).
``tests/test_interface_bc.py`` uses exactly this kind as a stand-in for two
materials meeting at an interface (see its ``_toy_spec`` docstring: "u =
T_solid, v = T_fluid"). This example mirrors that same real, working
construction -- through the same ``ProblemSpec`` / ``ConditionSpec`` /
``compile_problem`` pipeline -- and extends it into a standalone script
with outer boundary conditions and a real training loop:

  u(x, t): temperature-like field on the "solid" side, diffusivity Du
  v(x, t): temperature-like field on the "fluid" side, diffusivity Dv

  outer BC:      u(x=0, t) = 1.0   (fixed hot solid boundary)
                 v(x=1, t) = 0.0   (fixed cold fluid boundary)
  interface BC:  u(x=0.5, t) - v(x=0.5, t) = 0             (value continuity)
                 k_a*du/dx(0.5,t) - k_b*dv/dx(0.5,t) = 0   (flux continuity)

k_a != k_b models two materials of different conductivity meeting at the
interface x=0.5.

Run:
  python examples/pinn_solver/07_interface_bc_conjugate_heat_transfer.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pinneapple_physics.pde_environment.conditions import DirichletBC, InterfaceBC
from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec
from pinneapple_physics.pinn_solver.compiler.compile import compile_problem
from pinneapple_physics.pinn_solver.compiler.loss import LossWeights

X_INTERFACE = 0.5
K_A = 2.0  # "solid"-side conductivity weight
K_B = 0.5  # "fluid"-side conductivity weight


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 32, depth: int = 3):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_spec() -> ProblemSpec:
    coords = ("x", "t")
    fields = ("u", "v")
    pde = PDETermSpec(
        kind="reaction_diffusion",
        fields=fields,
        coords=coords,
        params={"Du": 0.5, "Dv": 0.1, "F": 0.04, "k": 0.06},
    )
    outer_u = DirichletBC(
        "outer_u", ("u",),
        value_fn=lambda X, ctx: np.ones((X.shape[0], 1), dtype=np.float32),
    )
    outer_v = DirichletBC(
        "outer_v", ("v",),
        value_fn=lambda X, ctx: np.zeros((X.shape[0], 1), dtype=np.float32),
    )
    interface = InterfaceBC(
        "region_interface", ("u", "v"), weight=5.0, k_a=K_A, k_b=K_B,
    )
    return ProblemSpec(
        name="conjugate_heat_transfer_1d",
        dim=1,
        coords=coords,
        fields=fields,
        pde=pde,
        conditions=(outer_u, outer_v, interface),
    )


def build_batch(n_col: int = 512, n_bc_each: int = 64, seed: int = 0) -> dict:
    """Sample collocation points plus a single combined boundary batch
    covering both outer walls and the interface, with per-condition
    boolean masks (the real ``mask_<condition_name>`` mechanism the
    compiler reads in ``pinneapple_physics/pinn_solver/compiler/compile.py``).
    """
    rng = np.random.default_rng(seed)

    x_col = np.stack(
        [rng.random(n_col).astype(np.float32), rng.random(n_col).astype(np.float32)],
        axis=1,
    )

    t_u = rng.random(n_bc_each).astype(np.float32)
    t_v = rng.random(n_bc_each).astype(np.float32)
    t_i = rng.random(n_bc_each).astype(np.float32)

    x_outer_u = np.stack([np.zeros(n_bc_each, dtype=np.float32), t_u], axis=1)
    x_outer_v = np.stack([np.ones(n_bc_each, dtype=np.float32), t_v], axis=1)
    x_interface = np.stack(
        [np.full(n_bc_each, X_INTERFACE, dtype=np.float32), t_i], axis=1
    )

    x_bc = np.concatenate([x_outer_u, x_outer_v, x_interface], axis=0)

    n_total = x_bc.shape[0]
    mask_outer_u = np.zeros(n_total, dtype=bool)
    mask_outer_v = np.zeros(n_total, dtype=bool)
    mask_interface = np.zeros(n_total, dtype=bool)
    mask_outer_u[:n_bc_each] = True
    mask_outer_v[n_bc_each:2 * n_bc_each] = True
    mask_interface[2 * n_bc_each:] = True

    # unit normal along x for every boundary/interface point (all three
    # sets lie on planes of constant x, so n = (1, 0) works for all of them)
    n_bc = np.zeros((n_total, 2), dtype=np.float32)
    n_bc[:, 0] = 1.0

    return {
        "x_col": torch.from_numpy(x_col).requires_grad_(False),
        "x_bc": torch.from_numpy(x_bc),
        "y_bc": None,
        "n_bc": torch.from_numpy(n_bc),
        "mask_outer_u": torch.from_numpy(mask_outer_u),
        "mask_outer_v": torch.from_numpy(mask_outer_v),
        "mask_region_interface": torch.from_numpy(mask_interface),
        "ctx": {},
    }


def interface_residual(model: nn.Module, batch: dict) -> float:
    """Direct value + weighted-flux continuity residual at the interface,
    computed independently of the compiled loss (a sanity check that the
    trained network actually satisfies InterfaceBC's condition)."""
    x_i = batch["x_bc"][batch["mask_region_interface"]].clone().detach().requires_grad_(True)
    y = model(x_i)
    u, v = y[:, 0:1], y[:, 1:2]
    (du,) = torch.autograd.grad(u, x_i, torch.ones_like(u), retain_graph=True)
    (dv,) = torch.autograd.grad(v, x_i, torch.ones_like(v))
    value_diff = (u - v).detach()
    flux_diff = (K_A * du[:, 0:1] - K_B * dv[:, 0:1]).detach()
    return float(torch.mean(value_diff ** 2 + flux_diff ** 2))


def main() -> None:
    torch.manual_seed(0)

    spec = build_spec()
    loss_fn = compile_problem(spec, weights=LossWeights(w_pde=1.0, w_bc=20.0))

    model = MLP(in_dim=2, out_dim=2, width=32, depth=3)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    batch = build_batch(n_col=512, n_bc_each=64, seed=0)

    steps = 3000
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        out = loss_fn(model, None, batch)
        total = out["total"]
        total.backward()
        opt.step()

        if step % 500 == 0 or step == 1:
            print(
                f"step={step:04d}  total={float(total.detach()):.4e}  "
                f"pde={float(out['pde'].detach()):.4e}  "
                f"bc_outer_u={float(out['bc_outer_u']):.4e}  "
                f"bc_outer_v={float(out['bc_outer_v']):.4e}  "
                f"bc_region_interface={float(out['bc_region_interface']):.4e}"
            )

    final_residual = interface_residual(model, batch)
    print(f"\nFinal direct interface continuity residual (value^2 + flux^2): {final_residual:.4e}")


if __name__ == "__main__":
    main()
