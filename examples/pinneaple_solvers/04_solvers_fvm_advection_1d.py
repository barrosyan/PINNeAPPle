"""
pinneaple_solvers: FVMSolver demo (1D linear advection, periodic)

PDE
---
u_t + a u_x = 0 on x in [0, 1), periodic.

What this shows
--------------
- How to use FVMSolver with a custom flux_fn.
- How to build a minimal 1D periodic topology (faces, areas, volumes).
- How to get a full trajectory tensor (steps+1, Nc, C).

Run
---
python examples/pinneaple_solvers/04_solvers_fvm_advection_1d.py
"""

from __future__ import annotations

from typing import Dict, Any

import torch

from pinneaple_solvers.fvm import FVMSolver


def make_periodic_topology(Nc: int, dx: float, device=None, dtype=None) -> Dict[str, Any]:
    """Cell i connects to i+1 by face i (periodic)."""
    faces = []
    for i in range(Nc):
        left = i
        right = (i + 1) % Nc
        faces.append([left, right])
    faces = torch.tensor(faces, device=device, dtype=torch.long)

    areas = torch.ones((Nc, 1), device=device, dtype=dtype)  # 1D: area=1
    volumes = torch.full((Nc, 1), float(dx), device=device, dtype=dtype)
    normals = torch.ones((Nc, 1), device=device, dtype=dtype)  # not used here
    return {"faces": faces, "areas": areas, "volumes": volumes, "normals": normals}


def upwind_flux(ui: torch.Tensor, uj: torch.Tensor, face_info: Dict[str, Any], params: Dict[str, Any]) -> torch.Tensor:
    """Upwind numerical flux for scalar advection."""
    a = float(params.get("a", 1.0))
    if a >= 0:
        return a * ui
    return a * uj


def main():
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.float64

    Nc = 256
    x = torch.linspace(0.0, 1.0, Nc + 1, device=device, dtype=dtype)[:-1]
    dx = float(1.0 / Nc)

    # initial condition: smooth bump
    u0 = torch.exp(-200.0 * (x - 0.35) ** 2)[:, None]  # (Nc,1)

    a = 1.0
    cfl = 0.5
    dt = cfl * dx / abs(a)
    steps = 200
    t_final = steps * dt

    topology = make_periodic_topology(Nc=Nc, dx=dx, device=device, dtype=dtype)

    solver = FVMSolver(flux_fn=upwind_flux)
    out = solver(mesh=None, u0=u0, steps=steps, dt=dt, topology=topology, params={"a": a})

    traj = out.result  # (steps+1, Nc, 1)
    uT = traj[-1, :, 0]

    # exact: shift by a*t
    shift = (a * t_final) % 1.0
    x_shift = (x - shift) % 1.0
    u_true = torch.exp(-200.0 * (x_shift - 0.35) ** 2)

    rmse = torch.sqrt(torch.mean((uT - u_true) ** 2)).item()

    print("--- FVM advection 1D (periodic)")
    print(f"Nc={Nc} | dx={dx:.6f} | dt={dt:.6f} | steps={steps} | t={t_final:.4f}")
    print(f"RMSE(u(T), u_exact(T)) = {rmse:.6e}")


if __name__ == "__main__":
    main()