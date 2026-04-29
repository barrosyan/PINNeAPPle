"""32_physics_validation.py — Physics consistency validation.

Demonstrates:
- PhysicsValidator: check that a trained PINN satisfies its governing equations
- ConservationLawChecker: verify energy / mass conservation integrals
- BoundaryConsistencyChecker: confirm BC satisfaction at sampled boundary points
- ValidationReport: structured per-check pass/fail with tolerance bands
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pinneaple_validate.physics_validator import PhysicsValidator, PhysicsValidatorConfig
from pinneaple_validate.conservation import ConservationLawChecker
from pinneaple_validate.boundary import BoundaryConsistencyChecker
from pinneaple_validate.report import ValidationReport


# ---------------------------------------------------------------------------
# Test problem: 2D Poisson  Δu = f  on [0,1]²
# Exact solution: u = sin(πx)sin(πy)
# We train a PINN and then run the full validation suite on it.
# ---------------------------------------------------------------------------

def f_source(xy: torch.Tensor) -> torch.Tensor:
    x, y = xy[:, 0:1], xy[:, 1:2]
    return -2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


def build_and_train(device, n_epochs: int = 4000) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 64), nn.Tanh(),
        nn.Linear(64, 1),
    ).to(device)
    phi = lambda xy: xy[:, 0:1] * (1 - xy[:, 0:1]) * xy[:, 1:2] * (1 - xy[:, 1:2])
    model = lambda xy: phi(xy) * net(xy)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for ep in range(n_epochs):
        opt.zero_grad()
        xy = torch.rand(2048, 2, device=device, requires_grad=True)
        u  = model(xy)
        g  = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        u_xx = torch.autograd.grad(g[:, 0:1].sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(g[:, 1:2].sum(), xy, create_graph=True)[0][:, 1:2]
        (u_xx + u_yy - f_source(xy)).pow(2).mean().backward()
        opt.step()

    return model


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Train ---------------------------------------------------------------
    print("Training Poisson PINN for validation ...")
    model = build_and_train(device, n_epochs=4000)
    print("Training complete.\n")

    # --- PhysicsValidator (PDE residual check) --------------------------------
    def pde_residual_fn(m, xy):
        xy = xy.requires_grad_(True)
        u  = m(xy)
        g  = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        u_xx = torch.autograd.grad(g[:, 0:1].sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(g[:, 1:2].sum(), xy, create_graph=True)[0][:, 1:2]
        return u_xx + u_yy - f_source(xy)

    pv_config = PhysicsValidatorConfig(
        n_check_points=4096,
        residual_tol=1e-2,         # pass if mean |residual| < tol
        percentile_tol=99,         # also check 99th percentile
        percentile_val=0.05,
        device=str(device),
    )
    pv = PhysicsValidator(
        model=model,
        residual_fn=pde_residual_fn,
        config=pv_config,
    )
    pv_result = pv.check()
    print("PDE residual check:")
    print(f"  mean |res| = {pv_result.mean_residual:.4e}  "
          f"({'PASS' if pv_result.passed else 'FAIL'})")
    print(f"  99th pct   = {pv_result.percentile_residual:.4e}")

    # --- ConservationLawChecker (∫ f dx dy = 0 for homogeneous Dirichlet) -----
    def energy_integral(m) -> float:
        """∫∫ |∇u|² dx dy  (Dirichlet energy)."""
        n = 50
        x = np.linspace(0, 1, n, dtype=np.float32)
        xx, yy = np.meshgrid(x, x)
        xy = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), device=device,
                          requires_grad=True)
        u  = m(xy)
        g  = torch.autograd.grad(u.sum(), xy)[0]
        energy = (g ** 2).sum(dim=1).mean().item()
        return energy

    expected_energy = (math.pi**2 / 2)  # analytic Dirichlet energy for sin(πx)sin(πy)
    cons_checker = ConservationLawChecker(
        model=model,
        law_fn=energy_integral,
        expected_value=expected_energy,
        relative_tol=0.05,
        name="Dirichlet energy",
    )
    cons_result = cons_checker.check()
    print(f"\nConservation check — Dirichlet energy:")
    print(f"  computed  = {cons_result.computed:.4f}")
    print(f"  expected  ≈ {expected_energy:.4f}")
    print(f"  rel error = {cons_result.relative_error:.4e}  "
          f"({'PASS' if cons_result.passed else 'FAIL'})")

    # --- BoundaryConsistencyChecker ------------------------------------------
    from pinneaple_geom.csg import CSGRectangle
    rect = CSGRectangle(0, 0, 1, 1)
    xy_bnd_np = rect.sample_boundary(n=512, seed=7)
    xy_bnd = torch.tensor(xy_bnd_np, dtype=torch.float32, device=device)

    bc_checker = BoundaryConsistencyChecker(
        model=model,
        bc_fn=lambda m, x: m(x),           # u should be 0 at boundary
        bc_value=torch.zeros(len(xy_bnd), 1, device=device),
        tol=1e-2,
    )
    bc_result = bc_checker.check(xy_bnd)
    print(f"\nBoundary consistency check (Dirichlet u=0):")
    print(f"  max |u_bc| = {bc_result.max_error:.4e}  "
          f"({'PASS' if bc_result.passed else 'FAIL'})")
    print(f"  mean|u_bc| = {bc_result.mean_error:.4e}")

    # --- Compile full report -------------------------------------------------
    report = ValidationReport(checks=[pv_result, cons_result, bc_result])
    print(f"\n{'='*50}")
    print("VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(report.summary())

    passed_all = report.all_passed()
    print(f"\nOverall: {'ALL CHECKS PASSED' if passed_all else 'SOME CHECKS FAILED'}")

    # --- Visualisation -------------------------------------------------------
    n_vis = 60
    x_ = np.linspace(0, 1, n_vis, dtype=np.float32)
    xx, yy = np.meshgrid(x_, x_)
    xy_vis = torch.tensor(
        np.stack([xx.ravel(), yy.ravel()], axis=1), device=device
    )
    with torch.no_grad():
        u_pred = model(xy_vis).cpu().numpy().reshape(n_vis, n_vis)
        xy_pde = torch.rand(1024, 2, device=device, requires_grad=True)
        res    = pde_residual_fn(model, xy_pde).abs().detach().cpu().numpy().ravel()

    xy_pde_np = xy_pde.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    im1 = axes[0].contourf(xx, yy, u_pred, levels=30, cmap="viridis")
    plt.colorbar(im1, ax=axes[0])
    axes[0].set_title("Predicted u(x,y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    sc = axes[1].scatter(xy_pde_np[:, 0], xy_pde_np[:, 1],
                         c=res, s=4, cmap="Reds", vmin=0)
    plt.colorbar(sc, ax=axes[1])
    axes[1].set_title("PDE residual |Δu - f| (spatial map)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("32_physics_validation_result.png", dpi=120)
    print("\nSaved 32_physics_validation_result.png")


if __name__ == "__main__":
    main()
