"""
Demo: SPH / ISPH / DFSPH

Creates particles in a 2D box, runs a few steps, prints shapes.
This is an MVP sanity test (not a physically-perfect fluid sim).
"""

import torch

from pinneaple_solvers.registry import SolverCatalog
from pinneaple_geom.sample.particles import sample_box_particles
from pinneaple_solvers.sph_boundaries import reflect_box


def main():
    torch.manual_seed(0)

    device = torch.device("cpu")
    dtype = torch.float32

    # 2D box
    bounds_min = torch.tensor([0.0, 0.0], device=device, dtype=dtype)
    bounds_max = torch.tensor([1.0, 1.0], device=device, dtype=dtype)

    # sample particles
    ps = sample_box_particles(bounds_min, bounds_max, spacing=0.05, jitter=0.1, rho0=1000.0, device=device, dtype=dtype)
    x0 = ps.x
    N, D = x0.shape
    v0 = torch.zeros((N, D), device=device, dtype=dtype)

    # give a small initial swirl so something happens
    center = (bounds_min + bounds_max) / 2.0
    r = x0 - center
    v0[:, 0] = -r[:, 1]
    v0[:, 1] = r[:, 0]
    v0 = 0.5 * v0

    cat = SolverCatalog()

    # --- WCSPH
    sph = cat.build(
        "sph",
        h=0.08,
        rho0=1000.0,
        k=2000.0,
        gamma=7.0,
        nu=0.02,
        gravity=torch.tensor([0.0, -9.81], device=device, dtype=dtype),
        record_history=True,
        max_neighbors=64,
    )
    out = sph(x0, v0, m=ps.m, dt=2e-3, steps=50, clamp_domain=(bounds_min, bounds_max))
    xT = out.result
    vT = out.extras["v"]
    print("\nWCSPH")
    print("  xT:", xT.shape, "vT:", vT.shape)
    print("  rho:", out.extras["rho"].shape, "p:", out.extras["p"].shape)
    print("  x_hist:", out.extras["x_hist"].shape)

    # --- ISPH (projection scaffold)
    isph = cat.build(
        "isph",
        h=0.08,
        rho0=1000.0,
        nu=0.02,
        gravity=torch.tensor([0.0, -9.81], device=device, dtype=dtype),
        poisson_iters=30,
        omega_relax=0.8,
        max_neighbors=64,
    )
    out = isph(x0, v0, dt=2e-3, steps=50)
    print("\nISPH (scaffold)")
    print("  xT:", out.result.shape, "v:", out.extras["v"].shape, "p:", out.extras["p"].shape)

    # --- DFSPH (divergence correction scaffold)
    dfsph = cat.build(
        "dfsph",
        h=0.08,
        rho0=1000.0,
        nu=0.02,
        gravity=torch.tensor([0.0, -9.81], device=device, dtype=dtype),
        iters=8,
        max_neighbors=64,
    )
    out = dfsph(x0, v0, dt=2e-3, steps=50)
    print("\nDFSPH (scaffold)")
    print("  xT:", out.result.shape, "v:", out.extras["v"].shape)

    # Optional: apply reflect boundary (external utility)
    xr, vr = reflect_box(xT, vT, bounds_min, bounds_max, restitution=0.3)
    print("\nBoundary reflect check:", xr.shape, vr.shape)


if __name__ == "__main__":
    main()