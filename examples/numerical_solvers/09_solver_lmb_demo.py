"""
Demo: LBM D2Q9 (very basic)

Simulates a small 2D flow field with constant force.
"""

import torch

from pinneaple_solvers.registry import SolverCatalog


def main():
    device = torch.device("cpu")
    dtype = torch.float32
    cat = SolverCatalog()

    H, W = 64, 128
    rho0 = torch.ones((H, W), device=device, dtype=dtype)
    u0 = torch.zeros((2, H, W), device=device, dtype=dtype)

    lbm = cat.build("lbm", tau=0.65, force=torch.tensor([1e-5, 0.0], device=device, dtype=dtype), record_history=False)
    out = lbm(H=H, W=W, steps=300, rho0=rho0, u0=u0)

    u = out.result
    rho = out.extras["rho"]
    print("\nLBM D2Q9")
    print("  u:", u.shape, "rho:", rho.shape)
    print("  u mean:", u.mean().item(), "rho mean:", rho.mean().item())


if __name__ == "__main__":
    main()