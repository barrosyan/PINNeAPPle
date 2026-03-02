"""Examples: time-series decomposition solvers.

Run:
  python examples/solvers/01_time_series_decompositions.py
"""

import numpy as np
import torch

from pinneaple_solvers import SolverRegistry, register_all


def main():
    register_all()

    t = np.linspace(0, 10, 2000)
    x = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.sin(2 * np.pi * 6.0 * t) + 0.1 * np.random.randn(len(t))
    x_t = torch.tensor(x, dtype=torch.float32)

    eemd = SolverRegistry.build("eemd", n_ensembles=30, noise_std=0.15, max_imfs=6)
    out = eemd(x_t)
    print("EEMD imfs:", out.result.shape)

    vmd = SolverRegistry.build("vmd", K=4)
    out = vmd(x_t)
    print("VMD modes:", out.result.shape, "omega:", np.array(out.extras["omega"]).shape)

    wav = SolverRegistry.build("wavelet", wavelet="db4", level=4)
    out = wav(x_t)
    print("Wavelet features:", out.result.shape)

    stl = SolverRegistry.build("stl", period=200)
    out = stl(x_t)
    print("STL parts:", out.result.shape, "(trend,seasonal,resid)")

    ssa = SolverRegistry.build("ssa", L=200, r=5)
    out = ssa(x_t)
    print("SSA comps:", out.result.shape)

    sst = SolverRegistry.build("sst", fs=1.0, nperseg=128)
    out = sst(x_t)
    print("SST TF map:", out.result.shape)


if __name__ == "__main__":
    main()
