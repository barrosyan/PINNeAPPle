"""
Demo: decomposition / time-frequency solvers

Runs:
  - EEMD
  - CEEMDAN
  - VMD
  - Wavelet CWT
  - SST
  - SSA
  - STL

Outputs a few tensor shapes + simple diagnostics.
"""

import torch

from pinneaple_solvers.registry import SolverCatalog


def make_signal(T: int = 2048, dt: float = 1.0) -> torch.Tensor:
    t = torch.arange(T) * dt
    # chirp-ish + seasonal + noise
    x = (
        0.9 * torch.sin(2 * torch.pi * (0.01 * t + 0.00001 * t * t))
        + 0.5 * torch.sin(2 * torch.pi * 0.08 * t)
        + 0.2 * torch.sin(2 * torch.pi * (1 / 64.0) * t)
        + 0.15 * torch.randn(T)
    )
    return x


def main():
    torch.manual_seed(0)
    x = make_signal(T=2048, dt=1.0)

    cat = SolverCatalog()

    # --- EEMD
    eemd = cat.build(
        "eemd",
        max_imfs=8,
        ensembles=32,
        noise_std=0.2,
        dt=1.0,
        compute_hilbert=True,
        seed=0,
    )
    out = eemd(x)
    print("\nEEMD")
    print("  imfs:", out.result.shape)
    print("  residual:", out.extras["residual"].shape)
    if "inst_freq" in out.extras:
        print("  inst_freq:", out.extras["inst_freq"].shape)

    # --- CEEMDAN
    ce = cat.build(
        "ceemdan",
        max_imfs=8,
        ensembles=32,
        noise_std=0.2,
        dt=1.0,
        seed=0,
    )
    out = ce(x)
    print("\nCEEMDAN")
    print("  imfs:", out.result.shape)
    print("  residual:", out.extras["residual"].shape)
    print("  K_eff:", out.extras["K_eff"])

    # --- VMD
    vmd = cat.build(
        "vmd",
        K=6,
        alpha=2000.0,
        tau=0.0,
        DC=False,
        init="uniform",
        tol=1e-6,
        max_iters=500,
    )
    out = vmd(x)
    print("\nVMD")
    print("  modes:", out.result.shape)
    print("  omega:", out.extras["omega"].shape)
    print("  iters:", out.extras["iters"])

    # --- Wavelet CWT
    cwt = cat.build(
        "wavelet",
        mode="cwt",
        dt=1.0,
        n_scales=48,
        s_min=1.0,
        s_max=128.0,
        morlet_w0=6.0,
    )
    out = cwt(x)
    print("\nCWT (Morlet)")
    print("  coeffs:", out.result.shape, out.result.dtype)
    print("  scales:", out.extras["scales"].shape)
    print("  freqs_approx:", out.extras["freqs_approx"].shape)

    # --- SST
    sst = cat.build(
        "sst",
        dt=1.0,
        n_scales=48,
        s_min=1.0,
        s_max=128.0,
        morlet_w0=6.0,
        n_freq_bins=128,
    )
    out = sst(x)
    print("\nSST")
    print("  tfr:", out.result.shape)
    print("  omega_hat:", out.extras["omega_hat"].shape)
    print("  freqs_bins:", out.extras["freqs_bins"].shape)

    # --- SSA
    ssa = cat.build("ssa", window=128, n_components=8, center=True)
    out = ssa(x)
    print("\nSSA")
    print("  comps:", out.result.shape)
    print("  singular_values:", out.extras["singular_values"].shape)

    # --- STL
    stl = cat.build("stl", period=64, trend_window=101, seasonal_window=21, robust=True, robust_iters=2)
    out = stl(x)
    print("\nSTL")
    print("  stacked:", out.result.shape)  # (3,T)
    print("  trend:", out.extras["trend"].shape)
    print("  seasonal:", out.extras["seasonal"].shape)
    print("  residual:", out.extras["residual"].shape)


if __name__ == "__main__":
    main()