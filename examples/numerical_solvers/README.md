# pinneaple_solvers examples

This folder contains runnable scripts showcasing the **numerical-solvers layer**:

- **FFT**: spectral features, dominant frequency detection, and ML integration.
- **Hilbert–Huang (EMD + Hilbert)**: non-stationary signal decomposition and instantaneous frequency.
- **FEM (scaffold)**: linear system assembly + BC application + solve.
- **FVM (scaffold)**: flux-based time stepping with a custom numerical flux.
- **Adapters + registry**: `SolverCatalog` and UPD-style extraction from field tensors.

## Run

From the repo root:

```bash
python examples/pinneaple_solvers/01_solvers_fft_feature_train.py
python examples/pinneaple_solvers/02_solvers_hilbert_huang_decompose.py
python examples/pinneaple_solvers/03_solvers_fem_poisson_1d.py
python examples/pinneaple_solvers/04_solvers_fvm_advection_1d.py
python examples/pinneaple_solvers/05_solvers_catalog_and_adapters.py
Tips

If you want plots, you can add matplotlib quickly (kept out here to keep examples minimal).

For serious FEM/FVM workloads, replace the scaffold assembly/topology with your pinneaple_geom mesh and higher-fidelity operators.