# pinneaple_pinn examples

This folder focuses on the **PINN-specific** layer of PINNeAPPle.

## Quickstart

```bash
python examples/pinneaple_pinn/01_symbolic_loss_basic.py
```

## Compiler API (autograd PDEs)

- `02_compiler_poisson_2d.py`
  - Poisson (2D) on a unit square with Dirichlet BC
  - Uses `pinneaple_environment.ProblemSpec` + `pinneaple_pinn.compile_problem()`

- `03_compiler_burgers_1d.py`
  - Viscous Burgers (1D + time) with initial and boundary conditions

## Symbolic API (SymPy -> torch)

- `04_factory_inverse_parameter_heat_1d.py`
  - Heat equation with *unknown* `alpha` learned as an inverse parameter
  - Uses `PINNFactory`, symbolic residual strings, and sparse supervised data

## UPD integration (Zarr -> sampled batches)

- `05_upd_dataset_data_only_regression.py`
  - Creates a tiny synthetic Zarr shard, samples points with `UPDDataset`,
    and trains a data-only neural surrogate through `PINNFactory`.
