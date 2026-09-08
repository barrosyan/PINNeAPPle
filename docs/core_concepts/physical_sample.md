# PhysicalSample

`PhysicalSample` is the framework's unified container for a piece of
physical data — a UPD-aligned (Unified Physical Data) record used
consistently across modeling, simulation, and PINN training so that data
coming from a numerical solver, an experiment, or a synthetic generator all
look the same to the rest of the pipeline.

## Where it lives

`pinneapple_data.physical_sample.PhysicalSample` (re-exported at
`pinneapple_data`). It's a dataclass with five fields:

- `state: xr.Dataset | Dict[str, Any]` — the core physical state, preferably
  an `xarray.Dataset` for structured grid data, or a dict for other layouts.
- `geometry: Optional[Any]` — an optional geometry asset describing the
  spatial structure the state lives on.
- `schema: Dict[str, Any]` — governing equations, boundary/initial
  conditions, forcing terms, unit policies.
- `domain: Dict[str, Any]` — domain interpretation metadata (grid, mesh,
  graph, points, ...).
- `provenance: Dict[str, Any]` — lineage metadata: identifiers, source,
  tiling, time span, and other traceability information.
- `extras: Dict[str, Any]` — an open-ended slot for feature caches, mesh
  labels, SDFs, and anything else that doesn't fit the above.

It also has small helper methods, e.g. `domain_type()` (reads
`domain["type"]`, defaulting to `"grid"` for an `xarray.Dataset` state) and
`is_grid()`.

## What else lives in `pinneapple_data`

- **Storage** — `UPDZarrStore` / `ZarrUPDIterable` for Zarr-backed datasets
  (plus HDF5 and PyTorch-native storage elsewhere in the package).
- **Collation** — `collate_upd_supervised`, `collate_pinn_batches`,
  `move_batch_to_device` for turning a list of samples into a training
  batch.
- **Collocation & active learning** — `CollocationSampler`/
  `CollocationConfig` for drawing collocation points, and
  `ActiveLearningConfig`/`ResidualBasedAL`/`VarianceBasedAL`/`CombinedAL`/
  `AdaptiveCollocationTrainer` for residual- or variance-driven adaptive
  sampling during training.
- **Transforms** — `Normalizer`, `StandardScaler`, `MinMaxScaler`.
- **Splits** — `SplitSpec`/`split_indices` for train/val/test partitioning.
- **PINN batch builders** — `build_from_bundle`, `build_from_solver`,
  `build_from_real_data` produce a `PINNBatch` from a data bundle, a
  `pinneapple_simulation` solver run, or real measured data respectively.
- `STLMesh`/`load_stl`/`load_stl_bytes` for mesh import, and `UPDItem`/
  `ConditionSpec`/`SamplingSpec`/`Batch` as the lower-level typed containers
  `PhysicalSample`-based datasets ultimately produce.

## How it's used

A numerical solver run
([Geometry & Domain](geometry_domain.md) sampling +
`pinneapple_simulation`), an experimental measurement, or a
`pinneapple_pdb`-built dataset can all be wrapped as one or more
`PhysicalSample` objects. From there, `pinneapple_data`'s collation and
collocation utilities turn them into the batches that
[Model](model.md)/[PINN / Physics](pinn.md)/[Solver](solver.md) consume —
`PhysicalSample` is the hand-off point between "where did this data come
from" and "how do I train on it."
