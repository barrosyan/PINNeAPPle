# pinneapple_geom examples

This folder focuses on geometry + meshing workflows that typically appear in Physics-AI / PINN pipelines:

- loading and repairing meshes
- sampling boundary / collocation points
- extracting simple geometric features
- building PINN-ready batches from raw geometry
- roundtripping through common file formats (STL, VTU/VTK via meshio)

## Quick run

From the repo root:

```bash
python examples/pinneapple_geom/01_load_mesh_and_sample.py
python examples/pinneapple_geom/02_repair_and_simplify.py
python examples/pinneapple_geom/03_parametric_boolean_and_export_stl.py
python examples/pinneapple_geom/04_curvature_importance_sampling.py
python examples/pinneapple_geom/05_stl_domain_batchbuilder_inlet_outlet_wall.py
python examples/pinneapple_geom/06_meshio_roundtrip_with_pointdata.py
python examples/pinneapple_geom/08_open_cad_datasets_showcase.py
```

Outputs (STL/VTU artifacts) are written to:

```text
examples/pinneapple_geom/_out/
```

## What each example demonstrates

### 01_load_mesh_and_sample.py
Load a trimesh geometry into Pinneapple's `MeshData` and sample many surface points.

### 02_repair_and_simplify.py
Repair a mesh and simplify it to a target face count.

### 03_parametric_boolean_and_export_stl.py
Procedural primitives + optional boolean operations, export to STL, and load back.

### 04_curvature_importance_sampling.py
Compute a curvature proxy and bias sampling toward high-curvature regions.

### 05_stl_domain_batchbuilder_inlet_outlet_wall.py
End-to-end: STL -> interior collocation points + boundary points + normals + tag masks + BC masks.

### 06_meshio_roundtrip_with_pointdata.py
Save/load meshes via meshio while preserving point features (useful for hybrid supervised losses).

### 08_open_cad_datasets_showcase.py
Downloads a handful of real, individually-addressable sample files from open CAD/mesh
research datasets (Thingi10K STL models via their Hugging Face mirror; a real STEP
test fixture from the Fusion 360 Gallery Dataset repo) and runs them through
PINNeAPPle's real importers (`pinneapple_data.stl_import.load_stl`,
`pinneapple_design.geometry.io.step.step_to_mesh`), then extracts features
(`pinneapple_data.geometry_features`) and voxelizes the result
(`pinneapple_design.geometry.ops.voxelize`). Downloaded files are written to
`examples/geometry/outputs/` and are not committed to the repo. Requires network
access; the STEP path additionally needs the optional `meshio`/`gmsh` dependencies
and degrades gracefully (with a clear message) if they're missing.
