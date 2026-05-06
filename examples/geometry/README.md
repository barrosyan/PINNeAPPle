# pinneaple_geom examples

This folder focuses on geometry + meshing workflows that typically appear in Physics-AI / PINN pipelines:

- loading and repairing meshes
- sampling boundary / collocation points
- extracting simple geometric features
- building PINN-ready batches from raw geometry
- roundtripping through common file formats (STL, VTU/VTK via meshio)

## Quick run

From the repo root:

```bash
python examples/pinneaple_geom/01_load_mesh_and_sample.py
python examples/pinneaple_geom/02_repair_and_simplify.py
python examples/pinneaple_geom/03_parametric_boolean_and_export_stl.py
python examples/pinneaple_geom/04_curvature_importance_sampling.py
python examples/pinneaple_geom/05_stl_domain_batchbuilder_inlet_outlet_wall.py
python examples/pinneaple_geom/06_meshio_roundtrip_with_pointdata.py
```

Outputs (STL/VTU artifacts) are written to:

```text
examples/pinneaple_geom/_out/
```

## What each example demonstrates

### 01_load_mesh_and_sample.py
Load a trimesh geometry into Pinneaple's `MeshData` and sample many surface points.

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
