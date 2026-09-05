# Real-writer fixtures for the CGNS / Exodus / Fluent / Abaqus readers

These files were **not** constructed by `pinneapple_simulation`'s own reader
code (unlike the earlier, purely self-consistent testing of those readers --
see the "Validation status" section of each reader module's docstring). Each
one was produced by an independent, real writer -- a different program than
the one being tested -- then read back with the reader in
`pinneapple_simulation/external_solvers/cfd_formats/` and checked against the
mesh's known-exact coordinates/fields. All fixtures encode the same 8-node,
6-tetrahedron unit-cube mesh (the standard 6-tet decomposition of a cube),
with two vertex fields wherever the format supports fields at all:
`temperature = x + 2y + 3z` and `pressure = sqrt(x^2+y^2+z^2)` -- both cheap
to recompute by hand, so a reader bug shows up as a wrong number, not just a
crash.

## `real_cgns_mll.cgns`

Written by a small C program (`generate_cgns_fixture.c`, kept alongside this
README) that links directly against the **real, official CGNS Mid-Level
Library** (`libcgns`, CGNS 4.5.2, installed via `brew install cgns`, which
also pulled in HDF5 2.2.0) and calls its public API
(`cg_open`/`cg_base_write`/`cg_zone_write`/`cg_coord_write`/
`cg_section_write`/`cg_sol_write`/`cg_field_write`/`cg_close`) -- the same
reference implementation used by SU2 and the CGNS project's own tooling, and
literally the "CGNS's own reference `cgns_utils`" mentioned in
`cgns_reader.py`'s docstring as an ideal validation source.

Before being read by our own `cgns_reader.py`, this file was independently
validated by the CGNS project's **own** reference checker (also installed by
the same `brew install cgns`):

```
$ cgnscheck real_cgns_mll.cgns
...
checking complete
7 warnings (7 shown)   # all benign (missing optional dataclass/family-name
                        # metadata, and one "faces shared by >2 volumes"
                        # warning from the toy 6-tet cube decomposition not
                        # being a clean watertight mesh) -- zero errors.
$ cgnslist real_cgns_mll.cgns
HDF5 MotherNode
  +-CGNSLibraryVersion
  +-Base
    +-Zone1
      +-ZoneType
      +-GridCoordinates
      | +-CoordinateX / CoordinateY / CoordinateZ
      +-Elements
      +-FlowSolution
        +-Temperature / Pressure
```

To regenerate:
```
brew install cgns
cc -std=c11 -I$(brew --prefix cgns)/include -o generate_cgns_fixture \
    generate_cgns_fixture.c -L$(brew --prefix cgns)/lib -lcgns -lhdf5
./generate_cgns_fixture real_cgns_mll.cgns
```

**Not covered by this fixture**: multi-zone bases, cell-centered
(`CellCenter`) solutions (only `Vertex` was exercised), and non-tetrahedral
element types.

## `real_meshio_classic.exo` / `real_meshio_netcdf4.exo`

Written by [`meshio`](https://github.com/nschloe/meshio) 5.3.5's real Exodus
writer (`meshio/exodus/_exodus.py`, which itself calls the real `netCDF4`
Python package -- `netCDF4` 1.7.x, itself a binding onto the genuine
netcdf-c library -- to do the actual encoding). meshio is an independent,
widely-used mesh-format conversion library, not code written for this
repository.

- `real_meshio_classic.exo`: `netCDF4.Dataset` was called with
  `format="NETCDF3_CLASSIC"` (meshio's own writer doesn't expose this kwarg,
  so `netCDF4.Dataset` was monkeypatched to default to it before calling
  `mesh.write(...)`) -- this is the classic-NetCDF variant
  `exodus_reader.py` is documented to support (via
  `scipy.io.netcdf_file`), and is the one actually validated end-to-end
  against known coordinate/field values.
- `real_meshio_netcdf4.exo`: the *other* real, valid Exodus file meshio
  produces by default (no format override) -- NetCDF-4/HDF5-based. This is
  also genuine Exodus (many modern solvers default to this container for
  large meshes), but `exodus_reader.py` explicitly documents this variant as
  out of scope (it only reads classic NetCDF). This fixture exists to check
  that the reader **fails loudly and clearly** on it rather than
  mis-parsing or silently returning nothing -- see the
  `NotImplementedError` path added to `read_exodus()` as a direct result of
  testing against this exact file.

One netCDF-C quirk surfaced while generating the classic-format fixture:
`createDimension(name, 0)` (used by meshio for an empty `num_node_sets`
count) is treated as a **second unlimited dimension** by classic NetCDF-3,
which only allows one (`time_step` already claims that role) --
`NC_UNLIMITED size already in use`. Worked around by giving the mesh one
dummy point set (`nset1`) so `num_node_sets=1`; this is an artifact of the
classic-NetCDF container format, unrelated to anything `exodus_reader.py`
reads (it does not read node sets).

Connectivity was written as `int32` -- classic NetCDF-3 has no 64-bit
integer type, and meshio's writer does not downcast for you.

## `real_meshio_ansys.msh`

Written by meshio's `ansys` writer (`meshio/ansys/_ansys.py`, `binary=False`
i.e. the ASCII encoding), which targets literally the same format
`fluent_mesh_reader.py` documents itself against: ANSYS/Fluent/TGrid's
`.msh` mesh format (meshio's own module docstring cites the same TGrid
user-guide appendix). ASCII was requested explicitly since
`fluent_mesh_reader.py` only supports the ASCII encoding by design (see its
module docstring) -- meshio's default is the binary encoding.

## `real_meshio_abaqus.inp`

Written by meshio's `abaqus` writer (`meshio/abaqus/_abaqus.py`), a plain
`*NODE`/`*ELEMENT, TYPE=C3D4` keyword deck -- the same open, documented
format `abaqus_reader.py`'s `read_abaqus_inp_mesh()` parses directly. This
validates only the `.inp` mesh-reading half of `abaqus_reader.py`; the
`.odb` results bridge (`export_odb_fields`) is a subprocess call into a
real, licensed Abaqus installation and could not be exercised at all in
this environment -- see the module docstring and `AUDIT_REPORT.md` for why.

## What was checked and ruled out: meshio's own `.cgns` writer

Before writing the C/libcgns fixture above, `meshio.Mesh.write(..., 'cube.cgns')`
(meshio's own, built-in CGNS writer) was tried first, since it was the
lowest-effort option. It produces an HDF5 file, but inspecting it directly
with `h5py` showed every single node is missing the CGNS-mandated `label`
and `type` HDF5 attributes (`CGNSBase_t`, `Zone_t`, `DataArray_t`, ...) that
the SIDS-to-HDF5 mapping requires and that `cgns_reader.py` keys off of --
and indeed meshio's own writer source
(`meshio/cgns/_cgns.py`) contains a literal `# TODO something is missing
here` comment and a matching reader that is hardcoded to meshio's own
non-standard layout (`f["Base"]["Zone1"]["GridCoordinates"]["CoordinateX"]`
by fixed path, not by walking `label` attributes) and admits `# TODO how to
distinguish cell types` (hardcoded to always assume tetrahedra). This is not
a spec-compliant CGNS/HDF5 file -- only meshio's own matching reader can
parse it -- so it was **not** used as "real writer" evidence for
`cgns_reader.py`, which correctly refused to parse it (raised
`ValueError: no CGNSBase_t node found`, exactly as it should for a file with
no CGNS label attributes at all). The real-CGNS-library fixture above is
the one that actually validates this reader.
