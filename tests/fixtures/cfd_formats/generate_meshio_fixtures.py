"""Regenerates the meshio-written fixtures in this directory
(``real_meshio_classic.exo``, ``real_meshio_netcdf4.exo``,
``real_meshio_ansys.msh``, ``real_meshio_abaqus.inp``) from scratch, using
meshio's own real writers (not PINNeAPPle code) -- see ``README.md`` in this
directory for what each one validates and why.

Not a pytest test file (the fixtures it produces are already committed and
the reader tests run against those directly) -- this is a standalone repro
script, run manually with ``python generate_meshio_fixtures.py``, requiring
``meshio`` and ``netCDF4`` (neither is a PINNeAPPle runtime dependency;
install into a scratch venv, e.g. ``pip install meshio netCDF4``).
"""
import os

import meshio
import netCDF4
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Same 8-node / 6-tet unit-cube mesh used by the CGNS fixture (generate_cgns_fixture.c).
POINTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
], dtype=np.float64)

TETS_I64 = np.array([
    [0, 1, 3, 4],
    [1, 2, 3, 6],
    [1, 3, 4, 6],
    [3, 4, 6, 7],
    [1, 4, 5, 6],
    [0, 3, 4, 1],
], dtype=np.int64)

TEMPERATURE = POINTS[:, 0] + 2 * POINTS[:, 1] + 3 * POINTS[:, 2]
PRESSURE = np.linalg.norm(POINTS, axis=1)


def write_exodus_files():
    # NETCDF4/HDF5-based variant -- meshio's default (no format override).
    mesh = meshio.Mesh(
        points=POINTS, cells=[("tetra", TETS_I64)],
        point_data={"temperature": TEMPERATURE, "pressure": PRESSURE},
    )
    mesh.write(os.path.join(HERE, "real_meshio_netcdf4.exo"))

    # Classic NetCDF-3 variant -- what exodus_reader.py actually supports.
    # meshio's writer doesn't expose a `format=` kwarg, so force it by
    # monkeypatching netCDF4.Dataset (still the real netCDF4 encoder underneath).
    orig_dataset = netCDF4.Dataset

    def classic_dataset(*args, **kwargs):
        kwargs.setdefault("format", "NETCDF3_CLASSIC")
        return orig_dataset(*args, **kwargs)

    netCDF4.Dataset = classic_dataset
    try:
        # Classic NetCDF-3 quirk: createDimension(name, 0) is treated as a
        # second unlimited dimension (only one allowed; time_step already
        # claims that role) -- give the mesh one dummy point set so
        # num_node_sets=1 instead of 0. Also, classic NetCDF-3 has no 64-bit
        # int type, so connectivity must be int32.
        mesh_classic = meshio.Mesh(
            points=POINTS, cells=[("tetra", TETS_I64.astype(np.int32))],
            point_data={"temperature": TEMPERATURE, "pressure": PRESSURE},
            point_sets={"nset1": np.array([0, 1, 2], dtype=np.int32)},
        )
        mesh_classic.write(os.path.join(HERE, "real_meshio_classic.exo"))
    finally:
        netCDF4.Dataset = orig_dataset


def write_ansys_fluent_mesh():
    mesh = meshio.Mesh(points=POINTS, cells=[("tetra", TETS_I64)])
    # binary=False: fluent_mesh_reader.py only supports the ASCII encoding.
    mesh.write(os.path.join(HERE, "real_meshio_ansys.msh"), file_format="ansys", binary=False)


def write_abaqus_inp():
    mesh = meshio.Mesh(points=POINTS, cells=[("tetra", TETS_I64)])
    mesh.write(os.path.join(HERE, "real_meshio_abaqus.inp"))


if __name__ == "__main__":
    write_exodus_files()
    write_ansys_fluent_mesh()
    write_abaqus_inp()
    print("Regenerated real_meshio_{classic,netcdf4}.exo, real_meshio_ansys.msh, real_meshio_abaqus.inp")
