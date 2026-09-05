"""Validates the CGNS / Exodus / Fluent-mesh / Abaqus-``.inp`` readers in
``pinneapple_simulation/external_solvers/cfd_formats/`` against real files
produced by independent, real writers -- not just the self-consistent
synthetic files those readers were originally tested against (see each
module's own "Validation status" docstring section, and
``tests/fixtures/cfd_formats/README.md`` for exactly how each fixture here
was produced and what it does/doesn't cover).

All fixtures encode the same 8-node, 6-tetrahedron unit-cube mesh, with
``temperature = x + 2y + 3z`` and ``pressure = sqrt(x^2+y^2+z^2)`` as the two
vertex fields wherever the format carries fields at all -- both cheap to
recompute here and compare exactly, so a reader bug shows up as a wrong
number, not just a "did it crash" check.
"""
import os

import numpy as np
import pytest

from pinneapple_simulation.external_solvers.cfd_formats.abaqus_reader import (
    read_abaqus_inp_mesh,
)
from pinneapple_simulation.external_solvers.cfd_formats.cgns_reader import (
    read_cgns_mesh_and_fields,
)
from pinneapple_simulation.external_solvers.cfd_formats.exodus_reader import read_exodus
from pinneapple_simulation.external_solvers.cfd_formats.fluent_mesh_reader import (
    read_fluent_mesh_nodes,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "cfd_formats")

CUBE_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
])
CUBE_TEMPERATURE = CUBE_POINTS[:, 0] + 2 * CUBE_POINTS[:, 1] + 3 * CUBE_POINTS[:, 2]
CUBE_PRESSURE = np.linalg.norm(CUBE_POINTS, axis=1)


def test_cgns_reader_against_real_cgns_mll_file():
    """``real_cgns_mll.cgns`` was written by a small C program linked
    directly against the real, official CGNS Mid-Level Library (CGNS 4.5.2
    via Homebrew) -- not any code from this repository -- and independently
    passed the CGNS project's own ``cgnscheck`` validator before this test
    was written (see ``tests/fixtures/cfd_formats/README.md``)."""
    pytest.importorskip("h5py")
    path = os.path.join(FIXTURES, "real_cgns_mll.cgns")
    result = read_cgns_mesh_and_fields(path)

    assert result["coords"].shape == (8, 3)
    assert np.allclose(result["coords"], CUBE_POINTS)
    assert set(result["fields"].keys()) == {"Temperature", "Pressure"}
    assert np.allclose(result["fields"]["Temperature"], CUBE_TEMPERATURE)
    assert np.allclose(result["fields"]["Pressure"], CUBE_PRESSURE)


def test_exodus_reader_against_real_classic_netcdf_file():
    """``real_meshio_classic.exo`` was written by meshio 5.3.5's real Exodus
    writer (itself using the real ``netCDF4`` package), forced to classic
    NetCDF-3 -- the variant ``exodus_reader.py`` documents itself as
    supporting via ``scipy.io.netcdf_file``."""
    path = os.path.join(FIXTURES, "real_meshio_classic.exo")
    result = read_exodus(path)

    assert result["coords"].shape == (8, 3)
    assert np.allclose(result["coords"], CUBE_POINTS)
    assert set(result["fields"].keys()) == {"temperature", "pressure"}
    assert np.allclose(result["fields"]["temperature"], CUBE_TEMPERATURE)
    assert np.allclose(result["fields"]["pressure"], CUBE_PRESSURE)


def test_exodus_reader_rejects_netcdf4_variant_cleanly():
    """``real_meshio_netcdf4.exo`` is *also* a real, valid Exodus file (same
    meshio writer, no format override -- meshio's own default), but it is
    NetCDF-4/HDF5-based, which ``exodus_reader.py`` explicitly documents as
    out of scope for ``scipy.io.netcdf_file``. This regression-tests the
    clear ``NotImplementedError`` added after this exact file first
    surfaced a confusing raw ``scipy`` ``TypeError`` instead."""
    path = os.path.join(FIXTURES, "real_meshio_netcdf4.exo")
    with pytest.raises(NotImplementedError, match="HDF5-based"):
        read_exodus(path)


def test_fluent_mesh_reader_against_real_ansys_msh_file():
    """``real_meshio_ansys.msh`` was written by meshio 5.3.5's ``ansys``
    writer in ASCII mode -- meshio's own docstring for that module cites the
    same Fluent/TGrid user-guide appendix (section 10 = node coordinates)
    that ``fluent_mesh_reader.py`` implements."""
    path = os.path.join(FIXTURES, "real_meshio_ansys.msh")
    coords = read_fluent_mesh_nodes(path)

    assert coords.shape == (8, 3)
    assert np.allclose(coords, CUBE_POINTS)


def test_abaqus_inp_reader_against_real_meshio_file():
    """``real_meshio_abaqus.inp`` was written by meshio 5.3.5's ``abaqus``
    writer -- a plain ``*NODE``/``*ELEMENT, TYPE=C3D4`` keyword deck, the
    same open format ``read_abaqus_inp_mesh`` parses directly. This
    validates only the ``.inp`` mesh-reading half of ``abaqus_reader.py``;
    its ``.odb`` results bridge requires a real, licensed Abaqus
    installation to exercise at all and could not be tested in this
    environment (see the module docstring)."""
    path = os.path.join(FIXTURES, "real_meshio_abaqus.inp")
    mesh = read_abaqus_inp_mesh(path)

    assert mesh["coords"].shape == (8, 3)
    assert np.allclose(mesh["coords"], CUBE_POINTS)
    assert list(mesh["node_ids"]) == list(range(1, 9))
    assert "C3D4" in mesh["elements"]
    assert mesh["elements"]["C3D4"]["ids"].shape == (6,)
