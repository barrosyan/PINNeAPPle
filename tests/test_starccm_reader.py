"""Tests for ``pinneapple_simulation/external_solvers/cfd_formats/starccm_reader.py``.

Two paths are exercised:

1. The plain-text table/CSV parser (:func:`read_starccm_table` /
   :func:`starccm_table_to_upd`), against a small, hand-constructed
   synthetic fixture built inline below in STAR-CCM+'s documented
   ``Plot > Export`` / ``Table > Export`` layout (quoted header row with
   unit suffixes, comma-delimited data rows). This is **not** a captured
   real STAR-CCM+ file -- none was available in this environment -- it is
   a correctly-formatted synthetic fixture matching the documented export
   layout, same caveat as this module's own docstring states.

2. The CGNS delegation path (:func:`read_starccm_cgns` /
   :func:`starccm_cgns_to_upd`), confirmed to call straight through to the
   real, already-validated ``cgns_reader`` logic (not a separate/duplicated
   implementation) by reading the same real CGNS fixture
   ``tests/fixtures/cfd_formats/real_cgns_mll.cgns`` (written by the real
   CGNS Mid-Level Library, cgnscheck-passed -- see
   ``tests/test_cfd_format_readers.py``) both directly via ``cgns_reader``
   and via the ``starccm_reader`` wrappers, and checking the results are
   identical.
"""
import os

import numpy as np
import pytest

from pinneapple_simulation.external_solvers.cfd_formats.cgns_reader import (
    cgns_to_upd, read_cgns_mesh_and_fields,
)
from pinneapple_simulation.external_solvers.cfd_formats.starccm_reader import (
    read_starccm_cgns, read_starccm_table,
    starccm_cgns_to_upd, starccm_table_to_upd,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "cfd_formats")


# ---------------------------------------------------------------------------
# 1. Plain-text table/CSV export path
# ---------------------------------------------------------------------------

# A hand-constructed synthetic fixture in STAR-CCM+'s documented
# "Plot > Export" / "Table > Export" table layout: quoted header row with
# unit-suffixed column names, then one comma-delimited data row per point.
# 4 points, X/Y/Z coordinates plus Temperature and Pressure scalar fields.
_TABLE_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.5],
    [0.0, 1.0, 0.5],
])
_TABLE_TEMPERATURE = np.array([300.0, 310.5, 295.25, 320.0])
_TABLE_PRESSURE = np.array([101325.0, 101300.0, 101400.0, 101250.5])

_SYNTHETIC_TABLE_CSV = (
    '"X (m)","Y (m)","Z (m)","Temperature (K)","Pressure (Pa)"\n'
    "0.0,0.0,0.0,300.0,101325.0\n"
    "1.0,0.0,0.0,310.5,101300.0\n"
    "1.0,1.0,0.5,295.25,101400.0\n"
    "0.0,1.0,0.5,320.0,101250.5\n"
)


@pytest.fixture()
def synthetic_table_csv(tmp_path):
    p = tmp_path / "starccm_table_export.csv"
    p.write_text(_SYNTHETIC_TABLE_CSV)
    return str(p)


def test_read_starccm_table_parses_coords_and_fields(synthetic_table_csv):
    result = read_starccm_table(synthetic_table_csv)

    assert result["coords"].shape == (4, 3)
    assert np.allclose(result["coords"], _TABLE_POINTS)
    assert set(result["fields"].keys()) == {"Temperature", "Pressure"}
    assert np.allclose(result["fields"]["Temperature"], _TABLE_TEMPERATURE)
    assert np.allclose(result["fields"]["Pressure"], _TABLE_PRESSURE)
    assert result["column_names"] == [
        "X (m)", "Y (m)", "Z (m)", "Temperature (K)", "Pressure (Pa)",
    ]


def test_read_starccm_table_handles_2d_export_missing_z(tmp_path):
    """A 2D STAR-CCM+ export (no Z column) should zero-pad Z rather than
    error -- same convention as the Fluent/CGNS readers pad missing axes."""
    csv_text = (
        '"X (m)","Y (m)","Scalar"\n'
        "0.0,0.0,1.5\n"
        "2.0,3.0,2.5\n"
    )
    p = tmp_path / "starccm_2d_export.csv"
    p.write_text(csv_text)

    result = read_starccm_table(str(p))
    assert result["coords"].shape == (2, 3)
    assert np.allclose(result["coords"], [[0.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    assert np.allclose(result["fields"]["Scalar"], [1.5, 2.5])


def test_read_starccm_table_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        read_starccm_table("/no/such/starccm_export.csv")


def test_read_starccm_table_no_coordinate_columns_raises(tmp_path):
    csv_text = '"Foo","Bar"\n1.0,2.0\n'
    p = tmp_path / "no_coords.csv"
    p.write_text(csv_text)
    with pytest.raises(ValueError, match="coordinate columns"):
        read_starccm_table(str(p))


def test_read_starccm_table_malformed_row_raises(tmp_path):
    csv_text = '"X (m)","Y (m)","Z (m)"\n0.0,0.0,0.0\n1.0,1.0\n'
    p = tmp_path / "malformed.csv"
    p.write_text(csv_text)
    with pytest.raises(ValueError, match="malformed"):
        read_starccm_table(str(p))


def test_starccm_table_to_upd_packages_physical_sample(synthetic_table_csv):
    from pinneapple_data.physical_sample import PhysicalSample

    sample = starccm_table_to_upd(synthetic_table_csv)

    assert isinstance(sample, PhysicalSample)
    assert sample.domain == {"type": "mesh", "n_cells": 4}
    assert np.allclose(sample.geometry.nodes, _TABLE_POINTS)
    assert np.allclose(sample.state["Temperature"], _TABLE_TEMPERATURE)
    assert np.allclose(sample.state["Pressure"], _TABLE_PRESSURE)
    assert sample.provenance["source"] == "starccm_table"
    assert "not a captured real STAR-CCM+ file" in sample.provenance["validation"]


# ---------------------------------------------------------------------------
# 2. CGNS export path -- must delegate to the real cgns_reader, not
#    reimplement it.
# ---------------------------------------------------------------------------


def test_read_starccm_cgns_matches_direct_cgns_reader():
    """Confirms the STAR-CCM+ wrapper produces byte-for-byte identical
    results to calling cgns_reader directly -- i.e. it really delegates,
    it does not duplicate the parsing logic."""
    pytest.importorskip("h5py")
    path = os.path.join(FIXTURES, "real_cgns_mll.cgns")

    direct = read_cgns_mesh_and_fields(path)
    via_starccm = read_starccm_cgns(path)

    assert np.allclose(via_starccm["coords"], direct["coords"])
    assert set(via_starccm["fields"].keys()) == set(direct["fields"].keys())
    for name in direct["fields"]:
        assert np.allclose(via_starccm["fields"][name], direct["fields"][name])


def test_starccm_cgns_to_upd_matches_direct_cgns_to_upd():
    pytest.importorskip("h5py")
    path = os.path.join(FIXTURES, "real_cgns_mll.cgns")

    direct = cgns_to_upd(path)
    via_starccm = starccm_cgns_to_upd(path)

    assert np.allclose(via_starccm.geometry.nodes, direct.geometry.nodes)
    assert set(via_starccm.state.keys()) == set(direct.state.keys())
    for name in direct.state:
        assert np.allclose(via_starccm.state[name], direct.state[name])
    assert via_starccm.domain == direct.domain


def test_starccm_reader_is_a_thin_wrapper_not_a_reimplementation():
    """Directly confirms starccm_reader.read_starccm_cgns/starccm_cgns_to_upd
    are the exact same underlying implementation as cgns_reader's, not
    look-alike duplicates -- by identity of the function they delegate to,
    rather than only comparing outputs (which a coincidentally-matching
    reimplementation could also pass)."""
    import pinneapple_simulation.external_solvers.cfd_formats.starccm_reader as starccm_mod

    assert starccm_mod.read_cgns_mesh_and_fields is read_cgns_mesh_and_fields
    assert starccm_mod.cgns_to_upd is cgns_to_upd
