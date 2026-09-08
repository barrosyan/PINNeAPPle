"""Tests for pinneapple_design.geometry.io.iges (IGES -> mesh import).

Neither ``gmsh`` nor ``meshio`` are installed in this environment (checked at
collection time below), so the real read/mesh path cannot be exercised here.
These tests therefore always cover config/validation logic and the
missing-dependency error paths, and additionally cover a real gmsh/OCC round
trip on a synthetic (in-process, gmsh-generated) IGES fixture *only* when
both optional dependencies happen to be present.
"""
from __future__ import annotations

import pytest

from pinneapple_design.geometry.io.iges import IgesImportConfig, iges_to_mesh


def _meshio_available() -> bool:
    try:
        import meshio  # noqa: F401
        return True
    except Exception:
        return False


def _gmsh_available() -> bool:
    try:
        import gmsh  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Config / validation (no optional deps required)
# ---------------------------------------------------------------------------

def test_iges_import_config_defaults():
    cfg = IgesImportConfig()
    assert cfg.kind == "surface"
    assert cfg.mesh_size == pytest.approx(0.02)
    assert cfg.algorithm_2d == 6
    assert cfg.algorithm_3d == 1
    assert cfg.curvature_refine is True
    assert cfg.optimize is True
    assert cfg.heal_shapes is True


def test_iges_import_config_override():
    cfg = IgesImportConfig(
        kind="volume",
        mesh_size=0.5,
        algorithm_2d=1,
        algorithm_3d=4,
        curvature_refine=False,
        optimize=False,
        heal_shapes=False,
    )
    assert cfg.kind == "volume"
    assert cfg.mesh_size == pytest.approx(0.5)
    assert cfg.algorithm_2d == 1
    assert cfg.algorithm_3d == 4
    assert cfg.curvature_refine is False
    assert cfg.optimize is False
    assert cfg.heal_shapes is False


# ---------------------------------------------------------------------------
# Missing-dependency error paths
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    _meshio_available(),
    reason="meshio is installed; this test targets the missing-meshio error path",
)
def test_iges_to_mesh_raises_clear_error_without_meshio(tmp_path):
    fake = tmp_path / "part.iges"
    fake.write_text("not a real iges file")
    with pytest.raises(RuntimeError, match="meshio is required"):
        iges_to_mesh(fake)


@pytest.mark.skipif(
    not _meshio_available() or _gmsh_available(),
    reason="requires meshio present and gmsh absent to target the missing-gmsh error path",
)
def test_iges_to_mesh_raises_clear_error_without_gmsh(tmp_path):
    fake = tmp_path / "part.iges"
    fake.write_text("not a real iges file")
    with pytest.raises(ImportError, match="gmsh is required"):
        iges_to_mesh(fake)


def test_iges_to_mesh_missing_file_raises_after_dependency_checks(tmp_path):
    if not (_meshio_available() and _gmsh_available()):
        pytest.skip("gmsh and meshio must both be installed to reach the file-existence check")
    missing = tmp_path / "does_not_exist.iges"
    with pytest.raises(FileNotFoundError):
        iges_to_mesh(missing)


# ---------------------------------------------------------------------------
# Real round trip (only runs when gmsh + meshio are both available)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_meshio_available() and _gmsh_available()),
    reason="gmsh and/or meshio not installed in this environment; cannot exercise a real IGES round trip",
)
def test_iges_to_mesh_round_trip_synthetic_box(tmp_path):
    import gmsh

    # Synthetic fixture: a plain unit box generated in-process via gmsh's own
    # OCC kernel and written out as IGES. This is NOT sourced from any
    # external CAD file -- it exists purely to exercise the real gmsh/OCC
    # IGES read + tessellation path end-to-end in this test.
    iges_path = tmp_path / "unit_box.iges"
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("synthetic_box")
        gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.write(str(iges_path))
    finally:
        gmsh.finalize()

    mesh = iges_to_mesh(iges_path, cfg=IgesImportConfig(mesh_size=0.3))
    assert mesh.vertices.shape[1] == 3
    assert mesh.faces.shape[1] == 3
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0
