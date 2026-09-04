import pytest

# A bare `import trimesh` here previously made the WHOLE test suite fail
# to collect (pytest aborts collection entirely on any module-level
# ImportError) for anyone without the optional `geom` extra installed --
# found via `pytest tests/` erroring out before running a single test.
# `importorskip` degrades this one file to a skip instead.
trimesh = pytest.importorskip("trimesh")
from pinneapple_geom.io.trimesh_bridge import TrimeshBridge
from pinneapple_geom.ops.repair import repair_mesh
from pinneapple_geom.ops.simplify import simplify_mesh

def test_repair_and_simplify_smoke():
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    bridge = TrimeshBridge()
    g = bridge.from_trimesh(mesh)
    g2 = repair_mesh(g)
    g3 = simplify_mesh(g2, target_faces=max(20, len(mesh.faces)//2))
    assert g3 is not None
