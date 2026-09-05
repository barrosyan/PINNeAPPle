"""Regression tests for the Blender bridge (see ``ROADMAP_PHYSICS_AI_HUB.md``,
"Blender bridge" row and P2.3): ``pinneapple_blender.export`` (no Blender
needed, pure numpy/matplotlib) and the ``bpy``-side add-on
(``pinneapple_blender/blender_addon/import_pinneapple_sequence.py``),
subprocess-bridged via ``pinneapple_blender.render.build_scene`` exactly
like the Abaqus ``.odb`` bridge.

Earlier in this session, ``ROADMAP_PHYSICS_AI_HUB.md`` carried the caveat
that the ``bpy``-side add-on "could not be executed in this session -- no
local Blender install." A real Blender install was added in this
follow-up pass specifically to close that gap, and doing so surfaced two
real, previously-undetected bugs (both fixed here, both regression-tested
below):

1. ``export.py``'s ``_colormap`` used ``matplotlib.cm.get_cmap``, which
   was removed in matplotlib 3.9 (deprecated since 3.7) -- on this
   machine's installed matplotlib 3.11.1 it silently raised
   ``AttributeError``, caught by a broad ``except Exception``, and fell
   through to a plain grayscale ramp. Every previously-exported PLY got
   grayscale vertex colours, never the requested colormap, with no
   warning. Confirmed by inspecting a real exported frame's raw RGB
   triplet (R==G==B) before the fix.
2. The add-on script never cleared Blender's factory-default Cube/
   Camera/Light before importing the PLY sequence -- confirmed by
   opening a real generated ``.blend`` and listing its objects: `Cube`,
   `Light`, `Camera` were all present, visible, and (in an actual test
   render) the default Cube obscured the imported point cloud entirely.
   Fixed via ``bpy.ops.wm.read_factory_settings(use_empty=True)`` at the
   start of the import.

The tests below that require a real Blender executable are skipped (not
failed) when one isn't on PATH, matching this repo's established
external-tool-dependency convention (see e.g.
``test_perception.py``'s ``ffmpeg`` skip).
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pinneapple_blender.export import _colormap, export_scene, export_trajectory
from pinneapple_blender.render import build_scene

_ADDON_SCRIPT = Path(__file__).resolve().parent.parent / "pinneapple_blender" / "blender_addon" / "import_pinneapple_sequence.py"


# ---------------------------------------------------------------------------
# Pure-numpy/matplotlib tests (no Blender required)
# ---------------------------------------------------------------------------

def test_colormap_produces_non_grayscale_colors_for_varying_values():
    """Regression test for bug 1: a genuine colormap (viridis) must
    produce R!=G!=B for at least some values -- a grayscale fallback
    (silently triggered by the removed matplotlib.cm.get_cmap API)
    always has R==G==B for every point, which this asserts against."""
    values = np.linspace(0.0, 1.0, 50)
    colors = _colormap(values, cmap_name="viridis")
    assert colors.shape == (50, 3)
    assert colors.dtype == np.uint8
    not_grayscale = ~((colors[:, 0] == colors[:, 1]) & (colors[:, 1] == colors[:, 2]))
    assert not_grayscale.any(), (
        "every color came out R==G==B (grayscale) -- the viridis colormap path is not "
        "actually being used (see the matplotlib.cm.get_cmap regression this test guards against)"
    )


def test_colormap_endpoints_match_known_viridis_rgb():
    """viridis(0.0) and viridis(1.0) are well-known, stable RGB values
    (dark purple / bright yellow) -- pin them down exactly so a future
    matplotlib API change is caught immediately rather than silently
    falling back to grayscale again."""
    colors = _colormap(np.array([0.0, 1.0]), cmap_name="viridis")
    # viridis(0.0) ~ (68, 1, 84); viridis(1.0) ~ (253, 231, 36/37) (matplotlib's
    # own published table -- the last channel's exact truncated-vs-rounded
    # uint8 value depends on the cast, hence the 1-off tolerance below rather
    # than a second hardcoded exact triplet).
    assert tuple(colors[0]) == (68, 1, 84)
    r, g, b = colors[1]
    assert (int(r), int(g)) == (253, 231)
    assert abs(int(b) - 37) <= 1


def test_export_trajectory_writes_valid_ply_sequence_with_real_colors(tmp_path):
    rng = np.random.default_rng(0)
    points = rng.uniform(-1, 1, size=(20, 3)).astype(np.float32)
    fields = [np.sin(points[:, 0] + t) for t in np.linspace(0, np.pi, 3)]
    paths = export_trajectory(points, fields, str(tmp_path), field_name="u")

    assert len(paths) == 3
    for p in paths:
        assert Path(p).exists()

    text = Path(paths[0]).read_text()
    assert text.startswith("ply\n")
    assert "element vertex 20" in text
    assert "property uchar red" in text

    # Parse the RGB triplets out of the vertex rows and confirm real
    # color variation across points (not a flat grayscale ramp for
    # every single vertex, which would indicate the fallback fired).
    lines = text.strip().split("\n")
    header_end = lines.index("end_header")
    rgb_triplets = []
    for row in lines[header_end + 1:]:
        cols = row.split()
        r, g, b = int(cols[3]), int(cols[4]), int(cols[5])
        rgb_triplets.append((r, g, b))
    not_grayscale = [t for t in rgb_triplets if not (t[0] == t[1] == t[2])]
    assert len(not_grayscale) > 0


# ---------------------------------------------------------------------------
# Real-Blender integration tests (skipped if Blender isn't installed)
# ---------------------------------------------------------------------------

def _blender_available() -> bool:
    return shutil.which("blender") is not None


def _list_scene_objects(blend_path: str) -> list:
    """Shell out to a second, independent Blender invocation to list every
    object in the saved scene -- deliberately NOT reusing any PINNeAPPle
    code path, so this is a genuine independent check of what
    ``build_scene`` actually wrote to disk, not a self-consistency check
    of the same code that wrote it."""
    script = f"""
import bpy, sys
bpy.ops.wm.open_mainfile(filepath={blend_path!r})
for o in bpy.context.scene.objects:
    print("OBJ", o.name, o.type, o.hide_viewport, o.hide_render)
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name
    proc = subprocess.run(
        ["blender", "--background", "--python", script_path],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    objects = []
    for line in proc.stdout.splitlines():
        if line.startswith("OBJ "):
            _, name, otype, hide_vp, hide_render = line.split()
            objects.append((name, otype, hide_vp == "True", hide_render == "True"))
    return objects


@pytest.mark.skipif(not _blender_available(), reason="blender not installed -- cannot test the real bpy-side add-on")
def test_build_scene_runs_real_blender_and_produces_a_blend_file(tmp_path):
    rng = np.random.default_rng(1)
    points = rng.uniform(-1, 1, size=(50, 3)).astype(np.float32)
    fields = [np.cos(points[:, 2] * 2 + t) for t in np.linspace(0, np.pi, 3)]
    ply_dir = tmp_path / "plys"
    export_trajectory(points, fields, str(ply_dir), field_name="p")

    out_blend = tmp_path / "scene.blend"
    result = build_scene(str(ply_dir), output_blend=str(out_blend))

    assert result == str(out_blend)
    assert out_blend.exists()
    assert out_blend.stat().st_size > 0


@pytest.mark.skipif(not _blender_available(), reason="blender not installed -- cannot test the real bpy-side add-on")
def test_build_scene_produces_a_clean_scene_with_no_leftover_defaults(tmp_path):
    """Regression test for bug 2: the saved .blend must contain ONLY the
    imported frame objects, not Blender's factory-default Cube/Camera/
    Light -- confirmed via a real, independent second Blender process
    listing every object in the saved file (not just trusting that
    build_scene() returned without raising)."""
    rng = np.random.default_rng(2)
    points = rng.uniform(-1, 1, size=(30, 3)).astype(np.float32)
    fields = [np.sin(points[:, 0] + t) for t in np.linspace(0, np.pi, 3)]
    ply_dir = tmp_path / "plys"
    export_trajectory(points, fields, str(ply_dir), field_name="u")

    out_blend = tmp_path / "scene.blend"
    build_scene(str(ply_dir), output_blend=str(out_blend))

    objects = _list_scene_objects(str(out_blend))
    names = {o[0] for o in objects}
    types = {o[1] for o in objects}

    assert "Cube" not in names, f"leftover default Cube found in objects: {names}"
    assert "CAMERA" not in types, f"leftover default Camera found: {objects}"
    assert "LIGHT" not in types, f"leftover default Light found: {objects}"
    assert len(objects) == 3, f"expected exactly 3 imported frame objects, got: {objects}"
    assert all(name.startswith("pinneapple_frame_") for name in names)


@pytest.mark.skipif(not _blender_available(), reason="blender not installed -- cannot test the real bpy-side add-on")
def test_build_scene_keyframes_visibility_so_exactly_one_frame_is_visible_at_a_time(tmp_path):
    rng = np.random.default_rng(3)
    points = rng.uniform(-1, 1, size=(30, 3)).astype(np.float32)
    n_frames = 4
    fields = [np.sin(points[:, 0] + t) for t in np.linspace(0, np.pi, n_frames)]
    ply_dir = tmp_path / "plys"
    export_trajectory(points, fields, str(ply_dir), field_name="u")

    out_blend = tmp_path / "scene.blend"
    build_scene(str(ply_dir), output_blend=str(out_blend))

    script = f"""
import bpy
bpy.ops.wm.open_mainfile(filepath={str(out_blend)!r})
scene = bpy.context.scene
assert scene.frame_start == 0
assert scene.frame_end == {n_frames - 1}
for f in range(scene.frame_start, scene.frame_end + 1):
    scene.frame_set(f)
    visible = [o.name for o in scene.objects if o.type == 'MESH' and not o.hide_render]
    assert len(visible) == 1, (f, visible)
    assert visible[0] == f"pinneapple_frame_{{f:04d}}", (f, visible)
print("KEYFRAME_CHECK_OK")
"""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script)
        script_path = f.name
    proc = subprocess.run(
        ["blender", "--background", "--python", script_path],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "KEYFRAME_CHECK_OK" in proc.stdout


@pytest.mark.skipif(not _blender_available(), reason="blender not installed -- cannot test the real bpy-side add-on")
def test_render_sequence_missing_blender_executable_raises_filenotfounderror(tmp_path):
    rng = np.random.default_rng(4)
    points = rng.uniform(-1, 1, size=(10, 3)).astype(np.float32)
    ply_dir = tmp_path / "plys"
    export_scene(points, np.zeros(10), str(ply_dir))

    with pytest.raises(FileNotFoundError):
        build_scene(str(ply_dir), blender_executable="definitely_not_a_real_executable_xyz")
