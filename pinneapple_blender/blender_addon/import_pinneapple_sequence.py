"""Runs INSIDE Blender's own Python (``bpy``) -- never imported by regular
PINNeAPPle code, same reasoning as ``cfd_formats/abaqus_reader.py``'s
``_abaqus_odb_export_script.py``: ``bpy`` is Blender's bundled interpreter,
not pip-installable into a normal environment, so this is launched as a
subprocess (``blender --background --python
import_pinneapple_sequence.py -- <args>``) by ``pinneapple_blender
.render.render_sequence``, not imported directly.

Imports every ``frame_XXXX.ply`` in a directory (as written by
``pinneapple_blender.export``), each into its own object, and keyframes
visibility so only frame N's object is visible at scene frame N -- the
simplest correct way to animate a sequence of separately-imported meshes
(as opposed to a single deforming mesh, which would need the point count
and topology to stay fixed across frames; a PLY-per-frame sequence makes
no such assumption, since a PINN surrogate evaluated at different
collocation samples per frame need not have a fixed point count at all).
"""
import glob
import os
import sys

import bpy


def _parse_args():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("usage: blender --background --python import_pinneapple_sequence.py -- <ply_dir> [output_blend]")
    args = argv[argv.index("--") + 1:]
    ply_dir = args[0]
    output_blend = args[1] if len(args) > 1 else None
    return ply_dir, output_blend


def main():
    ply_dir, output_blend = _parse_args()
    paths = sorted(glob.glob(os.path.join(ply_dir, "frame_*.ply")))
    if not paths:
        raise SystemExit(f"no frame_*.ply files found in {ply_dir}")

    # Start from a genuinely empty scene, not Blender's factory-default
    # Cube/Camera/Light -- confirmed by actually running this script and
    # inspecting the saved .blend that without this, the default Cube
    # was left behind, visible, sitting in the same scene as (and
    # visually obscuring, in a real test render) the imported point-cloud
    # frames. `use_empty=True` is the standard way to get a clean scene
    # from a headless script, rather than deleting the default objects
    # one by one after the fact.
    bpy.ops.wm.read_factory_settings(use_empty=True)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = len(paths) - 1

    for i, path in enumerate(paths):
        bpy.ops.wm.ply_import(filepath=path)
        obj = bpy.context.selected_objects[0]
        obj.name = f"pinneapple_frame_{i:04d}"

        # Visible only at its own frame: keyframe hide_viewport/hide_render
        # off at frame i, on everywhere else (the standard Blender pattern
        # for "swap which object is shown" animation).
        for f in range(len(paths)):
            visible = (f == i)
            obj.hide_viewport = not visible
            obj.hide_render = not visible
            obj.keyframe_insert(data_path="hide_viewport", frame=f)
            obj.keyframe_insert(data_path="hide_render", frame=f)

        # If the PLY carried per-vertex colour, add a Material using the
        # imported Color Attribute so it's visible in Eevee/Cycles render,
        # not just in vertex-paint view mode.
        if obj.data.color_attributes:
            mat = bpy.data.materials.new(name=f"pinneapple_mat_{i:04d}")
            # A material's node_tree (with a default Principled BSDF
            # already wired up) exists on creation in current Blender --
            # explicitly setting `use_nodes = True` is a no-op here and
            # raises "expected to be removed in Blender 6.0"
            # DeprecationWarning (confirmed against a real Blender 5.2.1
            # install), so it's dropped rather than kept for an
            # already-true default.
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            attr_node = mat.node_tree.nodes.new("ShaderNodeVertexColor")
            attr_node.layer_name = obj.data.color_attributes[0].name
            if bsdf is not None:
                mat.node_tree.links.new(attr_node.outputs["Color"], bsdf.inputs["Base Color"])
            obj.data.materials.append(mat)

    if output_blend:
        bpy.ops.wm.save_as_mainfile(filepath=output_blend)
        print(f"Saved {output_blend}")


if __name__ == "__main__":
    main()
