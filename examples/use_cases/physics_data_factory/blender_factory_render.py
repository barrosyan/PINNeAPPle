# -*- coding: utf-8 -*-
"""Blender Cycles — Industrial Pump Station Photorealistic Render
=================================================================

Run with:
    blender --background --python blender_factory_render.py -- --frames 192 --output ./outputs/blender

OR via the helper launcher:
    python blender_factory_launcher.py

What this script builds inside Blender
--------------------------------------
Scene — Industrial pump station with Cycles path-tracing:

  Environment
    Concrete floor (Principled BSDF — rough grey with bump)
    Concrete walls (matte grey with micro-detail)
    Industrial ceiling with skylights

  Equipment
    3x Centrifugal pumps (blue-painted steel, Principled BSDF, metallic=0.6)
    Pipe network with flanges and valves (brushed stainless steel)
    Shell-and-tube heat exchanger (polished stainless, high metallic)
    Yellow pipe insulation sleeves (matte plastic)
    Catwalk / maintenance platform (galvanised steel)
    Overhead pipe rack (structural steel)
    Pressure gauges (chrome dial + glass lens)
    Control panel (painted metal + LED indicators)

  Lighting
    HDRI environment (procedural sky — industrial hall)
    6x Area lights as industrial LED fixtures (5500K)
    1x Key sun light from skylight (8500K, soft)

  Camera
    Animated path — 7 key positions over N frames
    Depth of field (f/4, focus on main equipment)

  Render settings
    Cycles with path tracing
    Denoiser: Intel Open Image Denoise (OIDN) / NVIDIA Optix if GPU
    Samples: 256 (CPU) or 512 (GPU)
    Resolution: 1920x1080
    Color management: Filmic, High Contrast
"""
from __future__ import annotations

import math
import sys
import os
from pathlib import Path

# ── Parse command-line args passed after "--" ──────────────────────────────
argv  = sys.argv
args  = argv[argv.index("--") + 1:] if "--" in argv else []
_a    = {args[i][2:]: args[i+1] for i in range(0, len(args)-1, 2) if args[i].startswith("--")}

N_FRAMES   = int(_a.get("frames",  192))
# Always resolve to absolute path — Blender background mode can have unexpected CWD
_default_out = str(Path(__file__).parent / "outputs" / "blender")
OUTPUT_DIR   = Path(_a.get("output", _default_out)).resolve()
FPS        = int(_a.get("fps",      24))
W          = int(_a.get("width",  1920))
H          = int(_a.get("height", 1080))
SAMPLES    = int(_a.get("samples",  256))   # increase for cleaner result

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import bpy
import bmesh
from mathutils import Vector, Euler, Matrix
import numpy as np

RNG = np.random.default_rng(0)


# ============================================================================
# UTILITIES
# ============================================================================

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)


def link(obj):
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    return obj


def mat_principled(name, base_color, metallic=0.0, roughness=0.5,
                   specular=0.5, ior=1.45, clearcoat=0.0,
                   emission=(0,0,0,1), emission_strength=0.0) -> bpy.types.Material:
    """Create a Principled BSDF material."""
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt  = mat.node_tree
    nt.nodes.clear()
    out  = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
    nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    bsdf.inputs['Base Color'].default_value    = (*base_color[:3], 1.0)
    bsdf.inputs['Metallic'].default_value      = metallic
    bsdf.inputs['Roughness'].default_value     = roughness
    bsdf.inputs['Specular IOR Level'].default_value = specular
    bsdf.inputs['IOR'].default_value           = ior
    bsdf.inputs['Coat Weight'].default_value   = clearcoat
    if emission_strength > 0:
        bsdf.inputs['Emission Color'].default_value    = (*emission[:3], 1.0)
        bsdf.inputs['Emission Strength'].default_value = emission_strength
    return mat


def mat_concrete(name="concrete") -> bpy.types.Material:
    """Concrete floor/wall with procedural bump."""
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt  = mat.node_tree
    nt.nodes.clear()
    out    = nt.nodes.new('ShaderNodeOutputMaterial')
    bsdf   = nt.nodes.new('ShaderNodeBsdfPrincipled')
    noise  = nt.nodes.new('ShaderNodeTexNoise')
    bump   = nt.nodes.new('ShaderNodeBump')
    ramp   = nt.nodes.new('ShaderNodeValToRGB')
    coords = nt.nodes.new('ShaderNodeTexCoord')
    mappi  = nt.nodes.new('ShaderNodeMapping')
    # Layout
    for n, x in [(coords,-800),(mappi,-600),(noise,-400),(bump,-200),(ramp,-200),(bsdf,0),(out,300)]:
        n.location = (x, 0)
    noise.inputs['Scale'].default_value  = 12.0
    noise.inputs['Detail'].default_value = 8.0
    noise.inputs['Roughness'].default_value = 0.7
    bump.inputs['Strength'].default_value   = 0.25
    ramp.color_ramp.elements[0].color = (0.40, 0.40, 0.38, 1.0)
    ramp.color_ramp.elements[1].color = (0.58, 0.57, 0.54, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.92
    bsdf.inputs['Metallic'].default_value  = 0.0
    nt.links.new(coords.outputs['Generated'], mappi.inputs['Vector'])
    nt.links.new(mappi.outputs['Vector'],     noise.inputs['Vector'])
    nt.links.new(noise.outputs['Fac'],        ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'],       bsdf.inputs['Base Color'])
    nt.links.new(noise.outputs['Fac'],        bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'],      bsdf.inputs['Normal'])
    nt.links.new(bsdf.outputs['BSDF'],        out.inputs['Surface'])
    return mat


def assign(obj, mat):
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


# ============================================================================
# GEOMETRY BUILDERS
# ============================================================================

def add_cylinder(loc, rot=(0,0,0), radius=0.05, depth=1.0,
                 verts=32, name="cyl") -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=verts, radius=radius, depth=depth,
        location=loc, rotation=rot)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def add_pipe(p0, p1, radius=0.055, verts=20, name="pipe") -> bpy.types.Object:
    """Add a cylinder connecting two 3D points."""
    v0, v1 = Vector(p0), Vector(p1)
    mid    = (v0 + v1) / 2
    length = (v1 - v0).length
    diff   = v1 - v0
    # Rotation: align Z-axis to pipe direction
    z      = Vector((0, 0, 1))
    rot    = z.rotation_difference(diff.normalized()).to_euler()
    return add_cylinder(loc=mid, rot=rot, radius=radius, depth=length, verts=verts, name=name)


def add_flange(loc, axis, r_out=0.095, thick=0.025, name="flange") -> bpy.types.Object:
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=32, radius=r_out, depth=thick,
        location=loc,
        rotation=Vector((0,0,1)).rotation_difference(Vector(axis).normalized()).to_euler())
    obj = bpy.context.active_object
    obj.name = name
    # Drill inner hole using boolean (simplified: just mark as outer disc)
    return obj


def add_box(loc, dims, name="box", rot=(0,0,0)) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = (dims[0]/2, dims[1]/2, dims[2]/2)
    bpy.ops.object.transform_apply(scale=True)
    return obj


def add_sphere(loc, radius=0.5, verts=32, name="sphere") -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=verts, ring_count=verts//2, radius=radius, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def add_torus(loc, major_r=0.15, minor_r=0.055, name="elbow") -> bpy.types.Object:
    bpy.ops.mesh.primitive_torus_add(
        major_radius=major_r, minor_radius=minor_r,
        major_segments=24, minor_segments=12,
        location=loc)
    obj = bpy.context.active_object
    obj.name = name
    return obj


def join_objects(objs) -> bpy.types.Object:
    bpy.ops.object.select_all(action='DESELECT')
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    return bpy.context.active_object


def smooth(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    return obj


# ============================================================================
# MATERIAL LIBRARY
# ============================================================================

def mat_painted_metal(name, base_color, roughness_paint=0.55) -> bpy.types.Material:
    """
    Industrial painted metal: matte paint base with procedural scratches and
    surface imperfections — reveals bare metal under the paint in worn areas.
    Much more realistic than a flat Principled BSDF for pump casings.
    """
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out    = nt.nodes.new('ShaderNodeOutputMaterial')
    mix    = nt.nodes.new('ShaderNodeMixShader')       # blend paint ↔ bare metal
    paint  = nt.nodes.new('ShaderNodeBsdfPrincipled')  # painted surface
    metal  = nt.nodes.new('ShaderNodeBsdfPrincipled')  # exposed bare metal
    noise1 = nt.nodes.new('ShaderNodeTexNoise')        # large scratch pattern
    noise2 = nt.nodes.new('ShaderNodeTexNoise')        # fine bump texture
    bump   = nt.nodes.new('ShaderNodeBump')
    ramp   = nt.nodes.new('ShaderNodeValToRGB')        # threshold scratches
    coord  = nt.nodes.new('ShaderNodeTexCoord')

    # Paint layer — opaque matte industrial paint
    paint.inputs['Base Color'].default_value   = (*base_color, 1.0)
    paint.inputs['Metallic'].default_value     = 0.0
    paint.inputs['Roughness'].default_value    = roughness_paint
    paint.inputs['Specular IOR Level'].default_value = 0.15

    # Bare metal layer — shows where paint is chipped
    paint_rgb = base_color
    metal.inputs['Base Color'].default_value   = (0.55, 0.56, 0.58, 1.0)
    metal.inputs['Metallic'].default_value     = 0.95
    metal.inputs['Roughness'].default_value    = 0.22

    # Scratch / chip pattern (low frequency)
    noise1.inputs['Scale'].default_value      = 8.0
    noise1.inputs['Detail'].default_value     = 4.0
    noise1.inputs['Roughness'].default_value  = 0.8
    noise1.inputs['Distortion'].default_value = 0.5

    # Ramp: convert noise to binary scratch mask
    ramp.color_ramp.interpolation = 'CONSTANT'
    ramp.color_ramp.elements[0].position = 0.80   # 20% of surface is chipped
    ramp.color_ramp.elements[0].color    = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = 0.85
    ramp.color_ramp.elements[1].color    = (1.0, 1.0, 1.0, 1.0)

    # Surface bump (fine manufacturing texture)
    noise2.inputs['Scale'].default_value     = 40.0
    noise2.inputs['Detail'].default_value    = 8.0
    noise2.inputs['Roughness'].default_value = 0.6
    bump.inputs['Strength'].default_value    = 0.15
    bump.inputs['Distance'].default_value    = 0.003

    # Connect nodes
    nt.links.new(coord.outputs['Object'],  noise1.inputs['Vector'])
    nt.links.new(coord.outputs['Object'],  noise2.inputs['Vector'])
    nt.links.new(noise1.outputs['Fac'],    ramp.inputs['Fac'])
    nt.links.new(ramp.outputs['Color'],    mix.inputs['Fac'])
    nt.links.new(noise2.outputs['Fac'],    bump.inputs['Height'])
    nt.links.new(bump.outputs['Normal'],   paint.inputs['Normal'])
    nt.links.new(bump.outputs['Normal'],   metal.inputs['Normal'])
    nt.links.new(paint.outputs['BSDF'],    mix.inputs[1])
    nt.links.new(metal.outputs['BSDF'],    mix.inputs[2])
    nt.links.new(mix.outputs['Shader'],    out.inputs['Surface'])
    return mat


def build_materials():
    mats = {}
    # Brushed stainless steel  (pipes, flanges)
    mats["steel"]      = mat_principled("steel",      (0.72,0.72,0.74), metallic=0.92, roughness=0.20)
    # Polished stainless  (heat exchanger shell)
    mats["stainless"]  = mat_principled("stainless",  (0.80,0.82,0.84), metallic=0.96, roughness=0.08, clearcoat=0.2)
    # Industrial pump casing — matte paint with wear and scratch imperfections
    mats["pump_blue"]  = mat_painted_metal("pump_blue",  (0.06, 0.22, 0.55), roughness_paint=0.60)
    # Motor housing — slightly darker navy blue paint (same wear material)
    mats["pump_motor"] = mat_painted_metal("pump_motor", (0.04, 0.16, 0.42), roughness_paint=0.65)
    # Yellow pipe insulation  (rock-wool / armaflex wrap)
    mats["insulation"] = mat_principled("insulation",  (0.82, 0.60, 0.04), metallic=0.0, roughness=0.92)
    # Gate valve body  (dark cast iron)
    mats["valve"]      = mat_principled("valve",       (0.08, 0.08, 0.10), metallic=0.65, roughness=0.50)
    # Chrome / gauge face and dial
    mats["chrome"]     = mat_principled("chrome",      (0.92, 0.92, 0.90), metallic=0.98, roughness=0.05)
    # Concrete floor / wall with procedural aggregate texture
    mats["concrete"]   = mat_concrete("concrete")
    # Galvanised catwalk grating
    mats["galv"]       = mat_principled("galv",        (0.50, 0.54, 0.54), metallic=0.72, roughness=0.42)
    # Structural dark steel (columns, beams)
    mats["struct"]     = mat_principled("struct",      (0.10, 0.12, 0.13), metallic=0.88, roughness=0.58)
    # LED panel (warm white emissive)
    mats["led"]        = mat_principled("led", (1.0, 0.97, 0.90),
                                        emission=(1.0, 0.97, 0.90),
                                        emission_strength=8.0, metallic=0.0, roughness=0.3)
    # Control panel housing  (dark painted steel)
    mats["panel"]      = mat_principled("panel",       (0.10, 0.14, 0.18), metallic=0.42, roughness=0.58)
    # SCADA screen  (green emissive)
    mats["screen"]     = mat_principled("screen", (0.0, 0.75, 0.18),
                                        emission=(0.0, 0.75, 0.18),
                                        emission_strength=4.0, metallic=0.0, roughness=0.2)
    return mats


# ============================================================================
# PHYSICS DATA  (drives pipe colours per frame)
# ============================================================================

PIPE_DEFS = [
    # (p0, p1, radius, tag)
    ((-4.0, 0.58, 0.0),  (-1.2, 0.58, 0.0),  0.058, "intake"),
    ((-4.0, 0.58, 2.5),  (-1.2, 0.58, 2.5),  0.058, "intake"),
    ((-4.0, 0.58, 5.0),  (-1.2, 0.58, 5.0),  0.058, "intake"),
    ((-0.5, 0.58, 0.0),  (-0.5, 1.10, 0.0),  0.058, "riser"),
    ((-0.5, 0.58, 2.5),  (-0.5, 1.10, 2.5),  0.058, "riser"),
    ((-0.5, 0.58, 5.0),  (-0.5, 1.10, 5.0),  0.058, "riser"),
    ((-0.5, 1.10, 0.0),  (2.6,  1.10, 0.0),  0.058, "discharge"),
    ((-0.5, 1.10, 2.5),  (2.6,  1.10, 2.5),  0.058, "discharge"),
    ((-0.5, 1.10, 5.0),  (2.6,  1.10, 5.0),  0.058, "discharge"),
    ((2.6,  1.10, 0.0),  (2.6,  1.10, 5.0),  0.090, "manifold"),
    ((2.6,  1.10, 2.5),  (4.5,  1.10, 2.5),  0.090, "hot"),
    ((7.8,  1.10, 2.5),  (11.0, 1.10, 2.5),  0.070, "discharge_f"),
    ((7.8,  0.38, 2.5),  (2.6,  0.38, 2.5),  0.045, "recirc"),
]

INSULATION_DEFS = [
    ((2.6,  1.10, 2.5), (4.5, 1.10, 2.5), 0.090+0.040),
    ((7.8,  1.10, 2.5), (11.0,1.10, 2.5), 0.070+0.035),
    ((-0.5, 1.10, 0.0), (2.6, 1.10, 0.0), 0.058+0.032),
    ((-0.5, 1.10, 5.0), (2.6, 1.10, 5.0), 0.058+0.032),
]

PUMP_XYZ = [(-0.5, 0.15, 0.0), (-0.5, 0.15, 2.5), (-0.5, 0.15, 5.0)]
HX = (4.5, 7.8, 0.15, 1.12, 0.8, 4.2)   # x0,x1, y0,y1, z0,z1


def pressure_color(p: float):
    """[0,1] -> RGB for pipe pressure visualisation."""
    p = float(np.clip(p, 0, 1))
    if p < 0.5:
        return (0.0, p*2*0.5+0.1, 0.7 + p*0.2)
    else:
        q = (p - 0.5)*2
        return (0.1 + q*0.8, 0.6 - q*0.5, 0.4 - q*0.35)


def physics_timeline(n_frames):
    t    = np.linspace(0, 6*math.pi, n_frames)
    freq = np.array([0.8, 1.2, 1.7])
    P    = np.zeros((n_frames, len(PIPE_DEFS)))
    for i, (*_, tag) in enumerate(PIPE_DEFS):
        base = {"intake":0.28,"riser":0.50,"discharge":0.68,
                "manifold":0.82,"hot":0.75,"discharge_f":0.60,"recirc":0.35}.get(tag,0.4)
        P[:, i] = np.clip(base + 0.14*np.sin(freq[i%3]*t + i), 0, 1)
    T_in  = 78 + 9*np.sin(0.18*t)
    T_out = 44 + 5*np.sin(0.18*t + 0.3)
    return P, T_in, T_out


# ============================================================================
# SCENE BUILDER
# ============================================================================

def build_scene(mats):
    objs = {}

    # ── Floor ─────────────────────────────────────────────────────────────
    o = add_box((3.5, -0.05, 2.5), (18.0, 0.10, 10.0), name="floor")
    assign(o, mats["concrete"]); smooth(o)
    objs["floor"] = o

    # ── Walls ─────────────────────────────────────────────────────────────
    back_wall = add_box((3.5, 2.6, -1.9), (18.0, 5.2, 0.2), name="wall_back")
    assign(back_wall, mats["concrete"])
    left_wall = add_box((-4.95, 2.6, 2.5), (0.1, 5.2, 10.0), name="wall_left")
    assign(left_wall, mats["concrete"])
    right_wall = add_box((12.35, 2.6, 2.5), (0.1, 5.2, 10.0), name="wall_right")
    assign(right_wall, mats["concrete"])

    # ── Ceiling ───────────────────────────────────────────────────────────
    ceil_o = add_box((3.5, 5.1, 2.5), (18.0, 0.1, 10.0), name="ceiling")
    assign(ceil_o, mats["concrete"])

    # ── Structural columns ────────────────────────────────────────────────
    for xi in [-4.2, 2.5, 7.2, 11.5]:
        for zi in [-0.3, 5.3]:
            c = add_cylinder((xi, 2.5, zi), radius=0.08, depth=5.0, name=f"col_{xi}_{zi}")
            assign(c, mats["struct"]); smooth(c)

    # ── Overhead pipe rack ────────────────────────────────────────────────
    for zi in [-0.3, 5.3]:
        beam = add_pipe((-4.5, 3.8, zi), (11.5, 3.8, zi), radius=0.06, name=f"rack_{zi}")
        assign(beam, mats["struct"]); smooth(beam)
    for xi in [-4.2, 2.5, 7.2, 11.5]:
        cross = add_pipe((xi, 3.8, -0.3), (xi, 3.8, 5.3), radius=0.05, name=f"crossbeam_{xi}")
        assign(cross, mats["struct"]); smooth(cross)

    # ── Plinths ───────────────────────────────────────────────────────────
    for i, (px, _, pz) in enumerate(PUMP_XYZ):
        p = add_box((px, 0.08, pz), (0.9, 0.16, 0.8), name=f"plinth_{i}")
        assign(p, mats["concrete"])

    # ── Pumps ─────────────────────────────────────────────────────────────
    for i, (px, py, pz) in enumerate(PUMP_XYZ):
        # ── Volute casing: flat oval cross-section cylinder (looks like
        #    a real pump casing, not a balloon sphere) ──────────────────
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=48, radius=0.30, depth=0.38,
            location=(px, py+0.20, pz),
            rotation=(0, 0, math.pi/2))   # axis along X
        v = bpy.context.active_object
        v.name = f"volute_{i}"
        # Flatten the cross-section: squash in Z to look oval from the side
        v.scale.z = 0.80
        v.scale.y = 0.85
        bpy.ops.object.transform_apply(scale=True)
        assign(v, mats["pump_blue"]); smooth(v)

        # ── Split-case bolting flange (equatorial ring) ────────────────
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=0.32, depth=0.04,
            location=(px, py+0.20, pz),
            rotation=(0, math.pi/2, 0))
        flange_eq = bpy.context.active_object
        flange_eq.name = f"volute_flange_{i}"
        assign(flange_eq, mats["pump_blue"]); smooth(flange_eq)

        # ── Bolt studs around the split-case flange (8 bolts) ─────────
        for b in range(8):
            ang = b * math.pi / 4
            bx = px
            by = py + 0.20 + 0.30 * math.sin(ang)
            bz = pz            + 0.30 * math.cos(ang)
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=8, radius=0.018, depth=0.06,
                location=(bx, by, bz),
                rotation=(0, math.pi/2, 0))
            bolt = bpy.context.active_object
            bolt.name = f"bolt_v_{i}_{b}"
            assign(bolt, mats["chrome"]); smooth(bolt)

        # ── Motor housing (cylindrical, darker blue) ───────────────────
        m = add_cylinder((px-0.48, py+0.20, pz), rot=(0, math.pi/2, 0),
                         radius=0.175, depth=0.60, name=f"motor_{i}")
        assign(m, mats["pump_motor"]); smooth(m)

        # Motor end cap (flat disc)
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=32, radius=0.176, depth=0.02,
            location=(px-0.80, py+0.20, pz),
            rotation=(0, math.pi/2, 0))
        end_cap = bpy.context.active_object
        end_cap.name = f"motor_endcap_{i}"
        assign(end_cap, mats["pump_motor"]); smooth(end_cap)

        # Motor cooling fins (thin cylinder rings)
        for fi_idx, fx in enumerate([-0.48, -0.58, -0.68]):
            bpy.ops.mesh.primitive_torus_add(
                major_radius=0.195, minor_radius=0.012,
                major_segments=32, minor_segments=8,
                location=(px + fx, py+0.20, pz))
            fin = bpy.context.active_object
            fin.rotation_euler = Euler((0, math.pi/2, 0))
            fin.name = f"motor_fin_{i}_{fi_idx}"
            assign(fin, mats["pump_motor"]); smooth(fin)

        # ── Coupling guard (polished steel tube) ──────────────────────
        g = add_cylinder((px-0.16, py+0.20, pz), rot=(0, math.pi/2, 0),
                         radius=0.115, depth=0.14, name=f"guard_{i}")
        assign(g, mats["steel"]); smooth(g)

        # ── Suction nozzle (Z direction, entering volute side) ────────
        sn = add_pipe((px, py+0.20, pz-0.28), (px, py+0.20, pz-0.55),
                      radius=0.072, name=f"suction_{i}")
        assign(sn, mats["steel"]); smooth(sn)
        # Suction flange
        sf = add_cylinder((px, py+0.20, pz-0.32), radius=0.10, depth=0.025,
                          name=f"suc_fl_{i}")
        assign(sf, mats["steel"]); smooth(sf)

        # ── Discharge nozzle (Y direction, going up to header) ────────
        dn = add_pipe((px, py+0.34, pz), (px, py+0.68, pz),
                      radius=0.062, name=f"discharge_n_{i}")
        assign(dn, mats["steel"]); smooth(dn)
        # Discharge flange
        df = add_cylinder((px, py+0.38, pz), radius=0.090, depth=0.025,
                          name=f"dsc_fl_{i}")
        assign(df, mats["steel"]); smooth(df)

        # ── Baseplate (gusset style) ───────────────────────────────────
        bp = add_box((px-0.10, py+0.01, pz), (0.95, 0.03, 0.72), name=f"baseplate_{i}")
        assign(bp, mats["struct"])

        # Anchor bolts (4 corners)
        for abx, abz in [(px-0.35,pz-0.28),(px-0.35,pz+0.28),
                          (px+0.20,pz-0.28),(px+0.20,pz+0.28)]:
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=8, radius=0.022, depth=0.12,
                location=(abx, py+0.06, abz))
            ab = bpy.context.active_object
            ab.name = f"anchor_{i}_{abx}"
            assign(ab, mats["chrome"]); smooth(ab)

        # ── Motor fan cover (perforated look) ─────────────────────────
        fc = add_cylinder((px-0.78, py+0.20, pz), rot=(0, math.pi/2, 0),
                          radius=0.185, depth=0.025, name=f"fancover_{i}")
        assign(fc, mats["steel"]); smooth(fc)

        # ── Nameplate (small rectangle on casing) ─────────────────────
        np_obj = add_box((px, py+0.42, pz), (0.14, 0.002, 0.08), name=f"nameplate_{i}")
        assign(np_obj, mats["chrome"])

        objs[f"pump_{i}"] = v

    # ── Pipe network ──────────────────────────────────────────────────────
    pipe_objs = []
    for idx, (p0, p1, r, tag) in enumerate(PIPE_DEFS):
        po = add_pipe(p0, p1, radius=r, verts=24, name=f"pipe_{idx}")
        assign(po, mats["steel"]); smooth(po)
        pipe_objs.append(po)
        # Flanges at each end
        d = (Vector(p1) - Vector(p0)).normalized()
        for pt, sign in [(p0, 1), (p1, -1)]:
            fc_loc = Vector(pt) + d * sign * 0.08
            fl = add_cylinder(tuple(fc_loc),
                              rot=d.rotation_difference(Vector((0,0,1))).to_euler(),
                              radius=r+0.04, depth=0.025, name=f"fl_{idx}_{sign}")
            assign(fl, mats["steel"]); smooth(fl)

    objs["pipes"] = pipe_objs

    # ── Insulation sleeves ────────────────────────────────────────────────
    for j, (p0, p1, ri) in enumerate(INSULATION_DEFS):
        ins = add_pipe(p0, p1, radius=ri, verts=20, name=f"insul_{j}")
        assign(ins, mats["insulation"]); smooth(ins)

    # ── Valves ────────────────────────────────────────────────────────────
    valve_positions = [
        ((-3.0, 0.58, 0.0), (0,0,1)),
        ((-3.0, 0.58, 2.5), (0,0,1)),
        ((-3.0, 0.58, 5.0), (0,0,1)),
        (( 3.8, 1.10, 2.5), (1,0,0)),
        (( 9.0, 1.10, 2.5), (1,0,0)),
    ]
    for j, ((vx,vy,vz), ax) in enumerate(valve_positions):
        body = add_cylinder((vx,vy,vz),
                            rot=Vector((0,0,1)).rotation_difference(Vector(ax)).to_euler(),
                            radius=0.10, depth=0.17, name=f"valve_body_{j}")
        assign(body, mats["valve"]); smooth(body)
        stem = add_cylinder((vx, vy+0.22, vz), radius=0.022, depth=0.28, name=f"vstem_{j}")
        assign(stem, mats["steel"]); smooth(stem)
        wheel = add_cylinder((vx, vy+0.38, vz), radius=0.12, depth=0.018, name=f"vwheel_{j}")
        assign(wheel, mats["steel"]); smooth(wheel)

    # ── Heat Exchanger ────────────────────────────────────────────────────
    x0,x1,y0,y1,z0,z1 = HX
    cx=(x0+x1)/2; cy=(y0+y1)/2; cz=(z0+z1)/2; L=x1-x0
    shell = add_cylinder((cx,cy,cz), rot=(0,math.pi/2,0), radius=0.48, depth=L,
                         verts=48, name="hx_shell")
    assign(shell, mats["stainless"]); smooth(shell)
    for hx_x, hx_name in [(x0-0.12,"head_l"), (x1+0.12,"head_r")]:
        h = add_sphere((hx_x,cy,cz), radius=0.49, verts=32, name=hx_name)
        assign(h, mats["stainless"]); smooth(h)
    # Nozzles
    for nx,ny,nz,nr in [(x0,cy+0.58,cz,0.085),(x1,cy+0.58,cz,0.085),
                         (x0,cy-0.58,cz,0.072),(x1,cy-0.58,cz,0.072)]:
        n = add_pipe((nx,ny,nz), (nx,ny+0.22,nz), radius=nr, name=f"hx_noz_{nx}{ny}")
        assign(n, mats["stainless"]); smooth(n)
    # Support saddles
    for sx in [x0+0.4, x1-0.4]:
        sad = add_box((sx, (y0+cy-0.46)/2, cz), (0.3, (cy-0.46-y0), 1.2), name=f"saddle_{sx}")
        assign(sad, mats["struct"])

    # ── Catwalk ───────────────────────────────────────────────────────────
    cat_plat = add_box((-1.5, 1.30, 2.5), (1.0, 0.04, 6.0), name="catwalk_platform")
    assign(cat_plat, mats["galv"])
    for zi in np.linspace(-0.5, 5.5, 5):
        for xi in [-2.0, -1.0]:
            post = add_cylinder((xi, 1.75, zi), radius=0.018, depth=0.90, name=f"cat_post_{xi}_{zi}")
            assign(post, mats["galv"]); smooth(post)
    for xi in [-2.0, -1.0]:
        rail = add_pipe((xi, 2.20, -0.5), (xi, 2.20, 5.5), radius=0.022, name=f"cat_rail_{xi}")
        assign(rail, mats["galv"]); smooth(rail)

    # ── LED ceiling fixture array ─────────────────────────────────────────
    light_locs = [
        (-1.5, 4.9, 0.5), (-1.5, 4.9, 4.5),
        ( 3.0, 4.9, 0.0), ( 3.0, 4.9, 5.0),
        ( 7.0, 4.9, 0.0), ( 7.0, 4.9, 5.0),
        ( 9.5, 4.9, 2.5),
    ]
    for j, (lx,ly,lz) in enumerate(light_locs):
        # Fixture housing
        house = add_box((lx,ly+0.04,lz), (0.45, 0.08, 0.22), name=f"fixture_h_{j}")
        assign(house, mats["struct"])
        # LED panel (emissive)
        panel = add_box((lx,ly,lz), (0.40, 0.005, 0.18), name=f"fixture_led_{j}")
        assign(panel, mats["led"])

    # ── Control panel ─────────────────────────────────────────────────────
    cp = add_box((6.0, 1.20, -1.72), (1.2, 1.1, 0.14), name="control_panel")
    assign(cp, mats["panel"])
    scr = add_box((6.0, 1.55, -1.74), (0.90, 0.55, 0.01), name="screen")
    assign(scr, mats["screen"])

    # ── Pressure gauges ───────────────────────────────────────────────────
    gauge_locs = [(1.5,1.48,0.0),(1.5,1.48,2.5),(1.5,1.48,5.0),(2.6,1.48,4.0),(6.2,1.50,2.5)]
    for j, (gx,gy,gz) in enumerate(gauge_locs):
        dial = add_cylinder((gx,gy+0.08,gz), radius=0.082, depth=0.016, name=f"gauge_dial_{j}")
        assign(dial, mats["chrome"]); smooth(dial)
        gstem = add_cylinder((gx,gy,gz), radius=0.014, depth=0.10, name=f"gauge_stem_{j}")
        assign(gstem, mats["steel"]); smooth(gstem)

    return objs, light_locs


# ============================================================================
# LIGHTING
# ============================================================================

def setup_lighting(light_locs):
    # Clear existing lights
    for o in list(bpy.data.objects):
        if o.type == 'LIGHT':
            bpy.data.objects.remove(o, do_unlink=True)

    # ── Sun / Skylight — enters through high industrial windows ──────────────
    bpy.ops.object.light_add(type='SUN', location=(8, 12, 4))
    sun = bpy.context.active_object
    sun.name = "sun_key"
    sun.data.energy  = 5.0        # brighter sun for clear industrial day
    sun.data.color   = (1.00, 0.97, 0.90)   # slightly warm (morning light)
    sun.data.angle   = math.radians(2.5)     # soft penumbra shadows
    sun.rotation_euler = Euler((math.radians(-50), 0, math.radians(25)))

    # ── Large diffuse fill — skylights / north window ─────────────────────
    bpy.ops.object.light_add(type='AREA', location=(3.5, 8.0, 2.5))
    fill = bpy.context.active_object
    fill.name = "sky_fill"
    fill.data.energy  = 120.0     # strong fill to illuminate shadow side
    fill.data.color   = (0.82, 0.90, 1.00)  # cool blue-sky fill
    fill.data.size    = 8.0
    fill.rotation_euler = Euler((math.radians(90), 0, 0))

    # ── Floor bounce — light bounced from concrete floor ──────────────────
    bpy.ops.object.light_add(type='AREA', location=(3.5, -1.5, 2.5))
    bounce = bpy.context.active_object
    bounce.name = "floor_bounce"
    bounce.data.energy  = 40.0
    bounce.data.color   = (0.92, 0.90, 0.85)   # warm concrete-tinted bounce
    bounce.data.size    = 10.0
    bounce.rotation_euler = Euler((math.radians(-90), 0, 0))

    # ── Ceiling industrial LED high-bay fixtures ──────────────────────────
    for j, (lx, ly, lz) in enumerate(light_locs):
        bpy.ops.object.light_add(type='AREA', location=(lx, ly-0.06, lz))
        led = bpy.context.active_object
        led.name = f"led_light_{j}"
        led.data.energy  = 500.0     # bright industrial high-bay (5000 lm equiv.)
        led.data.color   = (1.00, 0.96, 0.82)   # warm white LED (4000K)
        led.data.size    = 0.50
        led.data.size_y  = 0.20
        led.data.spread  = math.radians(130)
        led.rotation_euler = Euler((math.radians(180), 0, 0))

    # ── Rim / back light — separates dark equipment from walls ────────────
    bpy.ops.object.light_add(type='AREA', location=(3.5, 4.8, 10.0))
    rim = bpy.context.active_object
    rim.name = "rim_light"
    rim.data.energy  = 90.0
    rim.data.color   = (0.78, 0.85, 1.00)
    rim.data.size    = 5.0
    rim.rotation_euler = Euler((math.radians(28), 0, 0))


# ============================================================================
# CAMERA ANIMATION
# ============================================================================

CAM_KEYS = [
    # (frame_pct, eye,                         focal_point,          f/stop)
    # All positions inside the plant (z:-1.5..7, x:-4.5..12, y:0.1..4.8)
    # Eye level at y=1.65-1.80 = realistic industrial inspector viewpoint

    # Shot 1 — Wide master: diagonal from front-right, shows full plant depth
    (0.00, ( 9.5, 1.75,  1.5),  (0.5, 0.85, 2.5),  5.6),

    # Shot 2 — Close-up: pump array side view, piping in foreground
    (0.20, ( 3.0, 1.65,  0.5),  (-0.5, 0.90, 2.5), 2.8),

    # Shot 3 — Walking shot: facing the 3 pumps from the walkway at left wall
    (0.40, (-3.2, 1.68,  2.5),  (3.0, 0.85, 2.5),  4.0),

    # Shot 4 — HX approach: from back-right, looking toward manifold & HX
    (0.57, ( 9.0, 1.72,  6.5),  (4.0, 0.85, 2.5),  4.0),

    # Shot 5 — HX close-up: ground level beside the heat exchanger vessel
    (0.73, ( 5.5, 1.65,  5.5),  (6.2, 0.90, 2.5),  2.0),

    # Shot 6 — Low angle: dramatic low-angle looking up at pipes and ceiling lights
    (0.87, ( 2.0, 0.80,  2.5),  (4.0, 2.50, 2.5),  4.0),

    # Shot 7 — Return to master
    (1.00, ( 9.5, 1.75,  1.5),  (0.5, 0.85, 2.5),  5.6),
]


def setup_camera(n_frames: int):
    bpy.ops.object.camera_add(location=(9, 2.8, -0.4))
    cam_obj  = bpy.context.active_object
    cam_obj.name = "Factory_Camera"
    cam      = cam_obj.data
    cam.lens         = 35.0      # 35mm focal length
    cam.dof.use_dof  = True
    cam.dof.aperture_fstop  = 4.0
    cam.dof.focus_distance  = 8.0
    bpy.context.scene.camera = cam_obj

    # Insert keyframes along path
    for (pct, eye, foc, fstop) in CAM_KEYS:
        frame = max(1, int(pct * n_frames))
        cam_obj.location = eye
        # Point camera at focal point
        direction = Vector(foc) - Vector(eye)
        rot = direction.to_track_quat('-Z', 'Y').to_euler()
        cam_obj.rotation_euler = rot
        cam.dof.aperture_fstop = fstop
        cam.dof.focus_distance = direction.length
        cam_obj.keyframe_insert('location',        frame=frame)
        cam_obj.keyframe_insert('rotation_euler',  frame=frame)
        cam.keyframe_insert('dof.aperture_fstop',  frame=frame)
        cam.keyframe_insert('dof.focus_distance',  frame=frame)

    # Smooth interpolation
    if cam_obj.animation_data:
        for fc in cam_obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'BEZIER'
    return cam_obj


# ============================================================================
# RENDER SETTINGS
# ============================================================================

def setup_render(output_dir: Path, n_frames: int):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end   = n_frames
    scene.render.fps  = FPS

    # Output
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '16'
    scene.render.filepath = str(output_dir / "frame_####")

    # Resolution
    scene.render.resolution_x = W
    scene.render.resolution_y = H
    scene.render.resolution_percentage = 100

    # Cycles
    scene.render.engine            = 'CYCLES'
    scene.cycles.device            = 'CPU'   # switch to 'GPU' if available
    scene.cycles.samples           = SAMPLES
    scene.cycles.use_denoising     = True
    scene.cycles.denoiser          = 'OPENIMAGEDENOISE'
    scene.cycles.use_adaptive_sampling     = True
    scene.cycles.adaptive_threshold        = 0.01
    scene.cycles.max_bounces        = 12
    scene.cycles.diffuse_bounces    = 4
    scene.cycles.glossy_bounces     = 4
    scene.cycles.transmission_bounces = 8

    # Color management — Filmic for photorealism
    scene.view_settings.view_transform  = 'Filmic'
    scene.view_settings.look            = 'High Contrast'
    scene.view_settings.exposure        = 0.3
    scene.view_settings.gamma           = 1.0

    # World (background / environment light)
    world = bpy.data.worlds['World']
    world.use_nodes = True
    wnt = world.node_tree
    wnt.nodes.clear()
    bg   = wnt.nodes.new('ShaderNodeBackground')
    sky  = wnt.nodes.new('ShaderNodeTexSky')
    out  = wnt.nodes.new('ShaderNodeOutputWorld')
    sky.sky_type = 'NISHITA'
    sky.sun_elevation   = math.radians(35)
    sky.sun_rotation    = math.radians(120)
    sky.sun_intensity   = 1.0
    sky.altitude        = 100.0
    bg.inputs['Strength'].default_value = 0.20   # subtle env light
    wnt.links.new(sky.outputs['Color'], bg.inputs['Color'])
    wnt.links.new(bg.outputs['Background'], out.inputs['Surface'])

    # Try GPU
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        scene.cycles.device = 'GPU'
        print("GPU render enabled")
    except Exception:
        print("CPU render (no GPU detected)")


# ============================================================================
# PIPE COLOUR ANIMATION
# ============================================================================

def animate_pipe_colours(pipe_objs: list, n_frames: int):
    """Keyframe pipe material base colour per frame for pressure animation."""
    P, T_in, T_out = physics_timeline(n_frames)
    for fi in range(1, n_frames+1, max(1, n_frames//60)):   # keyframe every ~4 frames
        bpy.context.scene.frame_set(fi)
        ti = fi - 1
        for si, po in enumerate(pipe_objs):
            col = pressure_color(float(P[ti, si]))
            mat = po.data.materials[0]
            bsdf = next(n for n in mat.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
            bsdf.inputs['Base Color'].default_value = (*col, 1.0)
            bsdf.inputs['Base Color'].keyframe_insert('default_value', frame=fi)
    bpy.context.scene.frame_set(1)


# ============================================================================
# PUMP ROTATION ANIMATION
# ============================================================================

def animate_pumps(pump_objs: dict, n_frames: int):
    """Rotate pump volutes to simulate spinning."""
    for i, pump_obj in pump_objs.items():
        if not isinstance(pump_obj, bpy.types.Object):
            continue
        rpm   = 1450 + i * 50
        rev_s = rpm / 60.0
        for fi in [1, n_frames]:
            bpy.context.scene.frame_set(fi)
            angle = (fi / FPS) * rev_s * 2 * math.pi
            pump_obj.rotation_euler = Euler((0, angle, 0))
            pump_obj.keyframe_insert('rotation_euler', frame=fi)
        # Linear interpolation for smooth spin
        if pump_obj.animation_data:
            for fc in pump_obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*65)
    print("  Blender Cycles — Industrial Factory Photorealistic Render")
    print(f"  Frames: {N_FRAMES}  |  {FPS} fps  |  {N_FRAMES/FPS:.0f}s  |  {W}x{H}")
    print(f"  Samples: {SAMPLES}  |  Output: {OUTPUT_DIR}")
    print("="*65 + "\n")

    # 1. Clear default scene
    clear_scene()

    # 2. Build materials
    print("[1/5] Building materials...")
    mats = build_materials()

    # 3. Build scene geometry
    print("[2/5] Building scene geometry...")
    objs, light_locs = build_scene(mats)

    # 4. Lighting
    print("[3/5] Setting up lighting...")
    setup_lighting(light_locs)

    # 5. Camera
    print("[4/5] Setting up camera animation...")
    setup_camera(N_FRAMES)

    # 6. Render settings
    setup_render(OUTPUT_DIR, N_FRAMES)

    # 7. Animate pipe colours
    print("[4/5] Animating physics-driven pipe colours...")
    animate_pipe_colours(objs["pipes"], N_FRAMES)

    # 8. Animate pump rotation
    pump_dict = {i: objs[f"pump_{i}"] for i in range(3) if f"pump_{i}" in objs}
    animate_pumps(pump_dict, N_FRAMES)

    # 9. Save .blend file for inspection
    blend_path = str(OUTPUT_DIR / "factory_scene.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"[5/5] Saved .blend -> {blend_path}")

    # 10. Render animation
    print(f"\nStarting Cycles render ({N_FRAMES} frames @ {SAMPLES} samples)...")
    print("This will take a while on CPU. Check progress in Blender console.")
    bpy.ops.render.render(animation=True)

    print("\nRender complete! Frames saved to:", OUTPUT_DIR)
    print("To assemble MP4, run:")
    print(f"  python blender_factory_launcher.py --assemble-only")


if __name__ == "__main__":
    main()
