# primitives_and_architected.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union

import numpy as np

from pinneaple_geom.core.mesh import MeshData


def _tm_to_meshdata(tm) -> MeshData:
    # trimesh.Trimesh -> MeshData
    return MeshData(
        vertices=tm.vertices.view(np.ndarray),
        faces=tm.faces.view(np.ndarray),
        normals=tm.face_normals.view(np.ndarray) if getattr(tm, "face_normals", None) is not None else None,
    )


def _as_3tuple(v: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        x = float(v)
        return (x, x, x)
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return (float(v[0]), float(v[1]), float(v[2]))
    raise TypeError(f"Expected a scalar or 3-tuple, got: {type(v).__name__} {v}")


def _as_centerline_points(v: Any) -> np.ndarray:
    pts = np.asarray(v, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("centerline must be array-like (N,3) with N>=2")
    return pts


def _linspace01(n: int) -> np.ndarray:
    n = int(n)
    if n < 2:
        raise ValueError("n must be >= 2")
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


# =============================================================================
# Registry / Builders
# =============================================================================
class MeshBuilder(Protocol):
    def __call__(self, **params): ...  # returns trimesh.Trimesh


_BUILDERS: Dict[str, MeshBuilder] = {}


def register_builder(*names: str):
    def deco(fn: MeshBuilder):
        for n in names:
            _BUILDERS[n.lower().strip()] = fn
        return fn

    return deco


# =============================================================================
# Transforms & Boolean
# =============================================================================
def _apply_transform(tm, params: dict):
    import trimesh

    T = np.eye(4, dtype=np.float64)

    # scale
    s = params.get("scale", None)
    if s is not None:
        sx, sy, sz = _as_3tuple(s, (1.0, 1.0, 1.0))
        S = np.diag([sx, sy, sz, 1.0])
        T = S @ T

    # rotate (radians) around axes: rotate=(rx,ry,rz)
    r = params.get("rotate", None)
    if r is not None:
        rx, ry, rz = _as_3tuple(r, (0.0, 0.0, 0.0))
        Rx = trimesh.transformations.rotation_matrix(rx, [1, 0, 0])
        Ry = trimesh.transformations.rotation_matrix(ry, [0, 1, 0])
        Rz = trimesh.transformations.rotation_matrix(rz, [0, 0, 1])
        T = (Rz @ Ry @ Rx) @ T

    # translate
    t = params.get("translate", None)
    if t is not None:
        tx, ty, tz = _as_3tuple(t, (0.0, 0.0, 0.0))
        Tr = np.eye(4, dtype=np.float64)
        Tr[:3, 3] = [tx, ty, tz]
        T = Tr @ T

    if not np.allclose(T, np.eye(4)):
        tm = tm.copy()
        tm.apply_transform(T)
    return tm


def _boolean_engines_available() -> Tuple[str, ...]:
    """
    Returns a tuple of boolean engines available in trimesh environment.
    Typical names can include: 'manifold', 'blender', 'scad', etc.
    """
    try:
        import trimesh

        eng = getattr(trimesh.boolean, "engines_available", None)
        if eng is None:
            return tuple()
        # engines_available can be set-like or list-like
        return tuple(sorted(list(eng)))
    except Exception:
        return tuple()


def _boolean_or_fallback(tm_a, tm_b, op: str, *, allow_fallback: bool = True):
    """
    Robust boolean if possible, otherwise fallback to concatenation.
    Fallback is NOT watertight and can self-intersect.
    """
    import trimesh
    import warnings

    op = op.lower().strip()
    engines = _boolean_engines_available()

    # Try real booleans only if something is available
    if engines:
        try:
            if op == "union":
                return tm_a.union(tm_b)
            if op in {"diff", "difference", "cut"}:
                return tm_a.difference(tm_b)
            if op in {"intersect", "intersection"}:
                return tm_a.intersection(tm_b)
            raise ValueError(f"Unknown boolean op: {op}")
        except Exception as e:
            if not allow_fallback:
                raise
            warnings.warn(
                f"[mesh] Boolean '{op}' failed even though engines {engines} exist. "
                f"Falling back to concatenate (may be non-watertight). Error: {e}"
            )
            return trimesh.util.concatenate([tm_a, tm_b])

    # No engines at all
    if not allow_fallback:
        raise RuntimeError(
            f"[mesh] No boolean engine available for trimesh. "
            f"Install/configure one (e.g., manifold3d / blender / cork) to get watertight booleans."
        )

    warnings.warn(
        "[mesh] No trimesh boolean engine available. Falling back to concatenate "
        "(NOT watertight; may self-intersect). "
        "For robust booleans, install/configure an engine such as manifold3d/blender/cork."
    )
    return trimesh.util.concatenate([tm_a, tm_b])


# =============================================================================
# Pipe / Thread construction (MVP) using capsules along polyline segments
# =============================================================================
def _polyline_capsule_tube(points: np.ndarray, radius: Union[float, np.ndarray], sections: int = 16):
    """
    Create a "tube" along a polyline by concatenating capsules per segment.
    MVP approach that doesn't require CAD/B-Rep.
    For watertight/production, use a boolean engine to union segments or use a true sweep in CAD.
    """
    import trimesh

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("points must be (N,3) with N>=2")

    if isinstance(radius, (int, float)):
        rad = np.full((len(pts) - 1,), float(radius), dtype=np.float64)
    else:
        rad = np.asarray(radius, dtype=np.float64)
        if rad.shape[0] != (len(pts) - 1):
            raise ValueError("radius array must have length N-1 (per segment)")

    seg_meshes = []
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        h = float(np.linalg.norm(b - a))
        if h < 1e-9:
            continue

        cap = trimesh.creation.capsule(radius=float(rad[i]), height=h, count=[sections, sections])

        direction = (b - a) / h
        T = trimesh.geometry.align_vectors([0, 0, 1], direction)
        T[:3, 3] = (a + b) * 0.5
        cap.apply_transform(T)
        seg_meshes.append(cap)

    if not seg_meshes:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(seg_meshes)


# =============================================================================
# Rotation-minimizing frames for a centerline (parallel transport)
# =============================================================================
def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _rmf_frames(centerline: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute rotation-minimizing frames (RMF) along a polyline centerline.

    Returns:
      C: (N,3) points
      T: (N,3) tangents (unit)
      N: (N,3) normal-ish (unit)
      B: (N,3) binormal-ish (unit)
    """
    C = np.asarray(centerline, dtype=np.float64)
    Np = len(C)

    # Tangents: forward differences + normalize
    T = np.zeros_like(C)
    for i in range(Np):
        if i == 0:
            d = C[1] - C[0]
        elif i == Np - 1:
            d = C[Np - 1] - C[Np - 2]
        else:
            d = C[i + 1] - C[i - 1]
        T[i] = _normalize(d)

    # Choose initial normal not parallel to T0
    t0 = T[0]
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(t0, ref)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    N = np.zeros_like(C)
    B = np.zeros_like(C)

    n0 = _normalize(ref - np.dot(ref, t0) * t0)
    b0 = _normalize(np.cross(t0, n0))
    N[0] = n0
    B[0] = b0

    # Parallel transport
    for i in range(1, Np):
        v = np.cross(T[i - 1], T[i])
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            # no significant bend
            N[i] = N[i - 1]
            B[i] = B[i - 1]
            continue

        v = v / nv
        # rotation angle
        c = float(np.clip(np.dot(T[i - 1], T[i]), -1.0, 1.0))
        angle = float(np.arccos(c))

        # rotate previous normal around axis v by angle
        # Rodrigues rotation
        n_prev = N[i - 1]
        n_rot = (
            n_prev * np.cos(angle)
            + np.cross(v, n_prev) * np.sin(angle)
            + v * np.dot(v, n_prev) * (1.0 - np.cos(angle))
        )
        n_rot = _normalize(n_rot)
        b_rot = _normalize(np.cross(T[i], n_rot))

        N[i] = n_rot
        B[i] = b_rot

    return C, T, N, B


# =============================================================================
# Public factory API
# =============================================================================
def build_mesh(name: str, **params) -> MeshData:
    """
    General factory:
      - calls a registered builder -> trimesh.Trimesh
      - optional transforms: scale / rotate / translate
      - optional boolean: boolean={op:'cut'|'union'|'intersect', other:{name:'sphere', ...}}
        -> If no boolean engine exists, it falls back to concatenate (non-watertight).
    """
    import trimesh

    key = (name or "").lower().strip()
    if key not in _BUILDERS:
        raise ValueError(f"Unsupported '{name}'. Available: {sorted(_BUILDERS.keys())}")

    tm = _BUILDERS[key](**params)

    tm = _apply_transform(tm, params)

    boolean = params.get("boolean", None)
    if boolean is not None:
        op = str(boolean["op"])
        other_spec = dict(boolean["other"])
        other_name = other_spec.pop("name")
        other_md = build_mesh(other_name, **other_spec)
        tm_other = trimesh.Trimesh(vertices=other_md.vertices, faces=other_md.faces, process=False)

        tm = _boolean_or_fallback(tm, tm_other, op, allow_fallback=True)

    return _tm_to_meshdata(tm)


# Backwards-compatible alias (your original API name)
def build_primitive(name: str, **params) -> MeshData:
    return build_mesh(name, **params)


# =============================================================================
# Primitive builders (your existing primitives)
# =============================================================================
@register_builder("box", "cube", "rect", "cuboid")
def _build_box(**params):
    import trimesh

    if "side" in params and (params.get("extents") is None and params.get("size") is None):
        extents = _as_3tuple(params.get("side"), (1.0, 1.0, 1.0))
    else:
        extents = _as_3tuple(params.get("extents") or params.get("size"), (1.0, 1.0, 1.0))

    ex = tuple(max(1e-12, float(e)) for e in extents)
    return trimesh.creation.box(extents=ex)


@register_builder("sphere")
def _build_sphere(**params):
    import trimesh

    radius = float(params.get("radius", 1.0))
    subdivisions = int(params.get("subdivisions", 3))
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


@register_builder("cylinder")
def _build_cylinder(**params):
    import trimesh

    radius = float(params.get("radius", 1.0))
    height = float(params.get("height", 1.0))
    sections = int(params.get("sections", 32))
    return trimesh.creation.cylinder(radius=radius, height=height, sections=sections)


@register_builder("plane")
def _build_plane(**params):
    import trimesh

    size = params.get("size") or (1.0, 1.0)
    sx, sy = float(size[0]), float(size[1])

    v = np.array(
        [
            [-sx / 2, -sy / 2, 0.0],
            [sx / 2, -sy / 2, 0.0],
            [sx / 2, sy / 2, 0.0],
            [-sx / 2, sy / 2, 0.0],
        ],
        dtype=np.float64,
    )
    f = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


@register_builder("channel")
def _build_channel(**params):
    import trimesh

    length = float(params.get("length", 1.0))
    width = float(params.get("width", 0.2))
    height = float(params.get("height", 0.2))
    return trimesh.creation.box(extents=(length, width, height))


# =============================================================================
# Architected builders (straight tube versions)
# =============================================================================
@register_builder("woven_tube")
def _build_woven_tube(**p):
    """
    Woven-like tubular metamaterial (straight axis).
    - Helix + radial waviness (A*cos(k*theta + phase)) to create "over/under" feel
    - Optional local thickness field (gaussian around thick_z0)
    """
    import trimesh
    import math

    R = float(p.get("R", 10.0))
    L = float(p.get("L", 60.0))
    n_threads = int(p.get("n_threads", 12))
    turns = float(p.get("turns", 8.0))
    A = float(p.get("A", 1.0))
    k = float(p.get("k", 4.0))
    wire_r0 = float(p.get("wire_r", 0.6))
    samples = int(p.get("samples", 1200))
    sections = int(p.get("sections", 16))

    # Optional local thickness
    z0 = p.get("thick_z0", None)
    thick_amp = float(p.get("thick_amp", 0.0))    # multiplicative bump
    thick_sigma = float(p.get("thick_sigma", 5.0))

    all_threads = []
    for t in range(n_threads):
        phi0 = 2 * math.pi * t / n_threads
        wave_phase = 0.0 if (t % 2 == 0) else math.pi

        pts = np.zeros((samples + 1, 3), dtype=np.float64)
        radii = np.zeros((samples,), dtype=np.float64)

        for i in range(samples + 1):
            s = i / samples
            z = L * (s - 0.5)

            theta = 2 * math.pi * turns * s + phi0
            r = R + A * math.cos(k * theta + wave_phase)

            x = r * math.cos(theta)
            y = r * math.sin(theta)
            pts[i] = (x, y, z)

            if i < samples:
                wr = wire_r0
                if z0 is not None and thick_amp != 0.0:
                    wr = wire_r0 * (1.0 + thick_amp * math.exp(-0.5 * ((z - float(z0)) / thick_sigma) ** 2))
                radii[i] = max(1e-4, wr)

        thread = _polyline_capsule_tube(pts, radii, sections=sections)
        all_threads.append(thread)

    return trimesh.util.concatenate(all_threads)


@register_builder("braid_tube")
def _build_braid_tube(**p):
    """
    Braided tubular material (straight axis):
    - Two sets of helices (right-handed and left-handed) interlaced.
    - Good base for "high load + high deformation" when combined with density/thickness fields.
    """
    import trimesh
    import math

    R = float(p.get("R", 10.0))
    L = float(p.get("L", 60.0))
    n_threads = int(p.get("n_threads", 16))  # total
    turns = float(p.get("turns", 10.0))
    wire_r = float(p.get("wire_r", 0.6))
    samples = int(p.get("samples", 1200))
    sections = int(p.get("sections", 16))

    n_half = max(1, n_threads // 2)

    def make_set(handed: float, phase_offset: float):
        meshes = []
        for t in range(n_half):
            phi0 = 2 * math.pi * t / n_half + phase_offset
            pts = np.zeros((samples + 1, 3), dtype=np.float64)
            for i in range(samples + 1):
                s = i / samples
                z = L * (s - 0.5)
                theta = handed * (2 * math.pi * turns * s) + phi0
                pts[i] = (R * math.cos(theta), R * math.sin(theta), z)
            meshes.append(_polyline_capsule_tube(pts, wire_r, sections=sections))
        return meshes

    right = make_set(+1.0, 0.0)
    left = make_set(-1.0, math.pi / max(1, n_half))  # small offset

    return trimesh.util.concatenate(right + left)


# =============================================================================
# Architected builders (centerline + frames versions)
# =============================================================================
@register_builder("woven_centerline")
def _build_woven_centerline(**p):
    """
    Woven mapped on an arbitrary centerline using RMF frames.

    Inputs:
      - centerline: array-like (N,3)
      - n_threads: int
      - turns: float  (helical turns along the whole centerline)
      - R: float or callable s->R(s)  OR array-like length M_samples
      - A, k: radial waviness parameters
      - wire_r: float or callable (s, thread_idx)->radius
      - samples: int (resampling along arc-length param)
      - sections: int (capsule resolution)

    This is the recommended "next step" for patient-specific vascular paths
    without doing full surface projection.
    """
    import trimesh
    import math

    centerline = _as_centerline_points(p["centerline"])
    n_threads = int(p.get("n_threads", 12))
    turns = float(p.get("turns", 8.0))
    A = float(p.get("A", 1.0))
    k = float(p.get("k", 4.0))
    samples = int(p.get("samples", 1200))
    sections = int(p.get("sections", 16))

    R_param = p.get("R", 10.0)
    wire_r_param = p.get("wire_r", 0.6)

    # Build an arc-length parametrization by resampling centerline
    # (simple cumulative length + linear interpolation)
    C = centerline
    seg = C[1:] - C[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(s_cum[-1])
    if total < 1e-9:
        raise ValueError("centerline length is ~0")

    u = s_cum / total  # 0..1 at original points
    uu = _linspace01(samples + 1)

    # Resample centerline
    Cr = np.zeros((samples + 1, 3), dtype=np.float64)
    for d in range(3):
        Cr[:, d] = np.interp(uu, u, C[:, d])

    # Frames
    Cr, Tr, Nr, Br = _rmf_frames(Cr)

    # radius function
    def R_of(s: float) -> float:
        if callable(R_param):
            return float(R_param(s))
        if isinstance(R_param, (list, tuple, np.ndarray)):
            arr = np.asarray(R_param, dtype=np.float64)
            # expect length samples+1 or samples
            if arr.shape[0] == samples + 1:
                # interpolate by index
                idx = int(np.clip(round(s * samples), 0, samples))
                return float(arr[idx])
            if arr.shape[0] == samples:
                idx = int(np.clip(round(s * (samples - 1)), 0, samples - 1))
                return float(arr[idx])
            raise ValueError("If R is array-like, expected length samples or samples+1.")
        return float(R_param)

    # thickness function
    def wire_r_of(s: float, tidx: int) -> float:
        if callable(wire_r_param):
            return max(1e-4, float(wire_r_param(s, tidx)))
        return max(1e-4, float(wire_r_param))

    all_threads = []
    for t in range(n_threads):
        phi0 = 2 * math.pi * t / n_threads
        wave_phase = 0.0 if (t % 2 == 0) else math.pi

        pts = np.zeros((samples + 1, 3), dtype=np.float64)
        radii = np.zeros((samples,), dtype=np.float64)

        for i in range(samples + 1):
            s = i / samples
            theta = 2 * math.pi * turns * s + phi0

            baseR = R_of(s)
            r = baseR + A * math.cos(k * theta + wave_phase)

            # position around centerline using local normal/binormal
            offset = (Nr[i] * (r * math.cos(theta))) + (Br[i] * (r * math.sin(theta)))
            pts[i] = Cr[i] + offset

            if i < samples:
                radii[i] = wire_r_of(s, t)

        thread = _polyline_capsule_tube(pts, radii, sections=sections)
        all_threads.append(thread)

    return trimesh.util.concatenate(all_threads)


@register_builder("braid_centerline")
def _build_braid_centerline(**p):
    """
    Braided mapped on an arbitrary centerline using RMF frames.

    Inputs:
      - centerline: (N,3)
      - n_threads: total threads (split half right/half left)
      - turns: turns along the whole centerline
      - R: float or callable s->R(s)
      - wire_r: float or callable (s, thread_idx)->radius
      - samples, sections
    """
    import trimesh
    import math

    centerline = _as_centerline_points(p["centerline"])
    n_threads = int(p.get("n_threads", 16))
    turns = float(p.get("turns", 10.0))
    samples = int(p.get("samples", 1200))
    sections = int(p.get("sections", 16))

    R_param = p.get("R", 10.0)
    wire_r_param = p.get("wire_r", 0.6)

    # Resample by arc-length
    C = centerline
    seg = C[1:] - C[:-1]
    seglen = np.linalg.norm(seg, axis=1)
    s_cum = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(s_cum[-1])
    if total < 1e-9:
        raise ValueError("centerline length is ~0")

    u = s_cum / total
    uu = _linspace01(samples + 1)
    Cr = np.zeros((samples + 1, 3), dtype=np.float64)
    for d in range(3):
        Cr[:, d] = np.interp(uu, u, C[:, d])

    Cr, Tr, Nr, Br = _rmf_frames(Cr)

    def R_of(s: float) -> float:
        if callable(R_param):
            return float(R_param(s))
        return float(R_param)

    def wire_r_of(s: float, tidx: int) -> float:
        if callable(wire_r_param):
            return max(1e-4, float(wire_r_param(s, tidx)))
        return max(1e-4, float(wire_r_param))

    n_half = max(1, n_threads // 2)

    def make_set(handed: float, phase_offset: float, tidx_offset: int):
        meshes = []
        for t in range(n_half):
            phi0 = 2 * math.pi * t / n_half + phase_offset
            pts = np.zeros((samples + 1, 3), dtype=np.float64)
            radii = np.zeros((samples,), dtype=np.float64)

            tidx = tidx_offset + t

            for i in range(samples + 1):
                s = i / samples
                theta = handed * (2 * math.pi * turns * s) + phi0
                r = R_of(s)

                offset = (Nr[i] * (r * math.cos(theta))) + (Br[i] * (r * math.sin(theta)))
                pts[i] = Cr[i] + offset
                if i < samples:
                    radii[i] = wire_r_of(s, tidx)

            meshes.append(_polyline_capsule_tube(pts, radii, sections=sections))
        return meshes

    right = make_set(+1.0, 0.0, 0)
    left = make_set(-1.0, math.pi / max(1, n_half), n_half)

    return trimesh.util.concatenate(right + left)


# =============================================================================
# Projection hook (not implemented): surface patient-specific mapping
# =============================================================================
def project_points_to_surface(
    points: np.ndarray,
    surface: MeshData,
    *,
    method: str = "nearest",
) -> np.ndarray:
    """
    Placeholder / hook for future "fidelity maximum" mapping:
      - Given points (N,3), project them onto a patient-specific surface mesh.

    This is intentionally NOT implemented generically here because:
      - robust projection requires spatial acceleration structures (kdtree/bvh),
      - careful handling of inside/outside normals,
      - and often a CAD kernel or dedicated geometry library.

    Recommended path:
      - Use a proper BVH nearest-point query (trimesh has proximity module),
      - then add constraints (stay within thickness, avoid foldovers),
      - and finally correct collisions with local thickness/spacing.

    Raise by default so you don't mistakenly assume it's working.
    """
    raise NotImplementedError(
        "Surface projection is not implemented in this module. "
        "Use centerline+frames builders for practical vascular mapping, "
        "or implement a robust mesh projection pipeline (BVH + constraints)."
    )
