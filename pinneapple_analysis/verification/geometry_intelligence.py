"""Geometry Intelligence -- the concrete implementation of the founder's
architectural point 4 ("CAD/mesh upload -> automatic semantic region
labeling -> boundary conditions"), built entirely from real,
already-tested PINNeAPPle geometry infrastructure
(``pinneapple_design.geometry.core.mesh.MeshData``,
``pinneapple_design.geometry.io.trimesh_bridge`` for real STL/OBJ/STEP
loading, ``pinneapple_design.geometry.gen.primitives.build_mesh`` for the
synthetic meshes this module's own tests are validated against).

Public API
----------
    classify_geometry(
        mesh: MeshData,
        problem_description: str = "",
        *,
        provider: Optional[str] = None,      # None -> geometry-only, no LLM call
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        angle_tolerance_deg: float = 20.0,
        planar_tolerance_deg: float = 5.0,
        opening_circularity_threshold: float = 0.8,
        opening_area_fraction_threshold: float = 0.3,
        symmetry_axis_tolerance_deg: float = 10.0,
        symmetry_midplane_tolerance_frac: float = 0.1,
        symmetry_area_fraction_threshold: float = 0.5,
        conversation_store=None,
    ) -> GeometryClassification

Two-stage design, mirroring the EXACT "checked menu, never invented
geometry" pattern already established by ``pinneapple_llm.geometry_draft``
/``pinneapple_llm.cad_draft`` for CAD generation:

1. **Real, mechanical geometric segmentation** (no LLM, fully
   deterministic, unit-tested against known-shape synthetic meshes):
   faces are grouped into connected patches by normal-vector similarity
   (a face-adjacency BFS that only crosses an edge when the two faces'
   normals are within ``angle_tolerance_deg`` of each other -- this
   naturally keeps a whole curved surface, e.g. a cylinder's side, in one
   patch while stopping cleanly at a sharp crease, e.g. a box corner).
   Each patch's boundary is traced into closed loops (crease edges
   between two different patches, or true open mesh-boundary edges), and
   each loop's shape is scored by the classical isoperimetric quotient
   ``4*pi*Area/Perimeter**2`` (<=1, =1 only for a perfect circle -- see
   e.g. any elementary differential-geometry text on the isoperimetric
   inequality). From these REAL, computed numbers, three geometric
   "candidate" labels are assigned by simple, honestly-documented
   heuristics:
     - **wall**: the default -- a patch that is not small+circular+
       single-loop (an "opening" candidate) and not large+axis-aligned+
       mid-bounding-box (a "symmetry_plane" candidate).
     - **opening** (a candidate inlet/outlet/port location): a patch
       whose entire boundary is ONE closed, near-planar loop, that loop's
       isoperimetric quotient is high (roughly circular), and the patch's
       own area is small relative to the mesh's largest cross-section --
       this only says "this is a port-shaped region", NEVER "this is
       specifically an inlet" or "specifically an outlet": pure geometry
       cannot know flow direction, so see stage 2 below.
     - **symmetry_plane**: a planar patch whose normal is aligned (within
       ``symmetry_axis_tolerance_deg``) with one of the mesh's own global
       axes AND whose position along that axis sits near the bounding
       box's midpoint along that axis AND whose area is large (a genuine
       cross-section, not a small port) -- this is the literal geometric
       signature the founder's own vision describes for a symmetry cut.
       Honest, stated limitation: a REAL half-domain mesh commonly has
       its symmetry face sitting at the EDGE of its own bounding box
       (because only half the original domain was kept), not at the
       middle -- that configuration is geometrically indistinguishable
       from a generic large flat wall face by this heuristic (or by any
       purely-geometric heuristic at all, without external context), and
       is correctly reported as a plain "wall" candidate here rather than
       guessed at.

2. **LLM-assisted semantic disambiguation** (only when ``provider`` is
   given): the LLM is shown the REAL, already-computed geometric
   properties of every region (never raw geometry, never asked to invent
   a region) plus the user's ``problem_description``, and assigns each
   region a final semantic label (wall/inlet/outlet/symmetry/periodic/
   ambiguous) and a boundary-condition type, from FIXED, checked
   vocabularies -- exactly ``pinneapple_llm.cad_draft``'s "propose only
   from a real, validated menu, reject any invented name" pattern, applied
   to region ids and label/bc-type strings instead of builder names.
   Without an LLM (``provider=None``), every "opening" candidate is
   conservatively reported as ``semantic_label="ambiguous"`` rather than
   guessed at -- geometry alone genuinely cannot tell an inlet from an
   outlet.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from pinneapple_design.geometry.core.mesh import MeshData

_ALLOWED_SEMANTIC_LABELS = {"wall", "inlet", "outlet", "symmetry", "periodic", "ambiguous"}
_ALLOWED_BC_TYPES = {
    "no_slip_wall", "slip_wall", "velocity_inlet", "pressure_inlet",
    "pressure_outlet", "mass_flow_outlet", "symmetry", "periodic", "unspecified",
}
_GLOBAL_AXES = np.eye(3)  # x, y, z unit vectors


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class BoundaryLoop:
    """One closed boundary loop bounding a surface patch -- either a crease
    edge chain shared with a DIFFERENT patch, or a true open mesh-boundary
    edge chain (the patch has an actual hole/opening, no covering face at
    all -- the convention many real CFD surface meshes use to represent an
    inlet/outlet port). Fields are ``None`` when not computable rather than
    silently defaulted (e.g. a loop whose trace could not be closed, or a
    non-planar loop for which "enclosed area"/"circularity" aren't
    meaningful)."""
    n_vertices: int
    closed: bool  # False if the loop tracer could not return to its start (e.g. a non-manifold junction)
    perimeter: float
    is_planar: bool
    planarity_rms_deviation: Optional[float]  # RMS distance of loop points from their best-fit plane
    enclosed_area: Optional[float]  # None if not planar enough to define a 2D polygon area
    circularity: Optional[float]  # 4*pi*Area/Perimeter^2, the isoperimetric quotient: <=1, =1 for a perfect circle
    centroid: Optional[Tuple[float, float, float]] = None  # mean of loop vertices, when n>=3
    normal: Optional[Tuple[float, float, float]] = None    # best-fit-plane normal, when n>=3
    vertex_ids: Tuple[int, ...] = ()  # the traced mesh vertex ids, used to de-duplicate the same physical
                                       # crease loop when it is encountered from both patches it separates


@dataclass
class SurfaceRegion:
    """One connected surface patch found by normal-vector-similarity
    segmentation, with every geometric property the classifier's
    heuristics (and, optionally, the LLM disambiguation step) actually
    use -- computed, real numbers, never placeholders."""
    region_id: int
    face_ids: List[int]
    n_faces: int
    area: float
    area_fraction_of_total_surface: float
    area_fraction_of_largest_cross_section: float  # area / max(Lx*Ly, Ly*Lz, Lx*Lz) of the mesh's own bbox
    centroid: Tuple[float, float, float]
    mean_normal: Tuple[float, float, float]
    normal_std_deg: float  # area-weighted angular spread of this patch's face normals, in degrees
    is_planar: bool
    boundary_loops: List[BoundaryLoop]
    geometric_candidate: str  # "wall" | "opening" | "symmetry_plane"
    geometric_reasoning: str


@dataclass
class GeometryClassification:
    regions: List[SurfaceRegion]
    semantic_labels: Dict[int, str] = field(default_factory=dict)          # region_id -> wall/inlet/outlet/symmetry/periodic/ambiguous
    boundary_condition_types: Dict[int, str] = field(default_factory=dict)  # region_id -> a real, checked BC-type string
    ambiguous_region_ids: List[int] = field(default_factory=list)
    llm_used: bool = False
    llm_reasoning: str = ""
    raw_llm_response: str = ""


# ---------------------------------------------------------------------------
# Stage 1: real, mechanical geometric segmentation
# ---------------------------------------------------------------------------

def _edge_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _build_edge_face_map(faces: np.ndarray) -> Dict[Tuple[int, int], List[int]]:
    edge_map: Dict[Tuple[int, int], List[int]] = {}
    for fidx in range(faces.shape[0]):
        tri = faces[fidx]
        for i in range(3):
            key = _edge_key(int(tri[i]), int(tri[(i + 1) % 3]))
            edge_map.setdefault(key, []).append(fidx)
    return edge_map


def _segment_patches(faces: np.ndarray, face_normals: np.ndarray, angle_tolerance_deg: float,
                      edge_map: Dict[Tuple[int, int], List[int]]) -> List[np.ndarray]:
    """Region-growing segmentation: two faces sharing an edge are merged
    into the same patch iff their normals are within ``angle_tolerance_deg``
    of each other. Because this check is applied pairwise between
    IMMEDIATE neighbors (not against a whole-patch average), a smoothly
    curved surface (e.g. a cylinder's side, tessellated into many small
    facets each only slightly rotated from its neighbor) correctly ends up
    as ONE patch, while a real crease (e.g. a box's 90-degree corner) stops
    the growth cleanly -- this is the "connected planar/curved patches"
    clustering the product's spec calls for, not a global-planarity check
    (that is instead ``is_planar`` on the finished patch, checked
    separately below)."""
    n_faces = faces.shape[0]
    adjacency: List[List[int]] = [[] for _ in range(n_faces)]
    for flist in edge_map.values():
        if len(flist) == 2:
            f0, f1 = flist
            adjacency[f0].append(f1)
            adjacency[f1].append(f0)
        # len==1: a true open mesh-boundary edge, no face neighbor to grow into.
        # len>2: a non-manifold edge; not treated as a growth path (conservative).

    cos_tol = np.cos(np.radians(angle_tolerance_deg))
    visited = np.zeros(n_faces, dtype=bool)
    patches: List[np.ndarray] = []
    for start in range(n_faces):
        if visited[start]:
            continue
        visited[start] = True
        stack = [start]
        comp = [start]
        while stack:
            f = stack.pop()
            nf = face_normals[f]
            for nb in adjacency[f]:
                if visited[nb]:
                    continue
                if float(np.dot(nf, face_normals[nb])) >= cos_tol:
                    visited[nb] = True
                    comp.append(nb)
                    stack.append(nb)
        patches.append(np.array(comp, dtype=np.int64))
    return patches


def _trace_boundary_loops(border_edges: List[Tuple[int, int]], vertices: np.ndarray) -> List[BoundaryLoop]:
    """Trace a patch's border edges (creases with another patch, or true
    open mesh-boundary edges) into closed loops by walking the induced
    vertex graph. Handles the simple-loop case correctly (every vertex on
    the border has degree 2, true for every synthetic test mesh this
    module validates against); at a higher-valence junction (3+ patches
    meeting at one point) the walk picks an arbitrary next edge and the
    resulting loop is marked ``closed=False`` if it cannot return to its
    start, an honest signal rather than a silently-wrong loop shape."""
    adj: Dict[int, List[int]] = defaultdict(list)
    for a, b in border_edges:
        adj[a].append(b)
        adj[b].append(a)

    visited_edges: Set[Tuple[int, int]] = set()
    loops: List[BoundaryLoop] = []
    max_steps = len(border_edges) + 2

    for a, b in border_edges:
        e0 = _edge_key(a, b)
        if e0 in visited_edges:
            continue
        visited_edges.add(e0)
        loop_vertices = [a, b]
        start, prev, cur = a, a, b
        closed = False
        for _ in range(max_steps):
            if cur == start:
                closed = True
                break
            nxt = None
            for cand in adj[cur]:
                e = _edge_key(cur, cand)
                if e not in visited_edges:
                    nxt = cand
                    visited_edges.add(e)
                    break
            if nxt is None:
                # can close by returning directly to start along an already-used edge shape
                if start in adj[cur] and _edge_key(cur, start) not in visited_edges:
                    visited_edges.add(_edge_key(cur, start))
                    closed = True
                break
            loop_vertices.append(nxt)
            prev, cur = cur, nxt
        loops.append(_loop_from_vertex_chain(loop_vertices, vertices, closed))
    return loops


def _loop_from_vertex_chain(vertex_ids: List[int], vertices: np.ndarray, closed: bool) -> BoundaryLoop:
    pts = vertices[np.array(vertex_ids, dtype=np.int64)]
    n = pts.shape[0]
    if n < 3:
        return BoundaryLoop(n_vertices=n, closed=False, perimeter=0.0, is_planar=False,
                             planarity_rms_deviation=None, enclosed_area=None, circularity=None)

    edges = pts[1:] - pts[:-1]
    perimeter = float(np.sum(np.linalg.norm(edges, axis=1)))
    if closed:
        perimeter += float(np.linalg.norm(pts[0] - pts[-1]))

    # Best-fit plane via PCA: the normal is the smallest-variance principal axis.
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    try:
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return BoundaryLoop(n_vertices=n, closed=closed, perimeter=perimeter, is_planar=False,
                             planarity_rms_deviation=None, enclosed_area=None, circularity=None)
    normal = vt[-1]
    signed_dist = centered @ normal
    rms_dev = float(np.sqrt(np.mean(signed_dist ** 2)))
    scale = max(float(np.max(np.linalg.norm(centered, axis=1))), 1e-12)
    is_planar = (rms_dev / scale) < 0.02  # loop points within 2% of the loop's own size from a flat plane
    normal_out = tuple(float(c) for c in normal)
    centroid_out = tuple(float(c) for c in centroid)

    enclosed_area = None
    circularity = None
    if is_planar and closed and n >= 3:
        u = centered[0]
        u = u - float(u @ normal) * normal
        u_norm = float(np.linalg.norm(u))
        if u_norm > 1e-12:
            u = u / u_norm
            v = np.cross(normal, u)
            xy = np.stack([centered @ u, centered @ v], axis=1)
            # Shoelace formula for a (possibly non-convex, but simple) planar polygon.
            x, y = xy[:, 0], xy[:, 1]
            enclosed_area = float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
            if perimeter > 1e-12:
                # Isoperimetric quotient: 4*pi*Area/Perimeter^2, <=1, ==1 only for a circle
                # (classical isoperimetric inequality) -- the real, standard "how circular is
                # this closed curve" scalar used here, not an ad hoc shape score.
                circularity = float(4.0 * np.pi * enclosed_area / (perimeter ** 2))

    return BoundaryLoop(
        n_vertices=n, closed=closed, perimeter=perimeter, is_planar=is_planar,
        planarity_rms_deviation=rms_dev, enclosed_area=enclosed_area, circularity=circularity,
        centroid=centroid_out, normal=normal_out, vertex_ids=tuple(int(v) for v in vertex_ids),
    )


def _segment_mesh(
    mesh: MeshData, *, angle_tolerance_deg: float, planar_tolerance_deg: float,
    opening_circularity_threshold: float, opening_area_fraction_threshold: float,
    symmetry_axis_tolerance_deg: float, symmetry_midplane_tolerance_frac: float,
    symmetry_area_fraction_threshold: float,
) -> List[SurfaceRegion]:
    face_normals = mesh.compute_face_normals()
    face_areas = mesh.face_areas()
    total_area = float(np.sum(face_areas))
    b0, b1 = mesh.bounds()
    bbox_size = b1 - b0
    lx, ly, lz = bbox_size
    cross_sections = [lx * ly, ly * lz, lx * lz]
    largest_cross_section = max(max(cross_sections), 1e-12)
    bbox_mid = 0.5 * (b0 + b1)

    edge_map = _build_edge_face_map(mesh.faces)
    face_patches = _segment_patches(mesh.faces, face_normals, angle_tolerance_deg, edge_map)

    face_patch_id = np.full(mesh.n_faces, -1, dtype=np.int64)
    for pid, comp in enumerate(face_patches):
        face_patch_id[comp] = pid

    border_edges_per_patch: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for key, flist in edge_map.items():
        if len(flist) == 1:
            border_edges_per_patch[int(face_patch_id[flist[0]])].append(key)
        elif len(flist) == 2:
            p0, p1 = int(face_patch_id[flist[0]]), int(face_patch_id[flist[1]])
            if p0 != p1:
                border_edges_per_patch[p0].append(key)
                border_edges_per_patch[p1].append(key)
        else:
            for f in flist:
                border_edges_per_patch[int(face_patch_id[f])].append(key)

    regions: List[SurfaceRegion] = []
    for pid, comp in enumerate(face_patches):
        comp_areas = face_areas[comp]
        area = float(np.sum(comp_areas))
        v0 = mesh.vertices[mesh.faces[comp, 0]]
        v1 = mesh.vertices[mesh.faces[comp, 1]]
        v2 = mesh.vertices[mesh.faces[comp, 2]]
        face_centroids = (v0 + v1 + v2) / 3.0
        centroid = np.average(face_centroids, axis=0, weights=comp_areas) if area > 0 else face_centroids.mean(axis=0)

        raw_mean_normal = np.average(face_normals[comp], axis=0, weights=comp_areas) if area > 0 else face_normals[comp].mean(axis=0)
        mn_norm = float(np.linalg.norm(raw_mean_normal))
        mean_normal = raw_mean_normal / mn_norm if mn_norm > 1e-12 else raw_mean_normal

        cos_dev = np.clip(face_normals[comp] @ mean_normal, -1.0, 1.0)
        angle_dev_deg = np.degrees(np.arccos(cos_dev))
        normal_std_deg = float(np.sqrt(np.average(angle_dev_deg ** 2, weights=comp_areas))) if area > 0 else float(np.mean(angle_dev_deg))
        is_planar = normal_std_deg <= planar_tolerance_deg

        loops = _trace_boundary_loops(border_edges_per_patch.get(pid, []), mesh.vertices)

        area_fraction_total = area / total_area if total_area > 0 else 0.0
        area_fraction_cross = area / largest_cross_section

        # -- classify: opening candidate -----------------------------------
        candidate = "wall"
        reasoning = "default: not small+circular+single-loop, not large+axis-aligned+mid-bbox"

        single_loop_planar_circular = (
            len(loops) == 1 and loops[0].closed and loops[0].is_planar
            and loops[0].circularity is not None and loops[0].circularity >= opening_circularity_threshold
        )
        if single_loop_planar_circular and area_fraction_cross <= opening_area_fraction_threshold:
            candidate = "opening"
            reasoning = (
                f"single closed planar loop, circularity={loops[0].circularity:.3f} "
                f">= {opening_circularity_threshold}, area/largest_cross_section="
                f"{area_fraction_cross:.3f} <= {opening_area_fraction_threshold} -- a candidate "
                "port location (inlet/outlet identity is NOT determinable from geometry alone)"
            )
        elif is_planar:
            # -- classify: symmetry-plane candidate --------------------------
            best_axis, best_align = None, 0.0
            for axis_idx in range(3):
                align = abs(float(np.dot(mean_normal, _GLOBAL_AXES[axis_idx])))
                if align > best_align:
                    best_axis, best_align = axis_idx, align
            axis_tol_cos = np.cos(np.radians(symmetry_axis_tolerance_deg))
            extent_along_axis = bbox_size[best_axis] if best_axis is not None else 0.0
            mid_tol = max(symmetry_midplane_tolerance_frac * extent_along_axis, 1e-9)
            near_midplane = (
                best_axis is not None
                and abs(float(centroid[best_axis]) - float(bbox_mid[best_axis])) <= mid_tol
            )
            if (
                best_axis is not None and best_align >= axis_tol_cos and near_midplane
                and area_fraction_cross >= symmetry_area_fraction_threshold
            ):
                candidate = "symmetry_plane"
                axis_name = "xyz"[best_axis]
                reasoning = (
                    f"planar, normal aligned to global {axis_name}-axis (|cos|={best_align:.3f}), "
                    f"centroid within {mid_tol:.4g} of the bbox midpoint along {axis_name}, "
                    f"area/largest_cross_section={area_fraction_cross:.3f} >= "
                    f"{symmetry_area_fraction_threshold} -- candidate symmetry plane. KNOWN LIMITATION: "
                    "a real half-domain mesh whose symmetry face sits at the EDGE (not the middle) of "
                    "its own bounding box is indistinguishable from a generic wall by geometry alone, "
                    "and is reported as 'wall' rather than guessed."
                )

        regions.append(SurfaceRegion(
            region_id=pid, face_ids=[int(f) for f in comp], n_faces=len(comp), area=area,
            area_fraction_of_total_surface=area_fraction_total,
            area_fraction_of_largest_cross_section=area_fraction_cross,
            centroid=tuple(float(c) for c in centroid), mean_normal=tuple(float(c) for c in mean_normal),
            normal_std_deg=normal_std_deg, is_planar=is_planar, boundary_loops=loops,
            geometric_candidate=candidate, geometric_reasoning=reasoning,
        ))

    # -- loop-level "hole punched through a wall, no covering material" ---
    # A genuine through-port (no blind-pocket cap, no interior disc left
    # behind) never gets its own connected surface patch above -- it shows
    # up ONLY as an extra, small, circular INNER boundary loop of a larger
    # wall patch (that patch's outer perimeter is its own separate, much
    # bigger loop). This is exactly the "a real hole in a wall has no
    # material of its own to form a region" case -- so it is reported here
    # as its own zero-area "opening" candidate (face_ids=[]), one per
    # qualifying inner loop, continuing the region id sequence rather than
    # silently folding it into the wall patch it's cut into (which would
    # hide a real, BC-relevant boundary from the caller).
    # A loop that is the ENTIRE boundary of some other real, already-built whole
    # patch (e.g. a blind pocket's flat bottom cap, a single-loop disc in its own
    # right) is the SAME physical crease as one of THIS patch's "inner" loops --
    # already fully represented as that other region, so it must not also be
    # re-reported here as a zero-area virtual duplicate of itself.
    single_loop_signatures: Set[frozenset] = {
        frozenset(r.boundary_loops[0].vertex_ids) for r in regions if len(r.boundary_loops) == 1
    }

    next_id = len(face_patches)
    sub_openings: List[SurfaceRegion] = []
    for r in regions:
        if len(r.boundary_loops) < 2:
            continue
        if not r.is_planar:
            # Only meaningful for a flat wall patch with a literal hole punched in it
            # (an "outer" loop plus one or more small "inner hole" loops). A CURVED
            # patch's multiple loops (e.g. a tube's two open ends) are its own genuine
            # boundary, not a hole-within-a-face, and the same physical crease loop
            # would otherwise be double-counted here once per side of the crease
            # (once from this patch, once already from the flat patch it borders) --
            # skipping non-planar patches avoids exactly that double-count.
            continue
        loop_areas = [l.enclosed_area if l.enclosed_area is not None else -1.0 for l in r.boundary_loops]
        outer_idx = int(np.argmax(loop_areas))
        for li, loop in enumerate(r.boundary_loops):
            if li == outer_idx:
                continue
            if not (loop.closed and loop.is_planar and loop.circularity is not None
                    and loop.circularity >= opening_circularity_threshold and loop.enclosed_area):
                continue
            if frozenset(loop.vertex_ids) in single_loop_signatures:
                continue  # already represented as its own whole single-loop patch elsewhere
            area_fraction_cross = loop.enclosed_area / largest_cross_section
            if area_fraction_cross > opening_area_fraction_threshold:
                continue
            sub_openings.append(SurfaceRegion(
                region_id=next_id, face_ids=[], n_faces=0, area=loop.enclosed_area,
                area_fraction_of_total_surface=(loop.enclosed_area / total_area) if total_area > 0 else 0.0,
                area_fraction_of_largest_cross_section=area_fraction_cross,
                centroid=loop.centroid or (0.0, 0.0, 0.0), mean_normal=loop.normal or (0.0, 0.0, 0.0),
                normal_std_deg=0.0, is_planar=True, boundary_loops=[loop],
                geometric_candidate="opening",
                geometric_reasoning=(
                    f"a hole with no covering material of its own, bounded by a single circular "
                    f"loop (circularity={loop.circularity:.3f} >= {opening_circularity_threshold}) "
                    f"cut through wall patch region {r.region_id} -- area/largest_cross_section="
                    f"{area_fraction_cross:.3f} <= {opening_area_fraction_threshold} -- a candidate "
                    "port location (inlet/outlet identity is NOT determinable from geometry alone)"
                ),
            ))
            next_id += 1
    regions.extend(sub_openings)

    return regions


# ---------------------------------------------------------------------------
# Stage 2: LLM-assisted semantic disambiguation (checked-menu pattern)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are assigning physical boundary-condition labels to \
a FIXED set of already-computed geometric surface regions on a mesh -- you \
are NOT inventing new geometry, regions, or region ids, only interpreting \
the ones given to you, using the user's problem description for context \
that pure geometry cannot supply (e.g. "is this port a flow inlet or a flow \
outlet" needs to know the flow direction, which geometry alone can never \
determine).

You are given TWO lists of regions: "OTHER REGIONS", already confidently \
classified by real geometry and shown to you ONLY for context (a wall's \
position can tell you which specific opening is "the one near the wall on \
the left", for instance) -- and "REGIONS THAT NEED YOUR DISAMBIGUATION", \
which is the ONLY list you must produce an assignment for.

You MUST respond with a single JSON object and nothing else, of the exact \
form:
{"assignments": [{"region_id": <int>, "label": "<one of wall|inlet|outlet|symmetry|periodic|ambiguous>", \
"bc_type": "<one of no_slip_wall|slip_wall|velocity_inlet|pressure_inlet|pressure_outlet|mass_flow_outlet|symmetry|periodic|unspecified>", \
"reasoning": "<one sentence>"}], "overall_reasoning": "<one or two sentences>"}

Rules:
- "region_id" MUST be exactly one of the region ids listed under "REGIONS THAT NEED YOUR \
DISAMBIGUATION". Never invent a new id, and never include a region id from "OTHER REGIONS".
- Provide EXACTLY one assignment per region id in "REGIONS THAT NEED YOUR DISAMBIGUATION", \
no more, no fewer.
- If the problem description gives you no way to tell whether a given region \
is an inlet or an outlet, respond with "label": "ambiguous" and "bc_type": "unspecified" \
for that region rather than guessing -- this is the CORRECT, expected answer when direction \
truly cannot be determined, not a failure.
"""


def _region_catalog(regions: List[SurfaceRegion]) -> List[Dict[str, Any]]:
    catalog = []
    for r in regions:
        catalog.append({
            "region_id": r.region_id,
            "n_faces": r.n_faces,
            "area": round(r.area, 6),
            "area_fraction_of_total_surface": round(r.area_fraction_of_total_surface, 4),
            "area_fraction_of_largest_cross_section": round(r.area_fraction_of_largest_cross_section, 4),
            "centroid": [round(c, 6) for c in r.centroid],
            "mean_normal": [round(c, 6) for c in r.mean_normal],
            "is_planar": r.is_planar,
            "n_boundary_loops": len(r.boundary_loops),
            "boundary_loop_circularity": [
                (round(l.circularity, 4) if l.circularity is not None else None) for l in r.boundary_loops
            ],
            "geometric_candidate": r.geometric_candidate,
            "geometric_reasoning": r.geometric_reasoning,
        })
    return catalog


def _llm_assign_semantics(
    regions: List[SurfaceRegion], problem_description: str, *,
    provider: str, model: Optional[str], api_key: Optional[str], conversation_store=None,
) -> Tuple[Dict[int, str], Dict[int, str], str, str]:
    """Ask the LLM to disambiguate ONLY the regions pure geometry could not
    already confidently resolve (``geometric_candidate == "opening"``) --
    wall and symmetry-plane candidates are never sent for re-assignment,
    both because the task genuinely doesn't need an LLM's help there (see
    the module docstring) and because a real, measured reliability problem
    was found doing it the other way: asking a small local model
    (``llama3.2:3b``) to assign EVERY region in a 9-region mesh reliably
    produced a response covering only 1 of the 9 -- exactly the kind of
    partial/truncated response this repo's other LLM-drafting modules
    (``cad_draft.py``, ``geometry_draft.py``) already document and design
    around, not a one-off fluke. Shrinking the required output to just the
    genuinely-ambiguous regions (typically 1-2, never the whole mesh)
    is both a more honest scoping of what actually needs an LLM opinion
    and a much more tractable ask for a small model."""
    import pinneapple_llm as pl

    opening_regions = [r for r in regions if r.geometric_candidate == "opening"]
    context_regions = [r for r in regions if r.geometric_candidate != "opening"]
    region_ids = {r.region_id for r in opening_regions}

    prompt = (
        f"USER PROBLEM DESCRIPTION:\n{problem_description or '(none given)'}\n\n"
        "OTHER REGIONS (already confidently classified by real geometric heuristics -- for "
        "context only, DO NOT include these in your \"assignments\"):\n"
        f"{json.dumps(_region_catalog(context_regions), indent=2)}\n\n"
        "REGIONS THAT NEED YOUR DISAMBIGUATION (assign EXACTLY one of these per entry, no more, "
        "no fewer):\n"
        f"{json.dumps(_region_catalog(opening_regions), indent=2)}\n"
    )
    raw = pl.call_llm(
        prompt, provider=provider, model=model, api_key=api_key, system=_SYSTEM_PROMPT,
        json_mode=True, module="geometry_intelligence", conversation_store=conversation_store,
    )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON:\n{raw}") from e

    assignments = parsed.get("assignments")
    overall_reasoning = parsed.get("overall_reasoning", "")
    if not isinstance(assignments, list):
        raise ValueError(f"LLM response missing a JSON list under 'assignments': {raw}")

    seen_ids: Set[int] = set()
    labels: Dict[int, str] = {}
    bc_types: Dict[int, str] = {}
    for entry in assignments:
        if not isinstance(entry, dict):
            raise ValueError(f"LLM assignment entry is not a JSON object: {entry!r}")
        rid = entry.get("region_id")
        if not isinstance(rid, int) or rid not in region_ids:
            raise ValueError(
                f"LLM named region_id {rid!r}, not in the real region set {sorted(region_ids)} "
                "-- refusing to use a hallucinated region id."
            )
        label = entry.get("label")
        if label not in _ALLOWED_SEMANTIC_LABELS:
            raise ValueError(
                f"LLM proposed label {label!r} for region {rid}, not in the real allowed set "
                f"{sorted(_ALLOWED_SEMANTIC_LABELS)} -- refusing a hallucinated label."
            )
        bc_type = entry.get("bc_type")
        if bc_type not in _ALLOWED_BC_TYPES:
            raise ValueError(
                f"LLM proposed bc_type {bc_type!r} for region {rid}, not in the real allowed set "
                f"{sorted(_ALLOWED_BC_TYPES)} -- refusing a hallucinated bc_type."
            )
        seen_ids.add(rid)
        labels[rid] = label
        bc_types[rid] = bc_type

    missing = region_ids - seen_ids
    if missing:
        raise ValueError(f"LLM response did not assign every real region id -- missing {sorted(missing)}.")

    return labels, bc_types, overall_reasoning, raw


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_geometry(
    mesh: MeshData,
    problem_description: str = "",
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    angle_tolerance_deg: float = 20.0,
    planar_tolerance_deg: float = 5.0,
    opening_circularity_threshold: float = 0.8,
    opening_area_fraction_threshold: float = 0.3,
    symmetry_axis_tolerance_deg: float = 10.0,
    symmetry_midplane_tolerance_frac: float = 0.1,
    symmetry_area_fraction_threshold: float = 0.5,
    conversation_store=None,
) -> GeometryClassification:
    """Classify every boundary surface of ``mesh`` into a physically-
    meaningful region. See the module docstring for the full two-stage
    design (real geometric segmentation, then optional LLM semantic
    disambiguation) and exactly which heuristics are used for each of the
    three geometric candidate labels (wall/opening/symmetry_plane).

    Parameters
    ----------
    mesh : a real ``MeshData`` -- load one from an uploaded file with
        ``pinneapple_design.geometry.io.trimesh_bridge.TrimeshBridge().load(path)``,
        or build a synthetic one with
        ``pinneapple_design.geometry.gen.primitives.build_mesh``.
    problem_description : natural-language context for the LLM
        disambiguation step (e.g. "water flows in through the small round
        port on the left and exits through the large opening on the
        right"). Ignored (and unnecessary) if ``provider`` is None.
    provider : ``None`` (default) runs geometry-only classification --
        every "opening" candidate is reported with
        ``semantic_labels[id] == "ambiguous"`` rather than guessed at.
        ``"anthropic"``/``"openai"``/``"ollama"`` invoke the checked LLM
        disambiguation step (see :func:`_llm_assign_semantics`).

    Returns
    -------
    GeometryClassification
        ``regions`` always has one entry per real detected surface patch
        with every geometric property computed; ``semantic_labels``/
        ``boundary_condition_types`` are filled either by the default
        geometry-only rule (wall/symmetry_plane candidates pass straight
        through, every opening candidate becomes "ambiguous") or by the
        checked LLM step.
    """
    regions = _segment_mesh(
        mesh, angle_tolerance_deg=angle_tolerance_deg, planar_tolerance_deg=planar_tolerance_deg,
        opening_circularity_threshold=opening_circularity_threshold,
        opening_area_fraction_threshold=opening_area_fraction_threshold,
        symmetry_axis_tolerance_deg=symmetry_axis_tolerance_deg,
        symmetry_midplane_tolerance_frac=symmetry_midplane_tolerance_frac,
        symmetry_area_fraction_threshold=symmetry_area_fraction_threshold,
    )

    # Default, geometry-only labels for every region -- wall/symmetry_plane candidates are
    # ALWAYS assigned this way (never re-asked of an LLM, see _llm_assign_semantics's
    # docstring for the real reliability reason), and every "opening" candidate defaults to
    # "ambiguous" unless a working LLM step below resolves it: geometry alone can never tell
    # an inlet from an outlet.
    semantic_labels: Dict[int, str] = {}
    bc_types: Dict[int, str] = {}
    for r in regions:
        if r.geometric_candidate == "opening":
            semantic_labels[r.region_id] = "ambiguous"
            bc_types[r.region_id] = "unspecified"
        elif r.geometric_candidate == "symmetry_plane":
            semantic_labels[r.region_id] = "symmetry"
            bc_types[r.region_id] = "symmetry"
        else:
            semantic_labels[r.region_id] = "wall"
            bc_types[r.region_id] = "no_slip_wall"

    n_openings = sum(1 for r in regions if r.geometric_candidate == "opening")
    llm_used = False
    llm_reasoning = "no LLM provider given -- geometry-only classification (openings honestly left ambiguous)"
    raw_llm_response = ""

    if provider is not None and n_openings > 0:
        opening_labels, opening_bc_types, llm_reasoning, raw_llm_response = _llm_assign_semantics(
            regions, problem_description, provider=provider, model=model, api_key=api_key,
            conversation_store=conversation_store,
        )
        semantic_labels.update(opening_labels)
        bc_types.update(opening_bc_types)
        llm_used = True
    elif provider is not None:
        llm_reasoning = "LLM provider given, but no 'opening' candidate regions were found to disambiguate"

    ambiguous_ids = [rid for rid, label in semantic_labels.items() if label == "ambiguous"]
    return GeometryClassification(
        regions=regions, semantic_labels=semantic_labels, boundary_condition_types=bc_types,
        ambiguous_region_ids=ambiguous_ids, llm_used=llm_used, llm_reasoning=llm_reasoning,
        raw_llm_response=raw_llm_response,
    )
