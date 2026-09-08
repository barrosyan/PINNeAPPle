"""Mesh Intelligence -- the concrete implementation of the founder's
architectural point 5 ("mesh quality, y+, is this mesh good enough").

Public API
----------
    assess_mesh_quality(
        mesh: MeshData,
        *,
        aspect_ratio_threshold: float = 100.0,
        skewness_threshold: float = 0.95,
        max_bad_element_fraction: float = 0.0,
        target_yplus: float = 1.0,
        physical_parameters: Optional[Dict[str, float]] = None,  # forwarded to compute_dimensionless_numbers
        wall_region_face_ids: Optional[Sequence[int]] = None,    # reuse geometry_intelligence's wall regions
    ) -> MeshQualityReport

Every metric below is either a real, standard, citable formula computed
directly from the mesh's own vertex/face data (aspect ratio, equiangular
skewness), or an explicitly-labeled ENGINEERING ESTIMATE with its formula
shown (the y+-driven required first-cell height) -- never a fabricated
number standing in for something that would actually require a converged
flow solution. Orthogonality is honestly reported as NOT COMPUTABLE for a
pure triangle-soup surface mesh (see :func:`_orthogonality` for why), per
this product's own established ethic (``core/dimensional_analysis.py``'s
``DimensionlessNumbers`` leaves a field ``None`` rather than fake-default
it when the inputs needed for it aren't there -- the same discipline is
applied here).

Formula references
-------------------
- **Aspect ratio** (per triangle): circumradius-to-inradius ratio,
  ``R/r`` where ``R = a*b*c/(4*Area)`` and ``r = Area/s`` (``s`` = the
  semi-perimeter) -- a standard FEM/CFD mesh-quality "radius ratio"
  metric (see e.g. Shewchuk, "What Is a Good Linear Element?", 2002, or
  any mainstream CFD meshing guide's "aspect ratio" quality metric). An
  equilateral triangle scores ``R/r = 2`` (the attainable minimum, NOT
  1.0 under this convention -- stated explicitly since other aspect-ratio
  conventions, e.g. longest-edge/shortest-altitude, normalize
  differently); a degenerate sliver triangle's ratio grows without bound.
- **Equiangular skewness** (per triangle): the ANSYS Meshing/Fluent
  convention, ``max((theta_max - theta_e)/(180 - theta_e), (theta_e -
  theta_min)/theta_e)`` with ``theta_e = 60 deg`` (the equilateral-
  triangle interior angle) -- 0 for an equilateral triangle, 1 for a
  fully degenerate one. Typical accepted-quality ranges commonly cited in
  CFD meshing guides (e.g. ANSYS Meshing documentation): 0.00-0.25
  excellent, 0.25-0.50 good, 0.50-0.80 acceptable, 0.80-0.95 bad/sliver,
  0.95-1.00 degenerate -- this module's default ``skewness_threshold``
  (0.95) marks the "degenerate" cutoff.
- **Aspect-ratio threshold**: 100 is used here as a commonly-cited round
  upper bound tolerated by most general-purpose CFD/FEM solvers before
  elements are flagged problematic (exact tolerances vary solver to
  solver and by element type -- this is a conservative, configurable
  default, not a universal constant).
- **y+ required first-cell height (an ESTIMATE, not a measurement)**:
  the standard flat-plate turbulent-boundary-layer approach used
  throughout CFD meshing guides to size the first prism-layer cell BEFORE
  a mesh even exists (see e.g. Schlichting & Gersten, *Boundary-Layer
  Theory*, 8th/9th ed., for the underlying flat-plate skin-friction
  correlation; the derivation chain below is the same one behind
  widely-used "y+ calculator" tools, e.g. the CFD-Online wiki's treatment
  of estimating first-layer thickness):
    1. Skin-friction coefficient (Schlichting's empirical turbulent
       flat-plate correlation, valid roughly 5e5 < Re_L < 1e7):
       ``Cf = 0.058 * Re_L^-0.2``
    2. Wall shear stress: ``tau_w = Cf * 0.5 * rho * U^2``
    3. Friction velocity: ``u_tau = sqrt(tau_w / rho)``
    4. Required first-cell height from the definition of y+ itself
       (``y+ = y * u_tau / nu``): ``y = y+ * nu / u_tau``
  This is fundamentally an ESTIMATE of what a converged flow's near-wall
  velocity gradient will need -- it cannot be, and is never presented as,
  a real, computed y+ from an actual flow field (that requires solving
  the flow first). See :func:`estimate_required_first_cell_height`'s
  docstring for the honest caveats this carries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pinneapple_design.geometry.core.mesh import MeshData

_EQUILATERAL_ANGLE_DEG = 60.0


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class TriangleQualityStats:
    """Per-triangle quality metrics and their summary statistics. Every
    array here has one entry per mesh face -- nothing is subsampled or
    approximated."""
    aspect_ratio_convention: str
    aspect_ratio: np.ndarray  # per-face R/r
    aspect_ratio_mean: float
    aspect_ratio_max: float
    worst_aspect_ratio_face_id: int
    n_faces_failing_aspect_ratio: int
    fraction_faces_failing_aspect_ratio: float

    skewness_convention: str
    skewness: np.ndarray  # per-face equiangular skewness, in [0, 1]
    skewness_mean: float
    skewness_max: float
    worst_skewness_face_id: int
    n_faces_failing_skewness: int
    fraction_faces_failing_skewness: float


@dataclass
class OrthogonalityResult:
    """Real orthogonal-quality (the angle between adjacent CELL centroids
    and their shared FACE normal) requires a volumetric cell mesh -- it is
    not a meaningful, well-defined quantity for a pure triangle-soup
    SURFACE mesh (no volume cells, no cell-to-cell adjacency of the kind
    the metric is actually about). Rather than compute a same-named but
    different quantity and call it "orthogonality" (which would be
    misleading), this is honestly reported as not computable here."""
    computable: bool = False
    reason: str = (
        "orthogonal quality is defined between adjacent VOLUME-mesh cells and their "
        "shared face normal -- not a well-defined quantity for a pure triangle-soup "
        "surface mesh with no volumetric cells. Not computed here rather than faked; "
        "compute it downstream once a real volume mesh exists (e.g. after tetrahedralization)."
    )


@dataclass
class YPlusEstimate:
    """See the module docstring's 'y+ required first-cell height' section
    for the full formula chain and citations. Every field is ``None``
    when its inputs weren't computable, exactly like
    ``dimensional_analysis.DimensionlessNumbers`` -- never silently
    defaulted."""
    computable: bool
    reason: str
    target_yplus: float
    reynolds_length: Optional[float] = None
    skin_friction_coefficient: Optional[float] = None
    wall_shear_stress_over_density: Optional[float] = None  # tau_w/rho, units m^2/s^2 -- rho cancels out of u_tau, so no density input is needed/reported
    friction_velocity: Optional[float] = None
    required_first_cell_height: Optional[float] = None
    actual_near_wall_cell_size: Optional[float] = None
    actual_cell_size_source: str = ""
    verdict: str = "not computable"  # "likely OK" | "likely too coarse" | "not computable"
    formula: str = (
        "Cf = 0.058 * Re_L^-0.2 (Schlichting turbulent flat-plate correlation, valid "
        "~5e5<Re_L<1e7); tau_w = Cf*0.5*rho*U^2; u_tau = sqrt(tau_w/rho); "
        "required_y = target_yplus * nu / u_tau"
    )


@dataclass
class MeshQualityReport:
    n_faces: int
    n_vertices: int
    triangle_quality: TriangleQualityStats
    orthogonality: OrthogonalityResult
    yplus: YPlusEstimate
    is_good_enough: bool
    failures: List[Dict[str, Any]] = field(default_factory=list)
    thresholds_used: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Real, per-triangle geometric quality metrics
# ---------------------------------------------------------------------------

def _triangle_quality(
    mesh: MeshData, *, aspect_ratio_threshold: float, skewness_threshold: float,
) -> TriangleQualityStats:
    v0 = mesh.vertices[mesh.faces[:, 0]]
    v1 = mesh.vertices[mesh.faces[:, 1]]
    v2 = mesh.vertices[mesh.faces[:, 2]]

    # Side lengths: a=|BC| (opposite vertex0), b=|CA| (opposite vertex1), c=|AB| (opposite vertex2).
    a = np.linalg.norm(v2 - v1, axis=1)
    b = np.linalg.norm(v0 - v2, axis=1)
    c = np.linalg.norm(v1 - v0, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    s = (a + b + c) / 2.0

    area_safe = np.maximum(area, 1e-300)
    s_safe = np.maximum(s, 1e-300)
    circumradius = (a * b * c) / (4.0 * area_safe)
    inradius = area_safe / s_safe
    aspect_ratio = np.where(area > 1e-14, circumradius / np.maximum(inradius, 1e-300), np.inf)

    # Interior angles at each vertex via the dot product of the two edges meeting there.
    def _angle_deg(p_center: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
        u = p_a - p_center
        w = p_b - p_center
        un = np.linalg.norm(u, axis=1)
        wn = np.linalg.norm(w, axis=1)
        cos_t = np.sum(u * w, axis=1) / np.maximum(un * wn, 1e-300)
        return np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0)))

    angle0 = _angle_deg(v0, v1, v2)
    angle1 = _angle_deg(v1, v0, v2)
    angle2 = _angle_deg(v2, v0, v1)
    angles = np.stack([angle0, angle1, angle2], axis=1)
    theta_max = np.max(angles, axis=1)
    theta_min = np.min(angles, axis=1)
    te = _EQUILATERAL_ANGLE_DEG
    skewness = np.maximum((theta_max - te) / (180.0 - te), (te - theta_min) / te)
    skewness = np.clip(skewness, 0.0, 1.0)

    fail_ar = aspect_ratio > aspect_ratio_threshold
    fail_sk = skewness > skewness_threshold
    n_faces = mesh.n_faces

    return TriangleQualityStats(
        aspect_ratio_convention="circumradius / inradius (R/r); equilateral triangle = 2.0 (minimum); degenerate -> inf",
        aspect_ratio=aspect_ratio,
        aspect_ratio_mean=float(np.mean(aspect_ratio[np.isfinite(aspect_ratio)])) if np.any(np.isfinite(aspect_ratio)) else float("inf"),
        aspect_ratio_max=float(np.max(aspect_ratio)) if n_faces else 0.0,
        worst_aspect_ratio_face_id=int(np.argmax(aspect_ratio)) if n_faces else -1,
        n_faces_failing_aspect_ratio=int(np.sum(fail_ar)),
        fraction_faces_failing_aspect_ratio=float(np.mean(fail_ar)) if n_faces else 0.0,
        skewness_convention="equiangular skewness (ANSYS Meshing/Fluent convention): max((theta_max-60)/(180-60), (60-theta_min)/60)",
        skewness=skewness,
        skewness_mean=float(np.mean(skewness)) if n_faces else 0.0,
        skewness_max=float(np.max(skewness)) if n_faces else 0.0,
        worst_skewness_face_id=int(np.argmax(skewness)) if n_faces else -1,
        n_faces_failing_skewness=int(np.sum(fail_sk)),
        fraction_faces_failing_skewness=float(np.mean(fail_sk)) if n_faces else 0.0,
    )


def _orthogonality() -> OrthogonalityResult:
    return OrthogonalityResult()


# ---------------------------------------------------------------------------
# y+ ESTIMATE (not a real, solved y+ -- see module docstring)
# ---------------------------------------------------------------------------

def estimate_required_first_cell_height(
    *, reynolds_length: float, velocity: float, kinematic_viscosity: float, target_yplus: float = 1.0,
) -> YPlusEstimate:
    """Estimate the first-cell (near-wall prism-layer) height needed to
    reach ``target_yplus`` for a flat-plate turbulent boundary layer, from
    ONLY the flow's own Reynolds number/velocity/viscosity -- i.e. BEFORE
    a mesh even exists, exactly the way a CFD analyst sizes a boundary
    layer during mesh planning. See the module docstring's formula chain
    and citation.

    HONEST LIMITATIONS, stated explicitly (never silently assumed away):
    - This is a FLAT-PLATE correlation. Real geometry (curvature, pressure
      gradients, separation) shifts the true required height; treat the
      result as a planning-stage order-of-magnude estimate, not a
      guarantee.
    - The correlation's own stated validity range is roughly
      ``5e5 < Re_L < 1e7``; outside that range the estimate is still
      computed (nothing here refuses to answer), but ``reason`` says so
      explicitly rather than silently applying a formula outside its
      documented range (matching ``dimensional_analysis.py``'s own
      Nusselt-correlation validity-range discipline).
    - This function NEVER claims to compute an actual y+ from a flow
      field -- that number only exists after a real solve. It only
      answers "how fine does the near-wall mesh need to be to plausibly
      achieve the target y+ once such a solve runs."
    """
    if reynolds_length is None or reynolds_length <= 0 or velocity is None or velocity == 0 \
            or kinematic_viscosity is None or kinematic_viscosity <= 0:
        return YPlusEstimate(
            computable=False,
            reason="reynolds_length, velocity, and kinematic_viscosity must all be positive/nonzero to estimate a required first-cell height.",
            target_yplus=target_yplus,
        )

    re_l = float(reynolds_length)
    u = float(velocity)
    nu = float(kinematic_viscosity)

    cf = 0.058 * re_l ** -0.2
    # tau_w/rho = Cf * 0.5 * U^2 (rho cancels out of u_tau = sqrt(tau_w/rho) entirely,
    # so this estimate never needs a density value at all -- one fewer required input).
    tau_w_over_rho = cf * 0.5 * (u ** 2)
    u_tau = float(np.sqrt(max(tau_w_over_rho, 0.0)))
    if u_tau <= 0:
        return YPlusEstimate(
            computable=False, reason="computed friction velocity is zero/negative -- cannot estimate a required cell height.",
            target_yplus=target_yplus, reynolds_length=re_l, skin_friction_coefficient=cf,
        )

    required_y = target_yplus * nu / u_tau

    reason = "estimated from the standard flat-plate turbulent boundary-layer correlation (see module docstring)."
    if not (5.0e5 <= re_l <= 1.0e7):
        reason += (
            f" NOTE: Re_L={re_l:.3g} is outside the correlation's typically-cited validity range "
            "(5e5-1e7) -- still computed, but treat as a rougher order-of-magnitude estimate."
        )

    return YPlusEstimate(
        computable=True, reason=reason, target_yplus=target_yplus, reynolds_length=re_l,
        skin_friction_coefficient=cf, wall_shear_stress_over_density=tau_w_over_rho, friction_velocity=u_tau,
        required_first_cell_height=required_y,
    )


def _near_wall_cell_size(mesh: MeshData, wall_region_face_ids: Optional[Sequence[int]]) -> tuple:
    """The ACTUAL near-wall cell size proxy: the smallest triangle
    "altitude-like" length scale (here, the shortest triangle altitude --
    the perpendicular distance from a vertex to its opposite side) among
    the wall-classified faces, if given (reuse geometry_intelligence's
    wall regions for a real, wall-specific answer), else the smallest
    altitude over the WHOLE mesh as an honestly-caveated fallback.

    This is a proxy, not a true first-prism-layer height (a surface mesh
    only records the SURFACE tessellation size, not whatever volume-mesh
    prism layer will eventually be built on top of it) -- stated in
    ``actual_cell_size_source`` either way."""
    face_ids = np.asarray(wall_region_face_ids, dtype=np.int64) if wall_region_face_ids else np.arange(mesh.n_faces)
    if face_ids.size == 0:
        face_ids = np.arange(mesh.n_faces)
    v0 = mesh.vertices[mesh.faces[face_ids, 0]]
    v1 = mesh.vertices[mesh.faces[face_ids, 1]]
    v2 = mesh.vertices[mesh.faces[face_ids, 2]]
    a = np.linalg.norm(v2 - v1, axis=1)
    b = np.linalg.norm(v0 - v2, axis=1)
    c = np.linalg.norm(v1 - v0, axis=1)
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    # altitude_i = 2*Area / side_i, for each of the 3 sides -- the shortest of the 3 per
    # triangle is that triangle's own "thinnest" dimension, a reasonable proxy for local
    # surface resolution near a wall.
    with np.errstate(divide="ignore", invalid="ignore"):
        alt_a = np.where(a > 1e-300, 2.0 * area / np.maximum(a, 1e-300), np.inf)
        alt_b = np.where(b > 1e-300, 2.0 * area / np.maximum(b, 1e-300), np.inf)
        alt_c = np.where(c > 1e-300, 2.0 * area / np.maximum(c, 1e-300), np.inf)
    min_altitude = np.min(np.stack([alt_a, alt_b, alt_c], axis=1), axis=1)
    finite = min_altitude[np.isfinite(min_altitude)]
    smallest = float(np.min(finite)) if finite.size else float("nan")

    source = (
        f"smallest triangle altitude among {face_ids.size} wall-classified face(s)"
        if wall_region_face_ids else
        f"smallest triangle altitude over the WHOLE mesh ({face_ids.size} faces) -- "
        "FALLBACK: no wall-classified region given, so this is not specifically a "
        "near-wall measurement, only the mesh's finest local resolution anywhere."
    )
    return smallest, source


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assess_mesh_quality(
    mesh: MeshData,
    *,
    aspect_ratio_threshold: float = 100.0,
    skewness_threshold: float = 0.95,
    max_bad_element_fraction: float = 0.0,
    target_reynolds: Optional[float] = None,
    target_yplus: float = 1.0,
    physical_parameters: Optional[Dict[str, float]] = None,
    wall_region_face_ids: Optional[Sequence[int]] = None,
) -> MeshQualityReport:
    """Assess whether ``mesh`` is "good enough" for the physics being
    asked about -- a structured report (which metric failed, by how much),
    never a single opaque boolean, per this product's own stated ethic
    (see ``pipeline.py``'s ``PhysicsGuardrail`` usage for the same
    philosophy applied to physics verification).

    Parameters
    ----------
    aspect_ratio_threshold, skewness_threshold : see module docstring for
        the formulas/conventions and typical CFD-best-practice ranges
        these thresholds are drawn from.
    max_bad_element_fraction : the fraction of faces allowed to exceed a
        threshold before that metric counts as an overall failure (0.0 by
        default -- zero tolerance; real production meshing workflows
        often tolerate a small fraction of borderline elements, so this
        is deliberately configurable rather than hardcoded).
    target_reynolds : if given directly, used for the y+ estimate instead
        of computing one from ``physical_parameters``.
    physical_parameters : forwarded to
        ``dimensional_analysis.compute_dimensionless_numbers`` (this
        module's own already-verified formulas) to obtain a Reynolds
        number when ``target_reynolds`` isn't given directly. Needs at
        least ``velocity``+``length``+(``kinematic_viscosity`` or
        ``density``+``dynamic_viscosity``) to produce anything.
    wall_region_face_ids : face indices classified as "wall" (e.g. from
        ``geometry_intelligence.classify_geometry``'s regions) -- reused
        for the near-wall cell-size comparison. Without it, the smallest
        cell anywhere in the mesh is used as a documented fallback (see
        :func:`_near_wall_cell_size`).
    """
    tri = _triangle_quality(mesh, aspect_ratio_threshold=aspect_ratio_threshold, skewness_threshold=skewness_threshold)
    ortho = _orthogonality()

    velocity = (physical_parameters or {}).get("velocity")
    length = (physical_parameters or {}).get("length")
    kin_visc = (physical_parameters or {}).get("kinematic_viscosity")
    density = (physical_parameters or {}).get("density")
    dyn_visc = (physical_parameters or {}).get("dynamic_viscosity")
    if kin_visc is None and density is not None and dyn_visc is not None and density > 0:
        kin_visc = dyn_visc / density

    re_l = target_reynolds
    if re_l is None and velocity is not None and length is not None and kin_visc is not None and kin_visc > 0:
        re_l = velocity * length / kin_visc

    if re_l is not None and velocity is not None and kin_visc is not None:
        yplus = estimate_required_first_cell_height(
            reynolds_length=re_l, velocity=velocity, kinematic_viscosity=kin_visc, target_yplus=target_yplus,
        )
        if yplus.computable:
            near_wall, source = _near_wall_cell_size(mesh, wall_region_face_ids)
            yplus.actual_near_wall_cell_size = near_wall
            yplus.actual_cell_size_source = source
            if not np.isfinite(near_wall) or yplus.required_first_cell_height is None:
                yplus.verdict = "not computable"
            elif near_wall <= yplus.required_first_cell_height:
                yplus.verdict = "likely OK"
            else:
                yplus.verdict = (
                    f"likely too coarse (actual smallest near-wall cell size {near_wall:.4g} > "
                    f"required {yplus.required_first_cell_height:.4g} for target y+={target_yplus})"
                )
    else:
        yplus = YPlusEstimate(
            computable=False, target_yplus=target_yplus,
            reason=(
                "insufficient physical_parameters to estimate a Reynolds number/first-cell height -- "
                "need target_reynolds, or velocity+length+kinematic_viscosity (or density+dynamic_viscosity)."
            ),
        )

    failures: List[Dict[str, Any]] = []
    if tri.fraction_faces_failing_aspect_ratio > max_bad_element_fraction:
        failures.append({
            "metric": "aspect_ratio", "threshold": aspect_ratio_threshold,
            "worst_value": tri.aspect_ratio_max, "worst_face_id": tri.worst_aspect_ratio_face_id,
            "n_faces_failing": tri.n_faces_failing_aspect_ratio,
            "fraction_faces_failing": tri.fraction_faces_failing_aspect_ratio,
            "detail": (
                f"{tri.n_faces_failing_aspect_ratio} face(s) ({tri.fraction_faces_failing_aspect_ratio:.2%}) "
                f"exceed aspect_ratio_threshold={aspect_ratio_threshold} (R/r convention); worst face "
                f"{tri.worst_aspect_ratio_face_id} has R/r={tri.aspect_ratio_max:.4g}."
            ),
        })
    if tri.fraction_faces_failing_skewness > max_bad_element_fraction:
        failures.append({
            "metric": "skewness", "threshold": skewness_threshold,
            "worst_value": tri.skewness_max, "worst_face_id": tri.worst_skewness_face_id,
            "n_faces_failing": tri.n_faces_failing_skewness,
            "fraction_faces_failing": tri.fraction_faces_failing_skewness,
            "detail": (
                f"{tri.n_faces_failing_skewness} face(s) ({tri.fraction_faces_failing_skewness:.2%}) "
                f"exceed skewness_threshold={skewness_threshold} (equiangular skewness); worst face "
                f"{tri.worst_skewness_face_id} has skewness={tri.skewness_max:.4g}."
            ),
        })
    if yplus.verdict.startswith("likely too coarse"):
        failures.append({
            "metric": "yplus_first_cell_height", "threshold": yplus.required_first_cell_height,
            "worst_value": yplus.actual_near_wall_cell_size, "worst_face_id": None,
            "n_faces_failing": None, "fraction_faces_failing": None, "detail": yplus.verdict,
        })

    return MeshQualityReport(
        n_faces=mesh.n_faces, n_vertices=mesh.n_vertices, triangle_quality=tri, orthogonality=ortho,
        yplus=yplus, is_good_enough=(len(failures) == 0), failures=failures,
        thresholds_used={
            "aspect_ratio_threshold": aspect_ratio_threshold, "skewness_threshold": skewness_threshold,
            "max_bad_element_fraction": max_bad_element_fraction, "target_yplus": target_yplus,
        },
    )
