"""Shared aerodynamic-coefficient utilities (Cp, surface-force integration, CL/CD/CM).

Plain numpy in / numpy out -- no framework lock-in, so both PINN surrogates
(which typically hand back predictions as numpy arrays after `.detach()
.cpu().numpy()`) and non-PINN solvers (analytical, LBM, panel methods, ...)
can share one implementation instead of every example re-deriving the same
algebra inline.

Sign / axis conventions
------------------------
The pressure-coefficient definition (``Cp = (p - p_inf) / q_inf``) is the
standard aerodynamics convention and matches the inline computation in
``examples/benchmark_suite/15_naca0012_aerodynamic_surrogate.py`` (which uses
the incompressible Bernoulli special case ``Cp = 1 - |V|^2 / U_inf^2``, i.e.
``p - p_inf = q_inf * (1 - |V/U_inf|^2)``).

The wind-axis rotation used by :func:`force_to_coefficients` mirrors the
normal/axial -> lift/drag transform in
``examples/use_cases/missile_aero/missile_aero_pipeline.py``
(see the module docstring there, and lines ~262-267 of
``solve_aero()``)::

    CN = normal-force coefficient   (body axis, perpendicular to axis of symmetry)
    CA = axial-force coefficient    (body axis, along axis of symmetry)
    CL = CN * cos(alpha) - CA * sin(alpha)
    CD = CN * sin(alpha) + CA * cos(alpha)

That pipeline works in a single meridional plane, so "normal" and "axial"
map directly onto two coordinate axes. This module generalizes the same
rotation to a full 3D force vector by picking a 2D "aero plane" out of the
3D force via a ``(drag_axis, lift_axis)`` pair and rotating within that
plane by ``alpha``; any component of the force orthogonal to both (e.g. a
sideforce) is reported separately and is NOT folded into CL/CD.

The default convention is a 2D-in-3D setup consistent with the rest of the
repo's LBM/PINN CFD tooling (x = streamwise/free-stream direction, z =
"vertical"/lift direction, y = spanwise): alpha rotates the force from the
*body axes* (x = axial, z = normal) into *wind axes* (drag along the
free-stream direction, lift perpendicular to it) within the x-z plane,
using exactly the CN/CA -> CL/CD sign convention above with
``CN -> F_normal`` (the lift-axis component of body force) and
``CA -> F_axial`` (the drag-axis component of body force)::

    CL = CN * cos(alpha) - CA * sin(alpha)
    CD = CN * sin(alpha) + CA * cos(alpha)

Callers with a different body-axis layout (e.g. a wing with span along x)
should pass an explicit ``drag_axis`` / ``lift_axis`` pair.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


# ---------------------------------------------------------------------------
# Pressure coefficient
# ---------------------------------------------------------------------------

def compute_cp(
    p: np.ndarray,
    p_inf: float,
    q_inf: float,
) -> np.ndarray:
    """Pressure coefficient ``Cp = (p - p_inf) / q_inf``.

    Parameters
    ----------
    p     : array_like, static pressure at each sample point.
    p_inf : float, free-stream static pressure.
    q_inf : float, free-stream dynamic pressure (``0.5 * rho_inf * U_inf**2``).
             Must be nonzero.

    Returns
    -------
    np.ndarray, same shape as ``p``.
    """
    p = np.asarray(p, dtype=np.float64)
    if q_inf == 0.0:
        raise ValueError("q_inf must be nonzero")
    return (p - p_inf) / q_inf


# ---------------------------------------------------------------------------
# Surface force / moment integration
# ---------------------------------------------------------------------------

def _face_areas_and_centroids(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangle areas (M,) and centroids (M,3) from a (N,3)/(M,3) mesh."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    centroids = (v0 + v1 + v2) / 3.0
    return areas, centroids


def integrate_surface_forces(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_normals: np.ndarray,
    face_pressure: np.ndarray,
    face_shear: Optional[np.ndarray] = None,
    moment_center: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate per-face pressure (and optional shear) into a net force and
    moment over a triangulated surface.

    The pressure contribution on face ``i`` is
    ``-face_pressure[i] * area[i] * face_normals[i]`` (pressure acts INTO the
    surface, i.e. opposite the outward normal -- the standard convention for
    aerodynamic surface-pressure integration). If ``face_shear`` is given, it
    is added directly (already a vector traction per face, force = shear *
    area, no normal-vs-tangent decomposition is imposed on the caller -- pass
    the wall-shear-stress vector already projected tangent to the surface).

    Parameters
    ----------
    vertices      : (N, 3) float array of mesh vertex coordinates.
    faces         : (M, 3) int array of triangle vertex indices.
    face_normals  : (M, 3) float array of OUTWARD unit normals, one per face.
    face_pressure : (M,) float array of static pressure at each face.
    face_shear    : (M, 3) float array of wall-shear-stress vectors, optional.
    moment_center : (3,) float array, reference point for the moment.
                    Defaults to the origin.

    Returns
    -------
    force  : (3,) net aerodynamic force vector.
    moment : (3,) net aerodynamic moment vector about ``moment_center``.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_normals = np.asarray(face_normals, dtype=np.float64)
    face_pressure = np.asarray(face_pressure, dtype=np.float64).reshape(-1)

    if face_normals.shape[0] != faces.shape[0]:
        raise ValueError("face_normals must have one row per face")
    if face_pressure.shape[0] != faces.shape[0]:
        raise ValueError("face_pressure must have one entry per face")

    if moment_center is None:
        moment_center = np.zeros(3, dtype=np.float64)
    else:
        moment_center = np.asarray(moment_center, dtype=np.float64).reshape(3)

    # Normalize normals defensively (callers may pass unnormalized face normals).
    norm_mag = np.linalg.norm(face_normals, axis=1, keepdims=True)
    norm_mag = np.where(norm_mag > 1e-14, norm_mag, 1.0)
    unit_normals = face_normals / norm_mag

    areas, centroids = _face_areas_and_centroids(vertices, faces)

    # Pressure acts opposite the outward normal.
    df = -face_pressure[:, None] * areas[:, None] * unit_normals

    if face_shear is not None:
        face_shear = np.asarray(face_shear, dtype=np.float64)
        if face_shear.shape[0] != faces.shape[0]:
            raise ValueError("face_shear must have one entry per face")
        df = df + face_shear * areas[:, None]

    force = df.sum(axis=0)
    arm = centroids - moment_center[None, :]
    moment = np.cross(arm, df).sum(axis=0)

    return force, moment


# ---------------------------------------------------------------------------
# Force -> aerodynamic coefficients
# ---------------------------------------------------------------------------

@dataclass
class AeroCoefficients:
    """Non-dimensional aerodynamic coefficients about the wind axes.

    Attributes
    ----------
    CL : lift coefficient (along ``lift_axis``, wind-axis rotated).
    CD : drag coefficient (along the free-stream direction, wind-axis rotated).
    CM : moment coefficient about ``moment_axis`` (default: perpendicular to
         the drag/lift plane, i.e. the standard pitching-moment axis).
    CY : sideforce coefficient -- component of the force orthogonal to both
         the drag and lift axes, non-dimensionalized the same way as CD/CL.
         Zero for a purely 2D-in-3D problem.
    """
    CL: float
    CD: float
    CM: float
    CY: float = 0.0


def force_to_coefficients(
    force: np.ndarray,
    moment: np.ndarray,
    q_inf: float,
    ref_area: float,
    ref_length: float,
    alpha_deg: float,
    *,
    drag_axis: str = "x",
    lift_axis: str = "z",
    moment_axis: Optional[str] = None,
) -> AeroCoefficients:
    """Convert a net body-axis force/moment into wind-axis CL/CD/CM.

    Rotation convention (matches
    ``examples/use_cases/missile_aero/missile_aero_pipeline.py``,
    ``solve_aero()``, lines ~262-267)::

        CN = force component along ``lift_axis``   (body axis)
        CA = force component along ``drag_axis``   (body axis)
        CL = CN * cos(alpha) - CA * sin(alpha)
        CD = CN * sin(alpha) + CA * cos(alpha)

    where ``alpha`` is measured positive rotating the body's ``drag_axis``
    toward its ``lift_axis`` (i.e. increasing angle of attack tilts the body
    axis toward the lift direction, exactly as in the missile pipeline where
    increasing alpha increases the leeward body angle -- see
    ``body_angle_distribution()`` in that file).

    Parameters
    ----------
    force, moment : (3,) arrays, e.g. from :func:`integrate_surface_forces`.
    q_inf         : free-stream dynamic pressure.
    ref_area      : reference area for force normalization.
    ref_length    : reference length for moment normalization.
    alpha_deg     : angle of attack in degrees.
    drag_axis, lift_axis : one of "x", "y", "z" (default 3D-in-2D convention:
        x = free-stream/axial direction, z = "up"/normal direction). Must be
        distinct.
    moment_axis   : axis about which CM is reported. Defaults to whichever
        of "x", "y", "z" is neither ``drag_axis`` nor ``lift_axis`` (the
        standard pitching-moment axis for a 2D-in-3D problem).

    Returns
    -------
    AeroCoefficients(CL, CD, CM, CY)
    """
    force = np.asarray(force, dtype=np.float64).reshape(3)
    moment = np.asarray(moment, dtype=np.float64).reshape(3)

    if q_inf == 0.0:
        raise ValueError("q_inf must be nonzero")
    if ref_area == 0.0:
        raise ValueError("ref_area must be nonzero")
    if ref_length == 0.0:
        raise ValueError("ref_length must be nonzero")
    if drag_axis == lift_axis:
        raise ValueError("drag_axis and lift_axis must be distinct")
    if drag_axis not in _AXIS_INDEX or lift_axis not in _AXIS_INDEX:
        raise ValueError("drag_axis/lift_axis must each be one of 'x', 'y', 'z'")

    if moment_axis is None:
        remaining = [a for a in ("x", "y", "z") if a not in (drag_axis, lift_axis)]
        moment_axis = remaining[0]
    if moment_axis not in _AXIS_INDEX:
        raise ValueError("moment_axis must be one of 'x', 'y', 'z'")

    i_drag = _AXIS_INDEX[drag_axis]
    i_lift = _AXIS_INDEX[lift_axis]
    i_side = [i for i in range(3) if i not in (i_drag, i_lift)][0]
    i_mom = _AXIS_INDEX[moment_axis]

    CA = force[i_drag] / (q_inf * ref_area)   # body-axis axial-force coeff.
    CN = force[i_lift] / (q_inf * ref_area)   # body-axis normal-force coeff.
    CY = force[i_side] / (q_inf * ref_area)   # sideforce, not rotated

    alpha_rad = math.radians(alpha_deg)
    CL = CN * math.cos(alpha_rad) - CA * math.sin(alpha_rad)
    CD = CN * math.sin(alpha_rad) + CA * math.cos(alpha_rad)

    CM = moment[i_mom] / (q_inf * ref_area * ref_length)

    return AeroCoefficients(CL=float(CL), CD=float(CD), CM=float(CM), CY=float(CY))


__all__ = [
    "compute_cp",
    "integrate_surface_forces",
    "force_to_coefficients",
    "AeroCoefficients",
]
