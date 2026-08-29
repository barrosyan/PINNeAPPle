"""pinneapple_systems.process_components.curved_path_geometry -- the
"build-and-hold" curved-conduit path generator: a straight (vertical)
section, a constant-curvature build section transitioning to a target
inclination, and a straight tangent/hold section at that inclination.

Used for directional wellbore trajectories, but the geometry itself is
generic: any conduit whose inclination changes at a constant rate over a
transition length and is otherwise straight (a curved pipeline routing
segment, a bent-tube heat exchanger pass, ...) is the same path shape.

SELECTED FORMULATION: inclination is measured from horizontal (90 deg =
vertical, 0 deg = horizontal) as a function of measured (along-path)
depth: constant at 90 deg up to the vertical section's end, linear in
measured depth (i.e. constant curvature -- literally a circular arc)
through the build section, then constant at `final_inclination_deg`.
True vertical depth (TVD) and horizontal displacement (HD) are obtained
by integrating `d(TVD)/ds = sin(inclination(s))`,
`d(HD)/ds = cos(inclination(s))` over path length `s` via a centered
(midpoint) Riemann sum on uniform bins -- second-order accurate in bin
width, and exact for the vertical and hold sections where inclination is
constant.

Independent check: because the build section has constant curvature by
construction, its TVD/HD accumulation has a closed form (the chord
projections of a circular arc of radius `R = build_length /
delta_theta_rad`), used in this module's own test suite to cross-check
the discretized integration.

VALIDITY ENVELOPE: single-plane (2D) path only (no azimuth/turn
component -- a "J" or "S" profile in one vertical plane). For a fully
3D (azimuth-varying) trajectory, the minimum-curvature method used in
directional-drilling survey computation is the standard generalization
and is not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CurvedPathProfile:
    s_m: np.ndarray
    inclination_deg: np.ndarray
    tvd_m: np.ndarray
    hd_m: np.ndarray
    n_bins: int
    vertical_length_m: float
    total_length_m: float
    bin_width_m: float


def inclination_at_depth(
    s_m: np.ndarray, vertical_length_m: float, build_length_m: float, final_inclination_deg: float,
) -> np.ndarray:
    """Inclination [deg, from horizontal] at each measured depth `s_m`:
    90 up to `vertical_length_m`, linear (constant curvature) through the
    following `build_length_m`, held at `final_inclination_deg` after."""
    build_end_m = vertical_length_m + build_length_m
    frac = np.clip((s_m - vertical_length_m) / max(build_length_m, 1e-9), 0.0, 1.0)
    building = 90.0 - (90.0 - final_inclination_deg) * frac
    return np.where(s_m <= vertical_length_m, 90.0, np.where(s_m <= build_end_m, building, final_inclination_deg))


def build_and_hold_profile(
    total_length_m: float, build_length_m: float, final_inclination_deg: float,
    hold_length_m: float, bin_width_m: float,
) -> CurvedPathProfile:
    """Discretize the build-and-hold path into `bin_width_m`-wide bins
    (midpoint-evaluated) and integrate TVD/HD along it. `vertical_length_m
    = total_length_m - build_length_m - hold_length_m` is derived, not an
    independent input, so the three lengths are always consistent."""
    vertical_length_m = total_length_m - build_length_m - hold_length_m
    if vertical_length_m < 0:
        raise ValueError(
            f"build_length_m ({build_length_m}) + hold_length_m ({hold_length_m}) "
            f"exceed total_length_m ({total_length_m}) -- vertical_length_m would be "
            f"negative ({vertical_length_m:.3f})"
        )
    if not (0.0 <= final_inclination_deg <= 90.0):
        raise ValueError(f"final_inclination_deg must be in [0, 90], got {final_inclination_deg}")

    n_bins = max(1, int(np.floor(total_length_m / bin_width_m)))
    s_m = (np.arange(n_bins) + 0.5) * bin_width_m
    inc_deg = inclination_at_depth(s_m, vertical_length_m, build_length_m, final_inclination_deg)
    inc_rad = np.radians(inc_deg)
    sin_i, cos_i = np.sin(inc_rad), np.cos(inc_rad)
    tvd_m = np.cumsum(sin_i * bin_width_m) - 0.5 * sin_i * bin_width_m
    hd_m = np.cumsum(cos_i * bin_width_m) - 0.5 * cos_i * bin_width_m

    return CurvedPathProfile(
        s_m=s_m, inclination_deg=inc_deg, tvd_m=tvd_m, hd_m=hd_m,
        n_bins=n_bins, vertical_length_m=vertical_length_m,
        total_length_m=total_length_m, bin_width_m=bin_width_m,
    )


def circular_arc_tvd_hd(s_within_build_m: np.ndarray, build_length_m: float, final_inclination_deg: float) -> tuple:
    """Closed-form TVD/HD accumulated over the build section only, for a
    circular arc of constant curvature `kappa = delta_theta_rad /
    build_length_m` (delta_theta = 90 - final_inclination_deg, in rad):
    `tvd(s) = sin(kappa*s)/kappa`, `hd(s) = (1 - cos(kappa*s))/kappa`.
    Independent closed-form cross-check for `build_and_hold_profile`'s
    discretized build-section accumulation (see module docstring)."""
    delta_theta_rad = np.radians(90.0 - final_inclination_deg)
    kappa = delta_theta_rad / build_length_m
    if kappa < 1e-12:
        return s_within_build_m.copy(), np.zeros_like(s_within_build_m)
    tvd = np.sin(kappa * s_within_build_m) / kappa
    hd = (1.0 - np.cos(kappa * s_within_build_m)) / kappa
    return tvd, hd
