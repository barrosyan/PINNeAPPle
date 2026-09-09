"""Astropy-backed unit handling and coordinate transforms for astrophysics.

Everything in ``astrophysics.py`` (this package's hand-derived orbital-
mechanics / stellar-structure / GR / accretion-disk presets) works in raw
Python floats with an *implicit* unit convention documented only in
docstrings and parameter comments (e.g. ``mu: float = 398600.4418   # Earth
GM, km^3/s^2``, ``GM: float = 1.32712440018e20   # ... m^3/s^2``). There is
no unit-safety anywhere in that module: nothing stops a caller from passing
a value in the wrong unit system, and several physical constants are
hard-coded as bare floats of varying precision/vintage (see the module
docstring notes below for the discrepancies found against real
``astropy.constants`` values).

This module does NOT reimplement or replace anything in ``astrophysics.py``
(read-only there, untouched). It adds genuinely new, currently-missing
capability, backed by the real, standard `Astropy <https://astropy.org>`_
library (Astropy Collaboration et al. 2022, ApJ, 935, 167):

- ``ASTRO_CONSTANTS``: a dict of live ``astropy.constants.Constant``
  objects (not bare floats) -- each one carries its own real numeric
  value, unit, uncertainty, and literature reference/provenance string,
  queryable directly off the object.
- ``to_si`` / ``with_units``: thin, real wrappers around ``astropy.units``
  for unit-safe parsing and conversion (e.g. ``"10 AU"``, ``"5 Msun"``),
  which raise astropy's own real ``UnitConversionError`` on a physically
  meaningless conversion (mass -> length, etc.) instead of silently
  producing a wrong number the way raw-float code would.
- ``transform_coordinates``: a real wrapper around ``astropy.coordinates``
  implementing the GCRS <-> ITRS transform (respectively: the Geocentric
  Celestial Reference System, a real IAU-standard Earth-Centered Inertial
  frame -- the frame the Earth-centered ``kepler_two_body_orbit``,
  ``space_debris_cw_relative_motion`` and ``satellite_j2_perturbation``
  presets in ``astrophysics.py`` implicitly assume when they say "Earth-
  centered inertial frame" in their docstrings -- and the International
  Terrestrial Reference System, a real Earth-Centered Earth-Fixed/ECEF
  frame), using astropy's full IAU precession/nutation/Earth-rotation/
  polar-motion model rather than a simplified fixed-rate Earth-rotation
  formula.

Discrepancies found against real astropy.constants values (reported
honestly per the task; NOT fixed here -- ``astrophysics.py`` is read-only
and out of scope for this module):

- ``shakura_sunyaev_accretion_disk``'s hard-coded ``G = 6.674e-11`` differs
  from ``astropy.constants.G`` (CODATA 2022) ``= 6.6743e-11`` by
  ~4.5e-5 relative (~0.0045%) -- the hard-coded value is simply truncated
  to 4 significant figures.
- Its hard-coded ``M_sun = 1.98892e30`` kg differs from
  ``astropy.constants.M_sun`` (IAU 2015 nominal solar mass parameter /
  CODATA 2022 G) ``= 1.988409870698051e30`` kg by ~2.6e-4 relative
  (~0.026%) -- an older/different-vintage solar-mass value.
- ``kepler_two_body_orbit`` / ``satellite_j2_perturbation``'s hard-coded
  Earth ``mu = 398600.4418`` km^3/s^2 (Vallado/WGS-84) differs from
  ``astropy.constants.GM_earth`` (IAU 2015 nominal) ``= 398600.4`` km^3/s^2
  by ~1.0e-6 relative -- different reference standards, both legitimate.
- ``satellite_j2_perturbation``'s hard-coded ``Re = 6378.137`` km (WGS-84)
  differs from ``astropy.constants.R_earth`` (IAU 2015 nominal)
  ``= 6378.1`` km by ~5.8e-6 relative -- again different standards.
- ``schwarzschild_light_bending_weak_field``'s hard-coded solar
  ``GM = 1.32712440018e20`` m^3/s^2 and ``shakura_sunyaev_accretion_disk``'s
  ``sigma_SB``, ``m_p``, ``sigma_T`` all agree with the real astropy/CODATA
  values to the precision given (sub-parts-per-billion residual differences
  from CODATA-vintage updates, not meaningful).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import (
    GCRS,
    ITRS,
    CartesianDifferential,
    CartesianRepresentation,
)
from astropy.time import Time

# ===========================================================================
# CONSTANTS
# ===========================================================================

# Live astropy.constants.Constant objects -- each is simultaneously usable
# as a plain float (via arithmetic / float()) AND carries real provenance:
# `.value`, `.unit`, `.si`, `.uncertainty`, `.reference`. NOT bare floats
# copy-pasted out of astropy; these ARE astropy's own objects.
ASTRO_CONSTANTS: Dict[str, "const.Constant"] = {
    "G": const.G,               # Newtonian gravitational constant
    "c": const.c,                # speed of light in vacuum
    "M_sun": const.M_sun,        # solar mass
    "R_sun": const.R_sun,        # nominal solar radius
    "GM_sun": const.GM_sun,      # nominal solar mass parameter (IAU 2015)
    "M_earth": const.M_earth,    # Earth mass
    "R_earth": const.R_earth,    # nominal Earth equatorial radius (IAU 2015)
    "GM_earth": const.GM_earth,  # nominal Earth mass parameter (IAU 2015)
    "au": const.au,              # astronomical unit
    "pc": const.pc,               # parsec
    "sigma_sb": const.sigma_sb,  # Stefan-Boltzmann constant
    "m_p": const.m_p,            # proton mass
    "sigma_T": const.sigma_T,    # Thomson scattering cross-section
}


def astro_constant_si(name: str) -> float:
    """Return the SI-unit float value of ``ASTRO_CONSTANTS[name]``.

    Thin convenience wrapper -- ``ASTRO_CONSTANTS[name].si.value`` -- for
    call sites (like ``astrophysics.py``'s presets) that want a bare float
    in SI units but still want the value sourced live from astropy rather
    than hard-coded.
    """
    if name not in ASTRO_CONSTANTS:
        raise KeyError(
            f"Unknown astro constant {name!r}; available: {sorted(ASTRO_CONSTANTS)}"
        )
    return float(ASTRO_CONSTANTS[name].si.value)


def constant_provenance(name: str) -> str:
    """Return the real astropy literature reference string for a constant.

    E.g. ``constant_provenance("G")`` -> ``"CODATA 2022"``. Lets a caller
    cite exactly which standard/CODATA vintage a value in
    ``ASTRO_CONSTANTS`` came from.
    """
    if name not in ASTRO_CONSTANTS:
        raise KeyError(
            f"Unknown astro constant {name!r}; available: {sorted(ASTRO_CONSTANTS)}"
        )
    return str(ASTRO_CONSTANTS[name].reference)


# ===========================================================================
# UNIT-SAFE CONVERSION
# ===========================================================================

QuantityLike = Union[str, "u.Quantity", float, int]


def with_units(value: Union[float, int, str], unit_str: Optional[str] = None) -> "u.Quantity":
    """Attach real astropy units to a bare number, or parse a unit string.

    Two call forms:
      - ``with_units(10.0, "AU")`` -> ``<Quantity 10. AU>``
      - ``with_units("10 AU")``    -> ``<Quantity 10. AU>`` (unit_str omitted)

    This is a thin wrapper around ``astropy.units.Quantity`` -- it does not
    reimplement any unit logic, it just gives this codebase's presets a
    documented, single entry point for constructing one. Raises
    ``astropy.units.UnitsError`` (astropy's own real exception, not
    swallowed) for a string astropy itself cannot parse as a unit.
    """
    if unit_str is None:
        if not isinstance(value, str):
            raise TypeError(
                "with_units(value) with a single argument requires a string "
                f"like '10 AU'; got {type(value).__name__}. Pass a unit_str "
                "as the second argument for a bare number."
            )
        return u.Quantity(value)
    return u.Quantity(value, unit=u.Unit(unit_str))


def to_si(quantity: QuantityLike, target_unit: Optional[str] = None) -> float:
    """Convert an astropy Quantity (or a parsable string like "5 Msun") to
    a plain float, in SI units by default or in ``target_unit`` if given.

    Examples
    --------
    >>> to_si("10 AU")                  # -> 1.495978707e12 (meters, SI)
    >>> to_si("5 Msun")                 # -> 9.942...e30 (kg, SI)
    >>> to_si("5 Msun", target_unit="kg")
    >>> to_si(with_units(1.0, "yr"), target_unit="s")

    Raises ``astropy.units.UnitConversionError`` -- astropy's own real
    exception, propagated as-is, not caught or swallowed -- if the
    quantity's dimensionality is not convertible to the requested unit
    (e.g. converting a mass to a length). This is exactly the class of
    unit-mismatch bug that raw-float code in ``astrophysics.py`` has no way
    to catch.
    """
    if isinstance(quantity, str):
        q = u.Quantity(quantity)
    elif isinstance(quantity, u.Quantity):
        q = quantity
    elif isinstance(quantity, (int, float)):
        raise TypeError(
            "to_si() requires a Quantity or a unit string (e.g. '5 kg'); "
            "got a bare number with no unit. Use with_units(value, unit_str) "
            "first."
        )
    else:
        raise TypeError(f"Cannot interpret {quantity!r} as an astropy Quantity.")

    if target_unit is not None:
        return float(q.to_value(u.Unit(target_unit)))
    return float(q.si.value)


# ===========================================================================
# COORDINATE-FRAME TRANSFORMS
# ===========================================================================

_FRAME_CLASSES = {"gcrs": GCRS, "itrs": ITRS}


def transform_coordinates(
    x: float,
    y: float,
    z: float,
    obstime: str,
    frame_in: str = "gcrs",
    frame_out: str = "itrs",
    vx: Optional[float] = None,
    vy: Optional[float] = None,
    vz: Optional[float] = None,
    unit: str = "km",
) -> Dict[str, float]:
    """Transform a Cartesian position (and optional velocity) between GCRS
    and ITRS using real ``astropy.coordinates`` machinery.

    GCRS (Geocentric Celestial Reference System) is a real, IAU-standard
    Earth-Centered Inertial (ECI) frame -- the frame ``astrophysics.py``'s
    Earth-centered presets (``kepler_two_body_orbit``,
    ``space_debris_cw_relative_motion``'s reference orbit,
    ``satellite_j2_perturbation``) implicitly assume when their docstrings
    say "Earth-centered inertial frame", without ever pinning down which
    real inertial frame that is or providing any way to convert to an
    Earth-fixed (ECEF) frame. ITRS (International Terrestrial Reference
    System) is the real Earth-Centered Earth-Fixed frame used for ground-
    station tracking, geolocation, and re-entry/impact-point predictions.

    This wrapper uses astropy's full IAU precession/nutation/Earth-rotation-
    angle/polar-motion model (via ``GCRS.transform_to(ITRS)``), not a
    simplified fixed-rotation-rate approximation -- i.e. a genuine,
    literature-grade ECI<->ECEF transform, cross-checkable against any
    other IAU-compliant astrodynamics tool.

    Parameters
    ----------
    x, y, z : Cartesian position components, in ``unit`` (default km).
    obstime : an ISO-8601 UTC time string (e.g. "2024-01-01T00:00:00"),
        or anything ``astropy.time.Time`` accepts. Required because the
        GCRS<->ITRS rotation is time-dependent (Earth's orientation).
    frame_in, frame_out : "gcrs" or "itrs".
    vx, vy, vz : optional Cartesian velocity components, in ``unit``/s.
    unit : unit string for the position (and, with "/s" appended,
        velocity) components; default "km".

    Returns
    -------
    dict with "x", "y", "z" (and "vx", "vy", "vz" if velocity was given),
    in the same ``unit`` convention as the input.

    Sanity check (see tests/test_astropy_bridge.py): a purely radial point
    along Earth's rotation axis (x=y=0) is invariant in *norm* under this
    transform (a rotation preserves vector length exactly), and a full
    GCRS -> ITRS -> GCRS round trip returns the original coordinates to
    numerical precision.
    """
    frame_in_l, frame_out_l = frame_in.lower(), frame_out.lower()
    if frame_in_l not in _FRAME_CLASSES or frame_out_l not in _FRAME_CLASSES:
        raise ValueError(
            f"frame_in/frame_out must be one of {sorted(_FRAME_CLASSES)}; "
            f"got frame_in={frame_in!r}, frame_out={frame_out!r}"
        )

    t = obstime if isinstance(obstime, Time) else Time(obstime, scale="utc")
    pos_unit = u.Unit(unit)

    rep: Any = CartesianRepresentation(x * pos_unit, y * pos_unit, z * pos_unit)
    has_velocity = vx is not None and vy is not None and vz is not None
    if has_velocity:
        vel_unit = pos_unit / u.s
        diff = CartesianDifferential(vx * vel_unit, vy * vel_unit, vz * vel_unit)
        rep = rep.with_differentials(diff)

    in_cls = _FRAME_CLASSES[frame_in_l]
    out_cls = _FRAME_CLASSES[frame_out_l]
    coord_in = in_cls(rep, obstime=t)
    coord_out = coord_in.transform_to(out_cls(obstime=t))

    out_rep = coord_out.cartesian
    result: Dict[str, float] = {
        "x": float(out_rep.x.to_value(pos_unit)),
        "y": float(out_rep.y.to_value(pos_unit)),
        "z": float(out_rep.z.to_value(pos_unit)),
    }
    if has_velocity:
        out_diff = out_rep.differentials["s"]
        vel_unit = pos_unit / u.s
        result["vx"] = float(out_diff.d_x.to_value(vel_unit))
        result["vy"] = float(out_diff.d_y.to_value(vel_unit))
        result["vz"] = float(out_diff.d_z.to_value(vel_unit))
    return result
