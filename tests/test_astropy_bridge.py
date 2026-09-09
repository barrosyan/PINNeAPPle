"""Validation of `pinneapple_physics.pde_environment.presets.astropy_bridge`
-- a new, additive astropy-backed unit-safety and coordinate-transform
utility module (does not modify/replace anything in `astrophysics.py`).

Checks performed:

1. `ASTRO_CONSTANTS` values are exactly (not approximately) the same
   objects/values as `astropy.constants` itself -- i.e. genuinely sourced
   live from astropy, not copy-pasted bare floats that happen to match.
2. `with_units`/`to_si` correctly parse and round-trip real unit strings
   ("10 AU", "5 Msun", ...) to the correct SI float, cross-checked against
   the value computed directly from `astropy.constants`/`astropy.units`
   independently in this file.
3. `to_si` propagates astropy's own real `astropy.units.UnitConversionError`
   (not a generic/custom exception, not swallowed) when asked to convert
   between incompatible dimensions (mass -> length).
4. `transform_coordinates`'s GCRS<->ITRS transform produces a real, sane
   result: cross-checked two ways that don't rely on the implementation
   itself --
     (a) an independent, from-scratch call to `astropy.coordinates`
         (GCRS/ITRS/CartesianRepresentation) reproduces the same numbers
         (this is what the function wraps, so this checks the wrapper
         doesn't silently mangle anything -- units, argument order, frame
         direction);
     (b) two frame-independent physical invariants of any pure rotation:
         vector norm is preserved, and a round trip (frame A -> B -> A)
         returns the original coordinates to numerical precision.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

astropy = pytest.importorskip("astropy")

from astropy import constants as const  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.coordinates import GCRS, ITRS, CartesianRepresentation  # noqa: E402
from astropy.time import Time  # noqa: E402

from pinneapple_physics.pde_environment.presets.astropy_bridge import (  # noqa: E402
    ASTRO_CONSTANTS,
    astro_constant_si,
    constant_provenance,
    to_si,
    transform_coordinates,
    with_units,
)


# ===========================================================================
# ASTRO_CONSTANTS
# ===========================================================================

@pytest.mark.parametrize(
    "name,astropy_attr",
    [
        ("G", "G"),
        ("c", "c"),
        ("M_sun", "M_sun"),
        ("R_sun", "R_sun"),
        ("GM_sun", "GM_sun"),
        ("M_earth", "M_earth"),
        ("R_earth", "R_earth"),
        ("GM_earth", "GM_earth"),
        ("au", "au"),
        ("pc", "pc"),
        ("sigma_sb", "sigma_sb"),
        ("m_p", "m_p"),
        ("sigma_T", "sigma_T"),
    ],
)
def test_astro_constants_match_astropy_exactly(name, astropy_attr):
    ours = ASTRO_CONSTANTS[name]
    real = getattr(const, astropy_attr)
    # Exact equality, not np.isclose: these must BE astropy's own values.
    assert float(ours.si.value) == float(real.si.value)
    assert ours.unit == real.unit
    # Same reference/provenance string too.
    assert str(ours.reference) == str(real.reference)


def test_astro_constants_are_live_astropy_objects_not_bare_floats():
    # Constant objects carry real provenance metadata a bare float cannot.
    g = ASTRO_CONSTANTS["G"]
    assert hasattr(g, "reference")
    assert hasattr(g, "uncertainty")
    assert isinstance(g, u.Quantity)  # astropy.constants.Constant is-a Quantity
    assert "CODATA" in str(g.reference)


def test_astro_constant_si_helper():
    assert astro_constant_si("G") == pytest.approx(const.G.si.value, rel=0, abs=0) or \
        astro_constant_si("G") == const.G.si.value
    assert astro_constant_si("c") == 299792458.0  # SI-exact, no tolerance needed
    with pytest.raises(KeyError):
        astro_constant_si("not_a_real_constant")


def test_constant_provenance_helper():
    ref = constant_provenance("M_sun")
    assert isinstance(ref, str) and len(ref) > 0
    assert ref == str(const.M_sun.reference)
    with pytest.raises(KeyError):
        constant_provenance("nope")


def test_hardcoded_astrophysics_G_differs_from_real_astropy_value():
    """Documents (does not fix -- astrophysics.py is out of scope) the
    ~4.5e-5 relative discrepancy between `shakura_sunyaev_accretion_disk`'s
    hard-coded G=6.674e-11 and astropy's real CODATA-2022 G."""
    hardcoded_G = 6.674e-11
    real_G = astro_constant_si("G")
    rel_diff = abs(hardcoded_G - real_G) / real_G
    assert rel_diff > 1e-6  # confirms a genuine (if small) discrepancy exists
    assert rel_diff < 1e-3  # ...but it's a rounding-vintage issue, not a gross error


def test_hardcoded_astrophysics_M_sun_differs_from_real_astropy_value():
    """Documents the ~2.6e-4 relative discrepancy between
    `shakura_sunyaev_accretion_disk`'s hard-coded M_sun=1.98892e30 kg and
    astropy's real (IAU 2015 nominal parameter / CODATA 2022 G) M_sun."""
    hardcoded_M_sun = 1.98892e30
    real_M_sun = astro_constant_si("M_sun")
    rel_diff = abs(hardcoded_M_sun - real_M_sun) / real_M_sun
    assert rel_diff > 1e-5
    assert rel_diff < 1e-3


# ===========================================================================
# with_units / to_si
# ===========================================================================

def test_with_units_two_arg_form():
    q = with_units(10.0, "AU")
    assert isinstance(q, u.Quantity)
    assert q.unit == u.AU
    assert q.value == 10.0


def test_with_units_single_string_form():
    q = with_units("5 Msun")
    assert q.unit.physical_type == "mass"
    assert q.to_value(u.kg) == pytest.approx(5.0 * const.M_sun.si.value, rel=1e-12)


def test_with_units_rejects_bare_number_without_unit():
    with pytest.raises(TypeError):
        with_units(10.0)


def test_to_si_default_si_units():
    assert to_si("10 AU") == pytest.approx(10.0 * const.au.si.value, rel=1e-12)
    assert to_si("5 Msun") == pytest.approx(5.0 * const.M_sun.si.value, rel=1e-12)


def test_to_si_explicit_target_unit():
    # 1 AU in km, cross-checked independently via astropy.units directly.
    expected_km = (1.0 * u.AU).to_value(u.km)
    assert to_si("1 AU", target_unit="km") == pytest.approx(expected_km, rel=1e-12)


def test_to_si_accepts_quantity_object():
    q = with_units(2.0, "yr")
    expected_s = (2.0 * u.yr).to_value(u.s)
    assert to_si(q) == pytest.approx(expected_s, rel=1e-12)


def test_to_si_round_trip_with_units():
    original = 42.0
    q = with_units(original, "km")
    back = to_si(q, target_unit="km")
    assert back == pytest.approx(original, rel=1e-12)


def test_to_si_rejects_bare_number():
    with pytest.raises(TypeError):
        to_si(5.0)


def test_to_si_raises_real_unit_conversion_error_on_mismatch():
    """A raw-float codebase has no way to catch this class of bug --
    converting a mass to a length -- at all. This must be astropy's own
    real UnitConversionError, not a generic ValueError or a swallowed
    exception."""
    with pytest.raises(u.UnitConversionError):
        to_si("5 kg", target_unit="m")


def test_to_si_raises_real_unit_conversion_error_time_vs_length():
    with pytest.raises(u.UnitConversionError):
        to_si("3 s", target_unit="AU")


# ===========================================================================
# transform_coordinates (GCRS <-> ITRS)
# ===========================================================================

OBSTIME = "2024-01-01T00:00:00"


def test_transform_matches_independent_direct_astropy_call():
    """Cross-check (a): reproduce the wrapper's output with a from-scratch,
    independent call to astropy.coordinates (not going through
    transform_coordinates at all)."""
    x, y, z = 1000.0, 2000.0, 6800.0  # km, arbitrary LEO-ish position
    t = Time(OBSTIME, scale="utc")
    rep = CartesianRepresentation(x * u.km, y * u.km, z * u.km)
    expected = GCRS(rep, obstime=t).transform_to(ITRS(obstime=t)).cartesian

    got = transform_coordinates(x, y, z, obstime=OBSTIME, frame_in="gcrs", frame_out="itrs")

    assert got["x"] == pytest.approx(expected.x.to_value(u.km), rel=1e-10)
    assert got["y"] == pytest.approx(expected.y.to_value(u.km), rel=1e-10)
    assert got["z"] == pytest.approx(expected.z.to_value(u.km), rel=1e-10)


def test_transform_preserves_vector_norm():
    """Cross-check (b), invariant 1: GCRS<->ITRS is a pure rotation (to the
    precision of astropy's IAU Earth-orientation model), so the norm of the
    position vector must be preserved exactly (up to floating point)."""
    x, y, z = 4000.0, -3000.0, 5000.0
    r0 = math.sqrt(x ** 2 + y ** 2 + z ** 2)
    got = transform_coordinates(x, y, z, obstime=OBSTIME, frame_in="gcrs", frame_out="itrs")
    r1 = math.sqrt(got["x"] ** 2 + got["y"] ** 2 + got["z"] ** 2)
    assert r1 == pytest.approx(r0, rel=1e-9)


def test_transform_round_trip_returns_original():
    """Cross-check (b), invariant 2: GCRS -> ITRS -> GCRS must recover the
    original coordinates to numerical precision."""
    x, y, z = 7000.0, 100.0, -200.0
    step1 = transform_coordinates(x, y, z, obstime=OBSTIME, frame_in="gcrs", frame_out="itrs")
    step2 = transform_coordinates(
        step1["x"], step1["y"], step1["z"], obstime=OBSTIME, frame_in="itrs", frame_out="gcrs"
    )
    assert step2["x"] == pytest.approx(x, abs=1e-6)
    assert step2["y"] == pytest.approx(y, abs=1e-6)
    assert step2["z"] == pytest.approx(z, abs=1e-6)


def test_transform_polar_axis_point_stays_near_z_axis():
    """A well-documented sanity case: a point along Earth's rotation axis
    (x=y=0 in an Earth-centered inertial frame) should map to a point very
    close to the ITRS z-axis (small x/y offsets only from precession,
    nutation and polar motion -- these are genuinely small, order tens of
    km at most, not order-unity)."""
    got = transform_coordinates(0.0, 0.0, 7000.0, obstime=OBSTIME, frame_in="gcrs", frame_out="itrs")
    assert abs(got["x"]) < 50.0  # km
    assert abs(got["y"]) < 50.0  # km
    assert got["z"] == pytest.approx(7000.0, abs=1.0)


def test_transform_with_velocity():
    x, y, z = 7000.0, 0.0, 0.0
    vx, vy, vz = 0.0, 7.5, 0.0  # km/s, roughly LEO circular speed
    got = transform_coordinates(
        x, y, z, obstime=OBSTIME, frame_in="gcrs", frame_out="itrs", vx=vx, vy=vy, vz=vz
    )
    assert set(("vx", "vy", "vz")).issubset(got.keys())
    v_mag = math.sqrt(got["vx"] ** 2 + got["vy"] ** 2 + got["vz"] ** 2)
    # ITRS velocity picks up Earth's rotation (~0.4-0.5 km/s at this
    # latitude/radius); should stay in a physically sane ballpark, not
    # blow up or vanish.
    assert 5.0 < v_mag < 10.0


def test_transform_rejects_unknown_frame():
    with pytest.raises(ValueError):
        transform_coordinates(0.0, 0.0, 7000.0, obstime=OBSTIME, frame_in="eci", frame_out="itrs")
