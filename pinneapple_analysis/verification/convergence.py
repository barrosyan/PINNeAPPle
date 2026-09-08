"""Grid-convergence / mesh-independence study automation.

Public API
----------
    richardson_extrapolate(
        f_coarse, f_medium, f_fine, r, *,
        p_assumed=None, safety_factor=1.25, asymptotic_tolerance=0.10,
    ) -> ConvergenceResult

    mesh_independence_study(
        solve_fn, resolutions, quantity_extractor, *,
        refinement_ratio=None, p_assumed=None, safety_factor=1.25,
        asymptotic_tolerance=0.10,
    ) -> MeshIndependenceStudyResult

This closes the gap flagged in the capability audit: PINNeAPPle has a real
mesh-*quality* module (``mesh_intelligence.py``, geometric aspect
ratio/skewness/y+ estimates from a single mesh) and real numerical solvers
with configurable resolution (``pinneapple_simulation/numerical_solvers/
fdm.py``, ``fem.py``, ``lbm.py``, etc.), but nothing that runs a solver at
several systematically-refined resolutions and asks the orthogonal
question "is the SOLUTION converging as resolution increases, and can we
estimate the discretization error and the resolution-independent 'exact'
answer from that trend?" -- i.e. mesh-*independence*, not mesh-*quality*.

This module is deliberately solver-agnostic: it never imports anything
from ``pinneapple_simulation`` and knows nothing about FDM/FEM/LBM
specifics. ``mesh_independence_study`` takes an arbitrary ``solve_fn(resolution)``
callable and an arbitrary ``quantity_extractor(solution)`` callable, so any
solver that can be re-run at different resolutions (or any external
tool wrapped in a Python function) can be plugged in unchanged.

Formula references
-------------------
Both the apparent (observed) order of accuracy and the Grid Convergence
Index (GCI) implemented here are the standard formulas from:

    Roache, P.J., "Perspective: A Method for Uniform Reporting of Grid
    Refinement Studies," ASME Journal of Fluids Engineering, 116(3),
    405-413, 1994.

(the same formulas were later re-popularized, unchanged in substance, by
Celik, I.B. et al., "Procedure for Estimation and Reporting of Uncertainty
Due to Discretization in CFD Applications," ASME J. Fluids Eng., 130(7),
2008 -- the notation below follows that paper's widely-used f1/f2/f3
(fine/medium/coarse) labeling since it is the more commonly cited form of
the same Roache method).

Let ``r`` be the constant refinement ratio between three systematically
refined grids (``r = h_coarse/h_medium = h_medium/h_fine > 1``, e.g.
``r=2`` for grid doubling), and ``f_coarse``, ``f_medium``, ``f_fine`` the
solution's quantity of interest computed on each grid.

1. **Apparent (observed) order of accuracy** ``p``, solved from the three
   solution values themselves (no a-priori assumption about the scheme's
   nominal order is required)::

       p = ln(|(f_coarse - f_medium) / (f_medium - f_fine)|) / ln(r)

   (this is the constant-refinement-ratio special case of Roache/Celik's
   general formula, which also handles non-constant ``r`` via an
   iteratively-solved correction term ``q(p)`` that vanishes identically
   when ``r`` is constant, as it always is here).

2. **Richardson-extrapolated ("exact", grid-independent) value**::

       f_extrapolated = (r^p * f_fine - f_medium) / (r^p - 1)

3. **Grid Convergence Index** (a standardized, physically-interpretable
   discretization-error UNCERTAINTY BAND -- not just a raw error
   estimate -- expressed as a percentage-like number), for both the
   fine-medium pair and the medium-coarse pair::

       GCI_fine   = Fs * |(f_fine - f_medium) / f_fine|     / (r^p - 1)
       GCI_coarse = Fs * |(f_medium - f_coarse) / f_medium| / (r^p - 1)

   with ``Fs`` a recommended safety factor -- **1.25** for studies using
   three or more grids (this module always requires >=3), vs. the more
   conservative **3.0** Roache originally recommended for two-grid
   studies only (not applicable here; see Roache 1994 and Celik et al.
   2008 Table 1).

4. **Asymptotic-range check**: in the true asymptotic convergence regime,
   ``GCI_coarse`` and ``GCI_fine`` are related by the refinement ratio
   raised to the observed order, i.e. ``GCI_coarse / (r^p * GCI_fine)``
   should be close to 1. This module reports that ratio directly
   (``asymptotic_ratio``) and flags ``is_asymptotic`` when it falls within
   ``asymptotic_tolerance`` of 1.0 (default 10%, a commonly-cited round
   engineering threshold -- not a universal constant, exactly like
   ``mesh_intelligence.py``'s own documented-but-configurable thresholds).

Honesty note, matching this repo's established ethic: a grid-convergence
study is only ever as good as the three solves it is built from. This
module never fabricates or smooths a value it wasn't given -- if the
three input values don't imply monotonic convergence (fine-medium and
medium-coarse differences have different signs -- oscillatory
convergence), it computes the formulas exactly as Roache/Celik define
them (using absolute values, as they do) but raises a ``UserWarning`` so
the caller knows the asymptotic assumption behind ``p``/GCI may not hold.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np

QuantityArray = Union[float, np.ndarray]

__all__ = [
    "ConvergenceResult",
    "MeshIndependenceStudyResult",
    "richardson_extrapolate",
    "mesh_independence_study",
]

# Roache (1994) / Celik et al. (2008) recommended safety factor for grid
# convergence studies using three or more grids (this module always uses
# exactly three -- the finest three resolutions given). The alternative,
# more conservative Fs=3.0 is specifically for two-grid studies, which
# this module does not perform.
_DEFAULT_SAFETY_FACTOR = 1.25

# A commonly-cited, round engineering threshold for judging
# GCI_coarse/(r^p*GCI_fine) "close enough" to 1 to call the study
# asymptotic -- not a universal constant, deliberately configurable.
_DEFAULT_ASYMPTOTIC_TOLERANCE = 0.10


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ConvergenceResult:
    """The result of a three-grid Richardson extrapolation / GCI analysis.
    See the module docstring's "Formula references" section for the exact
    formulas and citation (Roache, 1994)."""

    observed_order: float
    extrapolated_value: QuantityArray
    gci_fine: float
    gci_coarse: float
    asymptotic_ratio: float
    is_asymptotic: bool
    refinement_ratio: float = 0.0
    safety_factor: float = _DEFAULT_SAFETY_FACTOR
    asymptotic_tolerance: float = _DEFAULT_ASYMPTOTIC_TOLERANCE
    citation: str = (
        "Roache, P.J., 'Perspective: A Method for Uniform Reporting of Grid Refinement "
        "Studies', ASME J. Fluids Eng., 116(3), 405-413, 1994."
    )


@dataclass
class MeshIndependenceStudyResult:
    """A full mesh-independence study: the raw per-resolution values (for
    plotting a convergence curve) plus the three-finest-grid
    :class:`ConvergenceResult`."""

    resolutions: List[Any]
    values: List[QuantityArray]
    refinement_ratios: List[float]
    refinement_ratio_used: float
    convergence: ConvergenceResult


# ---------------------------------------------------------------------------
# Richardson extrapolation / GCI
# ---------------------------------------------------------------------------

def _flat_norm(x: np.ndarray) -> float:
    """L2 norm after flattening -- used to reduce an array-valued quantity
    of interest (e.g. a full field) to the single scalar scale that the
    order/GCI formulas (stated by Roache/Celik for a single scalar QoI)
    need. For a genuinely scalar ``x`` this is just ``abs(x)``."""
    arr = np.asarray(x, dtype=np.float64).ravel()
    return float(np.linalg.norm(arr))


def richardson_extrapolate(
    f_coarse: QuantityArray,
    f_medium: QuantityArray,
    f_fine: QuantityArray,
    r: float,
    *,
    p_assumed: Optional[float] = None,
    safety_factor: float = _DEFAULT_SAFETY_FACTOR,
    asymptotic_tolerance: float = _DEFAULT_ASYMPTOTIC_TOLERANCE,
) -> ConvergenceResult:
    """Richardson-extrapolate a quantity of interest computed at three
    systematically-refined resolutions and compute Roache's (1994) Grid
    Convergence Index. See the module docstring for the exact formulas.

    Parameters
    ----------
    f_coarse, f_medium, f_fine : the quantity of interest evaluated on the
        coarse, medium, and fine grids respectively. Each may be a plain
        Python/numpy scalar or an array (e.g. a full solution field) of
        identical shape -- if arrays, the observed order and GCI are
        computed from the L2 norm of the pairwise differences (see
        :func:`_flat_norm`), while ``extrapolated_value`` is computed
        elementwise and keeps the input's shape.
    r : the constant refinement ratio between successive grids
        (``h_coarse/h_medium == h_medium/h_fine == r``), e.g. ``r=2`` for
        grid doubling. Must be > 1.
    p_assumed : if given, skip solving for the observed order from the
        three solution values and use this value directly for the
        extrapolation and GCI formulas instead (useful when the scheme's
        nominal order is already known/trusted, or to sanity-check a
        theoretical order against the data-derived one separately).
    safety_factor : Roache/Celik's ``Fs`` -- 1.25 by default (recommended
        for studies with >=3 grids; see module docstring).
    asymptotic_tolerance : relative tolerance (default 0.10 = 10%) used to
        decide ``is_asymptotic`` from ``asymptotic_ratio``.

    Raises
    ------
    ValueError
        if ``r <= 1``, or if ``f_medium`` equals ``f_fine`` (or
        ``f_coarse`` equals ``f_medium``) exactly, which would make the
        observed-order formula divide by zero -- rather than silently
        returning ``inf``/``nan``, this is reported as an explicit error
        so the caller knows the three solves did not actually differ
        enough to support a convergence-order estimate.
    """
    if not (r > 1.0):
        raise ValueError(f"refinement ratio r must be > 1 (constant refinement, coarse->fine); got r={r}")

    f_c = np.asarray(f_coarse, dtype=np.float64)
    f_m = np.asarray(f_medium, dtype=np.float64)
    f_f = np.asarray(f_fine, dtype=np.float64)
    if f_c.shape != f_m.shape or f_m.shape != f_f.shape:
        raise ValueError(
            f"f_coarse, f_medium, f_fine must have identical shapes; got {f_c.shape}, {f_m.shape}, {f_f.shape}"
        )
    is_scalar = f_c.ndim == 0

    diff_32 = f_c - f_m  # coarse - medium
    diff_21 = f_m - f_f  # medium - fine
    norm_32 = _flat_norm(diff_32)
    norm_21 = _flat_norm(diff_21)

    if norm_21 == 0.0:
        raise ValueError(
            "f_medium and f_fine are identical -- cannot estimate an observed order of accuracy "
            "or GCI from two solves that produced the same value (either the quantity of interest "
            "is insensitive to this refinement, or the two finest grids are not actually distinct)."
        )
    if norm_32 == 0.0:
        raise ValueError(
            "f_coarse and f_medium are identical -- cannot estimate an observed order of accuracy "
            "or GCI from two solves that produced the same value."
        )

    if is_scalar and np.sign(float(diff_32)) != np.sign(float(diff_21)):
        warnings.warn(
            "richardson_extrapolate: f_coarse-f_medium and f_medium-f_fine have opposite signs "
            "(oscillatory, non-monotonic convergence) -- the observed order and GCI are still "
            "computed using |.| per Roache (1994)/Celik et al. (2008), but the underlying "
            "asymptotic-convergence assumption they rely on may not hold here.",
            UserWarning,
            stacklevel=2,
        )

    if p_assumed is None:
        p = float(np.log(norm_32 / norm_21) / np.log(r))
    else:
        p = float(p_assumed)

    r_p = r ** p
    if np.isclose(r_p, 1.0):
        raise ValueError(
            f"r^observed_order is numerically 1.0 (r={r}, p={p}) -- the extrapolation/GCI formulas "
            "divide by (r^p - 1) and would be singular here; check that r and the solution values "
            "are correct."
        )

    extrapolated_value = (r_p * f_f - f_m) / (r_p - 1.0)
    extrapolated_out: QuantityArray = float(extrapolated_value) if is_scalar else extrapolated_value

    norm_f_fine = _flat_norm(f_f)
    norm_f_medium = _flat_norm(f_m)
    if norm_f_fine == 0.0 or norm_f_medium == 0.0:
        raise ValueError(
            "cannot form a relative error for the GCI formulas: f_fine and/or f_medium have zero "
            "norm (Roache's GCI is defined relative to the solution value itself)."
        )

    e_fine = abs(norm_21 / norm_f_fine)     # |(f_fine - f_medium) / f_fine|, via norms
    e_coarse = abs(norm_32 / norm_f_medium)  # |(f_medium - f_coarse) / f_medium|, via norms

    gci_fine = safety_factor * e_fine / (r_p - 1.0)
    gci_coarse = safety_factor * e_coarse / (r_p - 1.0)

    asymptotic_ratio = float(gci_coarse / (r_p * gci_fine))
    is_asymptotic = bool(abs(asymptotic_ratio - 1.0) <= asymptotic_tolerance)

    return ConvergenceResult(
        observed_order=p,
        extrapolated_value=extrapolated_out,
        gci_fine=float(gci_fine),
        gci_coarse=float(gci_coarse),
        asymptotic_ratio=asymptotic_ratio,
        is_asymptotic=is_asymptotic,
        refinement_ratio=float(r),
        safety_factor=float(safety_factor),
        asymptotic_tolerance=float(asymptotic_tolerance),
    )


# ---------------------------------------------------------------------------
# Mesh-independence study driver (solver-agnostic)
# ---------------------------------------------------------------------------

def mesh_independence_study(
    solve_fn: Callable[[Any], Any],
    resolutions: Sequence[Any],
    quantity_extractor: Callable[[Any], QuantityArray],
    *,
    refinement_ratio: Optional[float] = None,
    p_assumed: Optional[float] = None,
    safety_factor: float = _DEFAULT_SAFETY_FACTOR,
    asymptotic_tolerance: float = _DEFAULT_ASYMPTOTIC_TOLERANCE,
) -> MeshIndependenceStudyResult:
    """Run a mesh-independence / grid-convergence study by calling an
    arbitrary solver at increasing resolutions and Richardson-extrapolating
    the three finest results.

    This is deliberately solver-agnostic: it knows nothing about FDM, FEM,
    LBM, or any other discretization scheme. ``solve_fn`` and
    ``quantity_extractor`` are the only two hooks -- any solver in
    ``pinneapple_simulation.numerical_solvers`` (or any external tool
    wrapped in a Python callable) can be plugged in without this module
    changing.

    Parameters
    ----------
    solve_fn : called as ``solve_fn(resolution)`` for each entry of
        ``resolutions``, in order; must return whatever solution object
        ``quantity_extractor`` expects (a full field, a dict, a custom
        result type -- entirely up to the caller's solver).
    resolutions : an ASCENDING sequence of resolution values (e.g. grid
        point counts, cell counts, or any resolution knob the caller's
        solver accepts) representing systematically-refined grids, finest
        last. At least 3 values are required.
    quantity_extractor : called as ``quantity_extractor(solution)`` on
        each of ``solve_fn``'s return values to pull out the scalar or
        array quantity of interest (e.g. a probe value, a max/L2 norm of
        the field, a drag coefficient -- whatever the study is tracking).
    refinement_ratio : the constant ratio between successive resolutions'
        effective grid spacing. If not given, it is inferred from
        ``resolutions`` themselves as
        ``resolutions[i+1] / resolutions[i]`` (appropriate when
        ``resolutions`` are counts -- points/cells per some fixed domain
        length, so spacing h ~ 1/resolution and this ratio already equals
        h_coarser/h_finer); a ``UserWarning`` is raised if the inferred
        ratios are not roughly constant across the sequence (a genuine
        grid-convergence study requires a *systematic*, constant-ratio
        refinement -- Roache's formulas assume it).
    p_assumed, safety_factor, asymptotic_tolerance : forwarded to
        :func:`richardson_extrapolate`.

    Returns
    -------
    MeshIndependenceStudyResult
        The raw per-resolution quantities (for plotting a convergence
        curve) plus the :class:`ConvergenceResult` from the three finest
        resolutions.

    Raises
    ------
    ValueError
        if fewer than 3 resolutions are given, or ``resolutions`` is not
        strictly ascending.
    """
    resolutions = list(resolutions)
    if len(resolutions) < 3:
        raise ValueError(
            f"mesh_independence_study requires at least 3 resolutions (got {len(resolutions)}) -- "
            "a grid-convergence study needs three systematically-refined solves to estimate an "
            "observed order of accuracy and GCI (Roache, 1994)."
        )
    numeric_resolutions = [float(r) for r in resolutions]
    if any(b <= a for a, b in zip(numeric_resolutions, numeric_resolutions[1:])):
        raise ValueError(
            f"resolutions must be strictly ascending (coarsest first, finest last); got {resolutions}"
        )

    values: List[QuantityArray] = []
    for res in resolutions:
        solution = solve_fn(res)
        qty = quantity_extractor(solution)
        qty_arr = np.asarray(qty, dtype=np.float64)
        values.append(float(qty_arr) if qty_arr.ndim == 0 else qty_arr)

    consecutive_ratios = [
        b / a for a, b in zip(numeric_resolutions, numeric_resolutions[1:])
    ]

    if refinement_ratio is None:
        r_used = float(consecutive_ratios[-1])  # ratio between the two finest resolutions -- what the extrapolation actually uses
        if len(consecutive_ratios) > 1:
            ratios_arr = np.asarray(consecutive_ratios, dtype=np.float64)
            if not np.allclose(ratios_arr, ratios_arr[0], rtol=0.05, atol=1e-9):
                warnings.warn(
                    f"mesh_independence_study: inferred refinement ratios between consecutive "
                    f"resolutions are not roughly constant ({consecutive_ratios}) -- Roache's (1994) "
                    "Richardson extrapolation/GCI formulas assume a SYSTEMATIC, constant-ratio "
                    f"refinement; using the ratio between the two finest resolutions ({r_used:.4g}) "
                    "for the extrapolation. Consider using resolutions with a constant refinement ratio.",
                    UserWarning,
                    stacklevel=2,
                )
    else:
        r_used = float(refinement_ratio)

    convergence = richardson_extrapolate(
        values[-3], values[-2], values[-1], r_used,
        p_assumed=p_assumed, safety_factor=safety_factor, asymptotic_tolerance=asymptotic_tolerance,
    )

    return MeshIndependenceStudyResult(
        resolutions=resolutions,
        values=values,
        refinement_ratios=consecutive_ratios,
        refinement_ratio_used=r_used,
        convergence=convergence,
    )
