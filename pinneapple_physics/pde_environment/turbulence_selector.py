"""Unified turbulence-model selector.

Real turbulence closures already exist in this repo in three disjoint
places:

  * ``pinneapple_physics.pde_environment.turbulence_presets`` --
    ``KOmegaSSTResiduals`` (RANS, PINN-residual style, dispatches on
    ``x_col.shape[1]`` to give a 2D or 3D closure at *call* time) and
    ``WALEResiduals`` (LES, PINN-residual style, genuinely 3D only).
  * ``pinneapple_simulation.numerical_solvers.lbm`` -- Smagorinsky LES via
    the scalar ``Cs`` parameter accepted by ``LBMSolver``/``LBMSolver3D``
    (LBM-native: not a PINN residual object at all, just a number).

This module is pure dispatch/validation glue over those two families -- it
constructs and returns the real objects (or the real ``Cs`` parameter) and
does not reimplement any turbulence physics itself.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union

from pinneapple_physics.pde_environment.turbulence_presets import (
    KOmegaSSTResiduals,
    WALEResiduals,
)


class TurbulenceModel(str, Enum):
    """Turbulence closure selector.

    ``LAMINAR``          -- no turbulence closure.
    ``RANS_K_OMEGA_SST``  -- Menter (1994) k-omega SST, PINN-residual style
                             (:class:`KOmegaSSTResiduals`).
    ``LES_WALE``          -- Nicoud & Ducros (1999) WALE, PINN-residual
                             style, genuinely 3D only (:class:`WALEResiduals`).
    ``LES_SMAGORINSKY``   -- Smagorinsky LES, LBM-native (a scalar ``Cs``
                             constant, not a PINN residual object).
    """
    LAMINAR = "laminar"
    RANS_K_OMEGA_SST = "rans_k_omega_sst"
    LES_WALE = "les_wale"
    LES_SMAGORINSKY = "les_smagorinsky"


_DEFAULT_SMAGORINSKY_CS = 0.17  # Hou (1994) LES-LBM typical range 0.1-0.18


def get_turbulence_closure(
    model: Union[TurbulenceModel, str],
    dim: int = 3,
    solver_family: str = "pinn",
    **kwargs: Any,
) -> Optional[Any]:
    """Return the right turbulence closure object/parameter for a solver family.

    This function only dispatches to and validates against the *existing*
    closures -- it never computes turbulence physics itself.

    Parameters
    ----------
    model : TurbulenceModel | str
        One of ``TurbulenceModel``'s members (or the equivalent string
        value, e.g. ``"les_wale"``).
    dim : int
        Spatial dimension, 2 or 3. For ``solver_family="pinn"``:

        * ``RANS_K_OMEGA_SST`` -- :class:`KOmegaSSTResiduals` itself
          dispatches 2D vs. 3D at *call* time from the collocation points'
          shape (``x_col.shape[1]``), not at construction time, so ``dim``
          is only used here to validate it is 2 or 3 (it is not forwarded
          to the constructor, which has no ``dim`` parameter).
        * ``LES_WALE`` -- :class:`WALEResiduals` is a genuinely-3D-only
          closure (see its docstring); ``dim=2`` raises ``ValueError``.

        Ignored for ``solver_family="lbm"`` (the in-repo LBM solvers'
        Smagorinsky support is dimension-agnostic: the same scalar ``Cs``
        is accepted by both ``LBMSolver`` (D2Q9) and ``LBMSolver3D`` (D3Q19)).
    solver_family : {"pinn", "lbm"}
        Which closure family to dispatch into.
    **kwargs
        Forwarded to the underlying closure's constructor
        (``solver_family="pinn"``) or used to look up ``Cs``
        (``solver_family="lbm"``, ``model=LES_SMAGORINSKY``).

    Returns
    -------
    * ``solver_family="pinn"``: ``None`` (LAMINAR), a configured
      :class:`KOmegaSSTResiduals` (RANS_K_OMEGA_SST), or a configured
      :class:`WALEResiduals` (LES_WALE).
    * ``solver_family="lbm"``: ``0.0`` (LAMINAR, i.e. ``Cs=0`` / pure BGK)
      or a ``float`` Smagorinsky constant ``Cs`` (LES_SMAGORINSKY).

    Raises
    ------
    ValueError
        For any model/solver_family combination that has no implementation
        in this repo (e.g. LES_SMAGORINSKY under "pinn", or
        RANS_K_OMEGA_SST/LES_WALE under "lbm"), or for an unrecognized
        ``model``/``solver_family``/``dim``.
    """
    model = TurbulenceModel(model)

    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3; got {dim}")

    if solver_family == "pinn":
        return _pinn_closure(model, dim, **kwargs)
    if solver_family == "lbm":
        return _lbm_closure(model, dim, **kwargs)

    raise ValueError(
        f"Unknown solver_family {solver_family!r}; expected 'pinn' or 'lbm'"
    )


def _pinn_closure(model: TurbulenceModel, dim: int, **kwargs: Any) -> Optional[Any]:
    if model is TurbulenceModel.LAMINAR:
        return None

    if model is TurbulenceModel.RANS_K_OMEGA_SST:
        # KOmegaSSTResiduals has no `dim` constructor arg -- it dispatches
        # 2D vs. 3D at __call__ time from the collocation points passed in
        # (x_col.shape[1] == 2 or 3). `dim` was already validated above.
        return KOmegaSSTResiduals(**kwargs)

    if model is TurbulenceModel.LES_WALE:
        if dim != 3:
            raise ValueError(
                "LES_WALE (WALEResiduals) is a genuinely-3D-only LES closure "
                "(see its docstring in "
                "pinneapple_physics.pde_environment.turbulence_presets); "
                f"got dim={dim}. Use RANS_K_OMEGA_SST for a 2D PINN closure."
            )
        return WALEResiduals(**kwargs)

    if model is TurbulenceModel.LES_SMAGORINSKY:
        raise ValueError(
            "LES_SMAGORINSKY (Smagorinsky LES via the 'Cs' constant) is "
            "LBM-native -- it is a parameter of LBMSolver/LBMSolver3D "
            "(pinneapple_simulation.numerical_solvers.lbm), not a PINN "
            "residual closure. Pass solver_family='lbm' instead, or use "
            "LES_WALE for a PINN-residual LES closure."
        )

    raise ValueError(f"Unhandled TurbulenceModel: {model!r}")  # pragma: no cover


def _lbm_closure(model: TurbulenceModel, dim: int, **kwargs: Any) -> float:
    if model is TurbulenceModel.LAMINAR:
        return 0.0  # Cs=0 => pure BGK, no LES (see LBMSolver's Cs docstring)

    if model is TurbulenceModel.LES_SMAGORINSKY:
        return float(kwargs.get("Cs", _DEFAULT_SMAGORINSKY_CS))

    if model in (TurbulenceModel.RANS_K_OMEGA_SST, TurbulenceModel.LES_WALE):
        raise ValueError(
            f"{model.value} ({'KOmegaSSTResiduals' if model is TurbulenceModel.RANS_K_OMEGA_SST else 'WALEResiduals'}) "
            "is a PINN-residual closure "
            "(pinneapple_physics.pde_environment.turbulence_presets) -- it "
            "is not implemented for this repo's LBM solvers "
            "(pinneapple_simulation.numerical_solvers.lbm), which only "
            "support Smagorinsky LES via the scalar 'Cs' constant. Pass "
            "solver_family='pinn' instead, or use LES_SMAGORINSKY here."
        )

    raise ValueError(f"Unhandled TurbulenceModel: {model!r}")  # pragma: no cover


__all__ = ["TurbulenceModel", "get_turbulence_closure"]
