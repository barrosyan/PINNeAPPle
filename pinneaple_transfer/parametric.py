"""ParametricFamilyTransfer — manage a collection of PINNs fine-tuned for
different values of a physical parameter (e.g. viscosity ν, Reynolds number
Re, wavenumber k).

Key capabilities
----------------
* Store and retrieve fine-tuned model variants indexed by a scalar parameter.
* Retrieve the closest pre-trained variant for any query value.
* **Linear weight interpolation** between the two nearest variants to obtain a
  smooth model family without retraining.
* Persist and reload the entire family from disk via a single
  ``torch.save`` / ``torch.load`` round-trip.

Example
-------
>>> family = ParametricFamilyTransfer(base_model, param_name="nu")
>>> family.add_variant(0.01, model_nu_001)
>>> family.add_variant(0.05, model_nu_005)
>>> model_nu_003 = family.interpolate_weights(0.03)   # between 0.01 and 0.05
>>> family.save("burgers_family.pt")
>>> loaded = ParametricFamilyTransfer.load("burgers_family.pt")
"""
from __future__ import annotations

import copy
from typing import ClassVar, Dict, List, Optional

import torch
import torch.nn as nn


class ParametricFamilyTransfer:
    """Collection of PINNs fine-tuned for different scalar parameter values.

    Parameters
    ----------
    base_model:
        The pre-trained reference model (trained at the reference parameter
        value).  It is deep-copied on construction.
    param_name:
        Human-readable name of the varying physical parameter, e.g.
        ``"nu"``, ``"Re"``, ``"k"``.  Used only for labelling/serialisation.
    """

    def __init__(self, base_model: nn.Module, param_name: str) -> None:
        self.param_name = param_name
        self.base_model: nn.Module = copy.deepcopy(base_model)
        # Ordered mapping: param_value -> fine-tuned model (deep-copied on add)
        self._variants: Dict[float, nn.Module] = {}

    # ------------------------------------------------------------------
    # Variant management
    # ------------------------------------------------------------------

    def add_variant(self, param_value: float, fine_tuned_model: nn.Module) -> None:
        """Register a fine-tuned model for *param_value*.

        The model is deep-copied so subsequent mutations by the caller do not
        affect the stored variant.

        Parameters
        ----------
        param_value:
            The scalar parameter value this model was fine-tuned for.
        fine_tuned_model:
            A trained ``nn.Module`` with the same architecture as
            :attr:`base_model`.
        """
        self._variants[float(param_value)] = copy.deepcopy(fine_tuned_model)

    def get_model(self, param_value: float) -> nn.Module:
        """Return the stored variant closest to *param_value*.

        If an exact match exists it is returned directly (no copy).  Otherwise
        the nearest registered value is used.

        Parameters
        ----------
        param_value:
            Query parameter value.

        Returns
        -------
        nn.Module
            The closest stored variant (not a copy — do not mutate in-place
            unless you intend to modify the registry).

        Raises
        ------
        RuntimeError
            If no variants have been registered.
        """
        if not self._variants:
            raise RuntimeError(
                "No variants registered.  Call add_variant() first."
            )
        key = float(param_value)
        if key in self._variants:
            return self._variants[key]
        closest = min(self._variants.keys(), key=lambda v: abs(v - key))
        return self._variants[closest]

    def interpolate_weights(self, param_value: float) -> nn.Module:
        """Linearly interpolate weights between the two nearest variants.

        Given a query value ``q`` and the two nearest registered values
        ``v_lo <= q <= v_hi``, the returned model's state dict is::

            theta(q) = alpha * theta(v_lo) + (1 - alpha) * theta(v_hi)

        where ``alpha = (v_hi - q) / (v_hi - v_lo)``.

        If fewer than two variants are registered, or the query is outside the
        registered range, the closest single variant is returned (no
        extrapolation).

        Parameters
        ----------
        param_value:
            Query parameter value for which to construct an interpolated model.

        Returns
        -------
        nn.Module
            A new model instance (deep-copied from :attr:`base_model`) whose
            weights are the linear interpolation.  The caller owns this object.

        Raises
        ------
        RuntimeError
            If no variants have been registered.
        """
        if not self._variants:
            raise RuntimeError(
                "No variants registered.  Call add_variant() first."
            )

        sorted_vals = sorted(self._variants.keys())
        q = float(param_value)

        # Edge cases: only one variant, or query outside range.
        if len(sorted_vals) == 1 or q <= sorted_vals[0]:
            return copy.deepcopy(self._variants[sorted_vals[0]])
        if q >= sorted_vals[-1]:
            return copy.deepcopy(self._variants[sorted_vals[-1]])

        # Find bracketing pair.
        v_lo: Optional[float] = None
        v_hi: Optional[float] = None
        for v in sorted_vals:
            if v <= q:
                v_lo = v
            else:
                v_hi = v
                break

        assert v_lo is not None and v_hi is not None  # guaranteed by above logic

        alpha = (v_hi - q) / (v_hi - v_lo)  # weight for v_lo

        sd_lo = self._variants[v_lo].state_dict()
        sd_hi = self._variants[v_hi].state_dict()

        # Build interpolated state dict.
        interp_sd = {}
        for key in sd_lo:
            t_lo = sd_lo[key]
            t_hi = sd_hi[key]
            if t_lo.is_floating_point():
                interp_sd[key] = alpha * t_lo + (1.0 - alpha) * t_hi
            else:
                # Non-float buffers (e.g. running_mean num_batches_tracked):
                # use the lower-parameter variant's value.
                interp_sd[key] = t_lo.clone()

        new_model = copy.deepcopy(self.base_model)
        new_model.load_state_dict(interp_sd)
        return new_model

    def list_variants(self) -> List[float]:
        """Return a sorted list of all registered parameter values.

        Returns
        -------
        List[float]
            Sorted ascending.
        """
        return sorted(self._variants.keys())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the entire family to *path* using ``torch.save``.

        The file contains the base model state dict, parameter name, and all
        variant state dicts, keyed by their parameter values.

        Parameters
        ----------
        path:
            File path (e.g. ``"burgers_family.pt"``).
        """
        payload = {
            "__version__": 1,
            "param_name": self.param_name,
            "base_model_state": self.base_model.state_dict(),
            "variants": {
                str(k): v.state_dict() for k, v in self._variants.items()
            },
        }
        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        path: str,
        base_model: nn.Module,
        map_location: Optional[str] = None,
    ) -> "ParametricFamilyTransfer":
        """Load a previously saved family from *path*.

        Parameters
        ----------
        path:
            Path to a file saved by :meth:`save`.
        base_model:
            A model instance with the same architecture used during saving.
            Its weights are overwritten with the saved base-model state dict.
        map_location:
            Passed to ``torch.load`` (e.g. ``"cpu"`` for CPU-only machines).

        Returns
        -------
        ParametricFamilyTransfer
            Fully restored family object.
        """
        payload = torch.load(path, map_location=map_location, weights_only=False)
        param_name: str = payload["param_name"]

        # Restore base model.
        base_copy = copy.deepcopy(base_model)
        base_copy.load_state_dict(payload["base_model_state"])

        family = cls(base_copy, param_name)
        # Base model already deep-copied inside __init__; load once more to
        # overwrite the copy made in __init__.
        family.base_model.load_state_dict(payload["base_model_state"])

        for str_val, sd in payload["variants"].items():
            variant = copy.deepcopy(base_model)
            variant.load_state_dict(sd)
            family._variants[float(str_val)] = variant

        return family

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._variants)

    def __repr__(self) -> str:
        variants = self.list_variants()
        return (
            f"ParametricFamilyTransfer("
            f"param_name={self.param_name!r}, "
            f"variants={variants})"
        )
