"""PhysicsNeMo -> PINNeAPPle model adapters (UQ + Digital Twin).

Promotes the real "PhysicsNeMo trains, PINNeAPPle operates" glue code from
``examples/vs_physicsnemo/05_combined_fno_digital_twin/example.py`` (FASE 2,
``add_uq()``: wrap a trained surrogate in ``MCDropoutWrapper``; FASE 3,
``run_digital_twin()``: wrap the same surrogate in ``DigitalTwin``) into a
maintained, importable module, so this hand-off no longer requires
copy-pasting from the example script.

``physicsnemo`` (a.k.a. ``nvidia-physicsnemo``, formerly ``nvidia-modulus``)
is a genuinely optional dependency here: importing this module never
requires it, because a trained physicsnemo model is, by the time it reaches
PINNeAPPle, just a ``torch.nn.Module`` -- exactly what the example script
hands to ``MCDropoutWrapper``/``DigitalTwin`` directly. Only
:func:`require_physicsnemo_model` performs a real ``import physicsnemo`` and
raises a clear, actionable ``ImportError`` if the package is missing.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from pinneapple_analysis.uncertainty import MCDropoutConfig, MCDropoutWrapper
from pinneapple_systems.digital_twin.twin import DigitalTwin, DigitalTwinConfig


# ---------------------------------------------------------------------------
# Optional physicsnemo import
# ---------------------------------------------------------------------------

def _require_physicsnemo() -> Any:
    """Import ``physicsnemo`` (or its pre-rename package, ``modulus``).

    Raises a clear, actionable ``ImportError`` if neither is installed. This
    is the ONLY place in :mod:`pinneapple_neural.physicsnemo_bridge` that
    performs a real physicsnemo import -- everything else in this module
    (and in :mod:`pinneapple_neural.physicsnemo_bridge.active_learning_bridge`)
    works with any ``torch.nn.Module``, physicsnemo-trained or not.
    """
    try:
        import physicsnemo  # type: ignore
        return physicsnemo
    except ImportError:
        pass
    try:
        import modulus  # type: ignore  # pre-rename package name (NVIDIA Modulus)
        return modulus
    except ImportError as e:
        raise ImportError(
            "physicsnemo is required for this operation but is not installed. "
            "Install with: pip install nvidia-physicsnemo "
            "(see https://github.com/NVIDIA/physicsnemo). "
            "Note: the rest of pinneapple_neural.physicsnemo_bridge (model "
            "adapters, UQ/digital-twin wiring, the active-learning retrain "
            "loop) works on any torch.nn.Module without physicsnemo "
            "installed -- only operations that need a genuine physicsnemo "
            "import, such as this one, require it."
        ) from e


def is_physicsnemo_available() -> bool:
    """Return ``True`` if ``physicsnemo`` (or the legacy ``modulus`` name) can be imported."""
    try:
        _require_physicsnemo()
        return True
    except ImportError:
        return False


def require_physicsnemo_model(model: Any) -> Any:
    """Validate that ``model`` is a physicsnemo-compatible trained model.

    This requires physicsnemo to actually be importable (unlike everything
    else in this module) since it is the boundary where PINNeAPPle asserts
    "this really did come from a PhysicsNeMo training run" rather than just
    duck-typing any ``nn.Module``. Trained PhysicsNeMo models are themselves
    ``torch.nn.Module`` instances, so once physicsnemo is confirmed
    importable the only remaining check is that shape.

    Raises
    ------
    ImportError
        If physicsnemo is not installed.
    TypeError
        If physicsnemo is installed but ``model`` is not an ``nn.Module``.
    """
    _require_physicsnemo()
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Expected a torch.nn.Module (a trained physicsnemo model is "
            f"one), got {type(model)!r}."
        )
    return model


# ---------------------------------------------------------------------------
# Output-normalizing adapter
# ---------------------------------------------------------------------------

class PhysicsNemoModelAdapter(nn.Module):
    """Wrap a trained model so its ``forward()`` output is one plain tensor.

    ``MCDropoutWrapper`` expects its wrapped model's ``forward`` to return a
    plain tensor it can average/std over ``n_samples`` stochastic passes.
    ``DigitalTwin._torch_predict`` is more permissive -- it already unwraps a
    ``dict`` of named fields or an object exposing ``.y`` (the two output
    shapes PhysicsNeMo / PhysicsNeMo-Sym models commonly use) -- but that
    unwrapping happens only inside ``DigitalTwin``, not for a direct call.

    This adapter performs the same unwrapping once, up front, so a single
    adapted model can be handed to *both* ``MCDropoutWrapper`` (FASE 2 of the
    example) and ``DigitalTwin`` (FASE 3) exactly as the example script
    passes its plain-tensor-output FNO surrogate to each.

    Parameters
    ----------
    model:
        Any trained ``nn.Module`` -- a physicsnemo model or a plain
        PINNeAPPle one. Its parameters are untouched; this only wraps
        ``forward``.
    field_names:
        Names of the output fields, in the order they should be
        concatenated along the last dimension when ``model``'s raw output is
        a dict.
    """

    def __init__(self, model: nn.Module, field_names: Sequence[str]) -> None:
        super().__init__()
        self.model = model
        self.field_names = list(field_names)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._normalize_output(self.model(x))

    def _normalize_output(self, out: Any) -> torch.Tensor:
        if isinstance(out, torch.Tensor):
            return out

        if hasattr(out, "y"):  # PINNOutput / OperatorOutput-style wrapper
            out = out.y
            if isinstance(out, torch.Tensor):
                return out

        if isinstance(out, dict):
            missing = [f for f in self.field_names if f not in out]
            if missing:
                raise KeyError(
                    f"Model output dict is missing field(s) {missing}; "
                    f"available keys: {sorted(out.keys())}"
                )
            parts = []
            for f in self.field_names:
                v = out[f]
                v = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
                if v.dim() == 1:
                    v = v.unsqueeze(-1)
                parts.append(v)
            return torch.cat(parts, dim=-1)

        raise TypeError(
            f"Unsupported model output type {type(out)!r}; expected a "
            "torch.Tensor, a dict of named fields, or an object exposing "
            "`.y`."
        )


# ---------------------------------------------------------------------------
# FASE 2 -- UQ (examples/vs_physicsnemo/05_combined_fno_digital_twin/example.py::add_uq)
# ---------------------------------------------------------------------------

def wrap_for_uq(
    model: nn.Module,
    field_names: Optional[Sequence[str]] = None,
    *,
    mc_dropout_config: Optional[MCDropoutConfig] = None,
) -> MCDropoutWrapper:
    """Wrap a trained model for MC-Dropout uncertainty quantification.

    Promotes FASE 2 (``add_uq``) of
    ``examples/vs_physicsnemo/05_combined_fno_digital_twin/example.py``,
    which built ``MCDropoutWrapper(model, MCDropoutConfig(n_samples=50,
    dropout_p=0.05))`` directly around a plain-tensor-output FNO surrogate.

    Parameters
    ----------
    model:
        Trained ``nn.Module`` (physicsnemo or plain).
    field_names:
        If given and ``model`` isn't already a :class:`PhysicsNemoModelAdapter`,
        ``model`` is first wrapped in one -- needed when the raw model
        returns a dict of named fields or a ``.y``-style output rather than
        a plain tensor.
    mc_dropout_config:
        Defaults to the same settings the example used:
        ``MCDropoutConfig(n_samples=50, dropout_p=0.05)``.
    """
    adapted: nn.Module = model
    if field_names is not None and not isinstance(model, PhysicsNemoModelAdapter):
        adapted = PhysicsNemoModelAdapter(model, field_names)
    cfg = mc_dropout_config or MCDropoutConfig(n_samples=50, dropout_p=0.05)
    return MCDropoutWrapper(adapted, cfg)


# ---------------------------------------------------------------------------
# FASE 3 -- Digital Twin (examples/vs_physicsnemo/05.../example.py::run_digital_twin)
# ---------------------------------------------------------------------------

def build_digital_twin_for_model(
    model: nn.Module,
    field_names: Sequence[str],
    coord_names: Optional[Sequence[str]] = None,
    *,
    config: Optional[DigitalTwinConfig] = None,
) -> DigitalTwin:
    """Wrap a trained model into a :class:`DigitalTwin`.

    Promotes FASE 3 (``run_digital_twin``) of
    ``examples/vs_physicsnemo/05_combined_fno_digital_twin/example.py``,
    which passed a trained FNO surrogate directly as
    ``DigitalTwin(model=model, field_names=[...], config=cfg)``.

    ``DigitalTwin`` already unwraps dict/``.y`` outputs internally (see
    ``DigitalTwin._torch_predict``); wrapping via
    :class:`PhysicsNemoModelAdapter` here is done anyway so the *same*
    adapted model returned by this function and by :func:`wrap_for_uq` can
    be reused interchangeably across both entry points, matching the
    example's FASE 2 -> FASE 3 flow where one trained model feeds both.

    Parameters
    ----------
    model:
        Trained ``nn.Module`` (physicsnemo or plain).
    field_names:
        Names of the physical fields the model predicts (e.g. ``["u", "v", "p"]``).
    coord_names:
        Names of the coordinate inputs (default ``["x", "y"]``, per
        ``DigitalTwin``'s own default).
    config:
        Optional :class:`DigitalTwinConfig`.
    """
    adapted = (
        model
        if isinstance(model, PhysicsNemoModelAdapter)
        else PhysicsNemoModelAdapter(model, field_names)
    )
    return DigitalTwin(
        adapted,
        field_names=list(field_names),
        coord_names=list(coord_names) if coord_names is not None else None,
        config=config,
    )
