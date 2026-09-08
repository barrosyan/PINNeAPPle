"""pinneapple_neural.physicsnemo_bridge -- PhysicsNeMo <-> PINNeAPPle integration glue.

Promotes example-level PhysicsNeMo integration code from
``examples/pinneapple_and_physicsnemo/`` and ``examples/vs_physicsnemo/``
into a maintained core package. Those two directories contain real, working
scripts demonstrating three integration patterns
(``examples/vs_physicsnemo/README.md``, "Padroes de integracao"): (1)
PhysicsNeMo trains a model, PINNeAPPle operates/UQs/twins it; (2) PINNeAPPle
active-learning selects points, PhysicsNeMo retrains; (3) PhysicsNeMo
MeshGraphNet output validated by PINNeAPPle -- but the glue was never
promoted beyond the example scripts, so every user wanting it had to
copy-paste from ``example.py`` rather than ``import`` a real module. This
package fixes that for patterns 1 and 2 (pattern 3 -- physics validation and
ONNX/TorchScript export of a PhysicsNeMo MeshGraphNet, demonstrated in
``examples/vs_physicsnemo/06_combined_meshgraphnet_valid/example.py`` -- is
out of scope here; it belongs with PINNeAPPle's own validation/export
modules, not this bridge). The example scripts themselves are left
untouched.

This follows the same relocation convention as
``pinneapple_analysis.verification`` (a similar promotion of
previously-siloed logic done in a prior session): the real, working
integration logic is extracted into tested, reusable functions/classes
here, without reinventing anything the examples or PINNeAPPle's own modules
(``pinneapple_data.active_learning``, ``pinneapple_analysis.uncertainty``,
``pinneapple_systems.digital_twin``) already do for real.

Sub-modules
-----------
adapters
    Pattern 1 ("PhysicsNeMo trains, PINNeAPPle operates"):
    ``PhysicsNemoModelAdapter`` normalizes a trained model's output (plain
    tensor, dict of named fields, or a ``.y``-style wrapper) into the plain
    tensor shape ``MCDropoutWrapper`` and ``DigitalTwin`` expect;
    ``wrap_for_uq``/``build_digital_twin_for_model`` promote the FASE 2 /
    FASE 3 glue from
    ``examples/vs_physicsnemo/05_combined_fno_digital_twin/example.py``.

active_learning_bridge
    Pattern 2 ("PINNeAPPle active-learning selects points, PhysicsNeMo
    retrains"): ``active_learning_retrain_loop`` runs the real
    select -> retrain -> select -> ... loop, built directly on
    ``pinneapple_data.active_learning.ResidualBasedAL`` (delegated, never
    reimplemented).

Optional dependency
--------------------
``physicsnemo`` (a.k.a. ``nvidia-physicsnemo``, formerly ``nvidia-modulus``)
is genuinely optional: importing this package, or anything in it, never
requires physicsnemo to be installed -- every function here operates on any
``torch.nn.Module``, physicsnemo-trained or not, because a trained
physicsnemo model already IS a plain ``nn.Module`` by the time it reaches
PINNeAPPle. Only ``adapters.require_physicsnemo_model`` performs a real
``import physicsnemo`` and raises a clear, actionable ``ImportError`` (with
an install hint) if the package is missing.
"""
from __future__ import annotations

from pinneapple_neural.physicsnemo_bridge.adapters import (
    PhysicsNemoModelAdapter,
    build_digital_twin_for_model,
    is_physicsnemo_available,
    require_physicsnemo_model,
    wrap_for_uq,
)
from pinneapple_neural.physicsnemo_bridge.active_learning_bridge import (
    ALRetrainResult,
    ALRoundResult,
    RetrainConfig,
    active_learning_retrain_loop,
)

__all__ = [
    # adapters
    "PhysicsNemoModelAdapter",
    "is_physicsnemo_available",
    "require_physicsnemo_model",
    "wrap_for_uq",
    "build_digital_twin_for_model",
    # active_learning_bridge
    "RetrainConfig",
    "ALRoundResult",
    "ALRetrainResult",
    "active_learning_retrain_loop",
]
