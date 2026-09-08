"""pinneapple_registry.problem_store — ``ProblemStore``: versioned problem
definitions with history.

Two save paths, because a ``ProblemSpec`` cannot always be losslessly
round-tripped through JSON:

- :meth:`save_from_preset` — the RELIABLY reconstructable path. Presets
  are deterministic (``get_preset(preset_id, **kwargs)``), so this just
  records ``(preset_id, kwargs)`` and :meth:`load` calls ``get_preset``
  again rather than trying to serialize/deserialize ``ProblemSpec``'s
  nested dataclasses.
- :meth:`save_snapshot` — records ``dataclasses.asdict(spec)`` (or any
  plain dict) for history/inspection. This is explicitly NOT guaranteed
  round-trippable back into a live ``ProblemSpec``; :meth:`load_snapshot`
  returns the plain dict, not a reconstructed object.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


class ProblemStore:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _problem_dir(self, problem_id: str) -> str:
        return os.path.join(self.root, problem_id)

    def _new_version_dir(self, problem_id: str) -> tuple[str, str]:
        version = f"v{datetime.now():%Y%m%d_%H%M%S}"
        d = os.path.join(self._problem_dir(problem_id), version)
        os.makedirs(d, exist_ok=True)
        return version, d

    def save_from_preset(self, problem_id: str, preset_id: str, **preset_kwargs: Any) -> str:
        version, d = self._new_version_dir(problem_id)
        with open(os.path.join(d, "preset.json"), "w") as f:
            json.dump({"preset_id": preset_id, "preset_kwargs": preset_kwargs}, f, indent=2)
        return version

    def save_snapshot(self, problem_id: str, spec: Any) -> str:
        version, d = self._new_version_dir(problem_id)
        data = asdict(spec) if is_dataclass(spec) else spec
        with open(os.path.join(d, "snapshot.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)
        return version

    def versions(self, problem_id: str) -> List[str]:
        d = self._problem_dir(problem_id)
        if not os.path.isdir(d):
            return []
        return sorted(v for v in os.listdir(d) if os.path.isdir(os.path.join(d, v)))

    def load(self, problem_id: str, version: Optional[str] = None):
        """Reconstructs a live ``ProblemSpec`` — only works for versions
        saved via :meth:`save_from_preset`."""
        v = version or (self.versions(problem_id)[-1] if self.versions(problem_id) else None)
        if v is None:
            raise KeyError(f"No versions for problem_id='{problem_id}'")
        preset_path = os.path.join(self._problem_dir(problem_id), v, "preset.json")
        if not os.path.exists(preset_path):
            raise KeyError(
                f"Version '{v}' of '{problem_id}' was saved via save_snapshot(), not "
                "save_from_preset() -- use load_snapshot() instead, it is not reconstructable."
            )
        from pinneapple_physics.pde_environment.presets.registry import get_preset

        with open(preset_path) as f:
            data = json.load(f)
        return get_preset(data["preset_id"], **data["preset_kwargs"])

    def load_snapshot(self, problem_id: str, version: Optional[str] = None) -> Dict[str, Any]:
        v = version or (self.versions(problem_id)[-1] if self.versions(problem_id) else None)
        if v is None:
            raise KeyError(f"No versions for problem_id='{problem_id}'")
        snap_path = os.path.join(self._problem_dir(problem_id), v, "snapshot.json")
        if not os.path.exists(snap_path):
            raise KeyError(f"Version '{v}' of '{problem_id}' has no snapshot.json (saved via save_from_preset()).")
        with open(snap_path) as f:
            return json.load(f)

    def history(self, problem_id: str) -> List[Dict[str, Any]]:
        out = []
        for v in self.versions(problem_id):
            d = os.path.join(self._problem_dir(problem_id), v)
            kind = "preset" if os.path.exists(os.path.join(d, "preset.json")) else "snapshot"
            out.append({"version": v, "kind": kind})
        return out
