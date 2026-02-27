from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

from pinneaple_arena.bundle.loader import BundleData


@runtime_checkable
class Backend(Protocol):
    """Training backend protocol.

    Backends may return:
      - model: a torch.nn.Module taking (N,2)->(N,3)
      - predict_fn: callable numpy (N,2)->(N,3) for framework-agnostic evaluation
      - metrics: dict of floats
    """
    name: str

    def train(self, bundle: BundleData, run_cfg: Dict[str, Any]) -> Dict[str, Any]:
        ...