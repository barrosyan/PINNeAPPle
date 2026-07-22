from __future__ import annotations
"""Registry and catalog for PINN model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import PINNBase
from .vanilla import VanillaPINN
from .inverse import InversePINN
from .pielm import PIELM
from .pinn_lstm import PINNLSTM
from .pinnsformer import PINNsFormer
from .vpinn import VPINN
from .xpinn import XPINN
from .xtfc import XTFC


_REGISTRY: Dict[str, Type[PINNBase]] = {
    "vanilla_pinn": VanillaPINN,
    "pinn": VanillaPINN,
    "vanilla": VanillaPINN,

    "inverse_pinn": InversePINN,
    "inv_pinn": InversePINN,

    "pielm": PIELM,
    "pi_elm": PIELM,

    "pinn_lstm": PINNLSTM,

    "pinnsformer": PINNsFormer,
    "pinn_former": PINNsFormer,

    "vpinn": VPINN,
    "variational_pinn": VPINN,

    "xpinn": XPINN,

    "xtfc": XTFC,
    "extreme_tfc": XTFC,
}

def register_into_global() -> None:
    from pinneapple_neural.architectures._registry_bridge import register_family_registry

    def capabilities(name: str, cls) -> dict:
        # PINNLSTM.forward(*inputs) requires exactly in_dim separate 1-column
        # tensors and internally reshapes to (B, seq_len, in_dim) — a real
        # sequence model wearing a PINN name, unlike every other class here
        # (VanillaPINN/InversePINN/PIELM/PINNsFormer/VPINN/XTFC all accept a
        # single (N, in_dim) tensor via forward(x, ...) or forward(*inputs)
        # with inputs=(x,)).
        if cls is PINNLSTM:
            return {"input_kind": "sequence", "expects": ["x_past"], "predicts": ["u"]}
        return {"input_kind": "pointwise_coords", "expects": ["x"], "predicts": ["u"]}

    register_family_registry(_REGISTRY, family="pinns", capabilities_getter=capabilities)

@dataclass
class PINNCatalog:
    registry: Dict[str, Type[PINNBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[PINNBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown pinn model '{name}'. Available: {self.list()[:20]} ...")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> PINNBase:
        cls = self.get(name)
        return cls(**kwargs)
