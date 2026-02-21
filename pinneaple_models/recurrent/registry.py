from __future__ import annotations
"""Registry and catalog for recurrent model family."""

from dataclasses import dataclass
from typing import Dict, Type

from .base import RecurrentModelBase
from .gru import GRUModel, BiGRUModel
from .lstm import LSTMModel, BiLSTMModel
from .seq2seq import Seq2SeqRNN


_REGISTRY: Dict[str, Type[RecurrentModelBase]] = {
    "gru": GRUModel,
    "bidirectional_gru": BiGRUModel,
    "bigru": BiGRUModel,

    "lstm": LSTMModel,
    "bidirectional_lstm": BiLSTMModel,
    "bilstm": BiLSTMModel,

    "seq2seq_rnn": Seq2SeqRNN,
    "seq2seq": Seq2SeqRNN,
}

def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry
    register_family_registry(_REGISTRY, family="recurrent")


@dataclass
class RecurrentCatalog:
    registry: Dict[str, Type[RecurrentModelBase]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> Type[RecurrentModelBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown recurrent model '{name}'. Available: {self.list()[:20]} ...")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> RecurrentModelBase:
        cls = self.get(name)
        return cls(**kwargs)
