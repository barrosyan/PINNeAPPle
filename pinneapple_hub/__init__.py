"""Model hub client: ``push_to_hub``/``from_pretrained`` + ``ModelCard``.

See ``hub.py`` and ``model_card.py`` module docstrings for the design
rationale (piggybacking on the Hugging Face Hub's real infrastructure
rather than building new hosting, and why a ``ModelCard``'s
``validation_metrics`` are mandatory).
"""
from .model_card import ModelCard
from .hub import push_to_hub, from_pretrained

__all__ = ["ModelCard", "push_to_hub", "from_pretrained"]
