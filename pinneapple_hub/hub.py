"""``push_to_hub`` / ``from_pretrained`` for PINNeAPPle models, built on the
Hugging Face Hub's own storage/auth/versioning infrastructure rather than
a new hosting service this repository would have to build, operate, and
secure itself. This is a deliberate, common pattern (many domain-specific
libraries piggyback on HF Hub this way) -- the value PINNeAPPle adds is
the ``pinneapple_config.json`` sidecar convention that lets a checkpoint
be reconstructed into a live model without the downloader having to
already know its exact architecture kwargs, plus the ``ModelCard`` (see
``model_card.py``) reproducibility contract every push carries.

Requires the ``hub`` extra (``pip install "pinneapple[hub]"``, i.e.
``huggingface_hub``) and a Hugging Face account/token for anything beyond
downloading public repos.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional

from .model_card import ModelCard

_CONFIG_FILENAME = "pinneapple_config.json"
_CARD_FILENAME = "pinneapple_model_card.json"
_WEIGHTS_FILENAME = "pytorch_model.bin"


def _require_hub():
    try:
        import huggingface_hub  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pinneapple_hub requires the 'hub' extra: pip install \"pinneapple[hub]\" "
            "(huggingface_hub)."
        ) from e
    return huggingface_hub


def push_to_hub(
    model,
    repo_id: str,
    model_card: ModelCard,
    *,
    architecture: Optional[str] = None,
    architecture_config: Optional[Dict[str, Any]] = None,
    private: bool = False,
    token: Optional[str] = None,
    commit_message: str = "Push model via pinneapple_hub",
) -> str:
    """Upload a trained model's weights + architecture config + model card
    to a Hugging Face Hub model repo.

    Parameters
    ----------
    model : the trained ``nn.Module`` (its ``state_dict()`` is uploaded).
    repo_id : ``"username/repo-name"``.
    model_card : a :class:`ModelCard` -- ``model_card.validate()`` is run
        first and a non-empty problem list raises (a hub push with no
        checkable validation claim is refused by default; see
        ``ModelCard.validate``'s docstring for why).
    architecture, architecture_config : the
        ``pinneapple_neural.architectures.ModelRegistry`` name and the
        kwargs ``ModelRegistry.build(architecture, **architecture_config)``
        needs to reconstruct this exact model shape before loading the
        state_dict back in :func:`from_pretrained`. Defaults to
        ``model_card.architecture``/``model_card.architecture_config`` if
        not given explicitly.
    private : create the repo as private if it doesn't exist yet.
    token : HF token; falls back to the ``huggingface_hub`` CLI's cached
        login (``huggingface-cli login``) if not given.

    Returns
    -------
    The repo URL.
    """
    import torch

    hub = _require_hub()

    problems = model_card.validate()
    if problems:
        raise ValueError(
            f"ModelCard for '{repo_id}' failed validation, refusing to push:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )

    arch = architecture or model_card.architecture
    arch_cfg = architecture_config or model_card.architecture_config
    if not arch:
        raise ValueError("architecture must be given (either as an argument or on model_card.architecture)")

    api = hub.HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        weights_path = os.path.join(tmp, _WEIGHTS_FILENAME)
        torch.save(model.state_dict(), weights_path)

        config_path = os.path.join(tmp, _CONFIG_FILENAME)
        with open(config_path, "w") as f:
            json.dump({"architecture": arch, "architecture_config": arch_cfg}, f, indent=2)

        card_path = os.path.join(tmp, _CARD_FILENAME)
        model_card.save(card_path)

        readme_path = os.path.join(tmp, "README.md")
        with open(readme_path, "w") as f:
            f.write(model_card.to_markdown())

        api.upload_folder(repo_id=repo_id, folder_path=tmp, commit_message=commit_message)

    return f"https://huggingface.co/{repo_id}"


def from_pretrained(
    repo_id: str,
    *,
    revision: str = "main",
    device: str = "cpu",
    token: Optional[str] = None,
):
    """Download and reconstruct a model pushed with :func:`push_to_hub`.

    Returns
    -------
    ``(model, model_card)`` -- the live ``nn.Module`` (weights loaded,
    moved to ``device``, in eval mode) and its :class:`ModelCard`.
    """
    import torch
    import pinneapple_neural.architectures  # noqa: F401  registers the model zoo
    from pinneapple_neural.architectures.registry import ModelRegistry

    hub = _require_hub()

    config_path = hub.hf_hub_download(repo_id, _CONFIG_FILENAME, revision=revision, token=token)
    weights_path = hub.hf_hub_download(repo_id, _WEIGHTS_FILENAME, revision=revision, token=token)
    card_path = hub.hf_hub_download(repo_id, _CARD_FILENAME, revision=revision, token=token)

    with open(config_path, "r") as f:
        config = json.load(f)
    model_card = ModelCard.load(card_path)

    model = ModelRegistry.build(config["architecture"], **config["architecture_config"])
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, model_card
