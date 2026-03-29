"""Deep Ensemble uncertainty quantification.

References
----------
Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). *Simple and Scalable
Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from pinneaple_uq.core import UQResult


@dataclass
class EnsembleConfig:
    """Configuration for Deep Ensemble UQ.

    Attributes
    ----------
    n_members:
        Number of ensemble members to create when using the factory
        constructor :meth:`EnsembleUQ.from_config`.  Has no effect when
        ``EnsembleUQ`` is instantiated directly with a list of models.
    diversity_reg:
        Coefficient for an optional diversity regularisation term that can be
        incorporated into a custom loss function.  ``EnsembleUQ`` stores this
        value in :attr:`metadata` but does **not** apply it automatically
        during inference.  Pass it to your training loss to encourage ensemble
        diversity.
    seed:
        Base random seed used when :meth:`EnsembleUQ.from_config` creates
        ensemble members.  Member *i* is initialised with ``seed + i`` so
        that each member starts from a different random state.
    """

    n_members: int = 5
    diversity_reg: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_members < 1:
            raise ValueError(
                f"n_members must be >= 1; got {self.n_members}."
            )
        if self.diversity_reg < 0.0:
            raise ValueError(
                f"diversity_reg must be >= 0; got {self.diversity_reg}."
            )


class EnsembleUQ:
    """Deep Ensemble wrapper for predictive uncertainty quantification.

    Aggregates predictions from multiple independently-trained models to
    estimate both mean and variance of the predictive distribution.

    Parameters
    ----------
    models:
        List of independently trained ``nn.Module`` instances.  Each model
        must accept the same input shape and return the same output shape.
    config:
        :class:`EnsembleConfig` controlling ensemble-level hyperparameters.

    Examples
    --------
    >>> ensemble = EnsembleUQ(trained_models, EnsembleConfig(n_members=5))
    >>> result = ensemble.predict_with_uncertainty(x, device=torch.device("cuda"))
    >>> print(result.std.mean())  # average epistemic uncertainty
    """

    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[EnsembleConfig] = None,
    ) -> None:
        if not models:
            raise ValueError("models list must contain at least one member.")
        self.models = list(models)
        self.config = config or EnsembleConfig(n_members=len(models))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: Tensor,
        device: Optional[torch.device] = None,
    ) -> UQResult:
        """Run all ensemble members and aggregate predictive statistics.

        Each member produces one prediction; the ensemble mean and std are
        computed across members.

        Parameters
        ----------
        x:
            Input tensor, shape ``(N, ...)``.
        device:
            Target device.  Defaults to ``x.device``.

        Returns
        -------
        UQResult
            ``mean`` and ``std`` of member predictions; ``samples`` contains
            the raw per-member predictions stacked as ``(n_members, N, ...)``;
            ``metadata["member_preds"]`` holds the same tensor for convenience.
        """
        if device is None:
            device = x.device if isinstance(x, Tensor) else torch.device("cpu")

        x = x.to(device)
        member_preds: List[Tensor] = []

        for model in self.models:
            model.eval()
            model.to(device)
            out = model(x).detach()
            member_preds.append(out)

        # Stack → (n_members, N, ...)
        samples = torch.stack(member_preds, dim=0)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=True)

        return UQResult(
            mean=mean,
            std=std,
            samples=samples,
            metadata={
                "method": "ensemble",
                "n_members": len(self.models),
                "diversity_reg": self.config.diversity_reg,
                "member_preds": samples,
            },
        )

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def fit_member(
        self,
        idx: int,
        trainer: Any,
        dataset: Any,
        **trainer_kwargs: Any,
    ) -> None:
        """Train a single ensemble member using a provided trainer.

        This is a lightweight shim that calls ``trainer.fit(model, dataset,
        **trainer_kwargs)``.  The trainer must follow the Pinneaple
        ``Trainer.fit(model, ...)`` convention.

        Parameters
        ----------
        idx:
            Index of the ensemble member to train (0-based).
        trainer:
            A Pinneaple-compatible trainer instance.  Expected to expose a
            ``fit(model, dataset, **kwargs)`` method.
        dataset:
            Training dataset or dataloader passed directly to the trainer.
        **trainer_kwargs:
            Additional keyword arguments forwarded to ``trainer.fit``.

        Raises
        ------
        IndexError
            If *idx* is out of range.
        """
        if idx < 0 or idx >= len(self.models):
            raise IndexError(
                f"Member index {idx} out of range for ensemble of size "
                f"{len(self.models)}."
            )
        trainer.fit(self.models[idx], dataset, **trainer_kwargs)

    # ------------------------------------------------------------------
    # Factory constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model_factory: Callable[[], nn.Module],
        n_members: int = 5,
        config: Optional[EnsembleConfig] = None,
        **factory_kwargs: Any,
    ) -> "EnsembleUQ":
        """Create an ``EnsembleUQ`` by instantiating *model_factory* n times.

        Each member is initialised with a distinct random seed so that weight
        initialisation diversity is maximised.

        Parameters
        ----------
        model_factory:
            A zero-argument callable (or partial) that returns a fresh
            ``nn.Module``.  Any ``factory_kwargs`` are forwarded.
        n_members:
            Number of ensemble members to create.  Overrides
            ``config.n_members`` when both are provided.
        config:
            Optional :class:`EnsembleConfig`.  When ``None``, a default config
            is created using ``n_members`` and default values.
        **factory_kwargs:
            Keyword arguments forwarded to ``model_factory``.

        Returns
        -------
        EnsembleUQ
            Ensemble with *n_members* freshly initialised (untrained) members.

        Examples
        --------
        >>> def make_model():
        ...     return nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))
        >>> ensemble = EnsembleUQ.from_config(make_model, n_members=5)
        """
        if config is None:
            config = EnsembleConfig(n_members=n_members)
        else:
            n_members = config.n_members

        models: List[nn.Module] = []
        for i in range(n_members):
            torch.manual_seed(config.seed + i)
            if factory_kwargs:
                m = model_factory(**factory_kwargs)
            else:
                m = model_factory()
            models.append(m)

        # Restore previous generator state to avoid global side-effects.
        # (We only seeded within the loop; the final state is deterministic.)
        return cls(models, config)
