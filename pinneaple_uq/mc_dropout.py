"""Monte Carlo Dropout for epistemic uncertainty quantification.

References
----------
Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning*. ICML.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from pinneaple_uq.core import UQResult


@dataclass
class MCDropoutConfig:
    """Configuration for Monte Carlo Dropout.

    Attributes
    ----------
    n_samples:
        Number of stochastic forward passes used to estimate the posterior
        predictive distribution.  Higher values yield lower variance estimates
        at the cost of more compute.
    dropout_p:
        Dropout probability applied to every ``nn.Linear`` layer during
        stochastic inference.
    seed:
        Optional integer RNG seed for reproducible sampling.  When ``None``
        the global PyTorch RNG state is used.
    """

    n_samples: int = 100
    dropout_p: float = 0.1
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.dropout_p < 1.0:
            raise ValueError(
                f"dropout_p must be in (0, 1); got {self.dropout_p}."
            )
        if self.n_samples < 1:
            raise ValueError(
                f"n_samples must be >= 1; got {self.n_samples}."
            )


class MCDropoutWrapper(nn.Module):
    """Wraps any ``nn.Module`` and adds Monte Carlo Dropout at inference time.

    A thin ``nn.Dropout`` hook is registered *after* every ``nn.Linear`` layer
    in the wrapped model.  The wrapper owns the dropout modules; the wrapped
    model's parameters are **never** modified.

    Parameters
    ----------
    model:
        Any trained ``nn.Module``.  Its weights are frozen from the wrapper's
        perspective (no gradient bookkeeping is changed here).
    config:
        :class:`MCDropoutConfig` instance controlling dropout probability and
        number of samples.

    Examples
    --------
    >>> cfg = MCDropoutConfig(n_samples=200, dropout_p=0.15)
    >>> wrapper = MCDropoutWrapper(trained_model, cfg)
    >>> result = wrapper.predict_with_uncertainty(x, device=torch.device("cuda"))
    >>> lower, upper = result.confidence_interval(alpha=0.05)
    """

    def __init__(self, model: nn.Module, config: MCDropoutConfig) -> None:
        super().__init__()
        self.config = config
        self.model = model

        # Collect handles so we can remove them later if needed.
        self._hooks: List[Any] = []
        self._dropout_modules: List[nn.Dropout] = []
        self._dropout_enabled: bool = False

        self._register_dropout_hooks()

    # ------------------------------------------------------------------
    # Hook registration
    # ------------------------------------------------------------------

    def _register_dropout_hooks(self) -> None:
        """Attach a post-forward Dropout hook to every Linear sub-module."""
        p = self.config.dropout_p

        for _name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                dropout = nn.Dropout(p=p)
                self._dropout_modules.append(dropout)

                # Capture in closure.
                _dropout_ref = dropout

                def _hook(
                    mod: nn.Module,
                    inputs: tuple,
                    output: Tensor,
                    _d: nn.Dropout = _dropout_ref,
                ) -> Optional[Tensor]:
                    if self._dropout_enabled:
                        return _d(output)
                    return output

                handle = module.register_forward_hook(_hook)
                self._hooks.append(handle)

    # ------------------------------------------------------------------
    # Dropout toggling
    # ------------------------------------------------------------------

    def enable_dropout(self) -> None:
        """Enable stochastic dropout on all hooked Linear layers."""
        self._dropout_enabled = True
        for d in self._dropout_modules:
            d.train()

    def disable_dropout(self) -> None:
        """Disable dropout — model behaves deterministically."""
        self._dropout_enabled = False
        for d in self._dropout_modules:
            d.eval()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Standard deterministic forward pass (dropout disabled).

        Parameters
        ----------
        x:
            Input tensor, shape ``(N, ...)``.

        Returns
        -------
        Tensor
            Model output, same shape as the wrapped model's output.
        """
        self.disable_dropout()
        return self.model(x)

    # ------------------------------------------------------------------
    # Uncertainty estimation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> UQResult:
        """Run *n_samples* stochastic forward passes and aggregate statistics.

        The wrapped model is set to ``train()`` mode only for the duration of
        sampling so that ``BatchNorm`` layers (if any) still track running
        statistics correctly.  Dropout is enabled via the registered hooks.

        Parameters
        ----------
        x:
            Input tensor, shape ``(N, ...)``.
        n_samples:
            Number of Monte Carlo samples.  Overrides ``config.n_samples``
            when provided.
        device:
            Target device.  Defaults to ``x.device``.

        Returns
        -------
        UQResult
            ``mean`` and ``std`` computed over the *n_samples* predictions;
            ``samples`` contains the raw stacked tensor of shape
            ``(n_samples, N, ...)``.
        """
        n = n_samples if n_samples is not None else self.config.n_samples
        if device is None:
            device = x.device

        x = x.to(device)
        self.model.to(device)

        # Seed for reproducibility.
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        # Collect samples with dropout active.
        self.model.train()   # needed so nn.Dropout works at module level too
        self.enable_dropout()

        preds: List[Tensor] = []
        for _ in range(n):
            out = self.model(x)
            # Detach to avoid accumulating the computation graph.
            preds.append(out.detach())

        self.disable_dropout()
        self.model.eval()

        # Stack → (n_samples, N, ...)
        samples = torch.stack(preds, dim=0)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=True)

        return UQResult(
            mean=mean,
            std=std,
            samples=samples,
            metadata={
                "method": "mc_dropout",
                "n_samples": n,
                "dropout_p": self.config.dropout_p,
            },
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks from the wrapped model.

        Call this if you need to use the wrapped model independently after
        uncertainty estimation.
        """
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def __del__(self) -> None:  # pragma: no cover
        # Best-effort cleanup when the wrapper is garbage-collected.
        try:
            self.remove_hooks()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Standalone convenience class
# ---------------------------------------------------------------------------

class MCDropout:
    """Convenience class that applies MC Dropout to any callable model.

    Unlike :class:`MCDropoutWrapper`, this class accepts **any callable**
    (not just ``nn.Module``), making it suitable for wrapped models, partial
    functions, or ONNX runtimes that expose a ``__call__`` interface.

    Parameters
    ----------
    model:
        Any callable mapping a ``Tensor`` to a ``Tensor``.
    config:
        :class:`MCDropoutConfig` instance.  When ``None``, default config is
        used.

    Notes
    -----
    Because this class cannot inject hooks into an arbitrary callable, it
    relies on the model having dropout layers that respect ``model.train()``
    and ``model.eval()`` (i.e. the model must be an ``nn.Module``).  For
    non-module callables, the stochastic variation comes solely from any
    randomness already present in the callable.

    Examples
    --------
    >>> mc = MCDropout(model, MCDropoutConfig(n_samples=50))
    >>> result = mc.predict_with_uncertainty(x)
    """

    def __init__(
        self,
        model: Callable[[Tensor], Tensor],
        config: Optional[MCDropoutConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or MCDropoutConfig()

    @torch.no_grad()
    def predict_with_uncertainty(
        self,
        x: Tensor,
        device: Optional[torch.device] = None,
    ) -> UQResult:
        """Run stochastic forward passes and return uncertainty estimates.

        Parameters
        ----------
        x:
            Input tensor, shape ``(N, ...)``.
        device:
            Target device.  Defaults to ``x.device``.

        Returns
        -------
        UQResult
            Aggregated mean, std, and raw samples.
        """
        if device is None:
            device = x.device if isinstance(x, Tensor) else torch.device("cpu")

        x = x.to(device)

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

        # Enable train mode if the callable is an nn.Module.
        if isinstance(self.model, nn.Module):
            self.model.train()
            self.model.to(device)

        preds: List[Tensor] = []
        for _ in range(self.config.n_samples):
            preds.append(self.model(x).detach())

        if isinstance(self.model, nn.Module):
            self.model.eval()

        samples = torch.stack(preds, dim=0)
        mean = samples.mean(dim=0)
        std = samples.std(dim=0, unbiased=True)

        return UQResult(
            mean=mean,
            std=std,
            samples=samples,
            metadata={
                "method": "mc_dropout",
                "n_samples": self.config.n_samples,
                "dropout_p": self.config.dropout_p,
            },
        )
