"""PhysicsTransferAdapter — domain-shift adaptation for PINNs.

Provides two complementary tools for adapting a model trained on one physics
problem to a different (but related) physics problem:

1. **Combined source + target physics loss** — regularises the fine-tuning
   process by keeping the model consistent with the source domain physics
   while simultaneously satisfying the target domain PDE residual.

2. **MMD (Maximum Mean Discrepancy) domain-adaptation loss** — aligns the
   intermediate feature distributions of the model across the source and
   target spatial domains, reducing the co-variate shift that arises when
   the geometry or parameter range changes significantly.

Reference for MMD:
    Gretton et al., "A Kernel Two-Sample Test", JMLR 2012.

Example
-------
>>> adapter = PhysicsTransferAdapter(model, source_spec, target_spec)
>>> combined_loss = adapter.adapt_physics_loss(my_target_pde_fn)
>>> x_src = torch.rand(256, 2)
>>> x_tgt = torch.rand(256, 2)
>>> mmd = adapter.domain_adaptation_loss(x_src, x_tgt)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rbf_kernel(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """Compute the RBF (Gaussian) kernel matrix K(x, y).

    K_{ij} = exp(-||x_i - y_j||^2 / (2 * bandwidth^2))

    Parameters
    ----------
    x:
        Tensor of shape ``(n, d)``.
    y:
        Tensor of shape ``(m, d)``.
    bandwidth:
        Kernel bandwidth σ.

    Returns
    -------
    torch.Tensor
        Shape ``(n, m)``.
    """
    # Squared pairwise distances: (n, m)
    x_sq = (x ** 2).sum(dim=1, keepdim=True)   # (n, 1)
    y_sq = (y ** 2).sum(dim=1, keepdim=True)   # (m, 1)
    cross = x @ y.T                            # (n, m)
    dist_sq = x_sq + y_sq.T - 2.0 * cross
    return torch.exp(-dist_sq / (2.0 * bandwidth ** 2))


def _mmd_squared(
    phi_s: torch.Tensor,
    phi_t: torch.Tensor,
    bandwidth: float,
) -> torch.Tensor:
    """Unbiased estimate of MMD^2 between feature sets *phi_s* and *phi_t*.

    Parameters
    ----------
    phi_s:
        Source features, shape ``(n, d)``.
    phi_t:
        Target features, shape ``(m, d)``.
    bandwidth:
        RBF kernel bandwidth.

    Returns
    -------
    torch.Tensor
        Scalar tensor.
    """
    n = phi_s.shape[0]
    m = phi_t.shape[0]

    kss = _rbf_kernel(phi_s, phi_s, bandwidth)
    ktt = _rbf_kernel(phi_t, phi_t, bandwidth)
    kst = _rbf_kernel(phi_s, phi_t, bandwidth)

    # Unbiased estimator: zero out diagonal contributions for same-sample terms.
    kss_sum = (kss.sum() - kss.trace()) / max(n * (n - 1), 1)
    ktt_sum = (ktt.sum() - ktt.trace()) / max(m * (m - 1), 1)
    kst_sum = kst.sum() / max(n * m, 1)

    return kss_sum + ktt_sum - 2.0 * kst_sum


def _parse_loss(loss_out: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Normalise physics_fn output to ``(total_tensor, components)``."""
    if isinstance(loss_out, torch.Tensor):
        return loss_out, {"total": float(loss_out.item())}
    if isinstance(loss_out, dict):
        if "total" not in loss_out:
            raise ValueError(
                "physics_fn returned a dict without a 'total' key."
            )
        components = {
            k: float(v.item()) if isinstance(v, torch.Tensor) else float(v)
            for k, v in loss_out.items()
        }
        return loss_out["total"], components
    raise TypeError(
        "physics_fn must return torch.Tensor or dict[str, Tensor]."
    )


# ---------------------------------------------------------------------------
# Feature extractor hook
# ---------------------------------------------------------------------------

class _IntermediateFeatureHook:
    """Attach a forward hook to a named submodule and cache its output."""

    def __init__(self, model: nn.Module, layer_name: str) -> None:
        self._features: Optional[torch.Tensor] = None
        module = dict(model.named_modules()).get(layer_name)
        if module is None:
            raise ValueError(
                f"Layer {layer_name!r} not found in model.  "
                f"Available: {list(dict(model.named_modules()).keys())}"
            )
        self._handle = module.register_forward_hook(self._hook)

    def _hook(
        self,
        module: nn.Module,
        input: Any,
        output: Any,
    ) -> None:
        if isinstance(output, torch.Tensor):
            self._features = output
        elif hasattr(output, "y") and isinstance(output.y, torch.Tensor):
            self._features = output.y
        else:
            self._features = None

    @property
    def features(self) -> Optional[torch.Tensor]:
        return self._features

    def remove(self) -> None:
        self._handle.remove()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class PhysicsTransferAdapter:
    """Adapt a PINN trained on one physics domain to a related target domain.

    Parameters
    ----------
    model:
        The (potentially partially fine-tuned) ``nn.Module`` to adapt.
    source_spec:
        A ``ProblemSpec``-compatible object describing the source physics
        problem.  Only ``source_spec`` is stored for reference; it is not
        used algorithmically unless you subclass and override
        :meth:`_source_physics_fn`.
    target_spec:
        A ``ProblemSpec``-compatible object describing the target physics
        problem.  Same note as above.
    source_physics_fn:
        Optional callable ``(model, batch) -> Tensor | dict``.  When
        provided, :meth:`adapt_physics_loss` uses it as the source
        regularisation term.  If ``None``, the source term is omitted.
    source_weight:
        Weight applied to the source physics regularisation loss.
    mmd_bandwidth:
        Bandwidth σ of the RBF kernel used in the MMD computation.
    feature_layer:
        Name of the submodule whose output is used as the feature
        representation for MMD.  E.g. ``"net.2"`` or ``"trunk"``.
        If ``None``, the raw model input is used as the feature (zero
        domain-shift assumption in input space).

    Example
    -------
    >>> adapter = PhysicsTransferAdapter(
    ...     model,
    ...     source_spec=burgers_spec_nu001,
    ...     target_spec=burgers_spec_nu005,
    ...     source_physics_fn=burgers_loss_nu001,
    ...     feature_layer="net.2",
    ... )
    >>> loss_fn = adapter.adapt_physics_loss(burgers_loss_nu005)
    >>> total_loss = loss_fn(model, batch)
    """

    def __init__(
        self,
        model: nn.Module,
        source_spec: Any,
        target_spec: Any,
        source_physics_fn: Optional[Callable[[nn.Module, Dict[str, Any]], Any]] = None,
        source_weight: float = 0.1,
        mmd_bandwidth: float = 1.0,
        feature_layer: Optional[str] = None,
    ) -> None:
        self.model = model
        self.source_spec = source_spec
        self.target_spec = target_spec
        self._source_physics_fn = source_physics_fn
        self.source_weight = source_weight
        self.mmd_bandwidth = mmd_bandwidth
        self.feature_layer = feature_layer

    # ------------------------------------------------------------------
    # Physics adaptation
    # ------------------------------------------------------------------

    def adapt_physics_loss(
        self,
        target_physics_fn: Callable[[nn.Module, Dict[str, Any]], Any],
    ) -> Callable[[nn.Module, Dict[str, Any]], Dict[str, torch.Tensor]]:
        """Return a combined source + target physics loss callable.

        The returned callable has the standard pinneaple loss signature::

            loss_fn(model, batch) -> dict(total=..., target=..., source=...)

        Parameters
        ----------
        target_physics_fn:
            ``callable(model, batch) -> Tensor | dict``.  Computes the PDE
            residual for the *target* domain.

        Returns
        -------
        callable
            Combined loss function that can be passed directly to
            :class:`~pinneaple_train.trainer.Trainer` or
            :class:`~pinneaple_transfer.trainer.TransferTrainer`.
        """
        source_fn = self._source_physics_fn
        source_weight = self.source_weight

        def combined(
            model: nn.Module,
            batch: Dict[str, Any],
        ) -> Dict[str, torch.Tensor]:
            tgt_loss, _ = _parse_loss(target_physics_fn(model, batch))
            total = tgt_loss

            components: Dict[str, torch.Tensor] = {"target": tgt_loss}

            if source_fn is not None:
                src_loss, _ = _parse_loss(source_fn(model, batch))
                total = total + source_weight * src_loss
                components["source"] = src_loss

            components["total"] = total
            return components

        return combined

    # ------------------------------------------------------------------
    # MMD domain adaptation
    # ------------------------------------------------------------------

    def domain_adaptation_loss(
        self,
        x_source: torch.Tensor,
        x_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the MMD loss between source and target feature distributions.

        Runs a forward pass on both ``x_source`` and ``x_target``, extracts
        intermediate features (from :attr:`feature_layer` if set, otherwise
        uses the model output directly), and returns the squared MMD between
        the two feature sets.

        A low MMD encourages the model to produce similar internal
        representations for both domains, reducing co-variate shift.

        Parameters
        ----------
        x_source:
            Collocation / query points from the **source** domain,
            shape ``(n, d_in)``.
        x_target:
            Collocation / query points from the **target** domain,
            shape ``(m, d_in)``.

        Returns
        -------
        torch.Tensor
            Scalar tensor: the squared MMD between source and target features.
        """
        if self.feature_layer is not None:
            hook = _IntermediateFeatureHook(self.model, self.feature_layer)
            try:
                _ = self.model(x_source)
                phi_s = hook.features
                _ = self.model(x_target)
                phi_t = hook.features
            finally:
                hook.remove()

            if phi_s is None or phi_t is None:
                raise RuntimeError(
                    f"Feature hook on layer {self.feature_layer!r} did not "
                    "capture a tensor output.  Check the layer name and model "
                    "architecture."
                )
        else:
            # Fall back to using model output as features.
            def _to_tensor(out: Any) -> torch.Tensor:
                if isinstance(out, torch.Tensor):
                    return out
                for attr in ("y", "pred", "logits", "out"):
                    if hasattr(out, attr):
                        candidate = getattr(out, attr)
                        if isinstance(candidate, torch.Tensor):
                            return candidate
                raise TypeError(
                    "Model output is not a Tensor and has no known tensor attribute."
                )

            phi_s = _to_tensor(self.model(x_source))
            phi_t = _to_tensor(self.model(x_target))

        # Flatten spatial dimensions if needed: (n, *) -> (n, d).
        phi_s = phi_s.reshape(phi_s.shape[0], -1)
        phi_t = phi_t.reshape(phi_t.shape[0], -1)

        return _mmd_squared(phi_s, phi_t, self.mmd_bandwidth)
