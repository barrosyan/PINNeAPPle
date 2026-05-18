"""Distribution-based synthetic generator for Gaussian, uniform, and mixture-of-Gaussians sampling."""
from __future__ import annotations

from typing import List, Optional, Callable
import torch

from .base import SynthConfig, SynthOutput
from .pde import SimplePhysicalSample


class DistributionSynthGenerator:
    """
    Synthetic generator for sampling inputs from simple probability distributions.

    Supported distributions:
    - "gaussian": i.i.d. normal samples with configurable mean/std.
    - "uniform": i.i.d. uniform samples in [low, high].
    - "mog": mixture of Gaussians (component means/stds, optional weights).

    A supervised target `y` can be produced either by:
    - a default nonlinear mapping y = sum(x^2) (per-sample), or
    - a user-provided function `y_fn(x)`.

    Output is wrapped as a PhysicalSample-like object via `SimplePhysicalSample`,
    with `fields={"x": ..., "y": ...}` and basic metadata.

    Parameters
    ----------
    cfg : Optional[SynthConfig]
        Configuration controlling reproducibility, device, and dtype.
    """
    """
    Generate synthetic samples from distributions:
      - gaussian, uniform
      - mixture of gaussians (simple)
    Output as PhysicalSample-like objects: fields={"x":..., "y":...} etc.
    """
    def __init__(self, cfg: Optional[SynthConfig] = None):
        """
        Initialize the distribution-based synthetic generator.

        Parameters
        ----------
        cfg : Optional[SynthConfig]
            Optional generator configuration. If not provided, defaults are used.
        """
        self.cfg = cfg or SynthConfig()

    def _rng(self):
        """
        Create a deterministic CPU random number generator.

        Returns
        -------
        torch.Generator
            CPU generator seeded with `self.cfg.seed`.
        """
        return torch.Generator(device="cpu").manual_seed(int(self.cfg.seed))

    def generate(
        self,
        *,
        kind: str = "gaussian",
        n_samples: int = 1024,
        dim: int = 2,
        mean: float = 0.0,
        std: float = 1.0,
        low: float = -1.0,
        high: float = 1.0,
        mixture_means: Optional[List[List[float]]] = None,
        mixture_stds: Optional[List[float]] = None,
        mixture_weights: Optional[List[float]] = None,
        y_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> SynthOutput:
        """
        Generate synthetic samples from a specified distribution and compute targets.

        Parameters
        ----------
        kind : str, optional
            Distribution kind: "gaussian", "uniform", or "mog". Default is "gaussian".
        n_samples : int, optional
            Number of samples to draw. Default is 1024.
        dim : int, optional
            Dimensionality of each sample `x`. Default is 2.
        mean : float, optional
            Mean for Gaussian sampling (used when kind="gaussian"). Default is 0.0.
        std : float, optional
            Standard deviation for Gaussian sampling (used when kind="gaussian").
            Default is 1.0.
        low : float, optional
            Lower bound for uniform sampling (used when kind="uniform"). Default is -1.0.
        high : float, optional
            Upper bound for uniform sampling (used when kind="uniform"). Default is 1.0.
        mixture_means : Optional[List[List[float]]], optional
            List of component means for mixture-of-Gaussians sampling (used when kind="mog").
            Shape conceptually (K, dim). Default is None.
        mixture_stds : Optional[List[float]], optional
            List of component standard deviations for mixture-of-Gaussians sampling
            (used when kind="mog"). Length K. Default is None.
        mixture_weights : Optional[List[float]], optional
            Optional mixture weights for components (used when kind="mog"). If None,
            uses uniform weights. Default is None.
        y_fn : Optional[Callable[[torch.Tensor], torch.Tensor]], optional
            Optional function mapping x -> y. If None, uses default y = sum(x^2).
            Default is None.

        Returns
        -------
        SynthOutput
            Output containing one `SimplePhysicalSample` with fields:
            - "x": sampled inputs (n_samples, dim)
            - "y": targets (n_samples, 1) by default, or as returned by y_fn
            and extras including the number of generated points.

        Raises
        ------
        ValueError
            If `kind` is unsupported, or if mixture parameters are missing for "mog".
        """
        kind = kind.lower().strip()
        rng = self._rng()
        device = torch.device(self.cfg.device)
        dtype = getattr(torch, self.cfg.dtype)

        if kind == "gaussian":
            x = torch.randn((n_samples, dim), generator=rng, device=device, dtype=dtype) * float(std) + float(mean)
        elif kind == "uniform":
            x = (high - low) * torch.rand((n_samples, dim), generator=rng, device=device, dtype=dtype) + low
        elif kind == "mog":
            if not mixture_means or not mixture_stds:
                raise ValueError("mixture_means and mixture_stds required for mog")
            K = len(mixture_means)
            w = torch.tensor(mixture_weights or [1.0 / K] * K, device=device, dtype=dtype)
            w = w / w.sum()
            comp = torch.multinomial(w, num_samples=int(n_samples), replacement=True, generator=rng)
            means = torch.tensor(mixture_means, device=device, dtype=dtype)  # (K,dim)
            stds = torch.tensor(mixture_stds, device=device, dtype=dtype).view(K, 1)
            x = torch.randn((n_samples, dim), generator=rng, device=device, dtype=dtype) * stds[comp] + means[comp]
        else:
            raise ValueError("kind must be gaussian | uniform | mog")

        if y_fn is None:
            # default: nonlinear mapping to create supervised target
            y = (x ** 2).sum(dim=-1, keepdim=True)
        else:
            y = y_fn(x)

        sample = SimplePhysicalSample(
            fields={"x": x, "y": y},
            coords={},
            meta={"kind": kind, "dim": int(dim)},
        )
        return SynthOutput(samples=[sample], extras={"n_points": int(n_samples)})