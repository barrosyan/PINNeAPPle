"""Curve-fit synthetic generator for polynomial ridge regression and sample synthesis."""
from __future__ import annotations

from typing import Optional, Tuple
import torch

from .base import SynthConfig, SynthOutput
from .pde import SimplePhysicalSample


def _poly_features(x: torch.Tensor, degree: int) -> torch.Tensor:
    """
    Construct polynomial feature expansions for 1D inputs.

    Given an input tensor `x` of shape (N, 1), this builds a design matrix
    of shape (N, degree + 1) with columns:
        [1, x, x^2, ..., x^degree].

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (N, 1).
    degree : int
        Maximum polynomial degree to include.

    Returns
    -------
    torch.Tensor
        Polynomial feature matrix of shape (N, degree + 1).
    """
    # x: (N,1) -> (N, degree+1) [1, x, x^2, ...]
    feats = [torch.ones_like(x)]
    for d in range(1, degree + 1):
        feats.append(x ** d)
    return torch.cat(feats, dim=1)


class CurveFitSynthGenerator:
    """
    Synthetic generator that fits a polynomial trend to 1D data and synthesizes samples.

    This generator performs a simple polynomial regression with ridge regularization
    on provided (x, y) pairs, then generates a new set of x locations and produces
    corresponding y predictions. Optionally, missing values can be ignored during
    fitting and Gaussian noise can be added to synthesized outputs.

    Notes
    -----
    - Assumes `x` and `y` are shaped (N, 1).
    - Uses a closed-form ridge solution via `torch.linalg.solve`.
    - Outputs a `SimplePhysicalSample` containing synthesized x/y and learned weights.

    Parameters
    ----------
    cfg : Optional[SynthConfig]
        Configuration controlling device/dtype/seed behavior.
    """

    def __init__(self, cfg: Optional[SynthConfig] = None):
        """
        Initialize the curve-fit synthetic generator.

        Parameters
        ----------
        cfg : Optional[SynthConfig]
            Optional generator configuration. If not provided, defaults are used.
        """
        self.cfg = cfg or SynthConfig()

    def generate(
        self,
        *,
        x: torch.Tensor,            # (N,1)
        y: torch.Tensor,            # (N,1)
        degree: int = 3,
        ridge: float = 1e-6,
        n_new: int = 1024,
        x_range: Optional[Tuple[float, float]] = None,
        noise_std: float = 0.0,
        mask_missing: Optional[torch.Tensor] = None,  # (N,1) bool; True means missing
    ) -> SynthOutput:
        """
        Fit a polynomial ridge regression model and synthesize a new dataset.

        Parameters
        ----------
        x : torch.Tensor
            Input coordinates of shape (N, 1).
        y : torch.Tensor
            Target values of shape (N, 1).
        degree : int, optional
            Polynomial degree for the regression model. Default is 3.
        ridge : float, optional
            Ridge regularization strength (λ). Default is 1e-6.
        n_new : int, optional
            Number of new x points to synthesize. Default is 1024.
        x_range : Optional[Tuple[float, float]], optional
            If provided, defines (lo, hi) range for synthesized x. If None, uses
            the min/max of the provided `x`. Default is None.
        noise_std : float, optional
            Standard deviation of optional Gaussian noise added to synthesized y.
            Default is 0.0 (no noise).
        mask_missing : Optional[torch.Tensor], optional
            Boolean mask of shape (N, 1) where True indicates missing entries
            that should be excluded from fitting. Default is None.

        Returns
        -------
        SynthOutput
            Output containing one `SimplePhysicalSample` with fields:
            - "x": synthesized x values (n_new, 1)
            - "y": synthesized y values (n_new, 1)
            - "w": learned regression weights (degree+1, 1)
            and extras including the learned weights shape.
        """
        device = torch.device(self.cfg.device)
        dtype = getattr(torch, self.cfg.dtype)
        x = x.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        if mask_missing is not None:
            m = (~mask_missing).view(-1)
            x_fit = x[m]
            y_fit = y[m]
        else:
            x_fit, y_fit = x, y

        Phi = _poly_features(x_fit, int(degree))  # (N,D)
        # ridge solve: (Phi^T Phi + λI) w = Phi^T y
        D = Phi.shape[1]
        I = torch.eye(D, device=device, dtype=dtype)
        w = torch.linalg.solve(Phi.t() @ Phi + float(ridge) * I, Phi.t() @ y_fit)

        # generate new x
        if x_range is None:
            lo = float(x.min().item())
            hi = float(x.max().item())
        else:
            lo, hi = float(x_range[0]), float(x_range[1])

        x_new = torch.linspace(lo, hi, int(n_new), device=device, dtype=dtype).view(-1, 1)
        Phi_new = _poly_features(x_new, int(degree))
        y_new = Phi_new @ w

        if noise_std and noise_std > 0:
            y_new = y_new + noise_std * torch.randn_like(y_new)

        sample = SimplePhysicalSample(
            fields={"x": x_new, "y": y_new, "w": w},
            coords={},
            meta={"degree": int(degree), "ridge": float(ridge), "source": "curvefit"},
        )
        return SynthOutput(samples=[sample], extras={"weights_shape": tuple(w.shape)})