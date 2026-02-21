"""FFT solver for spectrum computation with rFFT/FFT support."""
from __future__ import annotations
from typing import Dict, Optional, Tuple, List

import torch

from .base import SolverBase, SolverOutput


class FFTSolver(SolverBase):
    """
    FFT Solver (PyTorch backend).

    Supports:
      - rFFT / FFT
      - 1D/2D/3D along specified dims
      - returns spectrum (complex split into real/imag) or magnitude/phase

    Typical use:
      - compute spectra of time series or spatial fields
      - build physics features (energy in bands, dominant freq, etc.)
    """
    def __init__(
        self,
        *,
        dims: Optional[Tuple[int, ...]] = None,
        real_input: bool = True,
        output: str = "magnitude_phase",  # "complex_ri" | "magnitude_phase" | "magnitude"
        norm: Optional[str] = None,       # None | "forward" | "backward" | "ortho"
    ):
        super().__init__()
        self.dims = dims
        self.real_input = bool(real_input)
        self.output = str(output).lower().strip()
        self.norm = norm

        if self.output not in {"complex_ri", "magnitude_phase", "magnitude"}:
            raise ValueError("output must be 'complex_ri', 'magnitude_phase', or 'magnitude'")

    def forward(
        self,
        x: torch.Tensor,
        *,
        d: Optional[float] = None,  # sample spacing (e.g., dt); if given, returns freqs
    ) -> SolverOutput:
        # Choose dims
        if self.dims is None:
            # default: last dimension
            dims = (-1,)
        else:
            dims = self.dims

        # FFT
        if self.real_input:
            # rfftn reduces last FFT axis size
            X = torch.fft.rfftn(x, dim=dims, norm=self.norm)
        else:
            X = torch.fft.fftn(x, dim=dims, norm=self.norm)

        extras: Dict[str, object] = {}

        # Frequencies for 1D convenience when dims is single axis
        if d is not None and len(dims) == 1:
            n = x.shape[dims[0]]
            freqs = torch.fft.rfftfreq(n, d=float(d)) if self.real_input else torch.fft.fftfreq(n, d=float(d))
            extras["freqs"] = freqs.to(device=x.device)

        if self.output == "complex_ri":
            out = torch.stack([X.real, X.imag], dim=-1)  # (..., 2)
        elif self.output == "magnitude":
            out = torch.abs(X)
        else:
            mag = torch.abs(X)
            phase = torch.angle(X)
            out = torch.stack([mag, phase], dim=-1)  # (..., 2)

        return SolverOutput(result=out, losses={"total": torch.tensor(0.0, device=x.device)}, extras=extras)
