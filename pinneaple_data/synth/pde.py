"""PDE/ODE synthetic generator for heat, advection, and logistic equations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import math
import torch

from .base import SynthConfig, SynthOutput


@dataclass
class SimplePhysicalSample:
    """
    MVP duck-typed PhysicalSample to keep synth module usable standalone.

    This lightweight container mirrors the expected interface of a richer
    `PhysicalSample` class used elsewhere in the codebase.

    Attributes:
        fields: Mapping from field name to tensor (e.g., {"u": (T+1, X)}).
        coords: Mapping from coordinate name to tensor (e.g., {"t": (T+1,), "x": (X,)}).
        meta: Free-form metadata describing the sample (equation params, BCs, etc.).
    """
    fields: Dict[str, torch.Tensor]
    coords: Dict[str, torch.Tensor]
    meta: Dict[str, Any]


class PDESynthGenerator:
    """
    PDE/ODE synthetic generator.

    MVP supports:
      - 1D heat equation: u_t = alpha u_xx
      - 1D advection:     u_t + c u_x = 0
      - ODE logistic:     u' = r u (1-u/K)

    Provide:
      kind: "heat1d" | "advection1d" | "logistic"
      domain/grid/time params
      initial condition function

    Notes:
        - For PDEs, returns trajectories for u(t, x) sampled on a 1D grid.
        - For the logistic ODE, returns u(t) reshaped to (T, 1) for consistency.
        - Boundary condition handling is minimal: "periodic" or clamped "dirichlet0".
    """

    def __init__(self, cfg: Optional[SynthConfig] = None):
        """Create a PDE/ODE synthetic data generator.

        Args:
            cfg: Optional `SynthConfig` controlling device, dtype, and RNG seed.
                If omitted, defaults to `SynthConfig()`.
        """
        self.cfg = cfg or SynthConfig()

    def _rng(self):
        """Create a deterministic CPU RNG based on the configured seed.

        Returns:
            A `torch.Generator` on CPU seeded with `self.cfg.seed`.
        """
        return torch.Generator(device="cpu").manual_seed(int(self.cfg.seed))

    def generate(
        self,
        *,
        kind: str,
        n_samples: int = 16,

        # common
        dt: float = 1e-3,
        steps: int = 200,

        # 1D PDE
        x_min: float = 0.0,
        x_max: float = 1.0,
        nx: int = 128,
        bc: str = "periodic",  # "periodic" | "dirichlet0"

        # params
        alpha: float = 0.01,   # heat
        c: float = 1.0,        # advection
        r: float = 2.0,        # logistic
        K: float = 1.0,        # logistic

        # IC
        ic_fn: Optional[Callable[[torch.Tensor, torch.Generator], torch.Tensor]] = None,
    ) -> SynthOutput:
        """Generate synthetic trajectories for a supported PDE/ODE kind.

        Depending on `kind`, this method simulates:
          - "heat1d": explicit Euler time stepping with central-difference Laplacian
          - "advection1d": explicit Euler time stepping with upwind spatial derivative
          - "logistic": explicit Euler stepping of the logistic ODE

        Args:
            kind: Which system to simulate. One of:
                - "heat1d"
                - "advection1d"
                - "logistic"
            n_samples: Number of independent trajectories to generate.
            dt: Time step size.
            steps: Number of time steps (trajectory length is steps+1).
            x_min: Minimum spatial coordinate (PDE cases only).
            x_max: Maximum spatial coordinate (PDE cases only).
            nx: Number of spatial grid points (PDE cases only).
            bc: Boundary condition mode for PDEs:
                - "periodic" uses wrap-around with `torch.roll`
                - other values are treated as "dirichlet0" (clamped endpoints)
            alpha: Diffusivity for heat equation (used when kind=="heat1d").
            c: Advection speed for advection equation (used when kind=="advection1d").
            r: Growth rate for logistic ODE (used when kind=="logistic").
            K: Carrying capacity for logistic ODE (used when kind=="logistic").
            ic_fn: Optional initial-condition sampler for PDEs:
                `ic_fn(x, rng) -> u0(x)`. If not provided, a random Fourier series is used.

        Returns:
            A `SynthOutput` containing:
              - samples: list of `SimplePhysicalSample` trajectories
              - extras: summary metadata including `n_samples` and `kind`

        Raises:
            ValueError: If `kind` is not one of: heat1d, advection1d, logistic.
        """
        kind = kind.lower().strip()
        rng = self._rng()

        device = torch.device(self.cfg.device)
        dtype = getattr(torch, self.cfg.dtype)

        samples: List[SimplePhysicalSample] = []

        if kind in ("heat1d", "advection1d"):
            x = torch.linspace(x_min, x_max, int(nx), device=device, dtype=dtype)
            dx = (x_max - x_min) / max(nx - 1, 1)

            def default_ic(x: torch.Tensor, g: torch.Generator):
                """Default PDE initial condition: random low-frequency Fourier series.

                Args:
                    x: Spatial grid tensor of shape (nx,).
                    g: RNG used to sample amplitudes and frequencies.

                Returns:
                    Initial condition u0(x) with shape (nx,).
                """
                # random sum of sines
                amps = torch.randn((4,), generator=g, device=x.device, dtype=x.dtype) * 0.3
                freqs = torch.randint(1, 6, (4,), generator=g, device=x.device)
                u0 = torch.zeros_like(x)
                for a, f in zip(amps, freqs):
                    u0 = u0 + a * torch.sin(2 * math.pi * f * (x - x_min) / (x_max - x_min))
                return u0

            ic = ic_fn or default_ic

            for i in range(int(n_samples)):
                u = ic(x, rng)
                traj = [u]

                for _ in range(int(steps)):
                    if kind == "heat1d":
                        # u_xx via central diff
                        if bc == "periodic":
                            u_xx = (torch.roll(u, -1) - 2 * u + torch.roll(u, 1)) / (dx * dx)
                        else:
                            # dirichlet0: clamp boundaries
                            u_pad = u.clone()
                            u_pad[0] = 0.0
                            u_pad[-1] = 0.0
                            u_xx = torch.zeros_like(u_pad)
                            u_xx[1:-1] = (u_pad[2:] - 2 * u_pad[1:-1] + u_pad[:-2]) / (dx * dx)
                        u = u + float(dt) * float(alpha) * u_xx

                    else:
                        # advection: upwind
                        if c >= 0:
                            u_x = (u - torch.roll(u, 1)) / dx if bc == "periodic" else torch.cat([u[:1]*0, (u[1:] - u[:-1])/dx])
                        else:
                            u_x = (torch.roll(u, -1) - u) / dx if bc == "periodic" else torch.cat([(u[1:] - u[:-1])/dx, u[-1:]*0])
                        u = u - float(dt) * float(c) * u_x

                    traj.append(u)

                U = torch.stack(traj, dim=0)  # (T+1, nx)
                t = torch.linspace(0.0, float(dt) * float(steps), steps + 1, device=device, dtype=dtype)

                samples.append(
                    SimplePhysicalSample(
                        fields={"u": U},
                        coords={"t": t, "x": x},
                        meta={
                            "kind": kind,
                            "dt": float(dt),
                            "dx": float(dx),
                            "alpha": float(alpha),
                            "c": float(c),
                            "bc": bc,
                        },
                    )
                )

        elif kind == "logistic":
            # ODE trajectory (T+1,)
            for i in range(int(n_samples)):
                u0 = torch.rand((1,), generator=rng, device=device, dtype=dtype) * 0.9 + 0.05
                u = u0
                traj = [u.squeeze(0)]
                for _ in range(int(steps)):
                    u = u + float(dt) * float(r) * u * (1.0 - u / float(K))
                    traj.append(u.squeeze(0))
                U = torch.stack(traj, dim=0)  # (T+1,)
                t = torch.linspace(0.0, float(dt) * float(steps), steps + 1, device=device, dtype=dtype)

                samples.append(
                    SimplePhysicalSample(
                        fields={"u": U[:, None]},  # (T,1) for consistency
                        coords={"t": t},
                        meta={"kind": kind, "dt": float(dt), "r": float(r), "K": float(K)},
                    )
                )
        else:
            raise ValueError("kind must be one of: heat1d, advection1d, logistic")

        return SynthOutput(samples=samples, extras={"n_samples": len(samples), "kind": kind})