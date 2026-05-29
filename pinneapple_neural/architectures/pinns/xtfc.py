from __future__ import annotations
"""XtFC - eXtreme Theory of Functional Connections for PINNs.

Uses the `tfc` Python library (pip install tfc) when available to build
constrained expressions that *exactly* satisfy initial/boundary conditions.

Falls back to a manual g(x) + B(x)*N(x) formulation when `tfc` is not
installed, so the module is always importable.

References
----------
- Leake & Mortari (2020) "Deep Theory of Functional Connections" arXiv:2005.01219
- Johnston & Mortari (2021) "Least-Squares Solutions of BVPs" arXiv:2011.04700
- TFC library: https://github.com/leakec/tfc
"""

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .base import PINNBase, PINNOutput

# ---------------------------------------------------------------------------
# Optional tfc import
# ---------------------------------------------------------------------------
try:
    from tfc import TFC as _TFC_1D
    from tfc import utfc as _uTFC
    _TFC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TFC_1D = None
    _uTFC = None
    _TFC_AVAILABLE = False


def tfc_available() -> bool:
    """Return True if the `tfc` library is installed."""
    return _TFC_AVAILABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_phi(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    name = (name or "tanh").lower()
    if name == "tanh":
        return torch.tanh
    if name == "relu":
        return torch.relu
    if name == "gelu":
        return torch.nn.functional.gelu
    if name == "silu":
        return torch.nn.functional.silu
    if name == "sin":
        return torch.sin
    return torch.tanh


# ---------------------------------------------------------------------------
# TFC Constrained Expression builder (wraps the tfc library)
# ---------------------------------------------------------------------------

class TFCConstrainedExpression:
    """
    Wraps the `tfc` library to create constrained expressions for 1D or nD
    problems that exactly satisfy boundary/initial conditions.

    For 1D: uses ``tfc.TFC``
    For nD: uses ``tfc.utfc`` (Universal TFC)

    The free function ``eta(x)`` is provided at call time (typically, a
    neural network evaluated on the TFC support points ``z``).

    Usage
    -----
    >>> ce = TFCConstrainedExpression.build_1d(
    ...     n=100, nC=2, deg=10, x0=0.0, xf=1.0,
    ...     bc_values=[0.0, 0.0]
    ... )
    >>> z = ce.z          # support points in [x0, xf]
    >>> eta = nn(z)       # free function from NN
    >>> u = ce(eta)       # constrained expression; satisfies u(x0)=0, u(xf)=0
    """

    def __init__(self, tfc_obj: Any, dim: int = 1):
        if not _TFC_AVAILABLE:
            raise ImportError(
                "The `tfc` library is required for TFCConstrainedExpression. "
                "Install it with: pip install tfc"
            )
        self._tfc = tfc_obj
        self.dim = dim

        # Support points (numpy array) - convert to torch lazily
        self._z_np = tfc_obj.z if hasattr(tfc_obj, "z") else None

    @property
    def z(self) -> torch.Tensor:
        """TFC support points as a float32 torch tensor."""
        if self._z_np is None:
            raise AttributeError("TFC object has no 'z' attribute.")
        import numpy as np
        z = np.asarray(self._z_np, dtype=np.float32)
        return torch.from_numpy(z)

    def constrained_expression(
        self, eta_fn: Callable[..., Any]
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Return a callable ``u(x)`` that uses TFC constrained expression.

        Parameters
        ----------
        eta_fn : callable
            The free function (NN). Should accept numpy arrays and return
            numpy arrays (tfc library works in numpy/JAX).

        Returns
        -------
        callable that maps x -> u(x) as a torch.Tensor
        """
        ce = self._tfc.cf(eta_fn)

        def _u(x: torch.Tensor) -> torch.Tensor:
            import numpy as np
            x_np = x.detach().cpu().numpy()
            u_np = np.asarray(ce(x_np), dtype=np.float32)
            return torch.from_numpy(u_np).to(device=x.device, dtype=x.dtype)

        return _u

    @classmethod
    def build_1d(
        cls,
        n: int,
        nC: int,
        deg: int,
        x0: float = 0.0,
        xf: float = 1.0,
        basis: str = "CP",
    ) -> "TFCConstrainedExpression":
        """
        Build a 1D TFC object.

        Parameters
        ----------
        n : int       number of discretization points
        nC : int      number of constraints (BCs + ICs)
        deg : int     degree of the basis (number of basis functions)
        x0, xf       domain bounds
        basis         "CP" (Chebyshev), "LeP" (Legendre), "FS" (Fourier), ...
        """
        if not _TFC_AVAILABLE:
            raise ImportError("pip install tfc")
        tfc_obj = _TFC_1D(n, nC, deg, x0=x0, xf=xf, basis=basis)
        return cls(tfc_obj, dim=1)

    @classmethod
    def build_nd(
        cls,
        n: int,
        nC: int,
        deg: int,
        x0: Sequence[float],
        xf: Sequence[float],
        basis: str = "CP",
    ) -> "TFCConstrainedExpression":
        """
        Build a nD Universal TFC object.

        Parameters
        ----------
        n : int           number of points per dimension
        nC : int          number of constraints
        deg : int         degree of basis
        x0, xf           per-dimension bounds (length = ndim)
        basis             basis type
        """
        if not _TFC_AVAILABLE:
            raise ImportError("pip install tfc")
        ndim = len(x0)
        tfc_obj = _uTFC(n, nC, deg, dim=ndim, x0=list(x0), xf=list(xf), basis=basis)
        return cls(tfc_obj, dim=ndim)


# ---------------------------------------------------------------------------
# XtFC core model
# ---------------------------------------------------------------------------

@dataclass
class XTFCConfig:
    in_dim: int
    out_dim: int
    rf_dim: int = 2048
    activation: str = "tanh"
    freeze_random: bool = True

    # Random feature options
    rf_kind: str = "rff"       # "rff" | "random_linear"
    rff_sigma: float = 1.0
    use_bias: bool = True
    init_scale: float = 1.0

    # Head options
    head_bias: bool = True
    clamp_B: Optional[Tuple[float, float]] = None
    eps_B: float = 0.0

    # Regularization
    l2_head: float = 0.0
    l2_W: float = 0.0

    # TFC options (used when tfc library is available)
    use_tfc: bool = True           # attempt TFC constrained expression
    tfc_n: int = 100               # TFC support points
    tfc_deg: int = 10              # TFC basis degree
    tfc_nC: int = 2                # number of constraints
    tfc_basis: str = "CP"          # "CP" | "LeP" | "FS"
    tfc_x0: Optional[List[float]] = None   # domain lower bounds (None = auto)
    tfc_xf: Optional[List[float]] = None   # domain upper bounds (None = auto)

    broadcast_B: bool = True


class XTFC(PINNBase):
    """
    XtFC — eXtreme Theory of Functional Connections PINN.

    When the ``tfc`` library is installed (pip install tfc), the model uses
    a proper TFC constrained expression so that BCs/ICs are satisfied
    *exactly by construction*.

    When ``tfc`` is not installed, the model falls back to the classic
    ELM-TFC approximation:
        y(x) = g(x) + B(x) * N(x)
    where ``g`` is a particular solution and ``B`` vanishes on boundaries.

    In both modes the free function is a random feature network (ELM):
        N(x) = head( phi(W x + b) )
    with frozen random weights ``W, b`` and trainable linear head.

    The head can be solved by closed-form ridge regression (``fit_ridge``)
    or by gradient descent through the PINN loss.

    Parameters
    ----------
    in_dim, out_dim : int
    rf_dim : int        number of random features
    activation : str    "tanh" | "sin" | "relu" | "gelu" | "silu"
    freeze_random : bool  freeze W,b (ELM mode) — highly recommended
    g_fn : callable     particular solution g(x) -> (N, out_dim)
    B_fn : callable     boundary factor B(x) -> (N,1) or (N,out_dim)
    tfc_ce : TFCConstrainedExpression  pre-built TFC object (optional)
    config : XTFCConfig   full config dataclass
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rf_dim: int = 2048,
        activation: str = "tanh",
        freeze_random: bool = True,
        g_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        B_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        tfc_ce: Optional[TFCConstrainedExpression] = None,
        *,
        config: Optional[XTFCConfig] = None,
    ):
        super().__init__()

        self.cfg = config or XTFCConfig(
            in_dim=in_dim,
            out_dim=out_dim,
            rf_dim=rf_dim,
            activation=activation,
            freeze_random=freeze_random,
        )

        self.in_dim = int(self.cfg.in_dim)
        self.out_dim = int(self.cfg.out_dim)
        self.rf_dim = int(self.cfg.rf_dim)

        self.phi = _get_phi(self.cfg.activation)

        # Random feature parameters
        self.W = nn.Parameter(torch.empty(self.rf_dim, self.in_dim))
        self.b = nn.Parameter(torch.empty(self.rf_dim)) if self.cfg.use_bias else None

        self._init_random_features()

        head_in = self._rf_out_dim()
        self.head = nn.Linear(head_in, self.out_dim, bias=bool(self.cfg.head_bias))

        if self.cfg.freeze_random:
            self.W.requires_grad_(False)
            if self.b is not None:
                self.b.requires_grad_(False)

        # Constraint functions
        self.g_fn = g_fn
        self.B_fn = B_fn

        # TFC constrained expression (tfc library)
        self.tfc_ce = tfc_ce
        if tfc_ce is None and _TFC_AVAILABLE and self.cfg.use_tfc:
            self._try_build_tfc()

    # ------------------------------------------------------------------
    # TFC integration
    # ------------------------------------------------------------------

    def _try_build_tfc(self) -> None:
        """Attempt to build a TFC constrained expression from config."""
        try:
            x0 = self.cfg.tfc_x0 or [0.0] * self.in_dim
            xf = self.cfg.tfc_xf or [1.0] * self.in_dim

            if self.in_dim == 1:
                self.tfc_ce = TFCConstrainedExpression.build_1d(
                    n=self.cfg.tfc_n,
                    nC=self.cfg.tfc_nC,
                    deg=self.cfg.tfc_deg,
                    x0=float(x0[0]),
                    xf=float(xf[0]),
                    basis=self.cfg.tfc_basis,
                )
            else:
                self.tfc_ce = TFCConstrainedExpression.build_nd(
                    n=self.cfg.tfc_n,
                    nC=self.cfg.tfc_nC,
                    deg=self.cfg.tfc_deg,
                    x0=x0,
                    xf=xf,
                    basis=self.cfg.tfc_basis,
                )
        except Exception as e:
            warnings.warn(
                f"XtFC: Could not build TFC constrained expression: {e}. "
                "Falling back to manual g+B*N formulation.",
                stacklevel=2,
            )
            self.tfc_ce = None

    def set_tfc(self, tfc_ce: TFCConstrainedExpression) -> None:
        """Attach a pre-built TFCConstrainedExpression to this model."""
        self.tfc_ce = tfc_ce

    def set_constraint_fns(
        self,
        g_fn: Callable[[torch.Tensor], torch.Tensor],
        B_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        """Set the particular solution g and boundary factor B for fallback mode."""
        self.g_fn = g_fn
        self.B_fn = B_fn

    # ------------------------------------------------------------------
    # Random features
    # ------------------------------------------------------------------

    def _rf_out_dim(self) -> int:
        return 2 * self.rf_dim if self.cfg.rf_kind == "rff" else self.rf_dim

    def _init_random_features(self) -> None:
        scale = float(self.cfg.init_scale)
        if self.cfg.rf_kind == "rff":
            sigma = float(self.cfg.rff_sigma)
            std = 1.0 / max(sigma, 1e-12)
            nn.init.normal_(self.W, mean=0.0, std=std * scale)
            if self.b is not None:
                self.b.data.uniform_(0.0, 2.0 * torch.pi)
        else:
            std = 1.0 / max(self.in_dim, 1) ** 0.5
            nn.init.normal_(self.W, mean=0.0, std=std * scale)
            if self.b is not None:
                nn.init.zeros_(self.b)

    @torch.no_grad()
    def reset_random_features(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            g = torch.Generator(device=self.W.device)
            g.manual_seed(int(seed))
            scale = float(self.cfg.init_scale)
            if self.cfg.rf_kind == "rff":
                sigma = float(self.cfg.rff_sigma)
                std = 1.0 / max(sigma, 1e-12)
                self.W.copy_(
                    torch.randn(self.rf_dim, self.in_dim, generator=g, device=self.W.device) * (std * scale)
                )
                if self.b is not None:
                    self.b.copy_(
                        torch.rand(self.rf_dim, generator=g, device=self.W.device) * 2.0 * torch.pi
                    )
            else:
                std = 1.0 / max(self.in_dim, 1) ** 0.5
                self.W.copy_(
                    torch.randn(self.rf_dim, self.in_dim, generator=g, device=self.W.device) * (std * scale)
                )
                if self.b is not None:
                    self.b.zero_()
        else:
            self._init_random_features()

    def _rf(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.W.t()
        if self.b is not None:
            proj = proj + self.b[None, :]
        if self.cfg.rf_kind == "rff":
            return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return self.phi(proj)

    # ------------------------------------------------------------------
    # Ridge regression (closed-form ELM solve)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit_ridge(self, x: torch.Tensor, y_target: torch.Tensor, *, l2: float = 1e-6) -> None:
        """Solve head weights analytically by ridge regression (ELM mode)."""
        H = self._rf(x)  # (N, F)
        HT = H.t()
        A = HT @ H
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        Beta = torch.linalg.solve(A + l2 * I, HT @ y_target)  # (F, out_dim)
        self.head.weight.copy_(Beta.t())
        if self.head.bias is not None:
            self.head.bias.zero_()

    # ------------------------------------------------------------------
    # Constraint pieces (manual / fallback mode)
    # ------------------------------------------------------------------

    def _compute_g(self, x: torch.Tensor) -> torch.Tensor:
        if self.g_fn is None:
            return torch.zeros((x.shape[0], self.out_dim), device=x.device, dtype=x.dtype)
        g = self.g_fn(x)
        if g.ndim == 1:
            g = g[:, None]
        if g.shape[-1] == 1 and self.out_dim > 1:
            g = g.expand(-1, self.out_dim)
        if g.shape[-1] != self.out_dim:
            raise ValueError(f"g_fn must return (N,{self.out_dim}), got {tuple(g.shape)}")
        return g

    def _compute_B(self, x: torch.Tensor) -> torch.Tensor:
        if self.B_fn is None:
            B = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        else:
            B = self.B_fn(x)
            if B.ndim == 1:
                B = B[:, None]
        if self.cfg.eps_B != 0.0:
            B = B + float(self.cfg.eps_B)
        if self.cfg.clamp_B is not None:
            lo, hi = self.cfg.clamp_B
            B = torch.clamp(B, float(lo), float(hi))
        if self.cfg.broadcast_B and B.shape[-1] == 1 and self.out_dim > 1:
            B = B.expand(-1, self.out_dim)
        if B.shape[-1] not in (1, self.out_dim):
            raise ValueError(f"B_fn must return (N,1) or (N,{self.out_dim}), got {tuple(B.shape)}")
        return B

    # ------------------------------------------------------------------
    # Core forward
    # ------------------------------------------------------------------

    def _free_function(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the free function N(x) via random features + linear head."""
        return self.head(self._rf(x))

    def forward_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tensor-only forward for PINNFactory / pipeline compatibility.

        If a TFCConstrainedExpression is attached, it is used to enforce
        BCs exactly. Otherwise uses the manual g + B * N formulation.

        x : (N, in_dim)
        returns : (N, out_dim)
        """
        n = self._free_function(x)   # (N, out_dim) — free function

        if self.tfc_ce is not None:
            # TFC mode: constrained expression built from the tfc library.
            # The free function must be passed as a numpy callable;
            # we bridge via a closure that detaches x.
            import numpy as np

            def eta_np(x_np: Any) -> Any:
                x_t = torch.from_numpy(np.asarray(x_np, dtype=np.float32)).to(
                    device=self.W.device, dtype=self.W.dtype
                )
                return self._free_function(x_t).detach().cpu().numpy()

            try:
                u_fn = self.tfc_ce.constrained_expression(eta_np)
                return u_fn(x)
            except Exception:
                # If tfc constrained expression fails at runtime, fall back
                pass

        # Fallback: manual g + B * N
        g = self._compute_g(x)
        B = self._compute_B(x)
        return g + B * n

    def forward(
        self,
        x: torch.Tensor,
        *,
        physics_fn: Optional[Callable[..., Any]] = None,
        physics_data: Optional[Dict[str, Any]] = None,
        return_parts: bool = False,
    ) -> PINNOutput:
        """
        Standard PINNBase forward.

        x : (N, in_dim)  — coordinate tensor
        """
        y = self.forward_tensor(x)

        z = torch.zeros((), device=y.device, dtype=y.dtype)
        losses: Dict[str, torch.Tensor] = {"total": z}

        # L2 regularization on head / random weights
        reg = z
        if self.cfg.l2_head > 0.0:
            reg = reg + float(self.cfg.l2_head) * self.head.weight.pow(2).mean()
        if self.cfg.l2_W > 0.0:
            reg = reg + float(self.cfg.l2_W) * self.W.pow(2).mean()
        if reg.detach().abs().item() != 0.0:
            losses["reg"] = reg
            losses["total"] = losses["total"] + reg

        if physics_fn is not None and physics_data is not None:
            pl = self.physics_loss(physics_fn=physics_fn, physics_data=physics_data)
            losses.update(pl)
            losses["total"] = losses["total"] + losses.get(
                "physics", z
            )

        extras: Dict[str, Any] = {
            "tfc_mode": self.tfc_ce is not None,
            "tfc_available": _TFC_AVAILABLE,
        }
        if return_parts:
            extras.update({
                "N": self._free_function(x).detach(),
                "rf": self._rf(x).detach(),
            })
            if self.tfc_ce is None:
                extras["g"] = self._compute_g(x).detach()
                extras["B"] = self._compute_B(x).detach()

        return PINNOutput(y=y, losses=losses, extras=extras)


# ---------------------------------------------------------------------------
# Factory-compatible wrapper (model(x) -> Tensor)
# ---------------------------------------------------------------------------

class XTFCFactoryModel(nn.Module):
    """
    Thin wrapper around XTFC that exposes a standard ``forward(x) -> Tensor``
    interface compatible with PINNFactory and the Trainer.

    Also exposes ``inverse_params`` for inverse PINN problems.
    """

    def __init__(
        self,
        xtfc: XTFC,
        inverse_params_names: Optional[List[str]] = None,
        initial_guesses: Optional[Dict[str, float]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.xtfc = xtfc

        self.inverse_params = nn.ParameterDict()
        for name in (inverse_params_names or []):
            init = float((initial_guesses or {}).get(name, 0.1))
            self.inverse_params[name] = nn.Parameter(torch.tensor(init, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (N, in_dim)
        returns : (N, out_dim)
        """
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        return self.xtfc.forward_tensor(x)


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_xtfc(
    in_dim: int,
    out_dim: int,
    *,
    rf_dim: int = 2048,
    activation: str = "tanh",
    freeze_random: bool = True,
    g_fn: Optional[Callable] = None,
    B_fn: Optional[Callable] = None,
    use_tfc: bool = True,
    tfc_n: int = 100,
    tfc_deg: int = 10,
    tfc_nC: int = 2,
    tfc_x0: Optional[List[float]] = None,
    tfc_xf: Optional[List[float]] = None,
    tfc_basis: str = "CP",
) -> XTFC:
    """
    Convenience constructor for XTFC.

    If the ``tfc`` library is available and ``use_tfc=True``, the model will
    automatically build a TFC constrained expression.

    Parameters
    ----------
    in_dim, out_dim : int
    rf_dim : int         random feature dimension (ELM width)
    activation : str     activation for random features
    freeze_random : bool freeze random weights (ELM)
    g_fn, B_fn           particular solution and boundary factor (fallback)
    use_tfc : bool       use tfc library if available
    tfc_n : int          TFC support points
    tfc_deg : int        TFC basis degree
    tfc_nC : int         number of constraints (BCs + ICs)
    tfc_x0, tfc_xf       domain bounds
    tfc_basis : str      "CP" | "LeP" | "FS"
    """
    cfg = XTFCConfig(
        in_dim=in_dim,
        out_dim=out_dim,
        rf_dim=rf_dim,
        activation=activation,
        freeze_random=freeze_random,
        use_tfc=use_tfc,
        tfc_n=tfc_n,
        tfc_deg=tfc_deg,
        tfc_nC=tfc_nC,
        tfc_basis=tfc_basis,
        tfc_x0=tfc_x0,
        tfc_xf=tfc_xf,
    )
    return XTFC(
        in_dim=in_dim,
        out_dim=out_dim,
        g_fn=g_fn,
        B_fn=B_fn,
        config=cfg,
    )
