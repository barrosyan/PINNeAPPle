"""Missing term identification for physics-informed models.

Given observations of a physical system and a *partially-known* governing
equation, these tools identify what is missing from the PDE.

Three complementary strategies:

  CandidateLibrary      — builds a matrix of symbolic candidate terms (monomials,
                          trig, user-defined) from collocation data.

  SINDyIdentifier       — sparse regression (STRidge / LASSO) over a candidate
                          library; returns the symbolic form of the missing term.
                          Based on Brunton et al. (2016) "Discovering governing
                          equations from data".

  ResidualAnalyzer      — evaluates the "known" PDE residual of a trained PINN;
                          nonzero residuals encode the missing physics.
                          Optionally runs SINDy on the residual field.

  NeuralTermDiscovery   — jointly trains the PINN and a small MLP τ(x; φ) that
                          parametrises the unknown term. After training τ can be
                          queried at any point and optionally distilled by SINDy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Candidate library
# ---------------------------------------------------------------------------

class CandidateLibrary:
    """Build a matrix Θ of candidate terms from collocation data.

    Θ ∈ R^{N × L} where each column is a candidate term evaluated at N points.
    Used as the regression target in SINDy: Θ ξ ≈ b (missing term).

    Parameters
    ----------
    poly_order : highest polynomial degree (0 = constant, 1 = linear, 2 = quadratic …)
    include_trig : add sin(x₀), cos(x₀), sin(x₁), cos(x₁) columns
    custom_terms : extra (name, callable) pairs where callable: (N,d) → (N,)

    Examples
    --------
    >>> lib = CandidateLibrary(poly_order=2, include_trig=True)
    >>> Theta, names = lib.build(X)   # X shape (N, d)
    """

    def __init__(
        self,
        poly_order: int = 2,
        include_trig: bool = False,
        custom_terms: Optional[List[Tuple[str, Callable]]] = None,
    ) -> None:
        self.poly_order = poly_order
        self.include_trig = include_trig
        self.custom_terms = custom_terms or []

    def build(self, X: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Build the library matrix and term name list.

        Parameters
        ----------
        X : (N, d) input coordinates

        Returns
        -------
        Theta : (N, L)
        names : list of L strings
        """
        N, d = X.shape
        cols: List[np.ndarray] = []
        names: List[str] = []

        cols.append(np.ones(N))
        names.append("1")

        for degree in range(1, self.poly_order + 1):
            for idxs in combinations_with_replacement(range(d), degree):
                cols.append(np.prod(X[:, list(idxs)], axis=1))
                names.append(" ".join(f"x{i}" for i in idxs))

        if self.include_trig:
            for i in range(min(2, d)):
                cols.append(np.sin(X[:, i]))
                names.append(f"sin(x{i})")
                cols.append(np.cos(X[:, i]))
                names.append(f"cos(x{i})")

        for name, fn in self.custom_terms:
            cols.append(np.asarray(fn(X)).ravel())
            names.append(name)

        return np.column_stack(cols), names


# ---------------------------------------------------------------------------
# SINDy sparse regression
# ---------------------------------------------------------------------------

@dataclass
class SINDyResult:
    """Result of a SINDy sparse regression."""
    coefficients: np.ndarray        # (L,) full coefficient vector
    active_terms: List[str]         # names of nonzero-coefficient terms
    active_coefficients: np.ndarray # values of nonzero coefficients
    residual_norm: float            # ‖b - Θ ξ‖₂
    n_terms_selected: int

    def equation(self, lhs: str = "missing_term") -> str:
        """Format the discovered equation as a readable string."""
        if not self.active_terms:
            return f"{lhs} = 0"
        parts = [
            f"{c:+.4g} * {t}" for c, t in zip(self.active_coefficients, self.active_terms)
        ]
        return f"{lhs} = " + "  ".join(parts)


class SINDyIdentifier:
    """Identify missing PDE terms via sparse regression (Brunton et al., 2016).

    Solves  Θ ξ ≈ b  where b is the observed residual (e.g. from ResidualAnalyzer)
    using Sequential Threshold Ridge Regression (STRidge), LASSO, or plain ridge.

    Parameters
    ----------
    threshold : sparsity threshold — coefficients below this are set to zero
    max_iter : STRidge iterations
    alpha : regularisation strength (ridge / LASSO)
    method : "stridge" | "lasso" | "ridge"
    """

    def __init__(
        self,
        threshold: float = 1e-3,
        max_iter: int = 20,
        alpha: float = 1e-5,
        method: str = "stridge",
    ) -> None:
        self.threshold = threshold
        self.max_iter = max_iter
        self.alpha = alpha
        self.method = method

    def fit(
        self,
        Theta: np.ndarray,
        b: np.ndarray,
        term_names: Optional[List[str]] = None,
    ) -> SINDyResult:
        """Fit sparse coefficients.

        Parameters
        ----------
        Theta : (N, L) candidate library matrix
        b : (N,) target residual vector
        term_names : optional list of L labels; auto-generated if None

        Returns
        -------
        SINDyResult
        """
        L = Theta.shape[1]
        if term_names is None:
            term_names = [f"term_{i}" for i in range(L)]

        if self.method == "stridge":
            xi = self._stridge(Theta, b)
        elif self.method == "lasso":
            try:
                from sklearn.linear_model import Lasso
            except ImportError:
                raise ImportError("scikit-learn is required for LASSO: pip install scikit-learn")
            model = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=10_000)
            model.fit(Theta, b)
            xi = model.coef_
        else:
            # ridge via normal equations
            A = Theta.T @ Theta + self.alpha * np.eye(L)
            xi = np.linalg.solve(A, Theta.T @ b)

        active = np.abs(xi) > self.threshold
        return SINDyResult(
            coefficients=xi,
            active_terms=[term_names[i] for i in np.where(active)[0]],
            active_coefficients=xi[active],
            residual_norm=float(np.linalg.norm(b - Theta @ xi)),
            n_terms_selected=int(active.sum()),
        )

    def _stridge(self, Theta: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Sequential Threshold Ridge Regression (STRidge)."""
        xi = np.linalg.lstsq(Theta, b, rcond=None)[0]
        for _ in range(self.max_iter):
            small = np.abs(xi) < self.threshold
            if small.all():
                break
            xi[small] = 0.0
            active = ~small
            if not active.any():
                break
            xi[active] = np.linalg.lstsq(Theta[:, active], b, rcond=None)[0]
        return xi


# ---------------------------------------------------------------------------
# Residual analyser
# ---------------------------------------------------------------------------

@dataclass
class ResidualAnalysisResult:
    """Output of ResidualAnalyzer.analyze()."""
    residual_field: np.ndarray   # (N,) PDE residual at collocation points
    sindy_result: Optional[SINDyResult]
    coords: np.ndarray           # (N, d)
    mean_abs_residual: float
    max_abs_residual: float

    def summary(self) -> str:
        lines = [
            f"Mean |residual|: {self.mean_abs_residual:.4e}",
            f"Max  |residual|: {self.max_abs_residual:.4e}",
        ]
        if self.sindy_result is not None:
            lines.append(f"SINDy: {self.sindy_result.equation()}")
        return "\n".join(lines)


class ResidualAnalyzer:
    """Identify missing PDE terms from the residual field of a trained PINN.

    The idea: if the governing equation is `Lu = f` but the *true* equation is
    `Lu = f + g(x)`, then after training the PINN to fit data, evaluating
    `Lu_pred - f` at collocation points yields an approximation to the missing
    term `g(x)`. SINDy can then discover its symbolic form.

    Parameters
    ----------
    pde_residual_fn : Callable (model, x_tensor) → tensor (N,)
        Evaluates the *known* (partial) PDE residual at x.
        Should return zero everywhere if the PDE is complete.
    library : optional CandidateLibrary for SINDy post-processing
    identifier : optional SINDyIdentifier
    """

    def __init__(
        self,
        pde_residual_fn: Callable,
        library: Optional[CandidateLibrary] = None,
        identifier: Optional[SINDyIdentifier] = None,
    ) -> None:
        self.pde_residual_fn = pde_residual_fn
        self.library = library
        self.identifier = identifier

    def analyze(
        self,
        model: Any,
        coords: np.ndarray,
        *,
        device: str = "cpu",
    ) -> ResidualAnalysisResult:
        """Compute the residual field and optionally run SINDy.

        Parameters
        ----------
        model : trained nn.Module
        coords : (N, d) collocation points
        device : torch device string

        Returns
        -------
        ResidualAnalysisResult
        """
        import torch

        x_t = torch.tensor(coords, dtype=torch.float32, device=device)
        x_t.requires_grad_(True)
        res_t = self.pde_residual_fn(model, x_t)
        residual = res_t.detach().cpu().numpy().ravel()

        sindy_result = None
        if self.library is not None and self.identifier is not None:
            Theta, names = self.library.build(coords)
            sindy_result = self.identifier.fit(Theta, residual, term_names=names)

        return ResidualAnalysisResult(
            residual_field=residual,
            sindy_result=sindy_result,
            coords=coords,
            mean_abs_residual=float(np.mean(np.abs(residual))),
            max_abs_residual=float(np.max(np.abs(residual))),
        )


# ---------------------------------------------------------------------------
# Neural term discovery
# ---------------------------------------------------------------------------

@dataclass
class NeuralTermConfig:
    """Configuration for joint PINN + neural term training.

    Parameters
    ----------
    hidden_dims : hidden layer widths of the term network τ(x; φ)
    activation : activation function for τ
    term_weight : weight of the missing-term loss relative to data loss
    data_weight : weight of the data/supervised loss
    n_iters : total optimisation steps
    lr : Adam learning rate
    device : torch device
    """
    hidden_dims: List[int] = field(default_factory=lambda: [32, 32])
    activation: str = "tanh"
    term_weight: float = 1.0
    data_weight: float = 1.0
    n_iters: int = 5000
    lr: float = 1e-3
    device: str = "cpu"


class NeuralTermDiscovery:
    """Discover a missing PDE term parametrised as a neural network.

    Jointly trains:
      - the PINN (forward model) to fit observations
      - a small MLP τ(x; φ) representing the unknown term

    Combined loss:
        L = w_data * L_data(u_pred, y) + w_term * ‖PDE_residual(u_pred) − τ(x)‖²

    After training, `predict_term(x)` approximates the missing source/reaction/
    forcing term at any query point. The network can be optionally distilled into
    a symbolic form with SINDyIdentifier.

    Parameters
    ----------
    known_residual_fn : Callable (model, x_tensor) → tensor (N,)
        The "known" PDE residual. At the true solution this equals τ(x).
    config : NeuralTermConfig
    """

    def __init__(
        self,
        known_residual_fn: Callable,
        config: Optional[NeuralTermConfig] = None,
    ) -> None:
        self.known_residual_fn = known_residual_fn
        self.config = config or NeuralTermConfig()
        self._term_net = None

    # ------------------------------------------------------------------

    def _build_term_net(self, in_dim: int) -> "torch.nn.Module":
        import torch.nn as nn
        acts: Dict[str, Any] = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}
        Act = acts.get(self.config.activation, nn.Tanh)
        dims = [in_dim] + self.config.hidden_dims + [1]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.extend([nn.Linear(a, b), Act()])
        layers.pop()  # remove trailing activation on output
        return nn.Sequential(*layers)

    def fit(
        self,
        pinn_model: Any,
        x_data: np.ndarray,
        y_data: np.ndarray,
    ) -> "NeuralTermDiscovery":
        """Joint training of the PINN and the term network.

        Parameters
        ----------
        pinn_model : nn.Module (forward pass must support autograd)
        x_data : (N, d) observation coordinates
        y_data : (N, F) observed field values

        Returns self for chaining.
        """
        import torch
        import torch.nn as nn

        cfg = self.config
        dev = torch.device(cfg.device)
        self._term_net = self._build_term_net(x_data.shape[1]).to(dev)
        pinn_model = pinn_model.to(dev)

        x_t = torch.tensor(x_data, dtype=torch.float32, device=dev)
        y_t = torch.tensor(y_data, dtype=torch.float32, device=dev)

        all_params = list(pinn_model.parameters()) + list(self._term_net.parameters())
        opt = torch.optim.Adam(all_params, lr=cfg.lr)

        for step in range(cfg.n_iters):
            opt.zero_grad()
            x_in = x_t.detach().requires_grad_(True)

            y_pred = pinn_model(x_in)
            if hasattr(y_pred, "y"):
                y_pred = y_pred.y

            l_data = cfg.data_weight * nn.functional.mse_loss(y_pred, y_t)

            known_res = self.known_residual_fn(pinn_model, x_in)
            tau = self._term_net(x_t).squeeze(-1)
            l_term = cfg.term_weight * nn.functional.mse_loss(known_res, tau)

            (l_data + l_term).backward()
            opt.step()

        return self

    def predict_term(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the discovered term τ(x) at new points.

        Parameters
        ----------
        x : (N, d) query coordinates

        Returns
        -------
        (N,) numpy array of term values
        """
        import torch
        if self._term_net is None:
            raise RuntimeError("Call fit() before predict_term().")
        dev = next(self._term_net.parameters()).device
        x_t = torch.tensor(x, dtype=torch.float32, device=dev)
        with torch.no_grad():
            tau = self._term_net(x_t).squeeze(-1)
        return tau.cpu().numpy()

    def distill(
        self,
        x: np.ndarray,
        library: Optional[CandidateLibrary] = None,
        identifier: Optional[SINDyIdentifier] = None,
    ) -> SINDyResult:
        """Distill the neural term into a symbolic form via SINDy.

        Evaluates τ(x) at the provided collocation points and runs sparse
        regression over the candidate library to find a compact symbolic form.

        Parameters
        ----------
        x : (N, d) collocation points for distillation
        library : CandidateLibrary (default: poly_order=3)
        identifier : SINDyIdentifier (default: STRidge, threshold=1e-3)

        Returns
        -------
        SINDyResult
        """
        lib = library or CandidateLibrary(poly_order=3)
        sid = identifier or SINDyIdentifier()
        tau_vals = self.predict_term(x)
        Theta, names = lib.build(x)
        return sid.fit(Theta, tau_vals, term_names=names)

    @property
    def term_network(self) -> Optional[Any]:
        """The trained term neural network (nn.Module), or None before fit()."""
        return self._term_net


__all__ = [
    "CandidateLibrary",
    "SINDyResult",
    "SINDyIdentifier",
    "ResidualAnalysisResult",
    "ResidualAnalyzer",
    "NeuralTermConfig",
    "NeuralTermDiscovery",
]
