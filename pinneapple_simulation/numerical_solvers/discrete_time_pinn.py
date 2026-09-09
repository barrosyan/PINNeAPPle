"""Discrete-time physics-informed neural network (PINN), per Raissi,
Perdikaris & Karniadakis, "Physics Informed Deep Learning (Part II):
Data-driven Discovery of Nonlinear Partial Differential Equations"
(arXiv:1711.10566), Section 2 ("Discrete Time Models"), specialized here to
the pure forward-solve (not data-driven-discovery) setting for `burgers_1d`
and `allen_cahn_1d`.

Unlike every other PINN in this package (and unlike `xtfc_ivp.py`/
`xtfc_pde.py`, which are gradient-free), this module DOES use ordinary
gradient-based training -- but with a distinctive architecture and time
discretization that is algorithmically unrelated to the usual "continuous
time" PINN (a network of (x,t) -> u trained on scattered space-time
collocation points): here, a SINGLE network of x -> (q+1) values represents
an entire q-STAGE IMPLICIT RUNGE-KUTTA STEP from a known time t_n to
t_{n+1} = t_n + dt, where dt can be a SUBSTANTIAL fraction of the whole time
horizon (the original paper's Burgers example takes ONE step of dt=0.8 using
q=500 stages) -- the high stage-count IRK method supplies enough temporal
accuracy (local truncation error is formally O(dt^(2q+1)) for the q-stage
Gauss-Legendre family; see `irk_gauss_legendre.py`) to make a single giant
step numerically meaningful, in exchange for a much harder per-step training
problem than an explicit few-step scheme would pose.

Method summary
--------------
1. IRK tableau: the exact q-stage Gauss-Legendre (A, b, c) tableau from
   `irk_gauss_legendre.get_irk_tableau(q)` -- imported and reused verbatim,
   NOT rederived here (this module owns none of the tableau algebra).
2. Network (`DiscreteTimeRKNet`): a plain x -> R^(q+1) MLP. Column i (i=0..
   q-1) is the network's prediction of the solution at the i-th INTERMEDIATE
   RK stage time t_n + c_i*dt; column q is its prediction at the FINAL time
   t_n + dt. (Raissi et al.'s own notation: U^n_i for the stage predictions,
   U^{n+1} for the final one.)
3. Stage derivatives via forward-mode AD (`stage_derivatives`): because the
   network is applied POINTWISE to a batch of scalar x's (batch dimension is
   independent samples, not a computation graph coupling them), the
   per-sample derivative d(output_k)/dx for EVERY one of the q+1 outputs at
   EVERY batch point is obtainable from a SINGLE `torch.func.jvp` call with
   tangent = ones_like(x) (since d(f(x))/dx * 1 = f'(x) elementwise) -- far
   cheaper than q+1 separate reverse-mode passes or per-output autograd.grad
   calls. The second (viscous/diffusive) derivative is obtained the same way
   by taking the jvp of the jvp itself (forward-over-forward AD), still just
   two total forward evaluations of the network's computation graph.
4. RK residual (`discrete_time_rk_residual`): given the nonlinear spatial
   operator N[u] (e.g. Burgers' -u*u_x + nu*u_xx, or Allen-Cahn's
   eps^2*u_xx + 5u - 5u^3) evaluated at every stage's (value, first, second
   x-derivative) triple, the implicit-RK consistency equations (paper's
   Eqs. 2.4-2.6, adapted to a pure forward solve: `u_n_data` plays the role
   of the paper's known snapshot at t_n) are:

       U_i     - dt * sum_j A[i,j] * N[U_j]  ==  u_n_data,   i = 1..q
       U_{q+1} - dt * sum_j b_j   * N[U_j]   ==  u_n_data

   i.e. running the SAME IRK scheme backward from any stage (or from the
   final time) to t_n must reproduce the known data at t_n -- this is the
   trick that lets a single network encode the whole implicit step without
   ever needing labeled data at any intermediate or final time. Boundary
   conditions (time-invariant Dirichlet data here) are enforced as an
   ordinary MSE penalty at x0/x1 applied to ALL q+1 columns (the BC holds at
   every one of the q+1 times spanned by the step).
5. `train_discrete_time`: an L-BFGS driver (matching the original paper's
   optimizer choice -- these problems are small, smooth least-squares-like
   fits that L-BFGS converges on much faster than SGD/Adam) minimizing the
   sum of the interior RK-residual MSE and the boundary MSE.

Caveats (explicit, not glossed over)
-------------------------------------
- This is a genuine gradient-trained network (autograd/backprop), unlike
  every other X-TFC-family module in this package -- intentionally so, since
  discrete-time collocation in x alone (no t input) has no fixed-ELM
  analogue as directly applicable as the continuous-time TFC construction in
  `xtfc_pde.py`.
- q here defaults much smaller than the original paper's q=500 (which needed
  a full-batch L-BFGS run of order minutes) so this module's own tests run in
  seconds; `train_discrete_time`'s `q` argument can be raised freely for a
  more paper-faithful (and slower) reproduction.
- Only a SINGLE discrete step (t_n -> t_n+dt) is implemented per
  `train_discrete_time` call -- chaining multiple discrete-time steps (each
  using the previous step's converged U_{q+1} as the next step's u_n_data,
  the way the paper does for e.g. Allen-Cahn) is left to the caller by simply
  calling `train_discrete_time` again with the previous call's output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from pinneapple_neural.architectures.base import BaseModel

from .base import SolverBase, SolverOutput
from .irk_gauss_legendre import get_irk_tableau
from .registry import SolverRegistry

Array = np.ndarray


class DiscreteTimeRKNet(BaseModel):
    """x -> (q+1) discrete-time IRK network (Raissi et al. Part II, Section
    2). Column i<q is the i-th RK stage prediction U_i(x); column q is the
    final-time prediction U_{q+1}(x). Plain tanh MLP, matching the original
    paper's architecture choice.

    Subclasses `pinneapple_neural.architectures.base.BaseModel`: its default
    `forward_batch` (`batch['x'] -> self.forward(x)`) is already exactly the
    right contract here since this network's only input is the spatial
    coordinate `x` -- no override needed.
    """

    family = "discrete_time_pinn"
    name = "discrete_time_rk_net"

    def __init__(self, q: int, hidden_layers: "tuple[int, ...]" = (50, 50, 50, 50),
                 activation: str = "tanh"):
        super().__init__()
        self.q = int(q)
        act_cls = {"tanh": nn.Tanh, "gelu": nn.GELU, "silu": nn.SiLU}[activation]
        dims = [1, *hidden_layers, self.q + 1]
        layers: "list[nn.Module]" = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_cls())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1) -> (N, q+1)."""
        return self.net(x)


def stage_derivatives(model: nn.Module, x: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """U(x), U_x(x), U_xx(x), each (N, q+1), via forward-mode AD.

    x: (N, 1). Uses `torch.func.jvp` with tangent=ones_like(x): since the
    network is applied pointwise (batch dim is independent samples, not a
    coupled computation), d(f(x))/dx * 1 = f'(x) for every sample and every
    output column SIMULTANEOUSLY from one jvp call -- and the second
    derivative from one more (forward-over-forward), rather than q+1
    reverse-mode passes. `model` is called directly (not via
    `torch.func.functional_call`): jvp only pushes a tangent through the
    INPUT here, so treating the model's parameters as fixed constants during
    the push-forward is exactly the semantics wanted (we are differentiating
    U(x) w.r.t. x for fixed weights, not w.r.t. the weights).
    """
    ones = torch.ones_like(x)

    def f(x_: torch.Tensor) -> torch.Tensor:
        return model(x_)

    U, U_x = torch.func.jvp(f, (x,), (ones,))

    def f_prime(x_: torch.Tensor) -> torch.Tensor:
        _, tangent_out = torch.func.jvp(f, (x_,), (ones,))
        return tangent_out

    _, U_xx = torch.func.jvp(f_prime, (x,), (ones,))
    return U, U_x, U_xx


def discrete_time_rk_residual(
    model: nn.Module,
    x: torch.Tensor,
    u_n_data: torch.Tensor,
    dt: float,
    A: torch.Tensor,
    b: torch.Tensor,
    nonlinear_op: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Stage and final-time IRK consistency residuals (module docstring
    point 4). Returns (res_stage, res_final), shapes (N, q) and (N, 1).

    x: (N, 1) interior collocation points.
    u_n_data: (N, 1) known solution at t_n, evaluated at these SAME x
        (typically the problem's initial condition, or a previous discrete
        step's converged final-time output).
    dt: step size t_{n+1} - t_n.
    A, b: IRK tableau (q,q) and (q,), e.g. from `get_irk_tableau(q)`, as
        torch tensors matching `x`'s dtype/device.
    nonlinear_op(u, u_x, u_xx) -> N[u], applied columnwise to ALL q stages at
        once (each argument (N, q)); returns (N, q).
    """
    U, U_x, U_xx = stage_derivatives(model, x)
    q = A.shape[0]
    U_stage = U[:, :q]
    Ux_stage = U_x[:, :q]
    Uxx_stage = U_xx[:, :q]
    U_final = U[:, q:q + 1]

    N_j = nonlinear_op(U_stage, Ux_stage, Uxx_stage)          # (N, q)
    rhs_stage = dt * (N_j @ A.T)                              # (N, q): sum_j A[i,j]*N_j[:,j]
    res_stage = U_stage - rhs_stage - u_n_data                # broadcast (N,1) -> (N,q)

    rhs_final = dt * (N_j @ b[:, None])                       # (N, 1)
    res_final = U_final - rhs_final - u_n_data
    return res_stage, res_final


def _burgers_nonlinear_op(u: torch.Tensor, u_x: torch.Tensor, u_xx: torch.Tensor,
                           nu: float) -> torch.Tensor:
    return -u * u_x + nu * u_xx


def _allen_cahn_nonlinear_op(u: torch.Tensor, u_x: torch.Tensor, u_xx: torch.Tensor,
                              eps: float) -> torch.Tensor:
    return eps ** 2 * u_xx + 5.0 * u - 5.0 * u ** 3


# Named PDE presets: nonlinear operator N[u], boundary value (time-invariant
# Dirichlet on both ends here, matching both
# `pinneapple_tools.benchmark_suite.tasks.burgers_1d.Burgers1DTask` and
# `...tasks.allen_cahn_1d.AllenCahn1DTask`, whose already-built method-of-
# lines/RK4 reference solutions this module's tests validate against instead
# of re-deriving a new one), and each preset's standard domain/parameter.
_PDE_PRESETS: Dict[str, Dict[str, Any]] = {
    "burgers_1d": dict(
        nonlinear_op=lambda u, ux, uxx, param: _burgers_nonlinear_op(u, ux, uxx, param),
        bc_value=0.0, x_span=(-1.0, 1.0), default_param=0.01 / np.pi,
        ic=lambda x: -np.sin(np.pi * x),
    ),
    "allen_cahn_1d": dict(
        nonlinear_op=lambda u, ux, uxx, param: _allen_cahn_nonlinear_op(u, ux, uxx, param),
        bc_value=-1.0, x_span=(-1.0, 1.0), default_param=0.01,
        ic=lambda x: x ** 2 * np.cos(np.pi * x),
    ),
}


@dataclass
class DiscreteTimeResult:
    model: DiscreteTimeRKNet
    A: Array
    b: Array
    c: Array
    q: int
    dt: float
    x_span: "tuple[float, float]"
    loss_history: "list[float]"

    def predict_final(self, x_query: Array) -> Array:
        """U_{n+1}(x_query) -- the converged solution at t_n + dt."""
        x_t = torch.as_tensor(np.asarray(x_query, dtype=np.float32)).reshape(-1, 1)
        with torch.no_grad():
            out = self.model(x_t)
        return out[:, -1].numpy()

    def predict_stage(self, i: int, x_query: Array) -> Array:
        """U_i(x_query) -- the i-th intermediate RK stage prediction, at
        time t_n + c[i]*dt."""
        x_t = torch.as_tensor(np.asarray(x_query, dtype=np.float32)).reshape(-1, 1)
        with torch.no_grad():
            out = self.model(x_t)
        return out[:, i].numpy()


def train_discrete_time(
    pde: str,
    x_n: Array,
    u_n: Array,
    dt: float,
    q: int = 32,
    x_span: "Optional[tuple[float, float]]" = None,
    hidden_layers: "tuple[int, ...]" = (30, 30, 30),
    activation: str = "tanh",
    param: "Optional[float]" = None,
    n_boundary: int = 20,
    bc_weight: float = 10.0,
    max_iter: int = 300,
    lr: float = 0.5,
    tol_grad: float = 1e-9,
    seed: int = 0,
) -> DiscreteTimeResult:
    """Train a `DiscreteTimeRKNet` for one implicit-RK step of `pde` in
    {"burgers_1d", "allen_cahn_1d"}, from known data (x_n, u_n) at t_n to
    t_n + dt, via L-BFGS (module docstring point 5).

    pde: selects the nonlinear operator N[u] and the (time-invariant
        Dirichlet) boundary value from `_PDE_PRESETS`.
    x_n, u_n: (N,) known solution samples at t_n (e.g. the problem's initial
        condition, or a previous `train_discrete_time` call's
        `predict_final` output for a chained multi-step run).
    dt: step size (t_n -> t_n+dt); may be a large fraction of the whole
        horizon given a large enough `q` (see module docstring).
    q: number of IRK stages (Gauss-Legendre order 2q). Paper-scale examples
        use q up to ~500; kept smaller by default here for test speed.
    x_span: domain for boundary points; defaults to the preset's standard
        domain when omitted.
    param: nu (burgers_1d) or eps (allen_cahn_1d); defaults to the preset's
        standard benchmark value when omitted.
    n_boundary: number of boundary collocation points per side.
    bc_weight: loss weight on the boundary MSE term relative to the interior
        RK-residual MSE.
    max_iter, lr, tol_grad: `torch.optim.LBFGS` settings.
    seed: torch RNG seed for network initialization.

    Returns a `DiscreteTimeResult` (model, tableau, loss history, and
    `predict_final`/`predict_stage` convenience methods).
    """
    if pde not in _PDE_PRESETS:
        raise ValueError(f"pde={pde!r} not in {list(_PDE_PRESETS)}")
    preset = _PDE_PRESETS[pde]
    x_span = x_span or preset["x_span"]
    param = preset["default_param"] if param is None else float(param)
    x0, x1 = x_span

    torch.manual_seed(seed)
    model = DiscreteTimeRKNet(q=q, hidden_layers=hidden_layers, activation=activation)

    A_np, b_np, c_np = get_irk_tableau(q)
    A = torch.tensor(A_np, dtype=torch.float32)
    b = torch.tensor(b_np, dtype=torch.float32)

    x_n_t = torch.as_tensor(np.asarray(x_n, dtype=np.float32)).reshape(-1, 1)
    u_n_t = torch.as_tensor(np.asarray(u_n, dtype=np.float32)).reshape(-1, 1)

    x_bc = torch.tensor(
        np.concatenate([np.full(n_boundary, x0), np.full(n_boundary, x1)]).astype(np.float32)
    ).reshape(-1, 1)
    bc_value = float(preset["bc_value"])

    nonlinear_op = lambda u, ux, uxx: preset["nonlinear_op"](u, ux, uxx, param)

    optimizer = torch.optim.LBFGS(
        model.parameters(), lr=lr, max_iter=max_iter, tolerance_grad=tol_grad,
        history_size=50, line_search_fn="strong_wolfe",
    )
    loss_history: "list[float]" = []

    def closure():
        optimizer.zero_grad()
        res_stage, res_final = discrete_time_rk_residual(model, x_n_t, u_n_t, dt, A, b, nonlinear_op)
        loss_interior = torch.mean(res_stage ** 2) + torch.mean(res_final ** 2)
        u_bc = model(x_bc)
        loss_bc = torch.mean((u_bc - bc_value) ** 2)
        loss = loss_interior + bc_weight * loss_bc
        loss.backward()
        loss_history.append(float(loss.item()))
        return loss

    optimizer.step(closure)

    return DiscreteTimeResult(
        model=model, A=A_np, b=b_np, c=c_np, q=q, dt=dt, x_span=x_span,
        loss_history=loss_history,
    )


@SolverRegistry.register(
    name="discrete_time_pinn",
    family="pde",
    description="Discrete-time PINN (Raissi et al. Part II): a single x->(q+1) network encodes "
                "one q-stage implicit-RK step for burgers_1d/allen_cahn_1d, using "
                "irk_gauss_legendre's exact tableau and forward-mode-AD stage derivatives.",
    tags=["pinn", "discrete-time", "irk", "gauss-legendre", "burgers", "allen-cahn"],
)
class DiscreteTimePINNSolver(SolverBase):
    """Thin `SolverBase`/registry wrapper around `train_discrete_time`. The
    functional API (`train_discrete_time`, `DiscreteTimeRKNet`,
    `stage_derivatives`, `discrete_time_rk_residual`) is the primary entry
    point.
    """

    def __init__(self, pde: str = "burgers_1d", q: int = 32,
                 hidden_layers: "tuple[int, ...]" = (30, 30, 30), activation: str = "tanh",
                 n_boundary: int = 20, bc_weight: float = 10.0, max_iter: int = 300,
                 lr: float = 0.5, seed: int = 0):
        super().__init__()
        self.pde = pde
        self.q = int(q)
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.n_boundary = int(n_boundary)
        self.bc_weight = float(bc_weight)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.seed = int(seed)

    def forward(
        self,
        x_n,
        u_n,
        dt: float,
        param: "Optional[float]" = None,
        x_span: "Optional[tuple[float, float]]" = None,
        n_query: int = 200,
    ) -> SolverOutput:
        result = train_discrete_time(
            self.pde, x_n, u_n, dt, q=self.q, x_span=x_span,
            hidden_layers=self.hidden_layers, activation=self.activation, param=param,
            n_boundary=self.n_boundary, bc_weight=self.bc_weight, max_iter=self.max_iter,
            lr=self.lr, seed=self.seed,
        )
        x_query = np.linspace(*result.x_span, n_query)
        u_query = result.predict_final(x_query)
        return SolverOutput(
            result=torch.from_numpy(u_query.astype(np.float32)),
            losses={"loss": torch.tensor(result.loss_history[-1] if result.loss_history else 0.0)},
            extras={
                "x_query": x_query.astype(np.float32),
                "loss_history": result.loss_history,
                "q": result.q,
                "dt": result.dt,
                "method": "discrete_time_pinn",
                "pde": self.pde,
            },
        )
