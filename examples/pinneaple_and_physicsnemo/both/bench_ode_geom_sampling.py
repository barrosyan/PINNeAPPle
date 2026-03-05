import math
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import torch
import torch.optim as optim

# PhysicsNeMo Sym (só para geometria/amostragem)
from sympy import Symbol
from physicsnemo.sym.geometry.primitives_1d import Line1D

# PINNeAPPle
from pinneaple_pinn.factory.pinn_factory import PINNProblemSpec, PINNFactory

from pinneaple_models.pinns.vanilla import VanillaPINN
from pinneaple_models.pinns.inverse import InversePINN


# ---------
# Config
# ---------
@dataclass
class Cfg:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    steps: int = 2000
    lr: float = 1e-3
    n_col: int = 2048
    n_bc: int = 256
    domain_lo: float = -math.pi
    domain_hi: float = +math.pi


# ---------
# Adapter: força output Tensor para o PINNFactory (evita PINNOutput - Tensor)
# ---------
class TensorOut(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        out = self.model(*inputs)
        # PINNeAPPle normalmente tem atributos: y/pred/logits/etc
        if isinstance(out, torch.Tensor):
            return out
        for attr in ("y", "pred", "logits", "out"):
            if hasattr(out, attr):
                v = getattr(out, attr)
                if isinstance(v, torch.Tensor):
                    return v
        raise TypeError("Model output is not a Tensor and has no known tensor attribute.")


# ---------
# Sampling via PhysicsNeMo Sym geometry
# ---------
def sample_points_line1d(geo: Line1D, n_interior: int, n_boundary: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Retorna:
      x_col: (n_interior,1) com requires_grad=True
      x_bc:  (n_boundary,1) com requires_grad=True (amostras nos endpoints)
    """
    # sample_interior / sample_boundary existem nos objetos de geometria :contentReference[oaicite:6]{index=6}
    interior = geo.sample_interior(n_interior)
    boundary = geo.sample_boundary(n_boundary)

    # keys em 1D geralmente usam "x"
    x_col = torch.from_numpy(interior["x"]).float().requires_grad_(True)
    x_bc = torch.from_numpy(boundary["x"]).float().requires_grad_(True)
    return x_col, x_bc


# ---------
# Problema: y''(x) + y(x) = 0 em [-pi, pi], com BC y(-pi)=0 e y(pi)=0
# Solução analítica escolhida: y(x)=sin(x) satisfaz a ODE e BC nesses pontos.
# ---------
def make_spec() -> PINNProblemSpec:
    return PINNProblemSpec(
        pde_residuals=["Derivative(u(x), x, 2) + u(x)"],
        conditions=[
            {"name": "bc_left",  "equation": "u(x)", "weight": 10.0},  # u(-pi)=0
            {"name": "bc_right", "equation": "u(x)", "weight": 10.0},  # u(pi)=0
        ],
        independent_vars=["x"],
        dependent_vars=["u"],
        inverse_params=[],
        loss_weights={"pde": 1.0, "conditions": 1.0, "data": 0.0},
        verbose=False,
    )


def train_one(name: str, model: torch.nn.Module, loss_fn, cfg: Cfg, x_col: torch.Tensor, x_bc: torch.Tensor) -> Dict[str, float]:
    device = torch.device(cfg.device)
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    # manda pontos para device
    x_col = x_col.to(device).requires_grad_(True)
    x_bc = x_bc.to(device).requires_grad_(True)

    # conditions no PINNFactory: você passa tuplas alinhadas com independent_vars (aqui só "x")
    # BC esquerda: x = -pi ; BC direita: x = +pi
    x_left  = torch.full((x_bc.shape[0] // 2, 1), cfg.domain_lo, device=device, requires_grad=True)
    x_right = torch.full((x_bc.shape[0] - x_left.shape[0], 1), cfg.domain_hi, device=device, requires_grad=True)

    for step in range(1, cfg.steps + 1):
        batch = {
            "collocation": (x_col,),
            "conditions": [(x_left,), (x_right,)],
            "data": None,
        }
        total, comps = loss_fn(model, batch)
        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

        if step % 200 == 0:
            print(f"[{name}] step={step:04d} total={comps['total']:.3e} pde={comps.get('pde',0):.3e} cond={comps.get('conditions',0):.3e}")

    # métrica simples: erro L2 relativo vs sin(x)
    with torch.no_grad():
        xt = torch.linspace(cfg.domain_lo, cfg.domain_hi, 512, device=device).unsqueeze(1)
        y_pred = model(xt)
        y_true = torch.sin(xt)
        rel_l2 = torch.linalg.norm(y_pred - y_true) / (torch.linalg.norm(y_true) + 1e-12)

    return {"rel_l2": float(rel_l2.item())}


def main():
    cfg = Cfg()

    # PhysicsNeMo geometry (1D line). Em 2D/3D você reaproveita a mesma ideia. :contentReference[oaicite:7]{index=7}
    x_sym = Symbol("x")
    geo = Line1D(cfg.domain_lo, cfg.domain_hi)

    x_col, x_bc = sample_points_line1d(geo, cfg.n_col, cfg.n_bc)

    # PINNeAPPle factory loss
    spec = make_spec()
    factory = PINNFactory(spec)
    loss_fn = factory.generate_loss_function()

    results: List[Dict[str, float]] = []

    # 1) Vanilla PINN
    vanilla_core = VanillaPINN(in_dim=1, out_dim=1, hidden=[64, 64, 64], activation="tanh")
    vanilla = TensorOut(vanilla_core)
    r1 = train_one("VanillaPINN", vanilla, loss_fn, cfg, x_col, x_bc)
    results.append({"model": "VanillaPINN", **r1})

    # 2) InversePINN (mesmo sem inversão aqui, é só pra mostrar que roda e comparar)
    inv_core = InversePINN(in_dim=1, out_dim=1, hidden=[64, 64, 64], activation="tanh", inverse_params=[], initial_guesses={})
    inv = TensorOut(inv_core.as_factory_model(independent_vars=["x"]))  # garante assinatura do factory
    r2 = train_one("InversePINN", inv, loss_fn, cfg, x_col, x_bc)
    results.append({"model": "InversePINN", **r2})

    print("\n=== RESULTS ===")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()