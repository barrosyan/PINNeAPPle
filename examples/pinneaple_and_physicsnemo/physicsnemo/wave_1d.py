import numpy as np
from sympy import Symbol, sin

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch, PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_1d import Line1D
from physicsnemo.sym.domain.constraint import PointwiseBoundaryConstraint, PointwiseInteriorConstraint
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key

from wave_equation import WaveEquation1D


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # PDE + rede
    we = WaveEquation1D(c=1.0)
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("t")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = we.make_nodes() + [wave_net.make_node(name="wave_network")]

    # geometria e domínio
    x_sym, t_sym = Symbol("x"), Symbol("t")
    L = float(np.pi)
    geo = Line1D(0, L)
    time_range = {t_sym: (0, 2 * L)}
    domain = Domain()

    # IC: u(x,0)=sin(x) e u_t(x,0)=sin(x)
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": sin(x_sym), "u__t": sin(x_sym)},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={"u": 1.0, "u__t": 1.0},
        parameterization={t_sym: 0.0},
    )
    domain.add_constraint(IC, "IC")

    # BC: u(0,t)=0 e u(L,t)=0
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0},
        batch_size=cfg.batch_size.BC,
        parameterization=time_range,
    )
    domain.add_constraint(BC, "BC")

    # interior: residual = 0
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"wave_equation": 0},
        batch_size=cfg.batch_size.interior,
        parameterization=time_range,
    )
    domain.add_constraint(interior, "interior")

    # validação (solução analítica do exemplo)
    deltaT, deltaX = 0.01, 0.01
    x = np.arange(0, L, deltaX)
    t = np.arange(0, 2 * L, deltaT)
    X, T = np.meshgrid(x, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    u = np.sin(X) * (np.cos(T) + np.sin(T))

    validator = PointwiseValidator(
        nodes=nodes,
        invar={"x": X, "t": T},
        true_outvar={"u": u},
        batch_size=128,
    )
    domain.add_validator(validator)

    # solver
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()