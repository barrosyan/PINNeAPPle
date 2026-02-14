from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pinneaple_arena.bundle.loader import BundleData


class PhysicsNeMoSymBackend:
    name = "physicsnemo_sym"

    def train(self, bundle: BundleData, run_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        This backend relies on PhysicsNeMo Sym being installed.
        It trains internally and returns metrics (no torch model object is guaranteed).
        """
        train_cfg = dict(run_cfg.get("train", {}))
        arena_cfg = dict(run_cfg.get("arena", {}))
        model_cfg = dict(run_cfg.get("model", {}))

        device = str(train_cfg.get("device", "cuda"))
        max_steps = int(train_cfg.get("max_steps", 20000))
        lr = float(train_cfg.get("lr", 1e-3))
        weights = dict(train_cfg.get("weights", {}))
        w_pde = float(weights.get("pde", 1.0))
        w_bc = float(weights.get("bc", 10.0))

        n_collocation = int(arena_cfg.get("n_collocation", 4096))
        n_boundary = int(arena_cfg.get("n_boundary", 2048))

        nr_layers = int(model_cfg.get("nr_layers", 6))
        layer_size = int(model_cfg.get("layer_size", 256))
        activation = str(model_cfg.get("activation", "tanh"))

        nu = float(bundle.manifest["nu"])

        try:
            # Common PhysicsNeMo Sym API style
            from physicsnemo.sym.hydra import ModulusConfig
            from physicsnemo.sym.domain.domain import Domain
            from physicsnemo.sym.solver import Solver
            from physicsnemo.sym.key import Key
            from physicsnemo.sym.models.fully_connected import FullyConnectedArch
            from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
            from physicsnemo.sym.domain.constraint.continuous import PointwiseInteriorConstraint, PointwiseBoundaryConstraint
        except Exception as e:
            raise RuntimeError(
                "PhysicsNeMo Sym backend requested but physicsnemo.sym is not available.\n"
                "Install steps:\n"
                "  1) pip install nvidia-physicsnemo-sym\n"
                "  2) ensure torch+CUDA works if using device=cuda\n"
                f"Import error: {e}"
            )

        # sample
        rng = np.random.default_rng(0)

        def _sample_df(df, n):
            n = min(int(n), len(df))
            if n <= 0:
                return df.iloc[:0]
            idx = rng.integers(0, len(df), size=n, endpoint=False)
            return df.iloc[idx]

        col = _sample_df(bundle.points_collocation, n_collocation)
        bnd = _sample_df(bundle.points_boundary, n_boundary)

        input_keys = [Key("x"), Key("y")]
        output_keys = [Key("u"), Key("v"), Key("p")]

        ns = NavierStokes(nu=nu, rho=1.0, dim=2, time=False)

        net = FullyConnectedArch(
            input_keys=input_keys,
            output_keys=output_keys,
            nr_layers=nr_layers,
            layer_size=layer_size,
            activation_fn=activation,
        )

        nodes = ns.make_nodes() + [net.make_node(name="flow_net")]

        domain = Domain()

        # interior constraint
        invar_interior = {"x": col["x"].to_numpy().reshape(-1, 1), "y": col["y"].to_numpy().reshape(-1, 1)}
        outvar_interior = {
            "continuity": np.zeros((len(col), 1), dtype=np.float32),
            "momentum_x": np.zeros((len(col), 1), dtype=np.float32),
            "momentum_y": np.zeros((len(col), 1), dtype=np.float32),
        }
        interior = PointwiseInteriorConstraint(
            nodes=nodes,
            invar=invar_interior,
            outvar=outvar_interior,
            batch_size=min(len(col), n_collocation),
            lambda_weighting={
                "continuity": w_pde,
                "momentum_x": w_pde,
                "momentum_y": w_pde,
            },
        )
        domain.add_constraint(interior, "interior")

        # boundary constraints helper
        def _mask(region: str):
            return bnd[bnd["region"].astype(str) == region]

        # inlet u=1 v=0
        inlet = _mask("inlet")
        if len(inlet) > 0:
            invar = {"x": inlet["x"].to_numpy().reshape(-1, 1), "y": inlet["y"].to_numpy().reshape(-1, 1)}
            outvar = {"u": np.ones((len(inlet), 1), dtype=np.float32), "v": np.zeros((len(inlet), 1), dtype=np.float32)}
            c = PointwiseBoundaryConstraint(
                nodes=nodes,
                invar=invar,
                outvar=outvar,
                batch_size=min(len(inlet), n_boundary),
                lambda_weighting={"u": w_bc, "v": w_bc},
            )
            domain.add_constraint(c, "bc_inlet")

        # walls, obstacle no-slip
        for reg, name in (("walls", "bc_walls"), ("obstacle", "bc_obstacle")):
            df = _mask(reg)
            if len(df) > 0:
                invar = {"x": df["x"].to_numpy().reshape(-1, 1), "y": df["y"].to_numpy().reshape(-1, 1)}
                outvar = {"u": np.zeros((len(df), 1), dtype=np.float32), "v": np.zeros((len(df), 1), dtype=np.float32)}
                c = PointwiseBoundaryConstraint(
                    nodes=nodes,
                    invar=invar,
                    outvar=outvar,
                    batch_size=min(len(df), n_boundary),
                    lambda_weighting={"u": w_bc, "v": w_bc},
                )
                domain.add_constraint(c, name)

        # outlet p=0
        outlet = _mask("outlet")
        if len(outlet) > 0:
            invar = {"x": outlet["x"].to_numpy().reshape(-1, 1), "y": outlet["y"].to_numpy().reshape(-1, 1)}
            outvar = {"p": np.zeros((len(outlet), 1), dtype=np.float32)}
            c = PointwiseBoundaryConstraint(
                nodes=nodes,
                invar=invar,
                outvar=outvar,
                batch_size=min(len(outlet), n_boundary),
                lambda_weighting={"p": w_bc},
            )
            domain.add_constraint(c, "bc_outlet")

        # minimal config
        mcfg = ModulusConfig()
        mcfg.training.max_steps = max_steps
        mcfg.optimizer.lr = lr
        # device in physicsnemo is often read from environment/config; solver will handle

        solver = Solver(mcfg, domain)
        solver.solve()

        # Return minimal metrics. Detailed logs depend on PhysicsNeMo config/loggers.
        return {
            "device": device,
            "metrics": {
                "nu": float(nu),
                "max_steps": float(max_steps),
                "lr": float(lr),
            },
        }
