"""Patch snippet: add capabilities_getter to register_into_global() in neural_operators/registry.py.

Apply manually into your existing pinneaple_models/neural_operators/registry.py.
"""

def register_into_global() -> None:
    from pinneaple_models._registry_bridge import register_family_registry

    def capabilities(name: str, cls):
        key = name.lower().strip()
        caps = {"predicts": ["u"], "supports_physics_loss": (key in ("pino", "physics_informed_neural_operator"))}

        if key in ("deeponet", "multiscale_deeponet"):
            caps.update({"input_kind": "operator_branch_trunk", "expects": ["u_branch", "coords"], "predicts": ["u"]})
            return caps

        if key in ("fno", "fourier_neural_operator"):
            caps.update({"input_kind": "grid_1d", "expects": ["u_grid_1d"], "predicts": ["u"]})
            return caps

        if key in ("gno", "galerkin_neural_operator"):
            caps.update({"input_kind": "points", "expects": ["u_points", "coords_points"], "predicts": ["u"]})
            return caps

        if key in ("uno", "universal_operator_network"):
            caps.update({"input_kind": "grid_or_points", "expects": ["u_grid"], "predicts": ["u"]})
            return caps

        if key in ("pino", "physics_informed_neural_operator"):
            caps.update({"input_kind": "grid_or_points", "expects": ["u_grid", "physics_fn", "physics_data"], "predicts": ["u"], "supports_physics_loss": True})
            return caps

        caps.update({"input_kind": "grid_or_points", "expects": ["u"]})
        return caps

    register_family_registry(_REGISTRY, family="neural_operators", capabilities_getter=capabilities)
