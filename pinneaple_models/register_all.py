from __future__ import annotations
"""Register all model families into the global ModelRegistry."""

def register_all() -> None:
    from pinneaple_models.autoencoders.registry import register_into_global as reg_autoenc
    from pinneaple_models.classical_ts.registry import register_into_global as reg_classical
    from pinneaple_models.continuous.registry import register_into_global as reg_continuous
    from pinneaple_models.convolutions.registry import register_into_global as reg_conv
    from pinneaple_models.graphnn.registry import register_into_global as reg_graph
    from pinneaple_models.neural_operators.registry import register_into_global as reg_ops
    from pinneaple_models.physics_aware.registry import register_into_global as reg_phys
    from pinneaple_models.pinns.registry import register_into_global as reg_pinns
    from pinneaple_models.recurrent.registry import register_into_global as reg_rnn
    from pinneaple_models.reservoir_computing.registry import register_into_global as reg_rc
    from pinneaple_models.rom.registry import register_into_global as reg_rom
    from pinneaple_models.transformers.registry import register_into_global as reg_tf

    reg_autoenc()
    reg_classical()
    reg_continuous()
    reg_conv()
    reg_graph()
    reg_ops()
    reg_phys()
    reg_pinns()
    reg_rnn()
    reg_rc()
    reg_rom()
    reg_tf()
