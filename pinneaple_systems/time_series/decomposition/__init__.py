from .fft_forecaster import FFTForecaster
from .fft_nn         import FFTNNForecaster
from .hht_nn         import HHTNNForecaster
from .fft_lstm       import (
    FFTDecomposer,
    ResidualLSTMConfig,
    ResidualLSTMForecaster,
    FFTLSTMPipeline,
)
from .hht_lstm import (
    HHTDecomposer,
    HHTLSTMPipeline,
)

# ---------------------------------------------------------------------------
# Classical signal decomposition — re-exported from pinneaple_solvers.
# Those solvers are the authoritative implementations; this re-export makes
# them discoverable from the time-series module where users expect them.
# ---------------------------------------------------------------------------
_CLASSICAL: list = []
try:
    from pinneaple_simulation.numerical_solvers.fft import FFTSolver
    from pinneaple_simulation.numerical_solvers.wavelet import WaveletSolver, wavedec_features
    from pinneaple_simulation.numerical_solvers.hilbert_huang import HilbertHuangSolver
    from pinneaple_simulation.numerical_solvers.eemd import EEMDSolver
    from pinneaple_simulation.numerical_solvers.ceemdan import CEEMDANSolver
    from pinneaple_simulation.numerical_solvers.vmd import VMDSolver
    from pinneaple_simulation.numerical_solvers.ssa import SSASolver
    from pinneaple_simulation.numerical_solvers.sst import SSTSolver
    from pinneaple_simulation.numerical_solvers.stl import STLSolver
    _CLASSICAL = [
        "FFTSolver", "WaveletSolver", "wavedec_features",
        "HilbertHuangSolver", "EEMDSolver", "CEEMDANSolver",
        "VMDSolver", "SSASolver", "SSTSolver", "STLSolver",
    ]
except ImportError:
    pass

__all__ = [
    # Neural decomposition forecasters
    "FFTForecaster",
    "FFTNNForecaster",
    "HHTNNForecaster",
    "FFTDecomposer",
    "ResidualLSTMConfig",
    "ResidualLSTMForecaster",
    "FFTLSTMPipeline",
    "HHTDecomposer",
    "HHTLSTMPipeline",
    # Classical signal decomposition (from pinneaple_solvers)
    *_CLASSICAL,
]
