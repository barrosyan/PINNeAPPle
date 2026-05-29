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
# Classical signal decomposition — re-exported from pinneapple_solvers.
# Those solvers are the authoritative implementations; this re-export makes
# them discoverable from the time-series module where users expect them.
# ---------------------------------------------------------------------------
_CLASSICAL: list = []
try:
    from pinneapple_simulation.numerical_solvers.fft import FFTSolver
    from pinneapple_simulation.numerical_solvers.wavelet import WaveletSolver, wavedec_features
    from pinneapple_simulation.numerical_solvers.hilbert_huang import HilbertHuangSolver
    from pinneapple_simulation.numerical_solvers.eemd import EEMDSolver
    from pinneapple_simulation.numerical_solvers.ceemdan import CEEMDANSolver
    from pinneapple_simulation.numerical_solvers.vmd import VMDSolver
    from pinneapple_simulation.numerical_solvers.ssa import SSASolver
    from pinneapple_simulation.numerical_solvers.sst import SSTSolver
    from pinneapple_simulation.numerical_solvers.stl import STLSolver
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
    # Classical signal decomposition (from pinneapple_solvers)
    *_CLASSICAL,
]
