"""``pinneapple_train.trainer`` compatibility submodule.

Found via the pre-existing test suite (``tests/pinneapple_train
/test_trainer_minimal.py``) failing to even collect: ``pinneapple_train``
had only a flat ``__init__.py`` re-exporting ``Trainer``/``TrainConfig``
at the top level, but at least seven example scripts
(``examples/numerical_solvers/01_solvers_fft_feature_train.py``,
``examples/time_series/{01_FNO,02_full_pipeline,07_fft_lstm_hybrid,
08_hht_lstm_hybrid}.py``, ``examples/trainer/01_audited_training.py``),
``pinneapple_tools/benchmark_suite/{timeseries_pipeline,physics_pipeline}
.py``, and the library's own internal NLP-to-PDE agent knowledge base
(``pinneapple_problemdesign/knowledge/{pinneapple_capabilities,mapping}
.py``) all import from ``pinneapple_train.trainer`` as a genuine
submodule (``from pinneapple_train.trainer import Trainer, TrainConfig``),
which raised ``ModuleNotFoundError`` unconditionally -- none of that code
could have ever run. This submodule (and its siblings ``losses.py``,
``metrics.py``) makes that import path real.
"""
from pinneapple_neural.trainer.trainer import Trainer, TrainConfig

__all__ = ["Trainer", "TrainConfig"]
