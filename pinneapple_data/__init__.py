try:
    from .stl_import import STLMesh, load_stl, load_stl_bytes
except Exception:
    pass

try:
    from .physical_sample import PhysicalSample
except Exception:
    pass

try:
    from .zarr_store import UPDZarrStore
    from .zarr_iterable import ZarrUPDIterable
except Exception:
    pass

try:
    from .collate import collate_upd_supervised, collate_pinn_batches, move_batch_to_device
except Exception:
    pass

try:
    from .collocation import CollocationSampler, CollocationConfig
except Exception:
    pass

try:
    from .active_learning import (
        ActiveLearningConfig,
        ResidualBasedAL,
        VarianceBasedAL,
        CombinedAL,
        AdaptiveCollocationTrainer,
    )
except Exception:
    pass

try:
    from .transforms import Normalizer, StandardScaler, MinMaxScaler
except Exception:
    pass

try:
    from .upd_types import UPDItem, ConditionSpec, SamplingSpec, Batch
except Exception:
    pass

try:
    from .splits import SplitSpec, split_indices
except Exception:
    pass

try:
    from .adapters.pinn_batch_builders import PINNBatch, build_from_bundle, build_from_solver, build_from_real_data
except Exception:
    pass

from .datasets import (
    DatasetInfo,
    DatasetRegistry,
    load_dataset,
    list_datasets,
    dataset_info,
    dataset_ids,
)
from . import datasets

__all__ = [
    # STL
    "STLMesh",
    "load_stl",
    "load_stl_bytes",
    # Physical sample
    "PhysicalSample",
    # UPD / Zarr pipeline
    "UPDZarrStore",
    "ZarrUPDIterable",
    # Collate
    "collate_upd_supervised",
    "collate_pinn_batches",
    "move_batch_to_device",
    # Collocation
    "CollocationSampler",
    "CollocationConfig",
    # Active learning
    "ActiveLearningConfig",
    "ResidualBasedAL",
    "VarianceBasedAL",
    "CombinedAL",
    "AdaptiveCollocationTrainer",
    # Transforms / normalizers
    "Normalizer",
    "StandardScaler",
    "MinMaxScaler",
    # UPD types
    "UPDItem",
    "ConditionSpec",
    "SamplingSpec",
    "Batch",
    # Splits
    "SplitSpec",
    "split_indices",
    # PINN batch builders
    "PINNBatch",
    "build_from_bundle",
    "build_from_solver",
    "build_from_real_data",
    # Datasets
    "datasets",
    "DatasetInfo",
    "DatasetRegistry",
    "load_dataset",
    "list_datasets",
    "dataset_info",
    "dataset_ids",
]
