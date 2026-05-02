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
    "STLMesh",
    "load_stl",
    "load_stl_bytes",
    # UPD / Zarr pipeline
    "PhysicalSample",
    "UPDZarrStore",
    "ZarrUPDIterable",
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
    # Datasets
    "datasets",
    "DatasetInfo",
    "DatasetRegistry",
    "load_dataset",
    "list_datasets",
    "dataset_info",
    "dataset_ids",
]
