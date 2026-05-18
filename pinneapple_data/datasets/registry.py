"""Central registry for pre-built PINNeAPPle datasets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class DatasetInfo:
    id: str
    name: str
    category: str       # "physics" | "geometry" | "timeseries"
    description: str
    fields: List[str]   # variable names returned by loader
    tags: List[str] = field(default_factory=list)
    license: str = "synthetic"
    reference: str = ""


class DatasetRegistry:
    """Central registry for pre-built PINNeAPPle datasets."""

    _store: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        info: DatasetInfo,
        loader: Callable[..., Dict[str, np.ndarray]],
    ) -> None:
        cls._store[info.id] = {"info": info, "loader": loader}

    @classmethod
    def load(cls, dataset_id: str, **kwargs) -> Dict[str, np.ndarray]:
        if dataset_id not in cls._store:
            raise KeyError(
                f"Dataset '{dataset_id}' not found. "
                f"Use list_datasets() to see available IDs.\n"
                f"Available: {cls.ids()}"
            )
        return cls._store[dataset_id]["loader"](**kwargs)

    @classmethod
    def get_info(cls, dataset_id: str) -> DatasetInfo:
        if dataset_id not in cls._store:
            raise KeyError(f"Dataset '{dataset_id}' not registered.")
        return cls._store[dataset_id]["info"]

    @classmethod
    def list_all(cls, category: Optional[str] = None) -> List[DatasetInfo]:
        infos = [v["info"] for v in cls._store.values()]
        if category is not None:
            infos = [i for i in infos if i.category == category]
        return sorted(infos, key=lambda x: (x.category, x.id))

    @classmethod
    def ids(cls, category: Optional[str] = None) -> List[str]:
        return [info.id for info in cls.list_all(category)]

    @classmethod
    def __contains__(cls, dataset_id: str) -> bool:
        return dataset_id in cls._store
