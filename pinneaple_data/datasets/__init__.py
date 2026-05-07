"""Pre-built datasets for PINNeAPPle — physics, geometry, and time series.

Quick start
-----------
    from pinneaple_data.datasets import load_dataset, list_datasets

    # List all available datasets
    list_datasets()

    # Load a specific dataset
    data = load_dataset("burgers_1d")     # physics
    data = load_dataset("lorenz63")        # time series
    data = load_dataset("naca0012")        # geometry

    # Load with custom parameters
    data = load_dataset("heat_1d", k=0.2, Nx=128)
    data = load_dataset("lorenz63", T=100.0, dt=0.01)

    # Browse by category
    list_datasets(category="timeseries")
    list_datasets(category="physics")
    list_datasets(category="geometry")
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .registry import DatasetInfo, DatasetRegistry

# Trigger registration of all datasets
from . import physics     # noqa: F401
from . import geometry    # noqa: F401
from . import timeseries  # noqa: F401

try:
    from . import real_world  # noqa: F401  (optional: requires network / optional libs)
except Exception:
    pass

try:
    from . import external_sources  # noqa: F401  (optional: requires network / API keys)
except Exception:
    pass


def load_dataset(dataset_id: str, **kwargs) -> Dict[str, np.ndarray]:
    """Load a pre-built dataset by ID.

    Parameters
    ----------
    dataset_id:
        Unique identifier (see ``list_datasets()``).
    **kwargs:
        Passed to the dataset loader (e.g. ``Nx=256``, ``k=0.2``).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of arrays. Common keys: ``x``, ``t``, ``u``, ``X``.
    """
    return DatasetRegistry.load(dataset_id, **kwargs)


def list_datasets(category: Optional[str] = None,
                  verbose: bool = True) -> List[DatasetInfo]:
    """Print and return all registered datasets.

    Parameters
    ----------
    category:
        Filter by ``"physics"``, ``"geometry"``, or ``"timeseries"``.
        If ``None``, all datasets are shown.
    verbose:
        Print a formatted table.

    Returns
    -------
    List[DatasetInfo]
    """
    infos = DatasetRegistry.list_all(category)
    if verbose:
        header = f"  {'ID':<30}{'Category':<14}{'Fields'}"
        sep = "─" * 78
        print(f"\n{sep}")
        cat_label = f" [{category}]" if category else ""
        print(f"  PINNeAPPle Pre-built Datasets{cat_label}")
        print(sep)
        print(header)
        print(sep)
        prev_cat = None
        for info in infos:
            if info.category != prev_cat:
                if prev_cat is not None:
                    print()
                prev_cat = info.category
            print(f"  {info.id:<30}{info.category:<14}{', '.join(info.fields[:4])}")
        print(f"{sep}\n  Total: {len(infos)} dataset(s)\n{sep}\n")
    return infos


def dataset_info(dataset_id: str) -> DatasetInfo:
    """Return metadata for a specific dataset."""
    return DatasetRegistry.get_info(dataset_id)


def dataset_ids(category: Optional[str] = None) -> List[str]:
    """Return list of all registered dataset IDs."""
    return DatasetRegistry.ids(category)


__all__ = [
    "DatasetInfo",
    "DatasetRegistry",
    "load_dataset",
    "list_datasets",
    "dataset_info",
    "dataset_ids",
    # External source clients
    "MaterialsProjectClient",
    "NOMADClient",
    "NASAClient",
    "NOAAClient",
    "CopernicusClient",
    "BrazilDataCubeClient",
    "CPTECClient",
    "ONSClient",
    "NRELClient",
    "DesignSafeClient",
    "OpenFOAMClient",
    "SU2Client",
]

try:
    from .external_sources import (
        MaterialsProjectClient,
        NOMADClient,
        NASAClient,
        NOAAClient,
        CopernicusClient,
        BrazilDataCubeClient,
        CPTECClient,
        ONSClient,
        NRELClient,
        DesignSafeClient,
        OpenFOAMClient,
        SU2Client,
    )
except Exception:
    pass
