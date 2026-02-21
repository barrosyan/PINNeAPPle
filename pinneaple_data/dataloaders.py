"""DataLoader builders for UPD shards and PhysicalSample-based PINN training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset

from pinneaple_pinn.io import UPDItem, UPDDataset, SamplingSpec, ConditionSpec, PINNMapping

from .collate import collate_pinn_batches
from .physical_sample import PhysicalSample


@dataclass
class DataLoaderSpec:
    """
    Configuration container for torch.utils.data.DataLoader parameters.

    Attributes
    ----------
    batch_size : int
        Number of shard items per batch. For PINN training, this is often 1
        because each shard may internally contain many sampled points.
    num_workers : int
        Number of subprocesses used for data loading.
    shuffle : bool
        Whether to shuffle dataset indices at every epoch.
    pin_memory : bool
        If True, the DataLoader will copy tensors into CUDA pinned memory
        before returning them.
    drop_last : bool
        Whether to drop the last incomplete batch.
    """
    batch_size: int = 1
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    drop_last: bool = False


class _UPDShardTorchDataset(Dataset):
    """
    Torch Dataset wrapper around a UPDDataset shard.

    This class adapts the UPDDataset sampling interface to the PyTorch
    Dataset API so it can be consumed by a DataLoader.

    Each call to __getitem__ produces a newly sampled batch of collocation,
    condition, and data points based on a SamplingSpec. The sampling seed
    is varied per index to ensure fresh stochastic sampling.
    """

    def __init__(
        self,
        item: UPDItem,
        mapping: PINNMapping,
        sampling: SamplingSpec,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        length: int = 10_000,
    ):
        """
        Initialize the shard-backed dataset.

        Parameters
        ----------
        item : UPDItem
            Descriptor pointing to Zarr dataset and metadata.
        mapping : PINNMapping
            Mapping describing how dataset variables correspond to PINN inputs/outputs.
        sampling : SamplingSpec
            Specification controlling collocation, condition, and data sampling.
        device : str or torch.device
            Device where sampled tensors will be allocated.
        dtype : torch.dtype
            Data type of sampled tensors.
        length : int
            Virtual dataset length. Since sampling is stochastic,
            this defines how many indices the dataset exposes.
        """
        self.item = item
        self.mapping = mapping
        self.sampling = sampling
        self.device = device
        self.dtype = dtype
        self.length = int(length)

        self._upd = UPDDataset(item=item, mapping=mapping, device=device, dtype=dtype)

    def __len__(self) -> int:
        """
        Return the virtual dataset length.

        Returns
        -------
        int
            Number of accessible indices for sampling.
        """
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Sample a new batch from the underlying UPDDataset.

        The random seed is offset by the index to ensure different
        stochastic samples across dataset indices.

        Parameters
        ----------
        idx : int
            Dataset index.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing collocation points, conditions,
            supervised data, and metadata required by collate_pinn_batches.
        """
        # vary seed per idx for fresh random points
        spec = SamplingSpec(
            n_collocation=self.sampling.n_collocation,
            conditions=self.sampling.conditions,
            n_data=self.sampling.n_data,
            replace=self.sampling.replace,
            seed=int(self.sampling.seed) + int(idx),
        )
        b = self._upd.sample(spec)

        out: Dict[str, Any] = {
            "collocation": b.collocation,
            "conditions": b.conditions,
            "data": b.data,
            "meta": {
                "idx": idx,
                "zarr_path": self.item.zarr_path,
                "meta_path": self.item.meta_path,
            },
        }
        return out


def build_upd_dataloader(
    *,
    zarr_path: str,
    meta_path: str,
    mapping: PINNMapping,
    sampling: SamplingSpec,
    loader: Optional[DataLoaderSpec] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    length: int = 10_000,
) -> DataLoader:
    """
    Construct a PyTorch DataLoader from a single UPD shard (Zarr + JSON).

    This function wraps a UPDDataset shard inside a Torch Dataset
    adapter and exposes it via a DataLoader configured with
    DataLoaderSpec parameters.

    Parameters
    ----------
    zarr_path : str
        Path to the Zarr dataset.
    meta_path : str
        Path to the metadata JSON file.
    mapping : PINNMapping
        Mapping between dataset variables and PINN structure.
    sampling : SamplingSpec
        Sampling configuration for collocation, conditions, and data.
    loader : Optional[DataLoaderSpec]
        DataLoader configuration. Defaults to DataLoaderSpec().
    device : str or torch.device
        Device for sampled tensors.
    dtype : torch.dtype
        Tensor data type.
    length : int
        Virtual dataset length for stochastic sampling.

    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader instance.
    """
    loader = loader or DataLoaderSpec()
    item = UPDItem(zarr_path=zarr_path, meta_path=meta_path)
    ds = _UPDShardTorchDataset(
        item=item,
        mapping=mapping,
        sampling=sampling,
        device=device,
        dtype=dtype,
        length=length,
    )
    return DataLoader(
        ds,
        batch_size=loader.batch_size,
        shuffle=loader.shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=collate_pinn_batches,
    )


def build_physical_sample_dataloader(
    sample: PhysicalSample,
    *,
    mapping: PINNMapping,
    sampling: SamplingSpec,
    loader: Optional[DataLoaderSpec] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
    length: int = 10_000,
) -> DataLoader:
    """
    Construct a PyTorch DataLoader from a PhysicalSample instance.

    Current MVP behavior:
      - If the sample represents a structured grid (xarray.Dataset),
        it is wrapped into an in-memory UPD-like interface.
      - Mesh-based samples are not yet supported and will be handled
        in a future MeshPhysicalDataset implementation.

    Parameters
    ----------
    sample : PhysicalSample
        Physical sample containing state, schema, domain, and provenance.
    mapping : PINNMapping
        Mapping between dataset variables and PINN structure.
    sampling : SamplingSpec
        Sampling configuration.
    loader : Optional[DataLoaderSpec]
        DataLoader configuration.
    device : str or torch.device
        Device for sampled tensors.
    dtype : torch.dtype
        Tensor data type.
    length : int
        Virtual dataset length for stochastic sampling.

    Returns
    -------
    DataLoader
        Configured PyTorch DataLoader instance.

    Raises
    ------
    NotImplementedError
        If the PhysicalSample is not a structured grid.
    """
    loader = loader or DataLoaderSpec()

    if not sample.is_grid():
        raise NotImplementedError("Mesh dataloader is MVP-2. For now, only grid PhysicalSample is supported.")

    # Create an in-memory UPDInput dict supported by UPDDataset
    upd_input = {"ds": sample.state, "meta": {"schema": sample.schema, "domain": sample.domain, "provenance": sample.provenance}}

    class _MemUPDItem:
        """
        Minimal in-memory UPDItem-like adapter.

        Provides the interface required by UPDDataset for
        datasets stored in memory instead of disk-backed Zarr files.
        """
        def __init__(self, ds, meta):
            """
            Initialize in-memory dataset wrapper.

            Parameters
            ----------
            ds : Any
                Dataset object (e.g., xarray.Dataset).
            meta : Dict[str, Any]
                Associated metadata dictionary.
            """
            self._ds = ds
            self._meta = meta
            self.zarr_path = "<in-memory>"
            self.meta_path = "<in-memory>"

        def open_dataset(self):
            """
            Return the in-memory dataset.
            """
            return self._ds

        def load_meta(self):
            """
            Return the in-memory metadata.
            """
            return self._meta

    mem_item = _MemUPDItem(upd_input["ds"], upd_input["meta"])

    # Use the same _UPDShardTorchDataset with the mem item
    ds = _UPDShardTorchDataset(
        item=mem_item,  # type: ignore
        mapping=mapping,
        sampling=sampling,
        device=device,
        dtype=dtype,
        length=length,
    )
    return DataLoader(
        ds,
        batch_size=loader.batch_size,
        shuffle=loader.shuffle,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        collate_fn=collate_pinn_batches,
    )