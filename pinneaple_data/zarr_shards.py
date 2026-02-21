"""Sharded Zarr writer for partitioning UPD samples by key with index-based discovery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import os
import json
import re

from .zarr_store import UPDZarrStore


@dataclass
class ShardSpec:
    """
    Simple shard naming.

    Example:
      key_fn(sample) -> "time=2020-01" or "region=NA" etc.
    """
    key_fn: Callable[[Any], str]
    max_per_shard: int = 10_000


def _safe_path_component(s: str) -> str:
    """
    Make shard keys filesystem-safe (Windows-friendly).
    Replaces invalid/suspicious chars with '_'.
    """
    s = str(s).strip()
    # Windows invalid: < > : " / \ | ? *  (also avoid control chars)
    s = re.sub(r'[<>:"/\\\\|?*\\x00-\\x1f]', "_", s)
    # collapse repeats
    s = re.sub(r"_+", "_", s)
    # avoid trailing dots/spaces (Windows)
    s = s.rstrip(". ").strip()
    return s or "shard"


class UPDZarrShardedWriter:
    """
    Writes multiple shards:
      root/
        shards/
          <key>/part=0000/data.zarr
          <key>/part=0001/data.zarr
        index.json

    Intended properties:
      - easy parallelism
      - easy partial downloads
      - natural partitioning by time/region/regime
    """

    def __init__(
        self,
        root: str,
        shard_spec: ShardSpec,
        *,
        shards_dirname: str = "shards",
        index_filename: str = "index.json",
    ):
        """
        Initialize the sharded Zarr writer.

        Parameters
        ----------
        root : str
            Root directory for the sharded layout.
        shard_spec : ShardSpec
            Specification for key extraction and max samples per shard.
        shards_dirname : str, optional
            Subdirectory name for shards. Default is "shards".
        index_filename : str, optional
            Filename for the index JSON. Default is "index.json".
        """
        self.root = str(root)
        self.shard_spec = shard_spec
        self.shards_dir = os.path.join(self.root, shards_dirname)
        os.makedirs(self.shards_dir, exist_ok=True)
        self.index_path = os.path.join(self.root, index_filename)

        self._index: Dict[str, Any] = {
            "format": "pinneaple.upd.shards",
            "version": "0.1",
            "root": os.path.basename(self.root.rstrip("/\\")),
            "shards": {},  # key -> list[{path,count,manifest}]
            "totals": {"num_keys": 0, "num_shards": 0, "num_samples": 0},
        }

    def write(
        self,
        samples: Sequence[Any],
        *,
        upd_version: str = "0.1",
        overwrite: bool = True,
        chunks: Optional[Dict[str, Tuple[int, ...]]] = None,
    ) -> None:
        """
        Partition samples by shard_spec.key_fn and write each part to its own Zarr store.
        """
        buckets: Dict[str, List[Any]] = {}
        for s in samples:
            k = _safe_path_component(self.shard_spec.key_fn(s))
            buckets.setdefault(k, []).append(s)

        total_samples = 0
        total_shards = 0

        for key, items in buckets.items():
            self._index["shards"].setdefault(key, [])

            # split into multiple shards if too big
            maxn = int(self.shard_spec.max_per_shard)
            if maxn <= 0:
                maxn = 10_000

            for part_i, start in enumerate(range(0, len(items), maxn)):
                part = items[start : start + maxn]
                shard_rel = os.path.join("shards", key, f"part={part_i:04d}", "data.zarr")
                shard_abs = os.path.join(self.root, shard_rel)

                os.makedirs(os.path.dirname(shard_abs), exist_ok=True)

                manifest = {
                    "upd_version": str(upd_version),
                    "count": int(len(part)),
                    "key": key,
                    "part": int(part_i),
                }

                UPDZarrStore.write(
                    shard_abs,
                    part,
                    manifest=manifest,
                    overwrite=overwrite,
                    chunks=chunks,
                )

                self._index["shards"][key].append(
                    {"path": shard_rel.replace("\\", "/"), "count": int(len(part)), "manifest": manifest}
                )

                total_samples += int(len(part))
                total_shards += 1

        self._index["totals"]["num_keys"] = int(len(self._index["shards"]))
        self._index["totals"]["num_shards"] = int(total_shards)
        self._index["totals"]["num_samples"] = int(total_samples)

        os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2)
