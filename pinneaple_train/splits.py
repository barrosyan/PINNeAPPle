"""Train/val/test split strategies (random, time, group)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Dict
import numpy as np


@dataclass(frozen=True)
class SplitSpec:
    """
    Split strategies:
      - random: shuffle then split by ratios
      - time: keep order; split by ratios (train earliest, test latest)
      - group: ensure groups don't leak (requires group_ids)
    """
    method: str = "time"  # "random" | "time" | "group"
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42

    def __post_init__(self):
        m = (self.method or "").lower().strip()
        object.__setattr__(self, "method", m)

        if m not in {"random", "time", "group"}:
            raise ValueError(f"Unknown split method '{self.method}'. Use: random | time | group")

        for name, v in [("train", self.train), ("val", self.val), ("test", self.test)]:
            if v < 0:
                raise ValueError(f"Split '{name}' must be >= 0, got {v}")

        s = float(self.train + self.val + self.test)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"Splits must sum to 1.0, got {s} (train+val+test)")


def _counts(n: int, train: float, val: float) -> tuple[int, int, int]:
    """
    Stable split counts:
      - floor train and val
      - remainder goes to test
    Guarantees: n_train + n_val + n_test == n
    """
    if n <= 0:
        return 0, 0, 0
    n_train = int(np.floor(train * n))
    n_val = int(np.floor(val * n))
    n_test = n - n_train - n_val
    # guard pathological cases (e.g. train=1.0)
    if n_test < 0:
        n_test = 0
        if n_val > 0:
            n_val = max(0, n - n_train)
        else:
            n_train = n
    return n_train, n_val, n_test


def split_indices(
    n: int,
    spec: SplitSpec,
    *,
    group_ids: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    idx = np.arange(n, dtype=np.int64)

    # ---- GROUP split (no leakage)
    if spec.method == "group":
        if group_ids is None:
            raise ValueError("group_ids required for group split")
        group_ids_arr = np.asarray(group_ids)
        if group_ids_arr.shape[0] != n:
            raise ValueError(f"group_ids must have length {n}, got {group_ids_arr.shape[0]}")

        uniq = np.unique(group_ids_arr)
        rng = np.random.default_rng(spec.seed)
        rng.shuffle(uniq)

        g_train_n, g_val_n, _ = _counts(len(uniq), spec.train, spec.val)

        g_train = set(uniq[:g_train_n])
        g_val = set(uniq[g_train_n : g_train_n + g_val_n])
        g_test = set(uniq[g_train_n + g_val_n :])

        train_idx = idx[np.isin(group_ids_arr, list(g_train))]
        val_idx = idx[np.isin(group_ids_arr, list(g_val))]
        test_idx = idx[np.isin(group_ids_arr, list(g_test))]

        return {"train": train_idx, "val": val_idx, "test": test_idx}

    # ---- RANDOM or TIME split over samples
    if spec.method == "random":
        rng = np.random.default_rng(spec.seed)
        rng.shuffle(idx)
    elif spec.method == "time":
        # keep order (idx is already ordered)
        pass

    n_train, n_val, _ = _counts(n, spec.train, spec.val)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    return {"train": train_idx, "val": val_idx, "test": test_idx}
