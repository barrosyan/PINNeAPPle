"""Auditability: seed, deterministic mode, RunLogger, env fingerprint."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import os
import json
import time
import platform
import hashlib
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_deterministic(deterministic: bool = True) -> None:
    if not deterministic:
        return
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def env_fingerprint() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda if hasattr(torch.version, "cuda") else None,
    }

@dataclass
class RunLogger:
    """
    JSONL logger for industry-grade auditability.
    """
    out_dir: str
    run_name: str = "run"
    flush_every: int = 1

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)
        self.path = os.path.join(self.out_dir, f"{self.run_name}.jsonl")
        self._n = 0

    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts_unix", int(time.time()))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        self._n += 1

    def save_config(self, cfg: Dict[str, Any]) -> None:
        p = os.path.join(self.out_dir, f"{self.run_name}.config.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def save_artifact(self, name: str, obj: Any) -> None:
        p = os.path.join(self.out_dir, name)
        if isinstance(obj, (dict, list)):
            with open(p, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2)
        else:
            torch.save(obj, p)
