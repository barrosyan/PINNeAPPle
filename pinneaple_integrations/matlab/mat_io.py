"""Read/write MATLAB .mat files via scipy.io."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


def load_mat(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load a .mat file and return a dict of numpy arrays.

    MATLAB internal variables (starting with '__') are filtered out.
    """
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("scipy is required for .mat I/O: pip install scipy")
    data = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return {
        k: np.asarray(v)
        for k, v in data.items()
        if not k.startswith("_") and np.ndim(v) >= 0
    }


def save_mat(path: Union[str, Path], arrays: Dict[str, Any]) -> None:
    """Save a dict of numpy arrays (or scalars) to a .mat file."""
    try:
        import scipy.io as sio
    except ImportError:
        raise ImportError("scipy is required for .mat I/O: pip install scipy")
    sio.savemat(str(path), arrays)
