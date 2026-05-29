"""MATLAB Engine API wrapper.

Requires a MATLAB installation and the MATLAB Engine for Python.
Install from MATLAB:
    cd(fullfile(matlabroot,'extern','engines','python'))
    python setup.py install
Or (R2022a+):
    python -m pip install matlabengine
"""
from __future__ import annotations

from typing import Any, List, Optional
import numpy as np


def _to_matlab(arr: Any, eng: Any) -> Any:
    import matlab
    if isinstance(arr, np.ndarray):
        return matlab.double(arr.tolist())
    return arr


def _from_matlab(val: Any) -> Any:
    try:
        import matlab
        if isinstance(val, matlab.double):
            return np.array(val)
    except Exception:
        pass
    return val


class MATLABEngine:
    """Thin wrapper around matlab.engine.MatlabEngine.

    Examples
    --------
    >>> with MATLABEngine() as eng:
    ...     result = eng.run_function("solve_heat", x_array, nargout=1)
    """

    def __init__(self, start_options: Optional[List[str]] = None) -> None:
        try:
            import matlab.engine as _me
        except ImportError:
            raise ImportError(
                "MATLAB Engine for Python not found. "
                "See: https://www.mathworks.com/help/matlab/matlab_external/"
                "get-started-with-matlab-engine-for-python.html"
            )
        self._eng = _me.start_matlab(" ".join(start_options or []))

    def run_function(self, func_name: str, *args, nargout: int = 1) -> Any:
        """Call a MATLAB function with Python/numpy arguments."""
        margs = [_to_matlab(a, self._eng) for a in args]
        result = getattr(self._eng, func_name)(*margs, nargout=nargout)
        if nargout == 1:
            return _from_matlab(result)
        return tuple(_from_matlab(r) for r in result)

    def run_script(self, script_path: str) -> None:
        """Execute a MATLAB script (.m file) by path."""
        from pathlib import Path
        p = Path(script_path).expanduser().resolve()
        self._eng.addpath(str(p.parent), nargout=0)
        getattr(self._eng, p.stem)(nargout=0)

    def eval(self, expr: str) -> None:
        """Evaluate a MATLAB expression string in the engine workspace."""
        self._eng.eval(expr, nargout=0)

    def get(self, name: str) -> np.ndarray:
        """Retrieve a workspace variable as a numpy array."""
        return _from_matlab(self._eng.workspace[name])

    def put(self, name: str, value: np.ndarray) -> None:
        """Set a workspace variable from a numpy array."""
        self._eng.workspace[name] = _to_matlab(value, self._eng)

    def quit(self) -> None:
        self._eng.quit()

    def __enter__(self) -> "MATLABEngine":
        return self

    def __exit__(self, *_: Any) -> None:
        self.quit()
